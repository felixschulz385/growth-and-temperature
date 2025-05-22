from gnt.data.common.index.base_index import BaseIndex
from typing import Dict, List, Set, Optional, Any, Tuple, Generator, Union
import os
from datetime import datetime
import logging
import hashlib
import tempfile
import sqlite3
import json
import threading
import shutil
import uuid
import re
import pandas as pd
import time
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)

class PreprocessingIndex(BaseIndex):
    """
    Two-stage preprocessing index for tracking data through the entire preprocessing pipeline.
    
    Stage 1 (Run on Kubernetes with GCS access):
    - Aggregate raw files into annual .zarr files
    - Record all files in the index database
    
    Stage 2 (Run on university cluster):
    - Spatially transform to unified grid
    - Mask water tiles and convert to tabular format
    - Update the index with completed files
    
    This index can be exported/imported to bridge between environments.
    """
    
    # Processing stage constants
    STAGE_RAW = "raw"
    STAGE_ANNUAL = "annual"  # Stage 1 output
    STAGE_SPATIAL = "spatial"  # Stage 2 output
    
    # Status constants
    STATUS_PENDING = "pending"
    STATUS_PROCESSING = "processing"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    
    def __init__(self, bucket_name: str, data_path: str, 
             version: str = "v1", client=None, temp_dir=None,
             save_interval_seconds=300, auto_save_thread=False):
        """
        Initialize the preprocessing index.
    
        Args:
            bucket_name: Name of the GCS bucket
            data_path: Base data path that identifies the data type
            version: Version string for the processing run
            client: Optional pre-configured storage client
            temp_dir: Optional directory for temporary files
            save_interval_seconds: Interval in seconds for auto-saving to GCS
            auto_save_thread: Whether to start an auto-save thread
        """
        # Extract dataset name from data path for more readable naming
        dataset_name = data_path.rstrip('/').split('/')[-1]
        
        # Initialize base index with appropriate paths
        super().__init__(
            bucket_name=bucket_name,
            index_name=f"preprocess_{dataset_name}_{version}",
            client=client
        )
        
        self.data_path = data_path
        self.version = version
        self.dataset_name = dataset_name
        
        # Path structure for different processing stages
        self.raw_path = f"{data_path}/"
        self.annual_path = f"{data_path}/annual/"
        self.spatial_path = f"{data_path}/spatial/"
        
        # Paths for the SQLite database
        self.db_path = f"_index/preprocess_{dataset_name}_{version}.sqlite"
        
        # Create a unique identifier for this instance to avoid conflicts
        self._instance_id = str(uuid.uuid4())[:8]
        
        # Set up temporary directory
        self._setup_temp_dir(temp_dir)
        
        # Local path for the SQLite database
        self.local_db_path = os.path.join(self._temp_dir, f"preprocess_{dataset_name}_{version}.sqlite")
        
        # Thread safety
        self._lock = threading.RLock()
        self._thread_local = threading.local()
        
        # Status cache to reduce database access
        self._status_cache = {}
        self._max_cache_size = 10000
        
        # Blob existence cache
        self._blob_existence_cache = {}
        self._blob_cache_max_size = 10000
        
        # Autosave settings
        self.save_interval_seconds = save_interval_seconds
        self._last_save_time = time.time()
        self._operations_since_save = 0
        
        # Initialize database
        self._setup_database()
        
        # Load metadata
        self._load_metadata()
        
        # Start auto-save thread if requested
        if auto_save_thread:
            self.create_auto_save_thread(save_interval_seconds)
    
    def _setup_temp_dir(self, temp_dir):
        """Set up temporary directory for the index."""
        # Use provided temp_dir or create a dedicated subdirectory in the system temp
        if temp_dir:
            self._temp_base_dir = os.path.abspath(temp_dir)
        else:
            self._temp_base_dir = os.path.join(tempfile.gettempdir(), 
                                              f"gnt_preprocessing_{self.dataset_name}")
                
        # Ensure the base temp directory exists
        os.makedirs(self._temp_base_dir, exist_ok=True)
            
        # Create a dedicated temp directory for this index instance
        self._temp_dir = os.path.join(self._temp_base_dir, 
                                     f"preprocess_{self.dataset_name}_{self.version}_{self._instance_id}")
        os.makedirs(self._temp_dir, exist_ok=True)
    
    def _setup_database(self):
        """Set up SQLite database with schema for two-stage processing."""
        blob = self.bucket.blob(self.db_path)
        try:
            if blob.exists():
                logger.info(f"Downloading existing preprocessing index database from {self.db_path} to {self.local_db_path}")
                blob.download_to_filename(self.local_db_path)
            else:
                logger.info(f"No existing preprocessing index found at {self.db_path}, creating new database")
        except Exception as e:
            logger.error(f"Error accessing GCS index: {e}. Creating new database.")
        
        # Connect and create tables if needed
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create the main files table with support for both processing stages
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            file_hash TEXT PRIMARY KEY,
            blob_path TEXT UNIQUE,
            data_type TEXT,            -- Data type identifier
            stage TEXT,                -- Processing stage: 'raw', 'annual', 'spatial'
            year INTEGER,              -- Year of the data
            grid_cell TEXT,            -- Grid cell identifier
            status TEXT,               -- Status: 'pending', 'processing', 'completed', 'failed'
            created_timestamp TEXT,    -- When the file was created
            updated_timestamp TEXT,    -- When the file was last updated
            parent_hash TEXT,          -- Hash of parent file (for tracking dependencies)
            metadata TEXT              -- JSON metadata about the file
        )
        ''')
        
        # Create indexes for common query patterns
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON files(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_year ON files(year)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_grid_cell ON files(grid_cell)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_stage ON files(stage)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_year_grid ON files(year, grid_cell)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_parent ON files(parent_hash)')
        
        # Create a special table for tracking file transfers between environments
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS file_transfers (
            file_hash TEXT PRIMARY KEY,
            status TEXT,               -- Transfer status: 'needed', 'downloaded', 'uploaded'
            created_timestamp TEXT,    -- When the transfer was first needed
            updated_timestamp TEXT,    -- When the transfer status was last updated
            destination TEXT,          -- Destination (e.g., 'cluster', 'gcs')
            FOREIGN KEY (file_hash) REFERENCES files (file_hash)
        )
        ''')
        
        conn.commit()
    
    def _get_connection(self):
        """Get thread-local SQLite connection with performance optimizations."""
        if not hasattr(self._thread_local, 'conn') or self._thread_local.conn is None:
            self._thread_local.conn = sqlite3.connect(self.local_db_path, isolation_level=None, check_same_thread=False)
            
            # Performance optimizations
            self._thread_local.conn.execute('PRAGMA journal_mode=WAL')
            self._thread_local.conn.execute('PRAGMA synchronous=NORMAL')
            self._thread_local.conn.execute('PRAGMA temp_store=MEMORY')
            self._thread_local.conn.execute('PRAGMA cache_size=-10000')  # ~10MB of cache
        
        return self._thread_local.conn
    
    def is_blob_exists(self, blob_path):
        """Check if a blob exists in the bucket with caching for performance."""
        # Check cache first
        if blob_path in self._blob_existence_cache:
            return self._blob_existence_cache[blob_path]
            
        # Query GCS
        exists = self.bucket.blob(blob_path).exists()
        
        # Update cache
        if len(self._blob_existence_cache) < self._blob_cache_max_size:
            self._blob_existence_cache[blob_path] = exists
            
        return exists
    
    def clear_cache(self):
        """Clear the blob existence cache to ensure fresh reads."""
        with self._lock:
            self._blob_existence_cache.clear()
            self._status_cache.clear()
    
    def _parse_blob_path(self, blob_path: str) -> Dict[str, Any]:
        """
        Parse a blob path to extract stage, year, grid cell, and other metadata.
        
        Expected formats:
        - {data_path}/raw/[...]/YYYY/[...]/filename.ext
        - {data_path}/annual/YYYY/gridcellXXXXX/filename.zarr
        - {data_path}/spatial/YYYY/gridcellXXXXX/filename.parquet
        
        Returns a dictionary with extracted information.
        """
        # Determine stage based on path
        stage = self.STAGE_RAW
        if blob_path.startswith(self.annual_path):
            stage = self.STAGE_ANNUAL
            relative_path = blob_path[len(self.annual_path):]
        elif blob_path.startswith(self.spatial_path):
            stage = self.STAGE_SPATIAL
            relative_path = blob_path[len(self.spatial_path):]
        elif blob_path.startswith(self.raw_path):
            stage = self.STAGE_RAW
            relative_path = blob_path[len(self.raw_path):]
        else:
            relative_path = blob_path
            
        # Extract year using regex (4 consecutive digits)
        year_match = re.search(r'/(\d{4})/', relative_path)
        year = int(year_match.group(1)) if year_match else None
        
        # Extract grid cell (assuming format like gridcellXXXXX)
        grid_match = re.search(r'gridcell(\w+)', relative_path)
        grid_cell = grid_match.group(1) if grid_match else None
        
        # If we couldn't extract both year and grid cell, try alternative parsing from filename
        if not year or not grid_cell:
            filename = os.path.basename(blob_path)
            
            # Look for year in filename
            if not year:
                year_match = re.search(r'_(\d{4})_', filename)
                if year_match:
                    year = int(year_match.group(1))
            
            # Look for grid cell in filename
            if not grid_cell:
                grid_match = re.search(r'_grid(\w+)_', filename)
                if grid_match:
                    grid_cell = grid_match.group(1)
        
        # Determine data type from the data path
        data_type = self.data_path.rstrip('/').split('/')[-1]
        
        return {
            "stage": stage,
            "year": year,
            "grid_cell": grid_cell,
            "data_type": data_type,
            "filename": os.path.basename(blob_path)
        }
    
    def index_existing_files(self):
        """
        Index all existing files across all stages.
        """
        logger.info(f"Indexing existing files for dataset {self.dataset_name}")
        
        # Get existing files from GCS
        from gnt.data.download.gcs.client import GCSClient
        gcs_client = GCSClient(self.bucket_name, client=self.client)
        
        try:
            # Index files in all stages
            indexed_count = 0
            for stage_path in [self.raw_path, self.annual_path, self.spatial_path]:
                existing_files = gcs_client.list_existing_files(prefix=stage_path)
                logger.info(f"Found {len(existing_files)} files in {stage_path}")
                
                # Get already indexed files
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT blob_path FROM files")
                indexed_files = {row[0] for row in cursor.fetchall()}
                
                # Filter to files that need indexing
                new_files = [path for path in existing_files if path not in indexed_files]
                logger.info(f"Found {len(new_files)} new files to index in {stage_path}")
                
                # Index in batches
                batch_size = 1000
                stage_indexed = 0
                
                for i in range(0, len(new_files), batch_size):
                    batch = new_files[i:i+batch_size]
                    values = []
                    
                    for blob_path in batch:
                        # Parse information from the blob path
                        info = self._parse_blob_path(blob_path)
                        
                        # Skip if we couldn't parse the required information
                        if not info["stage"]:
                            logger.warning(f"Skipping file that doesn't match expected format: {blob_path}")
                            continue
                        
                        # Generate hash
                        file_hash = hashlib.md5(blob_path.encode()).hexdigest()
                        
                        # Prepare values for insert
                        values.append((
                            file_hash,
                            blob_path,
                            info["data_type"],
                            info["stage"],
                            info["year"],
                            info["grid_cell"],
                            self.STATUS_COMPLETED,  # Assume existing files are completed
                            datetime.now().isoformat(),
                            datetime.now().isoformat(),
                            None,  # parent_hash
                            json.dumps({"filename": info["filename"]})
                        ))
                    
                    # Insert batch
                    if values:
                        try:
                            cursor.executemany('''
                            INSERT OR IGNORE INTO files 
                            (file_hash, blob_path, data_type, stage, year, grid_cell, status, 
                            created_timestamp, updated_timestamp, parent_hash, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', values)
                            conn.commit()
                            stage_indexed += len(values)
                            logger.info(f"Indexed {stage_indexed}/{len(new_files)} files in {stage_path}")
                            
                            # Increment operations count and check if we should save
                            self._operations_since_save += len(values)
                            self._check_save_needed()
                            
                        except Exception as e:
                            logger.error(f"Error inserting batch: {e}")
                            # Try to continue with next batch
                    
                indexed_count += stage_indexed
        
            # Update metadata with indexing stats
            self.metadata["last_indexed"] = datetime.now().isoformat()
            self.metadata["indexed_files"] = indexed_count
            self._save_metadata()
            
            logger.info(f"Completed indexing {indexed_count} files across all stages")
        
        except Exception as e:
            logger.error(f"Error indexing existing files: {e}")
            logger.debug(f"Full error: {traceback.format_exc()}")
            raise
    
    def add_file(self, stage: str, year: int, grid_cell: str = None, 
            status: str = "pending", blob_path: str = None, 
            parent_hash: str = None, metadata: Dict[str, Any] = None) -> str:
        """
        Add a file to the index.
        
        Args:
            stage: Processing stage (raw, annual, spatial)
            year: Year of the data
            grid_cell: Grid cell identifier (can be None for raw files)
            status: Initial status (pending, processing, completed, failed)
            blob_path: Optional explicit blob path; will be auto-generated if None
            parent_hash: Optional hash of parent file (for tracking dependencies)
            metadata: Optional metadata about the file
    
        Returns:
            The file hash that can be used to update the file later
        """
        with self._lock:
            try:
                # Generate a blob path if not provided
                if not blob_path:
                    if stage == self.STAGE_RAW:
                        blob_path = f"{self.raw_path}{year}/"
                    elif stage == self.STAGE_ANNUAL:
                        blob_path = f"{self.annual_path}{year}/gridcell{grid_cell}/"
                    elif stage == self.STAGE_SPATIAL:
                        blob_path = f"{self.spatial_path}{year}/gridcell{grid_cell}/"
                    else:
                        raise ValueError(f"Invalid stage: {stage}")
                    
                    # Add a filename if provided in metadata
                    if metadata and "filename" in metadata:
                        blob_path += metadata["filename"]
                    else:
                        # Generate a generic filename with appropriate extension
                        if stage == self.STAGE_ANNUAL:
                            ext = ".zarr"
                        elif stage == self.STAGE_SPATIAL:
                            ext = ".parquet"
                        else:
                            ext = ".nc"  # default
                    
                        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                        if grid_cell:
                            blob_path += f"{self.dataset_name}_{year}_grid{grid_cell}_{timestamp}{ext}"
                        else:
                            blob_path += f"{self.dataset_name}_{year}_{timestamp}{ext}"
            
                # Generate hash
                file_hash = hashlib.md5(blob_path.encode()).hexdigest()
                
                # Convert metadata to JSON if provided
                metadata_json = json.dumps(metadata) if metadata is not None else None
                
                # Insert into database
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Check if file already exists
                cursor.execute("SELECT status FROM files WHERE file_hash = ?", (file_hash,))
                result = cursor.fetchone()
                
                if result:
                    logger.info(f"File already exists with status: {result[0]}")
                    # Update status if provided
                    if status != self.STATUS_PENDING:
                        cursor.execute('''
                        UPDATE files 
                        SET status = ?, updated_timestamp = ?
                        WHERE file_hash = ?
                        ''', (
                            status,
                            datetime.now().isoformat(),
                            file_hash
                        ))
                        conn.commit()
                        
                        # Update cache if present
                        cache_key = f"file_data_{file_hash}"
                        if cache_key in self._status_cache:
                            self._status_cache[cache_key]["status"] = status
                            self._status_cache[cache_key]["updated_timestamp"] = datetime.now().isoformat()
                        
                    return file_hash
                
                # Insert new file
                try:
                    cursor.execute('''
                    INSERT INTO files 
                    (file_hash, blob_path, data_type, stage, year, grid_cell, status, 
                    created_timestamp, updated_timestamp, parent_hash, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        file_hash,
                        blob_path,
                        self.dataset_name,
                        stage,
                        year,
                        grid_cell,
                        status,
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                        parent_hash,
                        metadata_json
                    ))
                    
                    conn.commit()
                    logger.info(f"Added {stage} file for year {year}, grid cell {grid_cell}, hash {file_hash}")
                    
                    # Increment operations count and check if we should save
                    self._operations_since_save += 1
                    self._check_save_needed()
                    
                    return file_hash
                except Exception as e:
                    logger.error(f"Error adding file: {e}")
                    logger.debug(f"Full error: {traceback.format_exc()}")
                    raise
            except Exception as e:
                logger.error(f"Error in add_file: {e}")
                logger.debug(f"Full error: {traceback.format_exc()}")
                raise
    
    def update_file_status(self, file_hash: str, status: str, 
                    local_path: str = None, metadata: Dict[str, Any] = None):
        """Update file status with periodic save logic."""
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Update the file record
                update_parts = ["status = ?", "updated_timestamp = ?"]
                update_values = [status, datetime.now().isoformat()]
                
                if local_path:
                    update_parts.append("local_path = ?")
                    update_values.append(local_path)
                    
                if metadata:
                    # Get existing metadata to merge if it exists
                    cursor.execute("SELECT metadata FROM files WHERE file_hash = ?", (file_hash,))
                    result = cursor.fetchone()
                    
                    if result and result[0]:
                        try:
                            existing_metadata = json.loads(result[0])
                            # Merge with new metadata
                            existing_metadata.update(metadata)
                            metadata = existing_metadata
                        except json.JSONDecodeError:
                            # If existing metadata isn't valid JSON, just use new metadata
                            pass
                            
                    update_parts.append("metadata = ?")
                    update_values.append(json.dumps(metadata))
                    
                query = f"UPDATE files SET {', '.join(update_parts)} WHERE file_hash = ?"
                update_values.append(file_hash)
                
                cursor.execute(query, update_values)
                conn.commit()
                
                # Update cache if present
                cache_key = f"file_data_{file_hash}"
                if cache_key in self._status_cache:
                    self._status_cache[cache_key]["status"] = status
                    self._status_cache[cache_key]["updated_timestamp"] = update_values[1]
                    if local_path:
                        self._status_cache[cache_key]["local_path"] = local_path
                    if metadata:
                        self._status_cache[cache_key]["metadata"] = metadata
                
                # Increment operations count and check if we should save
                self._operations_since_save += 1
                self._check_save_needed()
                
            except Exception as e:
                logger.error(f"Error updating file status: {e}")
                logger.debug(f"Full error: {traceback.format_exc()}")
                raise
    
    def _check_save_needed(self, force=False):
        """
        Check if index should be saved based on time elapsed or operations count.
        
        Args:
            force: Whether to force a save
        
        Returns:
            bool: Whether save was performed
        """
        current_time = time.time()
        elapsed = current_time - self._last_save_time
        
        # Save if forced, time interval exceeded, or significant operations performed
        if force or elapsed >= self.save_interval_seconds or self._operations_since_save >= 1000:
            logger.info(f"Saving index (forced: {force}, elapsed: {elapsed:.1f}s, operations: {self._operations_since_save})")
            try:
                self.save()
            except Exception as e:
                logger.error(f"Error during autosave: {e}")
                logger.debug(f"Full error: {traceback.format_exc()}")
            
            self._last_save_time = current_time
            self._operations_since_save = 0
            return True
        return False
    
    def mark_for_transfer(self, file_hash: str, destination: str = "cluster") -> bool:
        """
        Mark a file as needed for transfer to another environment.
        
        Args:
            file_hash: Hash of the file to transfer
            destination: Destination environment ('cluster', 'gcs')
            
        Returns:
            True if file was successfully marked for transfer
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Check if file exists in the main index
                cursor.execute("SELECT 1 FROM files WHERE file_hash = ?", (file_hash,))
                if not cursor.fetchone():
                    logger.warning(f"Cannot mark non-existent file for transfer: {file_hash}")
                    return False
                
                # Check if transfer already exists
                cursor.execute("SELECT status FROM file_transfers WHERE file_hash = ?", (file_hash,))
                result = cursor.fetchone()
                
                if result:
                    # Update existing transfer record if status isn't already 'downloaded'
                    if result[0] != 'downloaded':
                        cursor.execute('''
                        UPDATE file_transfers
                        SET status = ?, updated_timestamp = ?, destination = ?
                        WHERE file_hash = ?
                        ''', (
                            'needed',
                            datetime.now().isoformat(),
                            destination,
                            file_hash
                        ))
                        conn.commit()
                    
                    logger.info(f"Updated transfer status for file {file_hash} to 'needed'")
                    return True
                
                # Insert new transfer record
                cursor.execute('''
                INSERT INTO file_transfers
                (file_hash, status, created_timestamp, updated_timestamp, destination)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    file_hash,
                    'needed',
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    destination
                ))
                
                conn.commit()
                logger.info(f"Marked file {file_hash} for transfer to {destination}")
                
                # Increment operations count and check if we should save
                self._operations_since_save += 1
                self._check_save_needed()
                
                return True
            except Exception as e:
                logger.error(f"Error marking file for transfer: {e}")
                logger.debug(f"Full error: {traceback.format_exc()}")
                return False
    
    def update_transfer_status(self, file_hash: str, status: str) -> bool:
        """
        Update the transfer status of a file.
        
        Args:
            file_hash: Hash of the file
            status: New transfer status ('needed', 'downloaded', 'uploaded')
            
        Returns:
            True if status was successfully updated
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Check if transfer exists
                cursor.execute("SELECT 1 FROM file_transfers WHERE file_hash = ?", (file_hash,))
                if not cursor.fetchone():
                    logger.warning(f"Cannot update transfer status for non-existent transfer: {file_hash}")
                    return False
                
                # Update status
                cursor.execute('''
                UPDATE file_transfers
                SET status = ?, updated_timestamp = ?
                WHERE file_hash = ?
                ''', (
                    status,
                    datetime.now().isoformat(),
                    file_hash
                ))
                
                conn.commit()
                logger.info(f"Updated transfer status for file {file_hash} to '{status}'")
                
                # Increment operations count and check if we should save
                self._operations_since_save += 1
                self._check_save_needed()
                
                return True
            except Exception as e:
                logger.error(f"Error updating transfer status: {e}")
                logger.debug(f"Full error: {traceback.format_exc()}")
                return False
    
    def get_pending_transfers(self, destination: str = None) -> List[Dict[str, Any]]:
        """
        Get files that need to be transferred.
        
        Args:
            destination: Optional destination to filter by
            
        Returns:
            List of file information dictionaries
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Build query
            query = """
            SELECT f.*, t.status as transfer_status, t.destination
            FROM files f
            JOIN file_transfers t ON f.file_hash = t.file_hash
            WHERE t.status = 'needed'
            """
            params = []
            
            if destination:
                query += " AND t.destination = ?"
                params.append(destination)
            
            # Execute query
            cursor.execute(query, params)
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Process results
            results = []
            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                
                # Parse metadata if it exists
                if result.get('metadata'):
                    try:
                        result['metadata'] = json.loads(result['metadata'])
                    except:
                        pass  # Keep as string if not valid JSON
                        
                results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error getting pending transfers: {e}")
            logger.debug(f"Full error: {traceback.format_exc()}")
            return []
    
    def get_file_by_hash(self, file_hash: str) -> Dict[str, Any]:
        """Get file by hash with caching for performance."""
        # Check cache first
        cache_key = f"file_data_{file_hash}"
        if cache_key in self._status_cache:
            return self._status_cache[cache_key]
        
        try:
            with self._lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM files WHERE file_hash = ?", 
                    (file_hash,)
                )
                row = cursor.fetchone()
                
                if row:
                    # Convert row to dict using column names
                    columns = [desc[0] for desc in cursor.description]
                    file_data = dict(zip(columns, row))
                    
                    # Parse JSON metadata if present
                    if file_data.get('metadata'):
                        try:
                            file_data['metadata'] = json.loads(file_data['metadata'])
                        except json.JSONDecodeError:
                            file_data['metadata'] = {}
                    
                    # Cache result if cache isn't too large
                    if len(self._status_cache) < self._max_cache_size:
                        self._status_cache[cache_key] = file_data
                        
                    return file_data
            return None
        except Exception as e:
            logger.error(f"Error getting file by hash: {e}")
            logger.debug(f"Full error: {traceback.format_exc()}")
            return None
    
    def get_files(self, stage: str = None, status: str = None, 
                year: int = None, grid_cell: str = None, 
                limit: int = None) -> List[Dict[str, Any]]:
        """
        Get files with optional filtering.
        
        Args:
            stage: Optional processing stage to filter by
            status: Optional status to filter by
            year: Optional year to filter by
            grid_cell: Optional grid cell to filter by
            limit: Maximum number of files to return
            
        Returns:
            List of dictionaries with file information
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Build query based on filters
            query = "SELECT * FROM files WHERE 1=1"
            params = []
            
            if stage is not None:
                query += " AND stage = ?"
                params.append(stage)
                
            if status is not None:
                query += " AND status = ?"
                params.append(status)
                
            if year is not None:
                query += " AND year = ?"
                params.append(year)
                
            if grid_cell is not None:
                query += " AND grid_cell = ?"
                params.append(grid_cell)
            
            if limit is not None:
                query += f" LIMIT {limit}"
            
            # Execute query
            cursor.execute(query, params)
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Convert rows to dictionaries
            results = []
            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                
                # Parse metadata if it exists
                if result.get('metadata'):
                    try:
                        result['metadata'] = json.loads(result['metadata'])
                    except:
                        pass  # Keep as string if not valid JSON
                        
                results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error getting files: {e}")
            logger.debug(f"Full error: {traceback.format_exc()}")
            return []
    
    def get_files_for_processing(self, stage: str, year: int = None, 
                            grid_cell: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get files that are ready for the next processing stage.
        
        Args:
            stage: The target processing stage ('annual' or 'spatial')
            year: Optional year to filter by
            grid_cell: Optional grid cell to filter by
            limit: Maximum number of files to return
            
        Returns:
            List of file information dictionaries
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Determine source stage based on target
            source_stage = self.STAGE_RAW if stage == self.STAGE_ANNUAL else self.STAGE_ANNUAL
            
            # Get completed files from the source stage
            query = """
            SELECT *
            FROM files
            WHERE stage = ? AND status = 'completed'
            """
            params = [source_stage]
            
            if year is not None:
                query += " AND year = ?"
                params.append(year)
                
            if grid_cell is not None:
                query += " AND grid_cell = ?"
                params.append(grid_cell)
            
            # Add subquery to exclude files that have already been processed
            if stage == self.STAGE_ANNUAL:
                # For annual stage, we check that no "annual" entry exists for this year
                query += """
                AND year NOT IN (
                    SELECT year FROM files 
                    WHERE stage = 'annual' 
                    AND (status = 'completed' OR status = 'processing')
                )
                """
            elif stage == self.STAGE_SPATIAL:
                # For spatial stage, we check that no "spatial" entry exists for this year and grid cell
                query += """
                AND (year, grid_cell) NOT IN (
                    SELECT year, grid_cell FROM files 
                    WHERE stage = 'spatial'
                    AND (status = 'completed' OR status = 'processing')
                )
                """
            
            if limit is not None:
                query += f" LIMIT {limit}"
            
            # Execute query
            cursor.execute(query, params)
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            # Process results
            results = []
            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                
                # Parse metadata if it exists
                if result.get('metadata'):
                    try:
                        result['metadata'] = json.loads(result['metadata'])
                    except:
                        pass  # Keep as string if not valid JSON
                        
                results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error getting files for processing: {e}")
            logger.debug(f"Full error: {traceback.format_exc()}")
            return []
    
    def export_index(self, output_path: str = None, include_completed: bool = True) -> str:
        """
        Export the index to a portable SQLite file.
        
        Args:
            output_path: Path to save the exported index. If None, a default path is used.
            include_completed: Whether to include completed files in the export
            
        Returns:
            Path to the exported index file
        """
        if output_path is None:
            output_path = os.path.join(
                self._temp_base_dir, 
                f"{self.dataset_name}_v{self.version}_export_{datetime.now().strftime('%Y%m%d%H%M%S')}.sqlite"
            )
        
        try:
            # Close all connections before copying
            self._close_all_connections()
            
            # Create a copy of the database for export
            shutil.copy2(self.local_db_path, output_path)
            
            # If requested, filter out completed files
            if not include_completed:
                export_conn = sqlite3.connect(output_path)
                cursor = export_conn.cursor()
                
                # Remove completed files
                cursor.execute("DELETE FROM files WHERE status = 'completed'")
                export_conn.commit()
                export_conn.close()
            
            logger.info(f"Exported index to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error exporting index: {e}")
            logger.debug(f"Full error: {traceback.format_exc()}")
            return None
    
    def import_index(self, import_path: str, merge_mode: str = "update") -> bool:
        """
        Import an index from a SQLite file.
        
        Args:
            import_path: Path to the imported index file
            merge_mode: How to handle conflicts ('update', 'overwrite', 'keep')
            
        Returns:
            True if import was successful
        """
        if not os.path.exists(import_path):
            logger.error(f"Import file not found: {import_path}")
            return False
        
        try:
            # Connect to the import database
            import_conn = sqlite3.connect(import_path)
            import_cursor = import_conn.cursor()
            
            # Get local connection
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # First import file transfers
            import_cursor.execute("SELECT * FROM file_transfers")
            transfers = import_cursor.fetchall()
            
            if transfers:
                # Get column names
                import_cursor.execute("PRAGMA table_info(file_transfers)")
                columns = [info[1] for info in import_cursor.fetchall()]
                
                for transfer in transfers:
                    transfer_dict = dict(zip(columns, transfer))
                    file_hash = transfer_dict["file_hash"]
                    
                    # Check if transfer already exists
                    cursor.execute("SELECT status FROM file_transfers WHERE file_hash = ?", (file_hash,))
                    result = cursor.fetchone()
                    
                    if result:
                        # Update existing transfer if merge mode allows it
                        if merge_mode in ["update", "overwrite"]:
                            cursor.execute('''
                            UPDATE file_transfers
                            SET status = ?, updated_timestamp = ?, destination = ?
                            WHERE file_hash = ?
                            ''', (
                                transfer_dict["status"],
                                transfer_dict["updated_timestamp"],
                                transfer_dict["destination"],
                                file_hash
                            ))
                    else:
                        # Insert new transfer
                        cursor.execute('''
                        INSERT INTO file_transfers
                        (file_hash, status, created_timestamp, updated_timestamp, destination)
                        VALUES (?, ?, ?, ?, ?)
                        ''', (
                            file_hash,
                            transfer_dict["status"],
                            transfer_dict["created_timestamp"],
                            transfer_dict["updated_timestamp"],
                            transfer_dict["destination"]
                        ))
            
            # Import files
            import_cursor.execute("SELECT * FROM files")
            files = import_cursor.fetchall()
            
            if files:
                # Get column names
                import_cursor.execute("PRAGMA table_info(files)")
                columns = [info[1] for info in import_cursor.fetchall()]
                
                for file_row in files:
                    file_dict = dict(zip(columns, file_row))
                    file_hash = file_dict["file_hash"]
                    
                    # Check if file already exists
                    cursor.execute("SELECT status, updated_timestamp FROM files WHERE file_hash = ?", (file_hash,))
                    result = cursor.fetchone()
                    
                    if result:
                        existing_status, existing_timestamp = result
                        
                        # Handle conflicts based on merge mode
                        if merge_mode == "overwrite":
                            # Always overwrite
                            cursor.execute('''
                            UPDATE files 
                            SET blob_path = ?, data_type = ?, stage = ?, year = ?, grid_cell = ?, 
                                status = ?, created_timestamp = ?, updated_timestamp = ?, 
                                parent_hash = ?, metadata = ?
                            WHERE file_hash = ?
                            ''', (
                                file_dict["blob_path"],
                                file_dict["data_type"],
                                file_dict["stage"],
                                file_dict["year"],
                                file_dict["grid_cell"],
                                file_dict["status"],
                                file_dict["created_timestamp"],
                                file_dict["updated_timestamp"],
                                file_dict["parent_hash"],
                                file_dict["metadata"],
                                file_hash
                            ))
                        elif merge_mode == "update":
                            # Only update if the import has a newer timestamp
                            import_timestamp = datetime.fromisoformat(file_dict["updated_timestamp"])
                            existing_dt = datetime.fromisoformat(existing_timestamp)
                            
                            if import_timestamp > existing_dt:
                                cursor.execute('''
                                UPDATE files 
                                SET status = ?, updated_timestamp = ?, metadata = ?
                                WHERE file_hash = ?
                                ''', (
                                    file_dict["status"],
                                    file_dict["updated_timestamp"],
                                    file_dict["metadata"],
                                    file_hash
                                ))
                        # For "keep" mode, do nothing - keep existing data
                    else:
                        # Insert new file
                        cursor.execute('''
                        INSERT INTO files 
                        (file_hash, blob_path, data_type, stage, year, grid_cell, status, 
                        created_timestamp, updated_timestamp, parent_hash, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            file_dict["file_hash"],
                            file_dict["blob_path"],
                            file_dict["data_type"],
                            file_dict["stage"],
                            file_dict["year"],
                            file_dict["grid_cell"],
                            file_dict["status"],
                            file_dict["created_timestamp"],
                            file_dict["updated_timestamp"],
                            file_dict["parent_hash"],
                            file_dict["metadata"]
                        ))
            
            conn.commit()
            import_conn.close()
            
            # Mark operations and check if we should save
            self._operations_since_save += len(files) + len(transfers)
            self._check_save_needed()
            
            logger.info(f"Successfully imported index from {import_path} using merge mode '{merge_mode}'")
            return True
            
        except Exception as e:
            logger.error(f"Error importing index: {e}")
            logger.debug(f"Full error: {traceback.format_exc()}")
            return False
    
    def export_to_csv(self, output_dir: str = None, separate_stages: bool = True) -> Dict[str, str]:
        """
        Export the index to CSV files for easier inspection.
        
        Args:
            output_dir: Directory to save CSV files. If None, the temp directory is used.
            separate_stages: Whether to create separate CSVs for each processing stage
            
        Returns:
            Dictionary mapping stage names to CSV file paths
        """
        if output_dir is None:
            output_dir = self._temp_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            conn = self._get_connection()
            
            result_files = {}
            
            if separate_stages:
                # Create separate CSV for each stage
                for stage in [self.STAGE_RAW, self.STAGE_ANNUAL, self.STAGE_SPATIAL]:
                    output_path = os.path.join(output_dir, f"{self.dataset_name}_{stage}_v{self.version}.csv")
                    
                    # Use pandas to create a nice CSV
                    df = pd.read_sql_query(
                        f"SELECT * FROM files WHERE stage = '{stage}'", 
                        conn
                    )
                    
                    if not df.empty:
                        df.to_csv(output_path, index=False)
                        result_files[stage] = output_path
            else:
                # Create a single CSV with all files
                output_path = os.path.join(output_dir, f"{self.dataset_name}_all_v{self.version}.csv")
                df = pd.read_sql_query("SELECT * FROM files", conn)
                df.to_csv(output_path, index=False)
                result_files["all"] = output_path
                
                # Also export transfers
                transfers_path = os.path.join(output_dir, f"{self.dataset_name}_transfers_v{self.version}.csv")
                df_transfers = pd.read_sql_query(
                    """
                    SELECT t.*, f.stage, f.year, f.grid_cell 
                    FROM file_transfers t 
                    JOIN files f ON t.file_hash = f.file_hash
                    """, 
                    conn
                )
                if not df_transfers.empty:
                    df_transfers.to_csv(transfers_path, index=False)
                    result_files["transfers"] = transfers_path
            
            logger.info(f"Exported index to CSV files in {output_dir}")
            return result_files
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            logger.debug(f"Full error: {traceback.format_exc()}")
            return {}
    
    def create_transfer_manifest(self, output_path: str = None) -> str:
        """
        Create a manifest file for transferring files between environments.
        
        Args:
            output_path: Path to save the manifest file. If None, a default path is used.
            
        Returns:
            Path to the manifest file
        """
        if output_path is None:
            output_path = os.path.join(
                self._temp_dir, 
                f"{self.dataset_name}_transfer_manifest_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
            )
        
        try:
            # Get files that need to be transferred
            transfer_files = self.get_pending_transfers()
            
            manifest = {
                "dataset": self.dataset_name,
                "version": self.version,
                "created": datetime.now().isoformat(),
                "files_to_transfer": len(transfer_files),
                "files": []
            }
            
            for file_info in transfer_files:
                manifest["files"].append({
                    "file_hash": file_info["file_hash"],
                    "blob_path": file_info["blob_path"],
                    "stage": file_info["stage"],
                    "year": file_info["year"],
                    "grid_cell": file_info["grid_cell"],
                    "destination": file_info["destination"]
                })
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"Created transfer manifest with {len(transfer_files)} files at {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error creating transfer manifest: {e}")
            logger.debug(f"Full error: {traceback.format_exc()}")
            return None
    
    def mark_stage1_complete(self, year: int, grid_cell: str) -> bool:
        """
        Mark that stage 1 (annual aggregation) is complete for a specific year and grid cell.
        
        This is a convenience method that:
        1. Updates the annual file status to 'completed'
        2. Marks the file for transfer to the cluster
        
        Args:
            year: Year of the data
            grid_cell: Grid cell identifier
            
        Returns:
            True if successful
        """
        try:
            # Find the annual file for this year and grid cell
            annual_files = self.get_files(
                stage=self.STAGE_ANNUAL,
                year=year,
                grid_cell=grid_cell,
                status=self.STATUS_PROCESSING
            )
            
            if not annual_files:
                logger.warning(f"No processing annual file found for year {year}, grid cell {grid_cell}")
                return False
            
            file_hash = annual_files[0]["file_hash"]
            
            # Update status to completed
            self.update_file_status(
                file_hash=file_hash,
                status=self.STATUS_COMPLETED,
                metadata={"stage1_completed": datetime.now().isoformat()}
            )
            
            # Mark for transfer to cluster
            return self.mark_for_transfer(file_hash, "cluster")
        except Exception as e:
            logger.error(f"Error marking stage 1 complete: {e}")
            logger.debug(f"Full error: {traceback.format_exc()}")
            return False
    
    def plan_stage2_processing(self, years: List[int] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Plan stage 2 processing by identifying annual files that need spatial transformation.
        
        Args:
            years: Optional list of years to process
            limit: Maximum number of files to include in plan
            
        Returns:
            List of file information dictionaries ready for stage 2 processing
        """
        try:
            # Get completed annual files that don't have corresponding spatial files
            files_to_process = []
            
            if years:
                # Process specified years
                for year in years:
                    batch = self.get_files_for_processing(
                        stage=self.STAGE_SPATIAL,
                        year=year,
                        limit=limit - len(files_to_process)
                    )
                    files_to_process.extend(batch)
                    
                    if len(files_to_process) >= limit:
                        break
            else:
                # Get any available files
                files_to_process = self.get_files_for_processing(
                    stage=self.STAGE_SPATIAL,
                    limit=limit
                )
            
            return files_to_process
        except Exception as e:
            logger.error(f"Error planning stage 2 processing: {e}")
            logger.debug(f"Full error: {traceback.format_exc()}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the preprocessing index."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            stats = {
                "dataset": self.dataset_name,
                "version": self.version,
                "data_path": self.data_path,
                "stages": {}
            }
            
            # Get counts by stage and status
            cursor.execute("""
            SELECT stage, status, COUNT(*) 
            FROM files 
            GROUP BY stage, status
            """)
            stage_status_counts = cursor.fetchall()
            
            for stage, status, count in stage_status_counts:
                if stage not in stats["stages"]:
                    stats["stages"][stage] = {"total": 0}
                
                stats["stages"][stage][status] = count
                stats["stages"][stage]["total"] += count
            
            # Get year ranges for each stage
            for stage in stats["stages"].keys():
                cursor.execute("""
                SELECT MIN(year), MAX(year) 
                FROM files 
                WHERE stage = ?
                """, (stage,))
                min_year, max_year = cursor.fetchone()
                
                stats["stages"][stage]["year_range"] = [min_year, max_year] if min_year and max_year else None
                
                # For annual and spatial, get grid cell counts
                if stage in [self.STAGE_ANNUAL, self.STAGE_SPATIAL]:
                    cursor.execute("""
                    SELECT COUNT(DISTINCT grid_cell) 
                    FROM files 
                    WHERE stage = ?
                    """, (stage,))
                    stats["stages"][stage]["grid_cell_count"] = cursor.fetchone()[0]
            
            # Get transfer counts
            cursor.execute("""
            SELECT status, COUNT(*) 
            FROM file_transfers 
            GROUP BY status
            """)
            transfer_counts = dict(cursor.fetchall())
            
            stats["transfers"] = {
                "total": sum(transfer_counts.values()) if transfer_counts else 0,
                "counts": transfer_counts
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            logger.debug(f"Full error: {traceback.format_exc()}")
            return {"error": str(e)}
    
    def save(self):
        """Save the index database to GCS."""
        logger.info("Saving preprocessing index to GCS")
        try:
            # Close all connections to ensure data is flushed
            self._close_all_connections()
            
            # Small delay to allow SQLite to release locks
            time.sleep(0.5)
            
            # Create a fresh connection for the checkpoint
            temp_conn = sqlite3.connect(self.local_db_path)
            temp_cursor = temp_conn.cursor()

            try:
                # Run VACUUM to compact the database
                temp_cursor.execute("VACUUM")
                temp_conn.commit()
            finally:
                temp_conn.close()
            
            # Upload database to GCS
            with open(self.local_db_path, 'rb') as f:
                self.bucket.blob(self.db_path).upload_from_file(f)
            
            # Update metadata
            self._save_metadata()
            
            # Reset last save time and operations counter
            self._last_save_time = time.time()
            self._operations_since_save = 0
            
            logger.info(f"Index saved to {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving index to GCS: {e}")
            logger.debug(f"Full error: {traceback.format_exc()}")
            return False
        
    def auto_save_interval(func):
        """Decorator to automatically save at intervals when operations accumulate."""
        def wrapper(self, *args, **kwargs):
            # Call the original function
            result = func(self, *args, **kwargs)
            
            # Increment operations counter
            self._operations_since_save += 1
            
            # Check if we should save
            self._check_save_needed()
            
            return result
        return wrapper

    def validate_against_gcs(self, gcs_client=None, force_file_list_update=False):
        """
        Validate the index against actual GCS bucket contents.
        
        This method performs several validations:
        1. Finds files in GCS that aren't in the index
        2. Identifies indexed files that don't exist in GCS
        3. Checks for consistency between index stages
        
        Args:
            gcs_client: Optional pre-configured GCS client
            force_file_list_update: Whether to force refresh of file list from GCS
        
        Returns:
            Dict with validation results
        """
        logger.info(f"Validating preprocessing index for {self.dataset_name} against GCS")
        
        # Create GCS client if not provided
        if gcs_client is None:
            from gnt.data.download.gcs.client import GCSClient
            gcs_client = GCSClient(self.bucket_name, client=self.client)
        
        # Get all existing files from GCS for each stage path
        gcs_files = {}
        total_gcs_files = 0
        
        for stage_name, stage_path in [
            (self.STAGE_RAW, self.raw_path),
            (self.STAGE_ANNUAL, self.annual_path),
            (self.STAGE_SPATIAL, self.spatial_path)
        ]:
            # Clear cache if forced update
            if force_file_list_update:
                self.clear_cache()
                
            # Get existing files
            logger.info(f"Listing files in GCS path: {stage_path}")
            try:
                files = gcs_client.list_existing_files(prefix=stage_path)
                gcs_files[stage_name] = set(files)
                total_gcs_files += len(files)
                logger.info(f"Found {len(files)} files for stage {stage_name} in GCS")
            except Exception as e:
                logger.error(f"Error listing files for stage {stage_name}: {e}")
                gcs_files[stage_name] = set()
        
        # Get all indexed files
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT file_hash, blob_path, stage, status FROM files")
        indexed_files = cursor.fetchall()
        
        # Organize indexed files by stage
        index_files = {stage: set() for stage in [self.STAGE_RAW, self.STAGE_ANNUAL, self.STAGE_SPATIAL]}
        index_path_to_hash = {}
        index_hash_to_data = {}
        
        for file_hash, blob_path, stage, status in indexed_files:
            index_files[stage].add(blob_path)
            index_path_to_hash[blob_path] = file_hash
            index_hash_to_data[file_hash] = {"path": blob_path, "stage": stage, "status": status}
        
        # Find inconsistencies
        results = {
            "missing_from_index": {},
            "missing_from_gcs": {},
            "stage_counts": {},
            "status_by_stage": {},
            "total_indexed": len(indexed_files),
            "total_in_gcs": total_gcs_files,
            "orphaned_transfers": []
        }
        
        # Files in GCS but not in index
        for stage in [self.STAGE_RAW, self.STAGE_ANNUAL, self.STAGE_SPATIAL]:
            in_gcs_not_in_index = gcs_files[stage] - index_files[stage]
            results["missing_from_index"][stage] = list(in_gcs_not_in_index)
            
            # Files in index but not in GCS
            in_index_not_in_gcs = index_files[stage] - gcs_files[stage]
            results["missing_from_gcs"][stage] = list(in_index_not_in_gcs)
            
            # Count by stage
            results["stage_counts"][stage] = len(index_files[stage])
        
        # Get status counts by stage
        cursor.execute("""
        SELECT stage, status, COUNT(*) FROM files
        GROUP BY stage, status
        """)
        
        for stage, status, count in cursor.fetchall():
            if stage not in results["status_by_stage"]:
                results["status_by_stage"][stage] = {}
            results["status_by_stage"][stage][status] = count
        
        # Check for orphaned transfers (transfers for non-existent files)
        cursor.execute("""
        SELECT t.file_hash FROM file_transfers t
        LEFT JOIN files f ON t.file_hash = f.file_hash
        WHERE f.file_hash IS NULL
        """)
        orphaned_transfers = cursor.fetchall()
        results["orphaned_transfers"] = [hash[0] for hash in orphaned_transfers]
        
        # Save validation results in metadata
        validation_summary = {
            "performed_at": datetime.now().isoformat(),
            "missing_from_index_count": sum(len(files) for files in results["missing_from_index"].values()),
            "missing_from_gcs_count": sum(len(files) for files in results["missing_from_gcs"].values()),
            "orphaned_transfers_count": len(results["orphaned_transfers"]),
            "total_indexed": results["total_indexed"],
            "total_in_gcs": results["total_gcs_files"]
        }
        
        self.metadata["last_validation"] = validation_summary
        self._save_metadata()
        
        logger.info(f"Validation complete: found {validation_summary['missing_from_index_count']} files in GCS not in index, " 
                    f"{validation_summary['missing_from_gcs_count']} files in index not in GCS")
        
        return results

    def cleanup_missing_files(self, fix_orphaned_transfers=True, remove_missing_from_index=False):
        """
        Clean up the index by removing or fixing inconsistencies.
        
        Args:
            fix_orphaned_transfers: Whether to remove orphaned transfers
            remove_missing_from_index: Whether to remove files from index that don't exist in GCS
            
        Returns:
            Dict with cleanup results
        """
        logger.info(f"Cleaning up index for {self.dataset_name}")
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        results = {
            "orphaned_transfers_removed": 0,
            "missing_files_removed": 0
        }
        
        # Fix orphaned transfers
        if fix_orphaned_transfers:
            cursor.execute("""
            DELETE FROM file_transfers WHERE file_hash IN (
                SELECT t.file_hash FROM file_transfers t
                LEFT JOIN files f ON t.file_hash = f.file_hash
                WHERE f.file_hash IS NULL
            )
            """)
            results["orphaned_transfers_removed"] = cursor.rowcount
            logger.info(f"Removed {results['orphaned_transfers_removed']} orphaned transfers")
        
        # Remove files that don't exist in GCS
        if remove_missing_from_index:
            # We'll need to check each file individually
            cursor.execute("SELECT file_hash, blob_path FROM files")
            all_files = cursor.fetchall()
            
            files_to_remove = []
            for file_hash, blob_path in all_files:
                if not self.is_blob_exists(blob_path):
                    files_to_remove.append(file_hash)
            
            if files_to_remove:
                # Remove from transfers first to maintain foreign key constraints
                placeholders = ','.join(['?'] * len(files_to_remove))
                cursor.execute(f"DELETE FROM file_transfers WHERE file_hash IN ({placeholders})", files_to_remove)
                
                # Then remove from main index
                cursor.execute(f"DELETE FROM files WHERE file_hash IN ({placeholders})", files_to_remove)
                results["missing_files_removed"] = len(files_to_remove)
                logger.info(f"Removed {results['missing_files_removed']} indexed files that don't exist in GCS")
        
        conn.commit()
        
        # Schedule a save operation
        self._operations_since_save += max(results["orphaned_transfers_removed"], results["missing_files_removed"])
        self._check_save_needed(force=True)
        
        return results

    def create_auto_save_thread(self, interval_seconds=300):
        """
        Create a background thread that automatically saves the index at intervals.
        
        Args:
            interval_seconds: How often to save the index (in seconds)
            
        Returns:
            The started thread
        """
        import threading
        
        def auto_save_worker():
            logger.info(f"Starting auto-save thread with {interval_seconds}s interval")
            while True:
                try:
                    # Sleep for the interval
                    time.sleep(interval_seconds)
                    
                    # Check if we need to save
                    with self._lock:
                        if self._operations_since_save > 0:
                            logger.info(f"Auto-save thread: saving after {self._operations_since_save} operations")
                            self.save()
                except Exception as e:
                    logger.error(f"Error in auto-save thread: {e}")
        
        # Create and start the thread
        thread = threading.Thread(target=auto_save_worker, daemon=True)
        thread.start()
        
        # Store thread reference
        self._auto_save_thread = thread
        
        return thread

    def stop_auto_save(self):
        """Stop the auto-save thread if running."""
        if hasattr(self, '_auto_save_thread'):
            # We can't really stop daemon threads, but we can clear the reference
            logger.info("Auto-save thread will terminate when program exits")
            delattr(self, '_auto_save_thread')
    
    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        try:
            logger.info(f"Cleaning up temporary directory: {self._temp_dir}")
            if os.path.exists(self._temp_dir):
                shutil.rmtree(self._temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory: {e}")
    
    def _close_all_connections(self):
        """Close all database connections."""
        with self._lock:
            try:
                # Close thread-local connections
                if hasattr(self._thread_local, 'conn') and self._thread_local.conn is not None:
                    self._thread_local.conn.commit()
                    self._thread_local.conn.close()
                    self._thread_local.conn = None
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.save()
        finally:
            self._close_all_connections()
            self._cleanup_temp_files()
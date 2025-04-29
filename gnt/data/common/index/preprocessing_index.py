from gnt.data.common.index.base_index import BaseIndex
from typing import Dict, List, Set, Optional, Any, Tuple, Generator
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

logger = logging.getLogger(__name__)

class DataPreprocessingIndex(BaseIndex):
    """
    Memory-efficient SQLite-based index for gridded preprocessing data.
    
    Indexes files by:
    1. Data path (data type)
    2. Year
    3. Grid cell
    4. Processing status and metadata
    """
    
    def __init__(self, bucket_name: str, data_path: str, 
                 version: str = "v1", client=None, temp_dir=None):
        """
        Initialize the preprocessing index.
        
        Args:
            bucket_name: Name of the GCS bucket
            data_path: Base data path that identifies the data type
            version: Version string for the processing run
            client: Optional pre-configured storage client
            temp_dir: Optional directory for temporary files
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
        
        # Intermediate directory path where processed files are stored
        self.intermediate_path = f"{data_path}/intermediate/"
        
        # Paths for the SQLite database in GCS
        self.db_path = f"_index/preprocess_{dataset_name}_{version}.sqlite"
        
        # Create a unique identifier for this instance to avoid conflicts
        self._instance_id = str(uuid.uuid4())[:8]
        
        # Use provided temp_dir or create a dedicated subdirectory in the system temp
        if temp_dir:
            self._temp_base_dir = os.path.abspath(temp_dir)
        else:
            self._temp_base_dir = os.path.join(tempfile.gettempdir(), "gnt_indices")
            
        # Ensure the base temp directory exists
        os.makedirs(self._temp_base_dir, exist_ok=True)
            
        # Create a dedicated temp directory for this index instance
        self._temp_dir = os.path.join(self._temp_base_dir, f"preprocess_{dataset_name}_{version}_{self._instance_id}")
        os.makedirs(self._temp_dir, exist_ok=True)
        
        # Local path for the SQLite database
        self.local_db_path = os.path.join(self._temp_dir, f"preprocess_{dataset_name}_{version}.sqlite")
        
        # Database connection
        self._conn = None
        self._lock = threading.RLock()
        
        # Download existing database or create new one
        self._setup_database()
        self._load_metadata()
        
        # Index existing files automatically
        self.index_existing_files()
    
    def _setup_database(self):
        """Set up SQLite database with a simplified schema focused on grid cells and years."""
        blob = self.bucket.blob(self.db_path)
        if blob.exists():
            logger.info(f"Downloading existing preprocessing index database from {self.db_path} to {self.local_db_path}")
            blob.download_to_filename(self.local_db_path)
        else:
            logger.info(f"No existing preprocessing index found at {self.db_path}, creating new database")
        
        # Connect and create tables if needed
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create a simplified files table focused on grid cells and years
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            file_hash TEXT PRIMARY KEY,
            blob_path TEXT UNIQUE,
            data_type TEXT,            -- Data type identifier (derived from data_path)
            year INTEGER,              -- Year of the data
            grid_cell TEXT,            -- Grid cell identifier
            status TEXT,               -- Status: 'pending', 'processing', 'completed', 'failed'
            created_timestamp TEXT,    -- When the file was created
            updated_timestamp TEXT,    -- When the file was last updated
            metadata TEXT              -- JSON metadata about the file
        )
        ''')
        
        # Create indexes for common query patterns
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON files(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_year ON files(year)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_grid_cell ON files(grid_cell)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_type ON files(data_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_year_grid ON files(year, grid_cell)')
        
        conn.commit()
    
    def _get_connection(self):
        """Get SQLite connection (create if needed)."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.local_db_path)
            # Enable optimizations
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
        return self._conn
    
    def _parse_blob_path(self, blob_path: str) -> Dict[str, Any]:
        """
        Parse a blob path to extract year, grid cell, and other metadata.
        
        Expected format: {data_path}/intermediate/[...]/YYYY/gridcellXXXXX/filename.ext
        
        Returns a dictionary with extracted information or None if it doesn't match the expected format.
        """
        # Remove the intermediate_path prefix to simplify parsing
        if blob_path.startswith(self.intermediate_path):
            relative_path = blob_path[len(self.intermediate_path):]
        else:
            relative_path = blob_path
            
        # Extract year using regex (4 consecutive digits)
        year_match = re.search(r'/(\d{4})/', relative_path)
        year = int(year_match.group(1)) if year_match else None
        
        # Extract grid cell (assuming format like gridcellXXXXX)
        grid_match = re.search(r'gridcell(\w+)', relative_path)
        grid_cell = grid_match.group(1) if grid_match else None
        
        # If we couldn't extract both year and grid cell, try alternative parsing
        if not year or not grid_cell:
            # Try to extract from filename directly
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
            "year": year,
            "grid_cell": grid_cell,
            "data_type": data_type,
            "filename": os.path.basename(blob_path)
        }
    
    def index_existing_files(self):
        """
        Index all existing files in the intermediate directory.
        
        This automatically runs on initialization to ensure the index is up-to-date.
        """
        logger.info(f"Indexing existing files in {self.intermediate_path}")
        
        # Get existing files from GCS
        from gnt.data.download.gcs.client import GCSClient
        gcs_client = GCSClient(self.bucket_name, client=self.client)
        
        try:
            existing_files = gcs_client.list_existing_files(prefix=self.intermediate_path)
            logger.info(f"Found {len(existing_files)} files in {self.intermediate_path}")
            
            # Get already indexed files
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT blob_path FROM files")
            indexed_files = {row[0] for row in cursor.fetchall()}
            
            # Filter to files that need indexing
            new_files = [path for path in existing_files if path not in indexed_files]
            logger.info(f"Found {len(new_files)} new files to index")
            
            # Index in batches
            batch_size = 1000
            files_indexed = 0
            
            for i in range(0, len(new_files), batch_size):
                batch = new_files[i:i+batch_size]
                values = []
                
                for blob_path in batch:
                    # Parse information from the blob path
                    info = self._parse_blob_path(blob_path)
                    
                    # Skip if we couldn't parse the required information
                    if not info["year"] or not info["grid_cell"]:
                        logger.warning(f"Skipping file that doesn't match expected format: {blob_path}")
                        continue
                    
                    # Generate hash
                    file_hash = hashlib.md5(blob_path.encode()).hexdigest()
                    
                    # Prepare values for insert
                    values.append((
                        file_hash,
                        blob_path,
                        info["data_type"],
                        info["year"],
                        info["grid_cell"],
                        "completed",  # Assume existing files are completed
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                        json.dumps({"filename": info["filename"]})
                    ))
                
                # Insert batch
                if values:
                    cursor.executemany('''
                    INSERT OR IGNORE INTO files 
                    (file_hash, blob_path, data_type, year, grid_cell, status, 
                    created_timestamp, updated_timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', values)
                    conn.commit()
                    files_indexed += len(values)
                    logger.info(f"Indexed {files_indexed}/{len(new_files)} files")
            
            # Update metadata with indexing stats
            self.metadata["last_indexed"] = datetime.now().isoformat()
            self.metadata["indexed_files"] = files_indexed
            self._save_metadata()
            
            logger.info(f"Completed indexing {files_indexed} files")
            
        except Exception as e:
            logger.error(f"Error indexing existing files: {e}")
            raise
    
    def add_pending_file(self, year: int, grid_cell: str, metadata: Dict[str, Any] = None):
        """
        Add a pending file to be processed.
        
        Args:
            year: Year of the data
            grid_cell: Grid cell identifier
            metadata: Optional metadata about the file
        
        Returns:
            The file hash that can be used to update the file later
        """
        with self._lock:
            # Generate a deterministic blob path
            blob_path = f"{self.intermediate_path}{year}/gridcell{grid_cell}/"
            
            # Add a filename if provided in metadata
            if metadata and "filename" in metadata:
                blob_path += metadata["filename"]
            else:
                # Generate a generic filename
                blob_path += f"{self.dataset_name}_{year}_grid{grid_cell}_{datetime.now().strftime('%Y%m%d%H%M%S')}.nc"
            
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
                return file_hash
            
            # Insert new pending file
            cursor.execute('''
            INSERT INTO files 
            (file_hash, blob_path, data_type, year, grid_cell, status, 
             created_timestamp, updated_timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                file_hash,
                blob_path,
                self.dataset_name,
                year,
                grid_cell,
                "pending",
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                metadata_json
            ))
            
            conn.commit()
            logger.info(f"Added pending file for year {year}, grid cell {grid_cell}, hash {file_hash}")
            
            return file_hash
    
    def update_file_status(self, file_hash: str, status: str, 
                          metadata: Dict[str, Any] = None):
        """
        Update the status of a file.
        
        Args:
            file_hash: Hash of the file
            status: New status ('pending', 'processing', 'completed', 'failed')
            metadata: Optional additional metadata
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check if file exists
            cursor.execute("SELECT metadata FROM files WHERE file_hash = ?", (file_hash,))
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"Attempted to update non-existent file: {file_hash}")
                return False
            
            # Merge with existing metadata if provided
            if metadata:
                try:
                    existing_metadata = json.loads(result[0]) if result[0] else {}
                    existing_metadata.update(metadata)
                    metadata_json = json.dumps(existing_metadata)
                except:
                    metadata_json = json.dumps(metadata)
            else:
                metadata_json = result[0]  # Keep existing metadata
            
            # Update status and metadata
            cursor.execute('''
            UPDATE files 
            SET status = ?, updated_timestamp = ?, metadata = ?
            WHERE file_hash = ?
            ''', (
                status,
                datetime.now().isoformat(),
                metadata_json,
                file_hash
            ))
            
            conn.commit()
            logger.info(f"Updated file {file_hash} to status: {status}")
            
            return True
    
    def get_file_by_hash(self, file_hash: str) -> Dict[str, Any]:
        """
        Get file information by hash.
        
        Args:
            file_hash: Hash of the file
            
        Returns:
            Dictionary with file information or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM files WHERE file_hash = ?", (file_hash,))
        columns = [description[0] for description in cursor.description]
        row = cursor.fetchone()
        
        if not row:
            return None
            
        result = dict(zip(columns, row))
        
        # Parse metadata if it exists
        if result.get('metadata'):
            try:
                result['metadata'] = json.loads(result['metadata'])
            except:
                pass  # Keep as string if not valid JSON
                
        return result
    
    def iter_pending_files(self, year: int = None, grid_cell: str = None,
                         chunk_size: int = 10) -> Generator[Dict[str, Any], None, None]:
        """
        Generator that yields pending files one at a time.
        
        Args:
            year: Optional year to filter by
            grid_cell: Optional grid cell to filter by
            chunk_size: Number of records to fetch at once (for efficiency)
            
        Yields:
            File information dictionary
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get column names once
        cursor.execute("SELECT * FROM files WHERE 1=0")
        columns = [description[0] for description in cursor.description]
        
        # Track last processed ID
        last_id = 0
        
        while True:
            # Build query based on filters
            query = "SELECT * FROM files WHERE status = 'pending' AND rowid > ?"
            params = [last_id]
            
            # Add year filter if provided
            if year is not None:
                query += " AND year = ?"
                params.append(year)
                
            # Add grid cell filter if provided
            if grid_cell is not None:
                query += " AND grid_cell = ?"
                params.append(grid_cell)
            
            query += f" ORDER BY rowid LIMIT {chunk_size}"
            
            # Execute query
            cursor.execute(query, params)
            
            # Process results
            rows = cursor.fetchall()
            if not rows:
                break  # No more results
            
            # Yield each result
            for row in rows:
                result = dict(zip(columns, row))
                
                # Parse metadata if it exists
                if result.get('metadata'):
                    try:
                        result['metadata'] = json.loads(result['metadata'])
                    except:
                        pass  # Keep as string if not valid JSON
                
                # Update last_id
                last_id = cursor.lastrowid or last_id + 1
                
                yield result
    
    def get_files(self, status: str = None, year: int = None, 
                grid_cell: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get files with optional filtering.
        
        Args:
            status: Optional status to filter by
            year: Optional year to filter by
            grid_cell: Optional grid cell to filter by
            limit: Maximum number of files to return
            
        Returns:
            List of dictionaries with file information
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Build query based on filters
        query = "SELECT * FROM files WHERE 1=1"
        params = []
        
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the preprocessing index."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get file counts by status
        cursor.execute("SELECT status, COUNT(*) FROM files GROUP BY status")
        status_counts = dict(cursor.fetchall())
        
        # Get counts by year
        cursor.execute("SELECT year, COUNT(*) FROM files GROUP BY year")
        year_counts = {str(year): count for year, count in cursor.fetchall()}
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM files")
        total_count = cursor.fetchone()[0]
        
        # Get distinct grid cell count
        cursor.execute("SELECT COUNT(DISTINCT grid_cell) FROM files")
        grid_cell_count = cursor.fetchone()[0]
        
        # Get year range
        cursor.execute("SELECT MIN(year), MAX(year) FROM files")
        min_year, max_year = cursor.fetchone()
        
        return {
            "dataset": self.dataset_name,
            "version": self.version,
            "data_path": self.data_path,
            "total_files": total_count,
            "status_counts": status_counts,
            "grid_cell_count": grid_cell_count,
            "year_range": [min_year, max_year] if min_year and max_year else None,
            "year_counts": year_counts
        }
    
    def save(self):
        """Save the index database and metadata to GCS."""
        logger.info(f"Saving preprocessing index database to {self.db_path}")
        
        # Update metadata statistics
        self.metadata["stats"] = self.get_stats()
        
        # Close connection to ensure all data is written
        if self._conn:
            self._conn.close()
            self._conn = None
        
        # Upload database to GCS
        self.bucket.blob(self.db_path).upload_from_filename(self.local_db_path)
        
        # Save metadata
        self._save_metadata()
    
    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        try:
            logger.info(f"Cleaning up temporary directory: {self._temp_dir}")
            if os.path.exists(self._temp_dir):
                shutil.rmtree(self._temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save everything and clean up."""
        try:
            self.save()
        finally:
            self._cleanup_temp_files()
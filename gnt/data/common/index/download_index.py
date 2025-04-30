from gnt.data.common.index.base_index import BaseIndex
from gnt.data.common.gcs.client import GCSClient
from gnt.data.download.sources.base import BaseDataSource
from typing import Dict, List, Iterator, Tuple, Optional, Generator
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
import time

logger = logging.getLogger(__name__)

class DataDownloadIndex(BaseIndex):
    """Memory-efficient SQLite-based index for downloading files."""
    
    def __init__(self, bucket_name: str, data_source: BaseDataSource, client=None, temp_dir=None, 
                 auto_index=True, save_interval_seconds=300):
        # Extract data source properties
        self.data_source_name = getattr(data_source, "DATA_SOURCE_NAME", "unknown")
        self.data_path = getattr(data_source, "data_path", "unknown")
        
        # Initialize base index with appropriate paths
        super().__init__(
            bucket_name=bucket_name,
            index_name=self.data_source_name, 
            client=client
        )
        
        # Paths for the SQLite database in GCS
        self.db_path = f"_index/download_{self.data_path.replace('/', '_')}.sqlite"
        self.blob_list_path = f"_index/download_{self.data_path.replace('/', '_')}.json"
        self.entrypoints_path = f"_index/download_{self.data_path.replace('/', '_')}_entrypoints.json"
        
        # Create a unique identifier for this instance to avoid conflicts
        self._instance_id = str(uuid.uuid4())[:8]
        
        # Set up temporary directory
        self._setup_temp_dir(temp_dir)
        
        # Database connection
        self._conn = None
        self._lock = threading.RLock()
        
        # Set up status cache
        self._status_cache = {}
        self._max_cache_size = 10000
        
        # Autosave settings
        self.save_interval_seconds = save_interval_seconds
        self._last_save_time = time.time()
        
        # Download existing database or create new one
        self._setup_database()
        self._load_metadata()
        
        # Build the index if auto_index is enabled
        if auto_index:
            self.refresh_index(data_source)
    
    def _setup_temp_dir(self, temp_dir):
        """Set up temporary directory for the index."""
        # Use provided temp_dir or create a dedicated subdirectory in the system temp
        if temp_dir:
            self._temp_base_dir = os.path.abspath(temp_dir)
        else:
            self._temp_base_dir = os.path.join(tempfile.gettempdir(), "gnt_indices")
            
        # Ensure the base temp directory exists
        os.makedirs(self._temp_base_dir, exist_ok=True)
            
        # Create a dedicated temp directory for this index instance
        self._temp_dir = os.path.join(self._temp_base_dir, f"index_{self.data_source_name}_{self._instance_id}")
        os.makedirs(self._temp_dir, exist_ok=True)
        
        # Local path for the SQLite database
        self.local_db_path = os.path.join(self._temp_dir, f"download_{self.data_path.replace('/', '_')}.sqlite")
    
    def _setup_database(self):
        """Set up SQLite database either by downloading existing or creating new."""
        blob = self.bucket.blob(self.db_path)
        if blob.exists():
            logger.info(f"Downloading existing index database from {self.db_path} to {self.local_db_path}")
            blob.download_to_filename(self.local_db_path)
        else:
            logger.info(f"No existing index found at {self.db_path}, creating new database")
        
        # Connect and create tables if needed
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create files table - simplified schema
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            file_hash TEXT PRIMARY KEY,
            relative_path TEXT,
            source_url TEXT,
            destination_blob TEXT,
            status TEXT,
            timestamp TEXT,
            error TEXT,
            file_size INTEGER,
            metadata TEXT
        )
        ''')
        
        # Create index for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON files(status)')
        
        conn.commit()
    
    def _get_connection(self):
        """Get SQLite connection (create if needed) in a thread-safe manner."""
        # Use thread-local storage for connections
        if not hasattr(self, '_thread_local'):
            self._thread_local = threading.local()
        
        # Create a connection for this thread if it doesn't exist
        if not hasattr(self._thread_local, 'conn') or self._thread_local.conn is None:
            self._thread_local.conn = sqlite3.connect(self.local_db_path)
            # Enable optimizations
            self._thread_local.conn.execute("PRAGMA foreign_keys = ON")
            self._thread_local.conn.execute("PRAGMA journal_mode = WAL")
            self._thread_local.conn.execute("PRAGMA synchronous = NORMAL")
            
        return self._thread_local.conn
    
    def _get_existing_gcs_files(self):
        """Get a set of existing files in GCS using the blob list cache if available."""
        gcs_client = GCSClient(self.bucket_name)
        
        logger.info(f"Fetching existing files in GCS with prefix '{self.data_path}'")
        # Try to load existing blob list from GCS
        blob_list = self.bucket.blob(self.blob_list_path)
        if blob_list.exists():
            logger.info(f"Loading existing blob list from {self.blob_list_path}")
            blob_list_json = blob_list.download_as_text()
            existing_files = set(json.loads(blob_list_json))
        else:
            logger.info(f"Creating new blob list")
            existing_files = gcs_client.list_existing_files(prefix=self.data_path)
            # Store for future use
            blob_list.upload_from_string(json.dumps(list(existing_files)))
        logger.info(f"Found {len(existing_files)} existing files in GCS with prefix '{self.data_path}'")
        
        return existing_files
        
    def _load_entrypoints(self, data_source):
        """Load or generate entrypoints from data source."""
        if not hasattr(data_source, "has_entrypoints") or not data_source.has_entrypoints:
            logger.info("Data source has no entrypoints")
            return []
            
        # Try to load entrypoints from GCS cache first, compute if unavailable
        blob = self.bucket.blob(self.entrypoints_path)
        all_entrypoints = []
        
        try:
            if blob.exists():
                logger.info(f"Loading entrypoints from cache at {self.entrypoints_path}")
                entrypoints_json = blob.download_as_text()
                all_entrypoints = json.loads(entrypoints_json)
                logger.info(f"Loaded {len(all_entrypoints)} entrypoints from cache")
            else:
                logger.info("No cached entrypoints found, computing from data source")
                all_entrypoints = data_source.get_all_entrypoints()
                logger.info(f"Computed {len(all_entrypoints)} total entrypoints")
                
                # Cache for future use
                logger.info(f"Caching entrypoints to {self.entrypoints_path}")
                blob.upload_from_string(json.dumps(all_entrypoints))
                
        except Exception as e:
            logger.error(f"Failed to get entrypoints: {e}")
            if getattr(data_source, "has_entrypoints", False):
                raise ValueError("Cannot build index without entrypoints")
        
        if getattr(data_source, "has_entrypoints", False) and not all_entrypoints:
            raise ValueError("No entrypoints found - cannot build index")
            
        return all_entrypoints
    
    def _get_missing_entrypoints(self, data_source, all_entrypoints):
        """Identify which entrypoints have not yet been processed."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get processed entrypoints from the database
        processed_entrypoints = set()
        cursor.execute("SELECT DISTINCT relative_path FROM files")
        for (relative_path,) in cursor.fetchall():
            ep = data_source.filename_to_entrypoint(relative_path)
            if ep:
                # Create a hashable representation of the entrypoint
                ep_key = f"{ep['year']}_{ep['day']}"
                processed_entrypoints.add(ep_key)
        
        logger.info(f"Found {len(processed_entrypoints)} already processed entrypoints")
        
        # Find which entrypoints need to be processed
        missing_entrypoints = []
        for ep in all_entrypoints:
            ep_key = f"{ep['year']}_{ep['day']}"
            if ep_key not in processed_entrypoints:
                missing_entrypoints.append(ep)
        
        # Sort missing entrypoints by year and day for ordered processing
        missing_entrypoints.sort(key=lambda x: (x['year'], x['day']))
        logger.info(f"Found {len(missing_entrypoints)} missing entrypoints to process")
        
        return missing_entrypoints
    
    def _process_files(self, data_source, files_to_process, existing_files):
        """Process a list of files and add them to the index."""
        conn = self._get_connection()
        cursor = conn.cursor()
        batch_size = 1000
        batch = []
        total_indexed = 0
        
        # Process all files
        for i, (relative_path, file_url) in enumerate(files_to_process):
            if i > 0 and i % 100 == 0:
                logger.info(f"Processing file {i}/{len(files_to_process)}...")
            
            # Generate a hash of the URL
            url_hash = hashlib.md5(file_url.encode()).hexdigest()
            destination_blob = data_source.gcs_upload_path(data_source.base_url, relative_path)
            
            # Check if file already exists in index
            cursor.execute("SELECT file_hash FROM files WHERE file_hash = ?", (url_hash,))
            if cursor.fetchone():
                continue  # Skip if already indexed
            
            # Check if file already exists in GCS (using our pre-fetched set)
            status = "success" if destination_blob in existing_files else "indexed"
            
            # Add to batch
            batch.append((
                url_hash,
                relative_path,
                file_url, 
                destination_blob,
                status,
                datetime.now().isoformat(),
                None,
                None,
                None
            ))
            
            # Execute batch insert when batch is full
            if len(batch) >= batch_size:
                cursor.executemany('''
                INSERT INTO files 
                (file_hash, relative_path, source_url, destination_blob, 
                 status, timestamp, error, file_size, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', batch)
                conn.commit()
                total_indexed += len(batch)
                batch = []
                
                # Log progress
                logger.info(f"Indexed {total_indexed} files")
                
                # Periodic save
                self._check_periodic_save()
        
        # Commit any remaining files
        if batch:
            cursor.executemany('''
            INSERT INTO files 
            (file_hash, relative_path, source_url, destination_blob, 
             status, timestamp, error, file_size, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', batch)
            conn.commit()
            total_indexed += len(batch)
            
            logger.info(f"Indexed {total_indexed} files")
            
        return total_indexed
    
    def refresh_index(self, data_source):
        """
        Build or refresh the index from the data source.
        Scans source files and adds missing items to the index.
        
        Args:
            data_source: The data source to scan for files
            
        Returns:
            int: Number of new files indexed
        """
        logger.info("Starting to refresh index from data source")
        total_indexed = 0
        
        try:
            # Get all existing files in GCS
            existing_files = self._get_existing_gcs_files()
            
            # Load or generate entrypoints
            all_entrypoints = self._load_entrypoints(data_source)
            
            # Process files based on whether the data source has entrypoints
            if getattr(data_source, "has_entrypoints", False):
                # Find which entrypoints need to be processed
                missing_entrypoints = self._get_missing_entrypoints(data_source, all_entrypoints)
                
                # If no missing entrypoints, we're done
                if not missing_entrypoints:
                    logger.info("No missing entrypoints found. Index is up to date.")
                    return 0
                
                # Process each missing entrypoint
                for i, entrypoint in enumerate(missing_entrypoints):
                    logger.info(f"Processing entrypoint {i+1}/{len(missing_entrypoints)}: year={entrypoint['year']}, day={entrypoint['day']}")
                    
                    # Call list_remote_files with this specific entrypoint
                    entrypoint_files = list(data_source.list_remote_files(entrypoint))
                    
                    if not entrypoint_files:
                        logger.warning(f"No files found for entrypoint: year={entrypoint['year']}, day={entrypoint['day']}")
                        continue
                        
                    logger.info(f"Found {len(entrypoint_files)} files for entrypoint: year={entrypoint['year']}, day={entrypoint['day']}")
                    
                    # Process all files for this entrypoint
                    indexed = self._process_files(data_source, entrypoint_files, existing_files)
                    total_indexed += indexed
                    
                    # Update the metadata after each entrypoint
                    self.metadata["last_processed_entrypoint"] = {
                        "year": entrypoint["year"],
                        "day": entrypoint["day"],
                        "timestamp": datetime.now().isoformat()
                    }
                    self._save_metadata()
                    
                    # Periodic save
                    self._check_periodic_save()
            else:
                # No entrypoints available, process all files directly
                logger.info("Data source has no entrypoints, scanning all available files")
                
                # Process all files without using entrypoints
                all_files = list(data_source.list_remote_files())
                logger.info(f"Found {len(all_files)} files to process")
                
                # Process all files
                total_indexed = self._process_files(data_source, all_files, existing_files)
                
        except Exception as e:
            logger.error(f"Error building index: {e}")
            raise
        
        finally:
            # Make sure we commit any remaining changes
            conn = self._get_connection()
            conn.commit()
            
            # Update metadata stats
            self._update_stats(total_indexed)
            
            # Final save
            self.save()
        
        return total_indexed
    
    def _check_periodic_save(self, force=False):
        """Check if we should perform a periodic save and do it if needed."""
        current_time = time.time()
        elapsed = current_time - self._last_save_time
        
        if force or elapsed >= self.save_interval_seconds:
            logger.info(f"Performing periodic save (elapsed: {elapsed:.1f}s)")
            self.save()
            self._last_save_time = current_time
            return True
        return False
    
    def _update_stats(self, total_indexed=0):
        """Update and log statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Count files by status
        cursor.execute("SELECT status, COUNT(*) FROM files GROUP BY status")
        status_counts = dict(cursor.fetchall())
        
        # Update metadata stats
        if "stats" not in self.metadata:
            self.metadata["stats"] = {}
        self.metadata["stats"]["total_indexed"] = total_indexed
        self.metadata["stats"]["successful_downloads"] = status_counts.get("success", 0)
        self.metadata["stats"]["failed_downloads"] = status_counts.get("failed", 0)
        self.metadata["stats"]["pending_downloads"] = status_counts.get("indexed", 0)
        self._save_metadata()
        
        # Log detailed results
        if "stats" in self.metadata:
            stats = self.metadata["stats"]
            already_success = stats.get("successful_downloads", 0)
            pending = stats.get("pending_downloads", 0)
            logger.info(f"Completed indexing: {total_indexed} files indexed")
            logger.info(f"  - {already_success} files already exist in GCS")
            logger.info(f"  - {pending} files need to be downloaded")
    
    def get_file_status(self, file_hash: str) -> Tuple[str, Optional[str]]:
        """Get status of a specific file by hash."""
        # Use cache for common lookups
        cache_key = f"file_status_{file_hash}"
        if cache_key in self._status_cache:
            return self._status_cache[cache_key]
        
        # Query database
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status, error FROM files WHERE file_hash = ?", 
            (file_hash,)
        )
        result = cursor.fetchone()
        
        if result:
            status, error = result
            # Cache result
            if len(self._status_cache) < self._max_cache_size:
                self._status_cache[cache_key] = (status, error)
            return status, error
        
        # File not found
        return "unknown", None

    def record_download_status(self, file_hash: str, source_url: str, 
                              destination_blob: str, status: str, error: str = None):
        """Update the status of a file download."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Update status
            cursor.execute(
                "UPDATE files SET status = ?, timestamp = ?, error = ? WHERE file_hash = ?",
                (status, datetime.now().isoformat(), error, file_hash)
            )
            conn.commit()
            
            # Update stats
            if "stats" not in self.metadata:
                self.metadata["stats"] = {}
                
            if status == "success":
                if "successful_downloads" not in self.metadata["stats"]:
                    self.metadata["stats"]["successful_downloads"] = 0
                self.metadata["stats"]["successful_downloads"] += 1
            
            elif status == "failed":
                if "failed_downloads" not in self.metadata["stats"]:
                    self.metadata["stats"]["failed_downloads"] = 0
                self.metadata["stats"]["failed_downloads"] += 1
            
            # Save metadata periodically
            if (self.metadata["stats"].get("successful_downloads", 0) +
                self.metadata["stats"].get("failed_downloads", 0)) % 100 == 0:
                self._save_metadata()
                
                # Check for periodic save of the whole database
                self._check_periodic_save()
            
            # Update cache
            cache_key = f"file_status_{file_hash}"
            self._status_cache[cache_key] = (status, error)
    
    def iter_pending_downloads(self, chunk_size=10) -> Generator[Dict, None, None]:
        """
        Generator that yields pending downloads one at a time.
        Uses small chunks internally for efficiency but exposes a one-by-one interface.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get column names once
        cursor.execute("SELECT * FROM files WHERE 1=0")
        columns = [description[0] for description in cursor.description]
        
        # Keep track of last processed ID for pagination
        last_id = 0
        
        while True:
            # Fetch the next chunk of pending files
            cursor.execute("""
                SELECT * FROM files 
                WHERE status = 'indexed' AND rowid > ? 
                ORDER BY rowid
                LIMIT ?
            """, (last_id, chunk_size))
            
            rows = cursor.fetchall()
            if not rows:
                break  # No more pending files
                
            # Process each file in the chunk
            for row in rows:
                file_info = dict(zip(columns, row))
                
                # Convert metadata if needed
                if file_info.get("metadata") and isinstance(file_info["metadata"], str):
                    try:
                        file_info["metadata"] = json.loads(file_info["metadata"])
                    except:
                        pass  # Keep as string if not valid JSON
                
                # Update last_id for next query
                last_id = cursor.lastrowid or last_id + 1
                
                # Yield one file at a time
                yield file_info
    
    def get_stats(self) -> Dict:
        """Get download statistics."""
        # Basic stats from metadata
        basic_stats = self.metadata.get("stats", {})
        
        # Add live stats from database
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Count files by status
        cursor.execute("SELECT status, COUNT(*) FROM files GROUP BY status")
        for status, count in cursor.fetchall():
            basic_stats[f"files_{status}"] = count
            
        return basic_stats
    
    def save(self):
        """Save the index database and metadata to GCS."""
        logger.info(f"Saving index database to {self.db_path}")
        
        # Close connection to ensure all data is written
        if self._conn:
            self._conn.close()
            self._conn = None
        
        # Upload database to GCS
        self.bucket.blob(self.db_path).upload_from_filename(self.local_db_path)
        
        # Save metadata
        self._save_metadata()
        
        # Update last save time
        self._last_save_time = time.time()
        
        # Reconnect to database
        _ = self._get_connection()
        
    def is_new_index(self) -> bool:
        """Check if this is a new index or it already exists."""
        # If database is empty, it's a new index
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM files")
        file_count = cursor.fetchone()[0]
        
        return file_count == 0
    
    def remove_blob_list(self):
        """Remove the blob list from GCS."""
        try:
            blob = self.bucket.blob(self.blob_list_path)
            if blob.exists():
                logger.info(f"Removing blob list from {self.blob_list_path}")
                blob.delete()
            else:
                logger.info(f"No blob list found at {self.blob_list_path}")
        except Exception as e:
            logger.error(f"Error removing blob list: {e}")
            raise
    
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
    
    def list_successful_files(self, prefix: str = None) -> List[str]:
        """
        List all successfully downloaded files with the given prefix.
        
        This is a convenience method for preprocessors to efficiently query the index
        without needing to directly interact with the SQLite database or understand
        its schema.
        
        Args:
            prefix: Optional path prefix to filter results
            
        Returns:
            List of full blob paths for successfully downloaded files
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if prefix:
            # Query with prefix filter
            cursor.execute(
                """
                SELECT destination_blob FROM files 
                WHERE status = 'success' AND 
                      destination_blob LIKE ? || '%'
                """,
                (prefix,)
            )
        else:
            # Query all successful downloads
            cursor.execute(
                """
                SELECT destination_blob FROM files 
                WHERE status = 'success'
                """
            )
        
        # Extract the file paths
        files = [row[0] for row in cursor.fetchall()]
        
        return files
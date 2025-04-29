from gnt.data.common.index.base_index import BaseIndex
from gnt.data.common.gcs.client import GCSClient
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

logger = logging.getLogger(__name__)

class DataDownloadIndex(BaseIndex):
    """Memory-efficient SQLite-based index for downloading files."""
    
    def __init__(self, bucket_name: str, data_source, data_source_name: str, client=None, temp_dir=None):
        # Initialize base index with appropriate paths
        super().__init__(
            bucket_name=bucket_name,
            index_name=data_source_name, 
            client=client
        )
        
        self.data_path = getattr(data_source, "data_path", "unknown")
        self.data_source_name = data_source_name
        
        # Paths for the SQLite database in GCS
        self.db_path = f"_index/download_{self.data_path.replace('/', '_')}.sqlite"
        self.blob_list_path = f"_index/download_{self.data_path.replace('/', '_')}.json"
        
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
        self._temp_dir = os.path.join(self._temp_base_dir, f"index_{self.data_source_name}_{self._instance_id}")
        os.makedirs(self._temp_dir, exist_ok=True)
        
        # Local path for the SQLite database
        self.local_db_path = os.path.join(self._temp_dir, f"download_{self.data_path.replace('/', '_')}.sqlite")
        
        # Database connection
        self._conn = None
        self._lock = threading.RLock()
        
        # Set up status cache
        self._status_cache = {}
        self._max_cache_size = 10000
        
        # Download existing database or create new one
        self._setup_database()
        self._load_metadata()
        self._build_from_source(data_source)          # Build the index if necessary (new extraction)
    
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
        """Get SQLite connection (create if needed)."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.local_db_path)
            # Enable optimizations
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
        return self._conn
    
    def _build_from_source(self, data_source):
        """
        Build index directly from data source.
        Scans source files and adds them to the index.
        Uses a single GCS API call to check all existing files.
        """
        logger.info("Starting to build index from data source")
        total_indexed = 0
        conn = self._get_connection()
        
        try:
            # First, get all existing files in GCS with a single API call using data_path as prefix
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
            
            # Start a transaction for better performance
            with conn:
                cursor = conn.cursor()
                # Process source files in streaming mode
                batch_size = 1000
                batch = []
                
                # Initialize entrypoint as None, will be determined based on database state
                entrypoint = None

                # Check if we already have files in the database
                cursor.execute("SELECT COUNT(*) FROM files")
                file_count = cursor.fetchone()[0]

                if file_count > 0:
                    # Determine the best starting point for crawling
                    # Either find the earliest missing year/day combination or use the latest one
                    
                    # First, identify all year/day combinations we already have
                    cursor.execute("""
                        SELECT SUBSTR(relative_path, 1, INSTR(SUBSTR(relative_path, INSTR(relative_path, '/') + 1), '/') + INSTR(relative_path, '/')) AS yearday
                        FROM files
                        GROUP BY yearday
                        ORDER BY yearday
                    """)
                    
                    existing_yeardays = set(row[0] for row in cursor.fetchall())
                    
                    # Get all expected year/day combinations from the data source
                    expected_yeardays = set(data_source.get_all_entrypoints())
                    
                    # Find missing year/day combinations
                    missing_yeardays = expected_yeardays - existing_yeardays
                    
                    if missing_yeardays:
                        # Find the minimum (earliest) missing year/day
                        entrypoint = min(missing_yeardays)
                        logger.info(f"Found earliest missing year/day: {entrypoint}")
                    else:
                        # No missing entries, get the latest entry to continue from there
                        cursor.execute("""
                            SELECT relative_path 
                            FROM files 
                            ORDER BY SUBSTR(relative_path, 1, INSTR(relative_path, '/', 1, 2) - 1) DESC
                            LIMIT 1
                        """)
                        last_entry = cursor.fetchone()
                        if last_entry:
                            entrypoint = data_source.filename_to_entrypoint(last_entry[0])
                            logger.info(f"No missing entries, continuing from latest: {entrypoint}")
                
                cursor.execute("SELECT relative_path FROM files ORDER BY rowid DESC LIMIT 1")
                
                # Get the entrypoint from the relative_path of the last file
                last_entry = cursor.fetchone()
                if last_entry:
                    entrypoint = data_source.filename_to_entrypoint(last_entry[0])
                
                # Process all remote files
                for relative_path, file_url in data_source.list_remote_files(entrypoint):
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
                        logger.info(f"Indexed {total_indexed} files so far")
                
                # Insert any remaining items in the batch
                if batch:
                    cursor.executemany('''
                    INSERT INTO files 
                    (file_hash, relative_path, source_url, destination_blob, 
                     status, timestamp, error, file_size, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', batch)
                    conn.commit()
                    total_indexed += len(batch)
                    logger.info(f"Indexed final batch: {len(batch)} files")
        except Exception as e:
            logger.error(f"Error building index: {e}")
            raise
        finally:
            # Make sure we commit any remaining changes
            conn.commit()
            
            # Count files by status
            cursor = conn.cursor()
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
        
        return total_indexed
    
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
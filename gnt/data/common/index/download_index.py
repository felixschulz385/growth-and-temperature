from gnt.data.common.index.base_index import BaseIndex
from gnt.data.common.gcs.client import GCSClient
from gnt.data.download.sources.base import BaseDataSource
from typing import Dict, List, Iterator, Tuple, Optional, Generator
import os
import json
import logging
import threading
import sqlite3
import tempfile
import hashlib
from datetime import datetime
import uuid
import time
import shutil

logger = logging.getLogger(__name__)

class DataDownloadIndex(BaseIndex):
    """Memory-efficient SQLite-based index for managing file downloads."""
    
    def __init__(self, bucket_name: str, data_source: BaseDataSource, client=None, 
                 temp_dir=None, auto_index=False, save_interval_seconds=300):
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
        
        # Instance identifier for thread safety
        self._instance_id = str(uuid.uuid4())[:8]
        
        # Set up directories
        self._setup_temp_dir(temp_dir)
        
        # Threading resources
        self._lock = threading.RLock()
        self._thread_local = threading.local()
        
        # Status cache to reduce database access
        self._status_cache = {}
        self._max_cache_size = 10000
        
        # Autosave settings
        self.save_interval_seconds = save_interval_seconds
        self._last_save_time = time.time()
        
        # Setup database
        self._setup_database()
        self._load_metadata()
        
        # Build index if requested
        if auto_index:
            self.refresh_index(data_source)
    
    def _setup_temp_dir(self, temp_dir):
        """Set up temporary directory for the index."""
        if temp_dir:
            self._temp_base_dir = os.path.abspath(temp_dir)
        else:
            self._temp_base_dir = os.path.join(tempfile.gettempdir(), "gnt_indices")
            
        os.makedirs(self._temp_base_dir, exist_ok=True)
        self._temp_dir = os.path.join(self._temp_base_dir, f"index_{self.data_source_name}_{self._instance_id}")
        os.makedirs(self._temp_dir, exist_ok=True)
        self.local_db_path = os.path.join(self._temp_dir, f"download_{self.data_path.replace('/', '_')}.sqlite")
    
    def _setup_database(self):
        """Set up SQLite database by downloading existing or creating new."""
        blob = self.bucket.blob(self.db_path)
        try:
            if blob.exists():
                logger.info(f"Downloading existing index from {self.db_path}")
                blob.download_to_filename(self.local_db_path)
            else:
                logger.info(f"Creating new index database")
        except Exception as e:
            logger.error(f"Error accessing GCS index: {e}. Creating new database.")
        
        # Connect and create tables
        conn = self._get_connection()
        cursor = conn.cursor()
        
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
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON files(status)')
        conn.commit()
    
    def _get_connection(self):
        """Get thread-local SQLite connection."""
        if not hasattr(self._thread_local, 'conn') or self._thread_local.conn is None:
            self._thread_local.conn = sqlite3.connect(self.local_db_path)
            
            # Performance optimizations
            self._thread_local.conn.execute("PRAGMA journal_mode = WAL")
            self._thread_local.conn.execute("PRAGMA synchronous = NORMAL")
            self._thread_local.conn.execute("PRAGMA temp_store = MEMORY")
            self._thread_local.conn.execute("PRAGMA cache_size = 5000")
            
        return self._thread_local.conn
    
    def refresh_index(self, data_source):
        """Build or update the index from the data source."""
        logger.info(f"Refreshing index from {self.data_source_name}")
        total_indexed = 0
        
        try:
            # Get existing files in GCS
            existing_files = self._get_existing_files()
            
            # Process files based on data source capabilities
            if getattr(data_source, "has_entrypoints", False):
                # Process entrypoint-based data sources
                all_entrypoints = self._load_entrypoints(data_source)
                missing_entrypoints = self._find_missing_entrypoints(data_source, all_entrypoints)
                
                if not missing_entrypoints:
                    logger.info("No missing entrypoints found. Index is up to date.")
                    return 0
                
                # Process each missing entrypoint
                for i, entrypoint in enumerate(missing_entrypoints):
                    logger.info(f"Processing entrypoint {i+1}/{len(missing_entrypoints)}: "
                                f"year={entrypoint['year']}, day={entrypoint['day']}")
                    
                    entrypoint_files = list(data_source.list_remote_files(entrypoint))
                    
                    if not entrypoint_files:
                        logger.warning(f"No files found for entrypoint: "
                                      f"year={entrypoint['year']}, day={entrypoint['day']}")
                        continue
                    
                    logger.info(f"Found {len(entrypoint_files)} files for this entrypoint")
                    
                    # Process this entrypoint's files
                    indexed = self._add_files_to_index(data_source, entrypoint_files, existing_files)
                    total_indexed += indexed
                    
                    # Update metadata after each entrypoint
                    self.metadata["last_processed_entrypoint"] = {
                        "year": entrypoint["year"],
                        "day": entrypoint["day"],
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Periodic save
                    self._check_save_needed(force=False)
            else:
                # Process simple data sources
                logger.info("Processing all available files")
                all_files = list(data_source.list_remote_files())
                logger.info(f"Found {len(all_files)} files")
                total_indexed = self._add_files_to_index(data_source, all_files, existing_files)
                
        except Exception as e:
            logger.error(f"Error refreshing index: {e}", exc_info=True)
            raise
        finally:
            # Ensure we commit changes and update stats
            try:
                conn = self._get_connection()
                conn.commit()
                self._update_stats(total_indexed)
                self.save()
            except Exception as e:
                logger.error(f"Error during final index save: {e}")
        
        return total_indexed
    
    def _get_existing_files(self):
        """Get set of existing files in GCS, using cached blob list if available."""
        gcs_client = GCSClient(self.bucket_name)
        
        blob_list = self.bucket.blob(self.blob_list_path)
        try:
            if blob_list.exists():
                logger.info(f"Loading cached blob list")
                blob_list_json = blob_list.download_as_text()
                existing_files = set(json.loads(blob_list_json))
            else:
                logger.info(f"Creating new blob list for {self.data_path}")
                existing_files = gcs_client.list_existing_files(prefix=self.data_path)
                blob_list.upload_from_string(json.dumps(list(existing_files)))
                
            logger.info(f"Found {len(existing_files)} existing files in GCS")
            return existing_files
        except Exception as e:
            logger.warning(f"Error getting blob list: {e}. Will assume no existing files.")
            return set()
    
    def _load_entrypoints(self, data_source):
        """Load or generate entrypoints for the data source."""
        if not getattr(data_source, "has_entrypoints", False):
            return []
        
        blob = self.bucket.blob(self.entrypoints_path)
        try:
            if blob.exists():
                logger.info(f"Loading cached entrypoints")
                entrypoints_json = blob.download_as_text()
                all_entrypoints = json.loads(entrypoints_json)
                logger.info(f"Loaded {len(all_entrypoints)} entrypoints from cache")
            else:
                logger.info("Computing entrypoints from data source")
                all_entrypoints = data_source.get_all_entrypoints()
                logger.info(f"Computed {len(all_entrypoints)} entrypoints")
                blob.upload_from_string(json.dumps(all_entrypoints))
        except Exception as e:
            logger.error(f"Failed to get entrypoints: {e}")
            if getattr(data_source, "has_entrypoints", False):
                raise ValueError("Cannot build index without entrypoints")
            all_entrypoints = []
            
        if getattr(data_source, "has_entrypoints", False) and not all_entrypoints:
            raise ValueError("No entrypoints found - cannot build index")
            
        return all_entrypoints
    
    def _find_missing_entrypoints(self, data_source, all_entrypoints):
        """Find entrypoints that haven't been processed yet."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get all processed entrypoints from the database
        processed_entrypoints = set()
        cursor.execute("SELECT DISTINCT relative_path FROM files")
        for (relative_path,) in cursor.fetchall():
            ep = data_source.filename_to_entrypoint(relative_path)
            if ep:
                ep_key = f"{ep['year']}_{ep['day']}"
                processed_entrypoints.add(ep_key)
        
        # Find missing entrypoints
        missing_entrypoints = []
        for ep in all_entrypoints:
            ep_key = f"{ep['year']}_{ep['day']}"
            if ep_key not in processed_entrypoints:
                missing_entrypoints.append(ep)
        
        # Sort by year and day
        missing_entrypoints.sort(key=lambda x: (x['year'], x['day']))
        logger.info(f"Found {len(missing_entrypoints)} missing entrypoints")
        
        return missing_entrypoints
    
    def _add_files_to_index(self, data_source, files_to_process, existing_files):
        """Add new files to the index."""
        conn = self._get_connection()
        cursor = conn.cursor()
        batch_size = 1000
        batch = []
        total_indexed = 0
        
        for i, (relative_path, file_url) in enumerate(files_to_process):
            if i > 0 and i % 100 == 0:
                logger.debug(f"Processing file {i}/{len(files_to_process)}...")
            
            # Generate hash for the file
            url_hash = hashlib.md5(file_url.encode()).hexdigest()
            destination_blob = data_source.gcs_upload_path(data_source.base_url, relative_path)
            
            # Check if file already exists in index
            cursor.execute("SELECT file_hash FROM files WHERE file_hash = ?", (url_hash,))
            if cursor.fetchone():
                continue  # Skip if already indexed
            
            # Mark as successful if already in GCS, otherwise as indexed
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
                logger.info(f"Indexed {total_indexed} files")
                self._check_save_needed()
        
        # Insert any remaining files
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
    
    def _check_save_needed(self, force=False):
        """Check if we should save the index based on elapsed time."""
        current_time = time.time()
        elapsed = current_time - self._last_save_time
        
        if force or elapsed >= self.save_interval_seconds:
            logger.info(f"Performing periodic save (elapsed: {elapsed:.1f}s)")
            try:
                self.save()
            except Exception as e:
                logger.error(f"Error during periodic save: {e}")
            self._last_save_time = current_time
            return True
        return False
    
    def _update_stats(self, new_indexed=0):
        """Update index statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Count files by status
        cursor.execute("SELECT status, COUNT(*) FROM files GROUP BY status")
        status_counts = dict(cursor.fetchall())
        
        if "stats" not in self.metadata:
            self.metadata["stats"] = {}
            
        self.metadata["stats"]["total_indexed"] = (
            self.metadata["stats"].get("total_indexed", 0) + new_indexed
        )
        self.metadata["stats"]["successful_downloads"] = status_counts.get("success", 0)
        self.metadata["stats"]["failed_downloads"] = status_counts.get("failed", 0)
        self.metadata["stats"]["pending_downloads"] = status_counts.get("indexed", 0)
        
        self._save_metadata()
        
        if new_indexed > 0:
            stats = self.metadata["stats"]
            logger.info(f"Index statistics:")
            logger.info(f"  - {stats.get('successful_downloads', 0)} files already in GCS")
            logger.info(f"  - {stats.get('pending_downloads', 0)} files need to be downloaded")
    
    def get_file_status(self, file_hash: str) -> Tuple[str, Optional[str]]:
        """Get the status and error message for a file."""
        # Check cache first
        cache_key = f"file_status_{file_hash}"
        if cache_key in self._status_cache:
            return self._status_cache[cache_key]
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT status, error FROM files WHERE file_hash = ?", 
                (file_hash,)
            )
            result = cursor.fetchone()
            
            if result:
                status, error = result
                # Cache the result
                if len(self._status_cache) < self._max_cache_size:
                    self._status_cache[cache_key] = (status, error)
                return status, error
        except Exception as e:
            logger.error(f"Error getting file status: {e}")
            
        return "unknown", None

    def record_download_status(self, file_hash: str, source_url: str, 
                              destination_blob: str, status: str, error: str = None):
        """Update the status of a file download with thread safety."""
        with self._lock:
            try:
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
                
                # Periodic metadata save
                completed = (
                    self.metadata["stats"].get("successful_downloads", 0) + 
                    self.metadata["stats"].get("failed_downloads", 0)
                )
                if completed % 100 == 0:
                    self._save_metadata()
                    self._check_save_needed()
                
                # Update cache
                cache_key = f"file_status_{file_hash}"
                self._status_cache[cache_key] = (status, error)
                
            except Exception as e:
                logger.error(f"Error recording download status: {e}")
    
    def iter_pending_downloads(self, batch_size=100) -> Generator[Dict, None, None]:
        """Efficiently iterate through files that need to be downloaded."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get column names once
            cursor.execute("SELECT * FROM files WHERE 1=0")
            columns = [description[0] for description in cursor.description]
            
            # Use pagination for memory efficiency
            offset = 0
            while True:
                cursor.execute("""
                    SELECT * FROM files 
                    WHERE status = 'indexed'
                    ORDER BY rowid
                    LIMIT ? OFFSET ?
                """, (batch_size, offset))
                
                rows = cursor.fetchall()
                if not rows:
                    break  # No more files to process
                
                # Process each file in the batch
                for row in rows:
                    file_info = dict(zip(columns, row))
                    
                    # Parse metadata if it exists
                    if file_info.get("metadata") and isinstance(file_info["metadata"], str):
                        try:
                            file_info["metadata"] = json.loads(file_info["metadata"])
                        except:
                            pass  # Keep as string if not valid JSON
                    
                    yield file_info
                
                # Move to next batch
                offset += batch_size
                
        except Exception as e:
            logger.error(f"Error iterating pending downloads: {e}")
    
    def get_stats(self) -> Dict:
        """Get download statistics."""
        try:
            # Get basic stats from metadata
            stats = self.metadata.get("stats", {}).copy()
            
            # Add current counts from database
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT status, COUNT(*) FROM files GROUP BY status")
            for status, count in cursor.fetchall():
                stats[f"files_{status}"] = count
                
            return stats
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
    
    def save(self):
        """Save the index database to GCS."""
        logger.info("Saving index to GCS")
        try:
            # Close all connections to ensure data is flushed
            if hasattr(self, '_thread_local') and hasattr(self._thread_local, 'conn') and self._thread_local.conn:
                self._thread_local.conn = None
                
            # Upload to GCS
            blob = self.bucket.blob(self.db_path)
            blob.upload_from_filename(self.local_db_path)
            
            # Save metadata
            self._save_metadata()
            
            # Update last save time
            self._last_save_time = time.time()
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
        finally:
            # Ensure we can reconnect if needed
            if hasattr(self._thread_local, 'conn'):
                self._thread_local.conn = None
    
    def list_successful_files(self, prefix: str = None) -> List[str]:
        """List all successfully downloaded files with optional prefix filter."""
        try:
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
                # All successful downloads
                cursor.execute(
                    """
                    SELECT destination_blob FROM files 
                    WHERE status = 'success'
                    """
                )
            
            # Extract and return file paths
            return [row[0] for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Error listing successful files: {e}")
            return []
    
    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        try:
            if os.path.exists(self._temp_dir):
                logger.info(f"Cleaning up temp directory: {self._temp_dir}")
                shutil.rmtree(self._temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save and clean up."""
        try:
            self.save()
        finally:
            self._cleanup_temp_files()
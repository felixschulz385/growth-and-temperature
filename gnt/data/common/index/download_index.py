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
                 temp_dir=None, save_interval_seconds=300):
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
        self.blob_list_path = f"_index/download_{self.data_path.replace('/', '_')}_bloblist.json"
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
        """
        DEPRECATED: Use build_index_from_source instead.
        This method is maintained for backward compatibility.
        """
        logger.warning(
            "refresh_index() is deprecated and will be removed in a future version. "
            "Use build_index_from_source(data_source, check_gcs=True, only_missing_entrypoints=True) instead."
        )
        return self.build_index_from_source(
            data_source, 
            rebuild=False, 
            check_gcs=True, 
            only_missing_entrypoints=True
        )
    
    def build_index_from_source(self, data_source, rebuild=False, check_gcs=False, 
                                only_missing_entrypoints=True, force_refresh_gcs=False):
        """
        Build index from data source with configurable behavior.
        
        Args:
            data_source: Data source to index
            rebuild: Whether to rebuild the index from scratch
            check_gcs: Whether to check GCS for existing files
            only_missing_entrypoints: Only process entrypoints not already in index
            force_refresh_gcs: Whether to force a refresh of the GCS file list
        
        Returns:
            int: Number of files indexed
        """
        logger.info(f"Building index for {data_source.DATA_SOURCE_NAME} from remote sources")
        total_indexed = 0
        
        # Clear existing index if requested
        if rebuild:
            logger.info("Rebuilding index from scratch")
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM files")
            conn.commit()
        
        try:
            # Check for existing files in GCS if requested
            existing_files = set()
            if check_gcs:
                existing_files = self._get_existing_files(force_refresh=force_refresh_gcs)
                
            # Process files based on data source capabilities
            if hasattr(data_source, "has_entrypoints") and data_source.has_entrypoints:
                # Process entrypoint-based data sources
                all_entrypoints = self._load_entrypoints(data_source)
                logger.info(f"Found {len(all_entrypoints)} entrypoints to process")
                
                # Filter to only missing entrypoints if requested
                entrypoints_to_process = all_entrypoints
                if only_missing_entrypoints:
                    entrypoints_to_process = self._find_missing_entrypoints(data_source, all_entrypoints)
                    logger.info(f"Filtered to {len(entrypoints_to_process)} missing entrypoints")
                    
                    if not entrypoints_to_process:
                        logger.info("No missing entrypoints found. Index is up to date.")
                        return 0
                
                # Process each entrypoint
                for i, entrypoint in enumerate(entrypoints_to_process):
                    logger.info(f"Processing entrypoint {i+1}/{len(entrypoints_to_process)}: {entrypoint}")
                    
                    try:
                        remote_files = list(data_source.list_remote_files(entrypoint))
                        
                        if remote_files:
                            logger.info(f"Found {len(remote_files)} files for this entrypoint")
                            # Add files to index
                            indexed = self._add_files_to_index(data_source, remote_files, existing_files)
                            total_indexed += indexed
                            
                            # Update metadata after each entrypoint
                            self.metadata["last_processed_entrypoint"] = entrypoint
                        else:
                            logger.warning(f"No files found for entrypoint: {entrypoint}")
                    except Exception as e:
                        logger.error(f"Error processing entrypoint {entrypoint}: {e}")
                    
                    # Periodic save
                    self._check_save_needed(force=False)
            else:
                # Simple data source
                logger.info("Processing all available files")
                try:
                    remote_files = list(data_source.list_remote_files())
                    logger.info(f"Found {len(remote_files)} remote files")
                    total_indexed = self._add_files_to_index(data_source, remote_files, existing_files)
                except Exception as e:
                    logger.error(f"Error listing remote files: {e}")
                
        except Exception as e:
            logger.error(f"Error building index: {e}", exc_info=True)
            raise
        finally:
            # Ensure we commit changes and update stats
            try:
                conn = self._get_connection()
                conn.commit()
                self._update_stats(total_indexed)
            except Exception as e:
                logger.error(f"Error during final index update: {e}")

        return total_indexed
    
    def _get_existing_files(self, force_refresh=False):
        """
        Get set of existing files in GCS with option to refresh.
        
        Args:
            force_refresh: If True, always fetch a fresh list from GCS
            
        Returns:
            set: Set of blob paths that exist in GCS
        """
        from datetime import datetime, timedelta
        import json
        
        gcs_client = GCSClient(self.bucket_name)
        blob_list = self.bucket.blob(self.blob_list_path)
        
        try:
            # Check if we should use cached list
            if not force_refresh and blob_list.exists():
                try:
                    logger.info(f"Loading cached blob list")
                    blob_list_json = blob_list.download_as_text()
                    existing_files = set(json.loads(blob_list_json))
                    logger.info(f"Found {len(existing_files)} existing files in GCS (from cache)")
                    return existing_files
                except Exception as e:
                    logger.warning(f"Error accessing cached blob list: {e}, will refresh")
                    
            # If we get here, we need to refresh the list
            logger.info(f"Creating new blob list for {self.data_path}")
            
            # Collect all blob names with paging
            blob_paths = set()
            logger.info(f"Scanning GCS bucket for files with prefix {self.data_path}/")
            blob_iterator = gcs_client.bucket.list_blobs(prefix=f"{self.data_path}/")
            page_count = 0
            
            for page in blob_iterator.pages:
                page_count += 1
                page_blobs = {blob.name for blob in page}
                blob_paths.update(page_blobs)
                logger.info(f"Processed page {page_count}, found {len(page_blobs)} blobs (total: {len(blob_paths)})")
                
            logger.info(f"Found {len(blob_paths)} existing files in GCS")
            
            # Save the blob list for future use
            try:
                blob_list.upload_from_string(json.dumps(list(blob_paths)))
                logger.debug(f"Updated cached blob list at gs://{self.bucket_name}/{self.blob_list_path}")
            except Exception as e:
                logger.warning(f"Error saving blob list: {e}")
                
            return blob_paths
            
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
    
    def _check_save_needed(self, force=False, operations_since_save=0):
        """
        Check if index should be saved based on time elapsed or operations count.
        
        Args:
            force: Whether to force a save
            operations_since_save: Number of operations since last save
        
        Returns:
            bool: Whether save was performed
        """
        current_time = time.time()
        elapsed = current_time - self._last_save_time
        
        # Save if forced, time interval exceeded, or significant operations performed
        if force or elapsed >= self.save_interval_seconds or operations_since_save >= 1000:
            logger.info(f"Saving index (forced: {force}, elapsed: {elapsed:.1f}s, operations: {operations_since_save})")
            try:
                self.save()
            except Exception as e:
                logger.error(f"Error during save: {e}")
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
        """Update the status of a file download with consolidated save logic."""
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
                
                # Track operations since last save
                if not hasattr(self, '_ops_since_save'):
                    self._ops_since_save = 0
                self._ops_since_save += 1
                
                # Check if we should save based on operations count
                if self._ops_since_save >= 1000:
                    save_performed = self._check_save_needed(operations_since_save=self._ops_since_save)
                    if save_performed:
                        self._ops_since_save = 0
                
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
            self._close_all_connections()
            
            # Small delay to allow SQLite to release locks
            time.sleep(0.5)
            
            # Create a fresh connection for the checkpoint
            temp_conn = sqlite3.connect(self.local_db_path)
            temp_cursor = temp_conn.cursor()

            try:
                # Step 1: Force a checkpoint to sync WAL to main DB
                temp_cursor.execute("PRAGMA wal_checkpoint(FULL)")
                
                # Step 2: Switch journal mode to DELETE to close WAL file
                temp_cursor.execute("PRAGMA journal_mode = DELETE")
                
                # Step 3: Commit any pending transactions
                temp_conn.commit()
            finally:
                # Close the temporary connection
                temp_cursor.close()
                temp_conn.close()
                
            # Upload to GCS
            blob = self.bucket.blob(self.db_path)
            blob.upload_from_filename(self.local_db_path)
            
            # Save metadata
            self._save_metadata()
            
            # Update last save time
            self._last_save_time = time.time()
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            # Add more detail to the error message
            import traceback
            logger.debug(f"Full error: {traceback.format_exc()}")

    def _close_all_connections(self):
        """Close all database connections."""
        with self._lock:
            try:
                if hasattr(self, '_thread_local') and hasattr(self._thread_local, 'conn') and self._thread_local.conn:
                    self._thread_local.conn.commit()  # Ensure all changes are committed
                    self._thread_local.conn.close()   # Properly close the connection
                    self._thread_local.conn = None
            except Exception as e:
                logger.warning(f"Error closing thread-local connections: {e}")
    
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
    
    def validate_against_gcs(self, gcs_client, force_file_list_update=False):
        """
        Validate index against GCS bucket contents.
        Uses centralized status management.
        
        Args:
            gcs_client: GCSClient instance to use
            force_file_list_update: Whether to force a refresh of the GCS file list

        Returns:
            dict: Validation statistics
        """
        stats = {"updated": 0, "added": 0, "orphaned": 0}
        logger.info(f"Validating index against GCS for {self.data_source_name}")
        
        # Get all existing files in GCS using centralized method
        all_blobs = self._get_existing_files(force_refresh=force_file_list_update)
        logger.info(f"Found {len(all_blobs)} files in GCS")
        
        try:
            # Process in batches
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 1. Fix statuses for files that exist in GCS but aren't marked as success
            logger.info("Updating status for files that exist in GCS but aren't marked as successful")
            cursor.execute("SELECT file_hash, destination_blob, source_url, status FROM files WHERE status != 'success'")
            rows = cursor.fetchall()
            
            # Store all rows in memory to avoid cursor issues if connection is reset
            all_rows = list(rows)
            logger.info(f"Found {len(all_rows)} files to check for status updates")
            
            batch_size = 500
            for i in range(0, len(all_rows), batch_size):
                batch = all_rows[i:i+batch_size]
                for file_hash, destination_blob, source_url, status in batch:
                    if destination_blob in all_blobs:
                        # Use centralized status update method
                        self.record_download_status(
                            file_hash=file_hash,
                            source_url=source_url or "",
                            destination_blob=destination_blob,
                            status="success"
                        )
                        stats["updated"] += 1
            
            if stats["updated"] > 0 and i % batch_size == 0:
                logger.info(f"Updated {stats['updated']} of {len(all_rows)} file statuses so far")
        
            if stats["updated"] > 0:
                logger.info(f"Updated a total of {stats['updated']} file statuses to 'success'")
            
            # 2. Find and add files in GCS not tracked in index
            logger.info("Finding files in GCS that aren't in the index")
            cursor.execute("SELECT destination_blob FROM files")
            # Store results in memory to avoid cursor issues
            indexed_blobs = {row[0] for row in cursor.fetchall()}
            
            unindexed_blobs = all_blobs - indexed_blobs
            if unindexed_blobs:
                logger.info(f"Found {len(unindexed_blobs)} files in GCS not in the index")
                
                # Add these to index
                for i, blob_name in enumerate(unindexed_blobs):
                    # Check connection is still valid
                    if i % 100 == 0:
                        try:
                            conn.execute("SELECT 1")
                        except sqlite3.ProgrammingError:
                            # Reconnect if needed
                            logger.warning("Database connection lost, reconnecting")
                            conn = self._get_connection()
                            cursor = conn.cursor()
                    
                    relative_path = blob_name.replace(f"{self.data_path}/", "", 1)
                    file_hash = hashlib.md5(relative_path.encode()).hexdigest()
                    
                    try:
                        # Insert the file first
                        cursor.execute(
                            """INSERT OR IGNORE INTO files 
                            (file_hash, relative_path, source_url, destination_blob, status, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?)""",
                            (file_hash, relative_path, None, blob_name, "indexed", 
                            datetime.now().isoformat())
                        )
                        
                        # Then update status using the centralized method
                        self.record_download_status(
                            file_hash=file_hash,
                            source_url=None,
                            destination_blob=blob_name,
                            status="success"
                        )
                        stats["added"] += 1
                        
                        if stats["added"] % 100 == 0:
                            conn.commit()
                            logger.info(f"Added {stats['added']} missing files to the index so far")
                    except Exception as e:
                        logger.warning(f"Error adding file {blob_name}: {e}")
                
                try:
                    conn.commit()
                except:
                    # Reconnect and continue if needed
                    conn = self._get_connection()
                    
                logger.info(f"Added a total of {stats['added']} missing files to the index")
            
            # 3. Find orphaned entries (marked as success but not in GCS)
            logger.info("Checking for orphaned entries (marked as success but missing from GCS)")
            cursor.execute("SELECT file_hash, destination_blob FROM files WHERE status = 'success'")
            
            # Store results in memory to avoid cursor issues
            success_files = list(cursor.fetchall())
            
            for i, (file_hash, destination_blob) in enumerate(success_files):
                # Check connection every so often
                if i % 100 == 0:
                    try:
                        conn.execute("SELECT 1")
                    except sqlite3.ProgrammingError:
                        # Reconnect if needed
                        logger.warning("Database connection lost, reconnecting")
                        conn = self._get_connection()
                
                if destination_blob not in all_blobs:
                    try:
                        # Use centralized status update method
                        self.record_download_status(
                            file_hash=file_hash,
                            source_url=None,
                            destination_blob=destination_blob,
                            status="indexed",
                            error="File marked as success but missing from GCS"
                        )
                        stats["orphaned"] += 1
                        
                        if stats["orphaned"] % 100 == 0:
                            logger.info(f"Found {stats['orphaned']} orphaned entries so far")
                    except Exception as e:
                        logger.warning(f"Error updating orphaned entry {file_hash}: {e}")
            
            if stats["orphaned"] > 0:
                logger.info(f"Found {stats['orphaned']} orphaned entries (reset to 'indexed' status)")
        
        except Exception as e:
            logger.error(f"Error during validation: {e}", exc_info=True)
            raise
            
        return stats


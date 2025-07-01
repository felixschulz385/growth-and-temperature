"""
SQLite-based index for managing file metadata with HPC synchronization.

This module provides a specialized index that works with HPC systems,
including features for synchronizing the index between local workstations 
and HPC clusters.
"""
import os
import time
import json
import logging
import sqlite3
import threading
from typing import Dict, Any, Optional, List, Union, Set, Tuple
from datetime import datetime
import subprocess
import tempfile
from pathlib import Path
import shutil

# Import base class
from gnt.data.common.index.download_index import DataDownloadIndex
from gnt.data.common.hpc.client import HPCClient, HPCIndexSynchronizer

logger = logging.getLogger(__name__)

class HPCDataDownloadIndex(DataDownloadIndex):
    """
    SQLite-based index for managing file metadata with HPC synchronization.
    
    This extends the base DataDownloadIndex with features specific to HPC environments:
    - Local index storage instead of cloud storage
    - Synchronization between local and HPC systems
    """
    
    def __init__(
        self, 
        bucket_name: str, 
        data_source, 
        local_index_dir: str = None, 
        client=None,
        temp_dir=None, 
        save_interval_seconds=300,
        key_file: str = None
    ):
        """
        Initialize the HPC index.
        
        Args:
            bucket_name: Storage bucket name (kept for compatibility)
            data_source: Data source for indexing
            local_index_dir: Local directory for index storage
            client: Storage client (optional but kept for interface compatibility)
            temp_dir: Temporary directory
            save_interval_seconds: How often to save the index
            key_file: Path to SSH private key file (optional)
        """
        # Set up HPC-specific attributes BEFORE calling parent init
        self.use_local_only = True
        self.local_index_dir = local_index_dir or os.path.expanduser("~/hpc_data_index")
        os.makedirs(self.local_index_dir, exist_ok=True)
        
        # Store key_file for use in synchronization
        self.key_file = key_file
        
        # Connection pooling improvements - SET BEFORE parent init
        self.connection_timeout = 120  # Longer timeout for HPC operations
        
        # Initialize base class
        super().__init__(
            bucket_name=bucket_name, 
            data_source=data_source,
            client=client,
            temp_dir=temp_dir,
            save_interval_seconds=save_interval_seconds
        )
        
        # Initialize last save time
        self._last_save_time = time.time()
    
    def _setup_database(self):
        """Set up SQLite database with optimized indexes for HPC operations."""
        # Set up local path FIRST
        self.local_db_path = os.path.join(
            self.local_index_dir, 
            f"index_{self.data_path.replace('/', '_')}.sqlite"
        )
        
        # Connect with optimized settings
        conn = self._get_connection(timeout=60)
        cursor = conn.cursor()
        
        # Set pragmas for better performance with large datasets
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=30000")  # 30 seconds
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA cache_size=50000")  # Much larger cache (50MB)
        cursor.execute("PRAGMA page_size=4096")     # Larger page size
        cursor.execute("PRAGMA mmap_size=1073741824")  # 1GB memory mapping
        cursor.execute("PRAGMA optimize")  # Auto-optimize for current workload
        
        # Create standard tables (same as parent)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            file_hash TEXT PRIMARY KEY,
            relative_path TEXT,
            source_url TEXT,
            destination_blob TEXT,
            timestamp TEXT,
            file_size INTEGER,
            metadata TEXT
        )
        ''')
        
        # Create OPTIMIZED indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_timestamp ON files(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_relative_path ON files(relative_path)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_destination_blob ON files(destination_blob)')
        
        # CRITICAL: Analyze tables to update statistics for query optimizer
        cursor.execute('ANALYZE files')
        
        conn.commit()
    
    def _get_connection(self, timeout=30):
        """
        Get a SQLite database connection with improved pooling and error handling.
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            sqlite3.Connection: A connection to the database
        """
        # Use longer timeout for HPC operations
        timeout = max(timeout, self.connection_timeout)
        
        # Check if database exists, if not make sure directory exists
        db_dir = os.path.dirname(self.local_db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        # Try to recover the database if it's corrupted
        if os.path.exists(self.local_db_path) and self._is_database_corrupted():
            logger.warning(f"Database appears to be corrupted, attempting recovery")
            self._recover_database()
            
        # Create thread local storage if doesn't exist
        if not hasattr(self, '_thread_local'):
            self._thread_local = threading.local()
            
        # Create connection if it doesn't exist for this thread
        if not hasattr(self._thread_local, 'connection') or self._thread_local.connection is None:
            try:
                # Connect with optimized settings for bulk operations
                self._thread_local.connection = sqlite3.connect(
                    self.local_db_path, 
                    timeout=timeout,
                    isolation_level=None  # Autocommit mode
                )
                
                # Optimize for bulk operations
                conn = self._thread_local.connection
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute("PRAGMA busy_timeout=60000")  # 60 seconds for HPC operations
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")  # Larger cache for better performance
                conn.execute("PRAGMA temp_store=MEMORY")
                conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory mapping
                
            except sqlite3.Error as e:
                logger.error(f"Error connecting to database: {e}")
                self._recover_database()
                # Try connecting one more time
                self._thread_local.connection = sqlite3.connect(
                    self.local_db_path, 
                    timeout=timeout,
                    isolation_level=None
                )
        
        return self._thread_local.connection
    
    def _is_database_corrupted(self):
        """Check if the database is corrupted by attempting a simple query."""
        try:
            # Try to connect with a short timeout
            conn = sqlite3.connect(self.local_db_path, timeout=5)
            cursor = conn.cursor()
            # Simple query to check integrity
            cursor.execute("PRAGMA integrity_check(1)")
            result = cursor.fetchone()
            conn.close()
            # If result is not "ok", database is corrupted
            return result[0] != "ok"
        except sqlite3.Error:
            # Any error is a sign of corruption
            return True
    
    def _recover_database(self):
        """Attempt to recover a corrupted database or create a new one if recovery fails."""
        logger.warning(f"Attempting to recover database: {self.local_db_path}")
        
        # Create backup of corrupted database
        if os.path.exists(self.local_db_path):
            backup_path = f"{self.local_db_path}.corrupted.{int(time.time())}"
            try:
                shutil.copy2(self.local_db_path, backup_path)
                logger.info(f"Created backup of corrupted database: {backup_path}")
                
                # Try to recover using sqlite3 .dump command
                dump_path = f"{self.local_db_path}.dump"
                try:
                    # Dump the database to SQL commands
                    with open(dump_path, 'w') as f:
                        subprocess.run(
                            ["sqlite3", self.local_db_path, ".dump"],
                            stdout=f,
                            stderr=subprocess.PIPE,
                            check=False,
                            timeout=120
                        )
                    
                    # Remove corrupted database
                    os.remove(self.local_db_path)
                    
                    # Create new database from dump
                    subprocess.run(
                        ["sqlite3", self.local_db_path],
                        input=open(dump_path, 'r').read().encode(),
                        stderr=subprocess.PIPE,
                        check=False,
                        timeout=120
                    )
                    
                    # Check if recovery worked
                    if not self._is_database_corrupted():
                        logger.info("Database recovery successful")
                        # Clean up dump file
                        os.remove(dump_path)
                        return True
                    else:
                        logger.warning("Database recovery failed, will create new database")
                except Exception as e:
                    logger.error(f"Error during recovery attempt: {e}")
            except Exception as e:
                logger.error(f"Error creating backup: {e}")
        
        # If recovery failed or no database existed, create a new empty database
        if os.path.exists(self.local_db_path):
            try:
                os.remove(self.local_db_path)
            except Exception as e:
                logger.error(f"Error removing corrupted database: {e}")
                # If we can't remove it, create a new filename
                self.local_db_path = f"{self.local_db_path}.new.{int(time.time())}"
        
        logger.info(f"Creating new empty database at {self.local_db_path}")
        # Database will be created when _setup_database is called
        self._close_all_connections()
        self._thread_local = threading.local()
        return False

    def save(self):
        """Save index to local storage with improved error handling."""
        logger.info(f"Saving index for {self.data_source_name} to local storage")
        try:
            # Close the current thread's connection to ensure data is flushed
            self._close_all_connections()
            
            # Allow time for SQLite to release locks
            time.sleep(0.5)
            
            try:
                # Create a fresh connection for the checkpoint (in this thread)
                temp_conn = sqlite3.connect(self.local_db_path, timeout=60)
                temp_cursor = temp_conn.cursor()
    
                # First check integrity
                temp_cursor.execute("PRAGMA integrity_check(1)")
                result = temp_cursor.fetchone()
                if result[0] != "ok":
                    logger.error("Database integrity check failed before saving")
                    self._recover_database()
                    return
    
                # Force a checkpoint to sync WAL to main DB
                temp_cursor.execute("PRAGMA wal_checkpoint(FULL)")
                
                # Commit any pending transactions
                temp_conn.commit()
                
                # Close temporary connection
                temp_cursor.close()
                temp_conn.close()
                
                # Save metadata file locally
                meta_path = os.path.join(
                    self.local_index_dir, 
                    f"index_{self.data_path.replace('/', '_')}_meta.json"
                )
                with open(meta_path, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
                
                # Update last save time
                self._last_save_time = time.time()
            except sqlite3.Error as e:
                logger.error(f"SQLite error during save: {e}")
                # Attempt recovery if error indicates corruption
                if "malformed" in str(e).lower() or "corrupt" in str(e).lower():
                    self._recover_database()
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            import traceback
            logger.debug(f"Full error: {traceback.format_exc()}")
    
    def build_index_from_source(
        self, 
        data_source, 
        rebuild=False, 
        only_missing_entrypoints=True
    ):
        """
        Build index from data source with configurable behavior.
        
        Args:
            data_source: Data source to index
            rebuild: Whether to rebuild the index from scratch
            only_missing_entrypoints: Only process entrypoints not already in index
        
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
                            indexed = self._add_files_to_index(data_source, remote_files)
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
                    total_indexed = self._add_files_to_index(data_source, remote_files)
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

    def _load_entrypoints(self, data_source) -> List[Dict[str, Any]]:
        """
        Load or generate entrypoints for the data source.
        
        Args:
            data_source: Data source to load entrypoints from
            
        Returns:
            list: List of entrypoints
        """
        if not getattr(data_source, "has_entrypoints", False):
            return []
        
        # For HPC version, we store entrypoints locally
        entrypoints_file = os.path.join(
            self.local_index_dir,
            f"entrypoints_{self.data_path.replace('/', '_')}.json"
        )
        
        try:
            if os.path.exists(entrypoints_file):
                logger.info(f"Loading cached entrypoints from {entrypoints_file}")
                with open(entrypoints_file, 'r') as f:
                    all_entrypoints = json.load(f)
                logger.info(f"Loaded {len(all_entrypoints)} entrypoints from cache")
            else:
                logger.info("Computing entrypoints from data source")
                all_entrypoints = data_source.get_all_entrypoints()
                logger.info(f"Computed {len(all_entrypoints)} entrypoints")
                with open(entrypoints_file, 'w') as f:
                    json.dump(all_entrypoints, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to get entrypoints: {e}")
            if getattr(data_source, "has_entrypoints", False):
                raise ValueError("Cannot build index without entrypoints")
            all_entrypoints = []
            
        if getattr(data_source, "has_entrypoints", False) and not all_entrypoints:
            raise ValueError("No entrypoints found - cannot build index")
            
        return all_entrypoints

    def _find_missing_entrypoints(self, data_source, all_entrypoints) -> List[Dict[str, Any]]:
        """
        Find entrypoints that haven't been processed yet.
        
        Args:
            data_source: Data source to check entrypoints for
            all_entrypoints: List of all entrypoints
            
        Returns:
            list: List of missing entrypoints
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get all processed entrypoints from the database
        processed_entrypoints = set()
        cursor.execute("SELECT DISTINCT relative_path FROM files")
        for (relative_path,) in cursor.fetchall():
            ep = data_source.filename_to_entrypoint(relative_path)
            if ep:
                ep_key = f"{ep['year']}_{ep['day']}" if 'day' in ep else str(ep['year'])
                processed_entrypoints.add(ep_key)
        
        # Find missing entrypoints
        missing_entrypoints = []
        for ep in all_entrypoints:
            ep_key = f"{ep['year']}_{ep['day']}" if 'day' in ep else str(ep['year'])
            if ep_key not in processed_entrypoints:
                missing_entrypoints.append(ep)
        
        # Sort by year and day if available
        if all(('year' in ep and 'day' in ep) for ep in missing_entrypoints if missing_entrypoints):
            missing_entrypoints.sort(key=lambda x: (x['year'], x['day']))
        elif all('year' in ep for ep in missing_entrypoints if missing_entrypoints):
            missing_entrypoints.sort(key=lambda x: x['year'])
        
        logger.info(f"Found {len(missing_entrypoints)} missing entrypoints")
        
        return missing_entrypoints

    def _add_files_to_index(self, data_source, remote_files) -> int:
        """Add files to the index."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        total_indexed = 0
        batch_size = 500
        batch = []
        
        for relative_path, file_url in remote_files:
            # Calculate file hash
            file_hash = data_source.get_file_hash(file_url)
            
            # Determine destination path - simplified without GCS
            destination_blob = f"{self.data_path}/{os.path.basename(relative_path)}"
            
            # Add to batch for insertion - no timestamp means pending
            batch.append((
                file_hash,
                relative_path,
                file_url,
                destination_blob,
                None,  # No timestamp - indicates pending
                None,  # File size unknown until download
                None   # No metadata yet
            ))
            
            # If batch is full, insert into database
            if len(batch) >= batch_size:
                self._insert_file_batch(cursor, batch)
                total_indexed += len(batch)
                if total_indexed % 5000 == 0:
                    logger.info(f"Indexed {total_indexed} files so far")
                batch = []
        
        # Insert any remaining files
        if batch:
            self._insert_file_batch(cursor, batch)
            total_indexed += len(batch)
        
        conn.commit()
        
        logger.info(f"Indexed {total_indexed} files in total")
        return total_indexed

    def _insert_file_batch(self, cursor, batch):
        """
        Insert a batch of files into the index.
        
        Args:
            cursor: SQLite cursor
            batch: List of file tuples to insert
        """
        try:
            cursor.executemany(
                '''INSERT OR IGNORE INTO files 
                   (file_hash, relative_path, source_url, destination_blob, 
                    timestamp, file_size, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)''', 
                batch
            )
        except sqlite3.Error as e:
            logger.error(f"SQLite error during batch insert: {e}")
            # Fall back to one-at-a-time insertion to skip problematic records
            for record in batch:
                try:
                    cursor.execute(
                        '''INSERT OR IGNORE INTO files 
                           (file_hash, relative_path, source_url, destination_blob, 
                            timestamp, file_size, metadata)
                           VALUES (?, ?, ?, ?, ?, ?, ?)''',
                        record
                    )
                except Exception as e2:
                    logger.error(f"Error inserting record {record[0]}: {e2}")

    def get_stats(self):
        """
        Get statistics about the current index.
        
        Returns:
            dict: Statistics about indexed files
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get total file count
            cursor.execute("SELECT COUNT(*) FROM files")
            total_files = cursor.fetchone()[0]
            
            # Get file counts by status using timestamp patterns
            cursor.execute("SELECT COUNT(*) FROM files WHERE timestamp IS NULL OR timestamp = ''")
            pending_files = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM files WHERE timestamp = 'DOWNLOADING'")
            downloading_files = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM files WHERE timestamp LIKE 'FAILED:%'")
            failed_files = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM files WHERE timestamp LIKE '20%'")
            completed_files = cursor.fetchone()[0]
            
            # Get file size statistics
            cursor.execute("SELECT SUM(file_size) FROM files WHERE file_size IS NOT NULL")
            total_size = cursor.fetchone()[0] or 0
            
            # Return comprehensive statistics
            stats = {
                'total_files': total_files,
                'pending_files': pending_files,
                'downloading_files': downloading_files,
                'failed_files': failed_files,
                'completed_files': completed_files,
                'total_size': total_size,
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                'total_files': 0,
                'pending_files': 0,
                'downloading_files': 0,
                'failed_files': 0,
                'completed_files': 0,
                'total_size': 0,
            }
    
    def compare_local_and_remote_index(self, hpc_target: str, key_file: str = None) -> Dict[str, Any]:
        """
        Compare local and remote index files to determine sync strategy.
        
        Args:
            hpc_target: SSH target in format user@host:/path
            key_file: Path to SSH private key (optional)
            
        Returns:
            dict: Comparison results with sync recommendations
        """
        logger.info("Comparing local and remote index files")
        
        try:
            client = HPCClient(hpc_target, key_file=key_file)
            
            # Build remote index path
            db_filename = f"index_{self.data_path.replace('/', '_')}.sqlite"
            remote_db_path = f"hpc_data_index/{db_filename}"
            
            # Check if local index exists
            local_exists = os.path.exists(self.local_db_path)
            
            # Check if remote index exists
            remote_exists = client.check_file_exists(remote_db_path)
            
            # Determine recommended action based on existence
            if local_exists and remote_exists:
                # Both exist - could compare timestamps, but for now default to push
                recommended_action = "push"
            elif local_exists and not remote_exists:
                # Only local exists - push to remote
                recommended_action = "push"
            elif not local_exists and remote_exists:
                # Only remote exists - pull from remote
                recommended_action = "pull"
            else:
                # Neither exists - default to push (will create new)
                recommended_action = "push"
            
            return {
                "local_exists": local_exists,
                "remote_exists": remote_exists,
                "recommended_action": recommended_action,
                "local_path": self.local_db_path,
                "remote_path": remote_db_path
            }
            
        except Exception as e:
            logger.error(f"Error comparing index files: {e}")
            return {
                "local_exists": os.path.exists(self.local_db_path),
                "remote_exists": False,
                "recommended_action": "push",
                "error": str(e)
            }

    def ensure_synced_index(
        self, 
        hpc_target: str, 
        sync_direction: str = "auto", 
        force: bool = False,
        key_file: str = None
    ) -> bool:
        """
        Ensure the index is synchronized with HPC, automatically determining direction.
        
        Args:
            hpc_target: SSH target in format user@host:/path
            sync_direction: 'auto', 'push', 'pull', or 'none'
            force: Whether to force sync regardless of timestamps
            key_file: Path to SSH private key (optional)
            
        Returns:
            bool: Whether sync was successful or not needed
        """
        if sync_direction == "none":
            logger.info("Index sync disabled")
            return True
            
        logger.info(f"Ensuring index is synced with HPC (direction: {sync_direction})")
        
        try:
            if sync_direction == "auto":
                # Compare files to determine best sync direction
                comparison = self.compare_local_and_remote_index(hpc_target, key_file)
                sync_direction = comparison.get("recommended_action", "push")
                logger.info(f"Auto-detected sync direction: {sync_direction}")
            
            # Perform the sync
            success = self.sync_index_with_hpc(
                hpc_target=hpc_target,
                direction=sync_direction,
                sync_entrypoints=True,
                force=force,
                key_file=key_file
            )
            
            if success:
                logger.info(f"Index sync completed successfully ({sync_direction})")
            else:
                logger.warning(f"Index sync failed ({sync_direction})")
                
            return success
            
        except Exception as e:
            logger.error(f"Error ensuring index sync: {e}")
            return False

    def refresh_index(self, data_source):
        """
        Legacy method for backward compatibility.
        Calls build_index_from_source with default parameters.
        
        Args:
            data_source: Data source to refresh from
        """
        logger.warning("refresh_index is deprecated, use build_index_from_source instead")
        return self.build_index_from_source(
            data_source=data_source,
            rebuild=False,
            only_missing_entrypoints=True
        )

    def _check_save_needed(self, force: bool = False):
        """
        Check if index needs to be saved based on time interval.
        
        Args:
            force: Whether to force save regardless of interval
        """
        current_time = time.time()
        if force or (current_time - self._last_save_time) > self.save_interval_seconds:
            self.save()

    def _update_stats(self, files_added: int):
        """
        Update metadata statistics after indexing.
        
        Args:
            files_added: Number of files added in this operation
        """
        self.metadata["last_modified"] = datetime.now().isoformat()
        self.metadata["files_added_last_run"] = files_added
        
        # Get current file counts
        stats = self.get_stats()
        self.metadata["total_files"] = stats.get("total_files", 0)
    
    def sync_index_with_hpc(
        self, 
        hpc_target: str, 
        direction: str = "push", 
        sync_entrypoints: bool = True, 
        force: bool = False,
        key_file: str = None
    ) -> bool:
        """
        Synchronize the index database with the HPC system using rsync.
        
        Args:
            hpc_target: SSH target in format user@host:/path
            direction: 'push' to send local index to HPC, 'pull' to get index from HPC
            sync_entrypoints: Whether to also sync the entrypoints file
            force: Whether to force the sync even if timestamps indicate no changes
            key_file: Path to SSH private key (optional)
        
        Returns:
            bool: Whether the sync was successful
        """
        logger.info(f"{direction.capitalize()}ing index to/from HPC")
        
        try:
            # Save index before pushing
            if direction.lower() == "push":
                self.save()
                
            # Close connections to ensure data is flushed
            self._close_all_connections()
            time.sleep(0.5)  # Allow time for SQLite to release locks
            
            # Use our client with optional key file
            client = HPCClient(hpc_target, key_file=key_file or self.key_file)
            synchronizer = HPCIndexSynchronizer(
                client=client,
                local_index_dir=self.local_index_dir,
                remote_index_dir="hpc_data_index"
            )
            
            success = synchronizer.sync_index(
                data_path=self.data_path,
                direction=direction,
                sync_entrypoints=sync_entrypoints,
                force=force
            )
            
            # Reconnect after pull
            if direction.lower() == "pull" and success:
                # Clear connection cache to force reconnection
                if hasattr(self, '_thread_local'):
                    self._thread_local = threading.local()
            
            return success
                
        except Exception as e:
            logger.error(f"Error syncing index with HPC: {e}")
            import traceback
            logger.debug(f"Full error: {traceback.format_exc()}")
            return False
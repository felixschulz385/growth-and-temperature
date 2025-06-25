"""
SQLite-based index for managing file downloads with HPC-specific fields.

This module provides a specialized index for data downloads that works with HPC
systems, including features for synchronizing the index between local workstations 
and HPC clusters.
"""
import os
import time
import json
import logging
import sqlite3
import threading
from typing import Dict, Any, Optional, List, Union, Set
from datetime import datetime
import subprocess
import tempfile
from pathlib import Path
import shutil

# Import base class
from gnt.data.common.index.download_index import DataDownloadIndex
from gnt.data.common.hpc.client import HPCClient, HPCIndexSynchronizer

logger = logging.getLogger(__name__)

# Define file statuses for consistency
class FileStatus:
    PENDING = "pending"        # Initial state - file needs to be downloaded
    DOWNLOADING = "downloading"  # File is currently being downloaded
    DOWNLOADED = "downloaded"   # File has been downloaded but not yet batched
    BATCHED = "batched"        # File has been added to a batch
    TRANSFERRING = "transferring"  # Batch containing file is being transferred
    TRANSFERRED = "transferred"  # Batch has been transferred but not extracted
    EXTRACTING = "extracting"  # File is being extracted on HPC
    EXTRACTED = "extracted"    # File has been extracted but not yet verified
    SUCCESS = "success"        # File has been successfully processed and is available on HPC
    FAILED = "failed"          # File download or processing failed
    
    # Helper for status groups
    @staticmethod
    def is_pending(status):
        """Return True if the status is considered pending processing"""
        return status in (FileStatus.PENDING, FileStatus.DOWNLOADING)
    
    @staticmethod
    def is_in_progress(status):
        """Return True if the file is somewhere in the processing pipeline"""
        return status in (FileStatus.DOWNLOADING, FileStatus.DOWNLOADED, 
                          FileStatus.BATCHED, FileStatus.TRANSFERRING,
                          FileStatus.TRANSFERRED, FileStatus.EXTRACTING,
                          FileStatus.EXTRACTED)
    
    @staticmethod
    def is_terminal(status):
        """Return True if this is a terminal status (success or failed)"""
        return status in (FileStatus.SUCCESS, FileStatus.FAILED)

# Define batch statuses for consistency
class BatchStatus:
    PENDING = "pending"        # Batch is being created
    READY = "ready"            # Batch is complete and ready for transfer
    QUEUED = "queued"          # Batch is queued for transfer
    TRANSFERRING = "transferring"  # Batch is being transferred
    TRANSFERRED = "transferred"  # Batch has been transferred but not extracted
    EXTRACTING = "extracting"  # Batch is being extracted on HPC
    SUCCESS = "success"        # Batch has been successfully processed
    FAILED = "failed"          # Batch processing failed
    FAILED_EXTRACTION = "failed_extraction"  # Batch extraction failed

class HPCDataDownloadIndex(DataDownloadIndex):
    """
    SQLite-based index for managing file downloads with HPC-specific fields.
    
    This extends the base DataDownloadIndex with features specific to HPC environments:
    - Local index storage instead of cloud storage
    - Batch management for efficient transfers
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
        key_file: str = None  # Added key_file parameter
    ):
        """
        Initialize the HPC download index.
        
        Args:
            bucket_name: GCS bucket name (kept for compatibility)
            data_source: Data source for downloads
            local_index_dir: Local directory for index storage (not in GCS)
            client: GCS client (optional but kept for interface compatibility)
            temp_dir: Temporary directory for downloads
            save_interval_seconds: How often to save the index
            key_file: Path to SSH private key file (optional)
        """
        # Set up HPC-specific attributes
        self.use_local_only = True
        self.local_index_dir = local_index_dir or os.path.expanduser("~/hpc_data_index")
        os.makedirs(self.local_index_dir, exist_ok=True)
        
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
        
        # Store key_file for use in synchronization
        self.key_file = key_file
    
    def _setup_database(self):
        """Set up SQLite database with HPC-specific tables."""
        # Set up local path
        self.local_db_path = os.path.join(
            self.local_index_dir, 
            f"download_{self.data_path.replace('/', '_')}.sqlite"
        )
        
        # Connect with higher timeout and other pragmas for robustness
        conn = self._get_connection(timeout=60)
        cursor = conn.cursor()
        
        # Set pragmas for better reliability
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=30000")  # 30 seconds
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.execute("PRAGMA foreign_keys=ON")
        
        # Create standard tables (same as parent)
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
        
        # Add HPC-specific tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS batches (
            batch_id TEXT PRIMARY KEY,
            tar_path TEXT,
            file_count INTEGER,
            total_size INTEGER,
            status TEXT,
            created_timestamp TEXT,
            transfer_timestamp TEXT,
            error TEXT
        )
        ''')
        
        # Add column for batch_id if it doesn't exist
        try:
            cursor.execute("SELECT batch_id FROM files LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE files ADD COLUMN batch_id TEXT")
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON files(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_batch_id ON files(batch_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_batch_status ON batches(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relative_path ON files(relative_path)')
        conn.commit()
    
    def _get_connection(self, timeout=30):
        """
        Get a SQLite database connection with improved error handling.
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            sqlite3.Connection: A connection to the database
        """
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
                # Connect with a timeout to avoid hanging
                self._thread_local.connection = sqlite3.connect(
                    self.local_db_path, 
                    timeout=timeout,
                    isolation_level=None  # Autocommit mode
                )
                
                # Enable extended error codes for better diagnostics
                self._thread_local.connection.execute("PRAGMA foreign_keys=ON")
                self._thread_local.connection.execute("PRAGMA busy_timeout=30000")  # 30 seconds
                
                # Use write-ahead logging for better concurrency
                self._thread_local.connection.execute("PRAGMA journal_mode=WAL")
                
                # Configure for better reliability with multiple threads
                self._thread_local.connection.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.Error as e:
                logger.error(f"Error connecting to database: {e}")
                # If we can't connect, try to recover or create a new database
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
        self._connections = {}
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
                    f"download_{self.data_path.replace('/', '_')}_meta.json"
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
        check_gcs=False,  # Ignored in HPC version
        only_missing_entrypoints=True, 
        force_refresh_gcs=False  # Ignored in HPC version
    ):
        """
        Build index from data source with configurable behavior.
        
        Args:
            data_source: Data source to index
            rebuild: Whether to rebuild the index from scratch
            check_gcs: Whether to check HPC for existing files (not used)
            only_missing_entrypoints: Only process entrypoints not already in index
            force_refresh_gcs: Not used in HPC version
        
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
            # No GCS check in HPC version
            existing_files = set()
                
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

    def _add_files_to_index(self, data_source, remote_files, existing_files=None) -> int:
        """
        Add files to the index.
        
        Args:
            data_source: Data source for the files
            remote_files: List of (relative_path, file_url) tuples
            existing_files: Set of existing file paths (not used in HPC version)
            
        Returns:
            int: Number of files added to index
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        total_indexed = 0
        batch_size = 500
        batch = []
        
        for relative_path, file_url in remote_files:
            # Calculate file hash
            file_hash = data_source.get_file_hash(file_url)
            
            # Determine destination path in local storage
            # Note: Files will be placed under raw/ subfolder by the BatchManager
            # but we keep the original destination path in the index for reference
            destination_blob = data_source.gcs_upload_path(self.data_path, relative_path)
            
            # Add to batch for insertion
            batch.append((
                file_hash,
                relative_path,
                file_url,
                destination_blob,
                FileStatus.PENDING,  # Use constants for consistency
                datetime.now().isoformat(),
                None,  # No error
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
                    status, timestamp, error, file_size, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
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
                            status, timestamp, error, file_size, metadata)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        record
                    )
                except Exception as e2:
                    logger.error(f"Error inserting record {record[0]}: {e2}")

    def _check_save_needed(self, force=False):
        """
        Check if the index needs to be saved and save if needed.
        
        Args:
            force: If True, force a save regardless of time elapsed
        """
        now = time.time()
        if force or (now - self._last_save_time > self.save_interval_seconds):
            # Use a dedicated thread for saving to avoid blocking
            save_thread = threading.Thread(target=self._thread_safe_save)
            save_thread.daemon = True
            save_thread.start() 
            
    def _thread_safe_save(self):
        """Thread-safe method to save the index. Creates its own connection."""
        try:
            # Don't close other threads' connections
            
            # Create a direct connection for checkpointing
            temp_conn = sqlite3.connect(self.local_db_path)
            temp_cursor = temp_conn.cursor()

            try:
                # Force a checkpoint to sync WAL to main DB
                temp_cursor.execute("PRAGMA wal_checkpoint(FULL)")
                temp_conn.commit()
                
                # Save metadata file locally
                meta_path = os.path.join(
                    self.local_index_dir, 
                    f"download_{self.data_path.replace('/', '_')}_meta.json"
                )
                with open(meta_path, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
                    
                # Update last save time
                self._last_save_time = time.time()
                
                logger.info(f"Index for {self.data_source_name} saved successfully")
                
            finally:
                temp_cursor.close()
                temp_conn.close()
        except Exception as e:
            logger.error(f"Error in thread-safe save: {e}")
            import traceback
            logger.debug(f"Full error: {traceback.format_exc()}")

    def _update_stats(self, total_indexed):
        """
        Update index metadata with statistics.
        
        Args:
            total_indexed: Total number of files indexed in this run
        """
        stats = self.get_stats()
        self.metadata.update({
            "last_update": datetime.now().isoformat(),
            "total_files": stats['total_files'],
            "total_indexed_last_run": total_indexed,
            "pending_files": stats['files_pending'],
            "transferred_files": stats.get('files_transferred', 0),
            "successful_files": stats['files_success'],
            "failed_files": stats['files_failed']
        })
        self.save()

    def refresh_index(self, data_source):
        """
        Refresh the index from the data source.
        
        Args:
            data_source: Data source to refresh from
            
        Returns:
            int: Number of files indexed
        """
        logger.info(f"Refreshing index for {data_source.DATA_SOURCE_NAME}")
        # For simple refresh, we just call build_index_from_source with default options
        # This will only index missing entrypoints, not rebuilding everything
        return self.build_index_from_source(
            data_source,
            rebuild=False,
            check_gcs=False,  # No GCS check for HPC version
            only_missing_entrypoints=True
        )
    
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
            
            # Use our new client with optional key file
            client = HPCClient(hpc_target, key_file=key_file)
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
                self._connections = {}
            
            return success
                
        except Exception as e:
            logger.error(f"Error syncing index with HPC: {e}")
            import traceback
            logger.debug(f"Full error: {traceback.format_exc()}")
            return False

    def compare_local_and_remote_index(self, hpc_target: str, key_file: str = None) -> Dict[str, Any]:
        """
        Compare local and remote index files to determine which is newer/better.
        
        Args:
            hpc_target: SSH target in format user@host:/path
            key_file: Path to SSH private key (optional)
        
        Returns:
            Dict with comparison results
        """
        try:
            client = HPCClient(hpc_target, key_file=key_file)
            synchronizer = HPCIndexSynchronizer(
                client=client,
                local_index_dir=self.local_index_dir,
                remote_index_dir="hpc_data_index"
            )
            
            return synchronizer.compare_indices(self.data_path)
                
        except Exception as e:
            logger.error(f"Error comparing indices: {e}")
            return {
                'local_exists': False,
                'remote_exists': False,
                'local_file_count': 0,
                'remote_file_count': 0,
                'local_modified': None,
                'remote_modified': None,
                'recommendation': 'push'  # Default
            }

    def ensure_synced_index(
        self, 
        hpc_target: str, 
        sync_direction: str = 'auto', 
        force: bool = False,
        key_file: str = None
    ) -> bool:
        """
        Ensure the index is synchronized with the HPC system.
        
        Args:
            hpc_target: SSH target in format user@host:/path
            sync_direction: 'auto', 'push', 'pull', or 'none'
            force: Whether to force sync even if timestamps indicate no changes
            key_file: Path to SSH private key file (optional)
        
        Returns:
            bool: Whether synchronization was successful
        """
        from gnt.data.common.hpc.client import HPCClient, HPCIndexSynchronizer
        
        # Use instance key_file if passed key_file is None
        key_file = key_file or getattr(self, 'key_file', None)
        
        # Create HPC client with key file if provided
        hpc_client = HPCClient(hpc_target, key_file=key_file)
        
        # Log the key being used
        logger = logging.getLogger(__name__)
        if key_file:
            expanded_key = os.path.expanduser(key_file) if '~' in key_file else key_file
            logger.debug(f"Using SSH key file for index sync: {key_file} (expanded to {expanded_key})")
        
        # Create synchronizer
        synchronizer = HPCIndexSynchronizer(
            client=hpc_client,
            local_index_dir=self.local_index_dir
        )
        
        # Perform sync and return result
        return synchronizer.ensure_synced_index(
            data_path=self.data_path,
            sync_direction=sync_direction,
            force=force
        )

    def get_stats(self) -> Dict[str, int]:
        """
        Get download statistics from the index.
        
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get total file count
            cursor.execute("SELECT COUNT(*) FROM files")
            stats['total_files'] = cursor.fetchone()[0]
            
            # Get individual status counts
            cursor.execute(f"SELECT COUNT(*) FROM files WHERE status = '{FileStatus.PENDING}'")
            stats['files_pending'] = cursor.fetchone()[0]
            
            cursor.execute(f"SELECT COUNT(*) FROM files WHERE status = '{FileStatus.DOWNLOADING}'")
            stats['files_downloading'] = cursor.fetchone()[0]
            
            cursor.execute(f"SELECT COUNT(*) FROM files WHERE status = '{FileStatus.BATCHED}'")
            stats['files_batched'] = cursor.fetchone()[0]
            
            cursor.execute(f"SELECT COUNT(*) FROM files WHERE status = '{FileStatus.TRANSFERRING}'")
            stats['files_transferring'] = cursor.fetchone()[0]
            
            cursor.execute(f"SELECT COUNT(*) FROM files WHERE status = '{FileStatus.TRANSFERRED}'")
            stats['files_transferred'] = cursor.fetchone()[0]
            
            cursor.execute(f"SELECT COUNT(*) FROM files WHERE status = '{FileStatus.EXTRACTED}'")
            stats['files_extracted'] = cursor.fetchone()[0]
            
            cursor.execute(f"SELECT COUNT(*) FROM files WHERE status = '{FileStatus.SUCCESS}'")
            stats['files_success'] = cursor.fetchone()[0]
            
            cursor.execute(f"SELECT COUNT(*) FROM files WHERE status = '{FileStatus.FAILED}'")
            stats['files_failed'] = cursor.fetchone()[0]
            
            # Add a composite in_progress count
            in_progress_statuses = (
                f"'{FileStatus.DOWNLOADING}', '{FileStatus.DOWNLOADED}', "
                f"'{FileStatus.BATCHED}', '{FileStatus.TRANSFERRING}', "
                f"'{FileStatus.TRANSFERRED}', '{FileStatus.EXTRACTING}', "
                f"'{FileStatus.EXTRACTED}'"
            )
            cursor.execute(f"SELECT COUNT(*) FROM files WHERE status IN ({in_progress_statuses})")
            stats['files_in_progress'] = cursor.fetchone()[0]
            
            # Count batches
            cursor.execute("SELECT COUNT(*) FROM batches")
            stats['total_batches'] = cursor.fetchone()[0]
            
            cursor.execute(f"SELECT COUNT(*) FROM batches WHERE status = '{BatchStatus.SUCCESS}'")
            stats['batches_success'] = cursor.fetchone()[0]
            
            cursor.execute(f"SELECT COUNT(*) FROM batches WHERE status = '{BatchStatus.FAILED}'")
            stats['batches_failed'] = cursor.fetchone()[0]
            
            # In-progress batches
            in_progress_batch_statuses = (
                f"'{BatchStatus.PENDING}', '{BatchStatus.READY}', "
                f"'{BatchStatus.QUEUED}', '{BatchStatus.TRANSFERRING}', "
                f"'{BatchStatus.TRANSFERRED}', '{BatchStatus.EXTRACTING}'"
            )
            cursor.execute(f"SELECT COUNT(*) FROM batches WHERE status IN ({in_progress_batch_statuses})")
            stats['batches_in_progress'] = cursor.fetchone()[0]
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            stats['error'] = str(e)
            
        return stats
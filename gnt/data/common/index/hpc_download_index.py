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

# Import base class
from gnt.data.common.index.download_index import DataDownloadIndex
from gnt.data.common.hpc.client import HPCClient, HPCIndexSynchronizer

logger = logging.getLogger(__name__)

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
        save_interval_seconds=300
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
    
    def _setup_database(self):
        """Set up SQLite database with HPC-specific tables."""
        # Set up local path
        self.local_db_path = os.path.join(
            self.local_index_dir, 
            f"download_{self.data_path.replace('/', '_')}.sqlite"
        )
        
        # Connect and create tables
        conn = self._get_connection()
        cursor = conn.cursor()
        
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
    
    def save(self):
        """Save index to local storage instead of cloud storage."""
        logger.info(f"Saving index for {self.data_source_name} to local storage")
        try:
            # Close the current thread's connection to ensure data is flushed
            self._close_all_connections()
            
            # Allow time for SQLite to release locks
            time.sleep(0.5)
            
            # Create a fresh connection for the checkpoint (in this thread)
            # Don't use the thread-local storage for this one-time operation
            temp_conn = sqlite3.connect(self.local_db_path)
            temp_cursor = temp_conn.cursor()

            try:
                # Force a checkpoint to sync WAL to main DB
                temp_cursor.execute("PRAGMA wal_checkpoint(FULL)")
                
                # Optionally switch journal mode if needed
                # temp_cursor.execute("PRAGMA journal_mode = DELETE")
                
                # Commit any pending transactions
                temp_conn.commit()
            finally:
                # Close the temporary connection
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
                "pending",  # All files start as pending
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
            key_file: Path to SSH private key (optional)
        
        Returns:
            bool: Whether synchronization was successful or wasn't needed
        """
        if sync_direction.lower() == 'none':
            logger.info("Index synchronization disabled")
            return True
            
        try:
            client = HPCClient(hpc_target, key_file=key_file)
            synchronizer = HPCIndexSynchronizer(
                client=client,
                local_index_dir=self.local_index_dir,
                remote_index_dir="hpc_data_index"
            )
            
            return synchronizer.ensure_synced_index(
                data_path=self.data_path,
                sync_direction=sync_direction,
                force=force
            )
                
        except Exception as e:
            logger.error(f"Error ensuring synced index: {e}")
            return False

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
            
            # Get status counts - simplified to just pending, success, failed
            cursor.execute(
                "SELECT COUNT(*) FROM files WHERE status = 'pending' OR status IN ('batched', 'transferring', 'extracted', 'transferred')"
            )
            stats['files_pending'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM files WHERE status = 'success'")
            stats['files_success'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM files WHERE status = 'failed'")
            stats['files_failed'] = cursor.fetchone()[0]
            
            # Count batches
            cursor.execute("SELECT COUNT(*) FROM batches")
            stats['total_batches'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM batches WHERE status = 'success'")
            stats['batches_success'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM batches WHERE status = 'failed'")
            stats['batches_failed'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM batches WHERE status NOT IN ('success', 'failed')")
            stats['batches_pending'] = cursor.fetchone()[0]
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            stats['error'] = str(e)
            
        return stats
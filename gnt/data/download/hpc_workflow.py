"""
HPC-specific workflow for managing downloads and transfers to HPC systems.
"""

import os
import time
import json
import yaml
import queue
import random
import logging
import threading
import tempfile
import subprocess
import shutil
import glob
import re
import traceback  # Add missing import for traceback
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from datetime import datetime
from tqdm import tqdm  # Add missing import for tqdm

from gnt.data.common.index.hpc_download_index import HPCDataDownloadIndex
from gnt.data.download.workflow import WorkflowContext, DownloadWorker, TaskHandlers
from gnt.data.download.workflow import setup_progress_reporting, queue_files_for_download, load_config_with_env_vars
from gnt.data.download.sources.factory import create_data_source
from gnt.data.common.hpc.client import HPCClient

# Configure logging
logger = logging.getLogger(__name__)

class HPCWorkflowContext(WorkflowContext):
    """Context for HPC workflow execution."""
    
    def __init__(self, hpc_target: str, local_index_dir: str = None, key_file: str = None):
        """
        Initialize the HPC workflow context.
        
        Args:
            hpc_target: SSH target for HPC (user@server:/path)
            local_index_dir: Directory for local index storage
            key_file: Path to SSH private key file (optional)
        """
        super().__init__(bucket_name=None)  # Bucket name not used
        self.hpc_target = hpc_target
        self.local_index_dir = local_index_dir or os.path.expanduser("~/hpc_data_index")
        self.key_file = key_file
        os.makedirs(self.local_index_dir, exist_ok=True)
        
        # Extract HPC host and path
        if ":" in hpc_target:
            parts = hpc_target.split(":", 1)
            self.hpc_host = parts[0]
            self.hpc_path = parts[1]
            
            # Remove trailing slash for consistency
            self.hpc_path = self.hpc_path.rstrip('/')
        else:
            self.hpc_host = hpc_target
            self.hpc_path = ""
            
        # Create staging directory
        self.staging_dir = os.path.join(self.local_index_dir, "staging")
        os.makedirs(self.staging_dir, exist_ok=True)
        
        logger.debug(f"Initialized HPC context with host: {self.hpc_host}, path: {self.hpc_path}")


class BatchManager:
    """Manager for creating and tracking batches of files for rsync transfer."""
    
    def __init__(self, context: HPCWorkflowContext, index: HPCDataDownloadIndex, 
                 batch_size: int = 500, max_batch_size_mb: int = 5000):
        """
        Initialize the batch manager.
        
        Args:
            context: HPC workflow context
            index: HPC download index
            batch_size: Maximum number of files per batch
            max_batch_size_mb: Maximum batch size in MB
        """
        self.context = context
        self.index = index
        self.batch_size = batch_size
        self.max_batch_size_bytes = max_batch_size_mb * 1024 * 1024
        
        # Current batch tracking
        self.current_batch = None
        self.current_batch_files = []
        self.current_batch_size = 0
        self.batch_counter = 0
        
        # Batch directory
        self.batch_dir = os.path.join(context.staging_dir, "batches")
        os.makedirs(self.batch_dir, exist_ok=True)
        
        # Add state tracking file
        self.state_file = os.path.join(context.local_index_dir, f"batch_state_{index.data_source_name}.json")
        self._load_state()
    
    def _load_state(self):
        """Load batch state from disk if it exists."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    
                logger.info(f"Loaded batch state from {self.state_file}")
                
                # Check for incomplete batches to resume
                if state.get('current_batch') and os.path.exists(os.path.join(self.batch_dir, state['current_batch']['id'])):
                    self.current_batch = state['current_batch']
                    self.current_batch_files = state.get('current_batch_files', [])
                    self.current_batch_size = state.get('current_batch_size', 0)
                    logger.info(f"Resuming incomplete batch {self.current_batch['id']} with {len(self.current_batch_files)} files")
            except Exception as e:
                logger.warning(f"Failed to load batch state: {e}, starting with clean state")
    
    def _save_state(self):
        """Save current batch state to disk."""
        state = {
            'current_batch': self.current_batch,
            'current_batch_files': self.current_batch_files,
            'current_batch_size': self.current_batch_size,
            'batch_counter': self.batch_counter,
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save batch state: {e}")
    
    def _create_new_batch(self):
        """Create a new batch."""
        self.batch_counter += 1
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self.batch_counter}"
        self.current_batch = {
            'id': batch_id,
            'created': datetime.now().isoformat(),
            'status': 'pending'
        }
        self.current_batch_files = []
        self.current_batch_size = 0
        
        # Create batch directory
        os.makedirs(os.path.join(self.batch_dir, batch_id), exist_ok=True)
        
        # Record in database
        conn = self.index._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO batches 
               (batch_id, status, created_timestamp, file_count, total_size)
               VALUES (?, ?, ?, ?, ?)''',
            (batch_id, 'pending', datetime.now().isoformat(), 0, 0)
        )
        conn.commit()
        
        logger.info(f"Created new batch: {batch_id}")
        return batch_id
    
    def add_file(self, file_info: Dict, local_path: str) -> str:
        """
        Add a downloaded file to a batch.
        
        Args:
            file_info: Dictionary with file information
            local_path: Path to the downloaded file
            
        Returns:
            batch_id: ID of the batch the file was added to
        """
        file_size = os.path.getsize(local_path)
        
        # Create a new batch if needed
        if not self.current_batch or (
            len(self.current_batch_files) >= self.batch_size or
            self.current_batch_size + file_size > self.max_batch_size_bytes
        ):
            self._create_new_batch()
        
        # Get batch ID and directory
        batch_id = self.current_batch['id']
        batch_dir = os.path.join(self.batch_dir, batch_id)
        
        # Get the destination path and make it properly relative
        dest_path = file_info.get('destination_blob') or file_info.get('destination_path')
        
        # Extract the truly relative path - avoid nested path duplication:
        # Just use the basename of the file without duplicating the data_path
        rel_path = os.path.basename(dest_path)
        
        # Create the directory structure in the batch directory - use 'raw' directory
        dest_dir = os.path.join(batch_dir, "raw")
        os.makedirs(dest_dir, exist_ok=True)
        
        # Copy the file to the batch directory
        batch_file_path = os.path.join(dest_dir, rel_path)
        shutil.copy2(local_path, batch_file_path)
        
        # Update batch tracking - store just the filename as the path
        # The raw/ prefix will be added during extraction
        self.current_batch_files.append({
            'file_hash': file_info['file_hash'],
            'path': rel_path,
            'size': file_size
        })
        self.current_batch_size += file_size
        
        # Update the file record in the index
        conn = self.index._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE files SET batch_id = ?, status = ?, file_size = ? WHERE file_hash = ?",
            (batch_id, "batched", file_size, file_info['file_hash'])
        )
        
        # Update batch record
        cursor.execute(
            "UPDATE batches SET file_count = ?, total_size = ? WHERE batch_id = ?",
            (len(self.current_batch_files), self.current_batch_size, batch_id)
        )
        conn.commit()
        
        # Save state after updating batch
        self._save_state()
        
        return batch_id
    
    def finalize_current_batch(self) -> Optional[str]:
        """
        Finalize the current batch by creating a tar ball.
        
        Returns:
            str: Path to the tar ball, or None if no batch to finalize
        """
        if not self.current_batch or not self.current_batch_files:
            return None
            
        batch_id = self.current_batch['id']
        batch_dir = os.path.join(self.batch_dir, batch_id)
        tar_path = os.path.join(self.context.staging_dir, f"{batch_id}.tar.gz")
        
        # Create tar ball
        logger.info(f"Creating tar ball for batch {batch_id} with {len(self.current_batch_files)} files")
        try:
            subprocess.run(
                ["tar", "-czf", tar_path, "-C", batch_dir, "."],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Update batch status
            conn = self.index._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE batches SET status = ?, tar_path = ? WHERE batch_id = ?",
                ("ready", tar_path, batch_id)
            )
            conn.commit()
            
            logger.info(f"Batch {batch_id} finalized, tar ball created at {tar_path}")
            
            # Clear current batch
            current_batch_copy = self.current_batch.copy()
            self.current_batch = None
            self.current_batch_files = []
            self.current_batch_size = 0
            
            # Clear state since batch is finalized
            if current_batch_copy:
                self._save_state()
                
            return tar_path
            
        except Exception as e:
            logger.error(f"Error creating tar ball for batch {batch_id}: {e}")
            
            # Update batch status
            conn = self.index._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE batches SET status = ?, error = ? WHERE batch_id = ?",
                ("failed", str(e), batch_id)
            )
            conn.commit()
            
            return None
    
    def cleanup_batch(self, batch_id: str, status: str = "success") -> None:
        """
        Clean up local files for a batch after successful processing.
        
        Args:
            batch_id: ID of the batch to clean up
            status: Status to set in DB ('success' or 'failed')
        """
        batch_dir = os.path.join(self.batch_dir, batch_id)
        tar_path = os.path.join(self.context.staging_dir, f"{batch_id}.tar.gz")
        
        # Update files and batch status in the DB
        conn = self.index._get_connection()
        cursor = conn.cursor()
        
        if status == "success":
            # If success, mark all files as successful
            cursor.execute(
                "UPDATE files SET status = ? WHERE batch_id = ?",
                ("success", batch_id)
            )
            
            # Update batch status
            cursor.execute(
                "UPDATE batches SET status = ? WHERE batch_id = ?",
                ("success", batch_id)
            )
        else:
            # If failed, mark as failed
            cursor.execute(
                "UPDATE files SET status = ? WHERE batch_id = ?",
                ("failed", batch_id)
            )
            
            # Update batch status
            cursor.execute(
                "UPDATE batches SET status = ? WHERE batch_id = ?",
                ("failed", batch_id)
            )
            
        conn.commit()
        
        # Remove batch directory
        if os.path.exists(batch_dir):
            try:
                shutil.rmtree(batch_dir)
                logger.debug(f"Removed batch directory: {batch_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove batch directory {batch_dir}: {e}")
        
        # Remove tar file
        if os.path.exists(tar_path):
            try:
                os.remove(tar_path)
                logger.debug(f"Removed tar file: {tar_path}")
            except Exception as e:
                logger.warning(f"Failed to remove tar file {tar_path}: {e}")

    def cleanup_all_processed_batches(self) -> None:
        """Clean up all successfully processed batches to free disk space."""
        conn = self.index._get_connection()
        cursor = conn.cursor()
        
        # Get all successfully extracted or failed batches
        cursor.execute("SELECT batch_id, status FROM batches WHERE status IN ('extracted', 'success', 'failed', 'failed_extraction')")
        batches = cursor.fetchall()
        
        for batch_id, status in batches:
            status_for_cleanup = "success" if status in ("extracted", "success") else "failed"
            self.cleanup_batch(batch_id, status=status_for_cleanup)
            
        logger.info(f"Cleaned up {len(batches)} processed batches")

    def cleanup_all(self) -> None:
        """Clean up all batch files regardless of status."""
        logger.info("Cleaning up all batch files")
        
        # Clean up current batch if exists
        if self.current_batch:
            batch_id = self.current_batch['id']
            batch_dir = os.path.join(self.batch_dir, batch_id)
            if os.path.exists(batch_dir):
                try:
                    shutil.rmtree(batch_dir)
                    logger.debug(f"Removed incomplete batch directory: {batch_dir}")
                except Exception as e:
                    logger.warning(f"Failed to remove batch directory {batch_dir}: {e}")
        
        # Clean up all tar files
        tar_files = glob.glob(os.path.join(self.context.staging_dir, "batch_*.tar.gz"))
        for tar_file in tar_files:
            try:
                os.remove(tar_file)
                logger.debug(f"Removed tar file: {tar_file}")
            except Exception as e:
                logger.warning(f"Failed to remove tar file {tar_file}: {e}")
                
        # Clean up all batch directories
        if os.path.exists(self.batch_dir):
            batch_dirs = os.listdir(self.batch_dir)
            for batch_dir_name in batch_dirs:
                batch_dir_path = os.path.join(self.batch_dir, batch_dir_name)
                if os.path.isdir(batch_dir_path):
                    try:
                        shutil.rmtree(batch_dir_path)
                        logger.debug(f"Removed batch directory: {batch_dir_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove batch directory {batch_dir_path}: {e}")

class RsyncTransferManager:
    """Manager for rsync transfers to HPC."""
    
    def __init__(self, context: HPCWorkflowContext, index: HPCDataDownloadIndex,
                 rsync_options: Dict = None, max_concurrent: int = 2):
        """
        Initialize the rsync transfer manager.
        
        Args:
            context: HPC workflow context
            index: HPC download index
            rsync_options: Dictionary of rsync options
            max_concurrent: Maximum concurrent rsync processes
        """
        self.context = context
        self.index = index
        self.rsync_options = rsync_options or {
            "compress": True,
            "archive": True,
            "partial": True,
            "verbose": True,
            "bwlimit": 0  # 0 means no limit
        }
        
        # Create HPC client with key file from context
        self.hpc_client = HPCClient(context.hpc_target, key_file=context.key_file)
        
        # Concurrency control
        self.max_concurrent = max_concurrent
        self.transfer_semaphore = threading.Semaphore(max_concurrent)
        
        # Queue for pending transfers
        self.transfer_queue = queue.Queue()
        
        # Transfer tracking
        self.transfers_in_progress = {}
        self.transfer_lock = threading.RLock()
    
    def queue_batch(self, batch_id: str, tar_path: str) -> bool:
        """
        Queue a batch for transfer.
        
        Args:
            batch_id: Batch ID
            tar_path: Path to the tar file
            
        Returns:
            bool: Whether the batch was queued
        """
        if not os.path.exists(tar_path):
            logger.error(f"Cannot queue batch {batch_id}: tar file not found at {tar_path}")
            return False
        
        # Update batch status
        conn = self.index._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE batches SET status = ? WHERE batch_id = ?",
            ("queued", batch_id)
        )
        conn.commit()
        
        # Add to queue
        self.transfer_queue.put((batch_id, tar_path))
        logger.info(f"Batch {batch_id} queued for transfer")
        return True
    
    def start_transfer_workers(self, num_workers: int = None):
        """
        Start worker threads for batch transfers.
        
        Args:
            num_workers: Number of worker threads
        """
        num_workers = num_workers or self.max_concurrent
        
        for i in range(num_workers):
            thread = threading.Thread(
                target=self._transfer_worker,
                name=f"transfer-worker-{i}"
            )
            thread.daemon = True
            thread.start()
            logger.info(f"Started transfer worker {i}")
    
    def _transfer_worker(self):
        """Worker function for transferring batches."""
        while True:
            try:
                # Get next batch from queue
                batch_id, tar_path = self.transfer_queue.get()
                
                # Acquire semaphore
                self.transfer_semaphore.acquire()
                
                try:
                    # Transfer the batch
                    self._transfer_batch(batch_id, tar_path)
                finally:
                    # Release semaphore
                    self.transfer_semaphore.release()
                    
                    # Mark task as done
                    self.transfer_queue.task_done()
            
            except Exception as e:
                logger.error(f"Error in transfer worker: {e}")
                # Continue processing
    
    def _transfer_batch(self, batch_id: str, tar_path: str) -> bool:
        """
        Transfer a batch to the HPC system using rsync.
        
        Args:
            batch_id: Batch ID
            tar_path: Path to the tar file
            
        Returns:
            bool: Whether the transfer was successful
        """
        conn = self.index._get_connection()
        cursor = conn.cursor()
        
        # Update batch status to transferring (just for internal tracking)
        cursor.execute(
            "UPDATE batches SET status = ? WHERE batch_id = ?",
            ("transferring", batch_id)
        )
        conn.commit()
        
        # Track in-progress transfer
        with self.transfer_lock:
            self.transfers_in_progress[batch_id] = {
                "start_time": time.time(),
                "tar_path": tar_path
            }
    
        # Construct tar directory relative to HPC base path
        # Use the context.hpc_path as the base
        tar_dir = f"{self.context.hpc_path}/{self.index.data_path}/tar"
    
        # Ensure target directory exists - need to create the full path
        self.hpc_client.ensure_directory(tar_dir)
    
        # Log the path for debugging
        logger.debug(f"Transferring to full remote path: {tar_dir}")
    
        try:
            # Execute the transfer using HPCClient
            logger.info(f"Transferring batch {batch_id} to HPC using rsync")
            success, summary = self.hpc_client.rsync_transfer(
                source_path=tar_path,
                target_path=tar_dir,
                source_is_local=True,
                options=self.rsync_options,
                show_progress=True
            )
            
            if success:
                # Status stays as "pending" until extraction completes
                cursor.execute(
                    "UPDATE batches SET transfer_timestamp = ? WHERE batch_id = ?",
                    (datetime.now().isoformat(), batch_id)
                )
                conn.commit()
                
                # Remove from in-progress tracking
                with self.transfer_lock:
                    if batch_id in self.transfers_in_progress:
                        transfer_time = time.time() - self.transfers_in_progress[batch_id]["start_time"]
                        del self.transfers_in_progress[batch_id]
                
                # Log result
                logger.info(f"Successfully transferred batch {batch_id} to HPC - {summary} in {transfer_time:.1f}s")
                return True
            else:
                # Transfer failed, mark as failed
                cursor.execute(
                    "UPDATE batches SET status = ?, error = ? WHERE batch_id = ?",
                    ("failed", f"Rsync transfer failed: {summary}", batch_id)
                )
                conn.commit()
                
                # Mark files as failed
                cursor.execute(
                    "UPDATE files SET status = ? WHERE batch_id = ?",
                    ("failed", batch_id)
                )
                conn.commit()
                
                # Remove from in-progress tracking
                with self.transfer_lock:
                    if batch_id in self.transfers_in_progress:
                        del self.transfers_in_progress[batch_id]
                
                logger.error(f"Rsync failed for batch {batch_id}: {summary}")
                return False
    
        except Exception as e:
            logger.error(f"Error transferring batch {batch_id}: {e}")
            cursor.execute(
                "UPDATE batches SET status = ?, error = ? WHERE batch_id = ?",
                ("failed", str(e), batch_id)
            )
            conn.commit()
            
            # Mark files as failed
            cursor.execute(
                "UPDATE files SET status = ? WHERE batch_id = ?",
                ("failed", batch_id)
            )
            conn.commit()
            
            # Remove from in-progress tracking
            with self.transfer_lock:
                if batch_id in self.transfers_in_progress:
                    del self.transfers_in_progress[batch_id]
            
            return False
    
    def extract_batch(self, batch_id: str) -> bool:
        """
        Request extraction of a batch on the HPC using SSH.
        
        Args:
            batch_id: ID of the batch to extract
            
        Returns:
            bool: Whether the extraction command was sent successfully
        """
        logger.debug(f"Starting extraction for batch {batch_id}")
        conn = self.index._get_connection()
        cursor = conn.cursor()
        
        # Get batch information
        cursor.execute("SELECT status FROM batches WHERE batch_id = ?", (batch_id,))
        result = cursor.fetchone()
        
        if not result:
            logger.error(f"Batch {batch_id} not found")
            return False
            
        status = result[0]
        logger.debug(f"Batch {batch_id} current status: {status}")
        
        # Execute extraction on HPC using HPCClient
        tar_filename = f"{batch_id}.tar.gz"
        
        # Build paths with full HPC path prefix
        base_dir = f"{self.context.hpc_path}/{self.index.data_path}"
        raw_dir = f"{base_dir}/raw"  # Add /raw subfolder for extracted files
        tar_dir = f"{base_dir}/tar"
        
        # Build full tar path
        tar_path = f"{tar_dir}/{tar_filename}"
        
        logger.info(f"Extracting batch {batch_id} on HPC from {tar_path} to {raw_dir}")
        
        try:
            # Ensure extraction directory exists (the raw directory)
            self.hpc_client.ensure_directory(raw_dir)
            logger.debug(f"Ensured raw directory exists: {raw_dir}")
            
            # Check if tar file exists on remote
            if not self.hpc_client.check_file_exists(tar_path):
                logger.error(f"Tar file {tar_path} not found on remote system")
                
                # Update batch status to failed
                cursor.execute(
                    "UPDATE batches SET status = ?, error = ? WHERE batch_id = ?",
                    ("failed", f"Tar file not found on remote: {tar_path}", batch_id)
                )
                conn.commit()
                return False
            
            # Use HPCClient to extract the tar file to the raw directory
            logger.debug(f"Extracting tar file: {tar_path}")
            success = self.hpc_client.extract_tar(tar_path, raw_dir)
            
            if success:
                # Update batch status to success
                cursor.execute(
                    "UPDATE batches SET status = ? WHERE batch_id = ?",
                    ("success", batch_id)
                )
                conn.commit()
                
                # Update file statuses to success
                cursor.execute(
                    "UPDATE files SET status = ? WHERE batch_id = ?",
                    ("success", batch_id)
                )
                conn.commit()
                
                # Log result
                logger.info(f"Successfully extracted batch {batch_id} on HPC to {raw_dir}")
                return True
            else:
                # Extraction failed
                cursor.execute(
                    "UPDATE batches SET status = ?, error = ? WHERE batch_id = ?",
                    ("failed", "Tar extraction failed on HPC", batch_id)
                )
                conn.commit()
                
                # Mark files as failed
                cursor.execute(
                    "UPDATE files SET status = ? WHERE batch_id = ?",
                    ("failed", batch_id)
                )
                conn.commit()
                
                logger.error(f"Failed to extract batch {batch_id} on HPC")
                return False
            
        except Exception as e:
            logger.error(f"Error requesting extraction for batch {batch_id}: {e}")
            cursor.execute(
                "UPDATE batches SET status = ?, error = ? WHERE batch_id = ?",
                ("failed", str(e), batch_id)
            )
            conn.commit()
            
            # Mark files as failed
            cursor.execute(
                "UPDATE files SET status = ? WHERE batch_id = ?",
                ("failed", batch_id)
            )
            conn.commit()
            
            return False

class HPCDownloadWorker(DownloadWorker):
    """Worker class for downloading files with HPC-specific handling."""
    
    def __init__(self, worker_id, job_queue, data_source, tmp_dir, 
                workflow_context, stop_event, download_index, batch_manager,
                rate_limiter=None, sleep_between_downloads=1):
        """
        Initialize the HPC download worker.
        
        Args:
            worker_id: Worker ID
            job_queue: Queue for download jobs
            data_source: Data source
            tmp_dir: Temporary directory
            workflow_context: HPC workflow context
            stop_event: Event to signal stopping
            download_index: HPC download index
            batch_manager: Batch manager for creating tar balls
            rate_limiter: Optional rate limiter
            sleep_between_downloads: Sleep time between downloads
        """
        super().__init__(
            worker_id=worker_id,
            job_queue=job_queue,
            data_source=data_source,
            tmp_dir=tmp_dir,
            workflow_context=workflow_context,
            stop_event=stop_event,
            download_index=download_index,
            rate_limiter=rate_limiter,
            sleep_between_downloads=sleep_between_downloads
        )
        self.batch_manager = batch_manager
    
    def process_file(self, file_info):
        """Process a single file download with HPC-specific handling."""
        file_hash = file_info["file_hash"]
        source_url = file_info["source_url"]
        relative_path = file_info["relative_path"]
        destination_blob = file_info["destination_blob"]
        
        # Skip if already processed successfully
        if self._should_skip_file(file_hash):
            logger.debug(f"Worker {self.worker_id}: File already downloaded: {relative_path}")
            return True
        
        # Download parameters
        max_retries = 5
        attempt = 0
        success = False
        
        # Check for session refresh
        self._manage_session()
        
        while attempt < max_retries and not success:
            attempt += 1
            try:
                # Acquire rate limiter if using one
                if self.rate_limiter:
                    self.rate_limiter.acquire()
                
                try:
                    # Prepare local path
                    local_path = os.path.join(self.tmp_dir, os.path.basename(relative_path))
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    
                    logger.info(f"Worker {self.worker_id}: Downloading {relative_path} (attempt {attempt}/{max_retries})")
                    
                    # Download file
                    self.data_source.download(source_url, local_path, session=self.session)
                    
                    # Update file size in index
                    file_size = os.path.getsize(local_path)
                    
                    # Add to batch instead of uploading to GCS
                    batch_id = self.batch_manager.add_file(file_info, local_path)
                    
                    # Clean up temporary file
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    
                    logger.info(f"Worker {self.worker_id}: Successfully downloaded {relative_path} to batch {batch_id}")
                    
                    # Update status cache - keep as "pending" until fully processed
                    self.status_cache[file_hash] = "pending"
                    success = True
                    
                finally:
                    if self.rate_limiter:
                        self.rate_limiter.release()
            
            except Exception as e:
                if not self._handle_download_error(e, attempt, max_retries, file_info):
                    break
        
        # Add a configurable sleep after each download attempt
        sleep_time = getattr(self, 'sleep_between_downloads', 1)
        if sleep_time > 0:
            logger.debug(f"Worker {self.worker_id}: Sleeping for {sleep_time}s before next download")
            time.sleep(sleep_time)
    
        return success


class HPCTaskHandlers(TaskHandlers):
    """Handlers for different HPC task modes."""
    
    @staticmethod
    def handle_index(data_source, download_index, context, task_config):
        """Handle index-building task."""
        logger.info(f"Building index for {data_source.DATA_SOURCE_NAME}")
        # Use either refresh_index (deprecated) or build_index_from_source
        if task_config.get("rebuild", False):
            download_index.build_index_from_source(
                data_source,
                rebuild=True,
                check_gcs=task_config.get("check_existing", False),
                only_missing_entrypoints=task_config.get("only_missing", True),
                force_refresh_gcs=task_config.get("force_refresh", False)
            )
        else:
            download_index.refresh_index(data_source)
        logger.info("Index built successfully")
    
    @staticmethod
    def handle_download(data_source, download_index, context, task_config):
        """Handle download task with HPC integration."""
        max_concurrent = task_config.get("max_concurrent_downloads", 10)
        max_queue_size = task_config.get("max_queue_size", 1000)
        
        # Batch configuration
        batch_size = task_config.get("batch_size", 500)
        max_batch_size_mb = task_config.get("max_batch_size_mb", 5000)
        
        # Create batch manager
        batch_manager = BatchManager(
            context=context,
            index=download_index,
            batch_size=batch_size,
            max_batch_size_mb=max_batch_size_mb
        )
        
        # Create transfer manager
        transfer_manager = RsyncTransferManager(
            context=context,
            index=download_index,
            rsync_options=task_config.get("rsync_options", {
                "compress": True,
                "archive": True,
                "partial": True,
                "verbose": True,
                "bwlimit": task_config.get("rsync_bandwidth_limit", 0)
            }),
            max_concurrent=task_config.get("max_concurrent_transfers", 2)
        )
        
        # Use temporary directory for downloads
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger.info(f"Using temporary directory: {tmp_dir}")
            
            # Create bounded job queue
            job_queue = queue.Queue(maxsize=max_queue_size)
            
            # Create stop event for workers
            stop_event = threading.Event()
            
            # Limit concurrent downloads to a reasonable number
            worker_count = min(max_concurrent, 20)
            
            # Create rate limiter
            rate_limiter = threading.BoundedSemaphore(value=worker_count)
            
            # Create and start workers
            workers = []
            for i in range(worker_count):
                worker = HPCDownloadWorker(
                    worker_id=i+1,
                    job_queue=job_queue,
                    data_source=data_source,
                    tmp_dir=tmp_dir,
                    workflow_context=context,
                    stop_event=stop_event,
                    download_index=download_index,
                    batch_manager=batch_manager,
                    rate_limiter=rate_limiter,
                    sleep_between_downloads=task_config.get("sleep_between_downloads", 1)
                )
                thread = threading.Thread(target=worker.run)
                thread.daemon = True
                thread.start()
                workers.append(thread)
            
            # Start transfer workers
            transfer_manager.start_transfer_workers(
                task_config.get("transfer_workers", transfer_manager.max_concurrent)
            )
            
            # Create a batch finalizer thread
            def batch_finalizer():
                while not stop_event.is_set() or not job_queue.empty():
                    # Sleep to allow files to accumulate in the current batch
                    time.sleep(10)
                    
                    # Only finalize if there are files and the queue is getting low
                    if (batch_manager.current_batch_files and 
                        (job_queue.empty() or len(batch_manager.current_batch_files) >= batch_manager.batch_size * 0.8)):
                        
                        tar_path = batch_manager.finalize_current_batch()
                        if tar_path:
                            batch_id = os.path.basename(tar_path).replace('.tar.gz', '')
                            transfer_manager.queue_batch(batch_id, tar_path)
                
                # Final cleanup - make sure last batch is finalized
                if batch_manager.current_batch_files:
                    tar_path = batch_manager.finalize_current_batch()
                    if tar_path:
                        batch_id = os.path.basename(tar_path).replace('.tar.gz', '')
                        transfer_manager.queue_batch(batch_id, tar_path)
            
            # Start batch finalizer thread
            finalizer_thread = threading.Thread(target=batch_finalizer)
            finalizer_thread.daemon = True
            finalizer_thread.start()
            
            try:
                # Start progress reporting thread
                progress_thread = setup_progress_reporting(download_index, job_queue, stop_event)
                
                # Queue files in batches
                start_time = time.time()
                logger.info("Starting to queue pending downloads")
                query_batch_size = 500
                
                total_queued = queue_files_for_download(download_index, job_queue, query_batch_size, max_queue_size)
                
                elapsed = time.time() - start_time
                logger.info(f"Finished queueing {total_queued} files in {elapsed:.1f} seconds")
                
                if job_queue.empty():
                    logger.info("No files to download")
                    return
                
                # Wait for queue to empty with monitoring
                logger.info("Waiting for downloads to complete")
                
                try:
                    # Wait until all tasks are done
                    job_queue.join()
                    
                    # Ensure batch finalizer has time to finalize the last batch
                    time.sleep(5)
                    
                    # Wait for transfer queue to empty
                    logger.info("Waiting for transfers to complete")
                    transfer_manager.transfer_queue.join()
                    
                except KeyboardInterrupt:
                    logger.warning("Download interrupted by user")
                except Exception as e:
                    logger.error(f"Error during download: {str(e)}")
                
                # Print final statistics
                stats = download_index.get_stats()
                logger.info(f"Download complete: {stats.get('files_success', 0)} successful, "
                          f"{stats.get('files_pending', 0)} pending, "
                          f"{stats.get('files_failed', 0)} failed")
                
                # Clean up processed batches after successful completion
                logger.info("Cleaning up processed batch files")
                batch_manager.cleanup_all_processed_batches()
                
            finally:
                # Signal workers to stop
                logger.info("Signaling workers to stop")
                stop_event.set()
                
                # Wait for workers to finish (briefly)
                for worker_thread in workers:
                    worker_thread.join(timeout=2.0)
                
                # Final cleanup of all batch files regardless of status
                if task_config.get("cleanup_on_exit", True):
                    logger.info("Final cleanup of all batch files")
                    batch_manager.cleanup_all()
                    
    @staticmethod
    def handle_extract(data_source, download_index, context, task_config):
        """Handle extraction of transferred batches on HPC."""
        logger.info(f"Extracting transferred batches for {data_source.DATA_SOURCE_NAME}")
        
        # Create transfer manager for extraction
        transfer_manager = RsyncTransferManager(
            context=context,
            index=download_index,
            rsync_options=task_config.get("rsync_options", {
                "compress": True,
                "archive": True,
                "partial": True,
                "verbose": True,
                "bwlimit": task_config.get("rsync_bandwidth_limit", 0)
            })
        )
        
        # Get pending batches
        conn = download_index._get_connection()
        cursor = conn.cursor()
        
        # Find batches that have been transferred but not yet extracted
        cursor.execute(
            "SELECT batch_id FROM batches WHERE status IN ('pending', 'transferring', 'transferred')"
        )
        batches = cursor.fetchall()
        
        if not batches:
            logger.info("No batches pending extraction")
            return
            
        logger.info(f"Found {len(batches)} batches to extract")
        
        # Extract each batch
        successful = 0
        failed = 0
        
        for batch_row in batches:
            batch_id = batch_row[0]
            logger.info(f"Extracting batch {batch_id}")
            
            try:
                if transfer_manager.extract_batch(batch_id):
                    successful += 1
                    logger.info(f"Successfully extracted batch {batch_id}")
                else:
                    failed += 1
                    logger.error(f"Failed to extract batch {batch_id}")
            except Exception as e:
                failed += 1
                logger.error(f"Error extracting batch {batch_id}: {e}")
                
                # Update batch status
                cursor.execute(
                    "UPDATE batches SET status = ?, error = ? WHERE batch_id = ?",
                    ("failed_extraction", str(e), batch_id)
                )
                conn.commit()
        
        logger.info(f"Extraction complete: {successful} successful, {failed} failed")
        
    @staticmethod
    def handle_validate(data_source, download_index, context, task_config):
        """
        Handle validation of download workflow for files on HPC.
        Checks if files exist on the HPC system and updates the index accordingly.
        """
        logger.info(f"Starting HPC validation for {data_source.DATA_SOURCE_NAME}")
        
        # Force refresh option from task config
        force_refresh = task_config.get("force_refresh_hpc", False)
        
        # Get ignore patterns from task config or use default
        ignore_patterns = task_config.get("ignore_patterns", ["/intermediate/", "/annual/"])
        if ignore_patterns:
            logger.info(f"Ignoring paths containing: {', '.join(ignore_patterns)}")
        
        # Sample percentage for validation (useful for large datasets)
        sample_percentage = task_config.get("sample_percentage", 100)
        if sample_percentage < 100:
            logger.info(f"Using sampling: validating {sample_percentage}% of files")
        
        # Fix issues option (redownload missing files)
        fix_issues = task_config.get("fix_issues", False)
        if fix_issues:
            logger.info("Fix issues mode enabled: will attempt to redownload missing files")
        
        # Create HPC client for validation
        hpc_client = context.hpc_client if hasattr(context, 'hpc_client') else HPCClient(
            context.hpc_target, 
            key_file=context.key_file
        )
        
        # Validate against HPC - changed function name here
        stats = validate_downloads(
            download_index=download_index,
            hpc_client=hpc_client,
            data_source=data_source,
            context=context,
            force_refresh=force_refresh,
            ignore_patterns=ignore_patterns,
            sample_percentage=sample_percentage,
            fix_issues=fix_issues
        )
        
        # Summarize results
        logger.info(f"Validation complete:")
        logger.info(f"  - Updated {stats['updated']} file statuses to 'success'") 
        logger.info(f"  - Added {stats['added']} new files to the index")
        logger.info(f"  - Found {stats['missing']} files missing from HPC (marked as 'pending')")
        
        # Save index
        download_index.save()
        
        # Show final statistics
        index_stats = download_index.get_stats()
        logger.info(f"Index status: {index_stats.get('files_success', 0)} successful files, "
                f"{index_stats.get('files_pending', 0)} pending, "
                f"{index_stats.get('files_failed', 0)} failed")
        
        # If fix issues is enabled and there are missing files, suggest rerunning download
        if fix_issues and stats['missing'] > 0:
            logger.info(f"Found {stats['missing']} missing files. To download them, run:")
            logger.info(f"  python run.py download --config <config_file> --source {data_source.DATA_SOURCE_NAME}")

def validate_downloads(download_index, hpc_client, data_source, context, 
                     force_refresh=False, ignore_patterns=None,
                     sample_percentage=100, fix_issues=False):
    """
    Validate download index against files actually on HPC.
    
    Args:
        download_index: Download index to validate
        hpc_client: HPC client for remote operations
        data_source: Data source object
        context: Workflow context
        force_refresh: Force refresh of file list
        ignore_patterns: List of patterns to ignore in file paths
        sample_percentage: Percentage of files to sample for validation
        fix_issues: Whether to mark missing files for redownload
        
    Returns:
        Stats dictionary with updated/added/missing counts
    """
    # Initialize stats
    stats = {"updated": 0, "added": 0, "missing": 0}
    logger.info(f"Validating index against HPC for {download_index.data_source_name}")
    
    try:
        # Get connection for database updates
        conn = download_index._get_connection()
        cursor = conn.cursor()
        
        # Get list of files in the index that should be on HPC
        cursor.execute(
            "SELECT file_hash, destination_blob, status FROM files WHERE status IN ('success', 'transferred', 'extracted')"
        )
        files_to_check = cursor.fetchall()
        
        # Apply sampling if requested
        if sample_percentage < 100:
            sample_size = int(len(files_to_check) * sample_percentage / 100)
            files_to_check = random.sample(files_to_check, sample_size)
        
        logger.info(f"Checking {len(files_to_check)} files on HPC")
        
        # Check files on HPC
        for file_hash, destination_blob, status in tqdm(files_to_check, desc="Validating files"):
            # Skip if file matches ignore pattern
            if ignore_patterns and any(re.search(pattern, destination_blob) for pattern in ignore_patterns):
                continue
                
            # Generate the path with "raw" just before the filename
            # First, split the path into directory and filename
            path_parts = destination_blob.split("/")
            if len(path_parts) >= 2:
                dir_parts = path_parts[:-1]  # All except filename
                filename = path_parts[-1]    # Just the filename
                
                # Insert "raw" directory just before the filename
                raw_path = "/".join(dir_parts) + "/raw/" + filename
                
                # Build the full remote path
                remote_path = f"{context.hpc_path}/{raw_path}"
                
                # Log the path transformation (in debug mode only)
                logger.debug(f"Checking for file at: {remote_path} (transformed from: {destination_blob})")
            else:
                # If there's no directory structure, prepend raw/
                raw_path = f"raw/{destination_blob}"
                remote_path = f"{context.hpc_path}/{raw_path}"
                logger.debug(f"Simple path, checking: {remote_path}")
            
            # Check if file exists on HPC
            file_exists = hpc_client.check_file_exists(remote_path)
            
            if not file_exists and fix_issues:
                # Mark file for redownload
                logger.warning(f"File missing on HPC: {raw_path} (from {destination_blob}) - marking for redownload")
                cursor.execute(
                    "UPDATE files SET status = 'pending', error = 'File missing on HPC' WHERE file_hash = ?",
                    (file_hash,)
                )
                stats["missing"] += 1
            elif not file_exists:
                # Just report missing file
                logger.warning(f"File missing on HPC: {raw_path} (from {destination_blob})")
                stats["missing"] += 1
        
        # Commit changes if any files were marked for redownload
        if stats["missing"] > 0 and fix_issues:
            conn.commit()
        
        logger.info(f"Validation complete: {stats['missing']} files missing, {stats['updated']} updated")
    
    except Exception as e:
        logger.error(f"Error validating downloads: {e}")
        logger.debug(traceback.format_exc())
        
    return stats

def run_hpc_workflow(config_path: Union[str, Path]) -> None:
    """
    Run the complete HPC download workflow defined in the configuration file.
    
    Args:
        config_path: Path to the workflow configuration file
    """
    # Load configuration with environment variables expanded
    config = load_config_with_env_vars(config_path)
    
    # Get HPC target
    hpc_target = config.get("hpc_target")
    if not hpc_target:
        raise ValueError("No HPC target specified in configuration")
    
    # Get sync configuration
    sync_strategy = config.get("index_sync", "auto")
    force_sync = config.get("force_sync", False)
    key_file = config.get("hpc_ssh_key_file")  # Get SSH key file path if specified
    
    # Get cleanup configuration
    cleanup_on_exit = config.get("cleanup_on_exit", True)
    
    # Create a shared workflow context
    context = HPCWorkflowContext(
        hpc_target=hpc_target,
        local_index_dir=config.get("local_index_dir"),
        key_file=key_file
    )
    
    # Get tasks from configuration
    tasks = config.get("tasks", [])
    if not tasks:
        raise ValueError("No tasks specified in the configuration")
    
    # Process each task in order
    logger.info(f"Starting workflow with {len(tasks)} tasks")
    
    for i, task in enumerate(tasks):
        logger.info(f"Task {i+1}/{len(tasks)}: {task.get('data_source', 'unknown')}")
        
        try:
            # Get data source name and mode
            data_source_name = task.pop("data_source")
            mode = task.pop("mode", "download")  # download, index, validate
            
            # Add cleanup flag to task config
            task["cleanup_on_exit"] = cleanup_on_exit
            
            # Create data source instance using the factory
            data_source = create_data_source(data_source_name, task)
            
            # Create download index to track progress
            index = HPCDataDownloadIndex(
                bucket_name="dummy", # not used
                data_source=data_source,
                local_index_dir=context.local_index_dir,
                temp_dir=context.staging_dir
            )
            
            # Synchronize the index at the beginning
            # For index operations, we want the latest from HPC
            # For download operations, we need to check which is best
            initial_sync = True
            if mode in ["index", "validate"]:
                # For these operations, we want to pull any existing data first
                # unless forcing a rebuild
                if "rebuild" in task and task["rebuild"] and mode == "index":
                    # If rebuilding, push is fine
                    initial_sync = index.ensure_synced_index(
                        hpc_target, sync_direction="auto", force=force_sync,
                        key_file=key_file
                    )
                else:
                    # Otherwise pull first to avoid losing data
                    initial_sync = index.ensure_synced_index(
                        hpc_target, sync_direction="pull", force=force_sync,
                        key_file=key_file
                    )
            else:
                # For download tasks, auto-detect is usually best
                initial_sync = index.ensure_synced_index(
                    hpc_target, sync_direction=sync_strategy, force=force_sync,
                    key_file=key_file
                )
                
            if not initial_sync:
                logger.warning(f"Initial index sync failed, proceeding with local index")
            
            # Log task start
            logger.info(f"Starting {data_source_name} task with mode '{mode}'")
            
            # Execute the task 
            start_time = time.time()
            
            # For download tasks, simplify intermediate statuses
            if mode == "download":
                # Simplify by consolidating intermediate statuses to 'pending'
                conn = index._get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE files SET status = ? WHERE status IN (?, ?, ?, ?)",
                    ("pending", "batched", "transferring", "extracted", "transferred")
                )
                conn.commit()
            
            # Select and execute appropriate handler based on mode
            if mode == "index":
                HPCTaskHandlers.handle_index(data_source, index, context, task)
            elif mode == "download":
                HPCTaskHandlers.handle_download(data_source, index, context, task)
            elif mode == "validate_download":
                HPCTaskHandlers.handle_validate(data_source, index, context, task)
            elif mode == "extract":
                HPCTaskHandlers.handle_extract(data_source, index, context, task)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            
            duration = time.time() - start_time
            logger.info(f"Completed {data_source_name} task in {duration:.1f} seconds")
            
            # Synchronize the index at the end of the operation to ensure HPC has latest
            # Skip sync if the operation failed
            if mode in ["index", "download"]:
                # Always push after index/download operations
                final_sync = index.ensure_synced_index(
                    hpc_target, sync_direction="push", force=False,
                    key_file=key_file
                )
                if not final_sync:
                    logger.warning(f"Final index sync failed, HPC index may be out of date")
            
            # Final cleanup if enabled
            if cleanup_on_exit and mode == "download":
                # Create a batch manager for cleanup
                batch_manager = BatchManager(context, index)
                
                # Clean up all processed batches
                logger.info("Final cleanup of all batch files")
                batch_manager.cleanup_all()
            
        except Exception as e:
            logger.error(f"Error processing task with {task.get('data_source', 'unknown')}: {str(e)}")
            
            # Create a batch manager for emergency cleanup
            try:
                if cleanup_on_exit:
                    batch_manager = BatchManager(
                        context=context,
                        index=index
                    )
                    batch_manager.cleanup_all()
                    logger.info("Performed emergency cleanup after error")
            except Exception as cleanup_error:
                logger.error(f"Error during emergency cleanup: {cleanup_error}")
            
            raise
    
    logger.info("Workflow completed successfully")

def run_hpc_workflow_with_config(config: Dict[str, Any]) -> None:
    """
    Run the HPC workflow with the unified configuration format.
    This converts the unified configuration to the task-based format expected by
    the existing workflow implementation.
    
    Args:
        config: Unified configuration dictionary
    """
    logger.debug("Converting unified config to task-based format")
    
    # Extract the source name and mode
    source_name = config.get('data_source')
    mode = config.get('mode', 'download')
    
    if not source_name:
        raise ValueError("No data source specified in configuration")
    
    # Create task from unified config
    task = config.copy()
    
    # Ensure data_source is set in the task config
    task['data_source'] = source_name
    task['mode'] = mode
    
    # Extract HPC target and SSH key file
    hpc_target = config.get('hpc_target')
    if not hpc_target and 'hpc' in config:
        hpc_target = config['hpc'].get('target')
    
    if not hpc_target:
        logger.warning("No HPC target specified in configuration")
        raise ValueError("HPC target is required for HPC workflow")
    
    # Get SSH key file - check both top level and under 'hpc' section
    ssh_key_file = config.get('ssh_key_file')
    if not ssh_key_file and 'hpc' in config:
        ssh_key_file = config['hpc'].get('ssh_key_file')
    
    # Get index synchronization settings
    sync_strategy = config.get('index_sync', 'auto')
    force_sync = config.get('force_sync', False)
    
    # Get cleanup settings
    cleanup_on_exit = config.get('cleanup_on_exit', True)
    
    # Create task-based config structure
    task_config = {
        "hpc_target": hpc_target,
        "index_sync": sync_strategy,
        "force_sync": force_sync,
        "local_index_dir": config.get('local_index_dir'),
        "ssh_key_file": ssh_key_file,  # Add SSH key file to task config
        "cleanup_on_exit": cleanup_on_exit,  # Add cleanup flag
        "tasks": [task]
    }
    
    # Debug log the SSH key file for troubleshooting
    logger.debug(f"Using SSH key file: {ssh_key_file}")
    
    # Copy additional global settings
    for key, value in config.items():
        if key.startswith('hpc_') and key != 'hpc_target':
            task_config[key] = value
    
    # Add bucket name if found
    if 'gcs_bucket_name' in config:
        task_config['bucket_name'] = config['gcs_bucket_name']
    
    # Use a temporary file for the task config
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as temp_file:
        yaml.dump(task_config, temp_file)
        temp_path = temp_file.name
    
    try:
        # Log what we're doing
        logger.info(f"Running HPC workflow for {source_name} in mode {mode}")
        logger.debug(f"Using temporary config file: {temp_path}")
        
        # Run the workflow
        run_hpc_workflow(temp_path)
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            logger.debug(f"Removed temporary config file: {temp_path}")


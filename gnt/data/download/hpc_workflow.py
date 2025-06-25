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
import shutil  # Ensure shutil is imported
import glob
import re
import traceback
import tarfile  # Add this import
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from datetime import datetime, timedelta
from tqdm import tqdm
import sqlite3  # Missing import added

from gnt.data.common.index.hpc_download_index import HPCDataDownloadIndex, FileStatus, BatchStatus
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
        # Ensure local_index_dir is fully expanded
        self.local_index_dir = os.path.expanduser(local_index_dir or "~/hpc_data_index")
        self.key_file = key_file
        os.makedirs(self.local_index_dir, exist_ok=True)
        
        # Log SSH key file info for debugging
        if key_file:
            logger.debug(f"Using SSH key file: {key_file}")
            expanded_key = os.path.expanduser(key_file) if '~' in key_file else key_file
            if os.path.exists(expanded_key):
                logger.debug(f"SSH key file exists: {expanded_key}")
            else:
                logger.warning(f"SSH key file does not exist: {expanded_key}")
        
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
            
        # Create staging directory - ensure path is fully expanded
        self.staging_dir = os.path.join(self.local_index_dir, "staging")
        os.makedirs(self.staging_dir, exist_ok=True)
        
        logger.debug(f"Initialized HPC context with host: {self.hpc_host}, path: {self.hpc_path}")
        logger.debug(f"Local index directory: {self.local_index_dir}")
        logger.debug(f"Staging directory: {self.staging_dir}")
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
    
    def _should_skip_file(self, file_hash):
        """
        Check if a file should be skipped because it's already processed.
        Extends the parent method to also check for files in batches.
        
        Args:
            file_hash: Hash of the file to check
            
        Returns:
            bool: True if the file should be skipped
        """
        # First check the cache for improved performance
        if file_hash in self.status_cache:
            status = self.status_cache[file_hash]
            # Skip files that are already in any stage of being processed
            if status in [FileStatus.SUCCESS, FileStatus.BATCHED, FileStatus.TRANSFERRED, 
                        FileStatus.EXTRACTING, FileStatus.EXTRACTED]:
                return True
        
        # Then check the database for this file
        conn = self.download_index._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status, batch_id FROM files WHERE file_hash = ?",
            (file_hash,)
        )
        result = cursor.fetchone()
        
        if not result:
            return False  # File not in database
            
        status, batch_id = result
        
        # Skip if file is already in a batch or transferred/extracted
        if status in [FileStatus.SUCCESS, FileStatus.BATCHED, FileStatus.TRANSFERRED, 
                    FileStatus.EXTRACTING, FileStatus.EXTRACTED]:
            # Update cache
            self.status_cache[file_hash] = status
            
            logger.debug(f"Skipping file {file_hash} with status {status}")
            
            # If the file is in a batch, check if the batch is being processed
            if batch_id:
                # Check batch status
                cursor.execute(
                    "SELECT status FROM batches WHERE batch_id = ?",
                    (batch_id,)
                )
                batch_result = cursor.fetchone()
                
                if batch_result and batch_result[0] in [BatchStatus.FAILED, BatchStatus.FAILED_EXTRACTION]:
                    # If the batch failed, we should reprocess the file
                    logger.info(f"File {file_hash} was in failed batch {batch_id}, will redownload")
                    return False
            
            return True
            
        return False
    
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
        db_error_count = 0
        
        # Check for session refresh
        self._manage_session()
        
        try:
            # Mark file as downloading
            conn = self.download_index._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE files SET status = ? WHERE file_hash = ? AND status = ?",
                (FileStatus.DOWNLOADING, file_hash, FileStatus.PENDING)
            )
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error when marking file as downloading: {e}")
            # Attempt to recover from database errors
            if hasattr(self.download_index, '_recover_database'):
                self.download_index._recover_database()
            return False
        
        # If the data source supports selenium, get or create a persistent session
        selenium_session = None
        persistent_session_key = getattr(self.data_source, "DATA_SOURCE_NAME", None)
        needs_selenium = hasattr(self.data_source, "requires_selenium") and self.data_source.requires_selenium
        
        if persistent_session_key and needs_selenium and hasattr(self.data_source, "get_selenium_session"):
            try:
                # Get persistent session from context, or create if needed
                selenium_session = self.workflow_context.get_persistent_session(
                    f"{persistent_session_key}_selenium",
                    self.data_source.get_selenium_session
                )
                # Make it available to the data source for this download
                self.data_source._selenium_session = selenium_session
            except Exception as e:
                logger.warning(f"Error getting persistent selenium session: {e}, will create temporary session")
        
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
                    
                    # Download file - data source will use the injected selenium_session if needed
                    self.data_source.download(source_url, local_path, session=self.session)
                    
                    try:
                        # Mark as downloaded temporarily before adding to batch
                        conn = self.download_index._get_connection()
                        cursor = conn.cursor()
                        cursor.execute(
                            "UPDATE files SET status = ? WHERE file_hash = ?",
                            (FileStatus.DOWNLOADED, file_hash)
                        )
                        conn.commit()
                        
                        # Update file size in index
                        file_size = os.path.getsize(local_path)
                        
                        # Update the batch_id in the database
                        batch_id = self.batch_manager.add_file(file_info, local_path)
                        
                        # Update batch_id in database if successfully added to a batch
                        if batch_id:
                            cursor.execute(
                                "UPDATE files SET batch_id = ?, status = ? WHERE file_hash = ?",
                                (batch_id, FileStatus.BATCHED, file_hash)
                            )
                            conn.commit()
                            
                            logger.info(f"Worker {self.worker_id}: Successfully downloaded {relative_path} to batch {batch_id}")
                            
                            # Update status cache
                            self.status_cache[file_hash] = FileStatus.BATCHED
                            success = True
                        else:
                            logger.error(f"Worker {self.worker_id}: Failed to add file to batch: {relative_path}")
                            success = False
                    except sqlite3.Error as e:
                        db_error_count += 1
                        logger.error(f"Database error during file processing: {e}")
                        
                        # If we've had multiple database errors, attempt recovery
                        if db_error_count >= 2 and hasattr(self.download_index, '_recover_database'):
                            logger.warning("Multiple database errors detected, attempting recovery")
                            self.download_index._recover_database()
                        
                        # Don't mark as success yet, but we can continue to next attempt
                        success = False
                    
                    # Clean up temporary file - only after it's been copied to the batch directory
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    
                finally:
                    if self.rate_limiter:
                        self.rate_limiter.release()
            
            except Exception as e:
                # Check if we need to refresh the selenium session
                if selenium_session is not None and ("connection" in str(e).lower() or 
                                                    "stale" in str(e).lower() or
                                                    "timeout" in str(e).lower()):
                    logger.warning(f"Selenium session may be stale. Closing and refreshing for next attempt.")
                    try:
                        # Close the current session
                        self.workflow_context.close_persistent_session(f"{persistent_session_key}_selenium")
                        # Reset the reference so we'll create a new one on next file
                        self.data_source._selenium_session = None
                        selenium_session = None
                    except Exception as close_error:
                        logger.error(f"Error closing stale selenium session: {close_error}")
                
                if not self._handle_download_error(e, attempt, max_retries, file_info):
                    break
        
        # Add a configurable sleep after each download attempt
        sleep_time = getattr(self, 'sleep_between_downloads', 1)
        if sleep_time > 0:
            logger.debug(f"Worker {self.worker_id}: Sleeping for {sleep_time}s before next download")
            time.sleep(sleep_time)
    
        return success
        
    def _handle_download_error(self, error, attempt, max_retries, file_info):
        """
        Handle download error with improved database error handling.
        
        Args:
            error: The exception that occurred
            attempt: Current attempt number
            max_retries: Maximum retry count
            file_info: Information about the file being downloaded
            
        Returns:
            bool: Whether to continue retrying
        """
        error_str = str(error)
        file_hash = file_info["file_hash"]
        source_url = file_info["source_url"]
        relative_path = file_info["relative_path"]
        
        # Special handling for database errors
        if isinstance(error, sqlite3.Error) or "database disk image is malformed" in error_str:
            logger.warning(f"Worker {self.worker_id}: Database error for {relative_path}: {error_str}")
            
            # Attempt database recovery if available
            if hasattr(self.download_index, '_recover_database'):
                logger.info(f"Attempting database recovery due to error")
                self.download_index._recover_database()
                
            # Always retry on database errors if we haven't hit max retries
            if attempt < max_retries:
                logger.info(f"Will retry after database error recovery")
                # Sleep a bit longer to allow recovery
                time.sleep(5)
                return True
        
        # For other errors, use normal handling from parent class
        try:
            # Log the error
            logger.warning(f"Worker {self.worker_id}: Download error for {relative_path} (attempt {attempt}/{max_retries}): {error_str}")
            
            # Handle specific error cases
            if "timeout" in error_str.lower() or "read timed out" in error_str.lower():
                # For timeouts, retry with increasing backoff
                backoff = min(30, attempt * 5)  # 5, 10, 15, ... seconds up to 30
                logger.info(f"Connection timeout, sleeping for {backoff}s before retry")
                time.sleep(backoff)
                return attempt < max_retries
                
            elif "connection" in error_str.lower() or "connection reset" in error_str.lower():
                # For connection issues, retry after a short delay
                backoff = min(20, 2 ** attempt)  # 2, 4, 8, 16 seconds
                logger.info(f"Connection error, sleeping for {backoff}s before retry")
                time.sleep(backoff)
                return attempt < max_retries
                
            elif "404" in error_str or "not found" in error_str.lower():
                # Don't retry file not found errors
                logger.error(f"File not found (404), marking as failed: {relative_path}")
                self.mark_file_failed(file_hash, f"File not found: {error_str}")
                return False
                
            # Mark as failed if we've hit max retries
            if attempt >= max_retries:
                logger.error(f"Too many failed attempts for {relative_path}, marking as failed")
                self.mark_file_failed(file_hash, str(error))
                return False
                
            # Otherwise, retry
            time.sleep(1)  # Brief pause before retry
            return True
            
        except Exception as e:
            logger.error(f"Error in error handler: {e}")
            return False  # Don't retry if error handler itself fails
            
    def mark_file_failed(self, file_hash, error_msg):
        """
        Mark a file as failed with error handling.
        
        Args:
            file_hash: Hash of the file
            error_msg: Error message to record
        """
        try:
            conn = self.download_index._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE files SET status = ?, error = ? WHERE file_hash = ?",
                (FileStatus.FAILED, error_msg[:1000], file_hash)  # Limit error message length
            )
            conn.commit()
            
            # Update cache
            self.status_cache[file_hash] = FileStatus.FAILED
            
        except sqlite3.Error as e:
            logger.error(f"Database error when marking file as failed: {e}")
            # Try to recover database and retry once
            if hasattr(self.download_index, '_recover_database'):
                self.download_index._recover_database()
                try:
                    # Retry update once more
                    conn = self.download_index._get_connection()
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE files SET status = ?, error = ? WHERE file_hash = ?",
                        (FileStatus.FAILED, error_msg[:1000], file_hash)
                    )
                    conn.commit()
                except Exception:
                    # Give up silently, but log the error
                    logger.error("Failed to mark file as failed even after recovery")


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
        
        # Create transfer manager first - ensure key_file is passed
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
        
        # Log whether key_file is being used
        if hasattr(context, 'key_file') and context.key_file:
            logger.info(f"Using SSH key file for transfers: {context.key_file}")
        else:
            logger.info("No SSH key file specified, using default authentication")
        
        # First check for incomplete transfers from previous runs
        recovered = transfer_manager.check_and_recover_transfers()
        if recovered > 0:
            logger.info(f"Recovered {recovered} incomplete transfers from previous run")
        
        # Start transfer workers
        transfer_manager.start_transfer_workers(
            task_config.get("transfer_workers", transfer_manager.max_concurrent)
        )
        
        # Create batch manager with reference to transfer manager
        batch_manager = BatchManager(
            context=context,
            index=download_index,
            batch_size=batch_size,
            max_batch_size_mb=max_batch_size_mb,
            transfer_manager=transfer_manager  # Pass transfer manager to batch manager
        )
        
        # Clean up stale batches before starting to prevent errors
        batch_manager.cleanup_stale_batches()
        
        # Attach transfer_manager to context for worker access
        context.transfer_manager = transfer_manager
        
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
                    job_queue.join()
                    
                    # After all downloads, check if there's a final batch that needs processing
                    # This handles cases where the last batch wasn't full
                    logger.info("All downloads complete, finalizing any remaining batches")
                    batch_manager.finalize_all_batches()
                    
                    # Wait for transfers to complete
                    logger.info("Waiting for transfers to complete")
                    if not transfer_manager.transfer_queue.empty():
                        transfer_manager.transfer_queue.join()
                    
                except KeyboardInterrupt:
                    logger.warning("Download interrupted by user")
                except Exception as e:
                    logger.error(f"Error during download: {str(e)}")
                    
                # Print final statistics
                stats = download_index.get_stats()
                logger.info(f"Download complete: {stats.get('files_success', 0)} successful, "
                          f"{stats.get('files_in_progress', 0)} in progress, "
                          f"{stats.get('files_pending', 0)} pending, "
                          f"{stats.get('files_failed', 0)} failed")
                          
                # Clean up processed batches after successful completion
                logger.info("Cleaning up processed batch files")
                batch_manager.cleanup_all_processed_batches()
                
            finally:
                # Signal workers to stop
                logger.info("Signaling workers to stop")
                stop_event.set()
                
                # Stop batch manager
                batch_manager.stop()
                
                # Stop transfer manager
                transfer_manager.stop()
                
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
        
        # Get connection for database updates
        conn = download_index._get_connection()
        cursor = conn.cursor()
        
        # Find batches that have been transferred but not yet extracted or failed extraction
        cursor.execute(
            "SELECT batch_id, status FROM batches WHERE status IN ('transferred', 'extracting', 'failed_extraction')"
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
            batch_id, status = batch_row
            logger.info(f"Extracting batch {batch_id} (current status: {status})")
            
            # Add retry tracking for extraction
            retry_count = 0
            max_retries = 2
            
            # If previous extraction failed, get retry count
            if status == 'failed_extraction':
                cursor.execute(
                    "SELECT retry_count FROM batches WHERE batch_id = ?",
                    (batch_id,)
                )
                result = cursor.fetchone()
                if result and result[0] is not None:
                    retry_count = result[0]
                
                # Update retry count for this attempt
                retry_count += 1
                cursor.execute(
                    "UPDATE batches SET retry_count = ? WHERE batch_id = ?",
                    (retry_count, batch_id)
                )
                conn.commit()
            
            # Skip if too many retries
            if retry_count > max_retries:
                logger.warning(f"Skipping batch {batch_id} after {retry_count} failed extraction attempts")
                failed += 1
                continue
            
            try:
                if transfer_manager.extract_batch(batch_id):
                    successful += 1
                    logger.info(f"Successfully extracted batch {batch_id}")
                else:
                    failed += 1
                    logger.error(f"Failed to extract batch {batch_id}")
                    
                    # Since extraction is handled in extract_batch function,
                    # status updates are already applied there
            except Exception as e:
                failed += 1
                logger.error(f"Error extracting batch {batch_id}: {e}")
                traceback_str = traceback.format_exc()
                logger.debug(f"Traceback: {traceback.format_exc()}")
                
                # Update batch status
                cursor.execute(
                    "UPDATE batches SET status = ?, error = ?, retry_count = ? WHERE batch_id = ?",
                    (BatchStatus.FAILED_EXTRACTION, str(e), retry_count, batch_id)
                )
                
                # Update file status
                cursor.execute(
                    "UPDATE files SET status = ? WHERE batch_id = ?",
                    (FileStatus.FAILED, batch_id)
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
                
            # FIX: Simplify the path transformation - just get the filename and check in the raw directory
            filename = os.path.basename(destination_blob)
            
            # Build the full remote path
            remote_path = f"{context.hpc_path}/{download_index.data_path}/raw/{filename}"
            
            # Log the path transformation (in debug mode only)
            logger.debug(f"Checking for file at: {remote_path} (from: {destination_blob})")
            
            # Check if file exists on HPC
            file_exists = hpc_client.check_file_exists(remote_path)
            
            if not file_exists and fix_issues:
                # Mark file for redownload
                logger.warning(f"File missing on HPC: {remote_path} - marking for redownload")
                cursor.execute(
                    "UPDATE files SET status = 'pending', error = 'File missing on HPC' WHERE file_hash = ?",
                    (file_hash,)
                )
                stats["missing"] += 1
            elif not file_exists:
                # Just report missing file
                logger.warning(f"File missing on HPC: {remote_path}")
                stats["missing"] += 1
            
        # Commit changes if any files were marked for redownload
        if stats["missing"] > 0 and fix_issues:
            conn.commit()
        
        logger.info(f"Validation complete: {stats['missing']} files missing, {stats['updated']} updated")
    
    except Exception as e:
        logger.error(f"Error validating downloads: {e}")
        logger.debug(traceback.format_exc())
        
    return stats

def run_hpc_workflow_with_config(config: Dict[str, Any]) -> None:
    """
    Run the HPC workflow using a direct configuration dictionary.
    
    Args:
        config: Configuration dictionary with all required settings
    """
    # Extract HPC target
    hpc_target = config.get("hpc_target")
    if not hpc_target:
        raise ValueError("No HPC target specified in configuration")
    
    # Extract key configuration parameters
    key_file = config.get("hpc_ssh_key_file")
    local_index_dir = config.get("hpc_local_index_dir") or config.get("local_index_dir")
    sync_strategy = config.get("hpc_index_sync", "auto") or config.get("index_sync", "auto")
    force_sync = config.get("hpc_force_sync", False) or config.get("force_sync", False)
    cleanup_on_exit = config.get("cleanup_on_exit", True)
    
    # Extract batch configuration parameters
    batch_size = config.get("batch_size", 100)
    max_batch_size_mb = config.get("max_batch_size_mb", 200)
    
    # Log the key file information
    if key_file:
        expanded_key = os.path.expanduser(key_file) if '~' in key_file else key_file
        logger.info(f"Using SSH key file: {key_file} (expanded to {expanded_key})")
        if not os.path.exists(expanded_key):
            logger.warning(f"SSH key file not found: {expanded_key}")
    
    # Create workflow context
    context = HPCWorkflowContext(
        hpc_target=hpc_target,
        local_index_dir=local_index_dir,
        key_file=key_file
    )
    
    # Extract operation details
    data_source_name = config.get("data_source")
    mode = config.get("mode", "download")
    
    if not data_source_name:
        raise ValueError("No data source specified in configuration")
    
    logger.info(f"Starting {mode} workflow for {data_source_name}")
    
    try:
        # Create data source instance
        data_source = create_data_source(data_source_name, config)
        
        # Create download index
        index = HPCDataDownloadIndex(
            bucket_name="dummy",  # Not used in HPC workflow
            data_source=data_source,
            local_index_dir=context.local_index_dir,
            temp_dir=context.staging_dir,
            key_file=context.key_file,
        )
        
        # Pass the key file to the index for syncing
        index.key_file = key_file
        
        # Synchronize the index based on operation mode
        initial_sync = True
        if mode in ["index", "validate_download"]:
            # For these operations, we want to pull any existing data first
            # unless forcing a rebuild
            if config.get("rebuild", False) and mode == "index":
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
            # For download and extract tasks, auto-detect is usually best
            initial_sync = index.ensure_synced_index(
                hpc_target, sync_direction=sync_strategy, force=force_sync,
                key_file=key_file
            )
            
        if not initial_sync:
            logger.warning(f"Initial index sync failed, proceeding with local index")
        
        # Execute the operation
        start_time = time.time()
        
        # Reset problematic file statuses for download mode
        if mode == "download":
            conn = index._get_connection()
            cursor = conn.cursor()
            
            # Reset problematic statuses to 'pending'
            problematic_statuses = (
                f"'{FileStatus.BATCHED}', '{FileStatus.TRANSFERRING}', "
                f"'{FileStatus.TRANSFERRED}', '{FileStatus.EXTRACTING}', "
                f"'{FileStatus.DOWNLOADING}'"
            )
            
            cursor.execute(
                f"UPDATE files SET status = ? WHERE status IN ({problematic_statuses})",
                (FileStatus.PENDING,)
            )
            
            # Also reset batch_id for files in non-terminal states
            cursor.execute(
                "UPDATE files SET batch_id = NULL WHERE status = ?",
                (FileStatus.PENDING,)
            )
            
            conn.commit()
        
        # Execute appropriate handler based on mode
        if mode == "index":
            HPCTaskHandlers.handle_index(data_source, index, context, config)
        elif mode == "download":
            HPCTaskHandlers.handle_download(data_source, index, context, config)
        elif mode == "validate_download":
            HPCTaskHandlers.handle_validate(data_source, index, context, config)
        elif mode == "extract":
            HPCTaskHandlers.handle_extract(data_source, index, context, config)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        duration = time.time() - start_time
        logger.info(f"Completed {data_source_name} task in {duration:.1f} seconds")
        
        # Push index after operation if needed
        if mode in ["index", "download"]:
            final_sync = index.ensure_synced_index(
                hpc_target, sync_direction="push", force=False,
                key_file=key_file
            )
            if not final_sync:
                logger.warning(f"Final index sync failed, HPC index may be out of date")
        
        # Final cleanup
        if cleanup_on_exit and mode == "download":
            batch_manager = BatchManager(context, index)
            logger.info("Final cleanup of all batch files")
            batch_manager.cleanup_all()
            context.close_all_persistent_sessions()
        
    except Exception as e:
        logger.error(f"Error processing {data_source_name} with mode {mode}: {str(e)}")
        
        # Emergency cleanup
        try:
            if cleanup_on_exit:
                batch_manager = BatchManager(context, index)
                batch_manager.cleanup_all()
                context.close_all_persistent_sessions()
                logger.info("Performed emergency cleanup after error")
        except Exception as cleanup_error:
            logger.error(f"Error during emergency cleanup: {cleanup_error}")

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
        
        # Log SSH key file info for debugging
        if hasattr(context, 'key_file') and context.key_file:
            logger.debug(f"RsyncTransferManager using key file: {context.key_file}")
            expanded_key = os.path.expanduser(context.key_file) if '~' in context.key_file else context.key_file
            if os.path.exists(expanded_key):
                logger.debug(f"SSH key file exists: {expanded_key}")
            else:
                logger.warning(f"SSH key file does not exist: {expanded_key}")

        # Concurrency control
        self.max_concurrent = max_concurrent
        self.transfer_semaphore = threading.Semaphore(max_concurrent)
        
        # Queue for pending transfers
        self.transfer_queue = queue.Queue()
        
        # Transfer tracking
        self.transfers_in_progress = {}
        self.transfer_lock = threading.RLock()
        
        # Add failed transfers queue for retry
        self.failed_transfers = queue.Queue()
        
        # Add transfer health monitor
        self.transfer_monitor_stop = threading.Event()
        self.health_check_interval = 300  # 5 minutes
        self.stalled_threshold = 900  # 15 minutes
        
        # Start health monitoring thread
        self._start_health_monitor()
    
    def _start_health_monitor(self):
        """Start the transfer health monitoring thread."""
        self.monitor_thread = threading.Thread(
            target=self._health_monitor_thread,
            name="transfer-health-monitor",
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Transfer health monitor thread started")
    
    def _health_monitor_thread(self):
        """Monitor the health of ongoing transfers and detect stalled operations."""
        logger.info("Transfer health monitor running")
        
        while not self.transfer_monitor_stop.is_set():
            try:
                # Check for stalled transfers
                self._check_transfer_health()
                
                # Process any failed transfers that need retry
                self._process_failed_transfers()
                
                # Wait for next check interval
                self.transfer_monitor_stop.wait(self.health_check_interval)
            except Exception as e:
                logger.error(f"Error in transfer health monitor: {e}")
                # Continue monitoring after error
                time.sleep(60)  # Wait a bit before retrying after error
        
        logger.info("Transfer health monitor stopping")
    
    def _check_transfer_health(self):
        """Check the health of ongoing transfers and detect stalled operations."""
        current_time = time.time()
        stalled_transfers = []
        
        with self.transfer_lock:
            # Copy to avoid modification during iteration
            transfers = self.transfers_in_progress.copy()
            
            for batch_id, transfer_info in transfers.items():
                # Skip if no progress update available
                if 'last_progress_time' not in transfer_info:
                    continue
                
                # Check if transfer appears stalled
                elapsed_since_progress = current_time - transfer_info['last_progress_time']
                
                if elapsed_since_progress > self.stalled_threshold:
                    logger.warning(f"Transfer for batch {batch_id} appears stalled "
                                 f"(no progress for {elapsed_since_progress:.1f} seconds)")
                    stalled_transfers.append(batch_id)
        
        # Take action for each stalled transfer
        for batch_id in stalled_transfers:
            self._handle_stalled_transfer(batch_id)
    
    def _handle_stalled_transfer(self, batch_id):
        """Handle a stalled transfer by terminating and requeuing it."""
        with self.transfer_lock:
            if batch_id not in self.transfers_in_progress:
                return  # Transfer completed or was already handled
            
            transfer_info = self.transfers_in_progress[batch_id]
            logger.info(f"Attempting to recover stalled transfer for batch {batch_id}")
            
            # Get transfer process if available
            process = transfer_info.get('process')
            if process:
                try:
                    # Terminate the stalled process
                    process.terminate()
                    process.wait(timeout=60)
                    logger.info(f"Terminated stalled transfer process for batch {batch_id}")
                except Exception as e:
                    logger.warning(f"Error terminating stalled process: {e}")
            
            # Record the failure
            tar_path = transfer_info.get('tar_path')
            
            # Remove from in-progress tracking
            del self.transfers_in_progress[batch_id]
            
            # Mark as retrying in database
            conn = self.index._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE batches SET status = ?, error = ? WHERE batch_id = ?",
                (BatchStatus.RETRYING, f"Transfer stalled, requeuing. Last progress: {datetime.fromtimestamp(transfer_info.get('last_progress_time', 0))}", batch_id)
            )
            conn.commit()
            
        # Requeue the transfer with retry count
        retry_count = transfer_info.get('retry_count', 0) + 1
        
        # Only retry up to 3 times
        if retry_count <= 3 and tar_path and os.path.exists(tar_path):
            logger.info(f"Requeuing stalled transfer for batch {batch_id} (retry {retry_count}/3)")
            self.queue_batch(batch_id, tar_path, retry_count=retry_count)
        else:
            logger.error(f"Giving up on stalled transfer for batch {batch_id} after {retry_count} attempts")
            # Mark as permanently failed
            conn = self.index._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE batches SET status = ?, error = ? WHERE batch_id = ?",
                (BatchStatus.FAILED, f"Transfer permanently failed after {retry_count} attempts", batch_id)
            )
            # Mark files as failed
            cursor.execute(
                "UPDATE files SET status = ? WHERE batch_id = ?",
                (FileStatus.FAILED, batch_id)
            )
            conn.commit()
    
    def _process_failed_transfers(self):
        """Process failed transfers that need retry."""
        try:
            while not self.failed_transfers.empty():
                batch_id, tar_path, retry_count = self.failed_transfers.get_nowait()
                
                # Check if we should retry
                if retry_count <= 3 and os.path.exists(tar_path):
                    # Wait before retrying to avoid hammering the system
                    wait_time = retry_count * 30  # Exponential backoff
                    logger.info(f"Waiting {wait_time}s before retrying transfer for batch {batch_id}")
                    time.sleep(wait_time)
                    
                    logger.info(f"Requeuing failed transfer for batch {batch_id} (retry {retry_count}/3)")
                    self.queue_batch(batch_id, tar_path, retry_count=retry_count)
                else:
                    logger.error(f"Giving up on failed transfer for batch {batch_id} after {retry_count} attempts")
                    # Mark as permanently failed
                    conn = self.index._get_connection()
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE batches SET status = ?, error = ? WHERE batch_id = ?",
                        (BatchStatus.FAILED, f"Transfer permanently failed after {retry_count} attempts", batch_id)
                    )
                    # Mark files as failed
                    cursor.execute(
                        "UPDATE files SET status = ? WHERE batch_id = ?",
                        (FileStatus.FAILED, batch_id)
                    )
                    conn.commit()
                
                # Mark as done
                self.failed_transfers.task_done()
        except queue.Empty:
            pass  # No more failed transfers to process
        except Exception as e:
            logger.error(f"Error processing failed transfers: {e}")
    
    def queue_batch(self, batch_id: str, tar_path: str, retry_count: int = 0) -> bool:
        """
        Queue a batch for transfer.
        
        Args:
            batch_id: Batch ID
            tar_path: Path to the tar file
            retry_count: Number of times this transfer has been retried
            
        Returns:
            bool: Whether the batch was queued
        """
        if not os.path.exists(tar_path):
            logger.error(f"Cannot queue batch {batch_id}: tar file not found at {tar_path}")
            
            # Update batch status to failed
            conn = self.index._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE batches SET status = ?, error = ? WHERE batch_id = ?",
                (BatchStatus.FAILED, f"Tar file not found at {tar_path}", batch_id)
            )
            conn.commit()
            return False
        
        # Update batch status
        conn = self.index._get_connection()
        cursor = conn.cursor()
        status = BatchStatus.QUEUED
        if retry_count > 0:
            status = BatchStatus.RETRYING
        
        cursor.execute(
            "UPDATE batches SET status = ? WHERE batch_id = ?",
            (status, batch_id)
        )
        conn.commit()
        
        # Add to queue with retry count
        self.transfer_queue.put((batch_id, tar_path, retry_count))
        logger.info(f"Batch {batch_id} queued for transfer {'' if retry_count == 0 else f'(retry {retry_count})' }")
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
                batch_id, tar_path, retry_count = self.transfer_queue.get()
                
                # Acquire semaphore
                self.transfer_semaphore.acquire()
                
                try:
                    # Transfer the batch
                    self._transfer_batch(batch_id, tar_path, retry_count)
                finally:
                    # Release semaphore
                    self.transfer_semaphore.release()
                    
                    # Mark task as done
                    self.transfer_queue.task_done()
            
            except Exception as e:
                logger.error(f"Error in transfer worker: {e}")
                # Continue processing
    
    def _update_transfer_progress(self, batch_id: str, bytes_transferred: int = 0, message: str = None) -> None:
        """Update transfer progress tracking for health monitoring."""
        with self.transfer_lock:
            if batch_id in self.transfers_in_progress:
                self.transfers_in_progress[batch_id]['last_progress_time'] = time.time()
                
                if bytes_transferred:
                    self.transfers_in_progress[batch_id]['bytes_transferred'] = bytes_transferred
                
                if message:
                    self.transfers_in_progress[batch_id]['last_message'] = message
    
    def _transfer_batch(self, batch_id: str, tar_path: str, retry_count: int = 0) -> bool:
        """
        Transfer a batch to the HPC system using rsync.
        
        Args:
            batch_id: Batch ID
            tar_path: Path to the tar file
            retry_count: Number of previous retry attempts
            
        Returns:
            bool: Whether the transfer was successful
        """
        conn = self.index._get_connection()
        cursor = conn.cursor()
        
        # Update batch status to transferring
        cursor.execute(
            "UPDATE batches SET status = ? WHERE batch_id = ?",
            (BatchStatus.TRANSFERRING, batch_id)
        )
        conn.commit()
        
        # Track in-progress transfer
        with self.transfer_lock:
            self.transfers_in_progress[batch_id] = {
                "start_time": time.time(),
                "tar_path": tar_path,
                "last_progress_time": time.time(),
                "retry_count": retry_count
            }
    
        # Construct tar directory relative to HPC base path
        # Use the context.hpc_path as the base
        tar_dir = f"{self.context.hpc_path}/{self.index.data_path}/tar"
    
        # Ensure target directory exists - need to create the full path
        self.hpc_client.ensure_directory(tar_dir)
    
        # Log the path for debugging
        logger.debug(f"Transferring to full remote path: {tar_dir}")
    
        try:
            # Enhance rsync options with partial-dir for resumption
            rsync_options = self.rsync_options.copy()
            
            # Always enable partial option for resumable transfers
            rsync_options["partial"] = True
            
            # Use a hidden partial directory
            rsync_options["partial-dir"] = ".rsync-partial"
            
            # Execute the transfer using HPCClient with progress callback
            logger.info(f"Transferring batch {batch_id} to HPC using rsync")
            
            def progress_callback(progress_info):
                self._update_transfer_progress(batch_id, progress_info.get('bytes_transferred', 0), 
                                             progress_info.get('message'))
            
            success, summary, transfer_process = self.hpc_client.rsync_transfer(
                source_path=tar_path,
                target_path=tar_dir,
                source_is_local=True,
                options=rsync_options,
                show_progress=True,
                progress_callback=progress_callback,
                return_process=True
            )
            
            # Store process reference for potential termination if stalled
            with self.transfer_lock:
                if batch_id in self.transfers_in_progress:
                    self.transfers_in_progress[batch_id]['process'] = transfer_process
            
            if success:
                # Update batch status to transferred
                cursor.execute(
                    "UPDATE batches SET status = ?, transfer_timestamp = ? WHERE batch_id = ?",
                    (BatchStatus.TRANSFERRED, datetime.now().isoformat(), batch_id)
                )
                conn.commit()
                
                # Update file status to transferred
                cursor.execute(
                    "UPDATE files SET status = ? WHERE batch_id = ?",
                    (FileStatus.TRANSFERRED, batch_id)
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
                # Transfer failed, handle the failure
                error_msg = f"Rsync transfer failed: {summary}"
                logger.error(f"Rsync failed for batch {batch_id}: {summary}")
                
                with self.transfer_lock:
                    if batch_id in self.transfers_in_progress:
                        del self.transfers_in_progress[batch_id]
                
                # Queue for retry if not too many attempts
                if retry_count < 3: # Allow up to 3 retries (4 total attempts)

                    logger.info(f"Will retry transfer for batch {batch_id} (attempt {retry_count+1}/3)")
                    
                    # Update status to retrying
                    cursor.execute(
                        "UPDATE batches SET status = ?, error = ? WHERE batch_id = ?",
                        (BatchStatus.RETRYING, error_msg, batch_id)
                    )
                    conn.commit()
                    
                    # Add to failed transfers queue for retry
                    self.failed_transfers.put((batch_id, tar_path, retry_count + 1))
                else:
                    # Too many retries, mark as permanently failed
                    logger.error(f"Giving up on transfer for batch {batch_id} after {retry_count} attempts")
                    
                    cursor.execute(
                        "UPDATE batches SET status = ?, error = ? WHERE batch_id = ?",
                        (BatchStatus.FAILED, f"{error_msg} (after {retry_count} retries)", batch_id)
                    )
                    
                    # Mark files as failed
                    cursor.execute(
                        "UPDATE files SET status = ? WHERE batch_id = ?",
                        (FileStatus.FAILED, batch_id)
                    )
                    conn.commit()
                
                return False
    
        except Exception as e:
            logger.error(f"Error transferring batch {batch_id}: {e}")
            traceback_str = traceback.format_exc()
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
            cursor.execute(
                "UPDATE batches SET status = ?, error = ? WHERE batch_id = ?",
                (BatchStatus.FAILED if retry_count >= 3 else BatchStatus.RETRYING, str(e), batch_id)
            )
            
            # Only mark files as failed if we're giving up
            if retry_count >= 3:
                cursor.execute(
                    "UPDATE files SET status = ? WHERE batch_id = ?",
                    (FileStatus.FAILED, batch_id)
                )
            
            conn.commit()
            
            # Remove from in-progress tracking
            with self.transfer_lock:
                if batch_id in self.transfers_in_progress:
                    del self.transfers_in_progress[batch_id]
            
            # Add to failed transfers queue for retry if not too many attempts
            if retry_count < 3:
                self.failed_transfers.put((batch_id, tar_path, retry_count + 1))
            
            return False
    
    
    def stop(self):
        """Stop the transfer manager and clean up."""
        logger.info("Stopping transfer manager")
        
        # Stop health monitor
        if hasattr(self, 'transfer_monitor_stop'):
            self.transfer_monitor_stop.set()
            
            # Wait for monitor to end
            if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5.0)
        
        # Terminate any in-progress transfers
        with self.transfer_lock:
            for batch_id, transfer_info in self.transfers_in_progress.items():
                process = transfer_info.get('process')
                if process:
                    try:
                        logger.info(f"Terminating in-progress transfer for batch {batch_id}")
                        process.terminate()
                    except Exception as e:
                        logger.warning(f"Error terminating process for batch {batch_id}: {e}")
    
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
        
        # Mark batch as extracting
        cursor.execute(
            "UPDATE batches SET status = ? WHERE batch_id = ?",
            (BatchStatus.EXTRACTING, batch_id)
        )
        
        # Mark files as extracting
        cursor.execute(
            "UPDATE files SET status = ? WHERE batch_id = ?",
            (FileStatus.EXTRACTING, batch_id)
        )
        conn.commit()
        
        # Execute extraction on HPC using HPCClient
        tar_filename = f"{batch_id}.tar.gz"
        
        # Build paths with full HPC path prefix
        base_dir = f"{self.context.hpc_path}/{self.index.data_path}"
        # FIX: The extraction target is directly the raw directory, not a nested one
        raw_dir = f"{base_dir}/raw"
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
                    (BatchStatus.FAILED, f"Tar file not found on remote: {tar_path}", batch_id)
                )
                
                # Update file statuses to failed
                cursor.execute(
                    "UPDATE files SET status = ? WHERE batch_id = ?",
                    (FileStatus.FAILED, batch_id)
                )
                conn.commit()
                return False
            
            # Use HPCClient to extract the tar file to the raw directory
            logger.debug(f"Extracting tar file: {tar_path}")
            # FIX: Using the correct extraction function to ensure files are placed correctly
            success = self.hpc_client.extract_tar(tar_path, raw_dir)
            
            if success:
                # Update batch status to extracted (success)
                cursor.execute(
                    "UPDATE batches SET status = ? WHERE batch_id = ?",
                    (BatchStatus.EXTRACTED, batch_id)
                )
                
                # Update file statuses to success
                cursor.execute(
                    "UPDATE files SET status = ? WHERE batch_id = ?",
                    (FileStatus.SUCCESS, batch_id)
                )
                conn.commit()
                
                # Log result
                logger.info(f"Successfully extracted batch {batch_id} on HPC to {raw_dir}")
                return True
            else:
                # Extraction failed
                cursor.execute(
                    "UPDATE batches SET status = ?, error = ? WHERE batch_id = ?",
                    (BatchStatus.FAILED_EXTRACTION, "Tar extraction failed on HPC", batch_id)
                )
                
                # Mark files as failed
                cursor.execute(
                    "UPDATE files SET status = ? WHERE batch_id = ?",
                    (FileStatus.FAILED, batch_id)
                )
                conn.commit()
                
                logger.error(f"Failed to extract batch {batch_id} on HPC")
                return False
            
        except Exception as e:
            logger.error(f"Error requesting extraction for batch {batch_id}: {e}")
            traceback_str = traceback.format_exc()
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
            cursor.execute(
                "UPDATE batches SET status = ?, error = ? WHERE batch_id = ?",
                (BatchStatus.FAILED_EXTRACTION, str(e), batch_id)
            )
            
            # Mark files as failed
            cursor.execute(
                "UPDATE files SET status = ? WHERE batch_id = ?",
                (FileStatus.FAILED, batch_id)
            )
            conn.commit()
            
            return False

    def check_and_recover_transfers(self):
        """
        Check for incomplete transfers and attempt recovery.
        This can be called at startup to resume interrupted transfers.
        """
        logger.info("Checking for incomplete transfers to recover")
        try:
            conn = self.index._get_connection()
            cursor = conn.cursor()
            
            # Find batches in transferring or queued state
            incomplete_statuses = [
                BatchStatus.TRANSFERRING, 
                BatchStatus.QUEUED, 
                BatchStatus.RETRYING
            ]
            status_placeholders = ', '.join(['?'] * len(incomplete_statuses))
            
            cursor.execute(
                f"SELECT batch_id, tar_path FROM batches WHERE status IN ({status_placeholders})",
                incomplete_statuses
            )
            incomplete_batches = cursor.fetchall()
            
            recovered_count = 0
            for batch_id, tar_path in incomplete_batches:
                # Check if tar file still exists
                if tar_path and os.path.exists(tar_path):
                    logger.info(f"Recovering interrupted transfer for batch {batch_id}")
                    
                    # Queue for retry with incremented retry count
                    cursor.execute(
                        "SELECT retry_count FROM batches WHERE batch_id = ?",
                        (batch_id,)
                    )
                    result = cursor.fetchone()
                    retry_count = (result[0] if result and result[0] is not None else 0) + 1
                    
                    # Update retry count
                    cursor.execute(
                        "UPDATE batches SET retry_count = ? WHERE batch_id = ?",
                        (retry_count, batch_id)
                    )
                    conn.commit()
                    
                    # Queue for transfer
                    self.queue_batch(batch_id, tar_path, retry_count)
                    recovered_count += 1
                else:
                    logger.warning(f"Cannot recover batch {batch_id}: tar file not found at {tar_path}")
                    
                    # Mark as failed
                    cursor.execute(
                        "UPDATE batches SET status = ?, error = ? WHERE batch_id = ?",
                        (BatchStatus.FAILED, "Tar file missing during recovery", batch_id)
                    )
                    
                    # Reset files to pending
                    cursor.execute(
                        "UPDATE files SET status = ?, batch_id = NULL WHERE batch_id = ?",
                        (FileStatus.PENDING, batch_id)
                    )
                    conn.commit()
            
            logger.info(f"Recovery complete: {recovered_count} transfers recovered")
            return recovered_count
                
        except Exception as e:
            logger.error(f"Error checking for incomplete transfers: {e}")
            return 0

class BatchManager:
    """Manager for creating and tracking batches of files for transfer to HPC."""
    
    def __init__(self, context: HPCWorkflowContext, index: HPCDataDownloadIndex, 
                 batch_size: int = 100, max_batch_size_mb: int = 200,
                 transfer_manager=None):
        """
        Initialize the batch manager.
        
        Args:
            context: HPC workflow context
            index: HPC download index
            batch_size: Maximum number of files per batch
            max_batch_size_mb: Maximum size of a batch in MB
            transfer_manager: Transfer manager for handling batch transfers
        """
        self.context = context
        self.index = index
        self.batch_size = batch_size
        self.max_batch_size_bytes = max_batch_size_mb * 1024 * 1024
        self.transfer_manager = transfer_manager
        
        # Create batch directory under staging
        self.batch_dir = os.path.join(context.staging_dir, "batch")
        os.makedirs(self.batch_dir, exist_ok=True)
        
        # Track current active batch
        self.current_batch_id = None
        self.current_batch_size = 0
        self.current_batch_files = 0
        self.current_batch_lock = threading.RLock()
        
        # Lookup dictionary for batches
        self.batch_lookup = {}
        
        # Recovery - load existing batch tracking info
        self._load_batch_info()
    
    def _load_batch_info(self):
        """Load existing batch information from database."""
        try:
            conn = self.index._get_connection()
            cursor = conn.cursor()
            
            # Get all batches that are pending, ready, or queued
            cursor.execute(
                "SELECT batch_id, tar_path, file_count, total_size FROM batches "
                "WHERE status IN ('pending', 'ready', 'queued')"
            )
            for batch_id, tar_path, file_count, total_size in cursor.fetchall():
                if batch_id and os.path.isdir(os.path.join(self.batch_dir, batch_id)):
                    batch_path = os.path.join(self.batch_dir, batch_id)
                    
                    # Check which batch has the most files
                    if not self.current_batch_id or file_count > self.current_batch_files:
                        self.current_batch_id = batch_id
                        self.current_batch_files = file_count
                        self.current_batch_size = total_size or 0
                        
                    # Store in lookup
                    self.batch_lookup[batch_id] = {
                        'path': batch_path,
                        'tar_path': tar_path,
                        'file_count': file_count,
                        'total_size': total_size
                    }
                    
            # If we found an active batch, log it
            if self.current_batch_id:
                logger.info(f"Recovered active batch {self.current_batch_id} with {self.current_batch_files} files")
            
        except Exception as e:
            logger.error(f"Error loading batch info: {e}")
    
    def add_file(self, file_info, local_path):
        """
        Add a file to the current batch.
        
        Args:
            file_info: Information about the file
            local_path: Path to the downloaded file
            
        Returns:
            str: Batch ID the file was added to
        """
        file_hash = file_info["file_hash"]
        relative_path = file_info["relative_path"]
        file_size = os.path.getsize(local_path)
        
        with self.current_batch_lock:
            # Create a new batch if needed
            if not self.current_batch_id:
                self.current_batch_id = f"batch_{int(time.time())}_{random.randint(1000, 9999)}"
                self.current_batch_size = 0
                self.current_batch_files = 0
                
                # Create batch directory
                batch_path = os.path.join(self.batch_dir, self.current_batch_id)
                os.makedirs(batch_path, exist_ok=True)
                
                # Store in lookup
                self.batch_lookup[self.current_batch_id] = {
                    'path': batch_path,
                    'tar_path': None,
                    'file_count': 0,
                    'total_size': 0
                }
                
                # Create batch record in database
                conn = self.index._get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO batches (batch_id, status, created_timestamp, file_count, total_size) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (self.current_batch_id, BatchStatus.PENDING, datetime.now().isoformat(), 0, 0)
                )
                conn.commit()
                
                logger.info(f"Created new batch {self.current_batch_id}")
            
            # Copy the file to batch directory
            batch_path = self.batch_lookup[self.current_batch_id]['path']
            dest_path = os.path.join(batch_path, os.path.basename(relative_path))
            
            try:
                # Ensure any parent directories exist
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                
                # Copy file to batch directory
                shutil.copy2(local_path, dest_path)
                
                # Update batch metrics
                self.current_batch_size += file_size
                self.current_batch_files += 1
                
                # Update in lookup
                self.batch_lookup[self.current_batch_id]['file_count'] = self.current_batch_files
                self.batch_lookup[self.current_batch_id]['total_size'] = self.current_batch_size
                
                # Update batch record
                conn = self.index._get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE batches SET file_count = ?, total_size = ? WHERE batch_id = ?",
                    (self.current_batch_files, self.current_batch_size, self.current_batch_id)
                )
                conn.commit()
                
                # Check if batch is full
                if (self.current_batch_files >= self.batch_size or 
                    self.current_batch_size >= self.max_batch_size_bytes):
                    logger.info(f"Batch {self.current_batch_id} full: "
                               f"{self.current_batch_files} files, "
                               f"{self.current_batch_size / (1024*1024):.2f} MB")
                    self.finalize_batch(self.current_batch_id)
                
                # Return the batch ID
                return self.current_batch_id
                
            except Exception as e:
                logger.error(f"Error adding file to batch: {e}")
                return None
    
    def finalize_batch(self, batch_id):
        """
        Finalize a batch by creating a tar file and queuing for transfer.
        
        Args:
            batch_id: ID of the batch to finalize
            
        Returns:
            bool: Whether finalization was successful
        """
        logger.info(f"Finalizing batch {batch_id}")
        
        with self.current_batch_lock:
            # Check if this is the current batch
            if batch_id == self.current_batch_id:
                # Clear current batch so new files go into a new batch
                self.current_batch_id = None
                self.current_batch_files = 0
                self.current_batch_size = 0
        
        # Only proceed if batch exists
        if batch_id not in self.batch_lookup:
            logger.error(f"Cannot finalize batch {batch_id}: not found")
            return False
        
        try:
            # Get batch path
            batch_path = self.batch_lookup[batch_id]['path']
            
            # Create tar file in batch directory
            tar_path = os.path.join(self.context.staging_dir, f"{batch_id}.tar.gz")
            
            # Create tar file with batch contents
            logger.info(f"Creating tar file for batch {batch_id}")
            
            with tarfile.open(tar_path, "w:gz") as tar:
                # Add all files in batch directory
                for root, _, files in os.walk(batch_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, batch_path)
                        tar.add(file_path, arcname=arcname)
            
            # Update batch record
            conn = self.index._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE batches SET tar_path = ?, status = ? WHERE batch_id = ?",
                (tar_path, BatchStatus.READY, batch_id)
            )
            conn.commit()
            
            # Update in lookup
            self.batch_lookup[batch_id]['tar_path'] = tar_path
            
            # Queue batch for transfer if manager is available
            if self.transfer_manager:
                self.transfer_manager.queue_batch(batch_id, tar_path)
            
            logger.info(f"Successfully finalized batch {batch_id}, tar file at {tar_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error finalizing batch {batch_id}: {e}")
            
            # Update batch record to failed
            try:
                conn = self.index._get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE batches SET status = ?, error = ? WHERE batch_id = ?",
                    (BatchStatus.FAILED, str(e), batch_id)
                )
                conn.commit()
            except Exception as db_error:
                logger.error(f"Error updating batch record: {db_error}")
            
            return False
    
    def finalize_all_batches(self):
        """Finalize all pending batches."""
        with self.current_batch_lock:
            # Finalize current batch if any
            if self.current_batch_id:
                self.finalize_batch(self.current_batch_id)
            
            # Look for any other pending batches
            conn = self.index._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT batch_id FROM batches WHERE status = ?", (BatchStatus.PENDING,))
            
            for (batch_id,) in cursor.fetchall():
                if batch_id != self.current_batch_id:
                    self.finalize_batch(batch_id)
    
    def cleanup_batch(self, batch_id):
        """
        Clean up a processed batch.
        
        Args:
            batch_id: ID of the batch to clean up
        """
        logger.info(f"Cleaning up batch {batch_id}")
        
        try:
            # Get the batch path
            batch_info = self.batch_lookup.get(batch_id)
            if batch_info:
                batch_path = batch_info['path']
                tar_path = batch_info['tar_path']
                
                # Remove batch directory
                if batch_path and os.path.exists(batch_path):
                    shutil.rmtree(batch_path)
                
                # Remove tar file if it exists
                if tar_path and os.path.exists(tar_path):
                    os.remove(tar_path)
                
                # Remove from lookup
                if batch_id in self.batch_lookup:
                    del self.batch_lookup[batch_id]
                
                logger.info(f"Batch {batch_id} cleaned up successfully")
            else:
                logger.warning(f"Cannot clean up batch {batch_id}: not found in lookup")
        except Exception as e:
            logger.error(f"Error cleaning up batch {batch_id}: {e}")
    
    def cleanup_all_processed_batches(self):
        """Clean up all successfully processed batches."""
        logger.info("Cleaning up processed batches")
        
        try:
            conn = self.index._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT batch_id FROM batches WHERE status IN (?, ?)",
                (BatchStatus.SUCCESS, BatchStatus.EXTRACTED)
            )
            
            for (batch_id,) in cursor.fetchall():
                self.cleanup_batch(batch_id)
            
            logger.info("Finished cleaning up processed batches")
        except Exception as e:
            logger.error(f"Error cleaning up processed batches: {e}")
    
    def cleanup_stale_batches(self):
        """Clean up stale batches from the file system."""
        logger.info("Cleaning up stale batches")
        
        # Look for batch directories
        try:
            # Scan batch directory for batch folders
            if os.path.exists(self.batch_dir):
                for entry in os.listdir(self.batch_dir):
                    batch_path = os.path.join(self.batch_dir, entry)
                    if os.path.isdir(batch_path) and entry.startswith("batch_"):
                        batch_id = entry
                        
                        # Check if batch is still tracked in database
                        conn = self.index._get_connection()
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT status FROM batches WHERE batch_id = ?",
                            (batch_id,)
                        )
                        result = cursor.fetchone()
                        
                        if not result:
                            # Batch not in database, clean it up







                            logger.warning(f"Found stale batch directory {batch_id}, cleaning up")
                            shutil.rmtree(batch_path)
                        elif result[0] in [BatchStatus.SUCCESS, BatchStatus.EXTRACTED]:
                            # Batch is done, clean it up
                            logger.info(f"Cleaning up completed batch {batch_id}")
                            self.cleanup_batch(batch_id)
            
            # Scan staging directory for stale tar files
            for entry in os.listdir(self.context.staging_dir):
                if entry.endswith(".tar.gz") and entry.startswith("batch_"):
                    tar_path = os.path.join(self.context.staging_dir, entry)
                    batch_id = entry.replace(".tar.gz", "")
                    
                    # Check if batch is still tracked in database
                    conn = self.index._get_connection()
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT status FROM batches WHERE batch_id = ?",
                        (batch_id,)
                    )
                    result = cursor.fetchone()
                    
                    if not result:
                        # Tar file not in database, clean it up
                        logger.warning(f"Found stale tar file {entry}, cleaning up")
                        os.remove(tar_path)
                    elif result[0] in [BatchStatus.SUCCESS, BatchStatus.EXTRACTED]:
                        # Batch is done, clean up the tar file
                        logger.info(f"Cleaning up tar file for completed batch {batch_id}")
                        if os.path.exists(tar_path):
                            os.remove(tar_path)
        
        except Exception as e:
            logger.error(f"Error cleaning up stale batches: {e}")
    
    def cleanup_all(self):
        """Clean up all batch files regardless of status."""
        logger.info("Cleaning up all batch files")
        
        try:
            # Clean up all batch directories
            if os.path.exists(self.batch_dir):
                for entry in os.listdir(self.batch_dir):
                    entry_path = os.path.join(self.batch_dir, entry)
                    if os.path.isdir(entry_path):
                        shutil.rmtree(entry_path)
            
            # Clean up all tar files
            for entry in os.listdir(self.context.staging_dir):
                if entry.endswith(".tar.gz") and entry.startswith("batch_"):
                    tar_path = os.path.join(self.context.staging_dir, entry)
                    os.remove(tar_path)
            
            logger.info("All batch files cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up all batch files: {e}")
    
    def stop(self):
        """Stop the batch manager and clean up resources."""
        logger.info("Stopping batch manager")
        
        # Finalize current batch if any
        with self.current_batch_lock:
            if self.current_batch_id:
                try:
                    self.finalize_batch(self.current_batch_id)
                except Exception as e:
                    logger.error(f"Error finalizing batch during stop: {e}")
                    
# Define batch statuses for consistency
class BatchStatus:
    PENDING = "pending"        # Batch is being created
    READY = "ready"            # Batch is complete and ready for transfer
    QUEUED = "queued"          # Batch is queued for transfer
    RETRYING = "retrying"      # Batch is being retried after a failure
    TRANSFERRING = "transferring"  # Batch is being transferred
    TRANSFERRED = "transferred"  # Batch has been transferred but not extracted
    EXTRACTING = "extracting"  # Batch is being extracted on HPC


    FAILED = "failed"          # Batch processing failed    SUCCESS = "success"        # Batch has been successfully processed    FAILED_EXTRACTION = "failed_extraction"  # Batch extraction failed
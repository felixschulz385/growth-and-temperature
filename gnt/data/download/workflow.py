"""
Base workflow for managing data downloads.
"""

import os
import re
import time
import yaml
import queue
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


def load_config_with_env_vars(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration file with environment variables expanded."""
    env_pattern = re.compile(r'\${([^}^{]+)}')
    
    # Function to replace environment variables in strings
    def replace_env_vars(value: str) -> str:
        def replace(match):
            env_var = match.group(1)
            return os.environ.get(env_var, '')
        return env_pattern.sub(replace, value)
    
    # Process all items in a structure
    def process_item(item):
        if isinstance(item, dict):
            return {k: process_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [process_item(i) for i in item]
        elif isinstance(item, str):
            return replace_env_vars(item)
        else:
            return item
            
    # Load the YAML file
    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Process environment variables
    return process_item(config)


class WorkflowContext:
    """Base context for workflow execution."""
    
    def __init__(self, bucket_name: Optional[str] = None):
        """Initialize the workflow context."""
        self.bucket_name = bucket_name
        self.staging_dir = os.path.join(os.getcwd(), "staging")
        os.makedirs(self.staging_dir, exist_ok=True)


class DownloadWorker:
    """Worker class for downloading files."""
    
    def __init__(self, worker_id, job_queue, data_source, tmp_dir, 
                workflow_context, stop_event, download_index,
                rate_limiter=None, sleep_between_downloads=1):
        """
        Initialize the download worker.
        
        Args:
            worker_id: Worker ID
            job_queue: Queue for download jobs
            data_source: Data source
            tmp_dir: Temporary directory
            workflow_context: Workflow context
            stop_event: Event to signal stopping
            download_index: Download index
            rate_limiter: Optional rate limiter
            sleep_between_downloads: Sleep time between downloads
        """
        self.worker_id = worker_id
        self.job_queue = job_queue
        self.data_source = data_source
        self.tmp_dir = tmp_dir
        self.workflow_context = workflow_context
        self.stop_event = stop_event
        self.download_index = download_index
        self.rate_limiter = rate_limiter
        self.sleep_between_downloads = sleep_between_downloads
        
        # Session for authenticated downloads
        self.session = None
        self.session_refresh_time = None
        
        # Cache of file statuses
        self.status_cache = {}
    
    def run(self):
        """Main worker loop."""
        logger.info(f"Worker {self.worker_id}: Starting download worker")
        
        try:
            # Initialize session
            self._manage_session()
            
            # Process files until stopped
            while not self.stop_event.is_set():
                try:
                    # Try to get a job from the queue with timeout
                    file_info = self.job_queue.get(timeout=1.0)
                    
                    try:
                        # Process the file
                        self.process_file(file_info)
                    except Exception as e:
                        logger.error(f"Worker {self.worker_id}: Error processing file: {str(e)}")
                    finally:
                        self.job_queue.task_done()
                        
                except queue.Empty:
                    # No work to do, check if we should exit
                    if self.job_queue.empty() and self.stop_event.is_set():
                        break
            
            logger.info(f"Worker {self.worker_id}: Stopping")
            
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Fatal error: {str(e)}")
    
    def _should_skip_file(self, file_hash):
        """Check if file should be skipped."""
        # Check cache first
        if file_hash in self.status_cache:
            if self.status_cache[file_hash] in ["success", "batched", "transferred"]:
                return True
            return False
            
        # Check index
        conn = self.download_index._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status FROM files WHERE file_hash = ?",
            (file_hash,)
        )
        result = cursor.fetchone()
        
        if result:
            status = result[0]
            self.status_cache[file_hash] = status
            return status in ["success", "batched", "transferred"]
        
        return False
    
    def _manage_session(self):
        """Initialize or refresh the session if needed."""
        # If source doesn't need authentication, skip
        if not getattr(self.data_source, "auth_required", False):
            return
            
        # Check if we need a new session
        now = time.time()
        if (self.session is None or 
            self.session_refresh_time is None or
            now - self.session_refresh_time > 1800):  # 30 minutes
            
            logger.debug(f"Worker {self.worker_id}: Creating new authenticated session")
            self.session = self.data_source.create_authenticated_session()
            self.session_refresh_time = now
    
    def _handle_download_error(self, error, attempt, max_retries, file_info):
        """Handle download error."""
        file_hash = file_info["file_hash"]
        relative_path = file_info["relative_path"]
        
        if attempt < max_retries:
            logger.warning(f"Worker {self.worker_id}: Download error for {relative_path} (attempt {attempt}/{max_retries}): {str(error)}")
            
            # Wait before retry (with exponential backoff)
            retry_delay = 2 ** attempt
            time.sleep(retry_delay)
            
            # On certain errors, refresh session
            refresh_on_errors = ["401", "403", "Session expired", "Authentication failed"]
            error_str = str(error).lower()
            if any(err.lower() in error_str for err in refresh_on_errors):
                logger.info(f"Worker {self.worker_id}: Refreshing session after auth error")
                self.session = None  # Force session refresh on next attempt
            
            return True  # Continue retrying
        else:
            # Max retries reached, mark as failed
            logger.error(f"Worker {self.worker_id}: Failed to download {relative_path} after {max_retries} attempts: {str(error)}")
            
            # Update status in index
            conn = self.download_index._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE files SET status = ?, error = ? WHERE file_hash = ?",
                ("failed", str(error), file_hash)
            )
            conn.commit()
            
            # Update cache
            self.status_cache[file_hash] = "failed"
            
            return False  # Stop retrying
    
    def process_file(self, file_info):
        """Process a single file download."""
        # Override in subclasses
        raise NotImplementedError("Subclasses must implement process_file")


class TaskHandlers:
    """Base class for task handlers."""
    
    @staticmethod
    def handle_index(data_source, download_index, context, task_config):
        """Handle index-building task."""
        raise NotImplementedError("Subclasses must implement handle_index")
    
    @staticmethod
    def handle_download(data_source, download_index, context, task_config):
        """Handle download task."""
        raise NotImplementedError("Subclasses must implement handle_download")
    
    @staticmethod
    def handle_validate(data_source, download_index, context, task_config):
        """Handle validation task."""
        raise NotImplementedError("Subclasses must implement handle_validate")


def setup_progress_reporting(download_index, job_queue, stop_event, interval=10):
    """
    Set up a thread for progress reporting.
    
    Args:
        download_index: Download index
        job_queue: Job queue
        stop_event: Stop event
        interval: Reporting interval in seconds
        
    Returns:
        Thread: Progress reporting thread
    """
    def progress_reporter():
        """Thread function for progress reporting."""
        last_stats = {}
        
        while not stop_event.is_set() or not job_queue.empty():
            try:
                # Get current stats
                stats = download_index.get_stats()
                queue_size = job_queue.qsize()
                
                # Calculate rates
                now = time.time()
                
                # Print progress
                logger.info(f"Progress: {stats.get('files_success', 0)} successful, "
                          f"{stats.get('files_batched', 0)} batched, "
                          f"{stats.get('files_transferred', 0)} transferred, "
                          f"{stats.get('files_failed', 0)} failed, "
                          f"{queue_size} in queue")
                
                # Record stats for next time
                last_stats = dict(stats)
                last_stats['time'] = now
                
                # Save index periodically
                download_index.save()
                
                # Sleep until next report
                for _ in range(interval):
                    if stop_event.is_set() and job_queue.empty():
                        break
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in progress reporter: {str(e)}")
                time.sleep(interval)
    
    # Start progress reporting thread
    progress_thread = threading.Thread(target=progress_reporter)
    progress_thread.daemon = True
    progress_thread.start()
    
    return progress_thread


def queue_files_for_download(download_index, job_queue, batch_size=500, max_queue_size=1000):
    """
    Queue files for download.
    
    Args:
        download_index: Download index
        job_queue: Job queue
        batch_size: Size of each batch to query
        max_queue_size: Maximum queue size
        
    Returns:
        int: Total number of files queued
    """
    total_queued = 0
    offset = 0
    
    while True:
        # Get a batch of files to download
        conn = download_index._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT file_hash, source_url, relative_path, destination_blob 
            FROM files 
            WHERE status IN ('pending', 'failed') 
            ORDER BY RANDOM()
            LIMIT ? OFFSET ?
            """,
            (batch_size, offset)
        )
        files = cursor.fetchall()
        
        # Break if no more files
        if not files:
            break
        
        # Queue files for download
        for file_hash, source_url, relative_path, destination_blob in files:
            # Wait if queue is full
            while job_queue.qsize() >= max_queue_size:
                time.sleep(1)
                
            file_info = {
                "file_hash": file_hash,
                "source_url": source_url,
                "relative_path": relative_path,
                "destination_blob": destination_blob
            }
            job_queue.put(file_info)
            total_queued += 1
        
        # Move to next batch
        offset += len(files)
        
        # Log progress
        logger.debug(f"Queued {total_queued} files")
    
    return total_queued
"""
Workflow management for downloading geodata.

This module provides functions to execute download workflows
for different types of geodata, handling the instantiation of 
appropriate data sources based on configuration settings.
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
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import hashlib
from datetime import datetime

from google.cloud import storage
from gnt.data.common.gcs.client import GCSClient
from gnt.data.download.sources.factory import create_data_source
from gnt.data.common.index.download_index import DataDownloadIndex

# Configure logging
logger = logging.getLogger(__name__)


class WorkflowContext:
    """Shared context for workflow execution to avoid redundant resource creation."""
    
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self._storage_client = None
        self._gcs_client = None
        
    @property
    def storage_client(self):
        if not self._storage_client:
            self._storage_client = storage.Client()
        return self._storage_client
        
    @property
    def gcs_client(self):
        if not self._gcs_client:
            self._gcs_client = GCSClient(self.bucket_name, client=self.storage_client)
        return self._gcs_client


class DownloadWorker:
    """Worker class for downloading files."""
    
    def __init__(self, worker_id, job_queue, data_source, tmp_dir, 
                workflow_context, stop_event, download_index, rate_limiter=None, 
                sleep_between_downloads=1):
        self.worker_id = worker_id
        self.job_queue = job_queue
        self.data_source = data_source
        self.tmp_dir = tmp_dir
        self.workflow_context = workflow_context
        self.stop_event = stop_event
        self.download_index = download_index
        self.rate_limiter = rate_limiter
        self.sleep_between_downloads = sleep_between_downloads
        
        # Session management
        self.session = None
        self.last_session_refresh = 0
        self.session_refresh_interval = 1800  # 30 minutes
        self._init_session()
        
        # Status cache to reduce DB access
        self.status_cache = {}
    
    def _init_session(self):
        """Initialize or refresh the worker's session."""
        if hasattr(self.data_source, "get_authenticated_session"):
            try:
                logger.info(f"Worker {self.worker_id}: Creating authenticated session")
                self.session = self.data_source.get_authenticated_session()
                self.last_session_refresh = time.time()
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: Failed to create session: {e}")
    
    def _manage_session(self):
        """Check and refresh session if needed."""
        if (self.session is not None and 
            time.time() - self.last_session_refresh > self.session_refresh_interval):
            self._init_session()
            return True
        return False
    
    def _should_skip_file(self, file_hash):
        """Check if file should be skipped."""
        status = self.status_cache.get(file_hash)
        if not status:
            status, _ = self.download_index.get_file_status(file_hash)
            self.status_cache[file_hash] = status
        
        return status == "success"
    
    def _handle_download_error(self, e, attempt, max_retries, file_info):
        """Handle download errors and determine if retry is appropriate."""
        if attempt >= max_retries:
            logger.error(f"Worker {self.worker_id}: Failed to download {file_info['relative_path']}: {e}")
            self.download_index.record_download_status(
                file_info["file_hash"], file_info["source_url"], 
                file_info["destination_blob"], "failed", str(e))
            self.status_cache[file_info["file_hash"]] = "failed"
            return False
        
        # Calculate retry delay with jitter
        base_retry_delay = 15
        retry_delay = base_retry_delay * (1.5 ** (attempt - 1))
        jitter = retry_delay * 0.3 * random.random()
        delay = retry_delay + jitter
        
        logger.warning(f"Worker {self.worker_id}: Attempt {attempt} failed: {e}")
        logger.info(f"Worker {self.worker_id}: Retrying in {delay:.1f}s")
        
        # Get new session at third attempt
        if attempt == 3:
            self._init_session()
        
        # Wait before retry
        time.sleep(delay)
        return True
    
    def process_file(self, file_info):
        """Process a single file download."""
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
                    
                    # Download and upload to GCS
                    self.data_source.download(source_url, local_path, session=self.session)
                    self.workflow_context.gcs_client.upload_file(local_path, destination_blob)
                    
                    # Clean up temporary file
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    
                    logger.info(f"Worker {self.worker_id}: Successfully downloaded {relative_path}")
                    self.download_index.record_download_status(
                        file_hash, source_url, destination_blob, "success")
                    
                    # Update cache
                    self.status_cache[file_hash] = "success"
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
    
    def run(self):
        """Main worker loop."""
        logger.info(f"Worker {self.worker_id} starting")
        
        while not self.stop_event.is_set() or not self.job_queue.empty():
            try:
                # Get job with timeout to check stop_event periodically
                try:
                    file_info = self.job_queue.get(timeout=2)
                except queue.Empty:
                    continue
                
                # Process the file
                try:
                    self.process_file(file_info)
                finally:
                    self.job_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: Unexpected error: {e}", exc_info=True)
        
        logger.info(f"Worker {self.worker_id} finished")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a JSON or YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    if config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    elif config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def load_config_with_env_vars(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a JSON or YAML file and expand environment variables.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration with expanded environment variables
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    # Load raw config
    if config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    # Expand environment variables
    return expand_env_vars(config)


def expand_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively expand environment variables in string values within a config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with environment variables expanded
    """
    if isinstance(config, dict):
        return {k: expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(i) for i in config]
    elif isinstance(config, str):
        return os.path.expandvars(config)
    else:
        return config


def setup_progress_reporting(download_index, job_queue, stop_event):
    """
    Set up a progress reporting thread.
    
    Args:
        download_index: Download index for statistics
        job_queue: Job queue to monitor
        stop_event: Event to signal thread termination
    
    Returns:
        Thread object for progress reporting
    """
    def report_progress():
        last_report = time.time()
        while not stop_event.is_set() or job_queue.qsize() > 0:
            time.sleep(5)
            if time.time() - last_report >= 30:  # Report every 30 seconds
                current_stats = download_index.get_stats()
                pending = job_queue.qsize()
                success = current_stats.get("successful_downloads", 0)
                failed = current_stats.get("failed_downloads", 0)
                logger.info(f"Progress: {success} successful, {failed} failed, ~{pending} queued")
                last_report = time.time()
    
    progress_thread = threading.Thread(target=report_progress)
    progress_thread.daemon = True
    progress_thread.start()
    return progress_thread


def validate_downloads(data_source, gcs_client, download_index):
    """
    Validate previously downloaded files.
    
    Args:
        data_source: Data source object
        gcs_client: GCS client
        download_index: Download index
    """
    logger.info(f"Validating downloads for {data_source.DATA_SOURCE_NAME}")
    
    # Get successfully downloaded files
    success_files = download_index.get_files_by_status("success")
    total = len(success_files)
    
    logger.info(f"Found {total} files to validate")
    
    # Sample a subset for validation if too many
    if total > 1000:
        sample_size = min(1000, int(total * 0.1))
        files_to_validate = random.sample(success_files, sample_size)
        logger.info(f"Validating a sample of {sample_size} files")
    else:
        files_to_validate = success_files
    
    # Check if files exist in GCS
    valid_count = 0
    invalid_count = 0
    
    for i, file_info in enumerate(files_to_validate):
        destination_blob = file_info["destination_blob"]
        
        # Check if file exists and has appropriate size
        if gcs_client.blob_exists(destination_blob):
            blob_size = gcs_client.get_blob_size(destination_blob)
            if blob_size > 0:
                valid_count += 1
            else:
                invalid_count += 1
                logger.warning(f"Zero-sized file found: {destination_blob}")
                download_index.record_download_status(
                    file_info["file_hash"], file_info["source_url"], 
                    destination_blob, "failed", "Zero-sized file")
        else:
            invalid_count += 1
            logger.warning(f"File missing from GCS: {destination_blob}")
            download_index.record_download_status(
                file_info["file_hash"], file_info["source_url"], 
                destination_blob, "failed", "File missing from GCS")
        
        # Log progress periodically
        if (i+1) % 100 == 0:
            logger.info(f"Validated {i+1}/{len(files_to_validate)} files")
    
    # Report validation results
    logger.info(f"Validation complete: {valid_count} valid, {invalid_count} invalid")
    if invalid_count > 0:
        percentage = invalid_count / len(files_to_validate) * 100
        logger.warning(f"Invalid files percentage: {percentage:.1f}%")


def download_with_threads(data_source, context, max_concurrent, max_queue_size, download_index, task_config):
    """
    Thread-based download implementation using worker threads and a queue.
    
    Args:
        data_source: Data source object
        context: Workflow context with shared resources
        max_concurrent: Maximum concurrent downloads
        max_queue_size: Maximum queue size
        download_index: Download index for tracking progress
        task_config: Configuration parameters
    """
    logger.info(f"Using thread-based download for {data_source.DATA_SOURCE_NAME}")
    
    # Create bounded job queue
    job_queue = queue.Queue(maxsize=max_queue_size)
    
    # Create stop event for workers
    stop_event = threading.Event()
    
    # Limit concurrent downloads to a reasonable number
    worker_count = min(max_concurrent, 20)
    
    # Create rate limiter
    rate_limiter = threading.BoundedSemaphore(value=worker_count)
    
    # Use temporary directory for downloads
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info(f"Using temporary directory: {tmp_dir}")
        
        # Create and start workers
        workers = []
        for i in range(worker_count):
            worker = DownloadWorker(
                worker_id=i+1,
                job_queue=job_queue,
                data_source=data_source,
                tmp_dir=tmp_dir,
                workflow_context=context,
                stop_event=stop_event,
                download_index=download_index,
                rate_limiter=rate_limiter
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
            batch_size = 500
            total_queued = queue_files_for_download(download_index, job_queue, batch_size, max_queue_size)
            
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
            except KeyboardInterrupt:
                logger.warning("Download interrupted by user")
            except Exception as e:
                logger.error(f"Error during download: {str(e)}")
            
            # Print final statistics
            stats = download_index.get_stats()
            logger.info(f"Download complete: {stats.get('successful_downloads', 0)} successful, "
                      f"{stats.get('failed_downloads', 0)} failed")
            
        finally:
            # Signal workers to stop
            logger.info("Signaling workers to stop")
            stop_event.set()
            
            # Wait for workers to finish (briefly)
            for worker_thread in workers:
                worker_thread.join(timeout=2.0)


def queue_files_for_download(download_index, job_queue, batch_size, max_queue_size):
    """
    Queue files from the download index to the job queue.
    
    Args:
        download_index: Download index containing files to download
        job_queue: Queue to add files to
        batch_size: Size of batches to process at once
        max_queue_size: Maximum size of the queue
    
    Returns:
        Total number of files queued
    """
    total_queued = 0
    
    # Queue files in batches
    for batch in range(0, 1_000_000, batch_size):  # Arbitrary large range with batching
        files_in_batch = 0
        
        # Get a batch of files to download
        for file_info in download_index.iter_pending_downloads(batch_size=batch_size):
            try:
                # Try to add to queue with timeout
                job_queue.put(file_info, timeout=5)
                total_queued += 1
                files_in_batch += 1
                
                # Log progress periodically
                if total_queued % 50 == 0:
                    logger.info(f"Queued {total_queued} files for download")
                    
                # Avoid overwhelming the queue
                if job_queue.qsize() >= max_queue_size * 0.9:
                    logger.debug("Queue nearly full, pausing to let workers catch up")
                    time.sleep(5)
                    
            except queue.Full:
                logger.warning("Queue is full, waiting before retrying")
                time.sleep(2)
        
        # If no files were found in this batch, we're done
        if files_in_batch == 0:
            break
    
    return total_queued


def process_task(task_config: Dict[str, Any], context=None) -> None:
    """
    Process a single download task.
    
    Args:
        task_config: Configuration for the task
        context: Optional existing workflow context to reuse
    """
    # Add memory management parameters if specified
    memory_limit = task_config.pop("memory_limit", None)
    if memory_limit:
        try:
            import resource
            # Convert to bytes (assuming memory_limit is in MB)
            resource.setrlimit(resource.RLIMIT_AS, 
                              (memory_limit * 1024 * 1024, 
                               memory_limit * 1024 * 1024))
            logger.info(f"Memory limit set to {memory_limit}MB")
        except (ImportError, ValueError, resource.error) as e:
            logger.warning(f"Could not set memory limit: {str(e)}")
    
    try:
        # Get data source name
        data_source_name = task_config.pop("data_source")
        
        # Get bucket name
        bucket_name = task_config.pop("bucket_name", "growthandheat")
        
        # Get mode
        mode = task_config.pop("mode", "download")  # download, index, validate
        
        # Create data source instance using the factory
        data_source = create_data_source(data_source_name, task_config)
        
        # Log task start
        logger.info(f"Starting {data_source_name} task with mode '{mode}'")
        
        # Create or reuse workflow context
        if context is None:
            context = WorkflowContext(bucket_name)
        
        # Execute the task 
        start_time = time.time()
        
        # Use download index to track progress
        with DataDownloadIndex(
            bucket_name=bucket_name,
            data_source=data_source,
            client=context.storage_client,
        ) as download_index:
            
            # Select and execute appropriate handler based on mode
            if mode == "index":
                TaskHandlers.handle_index(data_source, download_index, context, task_config)
            elif mode == "download":
                TaskHandlers.handle_download(data_source, download_index, context, task_config)
            elif mode == "validate":
                TaskHandlers.handle_validate(data_source, download_index, context, task_config)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
        
        duration = time.time() - start_time
        logger.info(f"Completed {data_source_name} task in {duration:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Error processing task with {task_config.get('data_source', 'unknown')}: {str(e)}")
        raise


class TaskHandlers:
    """Handlers for different task modes."""
    
    @staticmethod
    def handle_index(data_source, download_index, context, task_config):
        """Handle index-building task."""
        logger.info(f"Building index for {data_source.DATA_SOURCE_NAME}")
        download_index.refresh_index(data_source)
        logger.info("Index built successfully")
    
    @staticmethod
    def handle_download(data_source, download_index, context, task_config):
        """Handle download task."""
        max_concurrent = task_config.get("max_concurrent_downloads", 10)
        max_queue_size = task_config.get("max_queue_size", 1000)
        
        # Use thread-based download approach
        download_with_threads(
            data_source=data_source,
            context=context,
            max_concurrent=max_concurrent,
            max_queue_size=max_queue_size,
            download_index=download_index,
            task_config=task_config
        )
    
    @staticmethod
    def handle_validate(data_source, download_index, context, task_config):
        """
        Handle validation task with simplified workflow.
        Delegates validation logic to index methods.
        """
        logger.info(f"Starting validation for {data_source.DATA_SOURCE_NAME}")
        
        # Force refresh option from task config
        force_refresh = task_config.get("force_refresh_gcs", False)
        
        # Delegate to index's validation method which uses _get_existing_files()
        stats = download_index.validate_against_gcs(context.gcs_client, force_file_list_update=force_refresh)
        
        # Summarize results
        logger.info(f"Validation complete:")
        logger.info(f"  - Updated {stats['updated']} file statuses to 'success'") 
        logger.info(f"  - Added {stats['added']} new files to the index")
        logger.info(f"  - Found {stats['orphaned']} orphaned entries (reset to 'indexed' status)")
        
        # Save index
        download_index.save()
        
        # Show final statistics
        index_stats = download_index.get_stats()
        logger.info(f"Index status: {index_stats.get('files_success', 0)} successful files, "
                   f"{index_stats.get('files_pending', 0)} pending, "
                   f"{index_stats.get('files_failed', 0)} failed")


def run_workflow(config_path: Union[str, Path]) -> None:
    """
    Run the complete download workflow defined in the configuration file.
    
    Args:
        config_path: Path to the workflow configuration file
    """
    # Load configuration with environment variables expanded
    config = load_config_with_env_vars(config_path)
    
    # Get workflow tasks
    tasks = config.get("tasks", [])
    if not tasks:
        logger.warning("No tasks defined in the workflow configuration")
        return
    
    # Create a shared workflow context
    shared_bucket = next((task.get("bucket_name", "growthandheat") for task in tasks), "growthandheat")
    context = WorkflowContext(shared_bucket)
    
    # Process each task in order
    logger.info(f"Starting workflow with {len(tasks)} tasks")
    
    for i, task in enumerate(tasks):
        logger.info(f"Task {i+1}/{len(tasks)}: {task.get('data_source', 'unknown')}")
        process_task(task, context)
    
    logger.info("Workflow completed successfully")
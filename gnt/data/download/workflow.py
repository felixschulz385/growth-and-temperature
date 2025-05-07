# workflow.py
import os
import queue
import threading
import time
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage
import random

from gnt.data.common.gcs.client import GCSClient
from gnt.data.download.sources.base import BaseDataSource
from gnt.data.common.index.download_index import DataDownloadIndex

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DownloadWorker:
    """Worker class for downloading files."""
    
    def __init__(self, worker_id, job_queue, data_source, tmp_dir, 
                gcs_client, stop_event, download_index, rate_limiter=None):
        self.worker_id = worker_id
        self.job_queue = job_queue
        self.data_source = data_source
        self.tmp_dir = tmp_dir
        self.gcs = gcs_client
        self.stop_event = stop_event
        self.download_index = download_index
        self.rate_limiter = rate_limiter
        
        # Worker-specific session
        self.session = self._create_session()
        self.last_session_refresh = time.time()
        self.session_refresh_interval = 1800  # 30 minutes
        
        # Status cache to reduce DB access
        self.status_cache = {}
    
    def _create_session(self):
        """Create an authenticated session for this worker if supported."""
        session = None
        if hasattr(self.data_source, "get_authenticated_session"):
            try:
                logger.info(f"Worker {self.worker_id}: Creating authenticated session")
                session = self.data_source.get_authenticated_session()
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: Failed to create session: {e}")
        return session
    
    def _refresh_session(self):
        """Refresh the worker's session."""
        if hasattr(self.data_source, "get_authenticated_session"):
            try:
                logger.info(f"Worker {self.worker_id}: Refreshing session")
                self.session = self.data_source.get_authenticated_session()
                self.last_session_refresh = time.time()
                return True
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: Failed to refresh session: {e}")
        return False
    
    def _check_session_refresh(self):
        """Check if session needs periodic refresh."""
        if (self.session is not None and 
            time.time() - self.last_session_refresh > self.session_refresh_interval):
            return self._refresh_session()
        return False
    
    def process_file(self, file_info):
        """Process a single file download."""
        file_hash = file_info["file_hash"]
        source_url = file_info["source_url"]
        relative_path = file_info["relative_path"]
        destination_blob = file_info["destination_blob"]
        
        # Check if already processed
        status = self.status_cache.get(file_hash)
        if not status:
            status, _ = self.download_index.get_file_status(file_hash)
            self.status_cache[file_hash] = status
        
        if status == "success":
            logger.debug(f"Worker {self.worker_id}: File already downloaded: {relative_path}")
            return True
        
        # Download parameters
        max_retries = 5
        base_retry_delay = 15  # seconds
        attempt = 0
        success = False
        
        # Check for session refresh
        self._check_session_refresh()
        
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
                    self.gcs.upload_file(local_path, destination_blob)
                    
                    # Clean up and mark as success
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
                        # Add small random delay between requests
                        time.sleep(1 + random.random())
                        self.rate_limiter.release()
            
            except Exception as e:
                if attempt >= max_retries:
                    logger.error(f"Worker {self.worker_id}: Failed to download {relative_path}: {e}")
                    self.download_index.record_download_status(
                        file_hash, source_url, destination_blob, "failed", str(e))
                    self.status_cache[file_hash] = "failed"
                else:
                    # Calculate retry delay with jitter
                    retry_delay = base_retry_delay * (1.5 ** (attempt - 1))
                    jitter = retry_delay * 0.3 * random.random()
                    delay = retry_delay + jitter
                    
                    logger.warning(f"Worker {self.worker_id}: Attempt {attempt} failed: {e}")
                    logger.info(f"Worker {self.worker_id}: Retrying in {delay:.1f}s")
                    
                    # Check if error indicates auth issue
                    auth_errors = ["Authentication", "Authorization", "auth", "401", "token", "expired"]
                    if any(err.lower() in str(e).lower() for err in auth_errors):
                        self._refresh_session()
                    
                    # Wait before retry
                    time.sleep(delay)
        
        return success
    
    def run(self):
        """Main worker loop."""
        logger.info(f"Worker {self.worker_id} starting")
        
        while not self.stop_event.is_set() or not self.job_queue.empty():
            try:
                # Get job with timeout to check stop_event periodically
                try:
                    file_info = self.job_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Process the file
                try:
                    self.process_file(file_info)
                finally:
                    # Always mark job as done
                    self.job_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: Unexpected error: {e}", exc_info=True)
        
        logger.info(f"Worker {self.worker_id} finished")


def run(data_source: BaseDataSource, bucket_name: str, max_concurrent_downloads: int, 
        max_queue_size: int, auto_index: bool = False, build_index_only: bool = False):
    """
    Simplified download workflow that efficiently manages parallel downloads.
    
    Args:
        data_source: The data source to download from
        bucket_name: GCS bucket name for storing files
        max_concurrent_downloads: Maximum parallel downloads
        max_queue_size: Maximum size of the download queue
        auto_index: Whether to automatically build/refresh the index
        build_index_only: If True, only build the index without downloading
    """
    logger.info(f"Starting download workflow for {data_source.DATA_SOURCE_NAME} to bucket {bucket_name}")
    
    # Create GCS client
    storage_client = storage.Client()
    gcs = GCSClient(bucket_name)
    
    # Use context manager for DataDownloadIndex
    with DataDownloadIndex(
        bucket_name=bucket_name, 
        data_source=data_source, 
        client=storage_client, 
        auto_index=auto_index
    ) as download_index:
        
        # If only building index, exit
        if build_index_only:
            if not auto_index:
                logger.info("Building index only without downloading files")
                download_index.refresh_index(data_source)
            logger.info("Index built successfully")
            return
        
        # Create bounded job queue
        job_queue = queue.Queue(maxsize=max_queue_size)
        
        # Create stop event for workers
        stop_event = threading.Event()
        
        # Limit concurrent downloads to a reasonable number
        worker_count = min(max_concurrent_downloads, 20)
        
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
                    gcs_client=gcs,
                    stop_event=stop_event,
                    download_index=download_index,
                    rate_limiter=rate_limiter
                )
                thread = threading.Thread(target=worker.run)
                thread.daemon = True
                thread.start()
                workers.append(thread)
            
            try:
                # Queue files for download with batch processing
                total_queued = 0
                logger.info("Starting to queue pending downloads")
                start_time = time.time()
                
                # Use in-memory tracking for fast status updates
                stats = {"success": 0, "failed": 0, "queued": 0}
                
                # Create a progress reporting thread
                def report_progress():
                    last_report = time.time()
                    while not stop_event.is_set() or job_queue.qsize() > 0:
                        time.sleep(5)
                        if time.time() - last_report >= 30:  # Report every 30 seconds
                            current_stats = download_index.get_stats()
                            pending = current_stats.get("files_indexed", 0)
                            success = current_stats.get("files_success", 0)
                            failed = current_stats.get("files_failed", 0)
                            logger.info(f"Progress: {success} successful, {failed} failed, ~{pending} remaining")
                            last_report = time.time()
                
                # Start progress reporting thread
                progress_thread = threading.Thread(target=report_progress)
                progress_thread.daemon = True
                progress_thread.start()
                
                # Queue files in batches
                batch_size = 500
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
                
                elapsed = time.time() - start_time
                logger.info(f"Finished queueing {total_queued} files in {elapsed:.1f} seconds")
                
                if total_queued == 0:
                    logger.info("No files need to be downloaded")
                    stop_event.set()
                    return
                
                # Wait for queue to empty with monitoring
                logger.info(f"Waiting for downloads to complete")
                
                try:
                    # Wait for all downloads to complete or timeout
                    timeout = 3600  # 1 hour max wait
                    job_queue.join()
                    logger.info("All downloads have completed successfully")
                except Exception as e:
                    logger.error(f"Error while waiting for downloads: {e}")
                
                # Print final statistics
                stats = download_index.get_stats()
                logger.info(f"Download complete: {stats.get('successful_downloads', 0)} successful, "
                          f"{stats.get('failed_downloads', 0)} failed")
                
            finally:
                # Signal workers to stop
                logger.info("Signaling workers to stop")
                stop_event.set()
                
                # Wait for workers to finish (briefly)
                for i, worker_thread in enumerate(workers):
                    try:
                        worker_thread.join(timeout=10)
                        if worker_thread.is_alive():
                            logger.warning(f"Worker {i+1} didn't exit cleanly within timeout")
                    except Exception as e:
                        logger.warning(f"Error waiting for worker {i+1}: {e}")
# workflow.py
import tempfile
import os
import queue
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage

from gnt.data.common.gcs.client import GCSClient
from gnt.data.download.sources.base import BaseDataSource
from gnt.data.common.index.download_index import DataDownloadIndex

# Configure uniform logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_worker(job_queue, data_source, tmp_dir, gcs, stop_event, download_index, rate_limiter=None):
    """Worker function to download files from the queue"""
    worker_id = threading.get_ident()
    logger.info(f"Worker {worker_id} starting")
    
    # Create a thread-local cache for file statuses to reduce DB access
    status_cache = {}
    
    # Create a private session for this worker
    session = None
    if hasattr(data_source, "get_authenticated_session"):
        try:
            logger.info(f"Worker {worker_id} creating its own authenticated session")
            session = data_source.get_authenticated_session()
        except Exception as e:
            logger.error(f"Worker {worker_id} failed to create session: {e}")
            session = None
    
    # Keep track of session refresh time
    last_session_refresh = time.time()
    session_refresh_interval = 1800  # 30 minutes in seconds
    
    while not stop_event.is_set() or not job_queue.empty():
        try:
            # Get job with timeout to check stop_event periodically
            try:
                file_info = job_queue.get(timeout=1)
            except queue.Empty:
                # Check if we should exit
                if stop_event.is_set() and job_queue.empty():
                    break
                continue
            
            # Unpack the file info
            file_hash = file_info["file_hash"]
            relative_path = file_info["relative_path"]
            source_url = file_info["source_url"]
            destination_blob = file_info["destination_blob"]
            
            # Check if already downloaded (using local cache first)
            if file_hash in status_cache:
                status = status_cache[file_hash]
            else:
                try:
                    status, _ = download_index.get_file_status(file_hash)
                    # Cache status to avoid repeated DB access
                    status_cache[file_hash] = status
                    # Keep cache size reasonable
                    if len(status_cache) > 1000:
                        status_cache.clear()
                except Exception as db_error:
                    logger.warning(f"Database error checking file status: {db_error}. Assuming file needs download.")
                    status = "unknown"
            
            if status == "success":
                logger.debug(f"Skipping already downloaded: {relative_path}")
                job_queue.task_done()
                continue
            
            # Retry parameters
            max_retries = 10
            base_retry_delay = 30  # seconds
            attempt = 0
            success = False
            session_refreshed = False
            
            # Check if the session needs regular refresh (based on time)
            if hasattr(data_source, "get_authenticated_session") and session is not None:
                if time.time() - last_session_refresh > session_refresh_interval:
                    try:
                        logger.info(f"Worker {worker_id} performing periodic session refresh")
                        session = data_source.get_authenticated_session()
                        last_session_refresh = time.time()
                        session_refreshed = True
                    except Exception as e:
                        logger.error(f"Worker {worker_id} error refreshing session: {e}. Continuing with existing session.")
            
            while attempt < max_retries and not success:
                attempt += 1
                try:
                    # Acquire rate limiter before making request
                    if rate_limiter:
                        rate_limiter.acquire()
                    
                    try:
                        # Download the file and upload it to GCS
                        local_path = os.path.join(tmp_dir, os.path.basename(relative_path))
                        
                        # Create directory if needed
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                        
                        logger.info(f"Worker {worker_id} starting download: {relative_path} (attempt {attempt}/{max_retries})")
                        data_source.download(source_url, local_path, session=session)
                        gcs.upload_file(local_path, destination_blob)
                        os.remove(local_path)
                        logger.info(f"Worker {worker_id} completed: {relative_path}")
                        success = True
                        
                        # Record download success
                        download_index.record_download_status(
                            file_hash, source_url, destination_blob, "success")
                    finally:
                        # Always release the rate limiter when done
                        if rate_limiter:
                            # Add random delay between requests
                            delay = 2 + (time.time() % 3)
                            time.sleep(delay)
                            rate_limiter.release()
                
                except Exception as e:
                    if attempt >= max_retries:
                        logger.error(f"Worker {worker_id} failed to process {source_url} after {max_retries} attempts: {e}")
                        # Record download failure
                        download_index.record_download_status(
                            file_hash, source_url, destination_blob, "failed", str(e))
                    else:
                        # Calculate delay with jitter for retries
                        retry_delay = base_retry_delay * (1.5 ** (attempt - 1))
                        jitter = retry_delay * 0.2 * (time.time() % 1)  # 20% random jitter
                        total_delay = retry_delay + jitter
                        
                        logger.warning(f"Worker {worker_id} attempt {attempt}/{max_retries} failed for {relative_path}: {e}. Retrying in {total_delay:.1f}s...")
                        
                        # Determine if we need to refresh the session based on error type
                        auth_errors = ["Authentication", "Authorization", "auth", "Unauthorized", 
                                      "401", "403", "token", "expired", "login",
                                      "ConnectionError", "ConnectTimeout", "Connection refused", "reset by peer"]
                        
                        need_refresh = any(err.lower() in str(e).lower() for err in auth_errors)
                        
                        # Only try to refresh if we have the method and haven't just refreshed
                        if need_refresh and hasattr(data_source, "get_authenticated_session") and not session_refreshed:
                            logger.info(f"Worker {worker_id} authentication/connection issue detected. Refreshing session for {relative_path}...")
                            try:
                                session = data_source.get_authenticated_session()
                                last_session_refresh = time.time()
                                session_refreshed = True
                                # Reduce the delay when refreshing the session
                                total_delay = min(total_delay, 5)
                                logger.info(f"Worker {worker_id} session refreshed successfully, continuing with shortened delay: {total_delay}s")
                            except Exception as refresh_error:
                                logger.error(f"Worker {worker_id} failed to refresh session: {refresh_error}")
                        
                        # Wait before retrying
                        time.sleep(total_delay)
            
            # Mark job as done regardless of success
            job_queue.task_done()
        except Exception as e:
            logger.error(f"Unexpected error in worker {worker_id}: {e}", exc_info=True)
            # Make sure we mark the job as done even in case of error
            try:
                job_queue.task_done()
            except:
                pass
    
    logger.info(f"Worker {worker_id} finished")


def run(data_source: BaseDataSource, bucket_name: str, max_concurrent_downloads: int, max_queue_size: int, 
        auto_index: bool = True, build_index_only: bool = False):
    """
    Main function to run the download workflow with optimized memory usage
    
    Args:
        data_source: The data source to download from
        bucket_name: GCS bucket name for storing downloaded files
        max_concurrent_downloads: Maximum number of concurrent downloads
        max_queue_size: Maximum size of the download queue
        auto_index: Whether to automatically build/refresh the index at startup
        build_index_only: If True, only build the index without downloading files
    """
    logger.info(f"Starting download workflow for {getattr(data_source, 'DATA_SOURCE_NAME', 'unknown')} to bucket {bucket_name}")
    
    # Create GCS client
    storage_client = storage.Client()
    gcs = GCSClient(bucket_name)
    
    # Use context manager for proper cleanup of DataDownloadIndex
    with DataDownloadIndex(
        bucket_name=bucket_name, 
        data_source=data_source, 
        client=storage_client, 
        auto_index=auto_index,
        save_interval_seconds=300
    ) as download_index:
        
        # If we're only building the index, refresh it if not done automatically and exit
        if build_index_only:
            if not auto_index:
                logger.info("Building index only, without downloading files")
                download_index.refresh_index(data_source)
            logger.info("Index built successfully, exiting without downloading files")
            return
            
        # Create job queue
        job_queue = queue.Queue(maxsize=max_queue_size)
        
        # Create event to signal workers to stop
        stop_event = threading.Event()

        # Use safer concurrency settings
        actual_workers = min(max_concurrent_downloads, 10) 
        
        # Create a temporary directory for downloads
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger.info(f"Using temporary directory: {tmp_dir}")
            
            # Create a shared rate limiter semaphore
            rate_limiter = threading.Semaphore(value=actual_workers)
            
            # Start worker threads first - each will create its own session
            logger.info(f"Starting {actual_workers} download workers")
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                workers = []
                for _ in range(actual_workers):
                    worker = executor.submit(
                        download_worker, 
                        job_queue, 
                        data_source, 
                        tmp_dir, 
                        gcs, 
                        stop_event, 
                        download_index, 
                        rate_limiter
                    )
                    workers.append(worker)
                
                try:
                    # Process pending downloads using generator pattern
                    total_queued = 0
                    files_being_processed = 0  # Track files currently being processed
                                    
                    # Process files using the generator
                    logger.info("Starting to queue files for download...")
                    start_time = time.time()
                    queued_timeout = 60  # seconds to wait before giving up on queue progress
                    last_queue_time = time.time()
                    
                    for file_info in download_index.iter_pending_downloads():
                        # Check for timeout if we're not making progress
                        if time.time() - last_queue_time > queued_timeout:
                            logger.warning(f"No queuing progress for {queued_timeout} seconds, possible deadlock. Breaking.")
                            break
                            
                        try:
                            # Try to add file to download queue with timeout
                            job_queue.put(file_info, timeout=5)
                            total_queued += 1
                            files_being_processed += 1
                            last_queue_time = time.time()
                            
                            # Log progress periodically
                            if total_queued % 10 == 0:
                                logger.info(f"Queued {total_queued} files for download")
                            
                            # Control flow to avoid overwhelming the queue
                            if files_being_processed >= max_queue_size // 2:
                                logger.debug(f"Queue getting full ({files_being_processed} files), waiting for processing to catch up")
                                # Wait for some files to finish
                                timeout_start = time.time()
                                while files_being_processed >= max_queue_size // 4:
                                    # Check for timeout
                                    if time.time() - timeout_start > 60:  # 60 second timeout
                                        logger.warning("Timeout waiting for queue to drain, continuing anyway")
                                        break
                                        
                                    # Calculate remaining files
                                    current_stats = download_index.get_stats()
                                    completed = (
                                        current_stats.get("successful_downloads", 0) + 
                                        current_stats.get("failed_downloads", 0)
                                    )
                                    files_being_processed = max(0, total_queued - completed)
                                    logger.debug(f"Files in progress: {files_being_processed}, completed: {completed}/{total_queued}")
                                    
                                    # Brief pause
                                    time.sleep(2)
                                    
                        except queue.Full:
                            logger.warning("Queue is full, waiting before retrying")
                            time.sleep(5)
                    
                    elapsed = time.time() - start_time
                    logger.info(f"Finished queueing {total_queued} files for download in {elapsed:.1f} seconds")
                    
                    if total_queued == 0:
                        logger.info("No files need to be downloaded, all files are already in GCS or completed")
                        # Signal workers to stop since there's nothing to do
                        stop_event.set()
                        return
                    
                    # Wait for all jobs to complete with timeout
                    queue_empty = False
                    wait_timeout = 3600  # 1 hour max wait time
                    logger.info(f"Waiting for download queue to complete (timeout: {wait_timeout}s)")
                    
                    # Wait with timeout
                    wait_start = time.time()
                    while time.time() - wait_start < wait_timeout:
                        # Check if queue is empty
                        if job_queue.empty():
                            # Give a moment for any in-process task_done calls to complete
                            time.sleep(0.5)
                            # Use q.unfinished_tasks to check if all tasks are done
                            if job_queue.unfinished_tasks == 0:
                                logger.info("All download tasks completed successfully.")
                                queue_empty = True
                                break
                                
                        # Not empty yet or tasks still being processed
                        current_stats = download_index.get_stats()
                        completed = (
                            current_stats.get("successful_downloads", 0) + 
                            current_stats.get("failed_downloads", 0)
                        )
                        remaining = max(0, total_queued - completed)
                        in_queue = job_queue.qsize() if hasattr(job_queue, 'qsize') else "unknown"
                        
                        logger.info(f"Still waiting: {remaining}/{total_queued} files remaining, approximately {in_queue} in queue")
                        
                        # Check if all workers are still alive
                        active_workers = sum(1 for w in workers if not w.done())
                        if active_workers == 0:
                            logger.warning("All workers have exited. Stopping wait.")
                            break
                        
                        # Sleep before checking again
                        time.sleep(30)
                    
                    if not queue_empty:
                        logger.warning(f"Timed out after {wait_timeout}s waiting for downloads to complete")
                    
                finally:
                    # Signal workers to stop regardless of success/failure
                    logger.info("Signaling workers to stop")
                    stop_event.set()
                    
                    # Wait for all workers to finish
                    for i, worker in enumerate(workers):
                        try:
                            worker.result(timeout=30)  # Give each worker 30 seconds to stop
                            logger.info(f"Worker {i+1} finished successfully")
                        except Exception as e:
                            logger.warning(f"Worker {i+1} did not exit cleanly: {e}")
                
                # Print final statistics
                stats = download_index.get_stats()
                logger.info(f"Download complete: {stats.get('successful_downloads', 0)} files downloaded, {stats.get('failed_downloads', 0)} failed")
                logger.info(f"Total index entries: {stats.get('total_indexed', 0)}")
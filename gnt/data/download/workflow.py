# workflow.py
from gnt.data.common.gcs.client import GCSClient
from gnt.data.download.download.base import BaseDataSource
from gnt.data.common.index.download_index import DataDownloadIndex
import tempfile
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import logging
import hashlib
from google.cloud import storage
from datetime import datetime

# Configure uniform logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_worker(job_queue, data_source, tmp_dir, gcs, session, stop_event, download_index, rate_limiter=None):
    """Worker function to download files from the queue"""
    worker_id = threading.get_ident()
    logger.info(f"Worker {worker_id} starting")
    
    while not stop_event.is_set() or not job_queue.empty():
        try:
            # Get job with timeout to check stop_event periodically
            file_info = job_queue.get(timeout=1)
            
            # Unpack the file info
            file_hash = file_info["file_hash"]
            relative_path = file_info["relative_path"]
            source_url = file_info["source_url"]
            destination_blob = file_info["destination_blob"]
            
            # Check if already downloaded
            status, _ = download_index.get_file_status(file_hash)
            if status == "success":
                logger.debug(f"Skipping already downloaded: {relative_path}")
                job_queue.task_done()
                continue
            
            # Retry parameters
            max_retries = 10
            base_retry_delay = 30  # seconds
            attempt = 0
            success = False
            
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
                        
                        logger.info(f"Starting download: {relative_path} (attempt {attempt}/{max_retries})")
                        data_source.download(source_url, local_path, session=session)
                        gcs.upload_file(local_path, destination_blob)
                        os.remove(local_path)
                        logger.info(f"Completed: {relative_path}")
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
                        logger.error(f"Failed to process {source_url} after {max_retries} attempts: {e}")
                        # Record download failure
                        download_index.record_download_status(
                            file_hash, source_url, destination_blob, "failed", str(e))
                    else:
                        # Calculate delay with jitter for retries
                        retry_delay = base_retry_delay * (1.5 ** (attempt - 1))
                        jitter = retry_delay * 0.2 * (time.time() % 1)  # 20% random jitter
                        total_delay = retry_delay + jitter
                        
                        logger.warning(f"Attempt {attempt}/{max_retries} failed for {relative_path}: {e}. Retrying in {total_delay:.1f}s...")
                        time.sleep(total_delay)
                        
                        # Recreate session on connection errors
                        if any(err in str(e) for err in ["ConnectionError", "ConnectTimeout", "Connection refused", "reset by peer"]):
                            if hasattr(data_source, "get_authenticated_session"):
                                logger.info(f"Refreshing session for {relative_path}...")
                                session = data_source.get_authenticated_session()
            
            # Mark job as done regardless of success
            job_queue.task_done()
        except queue.Empty:
            continue
    
    logger.info(f"Worker {worker_id} finished")


def run(data_source: BaseDataSource, bucket_name: str, max_concurrent_downloads: int, max_queue_size: int):
    """
    Main function to run the download workflow with optimized memory usage
    """
    logger.info(f"Starting download workflow for {getattr(data_source, 'DATA_SOURCE_NAME', 'unknown')} to bucket {bucket_name}")
    
    # Create GCS client
    storage_client = storage.Client()
    gcs = GCSClient(bucket_name)
    
    #
    data_source_name = getattr(data_source, "DATA_SOURCE_NAME", "unknown")
    
    # Use context manager for proper cleanup of SQLite database
    with DataDownloadIndex(bucket_name, data_source, data_source_name, client=storage_client) as download_index:
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
            
            # Perform login once and reuse the session if required
            session = None
            if hasattr(data_source, "get_authenticated_session"):
                logger.info("Creating authenticated session")
                session = data_source.get_authenticated_session()

            # Start worker threads first
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
                        session, 
                        stop_event, 
                        download_index, 
                        rate_limiter
                    )
                    workers.append(worker)
                
                # Process pending downloads using generator pattern
                total_queued = 0
                files_being_processed = 0  # Track files currently being processed
                                
                # Process files using the generator
                for file_info in download_index.iter_pending_downloads():
                    # Add file to download queue
                    job_queue.put(file_info)
                    total_queued += 1
                    files_being_processed += 1
                    
                    # Log progress periodically
                    if total_queued % 100 == 0:
                        logger.info(f"Queued {total_queued} files for download")
                    
                    # Control flow to avoid overwhelming the queue
                    if files_being_processed >= max_queue_size // 2:
                        # Wait for some files to finish
                        while files_being_processed >= max_queue_size // 4:
                            # Calculate remaining files
                            files_being_processed = total_queued - (
                                download_index.get_stats().get("successful_downloads", 0) + 
                                download_index.get_stats().get("failed_downloads", 0)
                            )
                            # Brief pause
                            time.sleep(0.1)
                
                if total_queued == 0:
                    logger.info("No files need to be downloaded, all files are already in GCS or completed")
                    # Signal workers to stop since there's nothing to do
                    stop_event.set()
                    return
                    
                logger.info(f"Finished queueing {total_queued} files for download")
                
                # Wait for all jobs to complete
                job_queue.join()
                
                # Signal workers to stop
                stop_event.set()
                
                # Wait for all workers to finish
                for worker in workers:
                    worker.result()
                
                # Print final statistics
                stats = download_index.get_stats()
                logger.info(f"Download complete: {stats.get('successful_downloads', 0)} files downloaded, {stats.get('failed_downloads', 0)} failed")
                logger.info(f"Total index entries: {stats.get('total_indexed', 0)}")
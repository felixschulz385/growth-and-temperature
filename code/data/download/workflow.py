# workflow.py
from gcs.client import GCSClient
from download.base import BaseDataSource
import tempfile
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import time

def download_worker(job_queue, data_source, tmp_dir, gcs, session, stop_event):
    while not stop_event.is_set() or not job_queue.empty():
        try:
            # Get the next job from the queue with a timeout
            # This allows checking the stop_event periodically
            relative_path, file_url, destination_blob = job_queue.get(timeout=1)
            
            try:
                # Download the file and upload it to GCS
                local_path = os.path.join(tmp_dir, relative_path)
                print(f">>> Starting download: {relative_path}")
                data_source.download(file_url, local_path, session=session)
                gcs.upload_file(local_path, destination_blob)
                os.remove(local_path)
                print(f">>> Completed: {relative_path}")
            except Exception as e:
                print(f"!!! Failed to process {file_url}: {e} !!!")
            finally:
                # Mark this job as done
                job_queue.task_done()
        except queue.Empty:
            # No jobs available currently, but may be added later
            # unless stop_event is set
            continue

def run(data_source: BaseDataSource, bucket_name: str, max_concurrent_downloads: int, max_queue_size: int):
    # Create GCS client
    gcs = GCSClient(bucket_name)
    # List existing files in the GCS bucket
    existing_files = gcs.list_existing_files()
    
    # Create job queue
    job_queue = queue.Queue(maxsize=max_queue_size)
    
    # Create event to signal workers to stop
    stop_event = threading.Event()

    # Create a temporary directory for downloads
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"*** Using temp dir: {tmp_dir} ***")
        
        # Perform login once and reuse the session if required
        session = None
        if hasattr(data_source, "get_authenticated_session"):
            session = data_source.get_authenticated_session()

        # Start worker threads before discovering files
        print(f"*** Starting {max_concurrent_downloads} download workers ***")
        with ThreadPoolExecutor(max_workers=max_concurrent_downloads) as executor:
            workers = []
            for _ in range(max_concurrent_downloads):
                worker = executor.submit(download_worker, job_queue, data_source, tmp_dir, gcs, session, stop_event)
                workers.append(worker)
            
            # Now start discovering files and adding to queue
            print("*** Starting to discover files ***")
            for relative_path, file_url in data_source.list_remote_files():
                # Resolve the destination blob path
                destination_blob = data_source.gcs_upload_path(data_source.base_url, relative_path)

                # Check if the file already exists in GCS
                if destination_blob in existing_files:
                    print(f"*** Skipping existing: {relative_path} ***")
                    continue
                    
                # Add job to the queue - workers will pick it up immediately
                job_queue.put((relative_path, file_url, destination_blob))
                print(f">>> Queued: {relative_path}")
            
            # Signal that we're done discovering files
            print("*** Finished discovering files, waiting for downloads to complete ***")
            stop_event.set()
            
            # Wait for all jobs to complete
            job_queue.join()
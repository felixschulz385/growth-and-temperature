from download.fetcher import get_all_files, download_file
from gcs.client import GCSClient
from utils import build_destination_path
from config import GCS_PATH_PREFIX
import tempfile
import os

def run(base_urls, bucket_name):
    gcs = GCSClient(bucket_name)
    existing_files = gcs.list_existing_files()

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"*** Using temp dir: {tmp_dir} ***")

        for base_url in base_urls:
            files = get_all_files(base_url)
            
            for file_path, file_url in files:
                # Compute the final destination blob path with optional prefix
                destination_blob = build_destination_path(file_path, base_url, GCS_PATH_PREFIX) if GCS_PATH_PREFIX else file_path
                
                if destination_blob in existing_files:
                    print(f"*** Skipping existing: {file_path} ***")
                    continue

                try:
                    local_path = download_file(file_url, output_dir=tmp_dir)
                    gcs.upload_file(local_path, destination_blob)
                    os.remove(local_path)
                except Exception as e:
                    print(f"!!! Failed to process {file_url}: {e} !!!")
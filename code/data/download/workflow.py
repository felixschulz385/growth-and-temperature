from gcs.client import GCSClient
from download.base import BaseDataSource
import tempfile
import os

def run(data_source: BaseDataSource, bucket_name: str):
    # Create GCS client
    gcs = GCSClient(bucket_name)
    # List existing files in the GCS bucket
    existing_files = gcs.list_existing_files()

    # Create a temporary directory for downloads
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"*** Using temp dir: {tmp_dir} ***")

        # Get the list of files to download
        files = data_source.list_remote_files()
        
        # Perform login once and reuse the session if required
        session = None
        if hasattr(data_source, "get_authenticated_session"):
            session = data_source.get_authenticated_session()

        for relative_path, file_url in files:
            # Resolve the destination blob path
            destination_blob = data_source.gcs_upload_path(data_source.base_url, relative_path)

            # Check if the file already exists in GCS
            if destination_blob in existing_files:
                print(f"*** Skipping existing: {relative_path} ***")
                continue

            try:
                # Download the file and upload it to GCS
                local_path = os.path.join(tmp_dir, relative_path)
                data_source.download(file_url, local_path, session=session)
                gcs.upload_file(local_path, destination_blob)
                os.remove(local_path)
            except Exception as e:
                print(f"!!! Failed to process {file_url}: {e} !!!")

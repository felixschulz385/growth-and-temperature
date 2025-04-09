from google.cloud import storage
from google.oauth2 import service_account
from config import GCP_PROJECT_ID

class GCSClient:
    def __init__(self, bucket_name):
        self.client = storage.Client(project=GCP_PROJECT_ID, credentials=service_account.Credentials.from_service_account_file("/Users/felixschulz/Downloads/ee-growthandheat-e6c4eefc2bf3.json"))
        self.bucket = self.client.bucket(bucket_name)

    def list_existing_files(self, prefix=""):
        return {blob.name for blob in self.bucket.list_blobs(prefix=prefix)}

    def upload_file(self, local_path, destination_blob_name):
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_path)
        print(f"*** Uploaded {destination_blob_name} ***")

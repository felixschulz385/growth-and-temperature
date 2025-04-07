from google.cloud import storage
from google.oauth2 import service_account
from config import SERVICE_ACCOUNT_FILE, GCP_PROJECT_ID

class GCSClient:
    def __init__(self, bucket_name):
        credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
        self.client = storage.Client(credentials=credentials, project=GCP_PROJECT_ID)
        self.bucket = self.client.bucket(bucket_name)

    def list_existing_files(self, prefix=""):
        return {blob.name for blob in self.bucket.list_blobs(prefix=prefix)}

    def upload_file(self, local_path, destination_blob_name):
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_path)
        print(f"*** Uploaded {destination_blob_name} ***")

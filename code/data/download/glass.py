import os
from google.cloud import storage

## Initialize the Google Cloud Storage client
# Replace 'your-key.json' with the path to your JSON key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Q:/HEALECON/Felix/Research/Heat Islands/misc/ee-growthandheat-570de7e75ef6.json"
storage_client = storage.Client("ee-growthandheat")

# Check which files already exist in the bucket
bucket_name = "ee-growthandheat"
blobs = storage_client.list_blobs(bucket_name, prefix="glass/LST/AVHRR/0.05D/")

if blobs.num_results == 0:
    existing_files = []
else:
    existing_files = [blob.name for blob in blobs]
    
pass

#gcloud iam service-accounts create glass-download --project="ee-growthandheat"
#gcloud storage buckets add-iam-policy-binding gs://growthandheat --member="serviceAccount:glass-download@ee-growthandheat.iam.gserviceaccount.com" --role="roles/storage.objectViewer"
#gcloud storage buckets add-iam-policy-binding gs://growthandheat --member="serviceAccount:glass-download@ee-growthandheat.iam.gserviceaccount.com" --role="roles/storage.objectUser"



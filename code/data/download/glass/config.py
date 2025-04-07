import os

# Service account key path (absolute or relative)
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "Q:/HEALECON/Felix/Research/Heat Islands/misc/ee-growthandheat-e6c4eefc2bf3.json")

# Google Cloud project ID
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "ee-growthandheat")

# GCS bucket name
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "growthandheat")

# Optional path prefix (e.g., "raw-data/") for uploaded files
GCS_PATH_PREFIX = os.getenv("GCS_PATH_PREFIX", "glass").rstrip("/")  # ensures no trailing slash

# Base URLs to crawl for files
BASE_URLS = [
    "https://glass.hku.hk/archive/LST/AVHRR/0.05D/",
    "https://glass.hku.hk/archive/LST/MODIS/Daily/1KM/"
]
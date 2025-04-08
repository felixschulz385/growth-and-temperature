import os
from download.glass import GLASSDataSource

# Google Cloud project ID
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "ee-growthandheat")

# GCS bucket name
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "growthandheat")

# Data source class name
DATA_SOURCE_NAME = os.getenv("DATA_SOURCE_NAME", "glass")

# Mapping of data source names to classes
DATA_SOURCES = {
    "glass": GLASSDataSource,
}

# Base URLs to crawl for files
BASE_URLS = os.getenv("BASE_URL", "https://glass.hku.hk/archive/LST/AVHRR/0.05D/") #"https://glass.hku.hk/archive/LST/MODIS/Daily/1KM/"
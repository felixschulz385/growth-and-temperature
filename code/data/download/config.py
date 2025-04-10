import os
from download.glass import GLASSDataSource
from download.eog import EOGDataSource

# Google Cloud project ID
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")

# GCS bucket name
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# Data source class name
DATA_SOURCE_NAME = os.getenv("DATA_SOURCE_NAME")

# Mapping of data source names to classes
DATA_SOURCES = {
    "glass": GLASSDataSource,
    "eog": EOGDataSource,
}

FILE_EXTENSIONS = os.getenv("FILE_EXTENSIONS").split(",")

# Base URL to crawl for files
BASE_URL = os.getenv("BASE_URL") 
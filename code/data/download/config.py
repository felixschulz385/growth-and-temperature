import os
from download.glass import GLASSDataSource
from download.eog import EOGDataSource

# Allow local debugging by loading .env
if not os.getenv("KUBERNETES_SERVICE_HOST"):  # Only load .env if not in a K8s environment
    from dotenv import load_dotenv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)

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
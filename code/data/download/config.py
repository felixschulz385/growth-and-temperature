import os
from download.glass import GLASSDataSource
from download.eog import EOGDataSource

# Google Cloud project ID
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "ee-growthandheat")

# GCS bucket name
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "growthandheat")

# Data source class name
DATA_SOURCE_NAME = os.getenv("DATA_SOURCE_NAME", "eog")

# Mapping of data source names to classes
DATA_SOURCES = {
    "glass": GLASSDataSource,
    "eog": EOGDataSource,
}

FILE_EXTENSIONS = os.getenv("FILE_EXTENSIONS", ".v4b.global.stable_lights.avg_vis.tif").split(",")
# ".v4b.global.stable_lights.avg_vis.tif"

# Base URL to crawl for files
BASE_URL = os.getenv("BASE_URL", "https://eogdata.mines.edu/wwwdata/dmsp/v4composites_rearrange/") 
#"https://glass.hku.hk/archive/LST/MODIS/Daily/1KM/"
#"https://eogdata.mines.edu/wwwdata/viirs_products/dvnl/"
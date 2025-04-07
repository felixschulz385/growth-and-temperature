import os
from urllib.parse import urlparse

def build_destination_path(file_path, base_url, gcs_prefix):
    """
    Builds a destination path like: 'glass-MODIS-filename'
    """
    parsed = urlparse(base_url)
    parts = parsed.path.strip("/").split("/")

    # Example: ['archive', 'LST', 'MODIS', 'Daily', '1KM']
    datatype = "/".join(parts[1:]) if len(parts) > 2 else "unknown"

    filename = os.path.basename(file_path)
    return f"{gcs_prefix}/{datatype}/{filename}"

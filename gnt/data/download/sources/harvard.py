# download/harvard.py
import requests
import logging
import os
import time
import hashlib
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from gnt.data.download.sources.base import BaseDataSource

logger = logging.getLogger(__name__)

class HarvardDataSource(BaseDataSource):
    def __init__(self, base_url: str = None, file_extensions: list[str] = None, output_path: str = None, doi: str = None):
        self.DATA_SOURCE_NAME = "harvard"
        # Accept either base_url or doi for flexibility
        self.doi = doi or base_url or "doi:10.7910/DVN/YUS575"
        self.base_url = base_url or f"https://dataverse.harvard.edu/dataset.xhtml?persistentId={self.doi}"
        self.file_extensions = file_extensions or [".csv", ".nc", ".tif", ".zip"]
        self.has_entrypoints = False

        # Use output_path if provided, otherwise default to 'harvard/plad'
        self.data_path = output_path or f"{self.DATA_SOURCE_NAME}/plad"

    def list_remote_files(self, entrypoint: dict = None) -> list:
        """
        List all files from Harvard Dataverse. 
        For Harvard, we don't use entrypoints but still include the parameter for compatibility.
        """
        api_url = f"https://dataverse.harvard.edu/api/datasets/:persistentId?persistentId={self.doi}"
        try:
            logger.info(f"Fetching dataset information from {api_url}")
            response = requests.get(api_url)
            response.raise_for_status()
            dataset_info = response.json()
            files = dataset_info['data']['latestVersion']['files']
            result = []
            for file in files:
                label = file["label"]
                if not self.file_extensions or any(label.endswith(ext) for ext in self.file_extensions):
                    relative_path = file['dataFile'].get('originalFileName', label)
                    file_id = file['dataFile']['id']
                    full_url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
                    result.append((relative_path, full_url))
            logger.info(f"Found {len(result)} files in Harvard dataset")
            return result
        except Exception as e:
            logger.error(f"Error listing files from Harvard Dataverse: {str(e)}")
            return []

    def local_path(self, relative_path: str) -> str:
        # Assuming a local directory structure that mirrors the remote one
        return os.path.join("data", self.DATA_SOURCE_NAME, relative_path)

    def download(self, file_url: str, output_path: str, session: requests.Session = None) -> None:
        """Download a file from Harvard Dataverse"""
        try:
            # Use the provided session or create a new one
            s = session or requests.Session()
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
            
            r = s.get(file_url, stream=True)
            r.raise_for_status()

            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Download file in chunks to handle large files
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            logger.debug(f"Successfully downloaded {os.path.basename(output_path)}")
            
        except Exception as e:
            logger.error(f"Error downloading {file_url}: {str(e)}")
            raise

    def gcs_upload_path(self, base_url: str, relative_path: str) -> str:
        """Generate the GCS path for a file"""
        filename = os.path.basename(relative_path)
        return f"{self.data_path}/{filename}"

    def filename_to_entrypoint(self, relative_path: str) -> dict:
        """
        Harvard data doesn't use entrypoints, but we implement this method
        to maintain compatibility with the BaseDataSource API.
        """
        return None
    
    def get_authenticated_session(self) -> requests.Session:
        """
        Harvard API doesn't require authentication for public datasets,
        but we implement this in case we need to add tokens later.
        """
        return requests.Session()
    
    def get_all_entrypoints(self):
        """
        Harvard data doesn't use entrypoints because all files are in a single dataset.
        Return an empty list for compatibility.
        """
        return []

    def get_file_hash(self, file_url: str) -> str:
        """
        Generate a unique hash for a file based on its URL.
        Args:
            file_url: URL of the file
        Returns:
            str: A unique hash identifier for the file
        """
        return hashlib.md5(file_url.encode('utf-8')).hexdigest()
import logging
from typing import Set, Optional, Union, List
from pathlib import Path
import os

from google.cloud import storage
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

class GCSClient:
    """
    Google Cloud Storage client for interacting with cloud storage buckets.
    
    This client provides methods for listing, downloading, and uploading files
    to Google Cloud Storage buckets with appropriate error handling and logging.
    """
    
    def __init__(self, bucket_name: str, project_id: Optional[str] = None, 
                 credentials_path: Optional[str] = None, use_index: bool = True):
        """
        Initialize a Google Cloud Storage client.
        
        Args:
            bucket_name: Name of the GCS bucket to interact with
            project_id: Google Cloud project ID (optional)
            credentials_path: Path to service account credentials JSON file (optional)
            use_index: Whether to use the blob index for improved listing performance
        """
        try:
            if credentials_path:
                # Use explicit credentials if provided
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                self.client = storage.Client(project=project_id, credentials=credentials)
                logger.debug(f"Using explicit credentials from {credentials_path}")
            else:
                # Use default authentication (environment variables or application default)
                self.client = storage.Client(project=project_id)
                logger.debug("Using application default credentials")
                
            self.bucket = self.client.bucket(bucket_name)
            self.bucket_name = bucket_name
            logger.info(f"Connected to bucket {bucket_name}" + 
                        (f" in project {project_id}" if project_id else ""))
            
            # Initialize blob index options
            self.use_index = use_index
            self._indices = {}  # Store indices by prefix
                
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {str(e)}")
            raise

    def _get_or_create_index(self, prefix: str = ""):
        """Get or create a blob index for the given prefix."""
        if not self.use_index:
            return None
            
        # Use an appropriate prefix key to avoid creating too many indices
        prefix_key = prefix#prefix.split('/')[0] if prefix else ""
        
        if prefix_key not in self._indices:
            try:
                from gcs.blob_index import BlobIndex
                # Pass the client directly to avoid circular references
                self._indices[prefix_key] = BlobIndex(
                    self.bucket_name, 
                    prefix=prefix_key,
                    client=self.client
                )
            except Exception as e:
                logger.error(f"Failed to create blob index: {str(e)}")
                return None
        
        return self._indices[prefix_key]

    def list_existing_files(self, prefix: str = "") -> Set[str]:
        """
        List all files in the bucket with the given prefix.
        
        Args:
            prefix: File path prefix to filter results
            
        Returns:
            Set of file paths in the bucket matching the prefix
        """
        try:
            # Try to use the index if enabled
            index = self._get_or_create_index(prefix)
            if index:
                logger.debug(f"Using blob index for listing with prefix '{prefix}'")
                return set(index.get_files_by_prefix(prefix))
            
            # Fall back to direct API calls if index is disabled
            logger.debug(f"Using direct API for listing with prefix '{prefix}'")
            return {blob.name for blob in self.bucket.list_blobs(prefix=prefix)}
        except Exception as e:
            logger.error(f"Failed to list files with prefix '{prefix}': {str(e)}")
            return set()  # Return empty set instead of raising exception
    
    def download_file(self, source_blob_name: str, destination_file_name: str) -> bool:
        """
        Download a file from the bucket to local storage.
        
        Args:
            source_blob_name: Path of the file in the bucket
            destination_file_name: Local path to save the downloaded file
            
        Returns:
            True if download was successful, False otherwise
        """
        try:
            blob = self.bucket.blob(source_blob_name)
            blob.download_to_filename(destination_file_name)
            logger.debug(f"Downloaded {source_blob_name} to {destination_file_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {source_blob_name}: {str(e)}")
            return False

    def upload_file(self, local_path: Union[str, Path], 
                    destination_blob_name: Optional[str] = None) -> bool:
        """
        Upload a local file to the bucket.
        
        Args:
            local_path: Path to the local file
            destination_blob_name: Destination path in the bucket.
                                   If None, uses the basename of the local path
                                   
        Returns:
            True if upload was successful, False otherwise
        """
        try:
            local_path = Path(local_path)
            if not destination_blob_name:
                destination_blob_name = local_path.name
                
            if not local_path.exists():
                logger.error(f"Local file {local_path} does not exist")
                return False
                
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(str(local_path))
            logger.info(f"Uploaded {local_path} to {destination_blob_name}")
            
            # Update the index if enabled
            if self.use_index:
                # Find relevant indices to update
                for prefix_key, index in self._indices.items():
                    if destination_blob_name.startswith(prefix_key):
                        try:
                            index.add_to_index(destination_blob_name)
                        except Exception as e:
                            logger.warning(f"Failed to update index for {destination_blob_name}: {str(e)}")
                        
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {str(e)}")
            return False
            
    def file_exists(self, path: str) -> bool:
        """
        Check if a file exists in the bucket.
        
        Args:
            path: Path to check
            
        Returns:
            True if the file exists, False otherwise
        """
        try:
            blob = self.bucket.blob(path)
            return blob.exists()
        except Exception as e:
            logger.error(f"Error checking if file {path} exists: {str(e)}")
            return False

    def list_blobs_with_limit(self, prefix: str = "", limit: int = 1):
        """
        List blobs in the bucket with the given prefix, up to the specified limit.
        
        Args:
            prefix: File path prefix to filter results
            limit: Maximum number of blobs to retrieve
            
        Yields:
            Blob objects, up to the specified limit
        """
        try:
            # Try to use the index if enabled
            index = self._get_or_create_index(prefix)
            if index:
                logger.debug(f"Using blob index for limited listing with prefix '{prefix}'")
                files = index.get_files_by_prefix(prefix, max_results=limit)
                
                # Convert file names to blob objects
                for file_name in files[:limit]:
                    yield self.bucket.blob(file_name)
                return
            
            # Fall back to direct API calls if index is disabled
            logger.debug(f"Using direct API for limited listing with prefix '{prefix}'")
            blobs = self.bucket.list_blobs(prefix=prefix, max_results=limit)
            yield from blobs
        except Exception as e:
            logger.error(f"Failed to list blobs with prefix '{prefix}': {str(e)}")
            # Just yield nothing instead of raising an exception

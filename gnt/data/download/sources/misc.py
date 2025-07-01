import requests
import logging
import os
import time
import hashlib
import urllib.parse
from typing import Generator, Tuple, List, Dict, Any, Optional
from pathlib import Path

# Add these imports for async download support
import aiofiles
import asyncio

from gnt.data.download.sources.base import BaseDataSource

logger = logging.getLogger(__name__)

class MiscDataSource(BaseDataSource):
    """
    Data source for miscellaneous files from stable download links.
    Allows downloading arbitrary files defined in the configuration YAML.
    """
    
    DATA_SOURCE_NAME = "misc"
    
    def __init__(self, files: List[Dict[str, str]], output_path: str = None, 
                 timeout: int = 60, chunk_size: int = 8192):
        """
        Initialize the miscellaneous data source.
        
        Args:
            files: List of file dictionaries, each containing at minimum:
                  - url: Download URL
                  - name: Filename to use (or will be extracted from URL)
                  Additional optional fields:
                  - description: Human-readable description
                  - subfolder: Optional subfolder within output_path
                  - md5: Optional MD5 hash for validation
            output_path: Custom output path in GCS (defaults to 'misc')
            timeout: Connection timeout in seconds
            chunk_size: Chunk size for streaming downloads
        """
        self.files = files
        self.data_path = output_path or "misc"
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.has_entrypoints = False
        
        # Define schema types for Parquet consistency (same as other sources)
        self.schema_dtypes = {
            'year': 'int32',            # Consistent with other sources
            'day_of_year': 'int32',     # Consistent with other sources
            'timestamp_precision': 'ms', # Use millisecond precision for timestamps
            'file_size': 'int64',       # Consistent int64 for file sizes
            'download_status': 'string', # Consistent string type
            'status_category': 'string'  # Consistent string type
        }
        
        # Validate and normalize file configurations
        self._normalize_file_configs()
        
        logger.info(f"Initialized Misc data source with {len(self.files)} files")
        for file_info in self.files:
            logger.debug(f"File: {file_info['name']} - {file_info['url']}")

    def _normalize_file_configs(self):
        """Validate and normalize file configurations."""
        normalized_files = []
        
        for idx, file_info in enumerate(self.files):
            # Create a new dict to avoid modifying the original
            normalized = {}
            
            # Check required fields
            if 'url' not in file_info:
                logger.warning(f"Skipping file at index {idx}: missing required 'url' field")
                continue
                
            normalized['url'] = file_info['url']
            
            # Extract filename from URL if not provided
            if 'name' not in file_info:
                parsed_url = urllib.parse.urlparse(file_info['url'])
                path = parsed_url.path
                filename = os.path.basename(path)
                if not filename:
                    filename = f"file_{idx}"
                normalized['name'] = filename
            else:
                normalized['name'] = file_info['name']
                
            # Copy optional fields
            normalized['description'] = file_info.get('description', f"File {normalized['name']}")
            normalized['subfolder'] = file_info.get('subfolder', '')
            normalized['md5'] = file_info.get('md5', None)
            
            # Generate a file hash based on URL
            normalized['file_hash'] = hashlib.md5(normalized['url'].encode()).hexdigest()
            
            # Build the destination path
            subfolder = normalized['subfolder']
            if subfolder:
                normalized['destination'] = f"{self.data_path}/{subfolder}/{normalized['name']}"
            else:
                normalized['destination'] = f"{self.data_path}/{normalized['name']}"
            
            normalized_files.append(normalized)
            
        self.files = normalized_files

    def list_remote_files(self, entrypoint: dict = None) -> Generator[Tuple[str, str], None, None]:
        """
        List all files defined in the configuration.
        
        Args:
            entrypoint: Ignored parameter (not used for this source)
            
        Returns:
            Generator yielding tuples of (relative_path, file_url)
        """
        for file_info in self.files:
            # Use subfolder in relative path if provided
            subfolder = file_info.get('subfolder', '')
            if subfolder:
                relative_path = f"{subfolder}/{file_info['name']}"
            else:
                relative_path = file_info['name']
                
            logger.debug(f"Yielding file: {relative_path} - {file_info['url']}")
            yield (relative_path, file_info['url'])

    def download(self, file_url: str, output_path: str, session: requests.Session = None) -> None:
        """
        Download a file from the configured URL.
        
        Args:
            file_url: URL of the file to download
            output_path: Local path to save the file
            session: Optional requests session for authentication (if needed)
        """
        try:
            # Use provided session or create a new one
            session = session or requests.Session()
            
            # Log download start
            filename = os.path.basename(output_path)
            logger.info(f"Downloading {filename} from {file_url}")
            
            # Stream the download
            start_time = time.time()
            
            with session.get(file_url, stream=True, timeout=self.timeout) as response:
                response.raise_for_status()
                
                # Create directory if needed
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Get content length if available
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                # Download in chunks
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Log progress for large files
                            if total_size > 10*1024*1024 and downloaded % (5*1024*1024) < self.chunk_size:  # Log every 5MB
                                progress = (downloaded / total_size) * 100 if total_size > 0 else 0
                                logger.info(f"Downloaded {downloaded/(1024*1024):.1f}MB of {total_size/(1024*1024):.1f}MB ({progress:.1f}%)")
            
            # Validate download if MD5 hash is provided
            for file_info in self.files:
                if file_info['url'] == file_url and file_info.get('md5'):
                    self._validate_file_hash(output_path, file_info['md5'])
            
            elapsed = time.time() - start_time
            file_size_mb = os.path.getsize(output_path) / 1024 / 1024
            logger.info(f"Successfully downloaded {filename} ({file_size_mb:.2f}MB in {elapsed:.1f}s)")
            
        except requests.RequestException as e:
            logger.error(f"Error downloading from {file_url}: {str(e)}")
            # Clean up partial download
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            raise
        except Exception as e:
            logger.error(f"Unexpected error downloading from {file_url}: {str(e)}")
            # Clean up partial download
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            raise

    def _validate_file_hash(self, file_path: str, expected_md5: str) -> bool:
        """
        Validate a downloaded file against its expected MD5 hash.
        
        Args:
            file_path: Path to the downloaded file
            expected_md5: Expected MD5 hash
            
        Returns:
            True if valid, raises exception if invalid
        """
        logger.info(f"Validating MD5 hash for {os.path.basename(file_path)}")
        
        # Calculate MD5 hash
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        calculated_md5 = md5.hexdigest()
        
        # Compare hashes
        if calculated_md5.lower() != expected_md5.lower():
            logger.error(f"MD5 hash validation failed for {file_path}")
            logger.error(f"Expected: {expected_md5}")
            logger.error(f"Got: {calculated_md5}")
            os.remove(file_path)  # Remove invalid file
            raise ValueError(f"MD5 hash validation failed for {file_path}")
            
        logger.info(f"MD5 hash validation successful")
        return True

    def local_path(self, relative_path: str) -> str:
        """
        Generates local path for a file.
        
        Args:
            relative_path: Relative path of the file
            
        Returns:
            Local path for the file
        """
        return os.path.join("data", self.DATA_SOURCE_NAME, relative_path)

    def gcs_upload_path(self, base_url: str, relative_path: str) -> str:
        """
        Generate the GCS path for a file.
        
        Args:
            base_url: Base URL or file URL for this data source
            relative_path: Relative path of the file
            
        Returns:
            GCS path for the file
        """
        # Log inputs for debugging
        logger.debug(f"gcs_upload_path called with base_url={base_url}, relative_path={relative_path}")
        
        # For files with a specified destination, use that
        for file_info in self.files:
            # Match by URL
            if file_info['url'] == base_url:
                logger.debug(f"Matched file by URL: {file_info['destination']}")
                return file_info['destination']
            
            # Match by name
            filename = os.path.basename(relative_path)
            if filename == file_info['name']:
                logger.debug(f"Matched file by name: {file_info['destination']}")
                return file_info['destination']
        
        # Default fallback with more predictable path construction
        clean_path = relative_path.strip('/')
        result = f"{self.data_path}/{clean_path}"
        logger.debug(f"Using default path: {result}")
        return result

    def get_file_hash(self, file_url: str) -> str:
        """
        Generate a unique hash for a file URL.
        
        Args:
            file_url: URL of the file
            
        Returns:
            MD5 hash of the file URL
        """
        return hashlib.md5(file_url.encode()).hexdigest()

    def get_all_entrypoints(self) -> List[Dict[str, Any]]:
        """
        Returns an empty list as this data source doesn't use entrypoints.
        
        Returns:
            An empty list
        """
        return []

    def filename_to_entrypoint(self, relative_path: str) -> Optional[Dict[str, Any]]:
        """
        Not used for this data source.
        
        Returns:
            None
        """
        return None

    @property
    def base_url(self):
        """
        Virtual base_url property for compatibility with the indexing system.
        Returns a unique identifier for this data source.
        
        Returns:
            String identifier
        """
        # Return a fixed string to provide compatibility
        return f"misc-data-source-{self.data_path}"
    
    def get_gcs_prefix(self) -> str:
        """
        Get the GCS prefix for this data source.
        Used during validation to find files.
        
        Returns:
            GCS prefix path
        """
        return self.data_path

    async def download_async(self, file_url: str, output_path: str, session=None) -> None:
        """
        Asynchronous download method for compatibility with async workflow.
        
        Args:
            file_url: URL to download from
            output_path: Local path to save the file
            session: Optional aiohttp session for connection reuse
        """
        import aiohttp
        
        # Use provided session or create a new one
        if session is None:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                await self._download_with_session_async(session, file_url, output_path)
        else:
            await self._download_with_session_async(session, file_url, output_path)
    
    async def _download_with_session_async(self, session, file_url: str, output_path: str):
        """Helper method to download with a given async session."""
        try:
            filename = os.path.basename(output_path)
            logger.info(f"Downloading {filename} from {file_url} (async)")
            
            start_time = time.time()
            
            # Add retry logic for robustness
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with session.get(file_url) as response:
                        response.raise_for_status()
                        
                        # Ensure directory exists
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        # Get content length if available
                        total_size = int(response.headers.get('content-length', 0))
                        downloaded = 0
                        
                        # Write file asynchronously
                        async with aiofiles.open(output_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(self.chunk_size):
                                await f.write(chunk)
                                downloaded += len(chunk)
                                
                                # Log progress for large files
                                if total_size > 10*1024*1024 and downloaded % (5*1024*1024) < self.chunk_size:
                                    progress = (downloaded / total_size) * 100 if total_size > 0 else 0
                                    logger.info(f"Downloaded {downloaded/(1024*1024):.1f}MB of {total_size/(1024*1024):.1f}MB ({progress:.1f}%)")
                        
                        # Validate download if MD5 hash is provided
                        for file_info in self.files:
                            if file_info['url'] == file_url and file_info.get('md5'):
                                self._validate_file_hash(output_path, file_info['md5'])
                        
                        elapsed = time.time() - start_time
                        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
                        logger.info(f"Successfully downloaded {filename} ({file_size_mb:.2f}MB in {elapsed:.1f}s)")
                        return  # Success, exit retry loop
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Progressive backoff: 2s, 4s, 6s
                        logger.warning(f"Download attempt {attempt + 1} failed for {file_url}, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Failed to download {file_url} after {max_retries} attempts: {e}")
                        raise
                        
        except Exception as e:
            logger.error(f"Error downloading {file_url}: {e}")
            # Clean up partial file if it exists
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            raise
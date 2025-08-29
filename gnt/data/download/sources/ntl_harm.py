import re
import json
import logging
import time
import hashlib
import os
import asyncio
import aiohttp
import aiofiles
from datetime import datetime
from typing import Generator, Tuple, List, Dict, Any, Optional
from urllib.parse import urlparse

import requests

from gnt.data.download.sources.base import BaseDataSource

logger = logging.getLogger(__name__)

class NTLHarmDataSource(BaseDataSource):
    """
    Data source for harmonized nighttime lights data from Figshare.
    Downloads DMSP and VIIRS harmonized data from the Figshare repository.
    """
    
    # Figshare API endpoints
    FIGSHARE_API_BASE = "https://api.figshare.com/v2"
    DATASET_ID = "9828827"  # From the URL provided
    
    def __init__(self, base_url: str = None, file_extensions: list[str] = None, output_path: str = None):
        """
        Initialize NTL Harm data source.
        
        Args:
            base_url: Base URL (optional, uses Figshare API)
            file_extensions: List of file extensions to download (default: .tif, .zip)
            output_path: Custom output path (optional)
        """
        self.DATA_SOURCE_NAME = "ntl_harm"
        
        # Use Figshare API URL if base_url not provided
        self.base_url = base_url or f"{self.FIGSHARE_API_BASE}/articles/{self.DATASET_ID}"
        self.file_extensions = file_extensions or [".tif", ".zip", ".tar.gz", ".gz"]
        self.has_entrypoints = True  # We'll use years as entrypoints
        
        # Set data path
        if output_path:
            self.data_path = output_path
        else:
            self.data_path = f"{self.DATA_SOURCE_NAME}/harmonized"
            
        # Define schema types for Parquet consistency
        self.schema_dtypes = {
            'year': 'int32',            # Explicitly use int32 for year
            'day_of_year': 'int32',     # Explicitly use int32 for day_of_year
            'timestamp_precision': 'ms', # Use millisecond precision for timestamps
            'file_size': 'int64',       # Consistent int64 for file sizes
            'download_status': 'string', # Consistent string type
            'status_category': 'string'  # Consistent string type
        }
        
        # Cache for API responses
        self._files_cache = None
        self._cache_timestamp = None
        self._cache_duration = 3600  # Cache for 1 hour
        
        logger.info(f"Initialized NTL Harm data source with path: {self.data_path}")

    def get_selenium_session(self):
        """
        Returns None as this data source doesn't require Selenium.
        """
        return None

    def close_selenium_session(self, session):
        """
        No-op as this data source doesn't use Selenium.
        """
        pass

    def get_file_hash(self, file_url: str) -> str:
        """
        Generate a unique hash for a file based on its URL.
        
        Args:
            file_url: URL of the file
            
        Returns:
            str: A unique hash identifier for the file
        """
        return hashlib.md5(file_url.encode('utf-8')).hexdigest()

    def _get_figshare_files(self) -> List[Dict[str, Any]]:
        """
        Get file information from Figshare API with caching.
        
        Returns:
            List of file dictionaries from the API
        """
        current_time = time.time()
        
        # Return cached data if still valid
        if (self._files_cache is not None and 
            self._cache_timestamp is not None and 
            current_time - self._cache_timestamp < self._cache_duration):
            logger.debug("Using cached Figshare file data")
            return self._files_cache
        
        logger.info(f"Fetching file information from Figshare API: {self.base_url}")
        
        try:
            # Get article details
            response = requests.get(self.base_url)
            response.raise_for_status()
            article_data = response.json()
            
            # Extract files from the article
            files = article_data.get('files', [])
            
            logger.info(f"Found {len(files)} files in Figshare dataset")
            
            # Cache the results
            self._files_cache = files
            self._cache_timestamp = current_time
            
            return files
            
        except requests.RequestException as e:
            logger.error(f"Error fetching data from Figshare API: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response from Figshare API: {e}")
            return []

    def list_remote_files(self, entrypoint: dict = None) -> List[Tuple[str, str]]:
        """
        List files from the Figshare dataset.
        
        Args:
            entrypoint: Optional filter by year
            
        Returns:
            List of (relative_path, file_url) tuples
        """
        files = self._get_figshare_files()
        results = []
        
        for file_info in files:
            filename = file_info.get('name', '')
            download_url = file_info.get('download_url', '')
            
            if not filename or not download_url:
                continue
                
            # Check file extension
            if not any(filename.lower().endswith(ext.lower()) for ext in self.file_extensions):
                continue
            
            # If entrypoint is specified, filter by year
            if entrypoint:
                year_filter = entrypoint.get('year')
                if year_filter:
                    # Extract year from filename
                    file_year = self._extract_year_from_filename(filename)
                    if file_year != year_filter:
                        continue
            
            # Use filename as relative path (flat structure)
            relative_path = filename
            results.append((relative_path, download_url))
        
        logger.info(f"Listed {len(results)} files from Figshare dataset")
        return results

    def _extract_year_from_filename(self, filename: str) -> Optional[int]:
        """
        Extract year from filename using common patterns.
        
        Args:
            filename: Name of the file
            
        Returns:
            Year as integer, or None if not found
        """
        # Look for 4-digit year patterns in filename
        year_patterns = [
            r'(\d{4})',  # Any 4-digit number
            r'_(\d{4})_',  # Year surrounded by underscores
            r'\.(\d{4})\.',  # Year surrounded by dots
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, filename)
            if match:
                year = int(match.group(1))
                # Validate reasonable year range for nighttime lights data
                if 1992 <= year <= 2030:
                    return year
        
        return None

    def local_path(self, relative_path: str) -> str:
        """
        Generate local path for a file.
        
        Args:
            relative_path: Relative path of the file
            
        Returns:
            Local path for the file
        """
        return os.path.join("data", self.DATA_SOURCE_NAME, relative_path)

    def download(self, file_url: str, output_path: str, session: requests.Session = None) -> None:
        """
        Download a file from Figshare.
        
        Args:
            file_url: URL to download from
            output_path: Local path to save the file
            session: Optional requests session for connection reuse
        """
        # Use provided session or create a new one
        s = session or requests.Session()
        
        try:
            logger.info(f"Downloading {os.path.basename(output_path)} from Figshare")
            
            r = s.get(file_url, stream=True)
            r.raise_for_status()

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            logger.info(f"Successfully downloaded {os.path.basename(output_path)}")
            
        except requests.RequestException as e:
            logger.error(f"Error downloading {file_url}: {e}")
            # Clean up partial file if it exists
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            raise

    async def download_async(self, file_url: str, output_path: str, session: aiohttp.ClientSession = None) -> None:
        """
        Asynchronous download method with respectful rate limiting.
        
        Args:
            file_url: URL to download from
            output_path: Local path to save the file
            session: Optional aiohttp session for connection reuse
        """
        # Add a small delay to be respectful to the server
        await asyncio.sleep(0.3)  # 300ms delay between requests
        
        # Use provided session or create a new one
        if session is None:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=600, connect=60)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                await self._download_with_session(session, file_url, output_path)
        else:
            await self._download_with_session(session, file_url, output_path)

    async def _download_with_session(self, session: aiohttp.ClientSession, file_url: str, output_path: str):
        """Helper method to download with a given session."""
        try:
            logger.info(f"Async downloading {os.path.basename(output_path)} from Figshare")
            
            # Add retry logic for robustness
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with session.get(file_url) as response:
                        response.raise_for_status()
                        
                        # Ensure directory exists
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        # Write file asynchronously
                        async with aiofiles.open(output_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)
                        
                        logger.info(f"Successfully downloaded {os.path.basename(output_path)}")
                        return  # Success, exit retry loop
                        
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
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

    def gcs_upload_path(self, base_url: str, relative_path: str) -> str:
        """Generate destination path for the file."""
        filename = os.path.basename(relative_path)
        return f"{self.data_path}/{filename}"

    def filename_to_entrypoint(self, relative_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract entrypoint information from filename.
        
        Args:
            relative_path: Relative path of the file
            
        Returns:
            Dictionary with year information, or None if not extractable
        """
        filename = os.path.basename(relative_path)
        year = self._extract_year_from_filename(filename)
        
        if year is not None:
            return {
                'year': int(year),  # Ensure int type (will be cast to int32 in index)
                'day': int(1)       # Default day for annual data
            }
        
        return None

    def get_all_entrypoints(self) -> List[Dict[str, Any]]:
        """
        Returns a list of year entrypoints based on available files.
        
        Returns:
            List of dictionaries with year information
        """
        logger.info("Finding year entrypoints from NTL Harm dataset")
        
        files = self._get_figshare_files()
        years = set()
        
        for file_info in files:
            filename = file_info.get('name', '')
            year = self._extract_year_from_filename(filename)
            if year is not None:
                years.add(year)
        
        # Convert to entrypoint format
        entrypoints = []
        for year in sorted(years):
            entrypoints.append({
                'year': int(year),  # Will be cast to int32 in index
                'day': int(1)       # Default day for annual data
            })
        
        logger.info(f"Generated {len(entrypoints)} year entrypoints for NTL Harm data: {sorted(years)}")
        return entrypoints

    async def list_remote_files_async(self, entrypoint: dict = None) -> List[Tuple[str, str]]:
        """
        Asynchronous version of list_remote_files.
        
        Args:
            entrypoint: Optional entrypoint to filter results
            
        Returns:
            List of (relative_path, file_url) tuples
        """
        # Add a small delay to be respectful
        await asyncio.sleep(0.1)
        
        # Since Figshare API is relatively fast, we can call the sync version
        # In a real async context, we might want to use aiohttp for the API call
        try:
            files = await asyncio.get_event_loop().run_in_executor(
                None, self.list_remote_files, entrypoint
            )
            return files
        except Exception as e:
            logger.error(f"Error in async file listing: {e}")
            return []

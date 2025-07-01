# download/glass.py
import re
import calendar
from datetime import datetime
import logging
import time
import hashlib
import os
import asyncio
import aiohttp
import aiofiles

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from gnt.data.download.sources.base import BaseDataSource

class GlassLSTDataSource(BaseDataSource):
    def __init__(self, base_url: str, file_extensions: list[str] = None, output_path: str = None):
        self.DATA_SOURCE_NAME = "glass"
        self.base_url = base_url
        self.file_extensions = file_extensions or [".hdf"]
        self.has_entrypoints = True
        
        parsed = urlparse(base_url)
        parts = parsed.path.strip("/").split("/")
        datatype = "/".join(parts[1:]) if len(parts) > 2 else "unknown"
        
        # Use custom output path if provided, otherwise construct from URL
        if output_path:
            self.data_path = output_path
        else:
            self.data_path = f"{self.DATA_SOURCE_NAME}/{datatype}"

        # Don't store the session directly in the instance
        # Just keep a flag to check if we need selenium
        self.requires_selenium = False
        
        # Define schema types for Parquet consistency
        self.schema_dtypes = {
            'year': 'int32',            # Explicitly use int32 for year
            'day_of_year': 'int32',     # Explicitly use int32 for day_of_year
            'timestamp_precision': 'ms', # Use millisecond precision for timestamps
            'file_size': 'int64',       # Consistent int64 for file sizes
            'download_status': 'string', # Consistent string type
            'status_category': 'string'  # Consistent string type
        }

    def get_selenium_session(self):
        """
        Returns a selenium session (webdriver).
        Creates it if it does not exist.
        
        Note: The workflow context will be responsible for maintaining
        the persistent session, this method is just a factory.
        """
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        logger = logging.getLogger(__name__)
        logger.info("Creating new Selenium WebDriver for GLASS data source")
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=chrome_options)
        
        # Set longer page load timeout to handle slow connections
        driver.set_page_load_timeout(120)
        
        return driver

    def close_selenium_session(self, session):
        """
        Closes the selenium session.
        
        Args:
            session: The selenium session to close
        """
        if session is not None:
            try:
                logger = logging.getLogger(__name__)
                logger.info("Closing Selenium WebDriver for GLASS data source")
                session.quit()
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Error closing Selenium session: {str(e)}")

    def get_file_hash(self, file_url: str) -> str:
        """
        Generate a unique hash for a file based on its URL.
        
        Args:
            file_url: URL of the file
            
        Returns:
            str: A unique hash identifier for the file
        """
        # Use the URL as the basis for the hash
        # For GLASS data, the URL should be unique for each file
        return hashlib.md5(file_url.encode('utf-8')).hexdigest()

    def list_remote_files(self, entrypoint: dict = None) -> list:
        def crawl(url: str, relative_path: str = ""):
            time.sleep(.5)
            res = requests.get(url)
            res.raise_for_status()  # Add error handling
            soup = BeautifulSoup(res.text, "html.parser")

            # Sort links to process years and days in order
            links = sorted(soup.find_all("a"), key=lambda link: link.get("href", ""))
 
            for link in links:
                href = link.get("href")
                if not href or href in ("../", "./"):
                    continue

                full_url = urljoin(url, href)
                new_relative_path = relative_path + href

                # Handle directories based on entrypoint
                if href.endswith("/"):
                    href_stripped = href.rstrip("/")
                    try:
                        if entrypoint:
                            # Check if folder is a 4-digit year
                            if len(href_stripped) == 4 and href_stripped.isdigit():
                                year = int(href_stripped)
                                if year == entrypoint.get("year", 0):
                                    yield from crawl(full_url, new_relative_path)
                            # Check if folder is a 3-digit day
                            elif len(href_stripped) == 3 and href_stripped.isdigit():
                                day = int(href_stripped)
                                if day == entrypoint.get("day", 0):
                                    yield from crawl(full_url, new_relative_path)
                            else:
                                pass
                        else:
                            yield from crawl(full_url, new_relative_path)
                    except ValueError:
                        yield from crawl(full_url, new_relative_path)
                elif any(href.endswith(ext) for ext in self.file_extensions):
                    yield (new_relative_path, full_url)

        return list(crawl(self.base_url))  # Convert generator to list for safer handling

    def local_path(self, relative_path: str) -> str:
        # Assuming a local directory structure that mirrors the remote one
        return os.path.join("data", relative_path)
    
    def download(self, file_url: str, output_path: str, session: requests.Session = None) -> None:
        # Use provided session or create a new one
        s = session or requests.Session()
        
        r = s.get(file_url, stream=True)
        r.raise_for_status()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    async def download_async(self, file_url: str, output_path: str, session: aiohttp.ClientSession = None) -> None:
        """
        Asynchronous download method with respectful rate limiting.
        
        Args:
            file_url: URL to download from
            output_path: Local path to save the file
            session: Optional aiohttp session for connection reuse
        """
        # Add a small delay to be respectful to the server
        await asyncio.sleep(0.5)  # 500ms delay between requests
        
        # Use provided session or create a new one
        if session is None:
            connector = aiohttp.TCPConnector(limit=5, limit_per_host=2)  # Conservative limits
            timeout = aiohttp.ClientTimeout(total=300, connect=30)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                await self._download_with_session(session, file_url, output_path)
        else:
            await self._download_with_session(session, file_url, output_path)
    
    async def _download_with_session(self, session: aiohttp.ClientSession, file_url: str, output_path: str):
        """Helper method to download with a given session."""
        logger = logging.getLogger(__name__)
        
        try:
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
                        
                        logger.debug(f"Successfully downloaded {os.path.basename(output_path)}")
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
        """Generate destination path for the file (legacy method name)."""
        filename = os.path.basename(relative_path)
        return f"{self.data_path}/{filename}"
    
    def filename_to_entrypoint(self, relative_path: str) -> dict:
        filename = os.path.basename(relative_path)
        try:
            # Extract year and day from filename
            # Format: GLASS06A01.V01.A2000055.h00v10.2022021.hdf
            parts = filename.split('.')
            date_part = next(part for part in parts if part.startswith('A'))
            year = int(date_part[1:5])
            day = int(date_part[5:])
            
            # Return with explicit int32 type specification to ensure schema consistency
            return {
                'year': int(year),  # Ensure int type (will be cast to int32 in index)
                'day': int(day)     # Ensure int type (will be cast to int32 in index)
            }
        except (IndexError, ValueError, StopIteration):
            return None
    
    def get_all_entrypoints(self):
        """
        Returns a list of dictionaries containing year and day entrypoints
        by recursively examining the directory structure.
        
        First checks for 4-digit year directories, then either:
        - Looks for 3-digit day subdirectories
        - Or extracts day numbers from filenames
        
        Returns:
            A list of dicts with format {'year': YYYY, 'day': DDD}
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Finding year/day entrypoints from GLASS directory structure in {self.base_url}")
        
        # Pattern to extract year and day from GLASS filenames
        pattern = r'A(\d{4})(\d{3})'
        entrypoints = []

        # Get the initial directory structure
        try:
            res = requests.get(self.base_url)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            
            # Find all directories that are 4-digit numbers (years)
            year_links = [link.get("href") for link in soup.find_all("a") 
                         if link.get("href") and 
                         link.get("href").rstrip('/').isdigit() and 
                         len(link.get("href").rstrip('/')) == 4]
            
            # Sort year directories numerically
            year_links.sort()
            
            for year_link in year_links:
                year = int(year_link.rstrip('/'))
                year_url = urljoin(self.base_url, year_link)
                
                # Check the second level for days
                time.sleep(0.3)  # Avoid hammering the server
                year_res = requests.get(year_url)
                year_res.raise_for_status()
                year_soup = BeautifulSoup(year_res.text, "html.parser")
                
                # Look for 3-digit day directories
                day_links = [link.get("href") for link in year_soup.find_all("a")
                            if link.get("href") and 
                            link.get("href").rstrip('/').isdigit() and 
                            len(link.get("href").rstrip('/')) == 3]
                
                if day_links:
                    # If day directories exist, use them
                    for day_link in day_links:
                        day = int(day_link.rstrip('/'))
                        # Ensure consistent int types for schema consistency
                        entrypoints.append({
                            'year': int(year),  # Will be cast to int32 in index
                            'day': int(day)     # Will be cast to int32 in index
                        })
                else:
                    # If no day directories, only use the year with day=0
                    # Ensure consistent int types for schema consistency
                    entrypoints.append({
                        'year': int(year),  # Will be cast to int32 in index
                        'day': int(0)       # Will be cast to int32 in index
                    })
            
            logger.info(f"Generated {len(entrypoints)} year/day combinations for GLASS data")
        except Exception as e:
            logger.error(f"Error exploring GLASS directory structure: {str(e)}")
        
        return entrypoints

    async def list_remote_files_async(self, entrypoint: dict = None) -> list:
        """
        Asynchronous version of list_remote_files with respectful rate limiting.
        
        Args:
            entrypoint: Optional entrypoint to filter results
            
        Returns:
            List of (relative_path, file_url) tuples
        """
        async def crawl_async(session: aiohttp.ClientSession, url: str, relative_path: str = ""):
            # Be more respectful with rate limiting
            await asyncio.sleep(0.8)  # 800ms delay between directory requests
            
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    content = await response.text()
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to crawl {url}: {e}")
                return []
                
            soup = BeautifulSoup(content, "html.parser")
            links = sorted(soup.find_all("a"), key=lambda link: link.get("href", ""))
            
            tasks = []
            results = []
            
            for link in links:
                href = link.get("href")
                if not href or href in ("../", "./"):
                    continue

                full_url = urljoin(url, href)
                new_relative_path = relative_path + href

                # Handle directories based on entrypoint
                if href.endswith("/"):
                    href_stripped = href.rstrip("/")
                    try:
                        if entrypoint:
                            # Check if folder matches entrypoint criteria
                            if len(href_stripped) == 4 and href_stripped.isdigit():
                                year = int(href_stripped)
                                if year == entrypoint.get("year", 0):
                                    tasks.append(crawl_async(session, full_url, new_relative_path))
                            elif len(href_stripped) == 3 and href_stripped.isdigit():
                                day = int(href_stripped)
                                if day == entrypoint.get("day", 0):
                                    tasks.append(crawl_async(session, full_url, new_relative_path))
                        else:
                            tasks.append(crawl_async(session, full_url, new_relative_path))
                    except ValueError:
                        if not entrypoint:  # Only crawl unknown directories if no entrypoint filter
                            tasks.append(crawl_async(session, full_url, new_relative_path))
                elif any(href.endswith(ext) for ext in self.file_extensions):
                    results.append((new_relative_path, full_url))
            
            # Execute subdirectory crawls with limited concurrency
            if tasks:
                # Process in smaller batches to avoid overwhelming the server
                batch_size = 3
                for i in range(0, len(tasks), batch_size):
                    batch = tasks[i:i + batch_size]
                    subdirectory_results = await asyncio.gather(*batch, return_exceptions=True)
                    
                    for result in subdirectory_results:
                        if isinstance(result, list):
                            results.extend(result)
                        elif isinstance(result, Exception):
                            logger = logging.getLogger(__name__)
                            logger.warning(f"Error crawling subdirectory: {result}")
                    
                    # Add a small delay between batches
                    if i + batch_size < len(tasks):
                        await asyncio.sleep(1.0)
            
            return results

        # Use conservative connection limits
        connector = aiohttp.TCPConnector(limit=5, limit_per_host=2)
        timeout = aiohttp.ClientTimeout(total=600, connect=60)  # Longer timeouts for directory crawling
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            return await crawl_async(session, self.base_url)


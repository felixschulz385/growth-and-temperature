# download/glass.py
import re
import calendar
from datetime import datetime
import logging
import time
import hashlib
import os

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

    def gcs_upload_path(self, base_url: str, relative_path: str) -> str:
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
            return {'year': year, 'day': day}
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
                        entrypoints.append({'year': year, 'day': day})
                else:
                    # If no day directories, only use the year
                    entrypoints.append({'year': year, 'day': 0})
            
            logger.info(f"Generated {len(entrypoints)} year/day combinations for GLASS data")
        except Exception as e:
            logger.error(f"Error exploring GLASS directory structure: {str(e)}")
        
        return entrypoints


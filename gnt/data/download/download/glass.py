# download/glass.py
import re
import calendar
from datetime import datetime
import logging
import time

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os

from gnt.data.download.download.base import BaseDataSource

class GLASSDataSource(BaseDataSource):
    def __init__(self, base_url: str, file_extensions=None):
        self.DATA_SOURCE_NAME = "glass"
        self.base_url = base_url
        self.file_extensions = file_extensions or [".hdf"]
        
        parsed = urlparse(base_url)
        parts = parsed.path.strip("/").split("/")
        datatype = "/".join(parts[1:]) if len(parts) > 2 else "unknown"
        self.data_path = f"{self.DATA_SOURCE_NAME}/{datatype}"

    def list_remote_files(self, entrypoint: dict = None) -> list:
        def crawl(url: str, relative_path: str = ""):
            time.sleep(.5)
            res = requests.get(url)
            soup = BeautifulSoup(res.text, "html.parser")

            # Sort links to process years and days in order
            links = sorted(soup.find_all("a"), key=lambda link: link.get("href", ""))
            
            for link in links:
                href = link.get("href")
                if not href or href in ("../", "./"):
                    continue

                # Skip directories based on entrypoint
                if entrypoint and href.endswith("/"):
                    try:
                        # Check if the folder name is a year (4 digits)
                        num_folder = int(href.rstrip("/"))
                        if len(href.rstrip("/")) == 4 and num_folder < entrypoint.get("year", 0):
                            continue
                        if len(href.rstrip("/")) == 3 and num_folder < entrypoint.get("day", 0):
                            continue
                    except ValueError:
                        pass

                full_url = urljoin(url, href)
                new_relative_path = relative_path + href

                if href.endswith("/"):
                    yield from crawl(full_url, new_relative_path)
                else:
                    if any(href.endswith(ext) for ext in self.file_extensions):
                        yield (new_relative_path, full_url)

        return crawl(self.base_url)

    def local_path(self, relative_path: str) -> str:
        # Assuming a local directory structure that mirrors the remote one
        return os.path.join("data", relative_path)
    
    def download(self, file_url: str, output_path: str, session: requests.Session = None) -> None:
        r = requests.get(file_url, stream=True)
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
        Returns a list of all available entrypoints in format "YYYY/DDD/"
        by inferring from first and last available files.
        
        Finds date range by following first/last links to files with pattern:
        GLASS06A01.V01.A2000055.h00v10.2022021.hdf (year = 2000, day = 055)
        
        Returns:
            A list of strings in format "YYYY/DDD/"
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Finding year/day range from GLASS files in {self.base_url}")
        
        # Pattern to extract year and day from GLASS filenames
        pattern = r'A(\d{4})(\d{3})'
        
        def get_file_date(url, take_first=True, max_depth=10):
            depth = 0
            current_url = url
            
            while depth < max_depth:
                try:
                    time.sleep(0.1)
                    res = requests.get(current_url)
                    if res.status_code != 200:
                        return None, None
                    
                    soup = BeautifulSoup(res.text, "html.parser")
                    links = [link.get("href") for link in soup.find_all("a") 
                             if link.get("href") and link.get("href") not in ("../", "./")]
                    
                    for href in sorted(links, reverse=not take_first):
                        if any(href.lower().endswith(ext.lower()) for ext in self.file_extensions):
                            match = re.search(pattern, href)
                            if match:
                                return int(match.group(1)), int(match.group(2))
                        if href.endswith('/'):
                            current_url = urljoin(current_url, href)
                            break
                    depth += 1
                
                except Exception as e:
                    logger.warning(f"Error exploring URL {current_url}: {str(e)}")
                    return None, None
            
            return None, None
        
        first_year, first_day = get_file_date(self.base_url, take_first=True)
        last_year, last_day = get_file_date(self.base_url, take_first=False)
        
        if first_year is None or last_year is None:
            logger.warning("Could not find valid GLASS files. Using default range.")
            first_year, first_day = 2000, 1
            last_year, last_day = datetime.now().year, 365
        
        logger.info(f"Found GLASS data ranging from {first_year}/{first_day:03d} to {last_year}/{last_day:03d}")
        
        time_step = 8
        entrypoints = []
        current_year = first_year
        current_day = first_day
        
        while current_year < last_year or (current_year == last_year and current_day <= last_day):
            entrypoints.append(f"{current_year}/{current_day:03d}/")
            
            current_day += time_step
            days_in_year = 366 if calendar.isleap(current_year) else 365
            if current_day > days_in_year:
                current_day -= days_in_year
                current_year += 1
        
        logger.info(f"Generated {len(entrypoints)} year/day combinations for GLASS data")
        return entrypoints


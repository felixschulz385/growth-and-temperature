# download/eog.py
import requests
import logging
import os
import time
import hashlib
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import Generator, Tuple, List, Dict, Any, Optional

from gnt.data.download.sources.base import BaseDataSource

logger = logging.getLogger(__name__)

class EOGDataSource(BaseDataSource):
    """
    Data source for Earth Observation Group (EOG) nighttime lights data.
    Handles authentication and downloading from the EOG repository.
    """
    def __init__(self, base_url: str, file_extensions: list[str] = None, output_path: str = None):
        """
        Initialize EOG data source.
        
        Args:
            base_url: Base URL for the EOG repository
            file_extensions: List of file extensions to download (default: .tif, .tgz, .tar.gz)
            output_path: Custom output path in GCS (optional)
        """
        self.DATA_SOURCE_NAME = "eog"
        self.base_url = base_url
        self.file_extensions = file_extensions or [".tif", ".tgz", ".tar.gz"]
        self.has_entrypoints = False  # Changed to False to disable entrypoint functionality
        
        # Parse URL to extract data path
        parsed = urlparse(base_url)
        parts = parsed.path.strip("/").split("/")
        datatype = "/".join(parts[1:]) if len(parts) > 2 else "unknown"
        
        # Use custom output path if provided, otherwise construct from URL
        if output_path:
            self.data_path = output_path
        else:
            self.data_path = f"{self.DATA_SOURCE_NAME}/{datatype}"
        
        logger.info(f"Initialized EOG data source with path: {self.data_path}")

    def list_remote_files(self, entrypoint: dict = None) -> Generator[Tuple[str, str], None, None]:
        """
        Lists all files from the EOG data repository using a generator approach.
        Uses a simple tree-based crawling approach.
        
        Args:
            entrypoint: Ignored parameter (entrypoint functionality removed)
            
        Returns:
            Generator yielding tuples of (relative_path, file_url)
        """
        logger.info(f"Starting to crawl EOG data source: {self.base_url}")
        base_url_parsed = urlparse(self.base_url)
        
        def extract_path_from_url(url):
            """Extract the path component relative to base URL"""
            # Parse the URL
            parsed = urlparse(url)
            
            # If URL is on a different host, return None
            if parsed.netloc != base_url_parsed.netloc:
                return None
                
            # Get path relative to base URL
            if not parsed.path.startswith(base_url_parsed.path):
                return parsed.path.lstrip('/')
                
            relative = parsed.path[len(base_url_parsed.path):].lstrip('/')
            return relative
        
        def crawl(url):
            """Generator function that crawls a URL and yields found files."""
            
            try:
                # Add small delay to avoid hammering the server
                time.sleep(0.5)
                logger.debug(f"Crawling directory: {url}")
                
                res = requests.get(url)
                res.raise_for_status()
                soup = BeautifulSoup(res.text, "html.parser")
                
                # Sort links for more predictable processing
                links = [link for link in soup.find_all("td", {"class": "indexcolname"})]

                for link in links:
                    href_elem = link.find("a")
                    if not href_elem:
                        continue
                        
                    href = href_elem.get("href")
                    
                    # Skip invalid, parent directory and self-references
                    if not href or href in ("../", "./", "/"):
                        continue
                    
                    full_url = urljoin(url, href)
                    
                    # Skip links that would create loops
                    # (if the link equals the current URL or points to a parent directory)
                    if full_url == url or url.startswith(full_url):
                        continue
                    
                    if href.endswith("/"):  # Directory
                        # Recurse into directory - simple tree traversal
                        yield from crawl(full_url)
                    
                    # File with matching extension
                    elif not self.file_extensions or any(href.lower().endswith(ext.lower()) for ext in self.file_extensions):
                        # Get the relative path by extracting from URL
                        relative_path = extract_path_from_url(full_url)
                        if not relative_path:
                            continue
                            
                        logger.debug(f"Found file: {relative_path}")
                        yield (relative_path, full_url)

            except Exception as e:
                logger.error(f"Error crawling directory {url}: {str(e)}")
        
        # Start crawling from base URL and yield all files found
        yield from crawl(self.base_url)
        
        # Log summary after generator is exhausted
        logger.info(f"Completed crawling EOG data source.")

    def local_path(self, relative_path: str) -> str:
        """
        Generates local path for a file.
        
        Args:
            relative_path: Relative path of the file
            
        Returns:
            Local path for the file
        """
        # Assuming a local directory structure that mirrors the remote one
        return os.path.join("data", self.DATA_SOURCE_NAME, relative_path)
                
    def get_authenticated_session(self) -> requests.Session:
        """
        Creates an authenticated session for EOG data downloads.
        Uses Selenium to log in and capture cookies.
        
        Returns:
            requests.Session: Authenticated session with valid cookies
        """
        logger.info("Creating authenticated session for EOG data")
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
        except ImportError:
            logger.error("Selenium is required for EOG authentication. Please install with: pip install selenium")
            raise ImportError("Selenium is required for EOG authentication")

        # Use Selenium to log in and get cookies
        username = os.environ.get("EOG_USERNAME")
        password = os.environ.get("EOG_PASSWORD")

        if not username or not password:
            raise ValueError("EOG_USERNAME and EOG_PASSWORD must be set in environment variables.")

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        # This disables downloads
        prefs = {
            "download.prompt_for_download": False,
            "download.default_directory": "/dev/null",  # not writable in most cases
            "download_restrictions": 3  # 0 = allow all, 3 = block all
        }
        chrome_options.add_experimental_option("prefs", prefs)
        logger.info("Starting Chrome in headless mode")
        driver = webdriver.Chrome(options=chrome_options)
        
        # Use a small sample file URL that will trigger login
        login_prompting_url = "https://eogdata.mines.edu/wwwdata/dmsp/v4composites_rearrange/F10_1992/F101992.v4b.global.avg_vis.tif"
        logger.info(f"Navigating to login page via {login_prompting_url}")
        driver.get(login_prompting_url)

        try:
            # Fill login form
            logger.info("Filling login form")
            driver.find_element("id", "username").send_keys(username)
            driver.find_element("id", "password").send_keys(password)
            driver.find_element("name", "login").click()

            # Wait for login & redirect
            logger.info("Waiting for login to complete")
            time.sleep(5)

            # Download using cookies from the session
            cookies = driver.get_cookies()
            
            # Create session with cookies
            session = requests.Session()
            for c in cookies:
                session.cookies.set(c['name'], c['value'])
            
            logger.info("Authenticated session created successfully")
            return session
        
        finally:
            driver.quit()
            logger.info("Browser session closed")

    def download(self, file_url: str, output_path: str, session: requests.Session = None) -> None:
        """
        Download a file from EOG data repository.
        
        Args:
            file_url: URL of the file to download
            output_path: Local path to save the file
            session: Authenticated session (required for EOG)
        """
        if session is None:
            logger.info("No session provided, creating new authenticated session")
            session = self.get_authenticated_session()

        try:
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
            
            logger.info(f"Downloading {os.path.basename(output_path)} from EOG")
            r = session.get(file_url, stream=True)
            r.raise_for_status()

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Successfully downloaded {os.path.basename(output_path)}")
            
        except Exception as e:
            logger.error(f"Error downloading {file_url}: {str(e)}")
            raise

    def gcs_upload_path(self, base_url: str, relative_path: str) -> str:
        """
        Generate the GCS path for a file.
        
        Args:
            base_url: Base URL (not used for EOG)
            relative_path: Relative path of the file
            
        Returns:
            GCS path for the file
        """
        # Extract the relative path from the full URL structure
        path_parts = relative_path.split("/")
        
        # For DMSP data (e.g., F10_1992/F101992.v4b.global.avg_vis.tif)
        # Extract data type (dmsp or viirs)
        if "dmsp" in self.data_path.lower():
            # Keep filename and append to data_path
            filename = path_parts[-1]
            # Include satellite identifier if present in the path
            satellite = None
            for part in path_parts:
                if part.startswith("F") and "_" in part:
                    satellite = part.split("_")[0]
                    break
            
            if satellite:
                return f"{self.data_path}/dmsp/{satellite}/{filename}"
            else:
                return f"{self.data_path}/dmsp/{filename}"
        
        elif "viirs" in self.data_path.lower():
            # Keep filename and append to data_path
            filename = path_parts[-1]
            return f"{self.data_path}/viirs/{filename}"
        
        # Default case - use full relative path
        return f"{self.data_path}/{relative_path}"

    def get_file_hash(self, file_url: str) -> str:
        """
        Generate a unique hash for a file URL.
        Used by download index to track file status.
        
        Args:
            file_url: URL of the file
            
        Returns:
            MD5 hash of the file URL
        """
        return hashlib.md5(file_url.encode()).hexdigest()

    def filename_to_entrypoint(self, relative_path: str) -> Optional[Dict[str, Any]]:
        pass

    def get_all_entrypoints(self) -> List[Dict[str, Any]]:
        """
        Returns an empty list as this data source doesn't use entrypoints.
        Implementation required by the abstract base class.
        
        Returns:
            An empty list
        """
        logger.info("Entrypoints not used for this data source")
        return []



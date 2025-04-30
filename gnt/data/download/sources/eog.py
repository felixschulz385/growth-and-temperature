# download/glass.py
import requests
import logging
import os
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from gnt.data.download.download.base import BaseDataSource

logger = logging.getLogger(__name__)

class EOGDataSource(BaseDataSource):
    def __init__(self, base_url: str, file_extensions: list[str] = None):
        self.DATA_SOURCE_NAME = "eog"
        self.base_url = base_url
        self.file_extensions = file_extensions or [".tif", ".tgz", ".tar.gz"]
        self.has_entrypoints = False
        
        # Parse URL to extract data path
        parsed = urlparse(base_url)
        parts = parsed.path.strip("/").split("/")
        datatype = "/".join(parts[1:]) if len(parts) > 2 else "unknown"
        self.data_path = f"{self.DATA_SOURCE_NAME}/{datatype}"

    def list_remote_files(self, entrypoint: dict = None) -> list:
        """
        Lists all files from the EOG data repository.
        For EOG, we don't use entrypoints but include the parameter for compatibility.
        """
        def collect_files_from_url(url: str, relative_prefix: str = "") -> list:
            collected_files = []
            
            try:
                logger.debug(f"Fetching directory listing from {url}")
                page = requests.get(url)
                page.raise_for_status()
                soup = BeautifulSoup(page.text, "html.parser")

                for link_td in soup.find_all("td", {"class": "indexcolname"}):
                    href = link_td.a.get("href")
                    if not href:
                        continue
                    # Skip parent directory links
                    if urlparse(url).path.startswith(href) and href.count("/") == urlparse(url).path.count("/") - 1:
                        continue
                    
                    full_url = urljoin(url, href)

                    if href.endswith("/"):  # It's a directory, recurse into it
                        logger.debug(f"Found directory: {href}, recursing...")
                        new_prefix = os.path.join(relative_prefix, href) if relative_prefix else href
                        collected_files.extend(collect_files_from_url(full_url, new_prefix))
                    elif not self.file_extensions or any(href.endswith(ext) for ext in self.file_extensions):
                        # It's a file with matching extension
                        relative_path = os.path.join(relative_prefix, href) if relative_prefix else href
                        collected_files.append((relative_path, full_url))
                        logger.debug(f"Found file: {relative_path}")

            except Exception as e:
                logger.error(f"Error fetching directory listing from {url}: {str(e)}")
            
            return collected_files

        logger.info(f"Starting to list files from EOG source: {self.base_url}")
        result = collect_files_from_url(self.base_url)
        logger.info(f"Found {len(result)} files in EOG source")
        return result

    def local_path(self, relative_path: str) -> str:
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
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        import time

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
        driver.quit()
        logger.info("Browser session closed")

        # Create session with cookies
        session = requests.Session()
        for c in cookies:
            session.cookies.set(c['name'], c['value'])
        
        logger.info("Authenticated session created successfully")
        return session

    def download(self, file_url: str, output_path: str, session: requests.Session = None) -> None:
        """
        Download a file from EOG data repository.
        
        Args:
            file_url: URL of the file to download
            output_path: Local path to save the file
            session: Authenticated session (required for EOG)
        """
        if session is None:
            raise ValueError("Authenticated session must be provided for EOG downloads.")

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
        """Generate the GCS path for a file."""
        filename = os.path.basename(relative_path)
        return f"{self.data_path}/{filename}"
    
    def filename_to_entrypoint(self, relative_path: str) -> dict:
        """
        EOG data doesn't use entrypoints, but we implement this method
        to maintain compatibility with the BaseDataSource API.
        """
        return None
    
    def get_all_entrypoints(self):
        """
        EOG data doesn't use entrypoints because files are organized by satellite/year.
        Return an empty list for compatibility.
        """
        return []


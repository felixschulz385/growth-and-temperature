# download/eog.py
import logging
import os
import time
import hashlib
import tempfile
import shutil
import random
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import Generator, Tuple, List, Dict, Any, Optional

# Import Selenium components
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, StaleElementReferenceException
try:
    from webdriver_manager.chrome import ChromeDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError:
    WEBDRIVER_MANAGER_AVAILABLE = False

from gnt.data.download.sources.base import BaseDataSource

logger = logging.getLogger(__name__)

class EOGDataSource(BaseDataSource):
    """
    Data source for Earth Observation Group (EOG) nighttime lights data.
    Uses Selenium for authenticated browsing and downloading from the EOG repository.
    """
    # EOG login URL
    EOG_LOGIN_URL = "https://eogdata.mines.edu/nighttime_light/login/"
    
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
        self.file_extensions = file_extensions or [".tif", ".tgz", ".tar.gz", ".gz"]
        self.has_entrypoints = False
        
        # Parse URL to extract data path
        parsed = urlparse(base_url)
        parts = parsed.path.strip("/").split("/")
        datatype = "/".join(parts[1:]) if len(parts) > 2 else "unknown"
        
        # Use custom output path if provided, otherwise construct from URL
        if output_path:
            self.data_path = output_path
        else:
            self.data_path = f"{self.DATA_SOURCE_NAME}/{datatype}"
        
        # Selenium WebDriver
        self._driver = None
        self._download_dir = None
        self._is_logged_in = False
        
        # Credentials
        self._username = os.environ.get("EOG_USERNAME")
        self._password = os.environ.get("EOG_PASSWORD")
        
        if not self._username or not self._password:
            logger.warning("EOG credentials not set in environment variables (EOG_USERNAME, EOG_PASSWORD)")
        
        logger.info(f"Initialized EOG data source with path: {self.data_path}")

    def _init_selenium_driver(self):
        """Initialize the Selenium WebDriver for downloading files."""
        if self._driver is not None:
            return
            
        logger.info("Initializing Selenium WebDriver")
        
        try:
            # Create a temporary directory for downloads
            if self._download_dir is None:
                self._download_dir = tempfile.mkdtemp(prefix="eog_downloads_")
                logger.info(f"Created temporary download directory: {self._download_dir}")
            
            # Configure Chrome options
            chrome_options = Options()
            
            # Set download directory
            prefs = {
                "download.default_directory": self._download_dir,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": False
            }
            chrome_options.add_experimental_option("prefs", prefs)
            
            # Headless mode for server environments
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument('--ignore-ssl-errors=yes')
            chrome_options.add_argument('--ignore-certificate-errors')
            
            # Initialize the WebDriver
            if WEBDRIVER_MANAGER_AVAILABLE:
                self._driver = webdriver.Chrome(
                    service=Service(ChromeDriverManager().install()), 
                    options=chrome_options
                )
            else:
                self._driver = webdriver.Chrome(options=chrome_options)
            
            # Set page load timeout
            self._driver.set_page_load_timeout(120)
            
            logger.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Selenium WebDriver: {e}")
            raise

    def _close_selenium_driver(self):
        """Close the Selenium WebDriver if it exists."""
        if self._driver is not None:
            try:
                self._driver.quit()
                logger.info("Selenium WebDriver closed")
            except Exception as e:
                logger.warning(f"Error closing Selenium WebDriver: {e}")
            finally:
                self._driver = None
                self._is_logged_in = False

        # Clean up temporary download directory
        if self._download_dir and os.path.exists(self._download_dir):
            try:
                shutil.rmtree(self._download_dir)
                logger.info(f"Removed temporary download directory: {self._download_dir}")
                self._download_dir = None
            except Exception as e:
                logger.warning(f"Error removing temporary directory: {e}")

    def _check_and_handle_login(self):
        """
        Check if login form is present and handle login if needed.
        Note: On EOG, the login form may stay visible even after successful login,
        but the download will start automatically.
        """
        try:
            # Check for login form
            username_field = WebDriverWait(self._driver, 5).until(
                EC.presence_of_element_located((By.ID, "username"))
            )
            
            # If already logged in, the form might be visible but download already started
            # Check if any files have started downloading before attempting login
            if self._is_logged_in:
                logger.info("Login form visible but already logged in - checking for downloads")
                return True
            
            logger.info("Login form detected, attempting to log in")
            
            if not self._username or not self._password:
                raise ValueError("EOG_USERNAME and EOG_PASSWORD must be set in environment variables")
            
            # Fill in login form
            username_field.send_keys(self._username)
            self._driver.find_element(By.ID, "password").send_keys(self._password)
            
            # Submit the form - this should trigger the download directly
            login_button = self._driver.find_element(By.ID, "kc-login")
            login_button.click()
            
            # Wait a moment for the form submission to complete
            # Note: Form may still be visible after successful login
            time.sleep(3)
            
            # Check if login was successful by testing if the login button is still clickable
            try:
                if login_button.is_enabled():
                    logger.error("Login failed: login button still clickable")
                    return False
            except StaleElementReferenceException:
                pass  # Button not found, which is good - means we're logged in
            
            # Set login state to true - we'll verify success by checking for downloads
            self._is_logged_in = True
            logger.info("Login attempted, will verify success by checking for downloads")
            return True
            
        except TimeoutException:
            # No login form detected
            logger.debug("No login form detected, already logged in or not required")
            return True
        except Exception as e:
            logger.error(f"Error during login process: {e}")
            return False

    def list_remote_files(self, entrypoint: dict = None) -> Generator[Tuple[str, str], None, None]:
        """
        Lists all files from the EOG data repository using Selenium for authenticated browsing.
        
        Args:
            entrypoint: Ignored parameter (entrypoint functionality removed)
            
        Returns:
            Generator yielding tuples of (relative_path, file_url)
        """
        logger.info(f"Starting to crawl EOG data source: {self.base_url}")
        base_url_parsed = urlparse(self.base_url)
        
        # Initialize Selenium
        self._init_selenium_driver()
        
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
        
        def crawl(url, depth=0, max_depth=8):
            """Generator function that crawls a URL and yields found files."""
            
            # Prevent excessive recursion
            if depth > max_depth:
                logger.warning(f"Maximum recursion depth reached at {url}")
                return
                
            try:
                # Add small delay to avoid hammering the server
                time.sleep(1 + random.random())
                logger.debug(f"Crawling directory: {url} (depth {depth})")
                
                # Navigate to URL
                self._driver.get(url)
                
                # Check if we need to log in
                if not self._check_and_handle_login():
                    logger.error("Failed to log in, cannot continue crawling")
                    return
                
                # Wait for page to load
                WebDriverWait(self._driver, 30).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # Get page source and parse with BeautifulSoup
                html = self._driver.page_source
                soup = BeautifulSoup(html, "html.parser")
                
                # Find all links - look for table elements first (typical directory listing)
                links = []
                
                # Try first with index column names (common in Apache directory listings)
                td_links = soup.find_all("td", {"class": "indexcolname"})
                if td_links:
                    for td in td_links:
                        a_tag = td.find("a")
                        if a_tag and a_tag.get("href"):
                            links.append(a_tag)
                else:
                    # Fall back to all a tags
                    links = soup.find_all("a")

                # Process links
                for link in links:
                    href = link.get("href")
                    
                    # Skip invalid, parent directory and self-references
                    if not href or href in ("../", "./", "/"):
                        continue
                    
                    full_url = urljoin(url, href)
                    
                    # Skip links that would create loops
                    if full_url == url or url.startswith(full_url):
                        continue
                    
                    if href.endswith("/"):  # Directory
                        # Recurse into directory
                        yield from crawl(full_url, depth + 1, max_depth)
                    
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
        
        try:
            # Navigate to base URL first and check for login
            self._driver.get(self.base_url)
            if not self._check_and_handle_login():
                logger.error("Failed to log in to EOG portal")
                return
            
            # Start crawling from base URL and yield all files found
            yield from crawl(self.base_url)
            
            # Log summary after generator is exhausted
            logger.info(f"Completed crawling EOG data source.")
        finally:
            # Clean up Selenium resources
            self._close_selenium_driver()

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
                
    def get_authenticated_session(self) -> webdriver.Chrome:
        """
        Creates an authenticated WebDriver session for EOG data downloads.
        
        Returns:
            webdriver.Chrome: Authenticated WebDriver instance
        """
        logger.info("Creating authenticated Selenium session for EOG data")
        
        # Initialize driver if needed
        self._init_selenium_driver()
        
        # Navigate to base URL and check for login
        try:
            self._driver.get(self.base_url)
            if not self._check_and_handle_login():
                raise RuntimeError("Failed to authenticate with EOG portal")
        except Exception as e:
            logger.error(f"Error creating authenticated session: {e}")
            self._close_selenium_driver()
            raise
            
        return self._driver

    def download_file(self, file_url, output_path):
        """
        Download a file using Selenium WebDriver.
        On EOG, direct download URLs trigger downloads immediately after login,
        even though the login form may remain visible.
        
        Args:
            file_url: URL of the file to download
            output_path: Local path to save the file
            
        Returns:
            True if download succeeded, False otherwise
        """
        # Initialize driver if needed
        if self._driver is None:
            self._init_selenium_driver()
        
        try:
            # Get initial directory state
            before_files = set(os.listdir(self._download_dir))
            
            # Navigate to the file URL - this should either start download or show login
            logger.info(f"Navigating to file URL: {file_url}")
            self._driver.get(file_url)
            
            # Handle login if needed
            self._check_and_handle_login()
            
            # Wait for download to complete by checking the download directory
            max_wait_time = 300  # 5 minutes
            interval = 5
            elapsed = 0
            download_started = False
            
            while elapsed < max_wait_time:
                # Check for new files
                current_files = set(os.listdir(self._download_dir))
                new_files = current_files - before_files
                
                # Check for temporary download files - sign that download has started
                temp_files = [f for f in new_files 
                            if f.endswith('.tmp') or f.endswith('.crdownload')]
                
                if temp_files and not download_started:
                    download_started = True
                    logger.info("Download has started")
                
                # Filter out temporary download files to find completed ones
                completed_files = [f for f in new_files 
                                 if not f.endswith('.tmp') 
                                 and not f.endswith('.crdownload')]
                
                if completed_files:
                    # Get the most recently modified file
                    latest_file = max(
                        [os.path.join(self._download_dir, f) for f in completed_files],
                        key=os.path.getmtime
                    )
                    
                    # Make sure the directory exists
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Copy the file to the desired output location
                    shutil.copy2(latest_file, output_path)
                    
                    logger.info(f"Successfully downloaded file to: {output_path}")
                    return True
                
                # Wait and check again
                time.sleep(interval)
                elapsed += interval
                
                # If no download started after a while, try re-submitting login
                if not download_started and elapsed >= 30 and elapsed % 30 == 0:
                    logger.warning(f"No download started after {elapsed}s, attempting login again")
                    try:
                        # Try to resubmit the login form
                        submit_button = self._driver.find_element(By.ID, "kc-login") 
                        if not submit_button:
                            submit_button = self._driver.find_element(By.XPATH, "//button[@type='submit']")
                        
                        submit_button.click()
                        logger.info("Re-submitted login form")
                    except Exception as e:
                        logger.warning(f"Could not re-submit login: {e}")
                
                if elapsed % 30 == 0:  # Log progress every 30 seconds
                    logger.info(f"Waiting for download to complete... ({elapsed}s elapsed)")
            
            logger.error("Download timeout exceeded")
            return False
            
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return False

    def download(self, file_url: str, output_path: str, session=None) -> None:
        """
        Download a file from EOG data repository using Selenium.
        
        Args:
            file_url: URL of the file to download
            output_path: Local path to save the file
            session: Optional WebDriver session (will be created if None)
        """
        close_driver = False
        
        try:
            if session is None:
                # If session is not provided, use our own driver
                if self._driver is None:
                    self._init_selenium_driver()
                close_driver = True
            else:
                # If a session was provided, use it directly
                # This is assuming the session is a WebDriver instance
                self._driver = session
            
            # Perform the download
            success = self.download_file(file_url, output_path)
            
            if not success:
                raise RuntimeError(f"Failed to download {file_url}")
                
        finally:
            # Clean up if we created our own driver
            if close_driver:
                self._close_selenium_driver()

    def gcs_upload_path(self, base_url: str, relative_path: str) -> str:
        """
        Generate the GCS path for a file relative to configured target.
        
        Args:
            base_url: Base URL (not used for EOG)
            relative_path: Relative path of the file
            
        Returns:
            Path for the file relative to configured target path
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
                return f"{satellite}/{filename}"
            else:
                return filename
        
        elif "viirs" in self.data_path.lower():
            # Keep filename and append to data_path
            filename = path_parts[-1]
            return filename
        
        # Default case - use full relative path without the data_path prefix
        return relative_path

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
        """
        Not implemented for EOG data source.
        
        Args:
            relative_path: Relative path of the file
            
        Returns:
            None as entrypoints are not used
        """
        return None

    def get_all_entrypoints(self) -> List[Dict[str, Any]]:
        """
        Returns an empty list as this data source doesn't use entrypoints.
        Implementation required by the abstract base class.
        
        Returns:
            An empty list
        """
        logger.info("Entrypoints not used for this data source")
        return []
        
    def __del__(self):
        """Clean up resources when the object is destroyed."""
        self._close_selenium_driver()



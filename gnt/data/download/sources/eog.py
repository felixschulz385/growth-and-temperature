# download/eog.py
import logging
import os
import time
import hashlib
import tempfile
import shutil
import random
import asyncio
import aiohttp
import aiofiles
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
            output_path: Custom output path in GCS (required)
        """
        # Initialize all attributes first to avoid issues in __del__
        self.DATA_SOURCE_NAME = "eog"
        self.base_url = base_url
        self.file_extensions = file_extensions or [".tif", ".tgz", ".tar.gz", ".gz"]
        self.has_entrypoints = False
        
        # Selenium WebDriver attributes - initialize early
        self._driver = None
        self._download_dir = None
        self._is_logged_in = False
        
        # Validate required parameters
        if not output_path:
            logger.error("No output_path defined for EOGDataSource; cannot set data_path.")
            raise ValueError("output_path must be defined for EOGDataSource.")
        
        self.data_path = output_path
        
        # Define schema types for Parquet consistency
        self.schema_dtypes = {
            'year': 'int32',            # Explicitly use int32 for year
            'day_of_year': 'int32',     # Explicitly use int32 for day_of_year
            'timestamp_precision': 'ms', # Use millisecond precision for timestamps
            'file_size': 'int64',       # Consistent int64 for file sizes
            'download_status': 'string', # Consistent string type
            'status_category': 'string'  # Consistent string type
        }
        
        # Credentials
        self._username = os.environ.get("EOG_USERNAME")
        self._password = os.environ.get("EOG_PASSWORD")
        
        if not self._username or not self._password:
            logger.warning("EOG credentials not set in environment variables (EOG_USERNAME, EOG_PASSWORD)")
        
        logger.info(f"Initialized EOG data source with path: {self.data_path}")

    def get_selenium_session(self):
        """
        Returns a selenium session (webdriver).
        Creates it if it does not exist.
        
        Note: The workflow context will be responsible for maintaining
        the persistent session, this method is just a factory.
        """
        logger.info("Creating new Selenium WebDriver for EOG data source")
        
        try:
            # Create a shared temporary directory for all downloads
            # Use a consistent name so all sessions can share the same directory
            download_dir = tempfile.mkdtemp(prefix="eog_shared_downloads_")
            logger.info(f"Created shared download directory: {download_dir}")
            
            # Configure Chrome options
            chrome_options = Options()
            
            # Set download directory
            prefs = {
                "download.default_directory": download_dir,
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
                driver = webdriver.Chrome(
                    service=Service(ChromeDriverManager().install()), 
                    options=chrome_options
                )
            else:
                driver = webdriver.Chrome(options=chrome_options)
            
            # Set page load timeout
            driver.set_page_load_timeout(120)
            
            # Store download directory and login state as attributes on the driver
            driver._eog_download_dir = download_dir
            driver._eog_is_logged_in = False
            driver._eog_username = self._username
            driver._eog_password = self._password
            
            logger.info("Selenium WebDriver initialized successfully")
            return driver
            
        except Exception as e:
            logger.error(f"Failed to initialize Selenium WebDriver: {e}")
            raise

    def close_selenium_session(self, session):
        """
        Closes the selenium session.
        
        Args:
            session: The selenium session to close
        """
        if session is not None:
            try:
                logger.info("Closing Selenium WebDriver for EOG data source")
                
                # Clean up download directory if it exists
                if hasattr(session, '_eog_download_dir') and os.path.exists(session._eog_download_dir):
                    try:
                        shutil.rmtree(session._eog_download_dir)
                        logger.info(f"Removed temporary download directory: {session._eog_download_dir}")
                    except Exception as e:
                        logger.warning(f"Error removing temporary directory: {e}")
                
                session.quit()
            except Exception as e:
                logger.warning(f"Error closing Selenium session: {e}")

    def _check_and_handle_login(self, driver=None):
        """
        Check if login form is present and handle login if needed.
        Uses the driver's stored login state for session persistence.
        """
        # Use provided driver or fall back to instance driver
        current_driver = driver or self._driver
        
        # Only proceed if we have a proper Selenium WebDriver
        if not hasattr(current_driver, 'find_element'):
            logger.warning("Login check requires Selenium WebDriver, but got different session type")
            return True  # Assume login not needed for non-Selenium sessions
            
        try:
            # Check for login form
            username_field = WebDriverWait(current_driver, 5).until(
                EC.presence_of_element_located((By.ID, "username"))
            )
            
            # Check session-specific login state
            session_logged_in = getattr(current_driver, '_eog_is_logged_in', False)
            
            if session_logged_in:
                logger.info("Session already logged in - checking for downloads")
                return True
            
            logger.info("Login form detected, attempting to log in")
            
            # Get credentials from session or instance
            username = getattr(current_driver, '_eog_username', None) or self._username
            password = getattr(current_driver, '_eog_password', None) or self._password
            
            if not username or not password:
                raise ValueError("EOG_USERNAME and EOG_PASSWORD must be set in environment variables")
            
            # Fill in login form
            username_field.send_keys(username)
            current_driver.find_element(By.ID, "password").send_keys(password)
            
            # Submit the form
            login_button = current_driver.find_element(By.ID, "kc-login")
            login_button.click()
            
            # Wait a moment for the form submission to complete
            time.sleep(3)
            
            # Check if login was successful
            try:
                if login_button.is_enabled():
                    logger.error("Login failed: login button still clickable")
                    return False
            except StaleElementReferenceException:
                pass  # Button not found, which is good
            
            # Set login state on the session
            current_driver._eog_is_logged_in = True
            logger.info("Login successful for session")
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

    def download_file(self, file_url, output_path, driver=None):
        """
        Download a file using Selenium WebDriver with proper session handling.
        """
        # Use provided driver or instance driver
        current_driver = driver or self._driver
        
        # Ensure we have a proper Selenium WebDriver
        if not hasattr(current_driver, 'get') or not hasattr(current_driver, 'find_element'):
            logger.error("EOG downloads require Selenium WebDriver")
            return False
            
        if current_driver is None:
            logger.error("No Selenium driver available")
            return False
        
        try:
            # Get download directory from the driver, with fallback to instance directory
            download_dir = getattr(current_driver, '_eog_download_dir', None)
            
            # If no download directory on session, try to use instance directory or create one
            if not download_dir or not os.path.exists(download_dir):
                if self._download_dir and os.path.exists(self._download_dir):
                    download_dir = self._download_dir
                    # Store it on the driver for future use
                    current_driver._eog_download_dir = download_dir
                    logger.info(f"Using instance download directory: {download_dir}")
                else:
                    # Create a new temporary download directory
                    download_dir = tempfile.mkdtemp(prefix="eog_session_downloads_")
                    current_driver._eog_download_dir = download_dir
                    logger.info(f"Created new download directory for session: {download_dir}")
            
            # Get initial directory state
            before_files = set(os.listdir(download_dir))
            
            # Navigate to the file URL
            logger.info(f"Navigating to file URL: {file_url}")
            current_driver.get(file_url)
            
            # Handle login if needed, passing the current driver
            self._check_and_handle_login(current_driver)
            
            # Wait for download to complete by checking the download directory
            max_wait_time = 300  # 5 minutes
            interval = 5
            elapsed = 0
            download_started = False
            
            while elapsed < max_wait_time:
                # Check for new files
                current_files = set(os.listdir(download_dir))
                new_files = current_files - before_files
                
                # Check for temporary download files
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
                        [os.path.join(download_dir, f) for f in completed_files],
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
        # Always use self.data_path as the output prefix
        filename = os.path.basename(relative_path)
        return f"{self.data_path}/{filename}"

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
        try:
            if hasattr(self, '_driver'):
                self._close_selenium_driver()
        except Exception as e:
            # Silently ignore cleanup errors during destruction
            pass

    async def download_async(self, file_url: str, output_path: str, session=None) -> None:
        """
        Asynchronous download method - uses Selenium in a thread pool since EOG requires authentication.
        
        Args:
            file_url: URL to download from
            output_path: Local path to save the file
            session: Optional Selenium session for authentication (not aiohttp.ClientSession)
        """
        # EOG requires Selenium, not aiohttp sessions
        if session is not None and not hasattr(session, 'find_element'):
            logger.warning("EOG async download received non-Selenium session, ignoring it")
            session = None
            
        # Add a small delay to be respectful to the server
        await asyncio.sleep(0.5)  # 500ms delay between requests
        
        # Run the Selenium download in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        
        try:
            # Use thread pool executor for the blocking Selenium operations
            await loop.run_in_executor(
                None,  # Use default thread pool
                self._download_sync_wrapper,
                file_url,
                output_path,
                session
            )
        except Exception as e:
            logger.error(f"Error in async download for {file_url}: {e}")
            # Clean up partial file if it exists
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            raise

    async def list_remote_files_async(self, entrypoint: dict = None) -> list:
        """
        Asynchronous version of list_remote_files.
        Since EOG requires Selenium for authentication, we run it in a thread pool.
        
        Args:
            entrypoint: Optional entrypoint to filter results (not used for EOG)
            
        Returns:
            List of (relative_path, file_url) tuples
        """
        loop = asyncio.get_event_loop()
        
        # Run the synchronous list_remote_files in a thread pool
        try:
            files = await loop.run_in_executor(
                None,  # Use default thread pool
                self._list_remote_files_sync,
                entrypoint
            )
            return files
        except Exception as e:
            logger.error(f"Error in async file listing: {e}")
            return []

    def _list_remote_files_sync(self, entrypoint: dict = None) -> list:
        """
        Synchronous wrapper for list_remote_files to be used in thread pool.
        """
        return list(self.list_remote_files(entrypoint))
        
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

    def _download_sync_wrapper(self, file_url: str, output_path: str, session=None):
        """
        Synchronous wrapper that properly handles Selenium sessions.
        """
        # Check if session is a Selenium WebDriver
        if session is not None and hasattr(session, 'find_element'):
            # Use the provided Selenium session
            success = self.download_file(file_url, output_path, driver=session)
            if not success:
                raise RuntimeError(f"Failed to download {file_url}")
        else:
            # No valid session provided, use instance driver (create if needed)
            close_driver = False
            try:
                if self._driver is None:
                    self._init_selenium_driver()
                    close_driver = True
                
                success = self.download_file(file_url, output_path, driver=self._driver)
                if not success:
                    raise RuntimeError(f"Failed to download {file_url}")
                    
            finally:
                if close_driver:
                    self._close_selenium_driver()


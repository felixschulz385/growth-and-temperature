import logging
import os
import shutil
import tempfile
import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
try:
    from webdriver_manager.chrome import ChromeDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError:
    WEBDRIVER_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)


class _SessionMixin:
    """Selenium session/auth lifecycle for EOG."""

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
                raise ValueError(
                    "EOG credentials not set -- create orchestration/secrets/eog.credentials.json "
                    "({\"username\": ..., \"password\": ...}) or set EOG_USERNAME/EOG_PASSWORD "
                    "(src/data/sources/eog/credentials.py)"
                )

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

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        try:
            if hasattr(self, '_driver'):
                self._close_selenium_driver()
        except Exception as e:
            # Silently ignore cleanup errors during destruction
            pass

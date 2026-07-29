import asyncio
import hashlib
import logging
import os
import shutil
import tempfile
import time
from typing import Any, Dict, Optional

from src.data.download.sources.base import BaseDataSource

from .crawler import _CrawlerMixin
from .session import _SessionMixin

logger = logging.getLogger(__name__)


class EOGDataSource(_CrawlerMixin, _SessionMixin, BaseDataSource):
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


NAMES = ("eog_dmsp", "eog_viirs", "eog_dvnl")


def from_config(dataset_name, config, *, base_url, file_extensions, output_path, source_config, **kwargs):
    """Build an EOGDataSource from the shared config-extraction the factory does."""
    logger.info(f"Creating EOG data source: {dataset_name}")
    if not output_path:
        raise ValueError("EOG data source requires 'output_path' or 'data_path' in configuration")
    return EOGDataSource(
        base_url=source_config['base_url'],
        file_extensions=source_config.get('file_extensions', None),
        output_path=source_config.get('data_path')
    )

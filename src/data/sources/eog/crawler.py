import asyncio
import logging
import random
import time
from typing import Any, Dict, Generator, List, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logger = logging.getLogger(__name__)


class _CrawlerMixin:
    """Remote listing/discovery for EOG, requiring an authenticated Selenium session."""

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

    def get_all_entrypoints(self) -> List[Dict[str, Any]]:
        """
        Returns an empty list as this data source doesn't use entrypoints.
        Implementation required by the abstract base class.

        Returns:
            An empty list
        """
        logger.info("Entrypoints not used for this data source")
        return []

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

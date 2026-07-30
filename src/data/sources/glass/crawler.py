import hashlib
import logging
import time
import asyncio

import aiohttp
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


class _CrawlerMixin:
    """Remote listing/discovery for GLASS directory trees."""

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

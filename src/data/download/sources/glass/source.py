import logging
import os
import asyncio
import aiohttp
import aiofiles

import requests
from urllib.parse import urlparse

from src.data.download.sources.base import BaseDataSource

from .crawler import _CrawlerMixin
from .session import _SessionMixin

logger = logging.getLogger(__name__)


class GlassLSTDataSource(_CrawlerMixin, _SessionMixin, BaseDataSource):
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

        # Don't store the session directly in the instance
        # Just keep a flag to check if we need selenium
        self.requires_selenium = False

        # Define schema types for Parquet consistency
        self.schema_dtypes = {
            'year': 'int32',            # Explicitly use int32 for year
            'day_of_year': 'int32',     # Explicitly use int32 for day_of_year
            'timestamp_precision': 'ms', # Use millisecond precision for timestamps
            'file_size': 'int64',       # Consistent int64 for file sizes
            'download_status': 'string', # Consistent string type
            'status_category': 'string'  # Consistent string type
        }

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

    async def download_async(self, file_url: str, output_path: str, session: aiohttp.ClientSession = None) -> None:
        """
        Asynchronous download method with respectful rate limiting.

        Args:
            file_url: URL to download from
            output_path: Local path to save the file
            session: Optional aiohttp session for connection reuse
        """
        # Add a small delay to be respectful to the server
        await asyncio.sleep(0.5)  # 500ms delay between requests

        # Use provided session or create a new one
        if session is None:
            connector = aiohttp.TCPConnector(limit=5, limit_per_host=2)  # Conservative limits
            timeout = aiohttp.ClientTimeout(total=300, connect=30)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                await self._download_with_session(session, file_url, output_path)
        else:
            await self._download_with_session(session, file_url, output_path)

    async def _download_with_session(self, session: aiohttp.ClientSession, file_url: str, output_path: str):
        """Helper method to download with a given session."""
        logger = logging.getLogger(__name__)

        try:
            # Add retry logic for robustness
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with session.get(file_url) as response:
                        response.raise_for_status()

                        # Ensure directory exists
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)

                        # Write file asynchronously
                        async with aiofiles.open(output_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)

                        logger.debug(f"Successfully downloaded {os.path.basename(output_path)}")
                        return  # Success, exit retry loop

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Progressive backoff: 2s, 4s, 6s
                        logger.warning(f"Download attempt {attempt + 1} failed for {file_url}, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Failed to download {file_url} after {max_retries} attempts: {e}")
                        raise

        except Exception as e:
            logger.error(f"Error downloading {file_url}: {e}")
            # Clean up partial file if it exists
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            raise

    def gcs_upload_path(self, base_url: str, relative_path: str) -> str:
        """Generate destination path for the file (legacy method name)."""
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

            # Return with explicit int32 type specification to ensure schema consistency
            return {
                'year': int(year),  # Ensure int type (will be cast to int32 in index)
                'day': int(day)     # Ensure int type (will be cast to int32 in index)
            }
        except (IndexError, ValueError, StopIteration):
            return None


NAMES = ("glass_modis", "glass_avhrr")


def from_config(dataset_name, config, *, base_url, file_extensions, output_path, source_config, **kwargs):
    """Build a GlassLSTDataSource from the shared config-extraction the factory does."""
    logger.info("Creating GLASS LST data source")
    return GlassLSTDataSource(
        base_url=base_url,
        file_extensions=file_extensions,
        output_path=output_path
    )

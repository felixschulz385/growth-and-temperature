"""
Data source for ACAG (Atmospheric Composition Analysis Group) PM2.5 data.

Downloads annual global PM2.5 estimates from the ACAG Box shared folder
hosted at Washington University in St. Louis (WashU).

Data: https://wustl.app.box.com/s/y143mciw7jz7ft2qe3hccjw65m3xe8f2/folder/327753085334
"""
import os
import re
import time
import hashlib
import logging
import asyncio
import aiohttp
import aiofiles
from typing import List, Tuple, Dict, Any, Optional

import requests

from gnt.data.download.sources.base import BaseDataSource

logger = logging.getLogger(__name__)


class ACAGDataSource(BaseDataSource):
    """
    Data source for ACAG PM2.5 annual global surface concentration estimates.

    Uses the Box content API with a shared-link bearer token to enumerate
    folder/file items without requiring an OAuth access token.

    References
    ----------
    Hammer et al. (2020) – GBD-MAPS Working Group, V6.GL.02.04
    https://sites.wustl.edu/acag/datasets/surface-pm2-5/
    """

    DATA_SOURCE_NAME = "acag"

    # ------------------------------------------------------------------ #
    # Box API constants
    # ------------------------------------------------------------------ #
    BOX_API_BASE = "https://api.box.com/2.0"
    # Public shared-link URL (no trailing slash)
    SHARED_LINK_URL = "https://wustl.app.box.com/s/y143mciw7jz7ft2qe3hccjw65m3xe8f2"
    # Root folder ID exposed by the shared link
    ROOT_FOLDER_ID = "327753085334"

    # How many items to request per page from the Box API
    PAGE_LIMIT = 1000

    # File extensions we are interested in
    DEFAULT_EXTENSIONS = [".nc4", ".nc", ".tif", ".tiff", ".h5"]

    # ------------------------------------------------------------------ #

    def __init__(
        self,
        base_url: str = None,
        file_extensions: List[str] = None,
        output_path: str = None,
        root_folder_id: str = None,
        shared_link_url: str = None,
    ):
        """
        Initialise the ACAG data source.

        Parameters
        ----------
        base_url:
            Ignored; kept for API compatibility with the factory.
        file_extensions:
            List of file suffixes to include (default: .nc4, .nc, .tif, .tiff, .h5).
        output_path:
            Sub-path under the HPC data root where files are stored
            (default: ``acag/pm25``).
        root_folder_id:
            Box folder ID at the top of the shared tree.
            Defaults to :attr:`ROOT_FOLDER_ID`.
        shared_link_url:
            Full Box shared-link URL.
            Defaults to :attr:`SHARED_LINK_URL`.
        """
        self.file_extensions = file_extensions or self.DEFAULT_EXTENSIONS
        self.data_path = output_path or "acag/pm25"
        self.root_folder_id = root_folder_id or self.ROOT_FOLDER_ID
        self.shared_link_url = shared_link_url or self.SHARED_LINK_URL

        # Schema dtypes for Parquet consistency (matches other sources)
        self.schema_dtypes = {
            "year": "int32",
            "day_of_year": "int32",
            "timestamp_precision": "ms",
            "file_size": "int64",
            "download_status": "string",
            "status_category": "string",
        }

        # In-memory cache: list of (relative_path, file_id) tuples
        self._file_cache: Optional[List[Tuple[str, str]]] = None
        self._cache_timestamp: Optional[float] = None
        self._cache_ttl = 3600  # seconds

        logger.info(
            "Initialised ACAG data source – root folder %s, output path: %s",
            self.root_folder_id,
            self.data_path,
        )

    # ------------------------------------------------------------------ #
    # Internal Box API helpers
    # ------------------------------------------------------------------ #

    def _box_headers(self) -> Dict[str, str]:
        """Return HTTP headers for anonymous Box API access via shared link."""
        return {
            "BoxApi": f"shared_link={self.shared_link_url}",
            # Box requires the Authorization header even for public links;
            # an empty bearer token is accepted for purely public items.
            "Authorization": "Bearer ",
        }

    def _list_folder(self, folder_id: str) -> List[Dict[str, Any]]:
        """
        Return all items (files and sub-folders) inside a Box folder, handling
        pagination automatically.

        Parameters
        ----------
        folder_id:
            Box folder ID to list.

        Returns
        -------
        List of item dicts as returned by the Box Items API.
        """
        items: List[Dict[str, Any]] = []
        offset = 0
        headers = self._box_headers()

        while True:
            url = (
                f"{self.BOX_API_BASE}/folders/{folder_id}/items"
                f"?limit={self.PAGE_LIMIT}&offset={offset}"
                f"&fields=id,name,type,size,modified_at"
            )
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                logger.error("Box API error listing folder %s: %s", folder_id, exc)
                break
            except ValueError as exc:
                logger.error("Box API returned non-JSON for folder %s: %s", folder_id, exc)
                break

            batch = data.get("entries", [])
            items.extend(batch)

            total = data.get("total_count", 0)
            offset += len(batch)

            if offset >= total or not batch:
                break

        return items

    def _walk_folder(
        self, folder_id: str, prefix: str = ""
    ) -> List[Tuple[str, str]]:
        """
        Recursively walk a Box folder tree and collect all matching files.

        Parameters
        ----------
        folder_id:
            Starting Box folder ID.
        prefix:
            Path prefix accumulated from parent folders.

        Returns
        -------
        List of ``(relative_path, file_id)`` tuples for every matching file.
        """
        results: List[Tuple[str, str]] = []
        items = self._list_folder(folder_id)

        for item in items:
            item_type = item.get("type")
            item_name = item.get("name", "")
            item_id = item.get("id", "")

            rel_path = f"{prefix}/{item_name}" if prefix else item_name

            if item_type == "folder":
                # Recurse into sub-folder
                logger.debug("Descending into Box sub-folder: %s (id=%s)", rel_path, item_id)
                results.extend(self._walk_folder(item_id, prefix=rel_path))

            elif item_type == "file":
                # Check extension
                if any(item_name.lower().endswith(ext.lower()) for ext in self.file_extensions):
                    results.append((rel_path, item_id))
                else:
                    logger.debug("Skipping file with unsupported extension: %s", item_name)

        return results

    def _get_all_files(self) -> List[Tuple[str, str]]:
        """
        Return a cached list of ``(relative_path, file_id)`` for all matching
        files reachable from the root shared folder.
        """
        now = time.time()
        if (
            self._file_cache is not None
            and self._cache_timestamp is not None
            and now - self._cache_timestamp < self._cache_ttl
        ):
            logger.debug("Using cached ACAG file list (%d entries)", len(self._file_cache))
            return self._file_cache

        logger.info("Scanning Box shared folder %s for ACAG files …", self.root_folder_id)
        files = self._walk_folder(self.root_folder_id)
        logger.info("Found %d matching files in ACAG Box folder", len(files))

        self._file_cache = files
        self._cache_timestamp = now
        return files

    def _file_download_url(self, file_id: str) -> str:
        """
        Return the direct-download URL for a Box file ID.

        The Box content endpoint returns a 302 redirect to a pre-signed CDN
        URL; ``requests`` follows the redirect automatically.
        """
        return f"{self.BOX_API_BASE}/files/{file_id}/content"

    # ------------------------------------------------------------------ #
    # BaseDataSource interface
    # ------------------------------------------------------------------ #

    def list_remote_files(self, entrypoint: dict = None) -> List[Tuple[str, str]]:
        """
        List files available in the ACAG shared Box folder.

        Parameters
        ----------
        entrypoint:
            Optional ``{"year": <int>}`` dict to restrict the listing to files
            whose filename contains the given four-digit year.

        Returns
        -------
        List of ``(relative_path, source_url)`` tuples where ``source_url``
        is a Box API content URL that yields the file bytes (with redirect).
        """
        all_files = self._get_all_files()
        results = []

        for rel_path, file_id in all_files:
            if entrypoint:
                year_filter = entrypoint.get("year")
                if year_filter is not None:
                    filename = os.path.basename(rel_path)
                    file_year = self._extract_year(filename)
                    if file_year != int(year_filter):
                        continue

            source_url = self._file_download_url(file_id)
            results.append((rel_path, source_url))

        logger.info(
            "list_remote_files returned %d files (entrypoint=%s)",
            len(results),
            entrypoint,
        )
        return results

    def local_path(self, relative_path: str) -> str:
        """Return a local filesystem path for a given relative path."""
        return os.path.join("data", self.DATA_SOURCE_NAME, relative_path)

    def download(
        self,
        source_url: str,
        output_path: str,
        session: requests.Session = None,
    ) -> None:
        """
        Download a single file from Box to *output_path*.

        Parameters
        ----------
        source_url:
            Box API content URL (``/files/<id>/content``).
        output_path:
            Local filesystem path to write the file to.
        session:
            Optional ``requests.Session`` for connection reuse.
        """
        s = session or requests.Session()
        headers = self._box_headers()

        try:
            logger.info("Downloading %s", os.path.basename(output_path))
            resp = s.get(source_url, headers=headers, stream=True, timeout=300)
            resp.raise_for_status()

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)

            logger.info("Saved %s", output_path)

        except requests.RequestException as exc:
            logger.error("Download failed for %s: %s", source_url, exc)
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            raise

    async def download_async(
        self,
        source_url: str,
        output_path: str,
        session: aiohttp.ClientSession = None,
    ) -> None:
        """Asynchronous download with retry logic."""
        await asyncio.sleep(0.2)  # polite rate-limiting

        headers = self._box_headers()

        async def _do_download(sess: aiohttp.ClientSession):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with sess.get(source_url, headers=headers) as resp:
                        resp.raise_for_status()
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        async with aiofiles.open(output_path, "wb") as fh:
                            async for chunk in resp.content.iter_chunked(8192):
                                await fh.write(chunk)
                        return  # success
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    if attempt < max_retries - 1:
                        wait = (attempt + 1) * 2
                        logger.warning(
                            "Attempt %d failed for %s, retrying in %ds: %s",
                            attempt + 1, source_url, wait, exc,
                        )
                        await asyncio.sleep(wait)
                    else:
                        logger.error(
                            "All %d attempts failed for %s: %s",
                            max_retries, source_url, exc,
                        )
                        raise

        if session is None:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=600, connect=60)
            async with aiohttp.ClientSession(
                connector=connector, timeout=timeout
            ) as sess:
                await _do_download(sess)
        else:
            await _do_download(session)

    def gcs_upload_path(self, base_url: str, relative_path: str) -> str:
        """Return the destination path (relative to the HPC data root)."""
        return f"{self.data_path}/{relative_path}"

    def filename_to_entrypoint(self, relative_path: str) -> Optional[Dict[str, Any]]:
        """
        Derive an entrypoint dict from a file path.

        For ACAG annual data the relevant axis is *year*.
        """
        filename = os.path.basename(relative_path)
        year = self._extract_year(filename)
        if year is not None:
            return {"year": int(year), "day": 1}
        return None

    def get_all_entrypoints(self) -> List[Dict[str, Any]]:
        """Return one entrypoint per year found in the shared folder."""
        all_files = self._get_all_files()
        years: set = set()

        for rel_path, _ in all_files:
            year = self._extract_year(os.path.basename(rel_path))
            if year is not None:
                years.add(year)

        entrypoints = [{"year": int(y), "day": 1} for y in sorted(years)]
        logger.info(
            "ACAG entrypoints: %d years – %s",
            len(entrypoints),
            sorted(years),
        )
        return entrypoints

    def get_authenticated_session(self) -> requests.Session:
        """Return a requests.Session pre-loaded with Box shared-link headers."""
        s = requests.Session()
        s.headers.update(self._box_headers())
        return s

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_year(filename: str) -> Optional[int]:
        """
        Extract a four-digit year from a filename.

        Tries ``_YYYY``, ``.YYYY.``, and a bare four-digit run.
        Returns the first match that falls in [1990, 2040], or None.
        """
        for pattern in (r"[._\-](\d{4})[._\-]", r"(\d{4})"):
            for match in re.finditer(pattern, filename):
                year = int(match.group(1))
                if 1990 <= year <= 2040:
                    return year
        return None

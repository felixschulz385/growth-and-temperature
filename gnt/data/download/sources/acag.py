"""
Data source for ACAG (Atmospheric Composition Analysis Group) PM2.5 data.

Downloads annual global PM2.5 estimates from the ACAG Box shared folder
hosted at Washington University in St. Louis (WashU).

Data: https://wustl.app.box.com/s/y143mciw7jz7ft2qe3hccjw65m3xe8f2/folder/327763146804

The file inventory is hardcoded from the Box shared-folder HTML (V6.GL.02.04 > EU > Annual).
Downloads use Box's public shared-link download endpoint – no API token required.
"""
import os
import re
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

    The file inventory is a hardcoded list derived from the Box shared-folder
    HTML pages. Downloads use Box's public shared-link download URL.

    References
    ----------
    Hammer et al. (2020) – GBD-MAPS Working Group, V6.GL.02.04
    https://sites.wustl.edu/acag/datasets/surface-pm2-5/
    """

    DATA_SOURCE_NAME = "acag"

    # Public shared-link URL (no trailing slash)
    SHARED_LINK_URL = "https://wustl.app.box.com/s/y143mciw7jz7ft2qe3hccjw65m3xe8f2"
    # Shared-link name token (the alphanumeric slug in the URL)
    SHARED_NAME = "y143mciw7jz7ft2qe3hccjw65m3xe8f2"

    # ------------------------------------------------------------------ #
    # Hardcoded file inventory
    # Sourced from Box shared-folder HTML: V6.GL.02.04 > EU > Annual
    # Each entry: (relative_path, box_file_id)
    # ------------------------------------------------------------------ #
    KNOWN_FILES: List[Tuple[str, str]] = [
        # Global files mirror the EU inventory with GL prefix
        ("GL/Annual/V6GL02.04.CNNPM25.GL.202301-202312.nc", "1904197590429"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.202201-202212.nc", "1904188293336"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.202101-202112.nc", "1904194844985"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.202001-202012.nc", "1904199632848"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.201901-201912.nc", "1904190302370"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.201801-201812.nc", "1904195082233"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.201701-201712.nc", "1904185892742"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.201601-201612.nc", "1904190764908"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.201501-201512.nc", "1904198007631"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.201401-201412.nc", "1904191060231"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.201301-201312.nc", "1904192466892"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.201201-201212.nc", "1904186701348"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.201101-201112.nc", "1904188948328"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.201001-201012.nc", "1904198202419"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.200901-200912.nc", "1904198860116"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.200801-200812.nc", "1904186910032"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.200701-200712.nc", "1904198384848"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.200601-200612.nc", "1904187236528"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.200501-200512.nc", "1904203088683"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.200401-200412.nc", "1904186071064"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.200301-200312.nc", "1904186044649"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.200201-200212.nc", "1904186910887"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.200101-200112.nc", "1904187033101"),
        ("GL/Annual/V6GL02.04.CNNPM25.EU.200001-200012.nc", "1904252406135"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.199901-199912.nc", "1904185160609"),
        ("GL/Annual/V6GL02.04.CNNPM25.GL.199801-199812.nc", "1904192358328"),
    ]

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
            Ignored; the hardcoded inventory already contains only .nc files.
        output_path:
            Sub-path under the HPC data root where files are stored
            (default: ``acag/pm25``).
        root_folder_id:
            Ignored; kept for API compatibility.
        shared_link_url:
            Full Box shared-link URL. Defaults to :attr:`SHARED_LINK_URL`.
        """
        self.data_path = output_path or "acag/pm25"
        self.shared_link_url = shared_link_url or self.SHARED_LINK_URL
        self.shared_name = self.shared_link_url.rstrip("/").split("/")[-1]

        # Schema dtypes for Parquet consistency (matches other sources)
        self.schema_dtypes = {
            "year": "int32",
            "day_of_year": "int32",
            "timestamp_precision": "ms",
            "file_size": "int64",
            "download_status": "string",
            "status_category": "string",
        }

        logger.info(
            "Initialised ACAG data source – %d known files, output path: %s",
            len(self.KNOWN_FILES),
            self.data_path,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _browser_headers() -> Dict[str, str]:
        """Minimal browser-like headers for Box download requests."""
        return {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def _file_download_url(self, file_id: str) -> str:
        """
        Return the shared-link download URL for a Box file ID.

        Box responds with a 302 redirect to a pre-signed CDN URL which
        ``requests`` follows automatically.
        """
        return (
            f"https://wustl.app.box.com/index.php"
            f"?rm=box_download_shared_file"
            f"&shared_name={self.shared_name}"
            f"&file_id=f_{file_id}"
        )

    def _get_all_files(self) -> List[Tuple[str, str]]:
        """Return the hardcoded list of ``(relative_path, file_id)`` tuples."""
        return list(self.KNOWN_FILES)

    def get_file_hash(self, file_url: str) -> str:
        """Return a stable unique hash for a file URL."""
        return hashlib.md5(file_url.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ #
    # BaseDataSource interface
    # ------------------------------------------------------------------ #

    def list_remote_files(self, entrypoint: dict = None) -> List[Tuple[str, str]]:
        """
        List files in the ACAG inventory.

        Parameters
        ----------
        entrypoint:
            Optional ``{"year": <int>}`` dict to restrict the listing to files
            whose filename contains the given four-digit year.

        Returns
        -------
        List of ``(relative_path, source_url)`` tuples.
        """
        results = []

        for rel_path, file_id in self._get_all_files():
            if entrypoint:
                year_filter = entrypoint.get("year")
                if year_filter is not None:
                    file_year = self._extract_year(os.path.basename(rel_path))
                    if file_year != int(year_filter):
                        continue

            results.append((rel_path, self._file_download_url(file_id)))

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
            Box shared-link download URL.
        output_path:
            Local filesystem path to write the file to.
        session:
            Optional ``requests.Session`` for connection reuse.
        """
        s = session or requests.Session()

        try:
            logger.info("Downloading %s", os.path.basename(output_path))
            resp = s.get(
                source_url, headers=self._browser_headers(), stream=True, timeout=300
            )
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

        headers = self._browser_headers()

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
        """Return one entrypoint per year present in the known file list."""
        years: set = set()

        for rel_path, _ in self.KNOWN_FILES:
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
        """Return a requests.Session pre-loaded with browser-like headers."""
        s = requests.Session()
        s.headers.update(self._browser_headers())
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

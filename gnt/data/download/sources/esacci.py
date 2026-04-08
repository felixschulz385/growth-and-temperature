"""
Data source for ESA CCI Land Cover data via the Copernicus CDS API.

Each year is treated as one "file" in the download index and retrieved with:

    cdsapi.Client().retrieve("satellite-land-cover", request, target)

Authentication
--------------
CDS API credentials must be stored in ``$HOME/.cdsapirc`` (or the path
pointed to by ``$CDSAPI_RC``) before any download is attempted::

    url: https://cds.climate.copernicus.eu/api
    key: <PERSONAL-ACCESS-TOKEN>

You must also accept the dataset Terms of Use once on the CDS website
before the API will honour requests.

Dataset versions
----------------
* 1992–2015 → ``v2_0_7cds``
* 2016–2022 → ``v2_1_1``

Both versions are included by default; the CDS service silently returns
the appropriate one for each year.
"""
import os
import re
import time
import logging
import hashlib
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlencode, urlparse, parse_qs

import requests

from gnt.data.download.sources.base import BaseDataSource

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cdsapi_url(year: int, versions: List[str]) -> str:
    """
    Encode a CDS API request as a virtual URI recognised by :meth:`download`.

    Format::

        cdsapi://satellite-land-cover?year=YYYY&version=v2_0_7cds&version=v2_1_1
    """
    params = [("year", str(year))]
    for v in versions:
        params.append(("version", v))
    return "cdsapi://satellite-land-cover?" + urlencode(params)


def _parse_cdsapi_url(url: str) -> Tuple[str, Dict[str, Any]]:
    """
    Decode a virtual CDS URI back into (dataset, request_dict).

    Returns:

        dataset  – e.g. ``"satellite-land-cover"``
        request  – dict suitable for ``cdsapi.Client().retrieve(dataset, request, …)``
    """
    parsed = urlparse(url)
    dataset = parsed.netloc  # e.g. "satellite-land-cover"
    qs = parse_qs(parsed.query)

    years = qs.get("year", [])
    versions = qs.get("version", [])

    request: Dict[str, Any] = {"variable": "all"}
    if years:
        request["year"] = years
    if versions:
        request["version"] = versions

    return dataset, request


# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------

class ESACCIDataSource(BaseDataSource):
    """
    Data source for ESA CCI Land Cover annual composites (NetCDF).

    Each calendar year is a separate index entry downloaded via the CDS API.
    No HTTP session is used; authentication is handled by ``cdsapi`` reading
    ``~/.cdsapirc``.
    """

    DATA_SOURCE_NAME = "esacci"

    DATASET = "satellite-land-cover"

    # Version map: which CDS versions to request
    DEFAULT_VERSIONS = ["v2_0_7cds", "v2_1_1"]

    # Year range covered by the ESACCI-LC product on CDS
    DEFAULT_YEAR_RANGE = (1992, 2022)

    def __init__(
        self,
        base_url: str = None,          # not used; kept for factory compatibility
        file_extensions: List[str] = None,  # not used; output is always .nc
        output_path: str = None,
        year_range: Tuple[int, int] = None,
        versions: List[str] = None,
        cdsapi_rc: str = None,
    ):
        """
        Parameters
        ----------
        output_path:
            Sub-path under the HPC data root (default: ``"esacci/landcover"``).
        year_range:
            ``(start_year, end_year)`` inclusive (default: 1992–2022).
        versions:
            List of CDS version tags to include in every request.
            Default: ``["v2_0_7cds", "v2_1_1"]``.
        cdsapi_rc:
            Path to a custom ``.cdsapirc`` config file.  If ``None`` cdsapi
            uses its own lookup chain (``$CDSAPI_RC`` → ``~/.cdsapirc``).
        """
        self.data_path = output_path or "esacci/landcover"
        self.year_start, self.year_end = year_range or self.DEFAULT_YEAR_RANGE
        self.versions = versions or self.DEFAULT_VERSIONS
        self.cdsapi_rc = cdsapi_rc

        # Schema dtypes (Parquet consistency)
        self.schema_dtypes = {
            "year": "int32",
            "day_of_year": "int32",
            "timestamp_precision": "ms",
            "file_size": "int64",
            "download_status": "string",
            "status_category": "string",
        }

        logger.info(
            "Initialised ESACCIDataSource  path=%s  years=%d–%d  versions=%s",
            self.data_path, self.year_start, self.year_end, self.versions,
        )

    # ------------------------------------------------------------------
    # BaseDataSource interface
    # ------------------------------------------------------------------

    def list_remote_files(self, entrypoint: dict = None) -> List[Tuple[str, str]]:
        """
        Return one ``(relative_path, source_url)`` entry per year.

        The *source_url* uses the virtual ``cdsapi://`` scheme; it is
        consumed by :meth:`download` and never opened directly with HTTP.

        Parameters
        ----------
        entrypoint:
            Optional ``{"year": <int>}`` to restrict to a single year.
        """
        if entrypoint and "year" in entrypoint:
            years = [int(entrypoint["year"])]
        else:
            years = list(range(self.year_start, self.year_end + 1))

        results = []
        for year in years:
            rel_path = f"{year}/ESACCI-LC-L4-LCCS-Map-300m-P1Y-{year}-v2.0.7.nc"
            source_url = _cdsapi_url(year, self.versions)
            results.append((rel_path, source_url))

        logger.info("list_remote_files: %d entries (entrypoint=%s)", len(results), entrypoint)
        return results

    def local_path(self, relative_path: str) -> str:
        """Return a local filesystem path for the given relative path."""
        return os.path.join("data", self.DATA_SOURCE_NAME, relative_path)

    def download(
        self,
        source_url: str,
        output_path: str,
        session: requests.Session = None,  # unused – kept for interface compat
    ) -> None:
        """
        Download one year of ESA CCI Land Cover data via the CDS API.

        Parameters
        ----------
        source_url:
            A ``cdsapi://`` URI produced by :func:`_cdsapi_url`.
        output_path:
            Local filesystem path to write the NetCDF file to.
        session:
            Ignored (CDS API uses its own authentication).

        Raises
        ------
        ImportError
            If ``cdsapi`` is not installed.
        RuntimeError
            If the CDS API request fails.
        """
        try:
            import cdsapi
        except ImportError as exc:
            raise ImportError(
                "cdsapi is required to download ESA CCI data.  "
                "Install it with: pip install cdsapi"
            ) from exc

        dataset, request = _parse_cdsapi_url(source_url)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        logger.info(
            "Requesting CDS dataset '%s'  year=%s  → %s",
            dataset, request.get("year"), output_path,
        )

        try:
            kwargs: Dict[str, Any] = {}
            if self.cdsapi_rc:
                kwargs["rc"] = self.cdsapi_rc

            client = cdsapi.Client(**kwargs)
            client.retrieve(dataset, request, output_path)

            logger.info("CDS download complete: %s", output_path)

        except Exception as exc:
            # Clean up any partial file
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            raise RuntimeError(f"CDS API request failed for {source_url}: {exc}") from exc

    def get_authenticated_session(self) -> None:
        """
        CDS API auth is handled by cdsapi reading ``~/.cdsapirc``.
        Returns None so the async downloader falls back to :meth:`download`.
        """
        return None

    def gcs_upload_path(self, base_url: str, relative_path: str) -> str:
        """Return the destination path relative to the HPC data root."""
        return f"{self.data_path}/{relative_path}"

    def get_file_hash(self, file_url: str) -> str:
        """Return a stable hash for a file URL.

        The unified index requires a ``get_file_hash`` method on every data
        source.  For ESACCI the URL string (which may be a ``cdsapi://``
        virtual URI) is hashed using MD5, matching the pattern used in other
        sources.
        """
        return hashlib.md5(file_url.encode("utf-8")).hexdigest()

    def filename_to_entrypoint(self, relative_path: str) -> Optional[Dict[str, Any]]:
        """Derive a year-based entrypoint from a file path."""
        year = self._extract_year(os.path.basename(relative_path))
        if year is not None:
            return {"year": int(year), "day": 1}
        return None

    def get_all_entrypoints(self) -> List[Dict[str, Any]]:
        """Return one entrypoint per year in the configured range."""
        entrypoints = [
            {"year": y, "day": 1}
            for y in range(self.year_start, self.year_end + 1)
        ]
        logger.info(
            "ESACCI entrypoints: %d years (%d–%d)",
            len(entrypoints), self.year_start, self.year_end,
        )
        return entrypoints

    # ------------------------------------------------------------------
    # Async shim – CDS API is synchronous; run it in an executor
    # ------------------------------------------------------------------

    async def download_async(
        self,
        source_url: str,
        output_path: str,
        session=None,
    ) -> None:
        """
        Async wrapper: runs the synchronous CDS API call in a thread-pool
        executor so it does not block the event loop.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.download, source_url, output_path, None)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_year(filename: str) -> Optional[int]:
        """Return the four-digit year found in *filename*, or None."""
        for pattern in (r"[._\-](\d{4})[._\-]", r"(\d{4})"):
            for m in re.finditer(pattern, filename):
                y = int(m.group(1))
                if 1990 <= y <= 2040:
                    return y
        return None

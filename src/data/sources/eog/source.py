"""EOG (DMSP/VIIRS/DVNL) nighttime lights: fetch + prepare.

PREPARE reprojects straight from each year's raw fetched file to the tiled
output; there is no intermediate annual zarr and no separate GRID step.

**Real bug fixed here, not silently ported** (unchanged from the original
migration): the old `EOGPreprocessor._generate_annual_targets` called
`self._extract_year_from_path(...)` and `self._select_best_file_for_year(...)`
-- neither method existed anywhere in that class (verified by direct
execution, see tests/data/preprocess/sources/test_characterization_eog.py),
so every call to `get_preprocessing_targets("annual", ...)` raised
`AttributeError`, silently caught, always returning `[]`. **EOG's annual/
PREPARE stage never produced a target.** This class implements both methods
for real: year extraction uses the same generic 4-digit-year regex every
other source in this codebase already uses (acag/esacci/ntl_harm's
`_extract_year`), with a DMSP-specific fallback for its undelimited
satellite+year filenames; best-file selection prefers extensions in the
order the source's own `file_extensions` config already declares (default
`.tif > .tgz > .tar.gz > .gz`).

**Bug fixed later, not part of the original migration**: `_derive_source_type`
used to guess the DMSP/VIIRS-annual/DVNL variant from substrings of
`cfg.data_path`/`base_url` (in that order, with a silent `viirs_dvnl` default
if nothing matched), rather than from `cfg.source_id` -- the literal
`sources.<id>:` config-block key this instance was actually built from
(`eog_dmsp`/`eog_viirs`/`eog_dvnl`, per `_build()`'s alias-block lookup in
`src/cli/data/handlers.py`). It happened to agree with the alias for
every config committed in `orchestration/configs/data.yaml`, but the two were
never actually pinned together -- editing `data_path`/`base_url` without
touching the block key would have silently mis-set `source_type`, which
drives PREPARE's output variable name and its output filename/`family`.
Now derived from `cfg.source_id` directly, the same authoritative signal
`GlassSource.__init__` already uses for its own MODIS/AVHRR variant
(`data_source_kind`).

**VIIRS annual composites: hardcoded year range, per-year discovery, not a
full recursive crawl.** `eogdata.mines.edu/nighttime_light/annual/v21/` lists
one subdirectory per year (`2012/`, `2013/`, ...); each year's subdirectory
mixes that year's canonical composite with intermediate/rolling reprocessing
periods (e.g. a `201204-201303` entry alongside `201204-201212`) and, per
file, several product variants
(`average`/`average_masked`/`cf_cvg`/`cvg`/`lit_mask`/`maximum`/`median`/
`median_masked`/`minimum`). The old whole-directory recursive crawl
(`_CrawlerMixin.list_remote_files()`, still used for DMSP) had no variant
filter at all -- it would have queued every variant of every period for
download, keyed only on `file_extensions`. `VIIRS_YEAR_RANGE` (2012-2021)
is hardcoded rather than discovered (`get_all_entrypoints()`), each year
mapped, per `data summary`/`data fetch`, to the file(s) whose period ends
in December of that year (excludes the rolling/intermediate periods) for
each variant in `VIIRS_VARIANTS` -- `average_masked` (the masked mean, the
standard used in nightlights economics literature), plus `median_masked`
and `cf_cvg`, which PREPARE lands as parallel columns
(`viirs_annual` / `viirs_annual_median` / `viirs_annual_cf_cvg`, each
resampled by its own method -- `VIIRS_RESAMPLING`). The *exact* URL (with its
unpredictable `_c<timestamp>` processing-code suffix) is still discovered
live, not templated -- `has_entrypoints=True` for this source_type routes
through the same per-entrypoint listing cache
(`src.data.common.fetch.catalog.required_files()`,
`_status/entrypoints/<year>.json`) esacci/acag/ntl_harm/glass already use,
so only the *first* `data summary`/`data fetch` after this file range is
adopted needs a live (authenticated) crawl per year; every run after that
reads the cached listing, unless explicitly refreshed.

**DMSP and DVNL are disabled** (`orchestration/configs/data.yaml`, commented
out like `berman_mining`) -- only VIIRS annual composites are wired into the
pipeline currently. Their code paths (the plain recursive crawl, `source_type`
derivation) are left intact, not deleted, so re-enabling either later is a
config uncomment, not a rewrite.

**Credentials**: `src/data/sources/eog/credentials.py::load_eog_credentials()`
-- a git-ignored `orchestration/secrets/eog.credentials.json`
(`{"username": ..., "password": ...}`), falling back to
`EOG_USERNAME`/`EOG_PASSWORD` environment variables if that file doesn't
exist. Replaces a bare `os.environ.get(...)` read that made every
credentialed EOG operation (including a live VIIRS listing crawl, above)
depend on the calling shell already having those two variables exported.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import xarray as xr

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.eog.crawler import _CrawlerMixin
from src.data.sources.eog.credentials import load_eog_credentials
from src.data.sources.eog.session import _SessionMixin
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection
from src.data.sources import verify

logger = logging.getLogger(__name__)


class EogSource(_CrawlerMixin, _SessionMixin, DataSource):
    """Earth Observation Group nighttime lights (DMSP-OLS, VIIRS annual, DVNL).

    FETCH   -- Selenium-authenticated crawl + download of the configured
               `base_url` archive (credentials via EOG_USERNAME/EOG_PASSWORD).
    PREPARE -- reproject every year's raw fetched file directly onto the
               canonical geobox, tile by tile; radiance field, so resampling
               defaults to area-weighted "sum" (docs/design/04-ingest.md
               §1), not `run_tiled_prepare`'s own "nearest" default.
    """

    ID = "eog"
    ALIASES = ("eog_dmsp", "eog_viirs", "eog_dvnl")
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE)
    DEFAULT_TRANSFER_MODE = "auto"

    DATA_SOURCE_NAME = "eog"  # matches old EOGDataSource: literally "eog", not per-alias
    STATIC_ENTRYPOINTS = True  # get_all_entrypoints() below is VIIRS_YEAR_RANGE-derived, no network call

    EOG_LOGIN_URL = "https://eogdata.mines.edu/nighttime_light/login/"

    #: Bumped whenever the raw-getter/reprojection logic here changes in a
    #: way that must invalidate every already-`complete` tile's status and
    #: force a full reprocess (`run_tiled_prepare`'s `processing_version`).
    PROCESSING_VERSION = "3-viirs-multivar"

    #: VIIRS annual-composite years worth fetching (module docstring) --
    #: hardcoded rather than discovered, since the directory mixes canonical
    #: composites with intermediate/rolling reprocessing periods with no
    #: reliable way to tell them apart except by which December they end in
    #: (see _viirs_annual_listing()).
    VIIRS_YEAR_RANGE = (2012, 2021)

    #: The VNL product variants fetched, each landing as its own column in
    #: the one `eog_viirs_annual` PREPARE output:
    #:   * `average_masked` -> `viirs_annual`          (the masked mean
    #:     composite -- background/fire/aurora-corrected; the standard used
    #:     in nightlights economics literature). Reprojected by area-weighted
    #:     `sum` (flux-conserving -- docs/design/04-ingest.md §1).
    #:   * `median_masked`  -> `viirs_annual_median`   (per-pixel masked
    #:     median radiance -- robust to transient bright nights). Reprojected
    #:     by `average`; a per-pixel diagnostic, NOT a flux ring sum.
    #:   * `cf_cvg`         -> `viirs_annual_cf_cvg`   (cloud-free coverage:
    #:     the count of cloud-free observations feeding each pixel's
    #:     composite -- an observation-density weight, not a day count).
    #:     Reprojected by `average`; stored as float32.
    VIIRS_VARIANTS = ("average_masked", "median_masked", "cf_cvg")

    #: Which VNL variant produces which output column (see VIIRS_VARIANTS).
    VIIRS_VARIANT_COLUMNS = {
        "average_masked": "viirs_annual",
        "median_masked": "viirs_annual_median",
        "cf_cvg": "viirs_annual_cf_cvg",
    }

    #: The primary variant -- a year without this file is not preparable
    #: (median/cf_cvg are NaN-filled if their file is missing, so every
    #: year's parquet part keeps the same 3-column schema).
    VIIRS_PRIMARY_VARIANT = "average_masked"

    #: Per-column resampling for the multi-variant `viirs_annual` output,
    #: threaded through `run_tiled_prepare` -> `process_tile_region` as a
    #: `{variable: method}` map (SpatialProcessor.resample_map_for).
    VIIRS_RESAMPLING = {
        "viirs_annual": "sum",
        "viirs_annual_median": "average",
        "viirs_annual_cf_cvg": "average",
    }

    #: Two live naming schemes, confirmed against eogdata.mines.edu/
    #: nighttime_light/annual/v21/<year>/: 2012 (the product's first,
    #: partial year -- VIIRS only launched in 2011/started this composite
    #: series in April 2012) is named with a month-range period, e.g.
    #: "VNL_v21_npp_201204-201212_global_vcmcfg_c202205302300.average_masked.dat.tif.gz";
    #: every year since (2013 on) drops the period entirely and uses a bare
    #: calendar year instead, e.g.
    #: "VNL_v21_npp_2013_global_vcmcfg_c202205302300.average_masked.dat.tif.gz"
    #: (also switching from "vcmcfg" to "vcmslcfg" partway through -- already
    #: covered by the `\w+` config-token wildcard below). Matched via two
    #: alternative groups: `start_year`/`start_month`/`end_year`/`end_month`
    #: for the period form, `year` for the bare-year form -- exactly one
    #: alternative's groups are populated per match.
    _VIIRS_FILENAME_RE = re.compile(
        r"VNL_v\d+_\w+_"
        r"(?:(?P<start_year>\d{4})(?P<start_month>\d{2})-(?P<end_year>\d{4})(?P<end_month>\d{2})|(?P<year>\d{4}))"
        r"_global_\w+_c\d+"
        r"\.(?P<variant>[a-z_]+)\.dat\.tif(?:\.gz)?$"
    )

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            # old EOGPreprocessor had no fallback here -- data_path/output_path was required.
            raise ValueError("'data_path' (or 'output_path') is required.")
        super().__init__(ctx, cfg)
        self.base_url: Optional[str] = cfg.raw.get("base_url")
        if not self.base_url:
            raise ValueError("'base_url' is required.")
        self.file_extensions: List[str] = cfg.raw.get("file_extensions") or [".tif", ".tgz", ".tar.gz", ".gz"]

        self._username, self._password = load_eog_credentials(cfg.raw.get("credentials_path"))
        if not self._username or not self._password:
            logger.warning(
                "EOG credentials not set (orchestration/secrets/eog.credentials.json or "
                "EOG_USERNAME/EOG_PASSWORD environment variables)"
            )
        self._driver = None
        self._download_dir = None
        self._is_logged_in = False

        self.source_type = self._derive_source_type()
        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix=f"eog_{self.source_type}_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)

        #: PREPARE resampling: VIIRS annual is the multi-variant output, so a
        #: per-column `{variable: method}` map (see VIIRS_RESAMPLING); DMSP/
        #: DVNL stay single-method. Either form is `verification`-style
        #: overridable via `resampling:` in the source's config block.
        default_resampling = self.VIIRS_RESAMPLING if self.source_type == "viirs_annual" else "sum"
        self.resampling = cfg.raw.get("resampling", default_resampling)

        #: Idle (no-progress) timeout for a single Selenium download, seconds
        #: -- see download_file(). `download.timeout` in config overrides it.
        self._download_stall_timeout_s = int(
            cfg.raw.get("download", {}).get("timeout", self.DOWNLOAD_STALL_TIMEOUT_S)
        )

        #: In-process cache for _viirs_annual_listing() -- populated by one
        #: crawl the first time any year's entrypoint is requested, reused
        #: for the rest (module docstring: avoids 10 separate logins to
        #: populate all 10 years' entrypoint caches in one run).
        self._viirs_listing_cache: Optional[Dict[int, List[Tuple[str, str]]]] = None

        from src.data.common import tiling

        self.tile_size = int(cfg.raw.get("tile_size", tiling.DEFAULT_TILE_SIZE))

    def _derive_source_type(self) -> str:
        """Which variant: derived from `cfg.source_id` -- the literal
        `sources.<id>:` config-block key this instance was built from
        (module docstring) -- mirroring `GlassSource.__init__`'s identical
        `cfg.source_id`-based derivation of its own MODIS/AVHRR variant.
        Raises rather than guessing from `data_path`/`base_url` content
        (the old behavior) if `source_id` doesn't name a known variant, so a
        misconfigured/renamed source_id fails loudly instead of silently
        mislabeling PREPARE's output variable/filename."""
        source_id = self.cfg.source_id.lower()
        if "dmsp" in source_id:
            return "dmsp"
        if "dvnl" in source_id:
            return "viirs_dvnl"
        if "viirs" in source_id:
            return "viirs_annual"
        raise ValueError(
            f"Cannot derive EOG source_type from source_id={self.cfg.source_id!r} -- "
            f"expected it to contain one of 'dmsp'/'dvnl'/'viirs' "
            f"(matching one of the registered aliases {EogSource.ALIASES})."
        )

    # ------------------------------------------------------------------
    # RemoteFileCatalog contract -- list_remote_files comes from
    # _CrawlerMixin, download/download_async from _SessionMixin-backed
    # EOGDataSource.download machinery ported below.
    # ------------------------------------------------------------------

    def local_path(self, relative_path: str) -> str:
        return os.path.join("data", self.DATA_SOURCE_NAME, relative_path)

    # get_file_hash: inherited from DataSource (src/data/sources/base.py).

    @staticmethod
    def _viirs_match_year(match: "re.Match") -> Optional[int]:
        """The canonical calendar year a `_VIIRS_FILENAME_RE` match names --
        `year` for the bare-year naming (2013 on), `end_year` (only if the
        period ends in December -- the canonical annual composite, not a
        rolling/intermediate one) for the older month-range naming (2012).
        `None` if neither applies (rolling period, wrong month, etc.)."""
        if match.group("year"):
            return int(match.group("year"))
        if match.group("end_month") == "12":
            return int(match.group("end_year"))
        return None

    def filename_to_entrypoint(self, relative_path: str) -> Optional[Dict[str, Any]]:
        """`None` for DMSP/DVNL (old EOGDataSource: entrypoints not used).
        VIIRS annual composites: the year is `_viirs_match_year()`'s reading
        of the filename, same regex `_viirs_annual_listing()` matches
        against -- lets `data summary` map an already-downloaded local file
        back to its year without needing a live crawl (`_summarize_fetch()`'s
        cached_required_files()-is-empty fallback)."""
        if self.source_type != "viirs_annual":
            return None
        match = self._VIIRS_FILENAME_RE.search(os.path.basename(relative_path))
        if not match:
            return None
        year = self._viirs_match_year(match)
        return {"year": year} if year is not None else None

    @property
    def has_entrypoints(self) -> bool:
        """VIIRS annual composites go through the per-year, cached listing
        below (module docstring) -- DMSP/DVNL (disabled sources) keep the
        plain whole-directory recursive crawl (`_CrawlerMixin`'s own
        `list_remote_files()`, unchanged), which needs no entrypoints."""
        return self.source_type == "viirs_annual"

    @property
    def RAW_LISTING_DEPTH(self) -> "int | None":
        """VIIRS annual composites land as a flat filename under the raw
        root (`_viirs_annual_listing()`'s `href` values have no
        subdirectory) -- depth 1. DMSP/DVNL's plain recursive crawler
        (`crawler.py`) allows up to 8 levels of real, variable directory
        nesting, so left unbounded (`None`) rather than guessed."""
        return 1 if self.source_type == "viirs_annual" else None

    def get_all_entrypoints(self) -> List[Dict[str, Any]]:
        if self.source_type != "viirs_annual":
            return []
        start, end = self.VIIRS_YEAR_RANGE
        return [{"year": year} for year in range(start, end + 1)]

    def list_remote_files(self, entrypoint: Optional[dict] = None):
        if self.source_type != "viirs_annual" or entrypoint is None:
            yield from super().list_remote_files(entrypoint)
            return
        year = entrypoint["year"]
        yield from self._viirs_annual_listing().get(year, [])

    def _viirs_annual_listing(self) -> Dict[int, List[Tuple[str, str]]]:
        """One Selenium page load per year subdirectory (`_CrawlerMixin.
        _list_single_directory()` -- non-recursive, no `file_extensions`
        filter of its own): `base_url` itself (`_list_single_directory()`
        called there) lists only the year folders (`2012/`, `2013/`, ...,
        confirmed live), each of which mixes the canonical one-per-calendar-
        year composite with intermediate/rolling reprocessing periods (e.g.
        a `201204-201303` entry alongside `201204-201212`) and, per file,
        several product variants (average/cf_cvg/median/...) -- filtered
        here to the `VIIRS_VARIANTS` we ingest (mean/median/cf_cvg), each
        entry whose period ends in December of its own folder's year. One
        (href, url) per (year, variant); a duplicate (year, variant) keeps
        the first. All year directories are crawled in one
        Selenium session/login and the combined result cached on `self` so
        every year in `get_all_entrypoints()` shares that one login instead
        of ten. Requests are throttled the same way `_CrawlerMixin.crawl()`
        throttles its own recursive page loads (`time.sleep(1 + random())`
        between requests), out of the same courtesy that crawler already
        extends."""
        if self._viirs_listing_cache is not None:
            return self._viirs_listing_cache

        listing: Dict[int, List[Tuple[str, str]]] = {}
        start, end = self.VIIRS_YEAR_RANGE
        base = self.base_url if self.base_url.endswith("/") else self.base_url + "/"
        # Reuse an already-open session (e.g. one _execute_fetch opened for
        # the whole run) rather than tearing it down in the finally below.
        opened_here = self._driver is None
        self._init_selenium_driver()
        try:
            for i, year in enumerate(range(start, end + 1)):
                if i > 0:
                    time.sleep(1 + random.random())
                year_url = urljoin(base, f"{year}/")
                entries = self._list_single_directory(year_url)
                logger.debug("EOG VIIRS %d: %s -- %d raw href(s)", year, year_url, len(entries))
                for href, full_url in entries:
                    match = self._VIIRS_FILENAME_RE.search(href)
                    if not match:
                        continue
                    if match.group("variant") not in self.VIIRS_VARIANTS:
                        continue
                    if self._viirs_match_year(match) != year:
                        continue
                    listing.setdefault(year, []).append((href, full_url))
        finally:
            if opened_here:
                self._close_selenium_driver()

        for year, matches in listing.items():
            seen: Dict[str, Tuple[str, str]] = {}
            for href, full_url in matches:
                variant = self._VIIRS_FILENAME_RE.search(href).group("variant")
                if variant in seen:
                    logger.warning(
                        "Multiple EOG VIIRS %s candidates for year %d: %s -- using %s",
                        variant, year, [m[0] for m in matches], seen[variant][0],
                    )
                    continue
                seen[variant] = (href, full_url)
            listing[year] = list(seen.values())

        self._viirs_listing_cache = listing
        return listing

    #: download_file() aborts a download only when it *stalls* -- no growth
    #: in the partial file for this many seconds -- not on a fixed total
    #: budget, since a healthy VNL composite is ~0.5-2 GB and can legitimately
    #: take many minutes on a slow link. Overridable per deployment via
    #: `sources.eog_viirs.download.timeout`. DOWNLOAD_MAX_TIMEOUT_S is a hard
    #: backstop against a download that dribbles forever without ever
    #: stalling long enough to trip the idle check.
    DOWNLOAD_STALL_TIMEOUT_S = 600
    DOWNLOAD_MAX_TIMEOUT_S = 4 * 3600
    DOWNLOAD_POLL_S = 5

    def download_file(self, file_url, output_path, driver=None):
        """Trigger *file_url* in Chrome and poll the Selenium download
        scratch dir until a non-partial file appears, then copy it to
        *output_path*. Returns True on success, False on stall/error --
        the caller turns a False into a retryable fetch failure."""
        import shutil
        import time

        # output_path is the fetch driver's atomic "<final>.part" temp -- log
        # the real name, not the .part.
        name = os.path.basename(output_path)
        if name.endswith(".part"):
            name = name[: -len(".part")]

        stall_timeout = self._download_stall_timeout_s

        current_driver = driver or self._driver
        if current_driver is None or not hasattr(current_driver, "get"):
            logger.error("EOG download %s: no Selenium WebDriver available", name)
            return False

        try:
            download_dir = getattr(current_driver, "_eog_download_dir", None)
            if not download_dir or not os.path.exists(download_dir):
                if self._download_dir and os.path.exists(self._download_dir):
                    download_dir = self._download_dir
                else:
                    download_dir = tempfile.mkdtemp(prefix="eog_session_downloads_")
                current_driver._eog_download_dir = download_dir

            before_files = set(os.listdir(download_dir))
            logger.info("EOG download: %s", name)
            logger.debug("EOG download URL: %s", file_url)
            started = time.monotonic()
            current_driver.get(file_url)
            self._check_and_handle_login(current_driver)

            next_heartbeat = 30
            last_size: Dict[str, int] = {}
            max_partial_bytes = 0
            last_progress_at = started
            while True:
                now = time.monotonic()
                elapsed = now - started
                new_files = set(os.listdir(download_dir)) - before_files
                # A finished download: a non-partial file that is non-empty
                # and whose size held steady across two polls (Chrome renames
                # <name>.crdownload -> <name> only at the end, but new-headless
                # / blocked-download failures can drop a 0-byte stub, and a
                # just-renamed file can still be flushing).
                ready = None
                partial_bytes = 0
                for f in new_files:
                    path = os.path.join(download_dir, f)
                    try:
                        size = os.path.getsize(path)
                    except OSError:
                        continue
                    if f.endswith((".tmp", ".crdownload")):
                        partial_bytes = max(partial_bytes, size)
                        continue
                    if size > 0 and last_size.get(f) == size:
                        ready = path
                        break
                    last_size[f] = size

                if ready:
                    size_mb = os.path.getsize(ready) / 1e6
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    shutil.copy2(ready, output_path)
                    try:
                        os.remove(ready)  # keep the scratch dir from growing across a run
                    except OSError:
                        pass
                    logger.info("EOG download complete: %s (%.1f MB, %.0fs)", name, size_mb, elapsed)
                    return True

                # Reset the stall clock whenever the partial file grows -- a
                # slow-but-flowing download must not be killed just for taking
                # a long time.
                if partial_bytes > max_partial_bytes:
                    max_partial_bytes = partial_bytes
                    last_progress_at = now

                idle = now - last_progress_at
                if idle >= stall_timeout or elapsed >= self.DOWNLOAD_MAX_TIMEOUT_S:
                    why = (
                        f"no progress for {idle:.0f}s"
                        if idle >= stall_timeout
                        else f"exceeded {self.DOWNLOAD_MAX_TIMEOUT_S}s hard cap"
                    )
                    holds = ", ".join(sorted(new_files)) or "nothing started"
                    logger.error(
                        "EOG download STALLED (%s) after %.0fs at %.1f MB: %s -- scratch dir holds: %s",
                        why, elapsed, max_partial_bytes / 1e6, name, holds,
                    )
                    return False

                if elapsed >= next_heartbeat:
                    logger.info(
                        "EOG download still running: %s -- %.0fs (%.1f MB so far)",
                        name, elapsed, max_partial_bytes / 1e6,
                    )
                    next_heartbeat += 30

                time.sleep(self.DOWNLOAD_POLL_S)
        except Exception:
            logger.exception("EOG download failed: %s", name)
            return False

    def download(self, file_url: str, output_path: str, session: Any = None) -> None:
        close_driver = False
        try:
            if session is None:
                if self._driver is None:
                    self._init_selenium_driver()
                close_driver = True
            else:
                self._driver = session
            if not self.download_file(file_url, output_path):
                raise RuntimeError(f"Failed to download {file_url}")
        finally:
            if close_driver:
                self._close_selenium_driver()

    async def download_async(self, file_url: str, output_path: str, session: Any = None) -> None:
        if session is not None and not hasattr(session, "find_element"):
            session = None
        await asyncio.sleep(0.5)
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._download_sync_wrapper, file_url, output_path, session)
        except Exception:
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            raise

    def _download_sync_wrapper(self, file_url: str, output_path: str, session=None):
        # download_file() has already logged the specific reason (timeout,
        # no driver, exception); this message only needs to mark the unit
        # failed for the fetch manifest.
        failed = RuntimeError("Selenium download did not produce a file (see log above)")
        if session is not None and hasattr(session, "find_element"):
            if not self.download_file(file_url, output_path, driver=session):
                raise failed
            return
        close_driver = False
        try:
            if self._driver is None:
                self._init_selenium_driver()
                close_driver = True
            if not self.download_file(file_url, output_path, driver=self._driver):
                raise failed
        finally:
            if close_driver:
                self._close_selenium_driver()

    # ------------------------------------------------------------------
    # plan()/execute() dispatch
    # ------------------------------------------------------------------

    def _plan(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        if step is PipelineStep.FETCH:
            return self._plan_fetch()
        if step is PipelineStep.PREPARE:
            return self._plan_prepare(selection)
        raise AssertionError(f"unreachable: {step}")

    def _execute(self, target: StepTarget) -> bool:
        if target.step is PipelineStep.FETCH:
            return self._execute_fetch(target)
        if target.step is PipelineStep.PREPARE:
            return self._execute_prepare(target)
        raise AssertionError(f"unreachable: {target.step}")

    # -- FETCH ----------------------------------------------------------

    def _plan_fetch(self) -> List[StepTarget]:
        return [
            StepTarget(
                source_id=self.cfg.source_id,
                step=PipelineStep.FETCH,
                key="all",
                output_path=self.output_root(PipelineStep.FETCH),
                completion=Completion.NEVER,
            )
        ]

    def _execute_fetch(self, target: StepTarget) -> bool:
        # `data transfer` (or transfer_mode=auto's inline push in run_fetch)
        # is what moves these bytes to HPC -- this step only downloads.
        #
        # Open ONE authenticated Chrome for the whole run and keep it on
        # `self._driver`: `_download_sync_wrapper` reuses a non-None
        # `self._driver` without tearing it down, so every VNL file rides the
        # same session instead of paying a fresh WebDriver launch + EOG login
        # (~10-15s each) per file -- the dominant cost now that each year
        # fetches three variants.
        from src.data.common.fetch.driver import run_fetch

        opened_here = self._driver is None
        if opened_here:
            self.get_authenticated_session()
        try:
            return run_fetch(self, **self.cfg.raw.get("download", {}))
        finally:
            if opened_here:
                self._close_selenium_driver()

    # -- PREPARE (raw fetched file -> tiled, reprojected output) ----------
    # the AttributeError bug fix lives here (module docstring)

    @staticmethod
    def _extract_year_from_path(path: str) -> Optional[int]:
        """FIX (see module docstring): the old code called this but never
        defined it.

        DMSP filenames concatenate the satellite code directly with the year,
        no delimiter (`F182019...` = satellite F18 + year 2019) -- the
        generic delimiter-based 4-digit pattern used elsewhere in this
        codebase (acag/esacci/ntl_harm) cannot isolate "2019" out of the
        6-digit run "182019", so that DMSP-specific shape is tried first
        (mirroring the `F(\\d+)(\\d{4})` satellite regex already used a few
        lines below in `_process_data_files`). VIIRS/DVNL filenames delimit
        the year normally (e.g. `..._2020_...`) and fall through to the
        generic pattern.
        """
        filename = os.path.basename(path)
        dmsp_match = re.search(r"F\d+(\d{4})", filename)
        if dmsp_match:
            year = int(dmsp_match.group(1))
            if 1990 <= year <= 2040:
                return year
        for pattern in (r"[._\-](\d{4})[._\-]", r"(\d{4})"):
            for match in re.finditer(pattern, filename):
                year = int(match.group(1))
                if 1990 <= year <= 2040:
                    return year
        return None

    def _select_best_file_for_year(self, year_files: List[str]) -> str:
        """FIX (see module docstring): the old code called this but never
        defined it. Prefers the source's own configured `file_extensions`
        order (default `.tif > .tgz > .tar.gz > .gz`)."""
        if len(year_files) == 1:
            return year_files[0]
        for ext in self.file_extensions:
            for file_path in year_files:
                if file_path.lower().endswith(ext.lower()):
                    return file_path
        return year_files[0]

    def _resolve_source_file_path(self, file_path: str) -> str:
        if os.path.isabs(file_path) or (self.ctx.data_root and file_path.startswith(self.ctx.data_root)):
            return file_path
        return os.path.join(self.output_root(PipelineStep.FETCH), file_path)

    def _output_columns(self) -> Tuple[str, ...]:
        """The data columns this source's PREPARE output carries. VIIRS
        annual is multi-variant (mean/median/cf_cvg -- VIIRS_VARIANT_COLUMNS);
        DMSP/DVNL are a single column named for the `source_type`."""
        if self.source_type == "viirs_annual":
            return tuple(self.VIIRS_VARIANT_COLUMNS[v] for v in self.VIIRS_VARIANTS)
        return (self.source_type,)

    @staticmethod
    def _maybe_gunzip(local_file: str) -> Tuple[str, Optional[str]]:
        """`(path_to_read, temp_to_delete_or_None)` -- decompresses a
        `.gz`-wrapped raw file next to itself; the caller owns deleting the
        returned temp only once the year's Dataset is evicted from cache
        (docs/design/13-prepare-memory-parallelism.md)."""
        if not local_file.endswith(".gz"):
            return local_file, None
        import gzip
        import shutil

        uncompressed = local_file[:-3]
        with gzip.open(local_file, "rb") as f_in, open(uncompressed, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        return uncompressed, uncompressed

    def _files_by_year(self) -> Dict[int, List[str]]:
        """Live crawl of FETCH's raw output directory: ground truth for
        which years have a fetched file (DMSP/DVNL -- one file per year).
        VIIRS annual has its own variant-aware crawl below."""
        raw_root = self.output_root(PipelineStep.FETCH)
        if not os.path.isdir(raw_root):
            return {}
        files_by_year: Dict[int, List[str]] = {}
        for dirpath, _dirnames, filenames in os.walk(raw_root):
            for fname in filenames:
                rel = os.path.relpath(os.path.join(dirpath, fname), raw_root)
                year = self._extract_year_from_path(fname)
                if year is not None:
                    files_by_year.setdefault(year, []).append(rel)
        return files_by_year

    def _viirs_files_by_year_variant(self) -> Dict[int, Dict[str, str]]:
        """`{year: {variant: relpath}}` for the fetched VNL composites --
        both the year and the product variant read off each filename via
        `_VIIRS_FILENAME_RE` (a plain 4-digit-year scan cannot tell
        `average_masked` from `median_masked` from `cf_cvg` for the same
        year). A duplicate (year, variant) keeps the first (`os.walk` yields
        the raw-root copy before any `<year>/` subdir copy): the *same*
        filename found in two places is just a stray extra copy (DEBUG); two
        *different* filenames claiming one (year, variant) is a real
        ambiguity (WARNING)."""
        raw_root = self.output_root(PipelineStep.FETCH)
        if not os.path.isdir(raw_root):
            return {}
        out: Dict[int, Dict[str, str]] = {}
        for dirpath, _dirnames, filenames in os.walk(raw_root):
            for fname in filenames:
                match = self._VIIRS_FILENAME_RE.search(fname)
                if not match:
                    continue
                year = self._viirs_match_year(match)
                variant = match.group("variant")
                if year is None or variant not in self.VIIRS_VARIANTS:
                    continue
                rel = os.path.relpath(os.path.join(dirpath, fname), raw_root)
                slot = out.setdefault(year, {})
                if variant in slot:
                    if os.path.basename(slot[variant]) == fname:
                        logger.debug(
                            "EOG VIIRS %d/%s: extra copy at %s, using %s", year, variant, rel, slot[variant]
                        )
                    else:
                        logger.warning(
                            "Conflicting EOG VIIRS files for %d/%s: %s vs %s -- using the first",
                            year, variant, slot[variant], rel,
                        )
                    continue
                slot[variant] = rel
        return out

    def _plan_prepare(self, selection: TargetSelection) -> List[StepTarget]:
        if self.source_type == "viirs_annual":
            return self._plan_prepare_viirs(selection)

        files_by_year = self._files_by_year()
        years = sorted(
            year for year in files_by_year if selection.matches_year(year) and selection.matches_key(str(year))
        )
        if not years:
            return []
        raw_files = {year: self._select_best_file_for_year(files_by_year[year]) for year in years}
        return [
            StepTarget(
                source_id=self.cfg.source_id,
                step=PipelineStep.PREPARE,
                key="all",
                output_path=self._output_path(),
                inputs=tuple(raw_files[year] for year in years),
                completion=Completion.MARKER,
                meta={
                    "years": years,
                    "raw_files": raw_files,
                    **verify.verification_meta(
                        self.cfg.raw,
                        expected_vars=(self.source_type,),
                        # DMSP is a classic 6-bit DN (0-63); VIIRS/DVNL
                        # radiance is continuous and can spike much higher
                        # over cities/flares.
                        value_range=(0, 63) if self.source_type == "dmsp" else (0, 1000),
                    ),
                },
            )
        ]

    def _plan_prepare_viirs(self, selection: TargetSelection) -> List[StepTarget]:
        """VIIRS annual: one PREPARE target carrying all `VIIRS_VARIANTS` as
        parallel columns. A year is preparable iff its primary (mean) file
        was fetched; a missing median/cf_cvg file is NaN-filled in
        `_load_year` so every year's parquet part keeps the same schema."""
        files = self._viirs_files_by_year_variant()
        years = sorted(
            year
            for year, variants in files.items()
            if self.VIIRS_PRIMARY_VARIANT in variants
            and selection.matches_year(year)
            and selection.matches_key(str(year))
        )
        if not years:
            return []
        raw_files = {year: files[year] for year in years}
        for year in years:
            missing = [v for v in self.VIIRS_VARIANTS if v not in raw_files[year]]
            if missing:
                logger.warning(
                    "EOG VIIRS %d missing variant file(s) %s -- corresponding column(s) NaN-filled",
                    year, missing,
                )
        columns = self._output_columns()
        return [
            StepTarget(
                source_id=self.cfg.source_id,
                step=PipelineStep.PREPARE,
                key="all",
                output_path=self._output_path(),
                inputs=tuple(path for year in years for path in raw_files[year].values()),
                completion=Completion.MARKER,
                meta={
                    "years": years,
                    "raw_files": raw_files,
                    **verify.verification_meta(
                        self.cfg.raw,
                        expected_vars=columns,
                        # One-sided magnitude guard: the upper bound catches a
                        # blown-up `sum` / garbage floats (legit flare cells
                        # reach ~1e6), the small negative floor tolerates VNL
                        # V2's signed background noise -- see the range-check
                        # note in orchestration/configs/data.yaml. Only the two
                        # radiance columns are range-checked; cf_cvg (an
                        # observation count) is left out of range_vars.
                        value_range=(-100, 1_000_000),
                        range_vars=("viirs_annual", "viirs_annual_median"),
                    ),
                },
            )
        ]

    def _load_year(self, source: "str | Dict[str, str]", year: int) -> "Optional[Tuple[xr.Dataset, List[str]]]":
        """Returns `(dataset, temp_files_to_delete)` -- stays dask-backed
        (`chunks="auto"`), not `.load()`'d, so the caller can clip to one
        output tile's bbox via `sel_bbox()` before compute()ing
        (docs/design/13-prepare-memory-parallelism.md). Any `.gz`-decompressed
        temp files must stay on disk until every tile for this year has been
        computed against them -- the caller (`_execute_prepare`'s
        `year_ds()`/`evict_cache()`) owns deleting them once this year's
        Dataset leaves the cache, not here.

        *source* is a `{variant: relpath}` map for VIIRS annual (one column
        per variant), or a single relpath string for DMSP/DVNL.
        """
        if isinstance(source, dict):
            return self._load_year_viirs(source, year)
        return self._load_year_single(source, year)

    def _load_year_single(self, file_path: str, year: int) -> "Optional[Tuple[xr.Dataset, List[str]]]":
        import rioxarray as rxr

        temps: List[str] = []
        try:
            local_file, tmp = self._maybe_gunzip(self._resolve_source_file_path(file_path))
            if tmp:
                temps.append(tmp)
            if not os.path.exists(local_file):
                logger.error("File does not exist: %s", local_file)
                return None

            da = rxr.open_rasterio(local_file, chunks="auto")
            da = da.expand_dims(dim={"time": 1}).assign_coords({"time": [pd.Timestamp(f"{year}-12-31")]})

            attrs = {}
            if self.source_type == "dmsp":
                filename = os.path.basename(file_path)
                match = re.search(r"F(\d+)(\d{4})", filename)
                if match:
                    attrs["satellite"] = f"F{match.group(1)}"

            ds = da.to_dataset(name=self.source_type)
            ds = ds.assign_attrs(**attrs)
            if ds.rio.crs is None:
                ds = ds.rio.write_crs(4326)
            return ds, temps
        except Exception:
            logger.exception("Error processing file %s.", file_path)
            for t in temps:
                if os.path.exists(t):
                    os.remove(t)
            return None

    def _load_year_viirs(self, paths: Dict[str, str], year: int) -> "Optional[Tuple[xr.Dataset, List[str]]]":
        """One Dataset with a column per `VIIRS_VARIANTS` entry
        (VIIRS_VARIANT_COLUMNS names them). A variant whose file wasn't
        fetched is NaN-filled onto the primary (mean) grid so the merged
        schema is stable across years."""
        import rioxarray as rxr

        temps: List[str] = []
        try:
            arrays: Dict[str, xr.DataArray] = {}
            for variant, rel in paths.items():
                local_file, tmp = self._maybe_gunzip(self._resolve_source_file_path(rel))
                if tmp:
                    temps.append(tmp)
                if not os.path.exists(local_file):
                    logger.error("EOG VIIRS file does not exist: %s", local_file)
                    return None
                da = rxr.open_rasterio(local_file, chunks="auto")
                da = da.expand_dims(dim={"time": 1}).assign_coords({"time": [pd.Timestamp(f"{year}-12-31")]})
                arrays[self.VIIRS_VARIANT_COLUMNS[variant]] = da

            primary_col = self.VIIRS_VARIANT_COLUMNS[self.VIIRS_PRIMARY_VARIANT]
            primary = arrays[primary_col]
            for variant in self.VIIRS_VARIANTS:
                col = self.VIIRS_VARIANT_COLUMNS[variant]
                if col not in arrays:
                    arrays[col] = xr.full_like(primary, np.nan, dtype="float32")

            ds = xr.Dataset(arrays)
            if ds.rio.crs is None:
                ds = ds.rio.write_crs(4326)
            return ds, temps
        except Exception:
            logger.exception("Error loading EOG VIIRS year %s", year)
            for t in temps:
                if os.path.exists(t):
                    os.remove(t)
            return None

    def _output_path(self) -> str:
        return layout.grid_store_path(
            self.ctx.data_root,
            self.cfg.data_path,
            grid_id=self.ctx.grid_id,
            family=f"eog_{self.source_type}",
            suffix="",  # cell_id-keyed parquet parts, not a Zarr store -- see grid_store_path docstring
        )

    def _execute_prepare(self, target: StepTarget) -> bool:
        from src.data.common.geobox import get_target_geobox
        from src.data.common.prepare.driver import run_tiled_prepare
        from src.data.common.raster.spatial import SpatialProcessor, sel_bbox
        from src.data.sources.steps import is_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping PREPARE -- already complete: %s", target.output_path)
            return True

        years: List[int] = target.meta["years"]
        # {year: relpath} for DMSP/DVNL, {year: {variant: relpath}} for VIIRS.
        raw_files: Dict[int, Any] = target.meta["raw_files"]
        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)

        target_geobox = get_target_geobox(self.ctx)

        with self._dask_client() as client:
            if client is None:
                return False
            processor = SpatialProcessor(
                hpc_root=self.ctx.data_root,
                temp_dir=self.temp_dir,
                dask_client=client,
                target_geobox=target_geobox,
            )
            with processor.setup_dask_config():
                # run_tiled_prepare walks units years-major, so only one
                # year's lazy Dataset (and, if `.gz`-wrapped, its
                # decompressed temp file) needs to stay alive at once --
                # evict_cache() below closes/removes the previous year's on
                # each year change. raw_getter clips to each tile's own bbox
                # and computes only that (docs/design/13-prepare-memory-
                # parallelism.md), instead of the whole annual global
                # raster this used to eagerly `.load()`.
                cache: Dict[int, Optional[Tuple[xr.Dataset, List[str]]]] = {}
                output_columns = self._output_columns()

                def evict_cache() -> None:
                    for old_year, old_entry in list(cache.items()):
                        if old_entry is not None:
                            old_ds, old_tmps = old_entry
                            old_ds.close()
                            for old_tmp in old_tmps:
                                if os.path.exists(old_tmp):
                                    os.remove(old_tmp)
                        del cache[old_year]

                def year_ds(year: int) -> Optional[xr.Dataset]:
                    if year not in cache:
                        evict_cache()
                        cache[year] = self._load_year(raw_files[year], year)
                    entry = cache[year]
                    return entry[0] if entry is not None else None

                def raw_getter(tile, year: int) -> Optional[xr.Dataset]:
                    ds = year_ds(year)
                    if ds is None:
                        return None
                    bbox = tile.geobox.pad(32, 32).extent.to_crs(ds.rio.crs).boundingbox
                    clipped = sel_bbox(ds, bbox, y_dim="y", x_dim="x")
                    if clipped.sizes.get("x", 0) == 0 or clipped.sizes.get("y", 0) == 0:
                        # Tile falls outside this year's raster coverage --
                        # legitimate tile state (e.g. poleward of VIIRS/
                        # DMSP's own extent), not a fetch failure. NaN-fill
                        # every output column on tile.geobox instead of None,
                        # same convention as MODIS/ESA-CCI/AVHRR's raw_getter.
                        dim_y, dim_x = tile.geobox.dims
                        return xr.Dataset(
                            {
                                col: ((dim_y, dim_x), np.full(tile.geobox.shape, np.nan, dtype=np.float32))
                                for col in output_columns
                            }
                        )
                    return clipped.compute()

                try:
                    return run_tiled_prepare(
                        output_path=target.output_path,
                        years=years,
                        variables=list(output_columns),
                        target_geobox=target_geobox,
                        processor=processor,
                        raw_getter=raw_getter,
                        tile_size=self.tile_size,
                        resampling=self.resampling,
                        processing_version=self.PROCESSING_VERSION,
                        override=self.cfg.override,
                    )
                finally:
                    evict_cache()

    # _dask_client: inherited from DataSource (src/data/sources/base.py).

registry.register(
    EogSource.ID,
    __name__,
    EogSource.__name__,
    EogSource.STEPS,
    aliases=EogSource.ALIASES,
)

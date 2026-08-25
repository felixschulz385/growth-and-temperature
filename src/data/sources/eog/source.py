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
mapped, per `data summary`/`data fetch`, to the one file whose period ends
in December of that year (excludes the rolling/intermediate periods) and
whose variant is `VIIRS_VARIANT` (`average_masked` -- the standard masked
composite used in nightlights economics literature, not raw `average` or
any of the coverage/min/max/median variants). The *exact* URL (with its
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
    PROCESSING_VERSION = "2-tiled"

    #: VIIRS annual-composite years worth fetching (module docstring) --
    #: hardcoded rather than discovered, since the directory mixes canonical
    #: composites with intermediate/rolling reprocessing periods with no
    #: reliable way to tell them apart except by which December they end in
    #: (see _viirs_annual_listing()).
    VIIRS_YEAR_RANGE = (2012, 2021)

    #: The VNL product variant fetched -- masked composite (background/
    #: fire/aurora-corrected), the standard used in nightlights economics
    #: literature, not raw "average" or the coverage/min/max/median variants
    #: also present in the same directory.
    VIIRS_VARIANT = "average_masked"

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
        self.resampling = cfg.raw.get("resampling", "sum")

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
        here for `VIIRS_VARIANT` entries whose period ends in December of
        their own folder's year. All year directories are crawled in one
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
                    if match.group("variant") != self.VIIRS_VARIANT:
                        continue
                    if self._viirs_match_year(match) != year:
                        continue
                    listing.setdefault(year, []).append((href, full_url))
        finally:
            self._close_selenium_driver()

        for year, matches in listing.items():
            if len(matches) > 1:
                logger.warning(
                    "Multiple %s candidates matched EOG VIIRS year %d: %s -- using the first",
                    self.VIIRS_VARIANT, year, [m[0] for m in matches],
                )
                listing[year] = matches[:1]

        self._viirs_listing_cache = listing
        return listing

    def download_file(self, file_url, output_path, driver=None):
        """Ported from EOGDataSource.download_file -- polls the shared
        Selenium download directory for the newest completed file."""
        import shutil
        import time

        current_driver = driver or self._driver
        if not hasattr(current_driver, "get") or not hasattr(current_driver, "find_element"):
            logger.error("EOG downloads require Selenium WebDriver")
            return False
        if current_driver is None:
            logger.error("No Selenium driver available")
            return False

        try:
            download_dir = getattr(current_driver, "_eog_download_dir", None)
            if not download_dir or not os.path.exists(download_dir):
                if self._download_dir and os.path.exists(self._download_dir):
                    download_dir = self._download_dir
                    current_driver._eog_download_dir = download_dir
                else:
                    download_dir = tempfile.mkdtemp(prefix="eog_session_downloads_")
                    current_driver._eog_download_dir = download_dir

            before_files = set(os.listdir(download_dir))
            current_driver.get(file_url)
            self._check_and_handle_login(current_driver)

            max_wait_time, interval, elapsed = 300, 5, 0
            while elapsed < max_wait_time:
                current_files = set(os.listdir(download_dir))
                new_files = current_files - before_files
                completed = [f for f in new_files if not f.endswith(".tmp") and not f.endswith(".crdownload")]
                if completed:
                    latest = max((os.path.join(download_dir, f) for f in completed), key=os.path.getmtime)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    shutil.copy2(latest, output_path)
                    return True
                time.sleep(interval)
                elapsed += interval
            logger.error("Download timeout exceeded")
            return False
        except Exception:
            logger.exception("Error downloading file")
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
        if session is not None and hasattr(session, "find_element"):
            if not self.download_file(file_url, output_path, driver=session):
                raise RuntimeError(f"Failed to download {file_url}")
            return
        close_driver = False
        try:
            if self._driver is None:
                self._init_selenium_driver()
                close_driver = True
            if not self.download_file(file_url, output_path, driver=self._driver):
                raise RuntimeError(f"Failed to download {file_url}")
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
        # FETCH is local-disk only now -- no HPC target required. `data
        # transfer` (separate, manual or auto per source config) is the only
        # thing that pushes to HPC.
        from src.data.common.fetch.driver import run_fetch

        return run_fetch(self, **self.cfg.raw.get("download", {}))

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

    def _files_by_year(self) -> Dict[int, List[str]]:
        """Live crawl of FETCH's raw output directory: ground truth for
        which years have a fetched file."""
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

    def _plan_prepare(self, selection: TargetSelection) -> List[StepTarget]:
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

    def _load_year(self, file_path: str, year: int) -> Optional[Tuple[xr.Dataset, Optional[str]]]:
        """Returns `(dataset, uncompressed_temp_path_or_None)` -- stays
        dask-backed (`chunks="auto"`), not `.load()`'d, so the caller can
        clip to one output tile's bbox via `sel_bbox()` before compute()ing
        (docs/design/13-prepare-memory-parallelism.md). For a `.gz`-wrapped
        source, the decompressed temp file must stay on disk until every
        tile for this year has been computed against it -- the caller (see
        `_execute_prepare`'s `year_ds()`/cache eviction) owns deleting it,
        only once this year's Dataset is evicted from its cache, not here."""
        import rioxarray as rxr

        uncompressed_file_to_delete = None
        try:
            local_file = file_path
            if file_path.endswith(".gz"):
                import gzip
                import shutil

                uncompressed = local_file[:-3]
                with gzip.open(local_file, "rb") as f_in, open(uncompressed, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                local_file = uncompressed
                uncompressed_file_to_delete = uncompressed

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
            return ds, uncompressed_file_to_delete
        except Exception:
            logger.exception("Error processing file %s.", file_path)
            if uncompressed_file_to_delete and os.path.exists(uncompressed_file_to_delete):
                os.remove(uncompressed_file_to_delete)
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
        raw_files: Dict[int, str] = target.meta["raw_files"]
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
                cache: Dict[int, Optional[Tuple[xr.Dataset, Optional[str]]]] = {}

                def evict_cache() -> None:
                    for old_year, old_entry in list(cache.items()):
                        if old_entry is not None:
                            old_ds, old_tmp = old_entry
                            old_ds.close()
                            if old_tmp and os.path.exists(old_tmp):
                                os.remove(old_tmp)
                        del cache[old_year]

                def year_ds(year: int) -> Optional[xr.Dataset]:
                    if year not in cache:
                        evict_cache()
                        source_file = self._resolve_source_file_path(raw_files[year])
                        cache[year] = self._load_year(source_file, year)
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
                        # on tile.geobox instead of None, same convention as
                        # MODIS/ESA-CCI/AVHRR's own raw_getter.
                        dim_y, dim_x = tile.geobox.dims
                        return xr.Dataset(
                            {self.source_type: ((dim_y, dim_x), np.full(tile.geobox.shape, np.nan, dtype=np.float32))}
                        )
                    return clipped.compute()

                try:
                    return run_tiled_prepare(
                        output_path=target.output_path,
                        years=years,
                        variables=[self.source_type],
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

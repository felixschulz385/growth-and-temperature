"""EOG (DMSP/VIIRS/DVNL) nighttime lights: fetch + prepare + grid.

docs/design/09-integrated-pipeline.md §5: one registered id with three
aliases (`eog_dmsp`, `eog_viirs`, `eog_dvnl`), matching today's `NAMES` tuple
in `src/data/download/sources/eog/source.py`. Merges that module's
`EOGDataSource` (Selenium-authenticated crawl/download, reused verbatim via
`_CrawlerMixin`/`_SessionMixin`) and
`src/data/preprocess/sources/eog.py::EOGPreprocessor` (`stage="annual"` ->
PREPARE, `stage="spatial"` -> GRID).

**Real bug fixed here, not silently ported**: the old `EOGPreprocessor.
_generate_annual_targets` calls `self._extract_year_from_path(...)` and
`self._select_best_file_for_year(...)` -- neither method exists anywhere in
that class (verified by direct execution, see
tests/data/preprocess/sources/test_characterization_eog.py), so every call to
`get_preprocessing_targets("annual", ...)` raises `AttributeError`, silently
caught, always returning `[]`. **EOG's annual/PREPARE stage has never
produced a target.** This migration implements both methods for real: year
extraction uses the same generic 4-digit-year regex every other source in
this codebase already uses (acag/esacci/ntl_harm's `_extract_year`); best-file
selection prefers extensions in the order the source's own
`file_extensions` config already declares (default `.tif > .tgz > .tar.gz >
.gz`) rather than inventing an unrelated preference.

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
drives PREPARE's output variable name and GRID's output filename/`v2_family`.
Now derived from `cfg.source_id` directly, the same authoritative signal
`GlassSource.__init__` already uses for its own MODIS/AVHRR variant
(`data_source_kind`).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd
import xarray as xr
from zarr.codecs import BloscCodec

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.eog.crawler import _CrawlerMixin
from src.data.sources.eog.session import _SessionMixin
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection
from src.data.sources import verify

if TYPE_CHECKING:
    from src.data.common.ledger.store import ArtifactRow

logger = logging.getLogger(__name__)


class EogSource(_CrawlerMixin, _SessionMixin, DataSource):
    """Earth Observation Group nighttime lights (DMSP-OLS, VIIRS annual, DVNL).

    FETCH   -- Selenium-authenticated crawl + download of the configured
               `base_url` archive (credentials via EOG_USERNAME/EOG_PASSWORD).
    PREPARE -- one annual zarr per year.
    GRID    -- reproject every annual zarr onto the canonical geobox; radiance
               field, so resampling defaults to area-weighted "sum"
               (docs/design/04-ingest.md §1), not SpatialProcessor's own
               "nearest" default.
    """

    ID = "eog"
    ALIASES = ("eog_dmsp", "eog_viirs", "eog_dvnl")
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE, PipelineStep.GRID)

    DATA_SOURCE_NAME = "eog"  # matches old EOGDataSource: literally "eog", not per-alias
    has_entrypoints = False

    EOG_LOGIN_URL = "https://eogdata.mines.edu/nighttime_light/login/"

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

        self._username = os.environ.get("EOG_USERNAME")
        self._password = os.environ.get("EOG_PASSWORD")
        if not self._username or not self._password:
            logger.warning("EOG credentials not set in environment variables (EOG_USERNAME, EOG_PASSWORD)")
        self._driver = None
        self._download_dir = None
        self._is_logged_in = False

        self.source_type = self._derive_source_type()
        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix=f"eog_{self.source_type}_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)

    def _derive_source_type(self) -> str:
        """Which variant: derived from `cfg.source_id` -- the literal
        `sources.<id>:` config-block key this instance was built from
        (module docstring) -- mirroring `GlassSource.__init__`'s identical
        `cfg.source_id`-based derivation of its own MODIS/AVHRR variant.
        Raises rather than guessing from `data_path`/`base_url` content
        (the old behavior) if `source_id` doesn't name a known variant, so a
        misconfigured/renamed source_id fails loudly instead of silently
        mislabeling PREPARE's output variable / GRID's output filename."""
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

    def filename_to_entrypoint(self, relative_path: str) -> Optional[Dict[str, Any]]:
        return None  # matches old EOGDataSource: entrypoints not used

    def get_all_entrypoints(self) -> List[Dict[str, Any]]:
        raise NotImplementedError  # matches BaseDataSource's unoverridden default; never called (has_entrypoints=False)

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
        if step is PipelineStep.GRID:
            return self._plan_grid(selection)
        raise AssertionError(f"unreachable: {step}")

    def _discover(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        """Ground truth for `data reconcile` (src/data/sources/base.py's
        `discover()`): FETCH has no crawl to redo (`_plan_fetch` is already a
        static, I/O-free target), so only PREPARE/GRID get a real live-crawl
        counterpart distinct from their ledger-backed `_plan_*`."""
        if step is PipelineStep.FETCH:
            return self._plan_fetch()
        if step is PipelineStep.PREPARE:
            return self._discover_prepare(selection)
        if step is PipelineStep.GRID:
            return self._discover_grid(selection)
        raise AssertionError(f"unreachable: {step}")

    def _execute(self, target: StepTarget) -> bool:
        if target.step is PipelineStep.FETCH:
            return self._execute_fetch(target)
        if target.step is PipelineStep.PREPARE:
            return self._execute_prepare(target)
        if target.step is PipelineStep.GRID:
            return self._execute_grid(target)
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
        if not self.ctx.ssh_target:
            logger.warning("Fetch requires an HPC/remote target to be configured.")
            return False

        from src.data.common.fetch.driver import run_fetch

        return run_fetch(self, **self.cfg.raw.get("download", {}))

    # -- PREPARE ("annual") -- the bug fix lives here --------------------

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

    def _plan_prepare(self, selection: TargetSelection) -> List[StepTarget]:
        """Ledger-backed fast path: reads existing `artifacts` rows instead
        of re-querying `completed_fetch_files()` and rebuilding the
        year-grouping every call. Falls back to `_discover_prepare()` --
        today's exact live logic -- if no ledger is configured yet, or
        `data reconcile --step prepare` hasn't populated one yet."""

        def build_target(row: "ArtifactRow", _ledger: Any) -> Optional[StepTarget]:
            year = row.meta.get("year")
            raw_relative_path = row.meta.get("raw_relative_path")
            if year is None or raw_relative_path is None or row.local_path is None:
                return None
            if not selection.matches_year(year):
                return None
            return StepTarget(
                source_id=self.cfg.source_id,
                step=PipelineStep.PREPARE,
                key=row.unit_id,
                output_path=row.local_path,
                inputs=(raw_relative_path,),
                completion=Completion.MARKER,
                meta=row.meta,
            )

        targets = self._plan_from_ledger(PipelineStep.PREPARE, selection, build_target)
        if targets is not None:
            return targets
        logger.warning(
            "No ledger for source='%s' step='prepare' -- falling back to live discovery; "
            "run `data reconcile --source %s --step prepare` for faster planning.",
            self.cfg.source_id, self.cfg.source_id,
        )
        return self._discover_prepare(selection)

    def _discover_prepare(self, selection: TargetSelection) -> List[StepTarget]:
        """Live ground truth for PREPARE: queries the ledger's FETCH-crawl
        catalog (`completed_fetch_files()`, not `artifacts`-for-PREPARE) and
        groups by year. Called from `discover()` (via `data reconcile`,
        which writes its result into `artifacts`) and as `_plan_prepare()`'s
        fallback when that table isn't populated yet."""
        from src.data.common.ledger.paths import ledger_path
        from src.data.common.ledger.store import SourceLedger

        local_ledger_path = ledger_path(self.ctx.local_index_dir, self.data_path)
        if not local_ledger_path or not os.path.exists(local_ledger_path):
            logger.warning("Ledger not found: %s", local_ledger_path)
            return []

        with SourceLedger.open_for_read(local_ledger_path, data_path=self.data_path) as ledger:
            relative_paths = ledger.completed_fetch_files()
        if not relative_paths:
            return []

        files_by_year: Dict[int, List[str]] = {}
        for rel_path in relative_paths:
            year = self._extract_year_from_path(rel_path)
            if year is None or not selection.matches_year(year):
                continue
            files_by_year.setdefault(year, []).append(rel_path)

        targets = []
        for year in sorted(files_by_year):  # sorted, unlike ntl_harm -- no insertion-order quirk to preserve here
            if not selection.matches_key(str(year)):
                continue
            year_files = files_by_year[year]
            selected = self._select_best_file_for_year(year_files)
            targets.append(
                StepTarget(
                    source_id=self.cfg.source_id,
                    step=PipelineStep.PREPARE,
                    key=str(year),
                    output_path=os.path.join(self.output_root(PipelineStep.PREPARE), f"{year}.zarr"),
                    inputs=(selected,),
                    completion=Completion.MARKER,
                    meta={"year": year, "total_candidates": len(year_files), "raw_relative_path": selected},
                )
            )
        return targets

    def _process_data_files(self, file_path: str, year: int) -> Optional[xr.DataArray]:
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

            ds = rxr.open_rasterio(local_file, chunks="auto")
            ds = ds.expand_dims(dim={"time": 1}).assign_coords({"time": [pd.Timestamp(f"{year}-12-31")]})

            if self.source_type == "dmsp":
                filename = os.path.basename(file_path)
                match = re.search(r"F(\d+)(\d{4})", filename)
                if match:
                    ds = ds.assign_attrs(satellite=f"F{match.group(1)}")

            if uncompressed_file_to_delete:
                ds.attrs["_cleanup_file"] = uncompressed_file_to_delete
            return ds
        except Exception:
            logger.exception("Error processing file %s.", file_path)
            return None

    def _create_annual_zarr(self, data_array: xr.DataArray, output_path: str) -> bool:
        cleanup_file = data_array.attrs.get("_cleanup_file")
        try:
            dataset = data_array.to_dataset(name=self.source_type)
            dataset.attrs.pop("_cleanup_file", None)
            dataset = dataset.chunk({"x": 1000, "y": 1000})

            compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle", blocksize=0)
            encoding = {var: {"compressors": (compressor,)} for var in dataset.data_vars}
            dataset.to_zarr(output_path, mode="w", encoding=encoding, zarr_format=3, consolidated=False)

            if cleanup_file and os.path.exists(cleanup_file):
                os.remove(cleanup_file)
            return True
        except Exception:
            logger.exception("Error creating zarr file at %s.", output_path)
            if cleanup_file and os.path.exists(cleanup_file):
                os.remove(cleanup_file)
            return False

    def _execute_prepare(self, target: StepTarget) -> bool:
        from src.data.sources.steps import is_complete, mark_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping year %s -- already complete: %s", target.key, target.output_path)
            return True

        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)
        year = target.meta["year"]
        source_file = self._resolve_source_file_path(target.inputs[0])

        data_array = self._process_data_files(source_file, year)
        if data_array is None:
            return False
        if not self._create_annual_zarr(data_array, target.output_path):
            return False
        mark_complete(target.output_path)
        return True

    # -- GRID ("spatial") -------------------------------------------------

    def _list_annual_zarrs(self) -> List[Dict[str, Any]]:
        annual_dir = self.output_root(PipelineStep.PREPARE)
        if not os.path.exists(annual_dir):
            return []
        results = []
        for fname in os.listdir(annual_dir):
            if fname.endswith(".zarr"):
                try:
                    year = int(os.path.splitext(fname)[0])
                    results.append({"year": year, "zarr_path": os.path.join(annual_dir, fname)})
                except ValueError:
                    pass
        return results

    def _grid_output_path(self) -> str:
        return layout.grid_store_path(
            self.ctx.data_root,
            self.cfg.data_path,
            f"{self.source_type}_timeseries_reprojected.zarr",
            namespace=self.cfg.namespace,
            grid_id=self.ctx.grid_id,
            layout=self.ctx.layout,
            v2_family=f"eog_{self.source_type}",
        )

    def _grid_meta(self, years_available: List[int]) -> Dict[str, Any]:
        return {
            "years_available": years_available,
            **verify.verification_meta(
                self.cfg.raw,
                expected_vars=(self.source_type,),
                # DMSP is a classic 6-bit DN (0-63); VIIRS/DVNL
                # radiance is continuous and can spike much higher
                # over cities/flares.
                value_range=(0, 63) if self.source_type == "dmsp" else (0, 1000),
            ),
        }

    def _discover_grid(self, selection: TargetSelection) -> List[StepTarget]:
        """Live ground truth for GRID: crawls the PREPARE output directory
        for annual zarrs. Called from `discover()` (via `data reconcile`)
        and as `_plan_grid()`'s fallback when the ledger has no GRID row yet."""
        annual_files = [f for f in self._list_annual_zarrs() if selection.matches_year(f["year"])]
        if not annual_files:
            return []
        return [
            StepTarget(
                source_id=self.cfg.source_id,
                step=PipelineStep.GRID,
                key="all",
                output_path=self._grid_output_path(),
                inputs=tuple(f["zarr_path"] for f in annual_files),
                completion=Completion.MARKER,
                meta=self._grid_meta([f["year"] for f in annual_files]),
            )
        ]

    def _plan_grid(self, selection: TargetSelection) -> List[StepTarget]:
        """Ledger-backed fast path: the single GRID target's `inputs` are
        re-derived live from PREPARE's `local_complete_units()` (a cheap
        indexed ledger query) rather than persisted -- persisting a
        filename-derived-year snapshot would go stale the moment a new
        PREPARE year lands without a matching GRID reconcile. Falls back to
        `_discover_grid()` if the ledger has no GRID row yet."""

        def build_target(row: "ArtifactRow", ledger: Any) -> Optional[StepTarget]:
            if row.local_path is None:
                return None
            annual = [
                (uid, path)
                for uid, path in ledger.local_complete_units("prepare")
                if uid.isdigit() and selection.matches_year(int(uid))
            ]
            if not annual:
                return None
            years = sorted(int(uid) for uid, _ in annual)
            return StepTarget(
                source_id=self.cfg.source_id,
                step=PipelineStep.GRID,
                key=row.unit_id,
                output_path=row.local_path,
                inputs=tuple(path for _, path in annual),
                completion=Completion.MARKER,
                meta=self._grid_meta(years),
            )

        targets = self._plan_from_ledger(PipelineStep.GRID, selection, build_target)
        if targets is not None:
            return targets
        logger.warning(
            "No ledger for source='%s' step='grid' -- falling back to live discovery; "
            "run `data reconcile --source %s --step grid` for faster planning.",
            self.cfg.source_id, self.cfg.source_id,
        )
        return self._discover_grid(selection)

    def _execute_grid(self, target: StepTarget) -> bool:
        from src.data.common.geobox import get_target_geobox
        from src.data.common.raster.spatial import SpatialProcessor
        from src.data.sources.steps import is_complete, mark_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping grid step -- already complete: %s", target.output_path)
            return True

        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)
        years = target.meta["years_available"]

        with self._dask_client() as client:
            if client is None:
                return False
            processor = SpatialProcessor(
                hpc_root=self.ctx.data_root,
                temp_dir=self.temp_dir,
                dask_client=client,
                target_geobox=get_target_geobox(self.ctx),
            )
            with processor.setup_dask_config():

                def year_from_path(p: str) -> Optional[int]:
                    try:
                        return int(os.path.splitext(os.path.basename(p))[0])
                    except ValueError:
                        return None

                def preprocess(ds: xr.Dataset) -> xr.Dataset:
                    if ds.rio.crs is None:
                        ds = ds.rio.write_crs(4326)
                    return ds

                def get_vars_and_attrs(file_path: str) -> Tuple[List[str], Dict]:
                    sample = xr.open_zarr(file_path, mask_and_scale=False, chunks="auto", consolidated=False)
                    variables = list(sample.data_vars.keys())
                    attrs = sample.attrs.copy()
                    sample.close()
                    return variables, attrs

                success = processor.process_spatial_standard(
                    source_files=list(target.inputs),
                    output_path=target.output_path,
                    years_to_process=years,
                    year_pattern_func=year_from_path,
                    preprocess_func=preprocess,
                    get_variables_func=get_vars_and_attrs,
                    resampling=self.resampling,
                )
                if success:
                    mark_complete(target.output_path)
                return success

    # _dask_client: inherited from DataSource (src/data/sources/base.py).

registry.register(
    EogSource.ID,
    __name__,
    EogSource.__name__,
    EogSource.STEPS,
    aliases=EogSource.ALIASES,
)

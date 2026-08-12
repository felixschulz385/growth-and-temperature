"""Harmonized DMSP-VIIRS nighttime lights (Figshare): fetch + prepare + grid.

docs/design/09-integrated-pipeline.md §5: same shape as `src/data/sources/acag.py`.
Merges `src/data/download/sources/ntl_harm.py::NTLHarmDataSource` (Figshare API
fetch) and `src/data/preprocess/sources/ntl_harm.py::NTLHarmPreprocessor`
(`stage="annual"` -> PREPARE, `stage="spatial"` -> GRID) into one class.

Two behavioural quirks are deliberately preserved unchanged (pinned by
tests/data/preprocess/sources/test_characterization_ntl_harm.py against the
old code, and tests/data/sources/ntl_harm/test_ntl_harm_plan.py against this
class): (1) PREPARE targets are emitted in file-insertion order, not sorted by
year; (2) the best-file-for-year preference is `.tif > .zip > .tar.gz > .gz`
(the opposite direction from acag/esacci's nc4-over-nc). A migration ports
behaviour; it does not silently improve it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import re
import tempfile
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
import requests
import xarray as xr
from zarr.codecs import BloscCodec

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection
from src.data.sources import verify

if TYPE_CHECKING:
    from src.data.common.ledger.store import ArtifactRow

logger = logging.getLogger(__name__)


class NtlHarmSource(DataSource):
    """Harmonized DMSP-VIIRS nighttime lights, Figshare dataset 9828827.

    FETCH   -- one Figshare file per year (cached article listing, 1h TTL).
    PREPARE -- one annual zarr per year (handles .gz/.zip decompression).
    GRID    -- reproject every annual zarr onto the canonical geobox; radiance
               field, so resampling defaults to area-weighted "sum"
               (docs/design/04-ingest.md §1), not SpatialProcessor's own
               "nearest" default.
    """

    ID = "ntl_harm"
    ALIASES = ("ntlharm", "harmonized_ntl")
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE, PipelineStep.GRID)

    DATA_SOURCE_NAME = "ntl_harm"
    has_entrypoints = True

    VARIABLE_NAME = "ntl_harm"
    FIGSHARE_API_BASE = "https://api.figshare.com/v2"
    DATASET_ID = "9828827"

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="ntl_harm/harmonized")  # old NTLHarmPreprocessor default
        super().__init__(ctx, cfg)
        self.base_url = cfg.raw.get("base_url") or f"{self.FIGSHARE_API_BASE}/articles/{self.DATASET_ID}"
        self.file_extensions = cfg.raw.get("file_extensions") or [".tif", ".zip", ".tar.gz", ".gz"]
        self.resampling = cfg.raw.get("resampling", "sum")
        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix="ntl_harm_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)
        self._files_cache: Optional[List[Dict[str, Any]]] = None
        self._cache_timestamp: Optional[float] = None
        self._cache_duration = 3600

    # ------------------------------------------------------------------
    # RemoteFileCatalog contract (ports NTLHarmDataSource verbatim)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_year_from_filename(filename: str) -> Optional[int]:
        for pattern in (r"(\d{4})", r"_(\d{4})_", r"\.(\d{4})\."):
            match = re.search(pattern, filename)
            if match:
                year = int(match.group(1))
                if 1992 <= year <= 2030:
                    return year
        return None

    # get_file_hash: inherited from DataSource (src/data/sources/base.py).

    def _get_figshare_files(self) -> List[Dict[str, Any]]:
        now = time.time()
        if self._files_cache is not None and self._cache_timestamp is not None and now - self._cache_timestamp < self._cache_duration:
            return self._files_cache
        try:
            response = requests.get(self.base_url)
            response.raise_for_status()
            files = response.json().get("files", [])
            self._files_cache = files
            self._cache_timestamp = now
            return files
        except (requests.RequestException, json.JSONDecodeError):
            logger.exception("Error fetching Figshare file listing.")
            return []

    def list_remote_files(self, entrypoint: Optional[dict] = None) -> List[Tuple[str, str]]:
        files = self._get_figshare_files()
        results = []
        for file_info in files:
            filename = file_info.get("name", "")
            download_url = file_info.get("download_url", "")
            if not filename or not download_url:
                continue
            if not any(filename.lower().endswith(ext.lower()) for ext in self.file_extensions):
                continue
            if entrypoint:
                year_filter = entrypoint.get("year")
                if year_filter and self._extract_year_from_filename(filename) != year_filter:
                    continue
            results.append((filename, download_url))
        return results

    def local_path(self, relative_path: str) -> str:
        return os.path.join("data", self.DATA_SOURCE_NAME, relative_path)

    def filename_to_entrypoint(self, relative_path: str) -> Optional[Dict[str, Any]]:
        year = self._extract_year_from_filename(os.path.basename(relative_path))
        return {"year": int(year), "day": 1} if year is not None else None

    def get_all_entrypoints(self) -> List[Dict[str, Any]]:
        years = {self._extract_year_from_filename(f.get("name", "")) for f in self._get_figshare_files()}
        years.discard(None)
        return [{"year": int(y), "day": 1} for y in sorted(years)]

    async def download_async(self, source_url: str, output_path: str, session: Optional[aiohttp.ClientSession] = None) -> None:
        from src.data.common.fetch.http import download_with_retries

        await asyncio.sleep(0.3)

        if session is None:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=600, connect=60)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as sess:
                await download_with_retries(sess, source_url, output_path)
        else:
            await download_with_retries(session, source_url, output_path)

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
                source_id=self.ID,
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

    # -- PREPARE ("annual") ---------------------------------------------

    def _resolve_source_file_path(self, file_path: str) -> str:
        if os.path.isabs(file_path) or (self.ctx.data_root and file_path.startswith(self.ctx.data_root)):
            return file_path
        return os.path.join(self.output_root(PipelineStep.FETCH), file_path)

    @staticmethod
    def _select_best_file_for_year(year_files: List[str]) -> str:
        """Prefer .tif > .zip > .tar.gz > .gz -- the order the old code used."""
        if len(year_files) == 1:
            return year_files[0]
        for ext in (".tif", ".zip", ".tar.gz", ".gz"):
            for file_path in year_files:
                if file_path.lower().endswith(ext):
                    return file_path
        return year_files[0]

    def _plan_prepare(self, selection: TargetSelection) -> List[StepTarget]:
        """Ledger-backed fast path: reads existing `artifacts` rows instead
        of re-querying `completed_fetch_files()` and rebuilding the
        year-grouping every call. Falls back to `_discover_prepare()` --
        today's exact live logic -- if no ledger is configured yet, or
        `data reconcile --step prepare` hasn't populated one yet.

        Note this fast path does NOT reproduce the insertion-order quirk
        `_discover_prepare()` preserves (module docstring): `artifacts_for_step`
        (`DataSource._plan_from_ledger`) orders rows by `unit_id`, i.e. by
        year, for every source that uses it -- there is no ledger-level
        concept of "figshare listing order" to replay. That quirk only
        applies to the live-crawl path (and to `plan()` when it falls back
        to it), which is exactly what every existing test here exercises
        (no `data reconcile` has been run against those fixtures).
        """

        def build_target(row: "ArtifactRow", _ledger: Any) -> Optional[StepTarget]:
            year = row.meta.get("year")
            raw_relative_path = row.meta.get("raw_relative_path")
            if year is None or raw_relative_path is None or row.local_path is None:
                return None
            if not selection.matches_year(year):
                return None
            return StepTarget(
                source_id=self.ID,
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
            self.ID, self.ID,
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

        # Quirk: preserve discovery-insertion order, not sorted-by-year
        # (ported from the old Parquet-row-order behavior).
        files_by_year: Dict[int, List[str]] = {}
        for rel_path in relative_paths:
            year = self._extract_year_from_filename(os.path.basename(rel_path))
            if year is None or not selection.matches_year(year):
                continue
            files_by_year.setdefault(year, []).append(rel_path)

        targets = []
        for year, year_files in files_by_year.items():
            if not selection.matches_key(str(year)):
                continue
            selected = self._select_best_file_for_year(year_files)
            targets.append(
                StepTarget(
                    source_id=self.ID,
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
            elif file_path.endswith(".zip"):
                import tempfile
                import zipfile

                extract_dir = tempfile.mkdtemp(prefix="ntl_harm_extract_")
                with zipfile.ZipFile(local_file, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
                extracted = [f for f in os.listdir(extract_dir) if f.endswith(".tif")]
                if not extracted:
                    raise ValueError(f"No .tif file found in zip archive: {file_path}")
                local_file = os.path.join(extract_dir, extracted[0])
                uncompressed_file_to_delete = extract_dir

            if not os.path.exists(local_file):
                logger.error("File does not exist: %s", local_file)
                return None

            ds = rxr.open_rasterio(local_file, chunks="auto")
            ds = ds.expand_dims(dim={"time": 1}).assign_coords({"time": [pd.Timestamp(f"{year}-12-31")]})
            if uncompressed_file_to_delete:
                ds.attrs["_cleanup_file"] = uncompressed_file_to_delete
            return ds
        except Exception:
            logger.exception("Error processing file %s.", file_path)
            return None

    @staticmethod
    def _create_annual_zarr(data_array: xr.DataArray, output_path: str) -> bool:
        import shutil

        cleanup_file = data_array.attrs.get("_cleanup_file")
        try:
            dataset = data_array.to_dataset(name=NtlHarmSource.VARIABLE_NAME)
            dataset.attrs.pop("_cleanup_file", None)
            dataset = dataset.chunk({"x": 512, "y": 512})
            dataset = dataset.assign_attrs(_FillValue=65535, scale_factor=1, add_offset=0.0)

            compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle", blocksize=0)
            encoding = {var: {"compressors": (compressor,)} for var in dataset.data_vars}
            dataset.to_zarr(output_path, mode="w", encoding=encoding)

            if cleanup_file and os.path.exists(cleanup_file):
                if os.path.isdir(cleanup_file):
                    shutil.rmtree(cleanup_file)
                else:
                    os.remove(cleanup_file)
            return True
        except Exception:
            logger.exception("Error creating zarr file at %s.", output_path)
            if cleanup_file and os.path.exists(cleanup_file):
                try:
                    if os.path.isdir(cleanup_file):
                        shutil.rmtree(cleanup_file)
                    else:
                        os.remove(cleanup_file)
                except OSError:
                    pass
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
            "ntl_harm_timeseries_reprojected.zarr",
            namespace=self.cfg.namespace,
            grid_id=self.ctx.grid_id,
            layout=self.ctx.layout,
            v2_family="ntl_harm",
        )

    def _discover_grid(self, selection: TargetSelection) -> List[StepTarget]:
        """Live ground truth for GRID: crawls the PREPARE output directory
        for annual zarrs. Called from `discover()` (via `data reconcile`)
        and as `_plan_grid()`'s fallback when the ledger has no GRID row yet."""
        annual_files = [f for f in self._list_annual_zarrs() if selection.matches_year(f["year"])]
        if not annual_files:
            return []
        return [
            StepTarget(
                source_id=self.ID,
                step=PipelineStep.GRID,
                key="all",
                output_path=self._grid_output_path(),
                inputs=tuple(f["zarr_path"] for f in annual_files),
                completion=Completion.MARKER,
                meta={
                    "years_available": [f["year"] for f in annual_files],
                    **verify.verification_meta(
                        self.cfg.raw, expected_vars=(self.VARIABLE_NAME,), value_range=(0, 2000)
                    ),
                },
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
                source_id=self.ID,
                step=PipelineStep.GRID,
                key=row.unit_id,
                output_path=row.local_path,
                inputs=tuple(path for _, path in annual),
                completion=Completion.MARKER,
                meta={
                    "years_available": years,
                    **verify.verification_meta(
                        self.cfg.raw, expected_vars=(self.VARIABLE_NAME,), value_range=(0, 2000)
                    ),
                },
            )

        targets = self._plan_from_ledger(PipelineStep.GRID, selection, build_target)
        if targets is not None:
            return targets
        logger.warning(
            "No ledger for source='%s' step='grid' -- falling back to live discovery; "
            "run `data reconcile --source %s --step grid` for faster planning.",
            self.ID, self.ID,
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
    NtlHarmSource.ID,
    __name__,
    NtlHarmSource.__name__,
    NtlHarmSource.STEPS,
    aliases=NtlHarmSource.ALIASES,
)

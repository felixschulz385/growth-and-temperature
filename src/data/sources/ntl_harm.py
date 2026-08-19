"""Harmonized DMSP-VIIRS nighttime lights (Figshare): fetch + prepare.

PREPARE goes straight from a year's raw fetched file to the reprojected
output, tile by tile (`src.data.common.prepare.driver.run_tiled_prepare`);
there is no intermediate whole-extent-per-year "annual" zarr and no separate
GRID step.

PREPARE is planned by a live crawl of FETCH's raw output directory
(`_files_by_year()`, sorted filenames). There is exactly one PREPARE target
(`key="all"`, one tiled output). The `.tif > .zip > .tar.gz > .gz`
best-file-for-year preference (`_select_best_file_for_year`) is the opposite
direction from acag/esacci's nc4-over-nc, and intentional.
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
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
import requests
import xarray as xr

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection
from src.data.sources import verify

logger = logging.getLogger(__name__)


class NtlHarmSource(DataSource):
    """Harmonized DMSP-VIIRS nighttime lights, Figshare dataset 9828827.

    FETCH   -- one Figshare file per year (cached article listing, 1h TTL).
    PREPARE -- reproject every year's raw file directly onto the canonical
               geobox, tile by tile; radiance field, so resampling defaults
               to area-weighted "sum" (docs/design/04-ingest.md §1), not
               SpatialProcessor's own "nearest" default.
    """

    ID = "ntl_harm"
    ALIASES = ("ntlharm", "harmonized_ntl")
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE)

    DATA_SOURCE_NAME = "ntl_harm"
    has_entrypoints = True
    STATIC_ENTRYPOINTS = True  # get_all_entrypoints() below is cfg.year_range-derived, no network call
    RAW_LISTING_DEPTH = 1  # flat filename, see list_remote_files() below

    VARIABLE_NAME = "ntl_harm"
    FIGSHARE_API_BASE = "https://api.figshare.com/v2"
    DATASET_ID = "9828827"

    #: Bumped whenever the raw-getter/reprojection logic here changes in a
    #: way that must invalidate every already-`complete` tile's status and
    #: force a full reprocess (`run_tiled_prepare`'s `processing_version`).
    PROCESSING_VERSION = "2-tiled"

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="ntl_harm/harmonized")  # old NTLHarmPreprocessor default
        super().__init__(ctx, cfg)
        self.base_url = cfg.raw.get("base_url") or f"{self.FIGSHARE_API_BASE}/articles/{self.DATASET_ID}"
        self.file_extensions = cfg.raw.get("file_extensions") or [".tif", ".zip", ".tar.gz", ".gz"]
        self.resampling = cfg.raw.get("resampling", "sum")
        from src.data.common import tiling

        self.tile_size = int(cfg.raw.get("tile_size", tiling.DEFAULT_TILE_SIZE))
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
        """Static from `cfg.year_range` (one file/year, matching
        `data.yaml`'s `[1992, 2024]`) -- no network call, unlike the old
        `_get_figshare_files()`-derived version this replaces. Resolving
        which figshare file id/URL a given year maps to still needs that
        API listing, but only once per entrypoint crawl
        (`list_remote_files()`, `_get_figshare_files()`'s own 1h cache),
        not just to enumerate which years exist."""
        if not self.cfg.year_range:
            return []
        return [{"year": y, "day": 1} for y in range(self.cfg.year_range[0], self.cfg.year_range[1] + 1)]

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
                source_id=self.ID,
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

    def _files_by_year(self) -> Dict[int, List[str]]:
        """Live crawl of FETCH's raw output directory: ground truth for
        which years have a fetched file (see module docstring)."""
        raw_root = self.output_root(PipelineStep.FETCH)
        if not os.path.isdir(raw_root):
            return {}
        files_by_year: Dict[int, List[str]] = {}
        for fname in sorted(os.listdir(raw_root)):
            if not os.path.isfile(os.path.join(raw_root, fname)):
                continue
            year = self._extract_year_from_filename(fname)
            if year is not None:
                files_by_year.setdefault(year, []).append(fname)
        return files_by_year

    def _output_path(self) -> str:
        return layout.grid_store_path(
            self.ctx.data_root,
            self.cfg.data_path,
            grid_id=self.ctx.grid_id,
            family="ntl_harm",
            suffix="",  # cell_id-keyed parquet parts, not a Zarr store -- see grid_store_path docstring
        )

    def _plan_prepare(self, selection: TargetSelection) -> List[StepTarget]:
        files_by_year = self._files_by_year()
        years = sorted(
            year
            for year in files_by_year
            if selection.matches_year(year) and selection.matches_key(str(year))
        )
        if not years:
            return []
        raw_files = {year: self._select_best_file_for_year(files_by_year[year]) for year in years}
        return [
            StepTarget(
                source_id=self.ID,
                step=PipelineStep.PREPARE,
                key="all",
                output_path=self._output_path(),
                inputs=tuple(raw_files[year] for year in years),
                completion=Completion.MARKER,
                meta={
                    "years": years,
                    "raw_files": raw_files,
                    **verify.verification_meta(
                        self.cfg.raw, expected_vars=(self.VARIABLE_NAME,), value_range=(0, 2000)
                    ),
                },
            )
        ]

    def _load_year(self, file_path: str, year: int) -> Optional[xr.Dataset]:
        """Open (decompressing if needed) one year's raw file, expanded with
        a `time` dim. Handed to every tile's `raw_getter` call for this year
        (cached by the caller, see `_execute_prepare`) -- `xr_reproject`
        crops/warps onto each tile's own geobox internally, so passing the
        same whole-year array to every tile is correct, just not the most
        I/O-efficient possible (acceptable for this source's raster size)."""
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

            da = rxr.open_rasterio(local_file, chunks="auto")
            da = da.expand_dims(dim={"time": 1}).assign_coords({"time": [pd.Timestamp(f"{year}-12-31")]})
            ds = da.to_dataset(name=self.VARIABLE_NAME)
            if ds.rio.crs is None:
                ds = ds.rio.write_crs(4326)
            # Materialize now (uncompressed_file_to_delete is a temp path
            # this method's caller cleans up right after this returns);
            # a dask-lazy array referencing a since-deleted temp file would
            # fail at reproject/compute time, not here.
            ds = ds.load()
            return ds
        except Exception:
            logger.exception("Error loading raw file %s for year %d.", file_path, year)
            return None
        finally:
            if uncompressed_file_to_delete and os.path.exists(uncompressed_file_to_delete):
                import shutil

                if os.path.isdir(uncompressed_file_to_delete):
                    shutil.rmtree(uncompressed_file_to_delete)
                else:
                    os.remove(uncompressed_file_to_delete)

    def _execute_prepare(self, target: StepTarget) -> bool:
        from src.data.common.geobox import get_target_geobox
        from src.data.common.prepare.driver import run_tiled_prepare
        from src.data.common.raster.spatial import SpatialProcessor
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
                cache: Dict[int, Optional[xr.Dataset]] = {}

                def load_year(year: int) -> Optional[xr.Dataset]:
                    if year not in cache:
                        source_file = self._resolve_source_file_path(raw_files[year])
                        cache[year] = self._load_year(source_file, year)
                    return cache[year]

                return run_tiled_prepare(
                    output_path=target.output_path,
                    years=years,
                    variables=[self.VARIABLE_NAME],
                    target_geobox=target_geobox,
                    processor=processor,
                    raw_getter=lambda tile, year: load_year(year),
                    tile_size=self.tile_size,
                    resampling=self.resampling,
                    processing_version=self.PROCESSING_VERSION,
                    override=self.cfg.override,
                )

    # _dask_client: inherited from DataSource (src/data/sources/base.py).

registry.register(
    NtlHarmSource.ID,
    __name__,
    NtlHarmSource.__name__,
    NtlHarmSource.STEPS,
    aliases=NtlHarmSource.ALIASES,
)

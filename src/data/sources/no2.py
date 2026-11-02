"""Global surface NO2, 1 km (Anenberg et al. 2022, figshare): fetch + prepare.

Annual ground-level NO2 in ppb on a 30 arc-second WGS84 grid (60 S-75 N),
one LZW GeoTIFF per year for 1990, 1995, 2000 and 2005-2020 (figshare
article 12968114, version 4, CC BY 4.0; 0.29-0.62 GB per file, about 8 GB
in all, so FETCH runs on the transfer node). A 2010-2012 land-use regression
surface is scaled to other years with satellite NO2 columns, so with region
or cell FE the variation over time comes from the satellite scaling.

PREPARE follows ``ntl_harm``: each year's raw file is reprojected straight
onto the canonical geobox, tile by tile (``run_tiled_prepare``), with
area-weighted ``average`` resampling (a concentration, not a total). The
raw nodata value -999 is masked on load.
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
import numpy as np
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


class No2Source(DataSource):
    """Anenberg et al. (2022) annual surface NO2, figshare article 12968114.

    FETCH   -- one figshare GeoTIFF per year (cached article listing, 1h TTL).
    PREPARE -- reproject every year's raw file onto the canonical geobox,
               tile by tile, with ``average`` resampling.
    """

    ID = "no2"
    ALIASES = ("no2_anenberg", "surface_no2")
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE)

    DATA_SOURCE_NAME = "no2"
    # "manual": FETCH runs on the transfer node, which mounts the same data_root,
    # so there is no separate HPC copy to push to or check against (an "auto"
    # push to the same path, then local cleanup, would delete the only copy).
    DEFAULT_TRANSFER_MODE = "manual"
    has_entrypoints = True
    STATIC_ENTRYPOINTS = True  # get_all_entrypoints() is KNOWN_YEARS-derived, no network call
    RAW_LISTING_DEPTH = 1  # flat filenames, "<year>_final_1km.tif"

    VARIABLE_NAME = "no2"
    FIGSHARE_API_BASE = "https://api.figshare.com/v2"
    DATASET_ID = "12968114"
    DATASET_VERSION = 4
    #: The years the dataset has: 1990, 1995, 2000 and every year 2005-2020.
    KNOWN_YEARS: Tuple[int, ...] = (1990, 1995, 2000, *range(2005, 2021))
    NODATA = -999.0

    #: Bumped whenever the raw-getter/reprojection logic here changes in a
    #: way that must invalidate every already-`complete` tile's status.
    PROCESSING_VERSION = "1-tiled"

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="no2/anenberg")
        super().__init__(ctx, cfg)
        version = cfg.raw.get("version", self.DATASET_VERSION)
        self.base_url = cfg.raw.get("base_url") or f"{self.FIGSHARE_API_BASE}/articles/{self.DATASET_ID}/versions/{version}"
        self.resampling = cfg.raw.get("resampling", "average")
        from src.data.common import tiling

        self.tile_size = int(cfg.raw.get("tile_size", tiling.DEFAULT_TILE_SIZE))
        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix="no2_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)
        self._files_cache: Optional[List[Dict[str, Any]]] = None
        self._cache_timestamp: Optional[float] = None
        self._cache_duration = 3600

    # ------------------------------------------------------------------
    # RemoteFileCatalog contract
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_year_from_filename(filename: str) -> Optional[int]:
        match = re.match(r"(\d{4})_final_1km\.tif$", filename)
        return int(match.group(1)) if match else None

    # get_file_hash: inherited from DataSource (src/data/sources/base.py).

    def _years(self) -> List[int]:
        lo, hi = self.cfg.year_range or (min(self.KNOWN_YEARS), max(self.KNOWN_YEARS))
        return [y for y in self.KNOWN_YEARS if lo <= y <= hi]

    def _get_figshare_files(self) -> List[Dict[str, Any]]:
        now = time.time()
        if self._files_cache is not None and self._cache_timestamp is not None and now - self._cache_timestamp < self._cache_duration:
            return self._files_cache
        try:
            response = requests.get(self.base_url, timeout=60)
            response.raise_for_status()
            files = response.json().get("files", [])
            self._files_cache = files
            self._cache_timestamp = now
            return files
        except (requests.RequestException, json.JSONDecodeError):
            logger.exception("Error fetching figshare file listing.")
            return []

    def list_remote_files(self, entrypoint: Optional[dict] = None) -> List[Tuple[str, str]]:
        years = set(self._years())
        results = []
        for file_info in self._get_figshare_files():
            filename = file_info.get("name", "")
            download_url = file_info.get("download_url", "")
            year = self._extract_year_from_filename(filename)
            if year is None or not download_url or year not in years:
                continue
            if entrypoint and entrypoint.get("year") and entrypoint["year"] != year:
                continue
            results.append((filename, download_url))
        return results

    def local_path(self, relative_path: str) -> str:
        return os.path.join("data", self.DATA_SOURCE_NAME, relative_path)

    def filename_to_entrypoint(self, relative_path: str) -> Optional[Dict[str, Any]]:
        year = self._extract_year_from_filename(os.path.basename(relative_path))
        return {"year": int(year), "day": 1} if year is not None else None

    def get_all_entrypoints(self) -> List[Dict[str, Any]]:
        return [{"year": y, "day": 1} for y in self._years()]

    async def download_async(self, source_url: str, output_path: str, session: Optional[aiohttp.ClientSession] = None) -> None:
        from src.data.common.fetch.http import download_with_retries

        await asyncio.sleep(0.3)
        if session is None:
            connector = aiohttp.TCPConnector(limit=4, limit_per_host=2)
            timeout = aiohttp.ClientTimeout(total=3600, connect=60)
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
        from src.data.common.fetch.driver import run_fetch

        return run_fetch(self, **self.cfg.raw.get("download", {}))

    # -- PREPARE (raw GeoTIFF -> tiled, reprojected output) ---------------

    def _files_by_year(self) -> Dict[int, str]:
        """Live crawl of FETCH's raw output directory."""
        raw_root = self.output_root(PipelineStep.FETCH)
        if not os.path.isdir(raw_root):
            return {}
        out: Dict[int, str] = {}
        for fname in sorted(os.listdir(raw_root)):
            year = self._extract_year_from_filename(fname)
            if year is not None and os.path.isfile(os.path.join(raw_root, fname)):
                out[year] = fname
        return out

    def _output_path(self) -> str:
        return layout.grid_store_path(
            self.ctx.data_root,
            self.cfg.data_path,
            grid_id=self.ctx.grid_id,
            family=self.VARIABLE_NAME,
            suffix="",
        )

    def _plan_prepare(self, selection: TargetSelection) -> List[StepTarget]:
        files = self._files_by_year()
        years = sorted(y for y in files if selection.matches_year(y) and selection.matches_key(str(y)))
        if not years:
            return []
        raw_files = {y: files[y] for y in years}
        return [
            StepTarget(
                source_id=self.ID,
                step=PipelineStep.PREPARE,
                key="all",
                output_path=self._output_path(),
                inputs=tuple(raw_files[y] for y in years),
                completion=Completion.MARKER,
                meta={
                    "years": years,
                    "raw_files": raw_files,
                    **verify.verification_meta(self.cfg.raw, expected_vars=(self.VARIABLE_NAME,), value_range=(0, 200)),
                },
            )
        ]

    @classmethod
    def _load_year(cls, file_path: str, year: int) -> Optional[xr.Dataset]:
        """One year's raw GeoTIFF as a lazy (dask-backed) Dataset with a
        ``time`` dim; nodata and negative values masked. The caller clips to
        each tile's bbox before computing."""
        import rioxarray as rxr

        try:
            da = rxr.open_rasterio(file_path, chunks="auto", masked=True)
            if "band" in da.dims:
                da = da.isel(band=0, drop=True)
            da = da.where(da >= 0).astype("float32")
            da = da.expand_dims(dim={"time": 1}).assign_coords(time=[pd.Timestamp(f"{year}-12-31")])
            ds = da.to_dataset(name=cls.VARIABLE_NAME)
            if ds.rio.crs is None:
                ds = ds.rio.write_crs(4326)
            return ds
        except Exception:
            logger.exception("Error loading raw file %s for year %d.", file_path, year)
            return None

    def _execute_prepare(self, target: StepTarget) -> bool:
        from src.data.common.geobox import get_target_geobox
        from src.data.common.prepare.driver import run_tiled_prepare
        from src.data.common.raster.spatial import SpatialProcessor, clip_to_tile
        from src.data.sources.steps import is_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping PREPARE -- already complete: %s", target.output_path)
            return True

        years: List[int] = target.meta["years"]
        raw_files: Dict[int, str] = target.meta["raw_files"]
        raw_root = self.output_root(PipelineStep.FETCH)
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
                # One year's lazy Dataset at a time (run_tiled_prepare walks
                # years-major); see ntl_harm for the same pattern.
                cache: Dict[int, Optional[xr.Dataset]] = {}

                def year_ds(year: int) -> Optional[xr.Dataset]:
                    if year not in cache:
                        for old_year, old_ds in list(cache.items()):
                            if old_ds is not None:
                                old_ds.close()
                            del cache[old_year]
                        cache[year] = self._load_year(os.path.join(raw_root, raw_files[year]), year)
                    return cache[year]

                def raw_getter(tile, year: int) -> Optional[xr.Dataset]:
                    ds = year_ds(year)
                    if ds is None:
                        return None
                    clipped = clip_to_tile(ds, tile.geobox, y_dim="y", x_dim="x")
                    if clipped.sizes.get("x", 0) == 0 or clipped.sizes.get("y", 0) == 0:
                        # outside the raster's 60 S-75 N extent: NaN-fill, not a failure
                        dim_y, dim_x = tile.geobox.dims
                        return xr.Dataset(
                            {self.VARIABLE_NAME: ((dim_y, dim_x), np.full(tile.geobox.shape, np.nan, dtype=np.float32))}
                        )
                    return clipped.compute()

                try:
                    return run_tiled_prepare(
                        output_path=target.output_path,
                        years=years,
                        variables=[self.VARIABLE_NAME],
                        target_geobox=target_geobox,
                        processor=processor,
                        raw_getter=raw_getter,
                        tile_size=self.tile_size,
                        resampling=self.resampling,
                        processing_version=self.PROCESSING_VERSION,
                        override=self.cfg.override,
                    )
                finally:
                    for ds in cache.values():
                        if ds is not None:
                            ds.close()

    # _dask_client: inherited from DataSource (src/data/sources/base.py).


registry.register(
    No2Source.ID,
    __name__,
    No2Source.__name__,
    No2Source.STEPS,
    aliases=No2Source.ALIASES,
)

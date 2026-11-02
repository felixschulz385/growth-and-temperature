"""GHS-POP R2023A population grid (JRC GHSL): fetch + prepare.

Residential population per cell at 1 km in World Mollweide (ESRI:54009),
one global ZIP per five-year epoch, 1975-2030 (2025 and 2030 are
projections; CC BY 4.0; ~300 MB per ZIP, so FETCH runs on the transfer
node). A Mollweide cell is exactly 1 km^2, so the value is persons per km^2.
Both grids are equal-area, so PREPARE reprojects the density with
``average`` resampling onto the EASE grid; persons per output cell are the
density times the EASE cell area (about 1.002 km^2 at 1 km).

Only epoch years are written (``part-<epoch>.parquet``). Annual values
between epochs are left to the analysis (interpolation), not made here.
The ZIP is read in place through GDAL's ``/vsizip/``; the raw nodata value
(-200) is masked on load.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd
import xarray as xr

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection
from src.data.sources import verify

logger = logging.getLogger(__name__)


class GhsPopSource(DataSource):
    """GHS-POP R2023A, 1 km Mollweide, five-year epochs.

    FETCH   -- one ZIP per epoch from the JRC open-data FTP (static URLs).
    PREPARE -- reproject each epoch's GeoTIFF (read inside the ZIP) onto the
               canonical geobox, tile by tile, with ``average`` resampling.
    """

    ID = "ghs_pop"
    ALIASES = ("ghsl_pop", "population")
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE)

    DATA_SOURCE_NAME = "ghs_pop"
    # "manual": FETCH runs on the transfer node, which mounts the same data_root,
    # so there is no separate HPC copy to push to or check against (an "auto"
    # push to the same path, then local cleanup, would delete the only copy).
    DEFAULT_TRANSFER_MODE = "manual"
    has_entrypoints = True
    STATIC_ENTRYPOINTS = True  # epochs are fixed; URLs follow a pattern, no network call
    RAW_LISTING_DEPTH = 1

    VARIABLE_NAME = "pop"
    RELEASE = "R2023A"
    BASE_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A"
    EPOCHS: Tuple[int, ...] = tuple(range(1975, 2031, 5))
    RESOLUTION_M = 1000

    PROCESSING_VERSION = "1-tiled"

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="ghs_pop/r2023a")
        super().__init__(ctx, cfg)
        self.base_url = cfg.raw.get("base_url", self.BASE_URL).rstrip("/")
        self.resampling = cfg.raw.get("resampling", "average")
        from src.data.common import tiling

        self.tile_size = int(cfg.raw.get("tile_size", tiling.DEFAULT_TILE_SIZE))
        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix="ghs_pop_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # RemoteFileCatalog contract
    # ------------------------------------------------------------------

    def _stem(self, epoch: int) -> str:
        return f"GHS_POP_E{epoch}_GLOBE_{self.RELEASE}_54009_{self.RESOLUTION_M}"

    def _zip_name(self, epoch: int) -> str:
        return f"{self._stem(epoch)}_V1_0.zip"

    @staticmethod
    def _extract_year_from_filename(filename: str) -> Optional[int]:
        match = re.match(r"GHS_POP_E(\d{4})_GLOBE_", filename)
        return int(match.group(1)) if match else None

    # get_file_hash: inherited from DataSource (src/data/sources/base.py).

    def _epochs(self) -> List[int]:
        lo, hi = self.cfg.year_range or (min(self.EPOCHS), max(self.EPOCHS))
        return [y for y in self.EPOCHS if lo <= y <= hi]

    def list_remote_files(self, entrypoint: Optional[dict] = None) -> List[Tuple[str, str]]:
        results = []
        for epoch in self._epochs():
            if entrypoint and entrypoint.get("year") and entrypoint["year"] != epoch:
                continue
            name = self._zip_name(epoch)
            results.append((name, f"{self.base_url}/{self._stem(epoch)}/V1-0/{name}"))
        return results

    def local_path(self, relative_path: str) -> str:
        return os.path.join("data", self.DATA_SOURCE_NAME, relative_path)

    def filename_to_entrypoint(self, relative_path: str) -> Optional[Dict[str, Any]]:
        year = self._extract_year_from_filename(os.path.basename(relative_path))
        return {"year": int(year), "day": 1} if year is not None else None

    def get_all_entrypoints(self) -> List[Dict[str, Any]]:
        return [{"year": y, "day": 1} for y in self._epochs()]

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

    # -- PREPARE (epoch ZIP -> tiled, reprojected output) -------------------

    def _files_by_year(self) -> Dict[int, str]:
        raw_root = self.output_root(PipelineStep.FETCH)
        if not os.path.isdir(raw_root):
            return {}
        out: Dict[int, str] = {}
        for fname in sorted(os.listdir(raw_root)):
            year = self._extract_year_from_filename(fname)
            if year is not None and fname.endswith(".zip") and os.path.isfile(os.path.join(raw_root, fname)):
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
                    **verify.verification_meta(
                        self.cfg.raw, expected_vars=(self.VARIABLE_NAME,), value_range=(0, 1_000_000)
                    ),
                },
            )
        ]

    @classmethod
    def _load_epoch(cls, path: str, year: int) -> Optional[xr.Dataset]:
        """One epoch as a lazy Dataset with a ``time`` dim. *path* is the
        ZIP; the GeoTIFF inside shares its stem. Nodata and negative values
        are masked; float64 is narrowed to float32."""
        import rioxarray as rxr

        try:
            if path.endswith(".zip"):
                tif = os.path.basename(path)[: -len(".zip")] + ".tif"
                path = f"/vsizip/{path}/{tif}"
            da = rxr.open_rasterio(path, chunks="auto", masked=True)
            if "band" in da.dims:
                da = da.isel(band=0, drop=True)
            da = da.where(da >= 0).astype("float32")
            da = da.expand_dims(dim={"time": 1}).assign_coords(time=[pd.Timestamp(f"{year}-12-31")])
            return da.to_dataset(name=cls.VARIABLE_NAME)
        except Exception:
            logger.exception("Error loading GHS-POP epoch %d from %s.", year, path)
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
                cache: Dict[int, Optional[xr.Dataset]] = {}

                def year_ds(year: int) -> Optional[xr.Dataset]:
                    if year not in cache:
                        for old_year, old_ds in list(cache.items()):
                            if old_ds is not None:
                                old_ds.close()
                            del cache[old_year]
                        cache[year] = self._load_epoch(os.path.join(raw_root, raw_files[year]), year)
                    return cache[year]

                def raw_getter(tile, year: int) -> Optional[xr.Dataset]:
                    ds = year_ds(year)
                    if ds is None:
                        return None
                    clipped = clip_to_tile(ds, tile.geobox, y_dim="y", x_dim="x")
                    if clipped.sizes.get("x", 0) == 0 or clipped.sizes.get("y", 0) == 0:
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
    GhsPopSource.ID,
    __name__,
    GhsPopSource.__name__,
    GhsPopSource.STEPS,
    aliases=GhsPopSource.ALIASES,
)

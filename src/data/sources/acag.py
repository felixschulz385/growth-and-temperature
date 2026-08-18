"""ACAG (Atmospheric Composition Analysis Group) PM2.5: fetch + prepare.

This is the shape every other bulk-archive raster source (esacci, eog,
glass) follows too. PREPARE reprojects straight from each year's raw
.nc/.nc4 file to the tiled output (`src.data.common.prepare.driver.
run_tiled_prepare`); there is no intermediate whole-extent annual zarr and
no separate GRID step.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
import requests
import rioxarray as rxr
import xarray as xr

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection
from src.data.sources import verify

logger = logging.getLogger(__name__)


class AcagSource(DataSource):
    """ACAG annual PM2.5 surface concentration data.

    FETCH   -- download the hardcoded Box-shared-folder .nc inventory.
    PREPARE -- reproject every year's raw .nc/.nc4 file directly onto the
               canonical geobox, tile by tile, into one multi-year
               timeseries zarr.
    """

    ID = "acag"
    ALIASES = ("acag_pm25", "pm25")
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE)

    # -- RemoteFileCatalog contract (src/data/sources/base.py) --------------
    DATA_SOURCE_NAME = "acag"
    has_entrypoints = True
    STATIC_ENTRYPOINTS = True  # get_all_entrypoints() derives from the static KNOWN_FILES list, no network call
    RAW_LISTING_DEPTH = 3  # GL/Annual/<file>, see KNOWN_FILES below

    # -- fetch-side inventory, unchanged from ACAGDataSource -----------------
    SHARED_LINK_URL = "https://wustl.app.box.com/s/y143mciw7jz7ft2qe3hccjw65m3xe8f2"
    KNOWN_FILES: List[Tuple[str, str]] = [
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
    _PM25_CANDIDATES = ["GWRPM25", "PM25", "pm25", "PM2_5", "pm2_5", "Annual_PM2.5"]

    #: Bumped whenever the raw-getter/reprojection logic here changes in a
    #: way that must invalidate every already-`complete` tile's status and
    #: force a full reprocess (`run_tiled_prepare`'s `processing_version`).
    PROCESSING_VERSION = "2-tiled"

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="acag/pm25")  # old ACAGPreprocessor default
        super().__init__(ctx, cfg)
        self.shared_name = self.SHARED_LINK_URL.rstrip("/").split("/")[-1]
        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix="acag_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)

        from src.data.common import tiling

        self.tile_size = int(cfg.raw.get("tile_size", tiling.DEFAULT_TILE_SIZE))

    # ------------------------------------------------------------------
    # RemoteFileCatalog contract (ports ACAGDataSource verbatim)
    # ------------------------------------------------------------------

    @staticmethod
    def _browser_headers() -> Dict[str, str]:
        return {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def _file_download_url(self, file_id: str) -> str:
        return (
            f"https://wustl.app.box.com/index.php"
            f"?rm=box_download_shared_file&shared_name={self.shared_name}&file_id=f_{file_id}"
        )

    # _extract_year: inherited from DataSource (src/data/sources/base.py).

    def list_remote_files(self, entrypoint: Optional[dict] = None) -> List[Tuple[str, str]]:
        results = []
        for rel_path, file_id in self.KNOWN_FILES:
            if entrypoint:
                year_filter = entrypoint.get("year")
                if year_filter is not None and self._extract_year(os.path.basename(rel_path)) != int(year_filter):
                    continue
            results.append((rel_path, self._file_download_url(file_id)))
        return results

    def local_path(self, relative_path: str) -> str:
        return os.path.join("data", self.DATA_SOURCE_NAME, relative_path)

    # get_file_hash: inherited from DataSource (src/data/sources/base.py).

    def filename_to_entrypoint(self, relative_path: str) -> Optional[Dict[str, Any]]:
        year = self._extract_year(os.path.basename(relative_path))
        return {"year": int(year), "day": 1} if year is not None else None

    def get_all_entrypoints(self) -> List[Dict[str, Any]]:
        years = {self._extract_year(os.path.basename(p)) for p, _ in self.KNOWN_FILES}
        years.discard(None)
        return [{"year": int(y), "day": 1} for y in sorted(years)]

    async def download_async(self, source_url: str, output_path: str, session: Optional[aiohttp.ClientSession] = None) -> None:
        from src.data.common.fetch.http import download_with_retries

        await asyncio.sleep(0.2)  # polite rate-limiting
        headers = self._browser_headers()

        if session is None:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=600, connect=60)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as sess:
                await download_with_retries(sess, source_url, output_path, headers=headers)
        else:
            await download_with_retries(session, source_url, output_path, headers=headers)

    # ------------------------------------------------------------------
    # plan()/execute() dispatch
    # ------------------------------------------------------------------

    def _plan(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        if step is PipelineStep.FETCH:
            return self._plan_fetch()
        if step is PipelineStep.PREPARE:
            return self._plan_prepare(selection)
        raise AssertionError(f"unreachable: {step}")  # _require_step already validated this

    def _execute(self, target: StepTarget) -> bool:
        if target.step is PipelineStep.FETCH:
            return self._execute_fetch(target)
        if target.step is PipelineStep.PREPARE:
            return self._execute_prepare(target)
        raise AssertionError(f"unreachable: {target.step}")

    # ------------------------------------------------------------------
    # FETCH -- one target = "sync + download whatever is missing", delegating
    # to common.fetch.driver.run_fetch (docs/design/10-fetch-ledger.md).
    # ------------------------------------------------------------------

    def _plan_fetch(self) -> List[StepTarget]:
        return [
            StepTarget(
                source_id=self.ID,
                step=PipelineStep.FETCH,
                key="all",
                output_path=self.output_root(PipelineStep.FETCH),
                completion=Completion.NEVER,  # always safe to re-run: only fetches what's missing
            )
        ]

    def _execute_fetch(self, target: StepTarget) -> bool:
        # FETCH is local-disk only now -- no HPC target required. `data
        # transfer` (separate, manual or auto per source config) is the only
        # thing that pushes to HPC.
        from src.data.common.fetch.driver import run_fetch

        return run_fetch(self, **self.cfg.raw.get("download", {}))

    # ------------------------------------------------------------------
    # PREPARE (raw .nc/.nc4 file -> tiled, reprojected output)
    # ------------------------------------------------------------------

    def _resolve_raw_path(self, relative_path: str) -> str:
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(self.output_root(PipelineStep.FETCH), relative_path)

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
                year = self._extract_year(fname)
                if year is not None:
                    files_by_year.setdefault(year, []).append(rel)
        return files_by_year

    @staticmethod
    def _select_best_file_for_year(candidates: List[str]) -> str:
        """Prefer .nc4 over .nc."""
        return next(
            (f for f in candidates if f.lower().endswith(".nc4")),
            next((f for f in candidates if f.lower().endswith(".nc")), candidates[0]),
        )

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
                source_id=self.ID,
                step=PipelineStep.PREPARE,
                key="all",
                output_path=self._output_path(),
                inputs=tuple(raw_files[year] for year in years),
                completion=Completion.MARKER,
                meta={
                    "years": years,
                    "raw_files": raw_files,
                    **verify.verification_meta(self.cfg.raw, expected_vars=("pm25",), value_range=(0, 500)),
                },
            )
        ]

    def _load_nc_as_dataset(self, file_path: str, year: int) -> Optional[xr.Dataset]:
        if not os.path.exists(file_path):
            logger.error("File not found: %s", file_path)
            return None
        try:
            data = rxr.open_rasterio(file_path, decode_coords="all", mask_and_scale=True, driver="HDF5")
            if "band" in data.dims and data.sizes.get("band", 1) > 1:
                data = data.isel(band=0)
            ds = data.to_dataset(name="pm25")
            ds.attrs["source_year"] = year
            ds.attrs["source_variable"] = "pm25"

            sp = 0.01
            if "x" in ds.coords:
                ds = ds.assign_coords(x=ds["x"] * sp - 180)
            if "y" in ds.coords:
                ds = ds.assign_coords(y=ds["y"] * sp - 60)

            dim_map = {}
            for d in ds.dims:
                dl = d.lower()
                if dl in ("lat", "latitude", "y"):
                    dim_map[d] = "latitude"
                elif dl in ("lon", "longitude", "x"):
                    dim_map[d] = "longitude"
            if dim_map:
                ds = ds.rename(dim_map)

            try:
                ds = ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
            except Exception:
                pass
            ds = ds.rio.write_crs("EPSG:4326")
            ds = ds.where(ds >= 0).astype("float32")

            if "time" not in ds.dims:
                ds = ds.expand_dims(dim={"time": 1}).assign_coords(time=[pd.Timestamp(f"{year}-12-31")])
            if "band" not in ds.dims:
                ds = ds.expand_dims(dim={"band": 1}).assign_coords(band=[1])
            return ds.transpose("time", "band", "latitude", "longitude")
        except Exception:
            logger.exception("Error loading %s.", file_path)
            return None

    def _output_path(self) -> str:
        return layout.grid_store_path(
            self.ctx.data_root,
            self.cfg.data_path,
            grid_id=self.ctx.grid_id,
            family="pm25",
            suffix="",  # cell_id-keyed parquet parts, not a Zarr store -- see grid_store_path docstring
        )

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
                        source_file = self._resolve_raw_path(raw_files[year])
                        cache[year] = self._load_nc_as_dataset(source_file, year)
                    return cache[year]

                return run_tiled_prepare(
                    output_path=target.output_path,
                    years=years,
                    variables=["pm25"],
                    target_geobox=target_geobox,
                    processor=processor,
                    raw_getter=lambda tile, year: load_year(year),
                    tile_size=self.tile_size,
                    processing_version=self.PROCESSING_VERSION,
                    override=self.cfg.override,
                )

    # _dask_client: inherited from DataSource (src/data/sources/base.py).

registry.register(
    AcagSource.ID,
    __name__,
    AcagSource.__name__,
    AcagSource.STEPS,
    aliases=AcagSource.ALIASES,
)

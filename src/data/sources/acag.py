"""ACAG (Atmospheric Composition Analysis Group) PM2.5: fetch + prepare + grid.

docs/design/09-integrated-pipeline.md §5: the reference migration -- the shape
every other bulk-archive raster source (esacci, eog, ntl_harm, glass) follows.
Merges `src/data/download/sources/acag.py::ACAGDataSource` (remote inventory /
fetch) and `src/data/preprocess/sources/acag.py::ACAGPreprocessor`
(`stage="annual"` -> PREPARE, `stage="spatial"` -> GRID) into one class,
unchanged in behaviour -- checked against
`tests/data/preprocess/sources/test_characterization_acag.py`'s pinned values
from the old code, and against `tests/data/sources/acag/test_plan.py`'s
equivalent assertions for this class.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
import aiohttp
import pandas as pd
import requests
import rioxarray as rxr
import xarray as xr
from zarr.codecs import BloscCodec

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection

logger = logging.getLogger(__name__)


class AcagSource(DataSource):
    """ACAG annual PM2.5 surface concentration data.

    FETCH   -- download the hardcoded Box-shared-folder .nc inventory.
    PREPARE -- one annual zarr per year, extracted from the raw .nc/.nc4 file
               (dims (time=1, band=1, latitude, longitude), EPSG:4326).
    GRID    -- reproject every annual zarr onto the canonical geobox into one
               multi-year timeseries zarr.
    """

    ID = "acag"
    ALIASES = ("acag_pm25", "pm25")
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE, PipelineStep.GRID)

    # -- RemoteFileCatalog contract (src/data/sources/base.py) --------------
    DATA_SOURCE_NAME = "acag"
    has_entrypoints = True

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

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="acag/pm25")  # old ACAGPreprocessor default
        super().__init__(ctx, cfg)
        self.shared_name = self.SHARED_LINK_URL.rstrip("/").split("/")[-1]
        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix="acag_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)

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

    def get_file_hash(self, file_url: str) -> str:
        return hashlib.md5(file_url.encode("utf-8")).hexdigest()

    def filename_to_entrypoint(self, relative_path: str) -> Optional[Dict[str, Any]]:
        year = self._extract_year(os.path.basename(relative_path))
        return {"year": int(year), "day": 1} if year is not None else None

    def get_all_entrypoints(self) -> List[Dict[str, Any]]:
        years = {self._extract_year(os.path.basename(p)) for p, _ in self.KNOWN_FILES}
        years.discard(None)
        return [{"year": int(y), "day": 1} for y in sorted(years)]

    async def download_async(self, source_url: str, output_path: str, session: Optional[aiohttp.ClientSession] = None) -> None:
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
                        return
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    if attempt < max_retries - 1:
                        await asyncio.sleep((attempt + 1) * 2)
                    else:
                        raise

        if session is None:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=600, connect=60)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as sess:
                await _do_download(sess)
        else:
            await _do_download(session)

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
        raise AssertionError(f"unreachable: {step}")  # _require_step already validated this

    def _execute(self, target: StepTarget) -> bool:
        if target.step is PipelineStep.FETCH:
            return self._execute_fetch(target)
        if target.step is PipelineStep.PREPARE:
            return self._execute_prepare(target)
        if target.step is PipelineStep.GRID:
            return self._execute_grid(target)
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
        if not self.ctx.ssh_target:
            logger.warning("Fetch requires an HPC/remote target to be configured.")
            return False

        from src.data.common.fetch.driver import run_fetch

        return run_fetch(self, **self.cfg.raw.get("download", {}))

    # ------------------------------------------------------------------
    # PREPARE ("annual" in the old vocabulary)
    # ------------------------------------------------------------------

    def _resolve_raw_path(self, relative_path: str) -> str:
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(self.output_root(PipelineStep.FETCH), relative_path)

    def _plan_prepare(self, selection: TargetSelection) -> List[StepTarget]:
        from src.data.common.ledger.paths import ledger_path
        from src.data.common.ledger.store import SourceLedger

        local_ledger_path = ledger_path(self.ctx.local_index_dir, self.data_path)
        if not local_ledger_path or not os.path.exists(local_ledger_path):
            logger.warning("Ledger not found: %s", local_ledger_path)
            return []

        with SourceLedger.open(local_ledger_path, data_path=self.data_path, read_only=True) as ledger:
            relative_paths = ledger.completed_fetch_files()
        if not relative_paths:
            return []

        files_by_year: Dict[int, List[str]] = {}
        for rel_path in relative_paths:
            year = self._extract_year(os.path.basename(rel_path))
            if year is None or not selection.matches_year(year):
                continue
            files_by_year.setdefault(year, []).append(rel_path)

        targets = []
        for year in sorted(files_by_year):
            if not selection.matches_key(str(year)):
                continue
            candidates = files_by_year[year]
            selected = next(
                (f for f in candidates if f.lower().endswith(".nc4")),
                next((f for f in candidates if f.lower().endswith(".nc")), candidates[0]),
            )
            targets.append(
                StepTarget(
                    source_id=self.ID,
                    step=PipelineStep.PREPARE,
                    key=str(year),
                    output_path=os.path.join(self.output_root(PipelineStep.PREPARE), f"{year}.zarr"),
                    inputs=(selected,),
                    completion=Completion.MARKER,
                    meta={"year": year, "raw_candidates": len(candidates)},
                )
            )
        return targets

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

    @staticmethod
    def _write_annual_zarr(ds: xr.Dataset, output_path: str) -> bool:
        try:
            ds = ds.chunk({"time": 1, "band": 1, "latitude": 512, "longitude": 512})
            compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle", blocksize=0)
            encoding = {var: {"compressors": (compressor,), "chunks": (1, 1, 512, 512)} for var in ds.data_vars}
            ds.to_zarr(output_path, mode="w", encoding=encoding, zarr_format=3, consolidated=False)
            return True
        except Exception:
            logger.exception("Error writing annual zarr to %s.", output_path)
            return False

    def _execute_prepare(self, target: StepTarget) -> bool:
        from src.data.sources.steps import is_complete, mark_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping year %s -- already complete: %s", target.key, target.output_path)
            return True

        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)
        year = target.meta["year"]
        raw_abs = self._resolve_raw_path(target.inputs[0])

        ds = self._load_nc_as_dataset(raw_abs, year)
        if ds is None:
            return False
        if not self._write_annual_zarr(ds, target.output_path):
            return False
        mark_complete(target.output_path)
        return True

    # ------------------------------------------------------------------
    # GRID ("spatial" in the old vocabulary)
    # ------------------------------------------------------------------

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

    def _plan_grid(self, selection: TargetSelection) -> List[StepTarget]:
        annual_files = self._list_annual_zarrs()
        annual_files = [f for f in annual_files if selection.matches_year(f["year"])]
        if not annual_files:
            return []
        return [
            StepTarget(
                source_id=self.ID,
                step=PipelineStep.GRID,
                key="all",
                output_path=layout.grid_store_path(
                    self.ctx.data_root,
                    self.cfg.data_path,
                    "acag_pm25_timeseries_reprojected.zarr",
                    namespace=self.cfg.namespace,
                    grid_id=self.ctx.grid_id,
                    layout=self.ctx.layout,
                    v2_family="pm25",
                ),
                inputs=tuple(f["zarr_path"] for f in annual_files),
                completion=Completion.MARKER,
                meta={"years_available": [f["year"] for f in annual_files]},
            )
        ]

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
                        ds = ds.rio.write_crs("EPSG:4326")
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
                )
                if success:
                    mark_complete(target.output_path)
                return success

    # _dask_client: inherited from DataSource (src/data/sources/base.py).

registry.register(
    AcagSource.ID,
    __name__,
    AcagSource.__name__,
    AcagSource.STEPS,
    aliases=AcagSource.ALIASES,
)

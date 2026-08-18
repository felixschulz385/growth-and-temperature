"""ESA CCI Land Cover annual composites: fetch + prepare.

PREPARE reprojects straight from each year's raw (possibly zip-wrapped)
NetCDF file to the tiled output; there is no intermediate annual zarr and
no separate GRID step.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import shutil
import tempfile
import zipfile
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse

import pandas as pd
import xarray as xr

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection
from src.data.sources import verify

logger = logging.getLogger(__name__)


def _cdsapi_url(year: int, versions: List[str]) -> str:
    params = [("year", str(year))]
    for v in versions:
        params.append(("version", v))
    return "cdsapi://satellite-land-cover?" + urlencode(params)


def _parse_cdsapi_url(url: str) -> Tuple[str, Dict[str, Any]]:
    parsed = urlparse(url)
    dataset = parsed.netloc
    qs = parse_qs(parsed.query)
    years = qs.get("year", [])
    versions = qs.get("version", [])
    request: Dict[str, Any] = {"variable": "all"}
    if years:
        request["year"] = years
    if versions:
        request["version"] = versions
    return dataset, request


class EsacciSource(DataSource):
    """ESA CCI Land Cover annual composites (categorical LCCS map).

    FETCH   -- one CDS API request per year (satellite-land-cover dataset).
    PREPARE -- reproject every year's raw (zip-wrapped) NetCDF file directly
               onto the canonical geobox, tile by tile (categorical -- always
               nearest, dst_nodata=0, no scale/offset packaging).
    """

    ID = "esacci"
    ALIASES = ("esa_cci", "esacci_lc", "landcover")
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE)

    DATA_SOURCE_NAME = "esacci"
    has_entrypoints = True
    STATIC_ENTRYPOINTS = True  # get_all_entrypoints() is cfg.year_range-derived, no network call
    RAW_LISTING_DEPTH = 2  # <year>/<file>, see list_remote_files() below

    DATASET = "satellite-land-cover"
    DEFAULT_VERSIONS = ["v2_0_7cds", "v2_1_1"]
    DEFAULT_VARIABLES = ["lccs_class"]

    #: Bumped whenever the raw-getter/reprojection logic here changes in a
    #: way that must invalidate every already-`complete` tile's status and
    #: force a full reprocess (`run_tiled_prepare`'s `processing_version`).
    PROCESSING_VERSION = "2-tiled"

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="esacci/landcover")  # old ESACCIPreprocessor default
        super().__init__(ctx, cfg)
        self.versions: List[str] = cfg.raw.get("versions") or self.DEFAULT_VERSIONS
        self.cdsapi_rc: Optional[str] = cfg.raw.get("cdsapi_rc")
        self.variables_to_keep: List[str] = cfg.raw.get("variables_to_keep") or self.DEFAULT_VARIABLES
        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix="esacci_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)

        from src.data.common import tiling

        self.tile_size = int(cfg.raw.get("tile_size", tiling.DEFAULT_TILE_SIZE))

    # ------------------------------------------------------------------
    # RemoteFileCatalog contract (ports ESACCIDataSource verbatim)
    # ------------------------------------------------------------------

    # _extract_year: inherited from DataSource (src/data/sources/base.py).

    def list_remote_files(self, entrypoint: Optional[dict] = None) -> List[Tuple[str, str]]:
        if entrypoint and "year" in entrypoint:
            years = [int(entrypoint["year"])]
        elif self.cfg.year_range:
            years = list(range(self.cfg.year_range[0], self.cfg.year_range[1] + 1))
        else:
            years = list(range(1992, 2023))
        results = []
        for year in years:
            rel_path = f"{year}/ESACCI-LC-L4-LCCS-Map-300m-P1Y-{year}-v2.0.7.nc"
            results.append((rel_path, _cdsapi_url(year, self.versions)))
        return results

    def local_path(self, relative_path: str) -> str:
        return os.path.join("data", self.DATA_SOURCE_NAME, relative_path)

    # get_file_hash: inherited from DataSource (src/data/sources/base.py).

    def filename_to_entrypoint(self, relative_path: str) -> Optional[Dict[str, Any]]:
        year = self._extract_year(os.path.basename(relative_path))
        return {"year": int(year), "day": 1} if year is not None else None

    def get_all_entrypoints(self) -> List[Dict[str, Any]]:
        if self.cfg.year_range:
            years = range(self.cfg.year_range[0], self.cfg.year_range[1] + 1)
        else:
            years = range(1992, 2023)
        return [{"year": y, "day": 1} for y in years]

    def download(self, source_url: str, output_path: str, session: Any = None) -> None:
        """Synchronous CDS API retrieve -- CDS auth reads ~/.cdsapirc, no HTTP session."""
        import cdsapi

        dataset, request = _parse_cdsapi_url(source_url)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        kwargs: Dict[str, Any] = {"rc": self.cdsapi_rc} if self.cdsapi_rc else {}
        try:
            cdsapi.Client(**kwargs).retrieve(dataset, request, output_path)
        except Exception as exc:
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            raise RuntimeError(f"CDS API request failed for {source_url}: {exc}") from exc

    async def download_async(self, source_url: str, output_path: str, session: Any = None) -> None:
        """CDS API is synchronous; run it in a thread-pool executor."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.download, source_url, output_path, None)

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

    # -- PREPARE (raw zip-wrapped NetCDF -> tiled, reprojected output) ----

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
                    **verify.verification_meta(
                        self.cfg.raw, expected_vars=("lccs_class",), value_range=(0, 220)
                    ),
                },
            )
        ]

    def _load_nc_as_dataset(self, file_path: str, year: int) -> Optional[xr.Dataset]:
        if not os.path.exists(file_path):
            logger.error("File not found: %s", file_path)
            return None
        try:
            try:
                with zipfile.ZipFile(file_path) as z:
                    nc_name = next((n for n in z.namelist() if n.endswith(".nc")), None)
                    if nc_name is None:
                        raise ValueError(f"No .nc entry found inside zip: {file_path}")
                    tmp_nc_path = os.path.join(self.temp_dir, f"esacci_{year}_{os.path.basename(nc_name)}")
                    with z.open(nc_name) as f_in, open(tmp_nc_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                nc_path = tmp_nc_path
            except zipfile.BadZipFile:
                nc_path = file_path

            raw = xr.open_dataset(nc_path, engine="h5netcdf", mask_and_scale=False, decode_coords="all", chunks="auto")

            if "lccs_class" not in raw.data_vars:
                logger.error(
                    "Variable 'lccs_class' not found in %s. Available: %s",
                    os.path.basename(file_path), sorted(raw.data_vars),
                )
                raw.close()
                return None

            ds = raw[["lccs_class"]]

            dim_map = {}
            for d in list(ds.dims):
                dl = d.lower()
                if dl in ("lat", "latitude", "y"):
                    dim_map[d] = "latitude"
                elif dl in ("lon", "longitude", "x"):
                    dim_map[d] = "longitude"
            if dim_map:
                ds = ds.rename(dim_map)

            if "time" in ds.dims:
                ds = ds.assign_coords(time=[pd.Timestamp(f"{year}-12-31")])
            else:
                ds = ds.expand_dims(dim={"time": 1}).assign_coords(time=[pd.Timestamp(f"{year}-12-31")])
            if "band" not in ds.dims:
                ds = ds.expand_dims(dim={"band": 1}).assign_coords(band=[1])

            ds = ds.transpose("time", "band", "latitude", "longitude")
            ds = ds.rio.write_crs("EPSG:4326")
            ds.attrs["source_year"] = year
            ds.attrs["source_file"] = os.path.basename(file_path)
            # Materialize now: `nc_path` may be a temp file this method's
            # own logic doesn't clean up on the same schedule as a lazy
            # dask read would need it to stay alive.
            return ds.load()
        except Exception:
            logger.exception("Error loading %s.", file_path)
            return None

    def _output_path(self) -> str:
        return layout.grid_store_path(
            self.ctx.data_root,
            self.cfg.data_path,
            grid_id=self.ctx.grid_id,
            family="land_cover",
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
        dim_y, dim_x = target_geobox.dimensions

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

                sample_ds = load_year(years[0])
                sample_attrs = dict(sample_ds.attrs) if sample_ds is not None else {}

                # ESA CCI LC is categorical -- always nearest (run_tiled_prepare's
                # own default), 0 as nodata, no scale/offset packaging.
                return run_tiled_prepare(
                    output_path=target.output_path,
                    years=years,
                    variables=["lccs_class"],
                    target_geobox=target_geobox,
                    processor=processor,
                    raw_getter=lambda tile, year: load_year(year),
                    target_dims=(dim_y, dim_x),
                    tile_size=self.tile_size,
                    dst_nodata=0,
                    packaging_attrs={},
                    dtype="uint16",
                    sample_attrs=sample_attrs,
                    processing_version=self.PROCESSING_VERSION,
                    override=self.cfg.override,
                )

    # _dask_client: inherited from DataSource (src/data/sources/base.py).

registry.register(
    EsacciSource.ID,
    __name__,
    EsacciSource.__name__,
    EsacciSource.STEPS,
    aliases=EsacciSource.ALIASES,
)

"""ESA CCI Land Cover annual composites: fetch + prepare + grid.

docs/design/09-integrated-pipeline.md §5: same shape as `src/data/sources/acag.py`
(the reference migration). Merges `src/data/download/sources/esacci.py::ESACCIDataSource`
(CDS API fetch, virtual `cdsapi://` URIs) and
`src/data/preprocess/sources/esacci.py::ESACCIPreprocessor` (`stage="annual"`
-> PREPARE, `stage="spatial"` -> GRID) into one class, unchanged in behaviour
-- checked against
`tests/data/preprocess/sources/test_characterization_esacci.py`'s pinned
values from the old code, and `tests/data/sources/esacci/test_plan.py`'s
equivalent assertions for this class.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import os
import shutil
import tempfile
import zipfile
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse

import pandas as pd
import xarray as xr
from zarr.codecs import BloscCodec

from src.data.common.hpc.client import HPCClient
from src.data.common.index.unified_index import UnifiedDataIndex
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection

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
    PREPARE -- one annual zarr per year, extracting `lccs_class` (uint8)
               from the raw zip-wrapped NetCDF.
    GRID    -- reproject every annual zarr onto the canonical geobox
               (categorical -> always nearest, dst_nodata=0).
    """

    ID = "esacci"
    ALIASES = ("esa_cci", "esacci_lc", "landcover")
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE, PipelineStep.GRID)

    DATA_SOURCE_NAME = "esacci"
    has_entrypoints = True

    DATASET = "satellite-land-cover"
    DEFAULT_VERSIONS = ["v2_0_7cds", "v2_1_1"]
    DEFAULT_VARIABLES = ["lccs_class"]

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="esacci/landcover")  # old ESACCIPreprocessor default
        super().__init__(ctx, cfg)
        self.versions: List[str] = cfg.raw.get("versions") or self.DEFAULT_VERSIONS
        self.cdsapi_rc: Optional[str] = cfg.raw.get("cdsapi_rc")
        self.variables_to_keep: List[str] = cfg.raw.get("variables_to_keep") or self.DEFAULT_VARIABLES
        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix="esacci_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)

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

    def get_file_hash(self, file_url: str) -> str:
        return hashlib.md5(file_url.encode("utf-8")).hexdigest()

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
        if step is PipelineStep.GRID:
            return self._plan_grid(selection)
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

        from src.data.common.fetch.async_downloader import run_async_download_workflow

        index = UnifiedDataIndex(
            bucket_name="",
            data_source=self,
            local_index_dir=self.ctx.local_index_dir,
            key_file=self.ctx.key_file,
            hpc_mode=bool(self.ctx.ssh_target),
        )
        index.build_index_from_source(data_source=self, rebuild=False, only_missing_entrypoints=True)
        index.save()

        hpc_client = HPCClient(target=self.ctx.ssh_target, key_file=self.ctx.key_file)
        download_cfg = dict(self.cfg.raw.get("download", {}))
        return asyncio.run(
            run_async_download_workflow(
                data_source=self, index=index, hpc_client=hpc_client, context=self.ctx, config=download_cfg
            )
        )

    # -- PREPARE ----------------------------------------------------------

    def _resolve_raw_path(self, relative_path: str) -> str:
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(self.output_root(PipelineStep.FETCH), relative_path)

    def _plan_prepare(self, selection: TargetSelection) -> List[StepTarget]:
        index_file = layout.index_path(self.ctx.local_index_dir, self.data_path)
        if not os.path.exists(index_file):
            logger.warning("Parquet index not found: %s", index_file)
            return []

        df = pd.read_parquet(index_file)
        status_col = "status_category" if "status_category" in df.columns else (
            "download_status" if "download_status" in df.columns else None
        )
        if status_col is None:
            return []
        df = df[df[status_col] == "completed"]
        if df.empty or "relative_path" not in df.columns:
            return []

        files_by_year: Dict[int, List[str]] = {}
        for rel_path in df["relative_path"].tolist():
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
            return ds
        except Exception:
            logger.exception("Error loading %s.", file_path)
            return None

    @staticmethod
    def _write_annual_zarr(ds: xr.Dataset, output_path: str) -> bool:
        try:
            ds = ds.chunk({"time": 1, "band": 1, "latitude": 512, "longitude": 512})
            compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle", blocksize=0)
            encoding = {
                var: {"compressors": (compressor,), "chunks": (1, 1, 512, 512), "dtype": str(ds[var].dtype)}
                for var in ds.data_vars
            }
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

    # -- GRID ----------------------------------------------------------

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
        annual_files = [f for f in self._list_annual_zarrs() if selection.matches_year(f["year"])]
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
                    "esacci_lc_timeseries_reprojected.zarr",
                    namespace=self.cfg.namespace,
                    grid_id=self.ctx.grid_id,
                    layout=self.ctx.layout,
                    v2_family="land_cover",
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

                # ESA CCI LC is categorical -- always nearest (SpatialProcessor's
                # own default), 0 as nodata, no scale/offset packaging.
                success = processor.process_spatial_standard(
                    source_files=list(target.inputs),
                    output_path=target.output_path,
                    years_to_process=years,
                    year_pattern_func=year_from_path,
                    preprocess_func=preprocess,
                    get_variables_func=get_vars_and_attrs,
                    dst_nodata=0,
                    packaging_attrs={},
                )
                if success:
                    mark_complete(target.output_path)
                return success

    # _dask_client: inherited from DataSource (src/data/sources/base.py).

registry.register(
    EsacciSource.ID,
    __name__,
    EsacciSource.__name__,
    EsacciSource.STEPS,
    aliases=EsacciSource.ALIASES,
)

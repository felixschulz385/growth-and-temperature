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
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
import aiohttp
import pandas as pd
import requests
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

    def get_file_hash(self, file_url: str) -> str:
        return hashlib.md5(file_url.encode("utf-8")).hexdigest()

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
        await asyncio.sleep(0.3)

        async def _do_download(sess: aiohttp.ClientSession):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with sess.get(source_url) as resp:
                        resp.raise_for_status()
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        async with aiofiles.open(output_path, "wb") as fh:
                            async for chunk in resp.content.iter_chunked(8192):
                                await fh.write(chunk)
                        return
                except (aiohttp.ClientError, asyncio.TimeoutError):
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

        # Quirk: preserve plain-dict insertion order, not sorted-by-year.
        files_by_year: Dict[int, List[str]] = {}
        for rel_path in df["relative_path"].tolist():
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
                    meta={"year": year, "total_candidates": len(year_files)},
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
                    "ntl_harm_timeseries_reprojected.zarr",
                    namespace=self.cfg.namespace,
                    grid_id=self.ctx.grid_id,
                    layout=self.ctx.layout,
                    v2_family="ntl_harm",
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

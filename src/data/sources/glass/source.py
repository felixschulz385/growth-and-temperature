"""GLASS (Global LAnd Surface Satellite) LST: fetch + prepare + grid.

docs/design/09-integrated-pipeline.md §5: registered as TWO separate ids,
`glass_modis` and `glass_avhrr` (not one id with two aliases) -- unlike EOG's
three aliases, which share identical behaviour derived from config, GLASS's
MODIS/AVHRR variants have genuinely different filename formats, output
layouts, and CRS handling, matching how `orchestration/configs/data.yaml`
already treats them as two distinct top-level `sources:` entries. Both ids
point at this one class, which derives which variant it is from its own
registered id (mirroring the old `GlassPreprocessor`'s `type` kwarg, which was
always exactly "glass_modis"/"glass_avhrr" in practice).

Merges `src/data/download/sources/glass/source.py::GlassLSTDataSource` (crawl
+ download, reused via `_CrawlerMixin`) and
`src/data/preprocess/sources/glass.py::GlassPreprocessor` (`stage="annual"`
-> PREPARE, `stage="spatial"` -> GRID) into one class. The heavy raster
compute (`_process_file_group_hpc`, `_calculate_statistics`,
`_process_years_chunked`, `_aggregate_year_files`, `_process_year_tiles`) is
ported mechanically, unchanged in behaviour -- including
`_calculate_statistics`'s naive `resample(time="1YE").mean()` from raw daily
data rather than from its own `monthly_stats`
([`docs/design/07-modis-ingest.md`](../../../../docs/design/07-modis-ingest.md)
§4 already flags this as "do not copy this pattern" for MODIS's own
compositing). Fixing it here is explicitly deferred to a separate,
labelled follow-on change (docs/design/09-integrated-pipeline.md §5/§14),
not silently folded into this mechanical migration.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import dask.array as da
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from odc.geo.geom import clip_lon180
from odc.geo.xr import xr_reproject
from zarr.codecs import BloscCodec

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.glass.crawler import _CrawlerMixin
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection

logger = logging.getLogger(__name__)


class GlassSource(_CrawlerMixin, DataSource):
    """GLASS LST, MODIS (multiple tiles/day) or AVHRR (one global file/day).

    FETCH   -- crawl + download the configured `base_url` directory tree.
    PREPARE -- one annual zarr per (year[, grid_cell]) with annual+monthly
               LST statistics (mean/median/std/max/min/rolling/threshold
               counts/valid-count).
    GRID    -- chunked tile-by-tile reprojection onto the canonical geobox
               (MODIS: native sinusoidal; AVHRR: EPSG:4326), region-written
               into one shared multi-year zarr. Does not use the shared
               `SpatialProcessor` -- GLASS's own bespoke tiled-reprojection
               path predates it and is ported as-is.
    """

    ID = "glass"  # not directly registered -- see the two registry.register() calls below
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE, PipelineStep.GRID)

    BUCKET_NAME = "growthandheat"
    MODIS_PATH_PREFIX = "glass/LST/MODIS/Daily/1KM/"
    AVHRR_PATH_PREFIX = "glass/LST/AVHRR/0.05D/"
    VARIABLE_NAME = "LST"

    DATA_SOURCE_NAME = "glass"
    has_entrypoints = True

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        # Which variant: derived from the registered id ("glass_modis" /
        # "glass_avhrr"), mirroring the old `type` kwarg -- always one of
        # exactly these two strings in every real config, so no third branch
        # is needed (docs/design/09-integrated-pipeline.md §5).
        self.data_source_kind = "AVHRR" if "avhrr" in cfg.source_id.lower() else "MODIS"
        self.path_prefix = self.MODIS_PATH_PREFIX if self.data_source_kind == "MODIS" else self.AVHRR_PATH_PREFIX

        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path=self.path_prefix.rstrip("/"))  # old GlassPreprocessor default
        super().__init__(ctx, cfg)

        self.base_url: Optional[str] = cfg.raw.get("base_url")
        if not self.base_url:
            raise ValueError("'base_url' is required.")
        self.file_extensions: List[str] = cfg.raw.get("file_extensions") or [".hdf"]
        self.version = cfg.raw.get("version", "v1")
        self.grid_cells: Optional[List[str]] = cfg.raw.get("grid_cells")
        self.chunk_size = cfg.raw.get("chunk_size") or {"band": 1, "x": 500, "y": 500}
        self.dashboard_port = cfg.raw.get("dashboard_port", ctx.dashboard_port)

        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix=f"glass_{self.data_source_kind}_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)

    def output_root(self, step: PipelineStep, *, namespace: str | None = None) -> str:
        """Overrides the base default: GLASS's output root is keyed by the
        fixed MODIS/AVHRR `path_prefix` constant, not `cfg.data_path` (which
        exists only for index-file naming, matching old
        `GlassPreprocessor.get_hpc_output_path` using `self.path_prefix`)."""
        return layout.output_root(
            self.ctx.data_root,
            self.path_prefix,
            step,
            namespace=namespace,
            grid_id=self.ctx.grid_id,
            layout=self.ctx.layout,
        )

    # ------------------------------------------------------------------
    # RemoteFileCatalog contract -- list_remote_files/get_all_entrypoints
    # come from _CrawlerMixin.
    # ------------------------------------------------------------------

    def local_path(self, relative_path: str) -> str:
        return os.path.join("data", relative_path)

    def download(self, file_url: str, output_path: str, session: Any = None) -> None:
        import requests

        s = session or requests.Session()
        r = s.get(file_url, stream=True)
        r.raise_for_status()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    async def download_async(self, file_url: str, output_path: str, session: Any = None) -> None:
        import asyncio

        import aiofiles
        import aiohttp

        await asyncio.sleep(0.5)

        async def _download_with_session(sess: aiohttp.ClientSession):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with sess.get(file_url) as response:
                        response.raise_for_status()
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        async with aiofiles.open(output_path, "wb") as f:
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)
                        return
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    if attempt < max_retries - 1:
                        await asyncio.sleep((attempt + 1) * 2)
                    else:
                        raise

        if session is None:
            connector = aiohttp.TCPConnector(limit=5, limit_per_host=2)
            timeout = aiohttp.ClientTimeout(total=300, connect=30)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as sess:
                await _download_with_session(sess)
        else:
            await _download_with_session(session)

    def filename_to_entrypoint(self, relative_path: str) -> Optional[Dict[str, Any]]:
        filename = os.path.basename(relative_path)
        try:
            parts = filename.split(".")
            date_part = next(part for part in parts if part.startswith("A"))
            return {"year": int(date_part[1:5]), "day": int(date_part[5:])}
        except (IndexError, ValueError, StopIteration):
            return None

    def get_file_hash(self, file_url: str) -> str:
        import hashlib

        return hashlib.md5(file_url.encode("utf-8")).hexdigest()

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

        import asyncio

        from src.data.common.fetch.async_downloader import run_async_download_workflow
        from src.data.common.hpc.client import HPCClient
        from src.data.common.index.unified_index import UnifiedDataIndex

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

    # -- PREPARE ("annual") -----------------------------------------------

    def _resolve_source_file_path(self, file_path: str) -> str:
        if os.path.isabs(file_path) or (self.ctx.data_root and file_path.startswith(self.ctx.data_root)):
            return file_path
        # Route through output_root(FETCH) rather than hand-building
        # "<path_prefix>/raw/..." -- that hardcodes the legacy shape and
        # ignores ctx.layout="v2", which relocates FETCH output to
        # "raw/<data_path>/..." (src/data/sources/layout.py).
        return os.path.join(self.output_root(PipelineStep.FETCH), file_path)

    def _parse_modis_filenames(self, filenames: List[str]) -> pd.DataFrame:
        """Expected format: GLASS06A01.V01.A2000055.h00v10.2022021.hdf"""
        result = []
        for filename in filenames:
            try:
                basename = os.path.basename(filename)
                if not basename.endswith(".hdf"):
                    continue
                year_day_match = basename.split(".")[2]
                if not (year_day_match.startswith("A") and len(year_day_match) == 8):
                    continue
                year = int(year_day_match[1:5])
                day = int(year_day_match[5:8])
                grid_match = basename.split(".")[3]
                if not (grid_match.startswith("h") and "v" in grid_match):
                    continue
                h = int(grid_match[1:].split("v")[0])
                v = int(grid_match.split("v")[1])
                result.append({"path": filename, "year": year, "day": day, "h": h, "v": v})
            except (IndexError, ValueError) as exc:
                logger.warning("Could not parse filename %s: %s", filename, exc)
        return pd.DataFrame(result)

    def _parse_avhrr_filenames(self, filenames: List[str]) -> pd.DataFrame:
        """Expected format: GLASS08B31.V40.A1982001.2021259.hdf"""
        result = []
        for filename in filenames:
            try:
                basename = os.path.basename(filename)
                if not basename.endswith(".hdf"):
                    continue
                year_day_match = basename.split(".")[2]
                if not (year_day_match.startswith("A") and len(year_day_match) == 8):
                    continue
                result.append(
                    {"path": filename, "year": int(year_day_match[1:5]), "day": int(year_day_match[5:8])}
                )
            except (IndexError, ValueError) as exc:
                logger.warning("Could not parse filename %s: %s", filename, exc)
        return pd.DataFrame(result)

    def _parse_filenames(self, filenames: List[str]) -> pd.DataFrame:
        if self.data_source_kind == "MODIS":
            return self._parse_modis_filenames(filenames)
        return self._parse_avhrr_filenames(filenames)

    def _plan_prepare(self, selection: TargetSelection) -> List[StepTarget]:
        index_file = layout.index_path(self.ctx.local_index_dir, self.data_path)
        if not index_file or not os.path.exists(index_file):
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

        files_df = self._parse_filenames(df["relative_path"].tolist())
        if files_df.empty:
            return []
        if selection.year_range:
            files_df = files_df[files_df["year"].between(selection.year_range[0], selection.year_range[1])]
        elif selection.years:
            files_df = files_df[files_df["year"].isin(selection.years)]

        targets = []
        if self.data_source_kind == "MODIS":
            if self.grid_cells:
                grid_filter = files_df.apply(lambda row: f"h{row['h']:02d}v{row['v']:02d}" in self.grid_cells, axis=1)
                files_df = files_df[grid_filter]
            for (year, h, v), group in files_df.groupby(["year", "h", "v"]):
                grid_cell = f"h{h:02d}v{v:02d}"
                key = f"{year}/{grid_cell}"
                if not selection.matches_key(key):
                    continue
                targets.append(
                    StepTarget(
                        source_id=self.cfg.source_id,
                        step=PipelineStep.PREPARE,
                        key=key,
                        output_path=os.path.join(self.output_root(PipelineStep.PREPARE), str(year), f"{grid_cell}.zarr"),
                        inputs=tuple(group["path"].tolist()),
                        completion=Completion.MARKER,
                        meta={"year": int(year), "grid_cell": grid_cell, "total_files": len(group)},
                    )
                )
        else:
            for year, group in files_df.groupby("year"):
                key = str(year)
                if not selection.matches_key(key):
                    continue
                targets.append(
                    StepTarget(
                        source_id=self.cfg.source_id,
                        step=PipelineStep.PREPARE,
                        key=key,
                        output_path=os.path.join(self.output_root(PipelineStep.PREPARE), f"{year}.zarr"),
                        inputs=tuple(group["path"].tolist()),
                        completion=Completion.MARKER,
                        meta={"year": int(year), "grid_cell": "global", "total_files": len(group)},
                    )
                )
        return targets

    def _dask_client(self):
        """Overrides `DataSource._dask_client`: GLASS supports a per-source
        config override of the dashboard port (`self.dashboard_port`, set in
        `__init__` from `cfg.raw` falling back to `ctx.dashboard_port`), so
        it can't use the shared base implementation as-is."""
        from src.data.common.dask.client import DaskClientContextManager

        return DaskClientContextManager(
            threads=self.ctx.dask_threads,
            memory_limit=self.ctx.dask_memory_limit,
            dashboard_port=self.dashboard_port,
            temp_dir=os.path.join(self.temp_dir, "dask_workspace"),
        )

    def _process_file_group_hpc(self, files: List[str], year: int, output_path: str, grid_cell: Optional[str] = None) -> bool:
        """Ported verbatim from GlassPreprocessor._process_file_group_hpc."""
        try:
            with self._dask_client() as client:
                if client is None:
                    logger.warning("Failed to initialize Dask client, proceeding without it")
                else:
                    dashboard_link = getattr(client, "dashboard_link", None)
                    if dashboard_link:
                        logger.info("Created Dask client for annual processing: %s", dashboard_link)

                files_with_day = []
                for file_path in files:
                    basename = os.path.basename(file_path)
                    year_day_match = basename.split(".")[2]
                    if year_day_match.startswith("A") and len(year_day_match) == 8:
                        day = int(year_day_match[5:8])
                        files_with_day.append((day, file_path))
                files_with_day.sort(key=lambda x: x[0])
                sorted_files = [f[1] for f in files_with_day]

                array_list = []
                days = []
                for file_path in sorted_files:
                    if not os.path.exists(file_path):
                        logger.warning("File does not exist: %s", file_path)
                        continue
                    basename = os.path.basename(file_path)
                    year_day_match = basename.split(".")[2]
                    day = int(year_day_match[5:8])
                    days.append(day)

                    chunks = {k: self.chunk_size[k] for k in ("band", "y", "x") if k in self.chunk_size}
                    ds = rxr.open_rasterio(file_path, decode_coords="all", chunks=chunks)
                    if self.data_source_kind == "MODIS":
                        lst_data = ds
                    else:
                        if hasattr(ds, "data_vars") and self.VARIABLE_NAME in ds.data_vars:
                            lst_data = ds[self.VARIABLE_NAME]
                        elif hasattr(ds, self.VARIABLE_NAME):
                            lst_data = getattr(ds, self.VARIABLE_NAME)
                        else:
                            logger.error("Could not find %s variable in %s", self.VARIABLE_NAME, file_path)
                            continue
                    array_list.append(lst_data)

                if not array_list:
                    logger.error("No valid files found for %s/%s", year, grid_cell)
                    return False

                combined_data = xr.concat(array_list, dim="day").rename({"day": "time"})
                combined_data = combined_data.assign_coords(
                    {"time": pd.to_datetime([f"{year}{day:03d}" for day in days], format="%Y%j")}
                )
                combined_data = combined_data.to_dataset(name=self.VARIABLE_NAME)

                logger.info("Calculating statistics with Dask")
                annual_stats, monthly_stats = self._calculate_statistics(combined_data)
                return self._create_annual_zarr_hpc(annual_stats, monthly_stats, output_path)
        except Exception:
            logger.exception("Error processing file group for %s/%s", year, grid_cell)
            return False

    def _calculate_statistics(self, data: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
        """Ported verbatim from GlassPreprocessor._calculate_statistics --
        INCLUDING its naive `resample(time="1YE").mean()` from raw daily
        data rather than from monthly_stats (see module docstring: a known
        bug, fixed in a separate follow-on change, not here)."""
        mask = da.logical_and(data[self.VARIABLE_NAME] >= 20000, data[self.VARIABLE_NAME] <= 35000)
        masked = data.where(mask)
        rechunked = masked.chunk(
            {"time": -1, "x": self.chunk_size.get("x", 500), "y": self.chunk_size.get("y", 500)}
        )
        attrs = {"_FillValue": 0, "scale_factor": 0.01, "add_offset": 0.0}

        def format_output(arr):
            return arr.fillna(0).assign_attrs(attrs).astype(np.uint16, casting="unsafe")

        def format_output_count(arr):
            return arr.fillna(0).astype(np.uint16, casting="unsafe")

        annual_stats = xr.Dataset(
            {
                "mean": format_output(rechunked[self.VARIABLE_NAME].resample(time="1YE").mean()),
                "median": format_output(rechunked[self.VARIABLE_NAME].resample(time="1YE").median()),
                "std": format_output(rechunked[self.VARIABLE_NAME].resample(time="1YE").std()),
                "max": format_output(rechunked[self.VARIABLE_NAME].resample(time="1YE").max()),
                "min": format_output(rechunked[self.VARIABLE_NAME].resample(time="1YE").min()),
                "rollmax3": format_output(
                    rechunked[self.VARIABLE_NAME].rolling(time=3, center=True).mean().resample(time="1YE").max()
                ),
                "rollmin3": format_output(
                    rechunked[self.VARIABLE_NAME].rolling(time=3, center=True).mean().resample(time="1YE").min()
                ),
                "gt30C": format_output_count((rechunked[self.VARIABLE_NAME] > 30315).resample(time="1YE").sum()),
                "lt0C": format_output_count((rechunked[self.VARIABLE_NAME] < 27315).resample(time="1YE").sum()),
                "valid_count": format_output_count(mask.resample(time="1YE").sum()),
            }
        )

        monthly_stats = xr.Dataset(
            {
                "mean": format_output(data[self.VARIABLE_NAME].resample(time="1ME").mean()),
                "median": format_output(data[self.VARIABLE_NAME].resample(time="1ME").median()),
                "std": format_output(data[self.VARIABLE_NAME].resample(time="1ME").std()),
                "valid_count": mask.resample(time="1ME").sum().fillna(0).astype(np.uint16, casting="unsafe"),
            }
        )

        chunk_dict = {"time": 1}
        if "x" in self.chunk_size:
            chunk_dict["x"] = self.chunk_size["x"]
        if "y" in self.chunk_size:
            chunk_dict["y"] = self.chunk_size["y"]
        return annual_stats.chunk(chunk_dict), monthly_stats.chunk(chunk_dict)

    @staticmethod
    def _create_annual_zarr_hpc(annual_stats: xr.Dataset, monthly_stats: xr.Dataset, output_path: str) -> bool:
        try:
            chunks = {"x": 1000, "y": 1000, "time": 1}
            annual_stats = annual_stats.chunk(chunks)
            monthly_stats = monthly_stats.chunk(chunks)

            compressor = BloscCodec(cname="zstd", clevel=3, shuffle=2)
            encoding = {var: {"compressor": compressor} for var in annual_stats.data_vars}

            annual_output_path = output_path
            monthly_output_path = output_path.replace(".zarr", "_monthly.zarr")

            annual_stats.to_zarr(annual_output_path, mode="w", encoding=encoding, consolidated=True, compute=False).compute()
            monthly_encoding = {var: {"compressor": compressor} for var in monthly_stats.data_vars}
            monthly_stats.to_zarr(
                monthly_output_path, mode="w", encoding=monthly_encoding, consolidated=True, compute=False
            ).compute()
            return True
        except Exception:
            logger.exception("Error creating zarr file at %s.", output_path)
            return False

    def _execute_prepare(self, target: StepTarget) -> bool:
        from src.data.sources.steps import is_complete, mark_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping %s -- already complete: %s", target.key, target.output_path)
            return True

        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)
        resolved_files = [self._resolve_source_file_path(f) for f in target.inputs]
        ok = self._process_file_group_hpc(
            resolved_files, target.meta["year"], target.output_path, target.meta.get("grid_cell")
        )
        if ok:
            mark_complete(target.output_path)
        return ok

    # -- GRID ("spatial") ---------------------------------------------------

    def _get_all_annual_files(self) -> List[Dict[str, Any]]:
        annual_dir = self.output_root(PipelineStep.PREPARE)
        if not os.path.exists(annual_dir):
            return []
        files = []
        if self.data_source_kind == "MODIS":
            for year_dir in os.listdir(annual_dir):
                year_path = os.path.join(annual_dir, year_dir)
                if not os.path.isdir(year_path):
                    continue
                try:
                    year = int(year_dir)
                except ValueError:
                    continue
                for fname in os.listdir(year_path):
                    if fname.endswith(".zarr") and not fname.endswith("_monthly.zarr"):
                        files.append(
                            {"year": year, "grid_cell": os.path.splitext(fname)[0], "zarr_path": os.path.join(year_path, fname)}
                        )
        else:
            for fname in os.listdir(annual_dir):
                if fname.endswith(".zarr") and not fname.endswith("_monthly.zarr"):
                    try:
                        year = int(os.path.splitext(fname)[0])
                        files.append({"year": year, "grid_cell": "global", "zarr_path": os.path.join(annual_dir, fname)})
                    except ValueError:
                        continue
        return files

    def _plan_grid(self, selection: TargetSelection) -> List[StepTarget]:
        annual_files = [f for f in self._get_all_annual_files() if selection.matches_year(f["year"])]
        if not annual_files:
            return []

        years_requested = (
            list(range(selection.year_range[0], selection.year_range[1] + 1)) if selection.year_range else None
        )
        missing = sorted(set(years_requested or []) - {f["year"] for f in annual_files})

        if self.data_source_kind == "MODIS":
            return [
                StepTarget(
                    source_id=self.cfg.source_id,
                    step=PipelineStep.GRID,
                    key="all_cells",
                    output_path=layout.grid_store_path(
                        self.ctx.data_root,
                        self.path_prefix,
                        "modis_timeseries_reprojected.zarr",
                        grid_id=self.ctx.grid_id,
                        layout=self.ctx.layout,
                        v2_family="glass_modis_lst",
                    ),
                    inputs=tuple(f["zarr_path"] for f in annual_files),
                    completion=Completion.MARKER,
                    meta={
                        "years_available": [f["year"] for f in annual_files],
                        "missing_years": missing,
                        "grid_cells": sorted({f["grid_cell"] for f in annual_files}),
                    },
                )
            ]
        return [
            StepTarget(
                source_id=self.cfg.source_id,
                step=PipelineStep.GRID,
                key="global",
                output_path=layout.grid_store_path(
                    self.ctx.data_root,
                    self.path_prefix,
                    "avhrr_timeseries_reprojected.zarr",
                    grid_id=self.ctx.grid_id,
                    layout=self.ctx.layout,
                    v2_family="glass_avhrr_lst",
                ),
                inputs=tuple(f["zarr_path"] for f in annual_files),
                completion=Completion.MARKER,
                meta={"years_available": [f["year"] for f in annual_files], "missing_years": missing},
            )
        ]

    def _execute_grid(self, target: StepTarget) -> bool:
        """Ported verbatim from GlassPreprocessor._process_spatial_target and
        its chunked-tile helpers -- GLASS's own bespoke tiled reprojection,
        not the shared SpatialProcessor (see module docstring)."""
        from src.data.common.geobox import get_target_geobox
        from src.data.sources.steps import is_complete, mark_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping grid step -- already complete: %s", target.output_path)
            return True

        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)

        try:
            with self._dask_client() as client:
                dashboard_link = getattr(client, "dashboard_link", None)
                if dashboard_link:
                    logger.info("Created Dask client for spatial processing: %s", dashboard_link)

                import dask

                with dask.config.set(
                    {
                        "array.slicing.split_large_chunks": True,
                        "array.chunk-size": "512MB",
                        "optimization.fuse.active": False,
                        "distributed.comm.compression": "lz4",
                    }
                ):
                    try:
                        target_geobox = get_target_geobox(self.ctx)
                    except Exception:
                        logger.exception("Failed to get target geobox")
                        return False

                    if not os.path.exists(target.output_path):
                        if not self._create_empty_target_zarr(target.output_path, target_geobox, target.inputs):
                            return False

                    years_to_process = target.meta["years_available"]
                    ok = self._process_years_chunked(list(target.inputs), target.output_path, target_geobox, years_to_process)
                    if ok:
                        mark_complete(target.output_path)
                    return ok
        except Exception:
            logger.exception("Error in GLASS spatial processing")
            return False

    def _create_empty_target_zarr(self, output_path: str, target_geobox, source_files: Tuple[str, ...]) -> bool:
        try:
            sample_ds = xr.open_zarr(source_files[0], mask_and_scale=False, chunks="auto", consolidated=False)
            variables = list(sample_ds.data_vars.keys())
            sample_attrs = sample_ds.attrs.copy()

            years = sorted(self._years_from_source_files(source_files))
            time_coords = pd.to_datetime([f"{year}-12-31" for year in years])

            ny, nx = target_geobox.shape
            dim_y, dim_x = target_geobox.dimensions
            y_coords = target_geobox.coords[dim_y].values.round(5)
            x_coords = target_geobox.coords[dim_x].values.round(5)

            data_vars = {}
            default_attrs = {"_FillValue": 0}
            packaging_attrs = {"scale_factor": 0.01, "add_offset": 0.0}
            for var in variables:
                var_attrs = sample_ds[var].attrs.copy() | default_attrs
                if "float" in str(sample_ds[var].dtype):
                    var_attrs |= packaging_attrs
                data_vars[var] = xr.DataArray(
                    da.zeros((len(years), 1, ny, nx), dtype=np.uint16, chunks=(1, 1, 512, 512)),
                    dims=["time", "band", dim_y, dim_x],
                    coords={"time": time_coords, "band": [1], dim_y: y_coords, dim_x: x_coords},
                    attrs=var_attrs,
                )
            sample_ds.close()

            empty_ds = xr.Dataset(data_vars, attrs=sample_attrs)
            empty_ds = empty_ds.rio.write_crs(target_geobox.crs)

            compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle", blocksize=0)
            encoding = {var: {"chunks": (1, 1, 512, 512), "compressors": (compressor,), "dtype": "uint16"} for var in variables}

            empty_ds.to_zarr(output_path, mode="w", encoding=encoding, compute=False, zarr_format=3, consolidated=False)
            return True
        except Exception:
            logger.exception("Error creating empty target zarr")
            return False

    @staticmethod
    def _years_from_source_files(source_files: Tuple[str, ...]) -> List[int]:
        years = set()
        for f in source_files:
            m = re.search(r"/(\d{4})/", f) or re.search(r"(\d{4})\.zarr", os.path.basename(f))
            if m:
                years.add(int(m.group(1)))
        return sorted(years)

    def _group_files_by_year(self, source_files: List[str]) -> Dict[int, List[str]]:
        files_by_year: Dict[int, List[str]] = {}
        for file_path in source_files:
            if self.data_source_kind == "MODIS":
                m = re.search(r"/(\d{4})/", file_path)
            else:
                m = re.search(r"(\d{4})\.zarr", os.path.basename(file_path))
            if m:
                files_by_year.setdefault(int(m.group(1)), []).append(file_path)
        return files_by_year

    def _process_years_chunked(self, source_files: List[str], output_path: str, target_geobox, years_to_process: List[int]) -> bool:
        try:
            files_by_year = self._group_files_by_year(source_files)
            from odc.geo import GeoboxTiles

            tile_size = 2048
            tiles = GeoboxTiles(target_geobox, (tile_size, tile_size))

            for year in sorted(files_by_year):
                if year not in years_to_process:
                    continue
                year_files = files_by_year[year]
                logger.info("Processing year %s with %d files", year, len(year_files))

                if len(year_files) > 1:
                    prepare_root = layout.output_root(
                        self.ctx.data_root, self.path_prefix, PipelineStep.PREPARE, layout=self.ctx.layout
                    )
                    annual_temp_path = os.path.join(prepare_root, str(year), "temp_combined.tzarr")
                    if not self._aggregate_year_files(year_files, annual_temp_path, year):
                        logger.error("Failed to aggregate files for year %s", year)
                        return False
                    year_source = annual_temp_path
                else:
                    year_source = year_files[0]

                if not self._process_year_tiles(year_source, output_path, target_geobox, tiles, year, tile_size):
                    logger.error("Failed to process tiles for year %s", year)
                    return False
            return True
        except Exception:
            logger.exception("Error in chunked year processing")
            return False

    def _aggregate_year_files(self, year_files: List[str], temp_output_path: str, year: int) -> bool:
        try:
            if os.path.exists(temp_output_path):
                logger.info("Temporary output path already exists")
                return True

            logger.info("Aggregating %d files for year %s", len(year_files), year)
            datasets = []
            for file_path in year_files:
                ds = xr.open_zarr(file_path, decode_coords="all", chunks="auto")
                ds.coords["x"] = ds.coords["x"].astype("int")
                ds.coords["y"] = ds.coords["y"].astype("int")
                datasets.append(ds)

            combined = xr.combine_by_coords(datasets, combine_attrs="drop_conflicts", join="outer")
            x_coords = combined.coords["x"]
            y_coords = combined.coords["y"]
            nx, ny = len(x_coords), len(y_coords)

            variables = list(ds.data_vars.keys())
            coordinates = list(ds.coords.keys())
            data_vars = {}
            default_attrs = {"_FillValue": 0}
            packaging_attrs = {"scale_factor": 0.01, "add_offset": 0.0}
            for var in variables:
                var_attrs = combined[var].attrs.copy() | default_attrs
                if "float" in str(datasets[0][var].dtype):
                    var_attrs |= packaging_attrs
                data_vars[var] = xr.DataArray(
                    da.zeros((1, 1, ny, nx), dtype=np.uint16, chunks=(1, 1, 300, 300)),
                    dims=["band", "time", "y", "x"],
                    coords={"band": [1], "time": [pd.to_datetime(f"{year}-12-31")], "y": y_coords, "x": x_coords},
                    attrs=var_attrs,
                )
            combined_ds = xr.Dataset(data_vars)

            crs = (
                "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
                if self.data_source_kind == "MODIS"
                else 4326
            )
            combined_ds = combined_ds.rio.write_crs(crs)

            compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle", blocksize=0)
            encoding = {
                var: {"chunks": (1, 1, 300, 300), "compressors": (compressor,), "dtype": "uint16"} for var in variables
            } | {coord: {"compressors": (compressor,)} for coord in coordinates}

            combined_ds.to_zarr(temp_output_path, mode="w", encoding=encoding, zarr_format=3, consolidated=False, compute=False)

            for i, ds in enumerate(datasets):
                try:
                    ds_clean = ds.rio.write_crs(crs)
                    ds_clean = ds_clean.drop_vars(["spatial_ref"]).drop_attrs()
                    ds_clean = ds_clean.chunk({"band": 1, "time": 1, "y": 300, "x": 300})
                    ds_clean = ds_clean.isel(y=slice(None, None, -1))
                    if ds_clean.sizes["x"] == 0 or ds_clean.sizes["y"] == 0:
                        continue
                    ds_clean.to_zarr(temp_output_path, region="auto", align_chunks=True)
                except Exception:
                    logger.warning("Error processing region %d/%d for year %s", i + 1, len(datasets), year, exc_info=True)
                    continue

            combined_ds.close()
            for ds in datasets:
                ds.close()
            return True
        except Exception:
            logger.exception("Error aggregating year files for %s", year)
            return False

    def _process_year_tiles(self, year_source: str, output_path: str, target_geobox, tiles, year: int, tile_size: int) -> bool:
        try:
            year_ds = xr.open_zarr(year_source, consolidated=False, decode_coords="all")

            if self.data_source_kind == "AVHRR":
                year_ds = year_ds.rio.write_crs(4326)
                year_ds = year_ds.sel(y=slice(None, None, -1))
            elif self.data_source_kind == "MODIS":
                year_ds = year_ds.rio.write_crs("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs")

            if year_ds.rio.crs is None:
                try:
                    year_ds = year_ds.rio.write_crs(year_ds.spatial_ref.crs_wkt)
                except Exception:
                    logger.warning("Error setting crs on dataset", exc_info=True)

            for ix in range(tiles.shape[0]):
                for iy in range(tiles.shape[1]):
                    try:
                        logger.info("Reprojecting tile [%d, %d] for year %s to: %s", ix, iy, year, output_path)
                        tile_geobox = tiles[ix, iy]

                        def extract_slice(ds, tg):
                            bbox = clip_lon180(tg.pad(32, 32).extent).to_crs(year_ds.rio.crs).boundingbox
                            return ds.sel(y=slice(bbox.bottom, bbox.top), x=slice(bbox.left, bbox.right))

                        clipped_ds = extract_slice(year_ds, tile_geobox).compute()
                        if clipped_ds.sizes["x"] == 0 or clipped_ds.sizes["y"] == 0:
                            continue

                        int_vars = [key for key, val in clipped_ds.dtypes.items() if np.issubdtype(val, np.integer)]
                        if int_vars:
                            clipped_ds[int_vars] = clipped_ds[int_vars].astype("float32").where(clipped_ds.valid_count > 0, np.nan)

                        reprojected_ds = xr_reproject(clipped_ds, tile_geobox, resampling="mode", dst_nodata=np.nan)
                        reprojected_ds = reprojected_ds.drop_vars(["spatial_ref"]).drop_attrs()
                        tile_dim_y, tile_dim_x = tile_geobox.dimensions
                        reprojected_ds.coords[tile_dim_x] = reprojected_ds.coords[tile_dim_x].round(5)
                        reprojected_ds.coords[tile_dim_y] = reprojected_ds.coords[tile_dim_y].round(5)
                        reprojected_ds = reprojected_ds.chunk({"band": 1, "time": 1, tile_dim_y: 512, tile_dim_x: 512})

                        reprojected_ds.to_zarr(output_path, region="auto", align_chunks=True, zarr_format=3, consolidated=False)
                        reprojected_ds.close()
                    except Exception:
                        logger.warning("Error processing tile [%d, %d] for year %s", ix, iy, year, exc_info=True)
                        continue

            year_ds.close()
            return True
        except Exception:
            logger.exception("Error processing tiles for year %s", year)
            return False


registry.register(
    "glass_modis",
    __name__,
    GlassSource.__name__,
    GlassSource.STEPS,
)
registry.register(
    "glass_avhrr",
    __name__,
    GlassSource.__name__,
    GlassSource.STEPS,
)

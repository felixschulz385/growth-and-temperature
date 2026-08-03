"""PLAD (Political Leaders and Development): fetch + grid, no separate prepare.

docs/design/09-integrated-pipeline.md §5. Merges
`src/data/download/sources/harvard.py::HarvardDataSource` (Harvard Dataverse
API fetch -- registered under aliases `harvard_plad`/`harvard`, never
actually wired to the old preprocessor, which built its own `create_source()`
`None` special-case instead) and
`src/data/preprocess/sources/plad.py::PLADPreprocessor` (`stage="spatial"`
-> GRID). **No PREPARE step**: unlike acag/esacci/glass, PLAD's panel
construction (merging the raw PLAD .dta table with GADM administrative
boundaries into a per-year boolean favoritism panel) and its rasterization
happen together in one stage the old code calls "spatial" -- there never was
a separate "annual"/"vector" pre-step, so none is invented here.

**Quirk preserved, not "fixed"**: `get_hpc_output_path` hardcodes the string
`"plad"` as the output path prefix, never `self.data_path` -- so even a
configured `data_path` override would not change where output lands. Modeled
here via an `output_root()` override, mirroring the identical pattern already
used for GLASS's `path_prefix`.

`REQUIRES` on gadm's **PREPARE** (not GRID) -- confirmed by reading
`_resolve_gadm_files_from_preprocessed`, which reads GADM's simplified vector
output (`gadm_levelADM_{1,2}_simplified.gpkg`, via `layout.output_root()` so
it resolves under either layout), not its rasterized grid.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
from itertools import product
from typing import Any, Dict, List, Optional

from zarr.codecs import BloscCodec

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection

logger = logging.getLogger(__name__)

DEFAULT_DOI = "doi:10.7910/DVN/YUS575"


class PlaDSource(DataSource):
    """Political Leaders and Development: regional-favoritism boolean grid."""

    ID = "plad"
    ALIASES = ("harvard_plad", "harvard")
    STEPS = (PipelineStep.FETCH, PipelineStep.GRID)
    REQUIRES = (("gadm", PipelineStep.PREPARE),)

    DATA_SOURCE_NAME = "harvard"
    has_entrypoints = False
    OUTPUT_PREFIX = "plad"  # hardcoded in the old code, see module docstring

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="plad")
        if cfg.year_range is None:
            cfg = dataclasses.replace(cfg, year_range=(1980, 2022))  # old PLADPreprocessor default
        super().__init__(ctx, cfg)

        self.doi = cfg.raw.get("doi") or cfg.raw.get("base_url") or DEFAULT_DOI
        self.base_url = cfg.raw.get("base_url") or f"https://dataverse.harvard.edu/dataset.xhtml?persistentId={self.doi}"
        self.file_extensions = cfg.raw.get("file_extensions") or [".csv", ".nc", ".tif", ".zip"]

        self.admin_level = cfg.raw.get("admin_level", 1)
        if self.admin_level not in (1, 2):
            raise ValueError("admin_level must be 1 or 2")

        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix="plad_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)

    def output_root(self, step: PipelineStep, *, namespace: str | None = None) -> str:
        if step is PipelineStep.GRID:
            return layout.output_root(
                self.ctx.data_root,
                self.OUTPUT_PREFIX,
                step,
                namespace=namespace,
                grid_id=self.ctx.grid_id,
                layout=self.ctx.layout,
            )
        return super().output_root(step, namespace=namespace)

    # ------------------------------------------------------------------
    # RemoteFileCatalog contract (ports HarvardDataSource verbatim)
    # ------------------------------------------------------------------

    def list_remote_files(self, entrypoint: Optional[dict] = None) -> List[tuple]:
        import requests

        api_url = f"https://dataverse.harvard.edu/api/datasets/:persistentId?persistentId={self.doi}"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            files = response.json()["data"]["latestVersion"]["files"]
            result = []
            for file in files:
                label = file["label"]
                if not self.file_extensions or any(label.endswith(ext) for ext in self.file_extensions):
                    relative_path = file["dataFile"].get("originalFileName", label)
                    file_id = file["dataFile"]["id"]
                    result.append((relative_path, f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"))
            return result
        except Exception:
            logger.exception("Error listing files from Harvard Dataverse")
            return []

    def local_path(self, relative_path: str) -> str:
        return os.path.join("data", self.DATA_SOURCE_NAME, relative_path)

    def get_file_hash(self, file_url: str) -> str:
        import hashlib

        return hashlib.md5(file_url.encode("utf-8")).hexdigest()

    def filename_to_entrypoint(self, relative_path: str) -> Optional[Dict[str, Any]]:
        return None

    def get_all_entrypoints(self) -> List[Dict[str, Any]]:
        return []

    def download(self, file_url: str, output_path: str, session: Any = None) -> None:
        import time

        import requests

        s = session or requests.Session()
        time.sleep(0.5)
        r = s.get(file_url, stream=True)
        r.raise_for_status()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    async def download_async(self, file_url: str, output_path: str, session: Any = None) -> None:
        import asyncio

        await asyncio.sleep(0.5)
        await asyncio.get_event_loop().run_in_executor(None, self.download, file_url, output_path, None)

    # ------------------------------------------------------------------
    # plan()/execute() dispatch
    # ------------------------------------------------------------------

    def _plan(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        if step is PipelineStep.FETCH:
            return [
                StepTarget(
                    source_id=self.ID, step=PipelineStep.FETCH, key="all",
                    output_path=self.output_root(PipelineStep.FETCH), completion=Completion.NEVER,
                )
            ]
        if step is PipelineStep.GRID:
            return [
                StepTarget(
                    source_id=self.ID, step=PipelineStep.GRID, key=f"adm{self.admin_level}",
                    output_path=layout.grid_store_path(
                        self.ctx.data_root,
                        self.OUTPUT_PREFIX,
                        f"plad_adm{self.admin_level}_timeseries_reprojected.zarr",
                        grid_id=self.ctx.grid_id,
                        layout=self.ctx.layout,
                        v2_family=f"admin_panel_adm{self.admin_level}",
                    ),
                    completion=Completion.PATH_EXISTS,
                    meta={"admin_level": self.admin_level, "year_range": self.cfg.year_range},
                )
            ]
        raise AssertionError(f"unreachable: {step}")

    def _execute(self, target: StepTarget) -> bool:
        if target.step is PipelineStep.FETCH:
            return self._execute_fetch(target)
        if target.step is PipelineStep.GRID:
            return self._execute_grid(target)
        raise AssertionError(f"unreachable: {target.step}")

    def _execute_fetch(self, target: StepTarget) -> bool:
        if not self.ctx.ssh_target:
            logger.warning("Fetch requires an HPC/remote target to be configured.")
            return False

        from src.data.common.fetch.driver import run_fetch

        return run_fetch(self, **self.cfg.raw.get("download", {}))

    # -- GRID ("spatial": panel construction + rasterization) ---------------

    def _resolve_gadm_files_from_preprocessed(self) -> Dict[str, str]:
        # Cross-source reference to gadm's own PREPARE output -- resolved
        # through layout.output_root() (not hardcoded to the legacy
        # misc/processed/stage_1/gadm shape) so this keeps finding gadm's
        # simplified vector files under ctx.layout="v2" too, matching
        # CountryClassificationsSource._plan_grid()'s own cross-source gadm
        # reference (src/data/sources/misc/country_classifications.py).
        gadm_base_path = layout.output_root(
            self.ctx.data_root, "misc", PipelineStep.PREPARE, namespace="gadm", layout=self.ctx.layout
        )
        data_files = {}
        adm1_path = os.path.join(gadm_base_path, "gadm_levelADM_1_simplified.gpkg")
        adm2_path = os.path.join(gadm_base_path, "gadm_levelADM_2_simplified.gpkg")
        if os.path.exists(adm1_path):
            data_files["gadm_adm1"] = adm1_path
        if os.path.exists(adm2_path):
            data_files["gadm_adm2"] = adm2_path
        return data_files

    def _resolve_plad_data_file(self) -> Optional[str]:
        from src.data.common.ledger.paths import ledger_path
        from src.data.common.ledger.store import SourceLedger

        local_ledger_path = ledger_path(self.ctx.local_index_dir, self.data_path)
        if not local_ledger_path or not os.path.exists(local_ledger_path):
            return None
        with SourceLedger.open(local_ledger_path, data_path=self.data_path, read_only=True) as ledger:
            relative_paths = ledger.completed_fetch_files()
        for rel_path in relative_paths:
            filename = os.path.basename(rel_path).lower()
            if "plad" in filename and filename.endswith(".dta"):
                raw = rel_path if os.path.isabs(rel_path) else os.path.join(self.output_root(PipelineStep.FETCH), rel_path)
                return raw
        return None

    def _get_or_create_geobox(self):
        from src.data.common.geobox import get_target_geobox

        return get_target_geobox(self.ctx)

    def _create_plad_panel(self):
        import geopandas as gpd
        import pandas as pd

        plad_data_path = self._resolve_plad_data_file()
        if not plad_data_path:
            raise ValueError("PLAD data file not found")
        gadm_files = self._resolve_gadm_files_from_preprocessed()

        plad = pd.read_table(plad_data_path)

        if self.admin_level == 1:
            gadm_path = gadm_files.get("gadm_adm1")
            if not gadm_path:
                raise ValueError("GADM ADM1 file not found - ensure gadm's PREPARE step has been run")
            gid_col, reg_fav_col = "GID_1", "reg_fav_adm_1"
        else:
            gadm_path = gadm_files.get("gadm_adm2")
            if not gadm_path:
                raise ValueError("GADM ADM2 file not found - ensure gadm's PREPARE step has been run")
            gid_col, reg_fav_col = "GID_2", "reg_fav_adm_2"

        adm_gdf = gpd.read_file(gadm_path)
        years_to_process = list(range(self.cfg.year_range[0], self.cfg.year_range[1] + 1))

        adm_panel = pd.DataFrame(list(product(adm_gdf[gid_col].unique(), years_to_process)), columns=[gid_col, "year"])
        adm_panel = pd.merge(adm_panel, adm_gdf[[gid_col, "geometry"]])

        plad_panel = pd.DataFrame(list(product(plad.gid_0.unique(), years_to_process)), columns=["gid_0", "year"])

        def processor(row):
            qresults = plad.loc[
                (plad.startyear <= row["year"]) & (plad.endyear >= row["year"]) & (plad.gid_0 == row["gid_0"]),
                ["gid_1", "gid_2"],
            ]
            return pd.Series() if qresults.empty else qresults.iloc[0]

        plad_panel[["reg_fav_adm_1", "reg_fav_adm_2"]] = plad_panel.apply(processor, axis=1)

        reg_fav_panel = pd.merge(
            adm_panel, plad_panel, left_on=[gid_col, "year"], right_on=[reg_fav_col, "year"], how="left"
        )
        reg_fav_panel["reg_fav"] = (~reg_fav_panel[reg_fav_col].isna()).astype(int)
        reg_fav_panel = reg_fav_panel.drop(
            columns=[c for c in ("gid_0", "reg_fav_adm_1", "reg_fav_adm_2") if c in reg_fav_panel.columns]
        )
        return gpd.GeoDataFrame(reg_fav_panel)

    def _create_empty_plad_zarr(self, output_path: str, geobox, years: List[int]) -> bool:
        import dask.array as da
        import pandas as pd
        import xarray as xr

        try:
            time_coords = pd.to_datetime([f"{year}-12-31" for year in years])
            ny, nx = geobox.shape
            dim_y, dim_x = geobox.dimensions
            y_coords = geobox.coords[dim_y].values.round(5)
            x_coords = geobox.coords[dim_x].values.round(5)

            data_var = xr.DataArray(
                da.zeros((len(years), 1, ny, nx), dtype=bool),
                dims=["time", "band", dim_y, dim_x],
                coords={"time": time_coords, "band": [1], dim_y: y_coords, dim_x: x_coords},
                attrs={
                    "long_name": "Regional Favoritism Indicator",
                    "description": f"Boolean indicator for regional favoritism at ADM{self.admin_level} level",
                    "admin_level": self.admin_level,
                    "dtype": "bool",
                },
            )
            empty_ds = xr.Dataset({"reg_fav": data_var}).rio.write_crs(geobox.crs)
            compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle", blocksize=0)
            empty_ds.to_zarr(
                output_path, mode="w",
                encoding={"reg_fav": {"chunks": (1, 1, 512, 512), "compressors": (compressor,), "dtype": "bool"}},
                compute=False, zarr_format=3, consolidated=False,
            )
            return True
        except Exception:
            logger.exception("Error creating empty PLAD zarr")
            return False

    @staticmethod
    def _rasterize_panel(panel_gdf, output_path: str, geobox, years: List[int]) -> bool:
        import pandas as pd
        import shapely
        import xarray as xr
        from odc.geo.geom import Geometry
        from odc.geo.xr import rasterize, xr_zeros

        try:
            for year in years:
                year_data = panel_gdf[panel_gdf["year"] == year]
                if year_data.empty:
                    data_array = xr_zeros(geobox, dtype=bool)
                else:
                    favoritism_regions = year_data[year_data["reg_fav"] == 1]
                    if favoritism_regions.empty:
                        data_array = xr_zeros(geobox, dtype=bool)
                    else:
                        geom_list = [geom for mgeom in favoritism_regions.geometry for geom in mgeom.geoms]
                        favoritism_polygons = shapely.MultiPolygon(geom_list)
                        geom = Geometry(favoritism_polygons, crs=str(year_data.crs))
                        data_array = rasterize(geom, geobox).astype(bool)

                data_array = data_array.expand_dims("time").assign_coords(time=[pd.Timestamp(f"{year}-12-31")])
                data_array = data_array.expand_dims("band").assign_coords(band=[1])
                dim_y, dim_x = geobox.dimensions
                data_array = data_array.assign_coords(
                    {dim_y: geobox.coords[dim_y].values.round(5), dim_x: geobox.coords[dim_x].values.round(5)}
                )
                data_array = data_array.drop_vars(["spatial_ref"])
                dataset = data_array.to_dataset(name="reg_fav")

                from zarr.codecs import BloscCodec as _BloscCodec

                compressor = _BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle", blocksize=0)
                encoding = {var: {"compressors": (compressor,), "dtype": "bool"} for var in dataset.data_vars}
                dataset.to_zarr(output_path, region="auto", consolidated=False)
            return True
        except Exception:
            logger.exception("Error rasterizing PLAD panel")
            return False

    def _execute_grid(self, target: StepTarget) -> bool:
        from src.data.sources.steps import is_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping spatial processing, output already exists: %s", target.output_path)
            return True

        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)
        try:
            panel_gdf = self._create_plad_panel()
            if panel_gdf is None or panel_gdf.empty:
                logger.error("Failed to create PLAD panel")
                return False

            geobox = self._get_or_create_geobox()
            years = sorted(panel_gdf["year"].unique())

            if not self._create_empty_plad_zarr(target.output_path, geobox, years):
                return False
            return self._rasterize_panel(panel_gdf, target.output_path, geobox, years)
        except Exception:
            logger.exception("Error in PLAD spatial processing")
            return False


registry.register(PlaDSource.ID, __name__, PlaDSource.__name__, PlaDSource.STEPS, aliases=PlaDSource.ALIASES, requires=PlaDSource.REQUIRES)

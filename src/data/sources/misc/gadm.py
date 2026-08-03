"""GADM administrative boundaries: fetch + prepare + grid.

docs/design/09-integrated-pipeline.md §7 (the misc split). Ports the
GADM-specific slice of `src/data/download/sources/misc.py::MiscDataSource`
and `src/data/preprocess/sources/misc.py::MiscPreprocessor`'s
`_process_gadm_target`/`_rasterize_gadm_target`/`_create_empty_gadm_zarr`/
`_process_gadm_tiles` (`stage="vector"` -> PREPARE, `stage="spatial"` ->
GRID). Output paths unchanged: `misc/processed/stage_1/gadm/gadm_level*_simplified.gpkg`,
`misc/processed/stage_2/gadm/countries_grid.zarr` (+ `country_code_mapping.json`)
-- so `src/analysis/subsets/registry.py`, `snl_mining`'s config, and
`src/data/common/neighbourhood/`'s cross-border masking (docs/design/03-neighbourhood-engine.md
§5) need no edits.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from zarr.codecs import BloscCodec

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.misc._fetch import ConfiguredFile, ConfiguredFilesFetchMixin
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-levels.zip"
DEFAULT_NAME = "gadm_410-levels.zip"


class GadmSource(ConfiguredFilesFetchMixin, DataSource):
    """GADM v4.1 administrative boundaries -- simplified vector levels +
    rasterized country/subdivision id grid."""

    ID = "gadm"
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE, PipelineStep.GRID)

    DATA_SOURCE_NAME = "gadm"

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="misc")
        if cfg.namespace is None:
            cfg = dataclasses.replace(cfg, namespace="gadm")
        super().__init__(ctx, cfg)

        url = cfg.raw.get("url", DEFAULT_URL)
        name = cfg.raw.get("name", DEFAULT_NAME)
        self.CONFIGURED_FILES: List[ConfiguredFile] = [ConfiguredFile(key="gadm", url=url, name=name)]

        self.simplify_tolerance = cfg.raw.get("simplify_tolerance", 0.001)
        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix="gadm_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)

    @property
    def data_path(self) -> str:
        return f"{self.cfg.data_path}/{self.cfg.namespace}"

    def _plan(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        if step is PipelineStep.FETCH:
            return [
                StepTarget(
                    source_id=self.ID, step=PipelineStep.FETCH, key="all",
                    output_path=self.output_root(PipelineStep.FETCH), completion=Completion.NEVER,
                )
            ]
        if step is PipelineStep.PREPARE:
            return self._plan_prepare()
        if step is PipelineStep.GRID:
            return self._plan_grid()
        raise AssertionError(f"unreachable: {step}")

    def _execute(self, target: StepTarget) -> bool:
        if target.step is PipelineStep.FETCH:
            return self._execute_fetch(target)
        if target.step is PipelineStep.PREPARE:
            return self._execute_prepare(target)
        if target.step is PipelineStep.GRID:
            return self._execute_grid(target)
        raise AssertionError(f"unreachable: {target.step}")

    def _execute_fetch(self, target: StepTarget) -> bool:
        if not self.ctx.ssh_target:
            logger.warning("Fetch requires an HPC/remote target to be configured.")
            return False

        from src.data.common.fetch.driver import run_fetch

        return run_fetch(self, **self.cfg.raw.get("download", {}))

    # -- PREPARE ("vector") -- produces one .gpkg per ADM level -------------

    def _raw_file_path(self) -> str:
        return os.path.join(self.output_root(PipelineStep.FETCH), self.CONFIGURED_FILES[0].name)

    def _plan_prepare(self) -> List[StepTarget]:
        raw_file = self._raw_file_path()
        if not os.path.exists(raw_file):
            index_file = layout.index_path(self.ctx.local_index_dir, self.data_path)
            if not index_file or not os.path.exists(index_file):
                return []
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.PREPARE, key="gadm",
                output_path=self.output_root(PipelineStep.PREPARE),
                inputs=(raw_file,),
                # Custom skip logic below (matches old "any level file exists"
                # check) rather than a single-file Completion policy.
                completion=Completion.NEVER,
            )
        ]

    def _execute_prepare(self, target: StepTarget) -> bool:
        import geopandas as gpd

        output_base = target.output_path
        if not self.cfg.override and os.path.exists(output_base):
            existing = [f for f in os.listdir(output_base) if f.startswith("gadm_level") and f.endswith("_simplified.gpkg")]
            if existing:
                logger.info("Skipping GADM processing, outputs already exist in: %s", output_base)
                return True

        os.makedirs(output_base, exist_ok=True)
        extract_dir = os.path.join(self.temp_dir, "gadm_extracted")
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(target.inputs[0], "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        geopackages = list(Path(extract_dir).glob("*.gpkg"))
        if not geopackages:
            logger.error("No geopackage found in GADM extract")
            return False

        geopackage_path = str(geopackages[0])
        layers = gpd.list_layers(geopackage_path)
        for level in layers.name.tolist():
            gdf = gpd.read_file(geopackage_path, engine="pyogrio", layer=level)
            gdf_simplified = gdf.copy()
            gdf_simplified["geometry"] = gdf_simplified.geometry.simplify(
                tolerance=self.simplify_tolerance, preserve_topology=True
            )
            out_path = f"{output_base}/gadm_level{level}_simplified.gpkg"
            gdf_simplified.to_file(out_path, driver="GPKG")
            logger.info("GADM level %s processing complete: %s", level, out_path)
        return True

    # -- GRID ("spatial") -- tiled rasterization -----------------------------

    def _plan_grid(self) -> List[StepTarget]:
        vector_dir = self.output_root(PipelineStep.PREPARE)
        adm0_file = os.path.join(vector_dir, "gadm_levelADM_0_simplified.gpkg")
        if not os.path.exists(adm0_file):
            return []
        adm1_file = os.path.join(vector_dir, "gadm_levelADM_1_simplified.gpkg")
        inputs = (adm0_file, adm1_file) if os.path.exists(adm1_file) else (adm0_file,)
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.GRID, key="gadm",
                output_path=layout.grid_store_path(
                    self.ctx.data_root,
                    self.cfg.data_path,
                    "countries_grid.zarr",
                    namespace=self.cfg.namespace,
                    grid_id=self.ctx.grid_id,
                    layout=self.ctx.layout,
                    v2_family="country_id",
                ),
                inputs=inputs, completion=Completion.MARKER,
            )
        ]

    def _execute_grid(self, target: StepTarget) -> bool:
        from odc.geo import GeoboxTiles

        from src.data.common.geobox import get_target_geobox
        from src.data.sources.steps import is_complete, mark_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping GADM rasterization, output already exists: %s", target.output_path)
            return True

        import geopandas as gpd

        output_dir = os.path.dirname(target.output_path)
        os.makedirs(output_dir, exist_ok=True)

        adm0_file = target.inputs[0]
        adm1_file = target.inputs[1] if len(target.inputs) > 1 else None

        gdf_adm0 = gpd.read_file(adm0_file, engine="pyogrio")
        country_codes = sorted(gdf_adm0["GID_0"].unique())
        country_code_to_id = {code: i + 1 for i, code in enumerate(country_codes)}

        gdf_adm1 = None
        subdivision_code_to_id: Dict[str, int] = {}
        if adm1_file:
            gdf_adm1 = gpd.read_file(adm1_file, engine="pyogrio")
            subdivision_codes = sorted(gdf_adm1["GID_1"].unique())
            subdivision_code_to_id = {code: i + 1 for i, code in enumerate(subdivision_codes)}

        with self._dask_client() as client:
            dashboard_link = getattr(client, "dashboard_link", None)
            if dashboard_link:
                logger.info("Created Dask client for GADM rasterization: %s", dashboard_link)

            geobox = get_target_geobox(self.ctx)
            tile_size = 2048
            tiles = GeoboxTiles(geobox, (tile_size, tile_size))

            if not self._create_empty_gadm_zarr(target.output_path, geobox, gdf_adm1 is not None):
                return False
            if not self._process_gadm_tiles(tiles, target.output_path, gdf_adm0, gdf_adm1, country_code_to_id, subdivision_code_to_id):
                return False

        with open(os.path.join(output_dir, "country_code_mapping.json"), "w") as f:
            json.dump(country_code_to_id, f, indent=2)
        if subdivision_code_to_id:
            with open(os.path.join(output_dir, "subdivision_code_mapping.json"), "w") as f:
                json.dump(subdivision_code_to_id, f, indent=2)

        mark_complete(target.output_path)
        return True

    # _dask_client: inherited from DataSource (src/data/sources/base.py).

    @staticmethod
    def _create_empty_gadm_zarr(output_path: str, geobox, include_subdivisions: bool) -> bool:
        import dask.array as da
        import numpy as np
        import xarray as xr

        try:
            ny, nx = geobox.shape
            dim_y, dim_x = geobox.dimensions
            y_coords = geobox.coords[dim_y].values.round(5)
            x_coords = geobox.coords[dim_x].values.round(5)

            data_vars = {
                "country": xr.DataArray(
                    da.zeros((ny, nx), dtype=np.uint16, chunks=(512, 512)),
                    dims=[dim_y, dim_x],
                    coords={dim_y: y_coords, dim_x: x_coords},
                    attrs={"description": "Country ID grid (0=no country)", "_FillValue": 0},
                )
            }
            if include_subdivisions:
                data_vars["subdivision"] = xr.DataArray(
                    da.zeros((ny, nx), dtype=np.uint16, chunks=(512, 512)),
                    dims=[dim_y, dim_x],
                    coords={dim_y: y_coords, dim_x: x_coords},
                    attrs={"description": "Subdivision ID grid (0=no subdivision)", "_FillValue": 0},
                )

            ds = xr.Dataset(
                data_vars,
                attrs={
                    "description": "GADM administrative boundaries grid",
                    "source": "GADM administrative boundaries",
                    "date_created": datetime.now().isoformat(),
                    "crs": str(geobox.crs),
                    "levels_included": "ADM_0 (countries)" + (" and ADM_1 (subdivisions)" if include_subdivisions else ""),
                },
            )
            ds = ds.rio.write_crs(geobox.crs)

            compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle", blocksize=0)
            encoding = {v: {"chunks": (512, 512), "compressors": compressor, "dtype": "uint16"} for v in data_vars}
            ds.to_zarr(output_path, mode="w", encoding=encoding, compute=False, consolidated=False)
            return True
        except Exception:
            logger.exception("Error creating empty GADM zarr")
            return False

    @staticmethod
    def _process_gadm_tiles(tiles, output_path: str, gdf_adm0, gdf_adm1, country_code_to_id: dict, subdivision_code_to_id: dict) -> bool:
        import shapely.geometry
        import xarray as xr
        import numpy as np
        from odc.geo.geom import Geometry
        from odc.geo.xr import rasterize

        try:
            total_tiles = tiles.shape[0] * tiles.shape[1]
            processed_tiles = 0

            for ix in range(tiles.shape[0]):
                for iy in range(tiles.shape[1]):
                    try:
                        tile_geobox = tiles[ix, iy]
                        tile_bounds = tile_geobox.boundingbox
                        tile_polygon = shapely.geometry.box(tile_bounds.left, tile_bounds.bottom, tile_bounds.right, tile_bounds.top)

                        overlapping_countries = gdf_adm0[gdf_adm0.geometry.intersects(tile_polygon)]
                        overlapping_subdivisions = gdf_adm1[gdf_adm1.geometry.intersects(tile_polygon)] if gdf_adm1 is not None else None

                        if len(overlapping_countries) == 0 and (gdf_adm1 is None or len(overlapping_subdivisions) == 0):
                            processed_tiles += 1
                            continue

                        tile_shape = tile_geobox.shape
                        country_tile = np.zeros(tile_shape, dtype=np.uint16)
                        subdivision_tile = np.zeros(tile_shape, dtype=np.uint16) if gdf_adm1 is not None else None

                        for _, row in overlapping_countries.iterrows():
                            value = country_code_to_id[row["GID_0"]]
                            geom = Geometry(row.geometry, crs=str(gdf_adm0.crs))
                            country_mask = rasterize(geom, tile_geobox)
                            country_tile = np.where(country_mask, value, country_tile)

                        if overlapping_subdivisions is not None and len(overlapping_subdivisions) > 0:
                            for _, row in overlapping_subdivisions.iterrows():
                                value = subdivision_code_to_id[row["GID_1"]]
                                geom = Geometry(row.geometry, crs=str(gdf_adm1.crs))
                                subdivision_mask = rasterize(geom, tile_geobox)
                                subdivision_tile = np.where(subdivision_mask, value, subdivision_tile)

                        tile_dim_y, tile_dim_x = tile_geobox.dimensions
                        tile_coords = {
                            tile_dim_y: tile_geobox.coords[tile_dim_y].values.round(5),
                            tile_dim_x: tile_geobox.coords[tile_dim_x].values.round(5),
                        }
                        tile_data_vars = {
                            "country": xr.DataArray(
                                country_tile, dims=[tile_dim_y, tile_dim_x], coords=tile_coords,
                            )
                        }
                        if subdivision_tile is not None:
                            tile_data_vars["subdivision"] = xr.DataArray(
                                subdivision_tile, dims=[tile_dim_y, tile_dim_x], coords=tile_coords,
                            )

                        xr.Dataset(tile_data_vars).to_zarr(output_path, region="auto", mode="r+", consolidated=False)
                        processed_tiles += 1
                        if processed_tiles % 100 == 0:
                            logger.info("Processed %d/%d tiles", processed_tiles, total_tiles)
                    except Exception:
                        logger.warning("Error processing tile [%d, %d]", ix, iy, exc_info=True)
                        processed_tiles += 1
                        continue

            logger.info("Completed processing all %d tiles", total_tiles)
            return True
        except Exception:
            logger.exception("Error processing GADM tiles")
            return False


registry.register(GadmSource.ID, __name__, GadmSource.__name__, GadmSource.STEPS)

"""OpenStreetMap land polygons: fetch + prepare.

docs/design/09-integrated-pipeline.md §7 (the misc split): one of the three
sources `misc.py` (both download and preprocess sides) used to bundle behind
config-key string matching. Ports the OSM-specific slice of
`src/data/download/sources/misc.py::MiscDataSource` (one configured file) and
`src/data/preprocess/sources/misc.py::MiscPreprocessor`'s
`_process_osm_target`/`_rasterize_osm_target`. Output paths:
`prepared/misc/osm/land_polygons_simplified.gpkg`, `grid/<grid_id>/land_mask.zarr`.

OSM's final output has no time dimension -- `run_tiled_prepare(years=None,
reproject=False, ...)`, one static rasterized `cell_id`-keyed parquet part
per tile (the vector rasterization happens inside `raw_getter`, directly on
each `tile.geobox`, so no raster resampling step is needed). There is no
separate GRID step.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import List

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.misc._fetch import ConfiguredFile, ConfiguredFilesFetchMixin
from src.data.sources.steps import Completion, PipelineStep, StepTarget
from src.data.sources import verify

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://osmdata.openstreetmap.de/download/land-polygons-complete-4326.zip"
DEFAULT_NAME = "land-polygons-complete-4326.zip"


class OsmSource(ConfiguredFilesFetchMixin, DataSource):
    """OpenStreetMap land polygons -- simplified vector + rasterized land mask."""

    ID = "osm"
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE)

    DATA_SOURCE_NAME = "osm"

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="misc")
        if cfg.namespace is None:
            cfg = dataclasses.replace(cfg, namespace="osm")
        super().__init__(ctx, cfg)

        url = cfg.raw.get("url", DEFAULT_URL)
        name = cfg.raw.get("name", DEFAULT_NAME)
        self.CONFIGURED_FILES: List[ConfiguredFile] = [ConfiguredFile(key="osm", url=url, name=name)]

        self.simplify_tolerance = cfg.raw.get("simplify_tolerance", 0.001)
        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix="osm_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)

    @property
    def data_path(self) -> str:
        """Distinct from `gadm`/`country_classifications`'s index files even
        though all three share `cfg.data_path="misc"` for output purposes --
        see `DataSource.data_path`'s docstring (src/data/sources/base.py)."""
        return f"{self.cfg.data_path}/{self.cfg.namespace}"

    # _plan_fetch/_execute_fetch/_plan/_execute: inherited from
    # ConfiguredFilesFetchMixin.

    # -- PREPARE (raw zip -> simplified vector -> rasterized land mask) ----

    def _raw_file_path(self) -> str:
        return os.path.join(self.output_root(PipelineStep.FETCH), self.CONFIGURED_FILES[0].name)

    def _vector_path(self) -> str:
        # MISC_AGG: a PREPARE-stage vector intermediate that feeds the
        # CRS_AGG land_mask.zarr rasterization below, but isn't itself
        # GID-keyed or an admin boundary file -- "everything else"
        # (src/data/sources/layout.py's crs/adm/misc split), same judgment
        # call as ecoregions_simplified.gpkg.
        return os.path.join(
            self.output_root(PipelineStep.PREPARE, agg=layout.MISC_AGG), "land_polygons_simplified.gpkg"
        )

    def _output_path(self) -> str:
        return layout.grid_store_path(
            self.ctx.data_root,
            self.cfg.data_path,
            grid_id=self.ctx.grid_id,
            family="land_mask",
            suffix="",  # cell_id-keyed parquet parts, not a Zarr store -- see grid_store_path docstring
        )

    def _plan_prepare(self) -> List[StepTarget]:
        raw_file = self._raw_file_path()
        if not os.path.exists(raw_file):
            return []
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.PREPARE, key="osm",
                output_path=self._output_path(),
                inputs=(raw_file,), completion=Completion.MARKER,
                meta={
                    "raw_file": raw_file,
                    **verify.verification_meta(self.cfg.raw, expected_vars=("land_mask",), value_range=(0, 1)),
                },
            )
        ]

    def _simplify_vector(self, raw_file: str, vector_path: str) -> bool:
        import geopandas as gpd

        os.makedirs(os.path.dirname(vector_path), exist_ok=True)
        extract_dir = os.path.join(self.temp_dir, "osm_extracted")
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(raw_file, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        shapefiles = list(Path(extract_dir).glob("**/*.shp"))
        if not shapefiles:
            raise RuntimeError("No shapefiles found in OSM extract")

        gdf = gpd.read_file(str(shapefiles[0]), engine="pyogrio")
        gdf_simplified = gdf.copy()
        gdf_simplified["geometry"] = gdf_simplified.geometry.simplify(
            tolerance=self.simplify_tolerance, preserve_topology=True
        )
        gdf_simplified.to_file(vector_path, driver="GPKG")
        return True

    def _rasterize_tile(self, land_polygons, source_crs: str, tile) -> "xr.Dataset":
        """One tile's rasterized land mask, on `tile.geobox` directly -- the
        `raw_getter` for `run_tiled_prepare(years=None, reproject=False,
        ...)`. `land_polygons`/`source_crs` are built once by the caller and
        closed over, not reloaded per tile."""
        import xarray as xr
        from odc.geo.geom import Geometry
        from odc.geo.xr import rasterize

        geom = Geometry(land_polygons, crs=source_crs)
        land_mask = rasterize(geom, tile.geobox)
        return xr.Dataset(data_vars={"land_mask": land_mask})

    def _rasterize(self, vector_path: str, output_path: str) -> bool:
        import geopandas as gpd
        import shapely

        from src.data.common.geobox import get_target_geobox
        from src.data.common.prepare.driver import run_tiled_prepare
        from src.data.common.raster.spatial import SpatialProcessor

        gdf = gpd.read_file(vector_path, engine="pyogrio")
        geobox = get_target_geobox(self.ctx)
        land_polygons = shapely.MultiPolygon(gdf.geometry.tolist())
        source_crs = str(gdf.crs)

        processor = SpatialProcessor(hpc_root=self.ctx.data_root, target_geobox=geobox)
        return run_tiled_prepare(
            output_path=output_path,
            years=None,
            target_geobox=geobox,
            processor=processor,
            raw_getter=lambda tile, year: self._rasterize_tile(land_polygons, source_crs, tile),
            reproject=False,
            processing_version="1-tiled",
            override=self.cfg.override,
        )

    def _execute_prepare(self, target: StepTarget) -> bool:
        from src.data.sources.steps import is_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping OSM processing -- already complete: %s", target.output_path)
            return True

        vector_path = self._vector_path()
        if self.cfg.override or not os.path.exists(vector_path):
            if not self._simplify_vector(target.inputs[0], vector_path):
                return False

        return self._rasterize(vector_path, target.output_path)


registry.register(OsmSource.ID, __name__, OsmSource.__name__, OsmSource.STEPS)

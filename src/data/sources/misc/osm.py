"""OpenStreetMap land polygons: fetch + prepare.

docs/design/09-integrated-pipeline.md §7 (the misc split): one of the three
sources `misc.py` (both download and preprocess sides) used to bundle behind
config-key string matching. Ports the OSM-specific slice of
`src/data/download/sources/misc.py::MiscDataSource` (one configured file) and
`src/data/preprocess/sources/misc.py::MiscPreprocessor`'s
`_process_osm_target`/`_rasterize_osm_target`. Output paths unchanged:
`misc/processed/stage_1/osm/land_polygons_simplified.gpkg`, `misc/processed/
stage_2/osm/land_mask.zarr` -- so `src/data/assemble/constants.py` needs no
edit.

OSM's final output is one whole-extent `rasterize()` call (no time dimension,
no per-year resumability need), so unlike the tiled raster sources
(acag/esacci/eog/ntl_harm/glass) this doesn't route through
`src.data.common.prepare.driver.run_tiled_prepare` -- it's simply one
PREPARE target that extracts+simplifies the vector, then rasterizes it, in
one call. There is no separate GRID step.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List

from zarr.codecs import BloscCodec

from src.data.common.raster.spatial import write_crs_and_grid_mapping_encoding
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.misc._fetch import ConfiguredFile, ConfiguredFilesFetchMixin
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection
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

    def _plan_fetch(self) -> List[StepTarget]:
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.FETCH, key="all",
                output_path=self.output_root(PipelineStep.FETCH), completion=Completion.NEVER,
            )
        ]

    def _plan(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        if step is PipelineStep.FETCH:
            return self._plan_fetch()
        if step is PipelineStep.PREPARE:
            return self._plan_prepare()
        raise AssertionError(f"unreachable: {step}")

    def _execute(self, target: StepTarget) -> bool:
        if target.step is PipelineStep.FETCH:
            return self._execute_fetch(target)
        if target.step is PipelineStep.PREPARE:
            return self._execute_prepare(target)
        raise AssertionError(f"unreachable: {target.step}")

    def _execute_fetch(self, target: StepTarget) -> bool:
        # FETCH is local-disk only now -- no HPC target required. `data
        # transfer` (separate, manual or auto per source config) is the only
        # thing that pushes to HPC.
        from src.data.common.fetch.driver import run_fetch

        return run_fetch(self, **self.cfg.raw.get("download", {}))

    # -- PREPARE (raw zip -> simplified vector -> rasterized land mask) ----

    def _raw_file_path(self) -> str:
        return os.path.join(self.output_root(PipelineStep.FETCH), self.CONFIGURED_FILES[0].name)

    def _vector_path(self) -> str:
        return os.path.join(self.output_root(PipelineStep.PREPARE), "land_polygons_simplified.gpkg")

    def _output_path(self) -> str:
        return layout.grid_store_path(
            self.ctx.data_root,
            self.cfg.data_path,
            "land_mask.zarr",
            namespace=self.cfg.namespace,
            grid_id=self.ctx.grid_id,
            layout=self.ctx.layout,
            v2_family="land_mask",
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

    def _rasterize(self, vector_path: str, output_path: str) -> bool:
        import geopandas as gpd
        import shapely
        import xarray as xr
        from odc.geo.geom import Geometry
        from odc.geo.xr import rasterize

        from src.data.common.geobox import get_target_geobox

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        gdf = gpd.read_file(vector_path, engine="pyogrio")
        geobox = get_target_geobox(self.ctx)

        land_polygons = shapely.MultiPolygon(gdf.geometry.tolist())
        geom = Geometry(land_polygons, crs=str(gdf.crs))
        land_mask = rasterize(geom, geobox)
        dim_y, dim_x = geobox.dimensions
        land_mask.coords[dim_y] = land_mask.coords[dim_y].values.round(5)
        land_mask.coords[dim_x] = land_mask.coords[dim_x].values.round(5)

        ds = xr.Dataset(
            data_vars={"land_mask": land_mask},
            attrs={
                "description": "Land/water mask (1=land, 0=water)",
                "source": "OpenStreetMap land polygons",
                "date_created": datetime.now().isoformat(),
            },
        )
        # Unlike every other GRID step's zarr writer, this one relied solely
        # on `rasterize()`'s own georeferencing rather than an explicit
        # `.rio.write_crs()` + "grid_mapping" encoding entry -- the same fix
        # gadm/ecoregions/snl_mining/glass/berman_mining each needed (see
        # write_crs_and_grid_mapping_encoding()'s docstring): without it, a
        # data variable's zarr encoding has no link to the CRS coordinate,
        # so `.rio.crs` (and any grid_mapping-based reader) returns None on
        # a later read even though the CRS metadata is otherwise present.
        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle", blocksize=0)
        base_encoding = {"land_mask": {"compressors": (compressor,)}}
        ds, encoding = write_crs_and_grid_mapping_encoding(ds, geobox, base_encoding)
        ds.to_zarr(output_path, encoding=encoding, mode="w")
        return True

    def _execute_prepare(self, target: StepTarget) -> bool:
        from src.data.sources.steps import is_complete, mark_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping OSM processing -- already complete: %s", target.output_path)
            return True

        vector_path = self._vector_path()
        if self.cfg.override or not os.path.exists(vector_path):
            if not self._simplify_vector(target.inputs[0], vector_path):
                return False

        if not self._rasterize(vector_path, target.output_path):
            return False
        mark_complete(target.output_path)
        return True


registry.register(OsmSource.ID, __name__, OsmSource.__name__, OsmSource.STEPS)

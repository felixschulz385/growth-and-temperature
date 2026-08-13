"""OpenStreetMap land polygons: fetch + prepare + grid.

docs/design/09-integrated-pipeline.md §7 (the misc split): one of the three
sources `misc.py` (both download and preprocess sides) used to bundle behind
config-key string matching. Ports the OSM-specific slice of
`src/data/download/sources/misc.py::MiscDataSource` (one configured file) and
`src/data/preprocess/sources/misc.py::MiscPreprocessor`'s
`_process_osm_target`/`_rasterize_osm_target` (`stage="vector"` -> PREPARE,
`stage="spatial"` -> GRID). Output paths unchanged: `misc/processed/stage_1/
osm/land_polygons_simplified.gpkg`, `misc/processed/stage_2/osm/land_mask.zarr`
-- so `src/data/assemble/constants.py` needs no edit.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from zarr.codecs import BloscCodec

from src.data.common.raster.spatial import write_crs_and_grid_mapping_encoding
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.misc._fetch import ConfiguredFile, ConfiguredFilesFetchMixin
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection
from src.data.sources import verify

if TYPE_CHECKING:
    from src.data.common.ledger.store import ArtifactRow

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://osmdata.openstreetmap.de/download/land-polygons-complete-4326.zip"
DEFAULT_NAME = "land-polygons-complete-4326.zip"


class OsmSource(ConfiguredFilesFetchMixin, DataSource):
    """OpenStreetMap land polygons -- simplified vector + rasterized land mask."""

    ID = "osm"
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE, PipelineStep.GRID)

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
        if step is PipelineStep.GRID:
            return self._plan_grid()
        raise AssertionError(f"unreachable: {step}")

    def _discover(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        """Ground truth for `data reconcile` -- see gadm.py's identical
        `_discover()` for the full rationale. OSM's targets are singletons
        with no year/key selection to apply; `selection` is accepted for
        interface symmetry only."""
        if step is PipelineStep.FETCH:
            return self._plan_fetch()
        if step is PipelineStep.PREPARE:
            return self._discover_prepare()
        if step is PipelineStep.GRID:
            return self._discover_grid()
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
        # FETCH is local-disk only now -- no HPC target required. `data
        # transfer` (separate, manual or auto per source config) is the only
        # thing that pushes to HPC.
        from src.data.common.fetch.driver import run_fetch

        return run_fetch(self, **self.cfg.raw.get("download", {}))

    # -- PREPARE ("vector") -------------------------------------------------

    def _raw_file_path(self) -> str:
        return os.path.join(self.output_root(PipelineStep.FETCH), self.CONFIGURED_FILES[0].name)

    def _plan_prepare(self) -> List[StepTarget]:
        """Ledger-backed fast path. Falls back to `_discover_prepare()` --
        today's exact live logic -- if no ledger is configured yet, or
        `data reconcile --step prepare` hasn't populated one yet."""

        def build_target(row: "ArtifactRow", _ledger: Any) -> Optional[StepTarget]:
            raw_file = row.meta.get("raw_file")
            if raw_file is None or row.local_path is None:
                return None
            return StepTarget(
                source_id=self.ID, step=PipelineStep.PREPARE, key=row.unit_id,
                output_path=row.local_path, inputs=(raw_file,),
                completion=Completion.PATH_EXISTS, meta=row.meta,
            )

        targets = self._plan_from_ledger(PipelineStep.PREPARE, TargetSelection(), build_target)
        if targets is not None:
            return targets
        logger.warning(
            "No ledger for source='%s' step='prepare' -- falling back to live discovery; "
            "run `data reconcile --source %s --step prepare` for faster planning.",
            self.ID, self.ID,
        )
        return self._discover_prepare()

    def _discover_prepare(self) -> List[StepTarget]:
        raw_file = self._raw_file_path()
        if not os.path.exists(raw_file):
            index_file = layout.index_path(self.ctx.local_index_dir, self.data_path)
            if not index_file or not os.path.exists(index_file):
                return []
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.PREPARE, key="osm",
                output_path=os.path.join(self.output_root(PipelineStep.PREPARE), "land_polygons_simplified.gpkg"),
                inputs=(raw_file,), completion=Completion.PATH_EXISTS,
                meta={"raw_file": raw_file},
            )
        ]

    def _execute_prepare(self, target: StepTarget) -> bool:
        import geopandas as gpd

        if not self.cfg.override and os.path.exists(target.output_path):
            logger.info("Skipping OSM processing, output already exists: %s", target.output_path)
            return True

        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)
        extract_dir = os.path.join(self.temp_dir, "osm_extracted")
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(target.inputs[0], "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        shapefiles = list(Path(extract_dir).glob("**/*.shp"))
        if not shapefiles:
            raise RuntimeError("No shapefiles found in OSM extract")

        gdf = gpd.read_file(str(shapefiles[0]), engine="pyogrio")
        gdf_simplified = gdf.copy()
        gdf_simplified["geometry"] = gdf_simplified.geometry.simplify(
            tolerance=self.simplify_tolerance, preserve_topology=True
        )
        gdf_simplified.to_file(target.output_path, driver="GPKG")
        return True

    # -- GRID ("spatial") ----------------------------------------------------

    def _plan_grid(self) -> List[StepTarget]:
        """Ledger-backed fast path. `inputs` (a single deterministic vector
        path) is persisted directly in `meta` at discovery time, same pattern
        as gadm's PREPARE `raw_file` -- see gadm.py's `_plan_prepare()`."""

        def build_target(row: "ArtifactRow", _ledger: Any) -> Optional[StepTarget]:
            vector_path = row.meta.get("vector_path")
            if vector_path is None or row.local_path is None:
                return None
            return StepTarget(
                source_id=self.ID, step=PipelineStep.GRID, key=row.unit_id,
                output_path=row.local_path, inputs=(vector_path,),
                completion=Completion.MARKER, meta=row.meta,
            )

        targets = self._plan_from_ledger(PipelineStep.GRID, TargetSelection(), build_target)
        if targets is not None:
            return targets
        logger.warning(
            "No ledger for source='%s' step='grid' -- falling back to live discovery; "
            "run `data reconcile --source %s --step grid` for faster planning.",
            self.ID, self.ID,
        )
        return self._discover_grid()

    def _discover_grid(self) -> List[StepTarget]:
        vector_path = os.path.join(self.output_root(PipelineStep.PREPARE), "land_polygons_simplified.gpkg")
        if not os.path.exists(vector_path):
            return []
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.GRID, key="osm",
                output_path=layout.grid_store_path(
                    self.ctx.data_root,
                    self.cfg.data_path,
                    "land_mask.zarr",
                    namespace=self.cfg.namespace,
                    grid_id=self.ctx.grid_id,
                    layout=self.ctx.layout,
                    v2_family="land_mask",
                ),
                inputs=(vector_path,), completion=Completion.MARKER,
                meta={
                    "vector_path": vector_path,
                    **verify.verification_meta(
                        self.cfg.raw, expected_vars=("land_mask",), value_range=(0, 1)
                    ),
                },
            )
        ]

    def _execute_grid(self, target: StepTarget) -> bool:
        import shapely
        import xarray as xr
        from odc.geo.geom import Geometry
        from odc.geo.xr import rasterize

        from src.data.common.geobox import get_target_geobox
        from src.data.sources.steps import is_complete, mark_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping OSM rasterization, output already exists: %s", target.output_path)
            return True

        import geopandas as gpd

        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)

        gdf = gpd.read_file(target.inputs[0], engine="pyogrio")
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
        ds.to_zarr(target.output_path, encoding=encoding, mode="w")
        mark_complete(target.output_path)
        return True


registry.register(OsmSource.ID, __name__, OsmSource.__name__, OsmSource.STEPS)

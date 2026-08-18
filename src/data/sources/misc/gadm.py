"""GADM administrative boundaries: fetch + prepare.

docs/design/09-integrated-pipeline.md §7 (the misc split). Ports the
GADM-specific slice of `src/data/download/sources/misc.py::MiscDataSource`
and `src/data/preprocess/sources/misc.py::MiscPreprocessor`'s
`_process_gadm_target`/`_rasterize_gadm_target`/`_create_empty_gadm_zarr`/
`_process_gadm_tiles` (`stage="vector"`, `stage="spatial"`, both under
PREPARE). Output paths: `prepared/misc/gadm/gadm_level*_simplified.gpkg`,
`grid/<grid_id>/country_id.zarr`.

PREPARE rasterizes every ADM level present in its own vector output (not just
ADM_0/ADM_1), one zarr variable per level named after its GADM id column
(`GID_0`, `GID_1`, ..., matching the level's own `ADM_N` suffix) rather than
`country`/`subdivision` names. Each variable holds uint32 integer ids (0 = no
unit at that level), with a `{GID_N}_code_mapping.json` sidecar per level
mapping GADM's string GID code to the integer id.
`src/analysis/subsets/registry.py` and `country_classifications.py` read
`GID_0_code_mapping.json` and the `GID_0` variable.

GADM's rasterization is tiled, has no time dimension, and every polygon is
already rasterized directly onto each tile's own geobox (no raster
resampling needed) -- `run_tiled_prepare(years=None, reproject=False, ...)`,
one static `cell_id`-keyed parquet part per tile, one column per GID level.
`_execute_prepare` stays its own bespoke two-phase driver (vector
extraction, then rasterization) since phase 1 (extracting/simplifying every
ADM level from the raw zip) isn't tile-shaped at all. There is no separate
GRID step.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from src.data.common.raster.spatial import reproject_for_tile_overlap
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.misc._fetch import ConfiguredFile, ConfiguredFilesFetchMixin
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-levels.zip"
DEFAULT_NAME = "gadm_410-levels.zip"

_LEVEL_FILENAME_RE = re.compile(r"^gadm_level(?P<level>ADM_\d+)_simplified\.gpkg$")


def _level_from_path(path: str) -> str | None:
    """Extract the GADM level name (e.g. "ADM_1") from a PREPARE-stage filename."""
    match = _LEVEL_FILENAME_RE.match(os.path.basename(path))
    return match.group("level") if match else None


def _gid_column_for_level(level: str) -> str:
    """GADM's per-level id column follows its layer name exactly: "ADM_1" -> "GID_1"."""
    return level.replace("ADM_", "GID_")


def gid_mapping_path(data_root: str, grid_id: str, gid_col: str) -> str:
    """Path to gadm PREPARE's `{gid_col}_code_mapping.json` sidecar for
    another source to consult (docs/design/09-integrated-pipeline.md §2:
    cross-source coupling is on artefact paths, never a class import).
    Shared by every source that needs to translate its own native GADM
    string codes (e.g. `GID_1` values like "USA.1_1") into the same integer
    ids gadm's own per-pixel `GID_N` grid uses, so a small per-GID table can
    be merged directly onto assembled rows instead of being rasterized
    (`src.data.assemble.processors.TileProcessor`'s `join_on` mechanism).

    `grid_id` is unused by the path itself (the mapping is grid-independent
    -- it's a string-code -> integer-id table, not pixel data) but kept in
    the signature for call-site symmetry with the `country_id.zarr` grid
    store it accompanies. ADM_AGG (src/data/sources/layout.py): a GID_N-keyed
    sidecar, not a pixel-grid store, filed alongside gadm's simplified
    `.gpkg` boundary files under the same "admin data" bucket."""
    adm_dir = layout.output_root(data_root, "misc", PipelineStep.PREPARE, namespace="gadm", agg=layout.ADM_AGG)
    return os.path.join(adm_dir, f"{gid_col}_code_mapping.json")


class GadmSource(ConfiguredFilesFetchMixin, DataSource):
    """GADM v4.1 administrative boundaries -- simplified vector levels +
    rasterized per-level GID id grid (GID_0, GID_1, ...)."""

    ID = "gadm"
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE)

    DATA_SOURCE_NAME = "gadm"
    #: bump to force a full reprocess (`run_tiled_prepare`'s `processing_version`)
    PROCESSING_VERSION = "1-tiled"

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

    # -- PREPARE (raw zip -> per-ADM-level vectors -> tiled rasterization) --

    def _raw_file_path(self) -> str:
        return os.path.join(self.output_root(PipelineStep.FETCH), self.CONFIGURED_FILES[0].name)

    def _vector_dir(self) -> str:
        # ADM_AGG: gadm's simplified `.gpkg` boundary files are admin-shaped
        # vector intermediates that feed the GID_N_code_mapping.json sidecars
        # (src/data/sources/layout.py's crs/adm/misc split), not pixel-grid
        # data -- kept with the code-mapping sidecars in the same bucket
        # (gid_mapping_path() above).
        return self.output_root(PipelineStep.PREPARE, agg=layout.ADM_AGG)

    def _grid_output_path(self) -> str:
        return layout.grid_store_path(
            self.ctx.data_root,
            self.cfg.data_path,
            grid_id=self.ctx.grid_id,
            family="country_id",
            suffix="",  # cell_id-keyed parquet parts, not a Zarr store -- see grid_store_path docstring
        )

    def _plan_prepare(self) -> List[StepTarget]:
        raw_file = self._raw_file_path()
        if not os.path.exists(raw_file):
            return []
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.PREPARE, key="gadm",
                output_path=self._grid_output_path(),
                inputs=(raw_file,),
                completion=Completion.MARKER,
                meta={"raw_file": raw_file},
            )
        ]

    def _existing_level_files(self, vector_dir: str) -> List[str]:
        return sorted(
            (str(p) for p in Path(vector_dir).glob("gadm_level*_simplified.gpkg")),
            key=lambda p: int(_level_from_path(p).split("_")[1]),
        )

    def _simplify_vector_levels(self, raw_file: str, vector_dir: str) -> Optional[List[str]]:
        """Phase 1: extract every ADM level from the raw zip and write a
        simplified .gpkg per level. Returns the written level file paths, or
        `None` on failure."""
        import geopandas as gpd

        os.makedirs(vector_dir, exist_ok=True)
        extract_dir = os.path.join(self.temp_dir, "gadm_extracted")
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(raw_file, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        geopackages = list(Path(extract_dir).glob("*.gpkg"))
        if not geopackages:
            logger.error("No geopackage found in GADM extract")
            return None

        geopackage_path = str(geopackages[0])
        layers = gpd.list_layers(geopackage_path)
        level_files = []
        for level in layers.name.tolist():
            gdf = gpd.read_file(geopackage_path, engine="pyogrio", layer=level)
            gdf_simplified = gdf.copy()
            gdf_simplified["geometry"] = gdf_simplified.geometry.simplify(
                tolerance=self.simplify_tolerance, preserve_topology=True
            )
            out_path = f"{vector_dir}/gadm_level{level}_simplified.gpkg"
            gdf_simplified.to_file(out_path, driver="GPKG")
            level_files.append(out_path)
            logger.info("GADM level %s processing complete: %s", level, out_path)
        return level_files

    @staticmethod
    def _rasterize_tile(
        level_gdfs: Dict[str, "gpd.GeoDataFrame"],
        level_code_to_id: Dict[str, Dict[str, int]],
        tile,
    ) -> "xr.Dataset":
        """One tile's rasterized GID-id grid, on `tile.geobox` directly --
        the `raw_getter` for `run_tiled_prepare(years=None, reproject=False,
        ...)`. `level_gdfs`/`level_code_to_id` are built once by the caller
        (`_rasterize_levels`) and closed over, not reloaded per tile.

        Always returns a dataset, one column per GID level, even when no
        polygon overlaps this tile: the untouched-pixel default (0 = "no
        unit at this level") is the correct output, and `run_tiled_prepare`
        would otherwise record a legitimate "nothing here" tile as a
        retryable failure. The per-level `.intersects()` pre-filter below is
        a performance optimization (skip rasterize() calls for polygons that
        can't possibly touch this tile), not a correctness requirement.
        """
        import shapely.geometry
        import xarray as xr
        import numpy as np
        from odc.geo.geom import Geometry
        from odc.geo.xr import rasterize

        tile_geobox = tile.geobox
        tile_bounds = tile_geobox.boundingbox
        tile_polygon = shapely.geometry.box(tile_bounds.left, tile_bounds.bottom, tile_bounds.right, tile_bounds.top)

        tile_data_vars = {}
        for gid_col, gdf in level_gdfs.items():
            level_tile = np.zeros(tile_geobox.shape, dtype=np.uint32)
            code_to_id = level_code_to_id[gid_col]
            overlap = gdf[gdf.geometry.intersects(tile_polygon)]
            for _, row in overlap.iterrows():
                value = code_to_id[row[gid_col]]
                geom = Geometry(row.geometry, crs=str(gdf.crs))
                mask = rasterize(geom, tile_geobox)
                level_tile = np.where(mask, value, level_tile)
            tile_data_vars[gid_col] = (tile_geobox.dims, level_tile)

        return xr.Dataset(tile_data_vars)

    def _rasterize_levels(self, level_files: List[str], output_path: str) -> bool:
        """Phase 2: tiled rasterization onto `run_tiled_prepare`."""
        from src.data.common.geobox import get_target_geobox
        from src.data.common.prepare.driver import run_tiled_prepare
        from src.data.common.raster.spatial import SpatialProcessor

        import geopandas as gpd

        # One GeoDataFrame + id mapping per ADM level, keyed by that level's
        # own GID column ("ADM_2" file -> "GID_2" column).
        level_gdfs: Dict[str, "gpd.GeoDataFrame"] = {}
        level_code_to_id: Dict[str, Dict[str, int]] = {}
        for level_file in level_files:
            level = _level_from_path(level_file)
            if level is None:
                logger.warning("Skipping unrecognized GADM level file: %s", level_file)
                continue
            gid_col = _gid_column_for_level(level)

            gdf = gpd.read_file(level_file, engine="pyogrio")
            if gid_col not in gdf.columns:
                logger.warning("Level file %s has no %s column, skipping", level_file, gid_col)
                continue

            codes = sorted(gdf[gid_col].unique())
            level_gdfs[gid_col] = gdf
            level_code_to_id[gid_col] = {code: i + 1 for i, code in enumerate(codes)}

        if "GID_0" not in level_gdfs:
            logger.error("GADM rasterization requires an ADM_0/GID_0 level file")
            return False

        with self._dask_client() as client:
            dashboard_link = getattr(client, "dashboard_link", None)
            if dashboard_link:
                logger.info("Created Dask client for GADM rasterization: %s", dashboard_link)

            geobox = get_target_geobox(self.ctx)

            # Reproject once, up front -- _rasterize_tile's per-tile overlap
            # pre-filter compares each level's geometries directly against a
            # tile_polygon built in the *target* geobox's CRS via plain
            # shapely `.intersects()`, which never reprojects itself. See
            # reproject_for_tile_overlap()'s docstring for why skipping this
            # silently produces ~100%-null GRID output with no exception
            # (the bug this line fixes, commit f653033).
            level_gdfs = {gid_col: reproject_for_tile_overlap(gdf, geobox.crs) for gid_col, gdf in level_gdfs.items()}

            processor = SpatialProcessor(hpc_root=self.ctx.data_root, target_geobox=geobox)
            ok = run_tiled_prepare(
                output_path=output_path,
                years=None,
                target_geobox=geobox,
                processor=processor,
                raw_getter=lambda tile, year: self._rasterize_tile(level_gdfs, level_code_to_id, tile),
                reproject=False,
                processing_version=self.PROCESSING_VERSION,
                override=self.cfg.override,
            )
            if not ok:
                return False

        # ADM_AGG, not the CRS_AGG grid-store directory -- these sidecars
        # are read via gid_mapping_path() above, which looks in the ADM_AGG
        # bucket alongside the simplified `.gpkg` vectors.
        adm_dir = self._vector_dir()
        os.makedirs(adm_dir, exist_ok=True)
        for gid_col, code_to_id in level_code_to_id.items():
            with open(os.path.join(adm_dir, f"{gid_col}_code_mapping.json"), "w") as f:
                json.dump(code_to_id, f, indent=2)
        return True

    def _execute_prepare(self, target: StepTarget) -> bool:
        from src.data.sources.steps import is_complete, mark_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping GADM processing -- already complete: %s", target.output_path)
            return True

        vector_dir = self._vector_dir()
        level_files = [] if self.cfg.override else self._existing_level_files(vector_dir)
        if not level_files:
            level_files = self._simplify_vector_levels(target.inputs[0], vector_dir)
            if not level_files:
                return False

        if not self._rasterize_levels(level_files, target.output_path):
            return False
        # Redundant with run_tiled_prepare's own internal mark_complete() in
        # the real path (harmless double-write of the same marker file) --
        # kept explicit here so `_execute_prepare`'s own completion contract
        # doesn't depend on what `_rasterize_levels` happens to delegate to
        # internally (e.g. tests stubbing `_rasterize_levels` directly).
        mark_complete(target.output_path)
        return True

    # _dask_client: inherited from DataSource (src/data/sources/base.py).


registry.register(GadmSource.ID, __name__, GadmSource.__name__, GadmSource.STEPS)

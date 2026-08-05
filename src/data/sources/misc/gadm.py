"""GADM administrative boundaries: fetch + prepare + grid.

docs/design/09-integrated-pipeline.md §7 (the misc split). Ports the
GADM-specific slice of `src/data/download/sources/misc.py::MiscDataSource`
and `src/data/preprocess/sources/misc.py::MiscPreprocessor`'s
`_process_gadm_target`/`_rasterize_gadm_target`/`_create_empty_gadm_zarr`/
`_process_gadm_tiles` (`stage="vector"` -> PREPARE, `stage="spatial"` ->
GRID). Output paths unchanged: `misc/processed/stage_1/gadm/gadm_level*_simplified.gpkg`,
`misc/processed/stage_2/gadm/countries_grid.zarr` -- so `snl_mining`'s config and
`src/data/common/neighbourhood/`'s cross-border masking (docs/design/03-neighbourhood-engine.md
§5) need no edits.

GRID rasterizes every ADM level present in PREPARE's output (not just ADM_0/ADM_1),
one zarr variable per level named after its GADM id column (`GID_0`, `GID_1`, ...,
matching the level's own `ADM_N` suffix) rather than the old `country`/`subdivision`
names. Each variable holds uint32 integer ids (0 = no unit at that level), with a
`{GID_N}_code_mapping.json` sidecar per level mapping GADM's string GID code to the
integer id. `country_code_mapping.json`/`subdivision_code_mapping.json` are gone --
`src/analysis/subsets/registry.py` and `country_classifications.py` were updated to
read `GID_0_code_mapping.json` and the `GID_0` variable instead.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
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
from src.data.sources import verify

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


def gid_mapping_path(data_root: str, grid_id: str, layout_mode: str, gid_col: str) -> str:
    """Path to gadm GRID's `{gid_col}_code_mapping.json` sidecar for another
    source to consult (docs/design/09-integrated-pipeline.md §2: cross-source
    coupling is on artefact paths, never a class import). Shared by every
    source that needs to translate its own native GADM string codes (e.g.
    `GID_1` values like "USA.1_1") into the same integer ids gadm's own
    per-pixel `GID_N` grid uses, so a small per-GID table can be merged
    directly onto assembled rows instead of being rasterized
    (`src.data.assemble.processors.TileProcessor`'s `join_on` mechanism)."""
    gadm_zarr = layout.grid_store_path(
        data_root, "misc", "countries_grid.zarr", namespace="gadm",
        grid_id=grid_id, layout=layout_mode, v2_family="country_id",
    )
    return os.path.join(os.path.dirname(gadm_zarr), f"{gid_col}_code_mapping.json")


class GadmSource(ConfiguredFilesFetchMixin, DataSource):
    """GADM v4.1 administrative boundaries -- simplified vector levels +
    rasterized per-level GID id grid (GID_0, GID_1, ...)."""

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
                # Directory output (variable number of per-level .gpkg files) --
                # same MARKER policy _execute_grid already uses for its own
                # directory (zarr) output, so `pipeline plan`/is_complete() can
                # actually see completion instead of always reporting pending.
                completion=Completion.MARKER,
            )
        ]

    def _execute_prepare(self, target: StepTarget) -> bool:
        import geopandas as gpd

        from src.data.sources.steps import is_complete, mark_complete

        output_base = target.output_path
        if not self.cfg.override and is_complete(target):
            logger.info("Skipping GADM processing, outputs already exist in: %s", output_base)
            return True
        if not self.cfg.override and os.path.exists(output_base):
            # Pre-existing runs from before the MARKER policy was added: the
            # marker won't exist yet even though the level files do. Fall
            # back to the old glob check so completed output isn't silently
            # redone, and write the marker now so future runs see it via
            # is_complete() above.
            existing = [f for f in os.listdir(output_base) if f.startswith("gadm_level") and f.endswith("_simplified.gpkg")]
            if existing:
                logger.info("Skipping GADM processing, outputs already exist in: %s", output_base)
                mark_complete(output_base)
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
        mark_complete(output_base)
        return True

    # -- GRID ("spatial") -- tiled rasterization -----------------------------

    def _plan_grid(self) -> List[StepTarget]:
        vector_dir = self.output_root(PipelineStep.PREPARE)
        adm0_file = os.path.join(vector_dir, "gadm_levelADM_0_simplified.gpkg")
        if not os.path.exists(adm0_file):
            return []
        level_files = sorted(
            Path(vector_dir).glob("gadm_level*_simplified.gpkg"),
            key=lambda p: int(_level_from_path(str(p)).split("_")[1]),
        )
        inputs = tuple(str(p) for p in level_files)
        gid_cols = tuple(_gid_column_for_level(_level_from_path(str(p))) for p in level_files)
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
                # No value_range by default: 0 = "no unit at this level",
                # else a sequential id -- there's no fixed upper bound
                # (depends on how many polygons the level has), so
                # verification only checks these variables are present and
                # not all-nodata unless a source `verification:` config
                # block adds one. verify.verification_meta() lets that
                # config block (orchestration/configs/data.yaml) override
                # any of these without a code change.
                meta=verify.verification_meta(self.cfg.raw, expected_vars=gid_cols),
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

        # One GeoDataFrame + id mapping per ADM level present in PREPARE's output,
        # keyed by that level's own GID column ("ADM_2" file -> "GID_2" column).
        level_gdfs: Dict[str, "gpd.GeoDataFrame"] = {}
        level_code_to_id: Dict[str, Dict[str, int]] = {}
        for level_file in target.inputs:
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
            tile_size = 2048
            tiles = GeoboxTiles(geobox, (tile_size, tile_size))

            if not self._create_empty_gadm_zarr(target.output_path, geobox, list(level_gdfs.keys())):
                return False
            if not self._process_gadm_tiles(tiles, target.output_path, level_gdfs, level_code_to_id):
                return False

        for gid_col, code_to_id in level_code_to_id.items():
            with open(os.path.join(output_dir, f"{gid_col}_code_mapping.json"), "w") as f:
                json.dump(code_to_id, f, indent=2)

        mark_complete(target.output_path)
        return True

    # _dask_client: inherited from DataSource (src/data/sources/base.py).

    @staticmethod
    def _create_empty_gadm_zarr(output_path: str, geobox, gid_columns: List[str]) -> bool:
        import dask.array as da
        import numpy as np
        import xarray as xr

        try:
            ny, nx = geobox.shape
            dim_y, dim_x = geobox.dimensions
            y_coords = geobox.coords[dim_y].values.round(5)
            x_coords = geobox.coords[dim_x].values.round(5)

            data_vars = {
                gid_col: xr.DataArray(
                    da.zeros((ny, nx), dtype=np.uint32, chunks=(512, 512)),
                    dims=[dim_y, dim_x],
                    coords={dim_y: y_coords, dim_x: x_coords},
                    attrs={"description": f"{gid_col} id grid (0=no unit at this level)", "_FillValue": 0},
                )
                for gid_col in gid_columns
            }

            ds = xr.Dataset(
                data_vars,
                attrs={
                    "description": "GADM administrative boundaries grid",
                    "source": "GADM administrative boundaries",
                    "date_created": datetime.now().isoformat(),
                    "levels_included": ", ".join(sorted(gid_columns)),
                },
            )
            # .rio.write_crs() records the CRS as each data variable's own
            # encoding["grid_mapping"] = "spatial_ref" (not an attr) and
            # strips any pre-existing "crs" attr key -- so both the
            # `"grid_mapping"` entry below (without it, the explicit
            # encoding= dict passed to to_zarr() silently drops the link)
            # and re-setting the "crs" attr *after* write_crs() (a redundant
            # fallback) must come after this call, not before.
            ds = ds.rio.write_crs(geobox.crs)
            ds.attrs["crs"] = str(geobox.crs)

            compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle", blocksize=0)
            encoding = {
                v: {"chunks": (512, 512), "compressors": compressor, "dtype": "uint32", "grid_mapping": "spatial_ref"}
                for v in data_vars
            }
            ds.to_zarr(output_path, mode="w", encoding=encoding, compute=False, consolidated=False)
            return True
        except Exception:
            logger.exception("Error creating empty GADM zarr")
            return False

    @staticmethod
    def _process_gadm_tiles(
        tiles,
        output_path: str,
        level_gdfs: Dict[str, "gpd.GeoDataFrame"],
        level_code_to_id: Dict[str, Dict[str, int]],
    ) -> bool:
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

                        overlapping_by_level = {
                            gid_col: gdf[gdf.geometry.intersects(tile_polygon)]
                            for gid_col, gdf in level_gdfs.items()
                        }

                        if all(len(overlap) == 0 for overlap in overlapping_by_level.values()):
                            processed_tiles += 1
                            continue

                        tile_shape = tile_geobox.shape
                        tile_dim_y, tile_dim_x = tile_geobox.dimensions
                        tile_coords = {
                            tile_dim_y: tile_geobox.coords[tile_dim_y].values.round(5),
                            tile_dim_x: tile_geobox.coords[tile_dim_x].values.round(5),
                        }

                        tile_data_vars = {}
                        for gid_col, overlap in overlapping_by_level.items():
                            level_tile = np.zeros(tile_shape, dtype=np.uint32)
                            gdf = level_gdfs[gid_col]
                            code_to_id = level_code_to_id[gid_col]
                            for _, row in overlap.iterrows():
                                value = code_to_id[row[gid_col]]
                                geom = Geometry(row.geometry, crs=str(gdf.crs))
                                mask = rasterize(geom, tile_geobox)
                                level_tile = np.where(mask, value, level_tile)
                            tile_data_vars[gid_col] = xr.DataArray(
                                level_tile, dims=[tile_dim_y, tile_dim_x], coords=tile_coords,
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

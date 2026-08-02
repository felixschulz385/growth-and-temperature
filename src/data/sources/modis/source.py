"""MODIS LST (Planetary Computer STAC streaming): prepare + grid, no fetch.

docs/design/09-integrated-pipeline.md §5/§6: MODIS was already closest to the
target shape before this migration -- no download-side counterpart exists
(it streams via STAC inside its own "annual"/PREPARE step), and its output is
already atomically-written GeoTIFFs with per-tile-year transfer units. This
migration is mostly a rename: `stage="annual"` -> PREPARE, `stage="spatial"`
-> GRID, ported from `src/data/preprocess/sources/modis.py::MODISPreprocessor`.

**Real bug fixed here, not silently ported**: the old
`get_preprocessor_class()` only matched `--source` values `"modis"`/
`"modis_lst"`; `orchestration/configs/data.yaml`'s second MODIS config block
is keyed `modis_robustness_11a1`, which does not match either string and
falls through to the generic camelcase-import branch, raising
`ModuleNotFoundError` (verified by direct execution) -- the exact same bug
class as `--source eog_viirs` (docs/design/09-integrated-pipeline.md §1).
Fixed here by registering `modis_robustness_11a1` as a real alias.

**Quirk deliberately preserved, not "fixed"**: unlike every other source's
GRID/spatial step, MODIS's `_process_spatial_target` never checks
`override`/output-existence before writing a year into the shared multi-year
zarr -- it always re-runs. Modeled here as `Completion.NEVER` on GRID
targets rather than inventing a skip-if-exists check that never existed
(pinned by tests/data/preprocess/sources/test_characterization_modis.py
against the old code).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.base import DataSource
from src.data.sources.modis import tiles as modis_util
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection, TransferUnit

logger = logging.getLogger(__name__)

BAND_SPECS = {
    "21A2": {
        "collection": "modis-21A2-061",
        "assets": {
            "lst": {"name": "LST_Night_1KM", "scale": 0.02, "offset": 0.0, "fill": 0},
            "qc": {"name": "QC_Night", "scale": None, "offset": None, "fill": None},
            "emis_29": {"name": "Emis_29", "scale": 0.002, "offset": 0.49, "fill": 0},
            "emis_31": {"name": "Emis_31", "scale": 0.002, "offset": 0.49, "fill": 0},
            "emis_32": {"name": "Emis_32", "scale": 0.002, "offset": 0.49, "fill": 0},
            "view_angle": {"name": "View_Angle_Night", "scale": 1.0, "offset": 0.0, "fill": None},
            "view_time": {"name": "View_Time_Night", "scale": 0.1, "offset": 0.0, "fill": None},
        },
    },
    "11A1": {
        "collection": "modis-11A1-061",
        "assets": {
            "lst": {"name": "LST_Night_1km", "scale": 0.02, "offset": 0.0, "fill": 0},
            "qc": {"name": "QC_Night", "scale": None, "offset": None, "fill": None},
            "emis_31": {"name": "Emis_31", "scale": 0.002, "offset": 0.0, "fill": 0},
            "emis_32": {"name": "Emis_32", "scale": 0.002, "offset": 0.0, "fill": 0},
            "view_angle": {"name": "Night_view_angl", "scale": 1.0, "offset": 0.0, "fill": 0},
            "view_time": {"name": "Night_view_time", "scale": 0.1, "offset": 0.0, "fill": 0},
        },
    },
}

DEFAULT_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SPATIAL_RESAMPLING = "nearest"


class ModisSource(DataSource):
    ID = "modis"
    ALIASES = ("modis_lst", "modis_robustness_11a1")
    STEPS = (PipelineStep.PREPARE, PipelineStep.GRID)

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        self.product = cfg.raw.get("product", "21A2")
        if self.product not in BAND_SPECS:
            raise ValueError(f"Unsupported product '{self.product}'. Use one of {list(BAND_SPECS)}.")
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path=f"modis/{self.product}")  # old MODISPreprocessor default
        super().__init__(ctx, cfg)

        self.band_spec = BAND_SPECS[self.product]
        self.collection_id = self.band_spec["collection"]
        self.platform = cfg.raw.get("platform", "aqua")

        self.lat_clip_deg = float(cfg.raw.get("lat_clip_deg", 60.0))
        land_tiles = cfg.raw.get("land_tiles")
        land_tiles_set = set(land_tiles) if land_tiles else None
        self.tiles = cfg.raw.get("tiles") or modis_util.get_modis_sinusoidal_tiles(
            self.lat_clip_deg, land_tiles=land_tiles_set
        )

        self.qc_max_lst_error_k = float(cfg.raw.get("qc_max_lst_error_k", 2.0))
        self.stac_url = cfg.raw.get("stac_url", DEFAULT_STAC_URL)

        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix="modis_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)

    def output_root(self, step: PipelineStep, *, namespace: str | None = None) -> str:
        """Overrides the base default: the old MODISPreprocessor hardcodes
        `stage_2_ease6933` for its spatial output regardless of any global
        grid config -- the one deliberate MODIS-only ad hoc case
        docs/design/05-migration.md §1 / docs/design/09-integrated-pipeline.md
        §3 describe. Force `grid_id="ease6933"` for GRID here rather than
        deferring to `self.ctx.grid_id` (which defaults to legacy_4326)."""
        from src.data.sources import layout
        from src.data.sources.layout import EASE_GRID_ID

        if step is PipelineStep.GRID:
            return layout.output_root(
                self.ctx.data_root,
                self.cfg.data_path,
                step,
                namespace=namespace,
                grid_id=EASE_GRID_ID,
                layout=self.ctx.layout,
            )
        return super().output_root(step, namespace=namespace)

    # ------------------------------------------------------------------
    # transfer_units -- per-(year, tile) files for PREPARE, default for GRID
    # ------------------------------------------------------------------

    def transfer_units(self, step: PipelineStep) -> List[TransferUnit]:
        self._require_step(step)
        if step is not PipelineStep.PREPARE:
            return super().transfer_units(step)

        stage1_root = self.output_root(PipelineStep.PREPARE)
        remote_base = self.ctx.remote_data_root or self.ctx.data_root
        units = []
        if not os.path.isdir(stage1_root):
            return units
        for year_name in sorted(os.listdir(stage1_root)):
            year_dir = os.path.join(stage1_root, year_name)
            if not os.path.isdir(year_dir):
                continue
            for tile_name in sorted(os.listdir(year_dir)):
                if not tile_name.endswith(".tif"):
                    continue
                local_path = os.path.join(year_dir, tile_name)
                units.append(
                    TransferUnit(
                        unit_id=f"{year_name}/{tile_name}",
                        local_path=local_path,
                        remote_path=os.path.relpath(local_path, remote_base),
                    )
                )
        return units

    # ------------------------------------------------------------------
    # plan()/execute() dispatch
    # ------------------------------------------------------------------

    def _plan(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        if step is PipelineStep.PREPARE:
            return self._plan_prepare(selection)
        if step is PipelineStep.GRID:
            return self._plan_grid(selection)
        raise AssertionError(f"unreachable: {step}")

    def _execute(self, target: StepTarget) -> bool:
        if target.step is PipelineStep.PREPARE:
            return self._execute_prepare(target)
        if target.step is PipelineStep.GRID:
            return self._execute_grid(target)
        raise AssertionError(f"unreachable: {target.step}")

    # -- PREPARE ("annual": STAC streaming ingest + compositing) -----------

    def _plan_prepare(self, selection: TargetSelection) -> List[StepTarget]:
        stage1_root = self.output_root(PipelineStep.PREPARE)
        targets = []
        for tile in self.tiles:
            years = self.cfg.year_range and range(self.cfg.year_range[0], self.cfg.year_range[1] + 1) or []
            for year in years:
                if not selection.matches_year(year):
                    continue
                key = f"{year}/{tile}"
                if not selection.matches_key(key):
                    continue
                targets.append(
                    StepTarget(
                        source_id=self.cfg.source_id,
                        step=PipelineStep.PREPARE,
                        key=key,
                        output_path=os.path.join(stage1_root, str(year), f"{tile}.tif"),
                        completion=Completion.PATH_EXISTS,
                        meta={"year": year, "tile": tile},
                    )
                )
        return targets

    def _get_stac_client(self):
        import planetary_computer
        import pystac_client

        return pystac_client.Client.open(self.stac_url, modifier=planetary_computer.sign_inplace)

    def _tile_bbox_4326(self, tile: str) -> List[float]:
        from pyproj import Transformer

        h, v = int(tile[1:3]), int(tile[4:6])
        x0, y0, x1, y1 = modis_util.tile_bounds_m(h, v)
        transformer = Transformer.from_crs(modis_util.SINUSOIDAL_PROJ4, "EPSG:4326", always_xy=True)
        lons, lats = transformer.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
        return [min(lons), min(lats), max(lons), max(lats)]

    def _search_items(self, tile: str, year: int) -> list:
        client = self._get_stac_client()
        bbox = self._tile_bbox_4326(tile)
        search = client.search(collections=[self.collection_id], bbox=bbox, datetime=f"{year}-01-01/{year}-12-31")
        items = list(search.items())

        id_prefix = "MYD" if self.platform == "aqua" else "MOD"
        filtered = []
        for item in items:
            platform_ok = item.properties.get("platform") == self.platform
            id_ok = item.id.startswith(id_prefix)
            if platform_ok != id_ok:
                logger.warning(
                    "STAC item %s: platform property (%s) and id prefix disagree", item.id, item.properties.get("platform")
                )
            if platform_ok:
                filtered.append(item)
        return filtered

    def _load_tile_year(self, items: list) -> Optional[xr.Dataset]:
        import odc.stac

        assets = self.band_spec["assets"]
        bands = [spec["name"] for spec in assets.values()]
        ds = odc.stac.load(items, bands=bands, chunks={"time": 1, "x": 2400, "y": 2400}, resampling="nearest")
        if not ds.data_vars:
            return None

        renamed = {}
        for key, spec in assets.items():
            asset_name = spec["name"]
            if asset_name not in ds.data_vars:
                continue
            data = ds[asset_name]
            if spec["scale"] is not None:
                fill = spec.get("fill")
                if fill is not None:
                    data = data.where(data != fill)
                data = data * spec["scale"] + (spec["offset"] or 0.0)
            renamed[key] = data
        return xr.Dataset(renamed).assign_attrs(ds.attrs)

    def _execute_prepare(self, target: StepTarget) -> bool:
        from src.data.common.raster.compositing import composite_to_annual
        from src.data.sources.steps import is_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping %s/%s -- output exists: %s", target.meta["year"], target.meta["tile"], target.output_path)
            return True

        year, tile = target.meta["year"], target.meta["tile"]
        items = self._search_items(tile, year)
        if not items:
            logger.warning("No STAC items found for tile=%s year=%d", tile, year)
            return False

        ds = self._load_tile_year(items)
        if ds is None or "lst" not in ds.data_vars or "qc" not in ds.data_vars:
            logger.error("Failed to load required bands for tile=%s year=%d", tile, year)
            return False

        valid_mask = modis_util.decode_qc_valid_mask(ds["qc"], self.qc_max_lst_error_k)
        annual_lst, monthly_lst, monthly_count, annual_count = composite_to_annual(ds["lst"], valid_mask)

        data_vars = {
            "lst_night": annual_lst.squeeze("time", drop=True).astype("float32"),
            "lst_night_monthly": monthly_lst.astype("float32"),
            "valid_period_count_monthly": monthly_count.astype("float32"),
            "valid_period_count_annual": annual_count.squeeze("time", drop=True).astype("float32"),
        }
        for key in ("emis_29", "emis_31", "emis_32", "view_angle", "view_time"):
            if key in ds.data_vars:
                annual_var, _, _, _ = composite_to_annual(ds[key], valid_mask)
                data_vars[key] = annual_var.squeeze("time", drop=True).astype("float32")

        out_ds = xr.Dataset(data_vars)
        out_ds = out_ds.rio.write_crs(modis_util.SINUSOIDAL_PROJ4)
        out_ds.attrs.update(
            {"source_type": "modis", "product": self.product, "tile": tile, "collection": self.collection_id, "platform": self.platform}
        )
        return self._write_annual_geotiff(out_ds, target.output_path)

    @staticmethod
    def _write_annual_geotiff(ds: xr.Dataset, output_path: str) -> bool:
        import rasterio

        band_arrays: List[np.ndarray] = []
        band_names: List[str] = []

        for var in ("lst_night", "valid_period_count_annual", "emis_29", "emis_31", "emis_32", "view_angle", "view_time"):
            if var not in ds.data_vars:
                continue
            arr = ds[var]
            for dim in ("time", "band"):
                if dim in arr.dims:
                    arr = arr.squeeze(dim, drop=True)
            band_arrays.append(np.asarray(arr.values, dtype="float32"))
            band_names.append(var)

        for var in ("lst_night_monthly", "valid_period_count_monthly"):
            if var not in ds.data_vars:
                continue
            monthly = ds[var]
            if "band" in monthly.dims:
                monthly = monthly.squeeze("band", drop=True)
            for i in range(monthly.sizes.get("time", 0)):
                month_label = pd.Timestamp(monthly["time"].values[i]).strftime("%m")
                band_arrays.append(np.asarray(monthly.isel(time=i).values, dtype="float32"))
                band_names.append(f"{var}_{month_label}")

        if not band_arrays:
            logger.error("No bands to write for %s", output_path)
            return False

        stacked = np.stack(band_arrays, axis=0)
        transform = ds.rio.transform()
        crs = ds.rio.crs

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tmp_path = output_path + ".tmp"
        try:
            with rasterio.open(
                tmp_path, "w", driver="GTiff", height=stacked.shape[1], width=stacked.shape[2],
                count=stacked.shape[0], dtype="float32", crs=crs, transform=transform,
                nodata=np.nan, compress="deflate", predictor=3, tiled=True,
            ) as dst:
                dst.write(stacked)
                for i, name in enumerate(band_names, start=1):
                    dst.set_band_description(i, name)
                dst.update_tags(**{k: str(v) for k, v in ds.attrs.items()})
            os.replace(tmp_path, output_path)
            return True
        except Exception:
            logger.exception("Error writing MODIS annual GeoTIFF to %s", output_path)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False

    # -- GRID ("spatial": mosaic tiles, reproject onto canonical EPSG:6933) --

    def _plan_grid(self, selection: TargetSelection) -> List[StepTarget]:
        from src.data.sources import layout
        from src.data.sources.layout import EASE_GRID_ID

        stage1_root = self.output_root(PipelineStep.PREPARE)
        output_path = layout.grid_store_path(
            self.ctx.data_root,
            self.cfg.data_path,
            f"modis_{self.product}_timeseries_reprojected.zarr",
            grid_id=EASE_GRID_ID,
            layout=self.ctx.layout,
            v2_family=f"modis_lst_{self.product.lower()}",
        )
        years = self.cfg.year_range and list(range(self.cfg.year_range[0], self.cfg.year_range[1] + 1)) or []

        targets = []
        for year in years:
            if not selection.matches_year(year) or not selection.matches_key(str(year)):
                continue
            year_dir = os.path.join(stage1_root, str(year))
            if not os.path.isdir(year_dir):
                logger.warning("No stage-1 output for year %d at %s", year, year_dir)
                continue
            tile_files = sorted(os.path.join(year_dir, f) for f in os.listdir(year_dir) if f.endswith(".tif"))
            if not tile_files:
                continue
            targets.append(
                StepTarget(
                    source_id=self.cfg.source_id,
                    step=PipelineStep.GRID,
                    key=str(year),
                    output_path=output_path,
                    inputs=tuple(tile_files),
                    # Quirk preserved, not invented -- see module docstring:
                    # the old spatial stage never checked override/existence.
                    completion=Completion.NEVER,
                    meta={"year": year, "years_all": years},
                )
            )
        return targets

    def _read_annual_geotiff(self, path: str, year: int) -> xr.Dataset:
        import rasterio
        import rioxarray as rxr

        da = rxr.open_rasterio(path, masked=True)
        with rasterio.open(path) as src:
            descriptions = src.descriptions

        time_coord = [pd.Timestamp(f"{year}-12-31")]
        data_vars = {}
        for i, name in enumerate(descriptions):
            if not name or "_monthly_" in name:
                continue
            band_da = da.isel(band=i, drop=True)
            band_da = band_da.expand_dims(time=time_coord, axis=0).expand_dims(band=[1], axis=1)
            data_vars[name] = band_da

        ds = xr.Dataset(data_vars)
        return ds.rio.write_crs(da.rio.crs)

    def _mosaic_tiles(self, tile_files: List[str], year: int) -> xr.Dataset:
        datasets = [self._read_annual_geotiff(f, year) for f in tile_files]
        return xr.combine_by_coords(datasets, combine_attrs="override")

    def _execute_grid(self, target: StepTarget) -> bool:
        from src.data.common.geobox import get_or_create_canonical_geobox
        from src.data.common.raster.spatial import SpatialProcessor

        year = target.meta["year"]
        try:
            with self._dask_client() as client:
                cache_path = os.path.join(self.ctx.data_root, "canonical_geobox.pkl")
                target_geobox = get_or_create_canonical_geobox(cache_path)

                spatial_processor = SpatialProcessor(
                    hpc_root=self.ctx.data_root, temp_dir=self.temp_dir, dask_client=client, target_geobox=target_geobox
                )

                with spatial_processor.setup_dask_config():
                    mosaic = self._mosaic_tiles(list(target.inputs), year)
                    if mosaic.rio.crs is None:
                        mosaic = mosaic.rio.write_crs(modis_util.SINUSOIDAL_PROJ4)

                    if not os.path.exists(target.output_path):
                        variables = list(mosaic.data_vars.keys())
                        if not spatial_processor.create_empty_target_zarr(
                            target.output_path, target_geobox, target.meta["years_all"], variables,
                            sample_attrs=mosaic.attrs, packaging_attrs={}, dst_nodata=float("nan"), dtype="float32",
                        ):
                            return False

                    return spatial_processor.write_year_to_zarr(
                        mosaic, target.output_path, year, target_geobox,
                        resampling=SPATIAL_RESAMPLING, dst_nodata=float("nan"),
                    )
        except Exception:
            logger.exception("Error in MODIS spatial processing for year %d.", year)
            return False

    # _dask_client: inherited from DataSource (src/data/sources/base.py).

registry.register(
    ModisSource.ID,
    __name__,
    ModisSource.__name__,
    ModisSource.STEPS,
    aliases=ModisSource.ALIASES,
)

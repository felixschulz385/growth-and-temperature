"""
Preprocessor for MODIS LST ingest from Microsoft Planetary Computer's STAC
catalog (docs/design/07-modis-ingest.md, docs/design/07a-modis-band-
reference.md).

Not a downloader: stage "annual" streams remote COGs via STAC and reduces
them to an annual composite in native sinusoidal projection in flight,
persisting only the composite and its diagnostics. Two-stage shape mirrors
`ACAGPreprocessor` (src/data/preprocess/sources/acag.py) -- the cleanest
existing instance of the "annual then spatial" pattern -- rather than
`GlassPreprocessor`, whose MODIS/AVHRR dual-source complexity this ingest
doesn't need; `GlassPreprocessor._calculate_statistics`'s resample-based
compositing is still the prior art `composite_to_annual`
(src/data/preprocess/common/compositing.py) generalizes, minus its
annual-directly-from-daily shortcut.
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from src.data.preprocess.sources.base import AbstractPreprocessor
from src.data.preprocess.sources import modis_util
from src.data.preprocess.common.compositing import composite_to_annual
from src.data.preprocess.common.spatial import SpatialProcessor

logger = logging.getLogger(__name__)

# Verified per-band facts (docs/design/07a-modis-band-reference.md). Assets
# not needed by this pipeline (hdf, metadata, day-time bands) are omitted.
# Emis offset for the 11-family is UNVERIFIED (07a: "do not assume the two
# products share the same offset") -- left at 0.0 pending the resolution
# logged in docs/design/06-open-questions.md; a wrong value here is a
# one-line fix, not a reason to block this module.
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
            "emis_31": {"name": "Emis_31", "scale": 0.002, "offset": 0.0, "fill": 0},  # UNVERIFIED offset, see 07a
            "emis_32": {"name": "Emis_32", "scale": 0.002, "offset": 0.0, "fill": 0},  # UNVERIFIED offset, see 07a
            "view_angle": {"name": "Night_view_angl", "scale": 1.0, "offset": 0.0, "fill": 0},
            "view_time": {"name": "Night_view_time", "scale": 0.1, "offset": 0.0, "fill": 0},
        },
    },
}

DEFAULT_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Resampling per variable for stage "spatial"'s reprojection onto EPSG:6933
# (docs/design/07-modis-ingest.md §6 table) -- every MODIS band is an
# intensive/categorical quantity, never averaged.
SPATIAL_RESAMPLING = "nearest"


class MODISPreprocessor(AbstractPreprocessor):
    """
    HPC-mode preprocessor for MODIS LST (MYD21A2 primary, MYD11A1 bounded
    robustness arm -- docs/design/07-modis-ingest.md §1).

    Stages
    ------
    annual  -- one target per (tile, year): stream night COGs for that
        sinusoidal tile/year from Planetary Computer's STAC catalog, apply
        QC, composite month-first-then-annual (`composite_to_annual`), and
        write a native-sinusoidal-projection zarr.
    spatial -- one target per year: mosaic that year's available tiles in
        native sinusoidal projection, reproject onto the canonical EPSG:6933
        GeoBox (nearest-neighbour, docs/design/01-grid.md), and region-write
        into a shared multi-year zarr.

    Configuration keys
    -------------------
    stage : "annual" | "spatial"
    product : "21A2" (default, primary) | "11A1" (robustness arm)
    platform : "aqua" (default, per docs/design/07-modis-ingest.md §1)
    year / year_range : one required
    hpc_target : required, local HPC root
    data_path : sub-path under hpc_root (default "modis/<product>")
    tiles : optional explicit tile-id list override (e.g. for the bounded
        robustness-arm subset); defaults to the computed |lat|<=lat_clip_deg
        tile list
    lat_clip_deg : default 60.0
    land_tiles : optional allowlist of land-covering tile ids
    qc_max_lst_error_k : default 2.0 (configurable per docs/design/07 §6 --
        the QC bit layout is UNVERIFIED, this threshold is not)
    stac_url : default Planetary Computer's STAC v1 endpoint
    override : re-process even when output already exists
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.stage = kwargs.get("stage", "annual")
        if self.stage not in ("annual", "spatial"):
            raise ValueError(f"Unsupported stage '{self.stage}'. Use 'annual' or 'spatial'.")

        self.product = kwargs.get("product", "21A2")
        if self.product not in BAND_SPECS:
            raise ValueError(f"Unsupported product '{self.product}'. Use one of {list(BAND_SPECS)}.")
        self.band_spec = BAND_SPECS[self.product]
        self.collection_id = self.band_spec["collection"]

        self.platform = kwargs.get("platform", "aqua")

        self.year = kwargs.get("year")
        self.year_range = kwargs.get("year_range")
        if self.year is None and self.year_range is None:
            raise ValueError("Either 'year' or 'year_range' must be specified.")
        if self.year is not None:
            self.years_to_process = [self.year]
        else:
            if not isinstance(self.year_range, (list, tuple)) or len(self.year_range) != 2:
                raise ValueError("'year_range' must be [start_year, end_year].")
            self.years_to_process = list(range(int(self.year_range[0]), int(self.year_range[1]) + 1))

        hpc_target = kwargs.get("hpc_target")
        self.hpc_root = self._strip_remote_prefix(hpc_target)
        if not self.hpc_root:
            raise ValueError("'hpc_target' is required.")

        self.data_path = kwargs.get("data_path") or f"modis/{self.product}"

        self.lat_clip_deg = float(kwargs.get("lat_clip_deg", 60.0))
        land_tiles = kwargs.get("land_tiles")
        land_tiles_set = set(land_tiles) if land_tiles else None
        self.tiles = kwargs.get("tiles") or modis_util.get_modis_sinusoidal_tiles(
            self.lat_clip_deg, land_tiles=land_tiles_set
        )

        self.qc_max_lst_error_k = float(kwargs.get("qc_max_lst_error_k", 2.0))
        self.stac_url = kwargs.get("stac_url", DEFAULT_STAC_URL)

        self.override = kwargs.get("override", False)
        self.dask_threads = kwargs.get("dask_threads")
        self.dask_memory_limit = kwargs.get("dask_memory_limit")

        self.temp_dir = kwargs.get("temp_dir") or tempfile.mkdtemp(prefix="modis_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)

        logger.info(
            "Initialised MODISPreprocessor product=%s collection=%s hpc_root=%s data_path=%s "
            "years=%d tiles=%d",
            self.product, self.collection_id, self.hpc_root, self.data_path,
            len(self.years_to_process), len(self.tiles),
        )

    # ------------------------------------------------------------------
    # AbstractPreprocessor interface
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MODISPreprocessor":
        return cls(**config)

    def get_hpc_output_path(self, stage: str) -> str:
        if stage == "annual":
            base = os.path.join(self.hpc_root, self.data_path, "processed", "stage_1")
        elif stage == "spatial":
            base = os.path.join(self.hpc_root, self.data_path, "processed", "stage_2_ease6933")
        else:
            raise ValueError(f"Unknown stage: {stage}")
        return self._strip_remote_prefix(base)

    def get_preprocessing_targets(
        self, stage: str, year_range: Tuple[int, int] = None
    ) -> List[Dict[str, Any]]:
        years = self.years_to_process
        if year_range:
            years = [y for y in years if year_range[0] <= y <= year_range[1]]

        if stage == "annual":
            return self._gen_annual_targets(years)
        elif stage == "spatial":
            return self._gen_spatial_targets(years)
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def _gen_annual_targets(self, years: List[int]) -> List[Dict[str, Any]]:
        targets = []
        stage1_root = self.get_hpc_output_path("annual")
        for year in years:
            for tile in self.tiles:
                output_path = os.path.join(stage1_root, str(year), f"{tile}.tif")
                targets.append({
                    "stage": "annual",
                    "year": year,
                    "tile": tile,
                    "output_path": output_path,
                    "dependencies": [],
                    "metadata": {"source_type": "modis", "product": self.product, "tile": tile},
                })
        return targets

    def _gen_spatial_targets(self, years: List[int]) -> List[Dict[str, Any]]:
        stage1_root = self.get_hpc_output_path("annual")
        output_path = os.path.join(
            self.get_hpc_output_path("spatial"), f"modis_{self.product}_timeseries_reprojected.zarr"
        )
        targets = []
        for year in years:
            year_dir = os.path.join(stage1_root, str(year))
            if not os.path.isdir(year_dir):
                logger.warning("No stage-1 output for year %d at %s", year, year_dir)
                continue
            tile_files = sorted(
                os.path.join(year_dir, f) for f in os.listdir(year_dir) if f.endswith(".tif")
            )
            if not tile_files:
                continue
            targets.append({
                "stage": "spatial",
                "year": year,
                "source_files": tile_files,
                "output_path": output_path,
                "dependencies": tile_files,
                "metadata": {"source_type": "modis", "product": self.product, "years_all": years},
            })
        return targets

    def get_transfer_units(self, stage: str) -> List[Dict[str, Any]]:
        """Per-(year, tile) transfer units for stage "annual" -- one GeoTIFF
        file per unit, so `transfer.py` pushes each directly (no tar/extract
        needed for a single file), with per-tile-year resumability matching
        the processing granularity (docs/design/08-hpc-transfer.md §2). Stage
        "spatial" falls back to the default single-unit behaviour since its
        output is already one shared store.
        """
        if stage != "annual":
            return super().get_transfer_units(stage)

        hpc_root = self._strip_remote_prefix(self.config.get("hpc_target")) or self.hpc_root
        stage1_root = self.get_hpc_output_path("annual")
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
                remote_path = os.path.relpath(local_path, hpc_root)
                units.append({
                    "unit_id": f"{year_name}/{tile_name}",
                    "local_path": local_path,
                    "remote_path": remote_path,
                })
        return units

    # ------------------------------------------------------------------
    # process_target dispatcher
    # ------------------------------------------------------------------

    def process_target(self, target: Dict[str, Any]) -> bool:
        stage = target.get("stage")
        try:
            if stage == "annual":
                return self._process_annual_target(target)
            elif stage == "spatial":
                return self._process_spatial_target(target)
            else:
                logger.error("Unknown stage: %s", stage)
                return False
        except Exception:
            logger.exception("Error processing target %s", target)
            return False

    # ------------------------------------------------------------------
    # Stage 1 -- annual (STAC streaming ingest + compositing)
    # ------------------------------------------------------------------

    def _get_stac_client(self):
        import pystac_client
        import planetary_computer
        return pystac_client.Client.open(self.stac_url, modifier=planetary_computer.sign_inplace)

    def _tile_bbox_4326(self, tile: str) -> List[float]:
        from pyproj import Transformer
        h, v = int(tile[1:3]), int(tile[4:6])
        x0, y0, x1, y1 = modis_util.tile_bounds_m(h, v)
        transformer = Transformer.from_crs(modis_util.SINUSOIDAL_PROJ4, "EPSG:4326", always_xy=True)
        xs, ys = [x0, x1, x0, x1], [y0, y0, y1, y1]
        lons, lats = transformer.transform(xs, ys)
        return [min(lons), min(lats), max(lons), max(lats)]

    def _search_items(self, tile: str, year: int) -> list:
        """STAC search for one tile/year, filtered to the configured platform.

        Cross-checks the `platform` property against the MOD/MYD id prefix
        per docs/design/07-modis-ingest.md §6 -- UNVERIFIED which signal is
        authoritative, so a disagreement is logged rather than silently
        trusted.
        """
        client = self._get_stac_client()
        bbox = self._tile_bbox_4326(tile)
        search = client.search(
            collections=[self.collection_id],
            bbox=bbox,
            datetime=f"{year}-01-01/{year}-12-31",
        )
        items = list(search.items())

        id_prefix = "MYD" if self.platform == "aqua" else "MOD"
        filtered = []
        for item in items:
            platform_ok = item.properties.get("platform") == self.platform
            id_ok = item.id.startswith(id_prefix)
            if platform_ok != id_ok:
                logger.warning(
                    "STAC item %s: platform property (%s) and id prefix disagree -- "
                    "see docs/design/07a-modis-band-reference.md platform-filter caveat",
                    item.id, item.properties.get("platform"),
                )
            if platform_ok:
                filtered.append(item)
        return filtered

    def _load_tile_year(self, items: list) -> Optional[xr.Dataset]:
        """Load this tile-year's assets via odc.stac.load.

        UNVERIFIED whether odc.stac.load applies STAC raster:bands
        scale/offset automatically (docs/design/07-modis-ingest.md §6) --
        manual application below is the conservative default; confirm
        against the installed odc-stac version's behaviour with a real read
        before a production run, since silent double-application would
        corrupt every band.
        """
        import odc.stac

        assets = self.band_spec["assets"]
        bands = [spec["name"] for spec in assets.values()]

        ds = odc.stac.load(
            items,
            bands=bands,
            chunks={"time": 1, "x": 2400, "y": 2400},
            resampling="nearest",
        )
        if not ds.data_vars:
            return None

        # Manual scale/offset application (see docstring above).
        renamed = {}
        for key, spec in assets.items():
            asset_name = spec["name"]
            if asset_name not in ds.data_vars:
                continue
            da = ds[asset_name]
            if spec["scale"] is not None:
                fill = spec.get("fill")
                if fill is not None:
                    da = da.where(da != fill)
                da = da * spec["scale"] + (spec["offset"] or 0.0)
            renamed[key] = da
        return xr.Dataset(renamed).assign_attrs(ds.attrs)

    def _process_annual_target(self, target: Dict[str, Any]) -> bool:
        year = target["year"]
        tile = target["tile"]
        output_path = self._strip_remote_prefix(target["output_path"])

        if not self.override and os.path.exists(output_path):
            logger.info("Skipping %s/%s -- output exists: %s", year, tile, output_path)
            return True

        items = self._search_items(tile, year)
        if not items:
            logger.warning("No STAC items found for tile=%s year=%d", tile, year)
            return False

        ds = self._load_tile_year(items)
        if ds is None or "lst" not in ds.data_vars or "qc" not in ds.data_vars:
            logger.error("Failed to load required bands for tile=%s year=%d", tile, year)
            return False

        valid_mask = modis_util.decode_qc_valid_mask(ds["qc"], self.qc_max_lst_error_k)

        annual_lst, monthly_lst, monthly_count, annual_count = composite_to_annual(
            ds["lst"], valid_mask
        )

        # Annual variables are squeezed out of "time" (size 1) here, before
        # dataset construction below -- combining them via xr.Dataset({...})
        # while they still carried a "time" dim would collide with the
        # monthly variables' *different* time coordinate (12 monthly labels
        # vs. 1 annual label), and xarray's dict-based Dataset constructor
        # silently outer-joins/broadcasts mismatched coordinates on a shared
        # dim name -- corrupting the annual variables' shape rather than
        # raising. Verified against a synthetic combine while testing this.
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
        out_ds.attrs.update({
            "source_type": "modis", "product": self.product, "tile": tile,
            "collection": self.collection_id, "platform": self.platform,
        })

        return self._write_annual_geotiff(out_ds, output_path)

    def _write_annual_geotiff(self, ds: xr.Dataset, output_path: str) -> bool:
        """Write this tile-year's composite as one atomic multi-band GeoTIFF.

        A single file, not a Zarr directory: `transfer.py` pushes it via a
        direct rsync with no tar/extract step needed (there's only one
        file), and the write below is trivially atomic (temp file + rename)
        -- a Zarr store's directory-of-many-chunk-files write is not, so a
        killed write could previously leave a partial store on disk that a
        bare `os.path.exists` resume check would wrongly treat as complete.

        Annual (single-band) variables and each month of the monthly
        diagnostic variables become separate, named bands in one file --
        band descriptions record which is which. All values share one
        dtype (float32) since classic GTiff doesn't support per-band
        dtypes; the small integer valid-period counts are exactly
        representable in float32, so nothing is lost.
        """
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
                tmp_path, "w", driver="GTiff",
                height=stacked.shape[1], width=stacked.shape[2], count=stacked.shape[0],
                dtype="float32", crs=crs, transform=transform,
                nodata=np.nan, compress="deflate", predictor=3, tiled=True,
            ) as dst:
                dst.write(stacked)
                for i, name in enumerate(band_names, start=1):
                    dst.set_band_description(i, name)
                dst.update_tags(**{k: str(v) for k, v in ds.attrs.items()})
            os.replace(tmp_path, output_path)
            logger.info("Wrote MODIS annual GeoTIFF: %s (%d bands)", output_path, len(band_names))
            return True
        except Exception:
            logger.exception("Error writing MODIS annual GeoTIFF to %s", output_path)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False

    # ------------------------------------------------------------------
    # Stage 2 -- spatial (mosaic tiles, reproject onto canonical EPSG:6933)
    # ------------------------------------------------------------------

    def _initialize_dask_client(self):
        from src.data.common.dask.client import DaskClientContextManager
        return DaskClientContextManager(
            threads=self.dask_threads,
            memory_limit=self.dask_memory_limit,
            dashboard_port=8787,
            temp_dir=os.path.join(self.temp_dir, "dask_workspace"),
        )

    def _read_annual_geotiff(self, path: str, year: int) -> xr.Dataset:
        """Reconstruct a tile-year's *annual* variables from its GeoTIFF.

        Only the annual (single-band) variables are carried onto the
        canonical-grid mosaic -- the monthly diagnostic bands
        (`*_monthly_MM`) are ingest-time QA/robustness-arm material
        (docs/design/07-modis-ingest.md §5), not one of the canonical-grid
        variables docs/design/04-ingest.md §5 lists, so they stay in the
        pushed tile-year file rather than being reprojected. `time`/`band`
        dims are re-added to match what `SpatialProcessor` expects.
        """
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
        """Merge one year's per-tile native-sinusoidal GeoTIFFs into one mosaic.

        MODIS sinusoidal tiles share resolution/CRS and tile exactly onto a
        shared global sinusoidal grid (docs/design/07-modis-ingest.md §3:
        "ordinary tile-boundary mosaicking, not a coordinate discontinuity"),
        so a coordinate-based combine is sufficient -- no reprojection needed
        at this step.
        """
        datasets = [self._read_annual_geotiff(f, year) for f in tile_files]
        return xr.combine_by_coords(datasets, combine_attrs="override")

    def _process_spatial_target(self, target: Dict[str, Any]) -> bool:
        from src.data.common.geobox import get_or_create_canonical_geobox

        year = target["year"]
        output_path = self._strip_remote_prefix(target["output_path"])
        source_files: List[str] = target.get("source_files", [])
        if not source_files:
            logger.error("No source files listed in spatial target for year %d.", year)
            return False

        try:
            with self._initialize_dask_client() as client:
                cache_path = os.path.join(self.hpc_root, "canonical_geobox.pkl")
                target_geobox = get_or_create_canonical_geobox(cache_path)

                spatial_processor = SpatialProcessor(
                    hpc_root=self.hpc_root,
                    temp_dir=self.temp_dir,
                    dask_client=client,
                    target_geobox=target_geobox,
                )

                with spatial_processor.setup_dask_config():
                    mosaic = self._mosaic_tiles(source_files, year)
                    if mosaic.rio.crs is None:
                        mosaic = mosaic.rio.write_crs(modis_util.SINUSOIDAL_PROJ4)

                    if not os.path.exists(output_path):
                        variables = list(mosaic.data_vars.keys())
                        # Physical (already-unpacked) float32 values -- store
                        # as float32, not the shared default packed uint16,
                        # to avoid truncating LST to whole-Kelvin precision
                        # (docs/design/07-modis-ingest.md §6 resampling table).
                        if not spatial_processor.create_empty_target_zarr(
                            output_path, target_geobox, target["metadata"]["years_all"], variables,
                            sample_attrs=mosaic.attrs, packaging_attrs={},
                            dst_nodata=float("nan"), dtype="float32",
                        ):
                            return False

                    success = spatial_processor.write_year_to_zarr(
                        mosaic, output_path, year, target_geobox,
                        resampling=SPATIAL_RESAMPLING, dst_nodata=float("nan"),
                    )
                    if success:
                        logger.info("MODIS spatial reprojection completed for year %d: %s", year, output_path)
                    return success

        except Exception:
            logger.exception("Error in MODIS spatial processing for year %d.", year)
            return False

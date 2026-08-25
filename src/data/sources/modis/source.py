"""MODIS LST (Planetary Computer STAC streaming): fetch + prepare.

docs/design/09-integrated-pipeline.md §5/§6: MODIS was already closest to the
target shape before that migration -- no separate download-side counterpart
exists (it streams via STAC inside its own "annual" step), and its output is
already atomically-written GeoTIFFs with per-tile-year transfer units. That
migration was mostly a rename: `stage="annual"` -> what was then called
PREPARE, `stage="spatial"` -> GRID, ported from
`src/data/preprocess/sources/modis.py::MODISPreprocessor`.

docs/design/10-fetch-ledger.md §6/§3 later named this step's placement the
one deliberate exception to "FETCH means a crawlable remote file catalog":
STAC search + `odc.stac.load` + QC-mask + annual-compositing genuinely *is*
the download from Planetary Computer, it just also transforms as it goes
(there is no separate raw-asset-on-disk stage to insert a real FETCH before).
This migration renames the step to FETCH -- matching what it actually is,
"fetch this source's remote data" -- rather than continuing to call it
PREPARE. It does **not** implement the full `RemoteFileCatalog` crawler
protocol other FETCH sources (GLASS, EOG, ...) satisfy: MODIS has no flat
remote file list to crawl, only per-(tile, year) STAC queries, so it is
excluded from `tests/data/sources/test_fetch_protocol.py`'s parametrization.

A failed (year, tile) unit's retry/error history lives in a small JSON
sidecar under FETCH's own output root, written via `manifest.record_failure`/
`clear_failure` (`src.data.common.fetch.manifest`, matching every other
source); completion is plain local-disk presence (`Completion.PATH_EXISTS`),
same as everywhere else -- no cross-machine remote-verification gate.
PREPARE (the mosaic + reprojection step, `STEPS = (FETCH, PREPARE)` --
renamed from GRID so MODIS's own step names/CLI/`data summary` columns line
up with every other source's shape; the physical output tier is still
"GRID", see `output_root()`) likewise reads FETCH's tile files straight off
local disk (`_discover_prepare` below).

**Real bug fixed in the original PREPARE migration, not silently ported**:
the old `get_preprocessor_class()` only matched `--source` values `"modis"`/
`"modis_lst"`; `orchestration/configs/data.yaml`'s second MODIS config block
is keyed `modis_robustness_11a1`, which does not match either string and
falls through to the generic camelcase-import branch, raising
`ModuleNotFoundError` (verified by direct execution) -- the exact same bug
class as `--source eog_viirs` (docs/design/09-integrated-pipeline.md §1).
Fixed here by registering `modis_robustness_11a1` as a real alias.

**PREPARE planning fixed to match GLASS's shape**: `_discover_prepare` used
to emit one `StepTarget` per year, all sharing the same `output_path` and
each `Completion.NEVER` (so `data summary` showed e.g. "0/23" for a single
combined zarr, and every run re-wrote every year regardless of what already
existed). It now emits a single `key="all"` target, `Completion.MARKER`,
mirroring GLASS's `_plan_prepare`/`_execute_prepare` split
(`src/data/sources/glass/source.py`): `_execute_prepare` loops over all
available years internally and only calls `mark_complete()` once, at the
end, once every year has been written.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from src.data.common import tiling
from src.data.common.fetch import manifest
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.base import DataSource
from src.data.sources.modis import tiles as modis_util
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection, TransferUnit
from src.data.sources import verify

logger = logging.getLogger(__name__)

BAND_SPECS = {
    # view_angle/view_time fill+offset confirmed against Table 11 ("The SDSs
    # in the MxD21A2 8-day product") in the MxD21 LST&E User Guide (Hulley et
    # al., JPL, March 2019) -- corrects the same class of bug the MOD11A1
    # block below independently caught: View_Angle_Night's offset was 0.0
    # (guide says -65.0) and fill was None/unmasked (guide says 255);
    # View_Time_Night's fill was likewise None (guide says 255). LST and
    # emissivity scale/offset/fill were already correct (matches Table 11
    # exactly).
    "21A2": {
        "collection": "modis-21A2-061",
        "assets": {
            "lst": {"name": "LST_Night_1KM", "scale": 0.02, "offset": 0.0, "fill": 0},
            "qc": {"name": "QC_Night", "scale": None, "offset": None, "fill": None},
            "emis_29": {"name": "Emis_29", "scale": 0.002, "offset": 0.49, "fill": 0},
            "emis_31": {"name": "Emis_31", "scale": 0.002, "offset": 0.49, "fill": 0},
            "emis_32": {"name": "Emis_32", "scale": 0.002, "offset": 0.49, "fill": 0},
            "view_angle": {"name": "View_Angle_Night", "scale": 1.0, "offset": -65.0, "fill": 255},
            "view_time": {"name": "View_Time_Night", "scale": 0.1, "offset": 0.0, "fill": 255},
        },
    },
    # Values below confirmed against Table 9 ("The SDSs in the MOD11A1
    # product") in the Collection-6 MODIS LST Products Users' Guide (Wan,
    # ERI/UCSB, June 2019) -- corrects three values a prior UNVERIFIED
    # assumption got wrong: Emis_31/Emis_32's offset (was 0.0, guide says
    # 0.49, matching MOD21A2's already-confirmed value below), and
    # Night_view_angl/Night_view_time's fill (was 0, guide says 255) --
    # Night_view_angl's offset (was 0.0, guide says -65.0, per that table's
    # note that a negative view angle means MODIS viewed the grid from the
    # east). With the old fill=0, a genuine 0 raw value (a valid -65 degree
    # view angle under the correct offset) was wrongly masked as nodata, and
    # a genuine fill=255 pixel was wrongly kept and scaled into a bogus
    # 255-degree "view angle".
    "11A1": {
        "collection": "modis-11A1-061",
        "assets": {
            "lst": {"name": "LST_Night_1km", "scale": 0.02, "offset": 0.0, "fill": 0},
            "qc": {"name": "QC_Night", "scale": None, "offset": None, "fill": None},
            "emis_31": {"name": "Emis_31", "scale": 0.002, "offset": 0.49, "fill": 0},
            "emis_32": {"name": "Emis_32", "scale": 0.002, "offset": 0.49, "fill": 0},
            "view_angle": {"name": "Night_view_angl", "scale": 1.0, "offset": -65.0, "fill": 255},
            "view_time": {"name": "Night_view_time", "scale": 0.1, "offset": 0.0, "fill": 255},
        },
    },
}

DEFAULT_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SPATIAL_RESAMPLING = "nearest"


#: main variant's lst_night stats (month-weighted -- composite_annual_stats)
MAIN_LST_STATS = ("mean", "median", "std", "valid_period_count", "valid_month_count", "count_above", "count_below")
#: extended variant's bands: everything besides lst_night itself, mean only.
EXTENDED_VARS = ("emis_29", "emis_31", "emis_32", "view_angle", "view_time")


class ModisSource(DataSource):
    """Registered twice (`registry.register()` calls at module end, mirroring
    GLASS's `glass_modis`/`glass_avhrr` dual-registration): `modis` (main --
    lst_night summary stats + valid counts) and `modis_extended` (emissivity
    + view geometry). One class, `self.variant` derived from `cfg.source_id`,
    so both share FETCH's STAC/QC plumbing; each writes its own tiles/PREPARE
    zarr (`data_path`/`family` include the variant so they never collide)."""

    ID = "modis"
    ALIASES = ("modis_lst", "modis_robustness_11a1")
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE)
    DEFAULT_TRANSFER_MODE = "auto"
    #: bump to force a full reprocess (`run_tiled_prepare`'s `processing_version`)
    PROCESSING_VERSION = "1-tiled"

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        self.product = cfg.raw.get("product", "21A2")
        if self.product not in BAND_SPECS:
            raise ValueError(f"Unsupported product '{self.product}'. Use one of {list(BAND_SPECS)}.")
        self.variant = "extended" if "extended" in cfg.source_id.lower() else "main"
        if cfg.data_path is None:
            suffix = "" if self.variant == "main" else "_extended"
            cfg = dataclasses.replace(cfg, data_path=f"modis/{self.product}{suffix}")
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
        # Explicit discrete-year override, e.g. modis_robustness_11a1's "3-5
        # years spanning early/mid/late mission" (docs/design/07-modis-ingest.md
        # §1) -- `year_range` alone can only express one contiguous span, not
        # a handful of non-adjacent years, so this is a separate config key
        # rather than overloading year_range's meaning.
        self.years = cfg.raw.get("years")

        self.qc_max_lst_error_k = float(cfg.raw.get("qc_max_lst_error_k", 2.0))
        # A "good" QC flag alone isn't sufficient: a small fraction of raw
        # pixels carry good QC bits alongside a corrupted decoded value (a
        # physically impossible temperature) -- see
        # decode_qc_valid_mask()'s docstring (src/data/sources/modis/tiles.py)
        # for the observed case that motivated this. Defaults mirror the
        # existing lst_night GRID-verification range (_discover_prepare's
        # value_range=(150, 350)) rather than inventing a second bound.
        self.lst_min_k = float(cfg.raw.get("lst_min_k", 150.0))
        self.lst_max_k = float(cfg.raw.get("lst_max_k", 350.0))
        # count_above/count_below thresholds for the main variant's
        # lst_night stats -- default lines mirror common night heat-health
        # (25C) and freezing (0C) thresholds; GLASS's own gt30C/lt0C
        # (src/data/sources/glass/source.py) is the naming precedent.
        self.heat_stress_k = float(cfg.raw.get("heat_stress_k", 298.15))
        self.cold_stress_k = float(cfg.raw.get("cold_stress_k", 273.15))
        self.stac_url = cfg.raw.get("stac_url", DEFAULT_STAC_URL)

        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix="modis_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.tile_size = int(cfg.raw.get("tile_size", tiling.DEFAULT_TILE_SIZE))

        self._stac_client = None

    def output_root(self, step: PipelineStep, *, namespace: str | None = None, agg: str | None = None) -> str:
        """Overrides the base default for PREPARE/GRID only -- MODIS's own
        PREPARE step (mosaic + reproject onto the canonical grid; was GRID
        before the FETCH/PREPARE rename) hardcodes `grid_id="ease6933"`
        regardless of any global grid config (docs/design/05-migration.md
        §1), and passes the literal `PipelineStep.GRID` through to
        `layout.output_root()` since the physical tier is still "GRID"
        (`grid/ease6933`), only MODIS's own declared step name changed. Also
        accepts the literal `PipelineStep.GRID` itself (not just PREPARE):
        `scripts/migrate_legacy_layout.py` calls `output_root(PipelineStep.GRID)`
        on every source regardless of whether GRID is still in that source's
        own `STEPS`.

        `agg` is accepted (so callers that pass `agg=` for every PREPARE call,
        e.g. `scripts/migrate_legacy_layout.py`, don't blow up) but unused:
        PREPARE is routed to the GRID tier here, so `layout.output_root()`
        never reaches the branch that requires `agg`.

        FETCH now uses the base class's default (`raw/<data_path>` --
        `layout.raw_root()`), same as every other FETCH-capable source; no
        longer a MODIS-only special case.
        """
        from src.data.sources import layout
        from src.data.sources.layout import EASE_GRID_ID

        if step in (PipelineStep.PREPARE, PipelineStep.GRID):
            return layout.output_root(
                self.ctx.data_root,
                self.cfg.data_path,
                PipelineStep.GRID,
                namespace=namespace,
                grid_id=EASE_GRID_ID,
            )
        return super().output_root(step, namespace=namespace, agg=agg)

    # ------------------------------------------------------------------
    # transfer_units -- per-(year, tile) files for FETCH, default for PREPARE
    # ------------------------------------------------------------------

    def transfer_units(self, step: PipelineStep) -> List[TransferUnit]:
        self._require_step(step)
        if step is not PipelineStep.FETCH:
            return super().transfer_units(step)

        stage1_root = self.output_root(PipelineStep.FETCH)
        # Relative to the LOCAL data root, not `remote_data_root` -- see
        # `DataSource.transfer_units()`'s docstring/comment (base.py) for why
        # relpath-ing a local path against a *remote* absolute path (often a
        # different machine/filesystem entirely, e.g. Windows local vs.
        # scicore POSIX) produces a meaningless, `../../..`-laden path
        # instead of "this unit's place under the remote tree". The
        # `.replace(os.sep, "/")` below is the same fix as base.py's: on
        # Windows, `os.path.relpath` emits backslash-separated output no
        # matter its input, but `remote_path` must be POSIX (the HPC target
        # is a remote Linux host).
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
                        remote_path=os.path.relpath(local_path, self.ctx.data_root).replace(os.sep, "/"),
                    )
                )
        return units

    # ------------------------------------------------------------------
    # plan()/execute() dispatch
    # ------------------------------------------------------------------

    def _plan(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        if step is PipelineStep.FETCH:
            return self._plan_fetch(selection)
        if step is PipelineStep.PREPARE:
            return self._discover_prepare(selection)
        raise AssertionError(f"unreachable: {step}")

    def _execute(self, target: StepTarget) -> bool:
        if target.step is PipelineStep.FETCH:
            return self._execute_fetch(target)
        if target.step is PipelineStep.PREPARE:
            return self._execute_prepare(target)
        raise AssertionError(f"unreachable: {target.step}")

    # -- FETCH ("annual": STAC streaming ingest + compositing) -------------

    def _plan_fetch(self, selection: TargetSelection) -> List[StepTarget]:
        from src.data.common.fetch.manifest import resolve_fetch_listing

        stage1_root = self.output_root(PipelineStep.FETCH)
        # `transfer_mode=auto` (default for MODIS -- ModisSource
        # .DEFAULT_TRANSFER_MODE above): each tile-year's
        # local .tif gets pushed to HPC right after FETCH and isn't kept
        # around indefinitely, so a bare local os.path.exists() would make
        # an already-pushed, locally-pruned tile look outstanding forever.
        # `from_remote` decides between `Completion.PRECOMPUTED` (checked
        # once here, against the HPC listing) and the original
        # `Completion.PATH_EXISTS` (checked later, per-target, against local
        # disk) -- `selection.local_only` (`data summary`'s deliberately
        # network-free targets) forces the latter regardless of transfer_mode.
        listing, from_remote = resolve_fetch_listing(self, stage1_root, allow_remote=not selection.local_only)

        targets = []
        for tile in self.tiles:
            years = self.years or (
                self.cfg.year_range and range(self.cfg.year_range[0], self.cfg.year_range[1] + 1)
            ) or []
            for year in years:
                if not selection.matches_year(year):
                    continue
                key = f"{year}/{tile}"
                if not selection.matches_key(key):
                    continue
                output_path = os.path.join(stage1_root, str(year), f"{tile}.tif")
                if from_remote:
                    completion = Completion.PRECOMPUTED
                    meta = {"year": year, "tile": tile, "complete": f"{year}/{tile}.tif" in listing}
                else:
                    completion = Completion.PATH_EXISTS
                    meta = {"year": year, "tile": tile}
                targets.append(
                    StepTarget(
                        source_id=self.cfg.source_id,
                        step=PipelineStep.FETCH,
                        key=key,
                        output_path=output_path,
                        completion=completion,
                        meta=meta,
                    )
                )
        return targets

    def _get_stac_client(self):
        """Lazily opened, reused across `_search_items()` calls within one
        run: `Client.open()` is an HTTP round trip to the STAC
        root/conformance document; `_execute_fetch()` runs once per (year,
        tile) `StepTarget`, so an uncached client would reopen the catalog
        thousands of times over a full run (e.g. ~20 years x ~300 land
        tiles) instead of once."""
        if self._stac_client is None:
            import planetary_computer
            import pystac_client

            self._stac_client = pystac_client.Client.open(self.stac_url, modifier=planetary_computer.sign_inplace)
        return self._stac_client

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

        # `properties.platform` vs the MOD/MYD id prefix -- checked directly
        # against 600 real STAC items (3 collections x 5 regions x 4 years,
        # 2026-08-09): zero disagreements (docs/design/06-open-questions.md
        # #8, now resolved). The two signals appear to always agree in
        # practice, so which one is "authoritative" is moot; this warning is
        # kept as a live tripwire in case that ever changes for some item,
        # not because a disagreement is expected.
        #
        # 2026-08-17: a real disagreement showed up -- a batch of items with
        # `platform` entirely empty (not wrong-platform, just missing), which
        # the original `platform_ok`-only filter silently dropped as if they
        # belonged to the other satellite. An empty `platform` carries no
        # information, so it can't outweigh an unambiguous id prefix: only
        # treat `platform` as authoritative when it's actually populated,
        # and fall back to the id prefix otherwise.
        id_prefix = "MYD" if self.platform == "aqua" else "MOD"
        filtered = []
        for item in items:
            platform_value = item.properties.get("platform")
            platform_ok = platform_value == self.platform
            id_ok = item.id.startswith(id_prefix)
            if platform_ok != id_ok:
                if platform_value:
                    # Populated and still wrong -- the real tripwire case.
                    logger.warning(
                        "STAC item %s: platform property (%s) and id prefix disagree", item.id, platform_value
                    )
                else:
                    # Empty is an expected, handled fallback (below), not an
                    # anomaly -- worth a record, not a WARNING-level alert.
                    logger.debug("STAC item %s: empty platform property, falling back to id prefix", item.id)
            include = id_ok if not platform_value else platform_ok
            if include:
                filtered.append(item)
        return filtered

    def _load_tile_year(self, items: list) -> Optional[xr.Dataset]:
        import odc.stac

        # odc.stac.load() (default kwargs, no `dtype=`/`groupby=` scale
        # processing requested) does NOT auto-apply STAC-declared
        # `raster:bands` scale/offset -- confirmed empirically against a
        # real modis-21A2-061 item (2026-08-09, odc-stac 0.5.3): the loaded
        # array stays `uint16` and its values match a raw rasterio read of
        # the same asset exactly (docs/design/06-open-questions.md #11, now
        # resolved). So the manual `raw * scale + offset` below is required,
        # not a double-application bug.
        assets = self.band_spec["assets"]
        bands = [spec["name"] for spec in assets.values()]
        # `crs=`/`resolution=` pinned explicitly: odc-stac otherwise
        # auto-guesses both from each item's STAC `proj` extension fields,
        # which some items lack entirely (observed 2026-08-17 -- items with
        # no `platform` property either, so likely a batch with generally
        # incomplete metadata) and then raises
        # `ValueError("Failed to auto-guess CRS/resolution.")` outright
        # instead of degrading gracefully. `crs=` alone isn't enough --
        # odc-stac still tries to auto-derive resolution from item metadata
        # when resolution is left unset, and hits the same error. The
        # sinusoidal grid and its 1km pixel size are fixed and already known
        # here (`_tile_bbox_4326`/`RESOLUTION_1KM_M`), so there's nothing to
        # guess.
        ds = odc.stac.load(
            items, bands=bands, crs=modis_util.SINUSOIDAL_PROJ4, resolution=modis_util.RESOLUTION_1KM_M,
            chunks={"time": 1, "x": 2400, "y": 2400}, resampling="nearest",
        )
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

    def _execute_fetch(self, target: StepTarget) -> bool:
        from src.data.common.raster.compositing import composite_annual_stats
        from src.data.sources.steps import is_complete

        status_dir = self.output_root(PipelineStep.FETCH)

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping %s/%s -- output exists: %s", target.meta["year"], target.meta["tile"], target.output_path)
            return True

        year, tile = target.meta["year"], target.meta["tile"]
        items = self._search_items(tile, year)
        if not items:
            logger.warning("No STAC items found for tile=%s year=%d", tile, year)
            manifest.record_failure(status_dir, target.key, "no STAC items found")
            return False

        ds = self._load_tile_year(items)
        if ds is None or "lst" not in ds.data_vars or "qc" not in ds.data_vars:
            logger.error("Failed to load required bands for tile=%s year=%d", tile, year)
            manifest.record_failure(status_dir, target.key, "failed to load required bands")
            return False

        valid_mask = modis_util.decode_qc_valid_mask(
            ds["qc"], self.qc_max_lst_error_k, product=self.product,
            lst=ds["lst"], min_lst_k=self.lst_min_k, max_lst_k=self.lst_max_k,
        )

        data_vars = {}
        if self.variant == "main":
            lst_stats = composite_annual_stats(
                ds["lst"], valid_mask, stats=MAIN_LST_STATS, thresholds=(self.cold_stress_k, self.heat_stress_k)
            )
            data_vars["lst_night_mean"] = lst_stats["mean"].squeeze("time", drop=True).astype("float32")
            data_vars["lst_night_median"] = lst_stats["median"].squeeze("time", drop=True).astype("float32")
            data_vars["lst_night_sd"] = lst_stats["std"].squeeze("time", drop=True).astype("float32")
            data_vars["lst_night_gt_heat"] = lst_stats["count_above"].squeeze("time", drop=True).astype("float32")
            data_vars["lst_night_lt_cold"] = lst_stats["count_below"].squeeze("time", drop=True).astype("float32")
            data_vars["valid_period_count_annual"] = lst_stats["valid_period_count"].squeeze("time", drop=True).astype("float32")
            data_vars["valid_month_count_annual"] = lst_stats["valid_month_count"].squeeze("time", drop=True).astype("float32")
        else:
            for key in EXTENDED_VARS:
                if key in ds.data_vars:
                    var_stats = composite_annual_stats(ds[key], valid_mask, stats=("mean",))
                    data_vars[key] = var_stats["mean"].squeeze("time", drop=True).astype("float32")

        out_ds = xr.Dataset(data_vars)
        out_ds = out_ds.rio.write_crs(modis_util.SINUSOIDAL_PROJ4)
        out_ds.attrs.update(
            {"source_type": "modis", "product": self.product, "tile": tile, "collection": self.collection_id, "platform": self.platform}
        )
        # Every data_var here is still dask-backed (odc.stac.load is lazy,
        # and QC-decode/composite_to_annual stay lazy too) -- compute the
        # whole Dataset in one shot before handing it to
        # `_write_annual_geotiff()`, which pulls `.values` per band *and*
        # per month for the two monthly vars (~20-30 separate accesses).
        # Dask's default scheduler shares no state across separate
        # `.compute()` calls, so without this each of those accesses
        # independently re-streamed the same COG bytes from Planetary
        # Computer and redid the QC-decode/resample chain from scratch.
        # One Dataset-level `.compute()` merges everything into a single
        # task graph, so the shared upstream work (raw read, QC decode,
        # valid_mask) runs once and every band/month below becomes a free
        # in-memory slice.
        out_ds = out_ds.compute()
        ok = self._write_annual_geotiff(out_ds, target.output_path)
        if ok:
            manifest.clear_failure(status_dir, target.key)
        else:
            manifest.record_failure(status_dir, target.key, "failed to write annual GeoTIFF")
        return ok

    @staticmethod
    def _write_annual_geotiff(ds: xr.Dataset, output_path: str) -> bool:
        """Expects `ds` already computed (in-memory, not dask-backed) --
        see the `.compute()` call at the end of `_execute_fetch()`. Each
        `.values` access below is a free slice of that in-memory array, not
        a fresh lazy evaluation."""
        import rasterio

        band_arrays: List[np.ndarray] = []
        band_names: List[str] = []

        for var, arr in ds.data_vars.items():
            for dim in ("time", "band"):
                if dim in arr.dims:
                    arr = arr.squeeze(dim, drop=True)
            band_arrays.append(np.asarray(arr.values, dtype="float32"))
            band_names.append(var)

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
                nodata=np.nan, compress="LERC_ZSTD", zstd_level=9, tiled=True,
                # ~32% smaller than DEFLATE+predictor3 at the same write
                # speed on this tile's sparse/NaN-heavy float32 bands
                # (benchmarked directly against a real tile) -- lossless
                # (default MAX_Z_ERROR=0), no predictor (LERC has its own
                # delta encoding, predictor is a DEFLATE/LZW-only concept).
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

    # -- PREPARE ("spatial": mosaic tiles, reproject onto canonical EPSG:6933) --
    # Named GRID until the rename that aligned MODIS's STEPS with every
    # other source's (FETCH, PREPARE) shape (module docstring) -- the
    # physical output tier is still "GRID" (see output_root() above), this
    # is MODIS's own declared step identity only.

    def _prepare_output_path(self) -> str:
        from src.data.sources import layout
        from src.data.sources.layout import EASE_GRID_ID

        # `family` alone determines the store's path (independent of
        # `data_path`) -- must include the variant or main/extended would
        # collide on the same zarr.
        variant_suffix = "" if self.variant == "main" else "_extended"
        return layout.grid_store_path(
            self.ctx.data_root,
            self.cfg.data_path,
            grid_id=EASE_GRID_ID,
            family=f"modis_lst_{self.product.lower()}{variant_suffix}",
            suffix="",  # cell_id-keyed parquet parts, not a Zarr store
        )

    def _discover_prepare(self, selection: TargetSelection) -> List[StepTarget]:
        """One PREPARE target for the whole source ("all"), matching GLASS's
        `_plan_prepare` -- `_execute_prepare` writes each available year into
        the shared multi-year zarr internally rather than one StepTarget per
        year sharing the same `output_path` (module docstring)."""
        stage1_root = self.output_root(PipelineStep.FETCH)
        output_path = self._prepare_output_path()
        years = self.years or (
            self.cfg.year_range and list(range(self.cfg.year_range[0], self.cfg.year_range[1] + 1))
        ) or []

        available_years = []
        for year in years:
            if not selection.matches_year(year) or not selection.matches_key(str(year)):
                continue
            year_dir = os.path.join(stage1_root, str(year))
            if not os.path.isdir(year_dir):
                logger.warning("No stage-1 output for year %d at %s", year, year_dir)
                continue
            if not any(f.endswith(".tif") for f in os.listdir(year_dir)):
                continue
            available_years.append(year)

        if not available_years:
            return []

        return [
            StepTarget(
                source_id=self.cfg.source_id,
                step=PipelineStep.PREPARE,
                key="all",
                output_path=output_path,
                completion=Completion.MARKER,
                meta={
                    "years": available_years,
                    # Already Kelvin (scale/offset applied at FETCH time)
                    # for the main variant's lst_night_mean, so no packed
                    # decode is needed here; extended has no LST band, so
                    # sanity-check emissivity's 0-1 range instead.
                    **(
                        verify.verification_meta(
                            self.cfg.raw, expected_vars=("lst_night_mean",), value_range=(150, 350)
                        )
                        if self.variant == "main"
                        else verify.verification_meta(
                            self.cfg.raw, expected_vars=("emis_31",), value_range=(0, 1)
                        )
                    ),
                },
            )
        ]

    @staticmethod
    def _read_annual_geotiff(path: str, year: int) -> xr.Dataset:
        """Stays dask-backed (`chunks=True`) -- the caller `xr.combine_by_coords`s
        every tile for a year into one lazy mosaic and only `.compute()`s
        after clipping to one output tile's bbox (`sel_bbox`), so this never
        eagerly reads a whole source GeoTIFF, let alone a whole year of them
        (docs/design/13-prepare-memory-parallelism.md)."""
        import rasterio
        import rioxarray as rxr

        da = rxr.open_rasterio(path, masked=True, chunks=True)
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

    def _execute_prepare(self, target: StepTarget) -> bool:
        from src.data.common.geobox import get_or_create_canonical_geobox
        from src.data.common.prepare.driver import run_tiled_prepare
        from src.data.common.raster.spatial import SpatialProcessor, sel_bbox

        # No top-level `is_complete(target)` short-circuit here: `target`'s
        # marker can already exist from a prior run while `target.meta["years"]`
        # (freshly discovered by `_discover_prepare`) has since grown with
        # newly-fetched years. `run_tiled_prepare` has its own finer-grained
        # per-unit status tracking (see its docstring) that cheaply skips
        # units already complete and only processes new ones, so it's
        # always safe and correct to call it rather than trusting the
        # coarse marker.
        stage1_root = self.output_root(PipelineStep.FETCH)
        years = target.meta["years"]

        # One year's *lazy* (dask-chunked, not yet materialized) mosaic
        # built at a time -- run_tiled_prepare walks units years-major, so
        # at most one year's mosaic graph is ever open here. Building it is
        # cheap regardless of tile count: xr.combine_by_coords over
        # chunks=True-opened tiles only assembles a task graph, no pixel
        # data is read. raw_getter clips that graph to one output tile's
        # bbox via sel_bbox() and only then .compute()s -- the actual
        # chunk reads that triggers are distributed across the (small)
        # Dask cluster and released immediately after; reprojection and the
        # parquet write then run serially in this process, against regular
        # (not worker) memory (docs/design/13-prepare-memory-parallelism.md,
        # same pattern as glass_avhrr's raw_getter).
        mosaic_cache: Dict[int, Optional[xr.Dataset]] = {}

        def year_mosaic(year: int) -> Optional[xr.Dataset]:
            if year not in mosaic_cache:
                mosaic_cache.clear()
                year_dir = os.path.join(stage1_root, str(year))
                tile_files = sorted(
                    os.path.join(year_dir, f) for f in os.listdir(year_dir) if f.endswith(".tif")
                ) if os.path.isdir(year_dir) else []
                if not tile_files:
                    logger.error("No stage-1 tiles for year %d at %s", year, year_dir)
                    mosaic_cache[year] = None
                    return None
                datasets = [ModisSource._read_annual_geotiff(f, year) for f in tile_files]
                mosaic = datasets[0] if len(datasets) == 1 else xr.combine_by_coords(
                    datasets, combine_attrs="override"
                )
                if mosaic.rio.crs is None:
                    mosaic = mosaic.rio.write_crs(modis_util.SINUSOIDAL_PROJ4)
                mosaic_cache[year] = mosaic
            return mosaic_cache[year]

        def raw_getter(tile, year: int) -> Optional[xr.Dataset]:
            mosaic = year_mosaic(year)
            if mosaic is None:
                return None
            bbox = tile.geobox.pad(32, 32).extent.to_crs(mosaic.rio.crs).boundingbox
            clipped = sel_bbox(mosaic, bbox, y_dim="y", x_dim="x")
            if clipped.sizes.get("x", 0) == 0 or clipped.sizes.get("y", 0) == 0:
                # This tile falls outside the mosaic's spatial coverage --
                # a legitimate tile state, not a fetch failure. Return a
                # NaN-filled dataset on tile.geobox instead of None so
                # run_tiled_prepare doesn't record it as a retryable
                # failure and permanently block mark_complete (same
                # convention as ecoregions/gadm/snl_mining's
                # _rasterize_tile).
                dim_y, dim_x = tile.geobox.dims
                return xr.Dataset(
                    {
                        var: ((dim_y, dim_x), np.full(tile.geobox.shape, np.nan, dtype=np.float32))
                        for var in mosaic.data_vars
                    }
                )
            return clipped.compute()

        try:
            with self._dask_client() as client:
                cache_path = os.path.join(self.ctx.data_root, "canonical_geobox.pkl")
                target_geobox = get_or_create_canonical_geobox(cache_path)

                spatial_processor = SpatialProcessor(
                    hpc_root=self.ctx.data_root, temp_dir=self.temp_dir, dask_client=client, target_geobox=target_geobox
                )

                with spatial_processor.setup_dask_config():
                    return run_tiled_prepare(
                        output_path=target.output_path,
                        years=years,
                        target_geobox=target_geobox,
                        processor=spatial_processor,
                        raw_getter=raw_getter,
                        tile_size=self.tile_size,
                        resampling=SPATIAL_RESAMPLING,
                        dst_nodata=float("nan"),
                        processing_version=self.PROCESSING_VERSION,
                        override=self.cfg.override,
                    )
        except Exception:
            logger.exception("Error in MODIS spatial processing for years %s.", years)
            return False

    # _dask_client: inherited from DataSource (src/data/sources/base.py).

registry.register(
    ModisSource.ID,
    __name__,
    ModisSource.__name__,
    ModisSource.STEPS,
    aliases=ModisSource.ALIASES,
)
registry.register(
    "modis_extended",
    __name__,
    ModisSource.__name__,
    ModisSource.STEPS,
)

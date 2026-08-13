"""MODIS LST (Planetary Computer STAC streaming): fetch + grid.

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
Instead it tracks each (year, tile) unit's local/remote state directly in the
generic `artifacts` table (`SourceLedger.ensure_artifact`/`set_local_state`),
giving `data summary`/retries the same per-unit visibility other sources
get from the crawl catalog, without pretending MODIS has one.

**Real bug fixed in the original PREPARE migration, not silently ported**:
the old `get_preprocessor_class()` only matched `--source` values `"modis"`/
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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.base import DataSource
from src.data.sources.modis import tiles as modis_util
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection, TransferUnit
from src.data.sources import verify

if TYPE_CHECKING:
    from src.data.common.ledger.store import ArtifactRow

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


class ModisSource(DataSource):
    ID = "modis"
    ALIASES = ("modis_lst", "modis_robustness_11a1")
    STEPS = (PipelineStep.FETCH, PipelineStep.GRID)

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
        # Explicit discrete-year override, e.g. modis_robustness_11a1's "3-5
        # years spanning early/mid/late mission" (docs/design/07-modis-ingest.md
        # §1) -- `year_range` alone can only express one contiguous span, not
        # a handful of non-adjacent years, so this is a separate config key
        # rather than overloading year_range's meaning.
        self.years = cfg.raw.get("years")

        self.qc_max_lst_error_k = float(cfg.raw.get("qc_max_lst_error_k", 2.0))
        self.stac_url = cfg.raw.get("stac_url", DEFAULT_STAC_URL)

        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix="modis_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)

        self._stac_client = None

    def output_root(self, step: PipelineStep, *, namespace: str | None = None) -> str:
        """Overrides the base default in two places:

        - GRID: the old MODISPreprocessor hardcodes `stage_2_ease6933` for
          its spatial output regardless of any global grid config -- the one
          deliberate MODIS-only ad hoc case docs/design/05-migration.md §1 /
          docs/design/09-integrated-pipeline.md §3 describe. Force
          `grid_id="ease6933"` here rather than deferring to `self.ctx.grid_id`
          (which defaults to legacy_4326).
        - FETCH: this step is a rename of what used to be called PREPARE
          (module docstring) -- the physical artifact tree
          (`processed/stage_1` legacy / `prepared/<data_path>` v2) is
          unchanged, it is still STAC-streamed-then-composited annual
          GeoTIFFs, not a bag of raw bytes under `layout.raw_root()`'s bare
          `<data_path>/raw` convention every crawler-based FETCH source uses.
          Reusing that convention here would silently orphan already-fetched
          local/HPC GeoTIFFs on this rename. `PipelineStep.PREPARE` below is
          a path-computation implementation detail only -- MODIS no longer
          declares that step (`STEPS`).
        """
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
        if step is PipelineStep.FETCH:
            return layout.output_root(
                self.ctx.data_root,
                self.cfg.data_path,
                PipelineStep.PREPARE,
                namespace=namespace if namespace is not None else self.cfg.namespace,
                grid_id=self.ctx.grid_id,
                layout=self.ctx.layout,
            )
        return super().output_root(step, namespace=namespace)

    # ------------------------------------------------------------------
    # transfer_units -- per-(year, tile) files for FETCH, default for GRID
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
        if step is PipelineStep.GRID:
            return self._plan_grid(selection)
        raise AssertionError(f"unreachable: {step}")

    def _discover(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        """Ground truth for `data reconcile`: FETCH is already a pure
        config cross-product (tile x year), not a crawl -- `_plan_fetch()`
        itself is the discovery, tracked directly in the ledger's `artifacts`
        table by `_execute_fetch()` (this class's own module docstring), not
        by `reconcile_step`. Only GRID gets a real live-crawl counterpart."""
        if step is PipelineStep.FETCH:
            return self._plan_fetch(selection)
        if step is PipelineStep.GRID:
            return self._discover_grid(selection)
        raise AssertionError(f"unreachable: {step}")

    def _execute(self, target: StepTarget) -> bool:
        if target.step is PipelineStep.FETCH:
            return self._execute_fetch(target)
        if target.step is PipelineStep.GRID:
            return self._execute_grid(target)
        raise AssertionError(f"unreachable: {target.step}")

    def _ledger_path(self) -> Optional[str]:
        from src.data.common.ledger.paths import ledger_path

        return ledger_path(self.ctx.local_index_dir, self.data_path)

    def _ledger_ensure_artifact(self, target: StepTarget) -> None:
        """Open a fresh, short-lived ledger connection, record `target` as a
        known artifact, close immediately.

        Deliberately NOT one connection cached across `_execute_fetch()`
        calls for the run's whole lifetime (this method used to route
        through such a cache, `_get_ledger()`) -- DuckDB allows only one
        read-write connection to a given `.duckdb` file at a time, across
        processes (`SourceLedger`'s module docstring), and a multi-hour FETCH
        run holding that connection open continuously starves any concurrent
        writer against the same file -- concretely, `data transfer
        --watch` running alongside a live FETCH, which is exactly the
        "second concurrent writer... needs its own connection strategy" case
        that docstring flags as unhandled. Each connection here touches the
        ledger for a few ms; the STAC search + odc.stac.load + compositing
        between calls (seconds to tens of seconds per tile-year) is real
        idle time a concurrent transfer pass can use the lock during instead.

        Opens via `open_with_retry()` (short exponential backoff on
        `duckdb.IOException`) rather than a bare `open()`: two processes each
        briefly holding the lock still occasionally collide mid-hold, and a
        multi-hour FETCH run should not abort entirely over one missed
        ledger write. If every retry is exhausted, this logs and moves on --
        losing one artifact-tracking row is not worse than losing the whole
        run; a later `data reconcile` (or simply re-running FETCH, which
        checks `is_complete()`/the file's own presence) recovers it.
        """
        path = self._ledger_path()
        if path is None:
            return
        import duckdb

        from src.data.common.ledger.store import SourceLedger

        try:
            with SourceLedger.open_with_retry(path, data_path=self.data_path) as ledger:
                ledger.ensure_artifact(PipelineStep.FETCH.value, target.key, local_path=target.output_path)
        except duckdb.IOException:
            logger.warning("Ledger busy after retries -- skipping artifact-tracking write for %s", target.key)

    def _ledger_set_state(self, key: str, state: str, *, size_bytes: Optional[int] = None) -> None:
        """Same short-lived-connection, retry-then-skip shape as
        `_ledger_ensure_artifact()`."""
        path = self._ledger_path()
        if path is None:
            return
        import duckdb

        from src.data.common.ledger.store import SourceLedger

        try:
            with SourceLedger.open_with_retry(path, data_path=self.data_path) as ledger:
                ledger.set_local_state(PipelineStep.FETCH.value, key, state, size_bytes=size_bytes)
        except duckdb.IOException:
            logger.warning("Ledger busy after retries -- skipping state update (%s) for %s", state, key)

    # -- FETCH ("annual": STAC streaming ingest + compositing) -------------

    def _plan_fetch(self, selection: TargetSelection) -> List[StepTarget]:
        stage1_root = self.output_root(PipelineStep.FETCH)
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
                targets.append(
                    StepTarget(
                        source_id=self.cfg.source_id,
                        step=PipelineStep.FETCH,
                        key=key,
                        output_path=os.path.join(stage1_root, str(year), f"{tile}.tif"),
                        completion=Completion.PATH_EXISTS,
                        # FETCH streams from Planetary Computer off-cluster
                        # (needs internet egress SLURM compute nodes may lack)
                        # and must be pushed to HPC before GRID's SLURM job
                        # can read it -- so "complete" here must mean
                        # HPC-verified, not just locally present (docs/design/10-fetch-ledger.md §6).
                        require_remote=True,
                        meta={"year": year, "tile": tile},
                    )
                )
        return targets

    def _get_stac_client(self):
        """Lazily opened, reused across `_search_items()` calls within one
        run -- unlike the ledger connections (`_ledger_ensure_artifact()`/
        `_ledger_set_state()` above, deliberately *not* cached this way, see
        their docstrings), caching this is safe: it's a plain HTTP client
        with no cross-process exclusive-lock semantics to starve a concurrent
        writer with. `Client.open()` is an HTTP round trip to the STAC
        root/conformance document;
        `_execute_fetch()` runs once per (year, tile) `StepTarget`, so an
        uncached client would reopen the catalog thousands of times over a
        full run (e.g. ~20 years x ~300 land tiles) instead of once."""
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

    def _execute_fetch(self, target: StepTarget) -> bool:
        from src.data.common.raster.compositing import composite_to_annual
        from src.data.sources.steps import is_complete

        self._ledger_ensure_artifact(target)

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping %s/%s -- output exists: %s", target.meta["year"], target.meta["tile"], target.output_path)
            return True

        year, tile = target.meta["year"], target.meta["tile"]
        items = self._search_items(tile, year)
        if not items:
            logger.warning("No STAC items found for tile=%s year=%d", tile, year)
            self._ledger_set_state(target.key, "failed")
            return False

        ds = self._load_tile_year(items)
        if ds is None or "lst" not in ds.data_vars or "qc" not in ds.data_vars:
            logger.error("Failed to load required bands for tile=%s year=%d", tile, year)
            self._ledger_set_state(target.key, "failed")
            return False

        valid_mask = modis_util.decode_qc_valid_mask(ds["qc"], self.qc_max_lst_error_k, product=self.product)
        annual_lst, monthly_lst, monthly_count, annual_count, annual_month_count = composite_to_annual(
            ds["lst"], valid_mask
        )

        data_vars = {
            "lst_night": annual_lst.squeeze("time", drop=True).astype("float32"),
            "lst_night_monthly": monthly_lst.astype("float32"),
            "valid_period_count_monthly": monthly_count.astype("float32"),
            "valid_period_count_annual": annual_count.squeeze("time", drop=True).astype("float32"),
            # Count of months that actually contributed to lst_night's own
            # averaging -- unlike valid_period_count_annual (a raw
            # observation-density count), this is the correctly-denominated
            # reliability diagnostic for a month-first-then-annual composite;
            # see composite_to_annual()'s docstring for why the two differ.
            "valid_month_count_annual": annual_month_count.squeeze("time", drop=True).astype("float32"),
        }
        for key in ("emis_29", "emis_31", "emis_32", "view_angle", "view_time"):
            if key in ds.data_vars:
                annual_var, *_ = composite_to_annual(ds[key], valid_mask)
                data_vars[key] = annual_var.squeeze("time", drop=True).astype("float32")

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
        size_bytes = os.path.getsize(target.output_path) if ok and os.path.exists(target.output_path) else None
        self._ledger_set_state(target.key, "complete" if ok else "failed", size_bytes=size_bytes)
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

        for var in (
            "lst_night", "valid_period_count_annual", "valid_month_count_annual",
            "emis_29", "emis_31", "emis_32", "view_angle", "view_time",
        ):
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

    def _grid_output_path(self) -> str:
        from src.data.sources import layout
        from src.data.sources.layout import EASE_GRID_ID

        return layout.grid_store_path(
            self.ctx.data_root,
            self.cfg.data_path,
            f"modis_{self.product}_timeseries_reprojected.zarr",
            grid_id=EASE_GRID_ID,
            layout=self.ctx.layout,
            v2_family=f"modis_lst_{self.product.lower()}",
        )

    def _plan_grid(self, selection: TargetSelection) -> List[StepTarget]:
        """Ledger-backed fast path: a year's GRID target's `inputs` (per-tile
        GeoTIFF paths, up to ~300/year) are re-derived from FETCH's
        `local_complete_units(key_prefix=f"{year}/")` rather than persisted
        -- the FETCH key format (`f"{year}/{tile}"`, `_plan_fetch()` above)
        makes the prefix scope exact. Falls back to `_discover_grid()` for a
        year with no ledger GRID row yet."""
        output_path = self._grid_output_path()
        years_all = self.years or (
            self.cfg.year_range and list(range(self.cfg.year_range[0], self.cfg.year_range[1] + 1))
        ) or []

        def build_target(row: "ArtifactRow", ledger: Any) -> Optional[StepTarget]:
            if not row.unit_id.isdigit():
                return None
            year = int(row.unit_id)
            if not selection.matches_year(year):
                return None
            tile_files = sorted(path for _, path in ledger.local_complete_units("fetch", key_prefix=f"{year}/"))
            if not tile_files:
                return None
            return StepTarget(
                source_id=self.cfg.source_id,
                step=PipelineStep.GRID,
                key=row.unit_id,
                output_path=output_path,
                inputs=tuple(tile_files),
                # Quirk preserved, not invented -- see module docstring:
                # the old spatial stage never checked override/existence.
                completion=Completion.NEVER,
                meta={
                    "year": year,
                    "years_all": years_all,
                    **verify.verification_meta(
                        self.cfg.raw, expected_vars=("lst_night",), value_range=(150, 350)
                    ),
                },
            )

        targets = self._plan_from_ledger(PipelineStep.GRID, selection, build_target)
        if targets is not None:
            return targets
        logger.warning(
            "No ledger for source='%s' step='grid' -- falling back to live discovery; "
            "run `data reconcile --source %s --step grid` for faster planning.",
            self.cfg.source_id, self.cfg.source_id,
        )
        return self._discover_grid(selection)

    def _discover_grid(self, selection: TargetSelection) -> List[StepTarget]:
        stage1_root = self.output_root(PipelineStep.FETCH)
        output_path = self._grid_output_path()
        years = self.years or (
            self.cfg.year_range and list(range(self.cfg.year_range[0], self.cfg.year_range[1] + 1))
        ) or []

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
                    meta={
                        "year": year,
                        "years_all": years,
                        # "lst_night" is always written by _execute_fetch's
                        # data_vars dict regardless of product; already Kelvin
                        # (scale/offset applied at FETCH time), so no packed
                        # decode is needed here.
                        **verify.verification_meta(
                            self.cfg.raw, expected_vars=("lst_night",), value_range=(150, 350)
                        ),
                    },
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

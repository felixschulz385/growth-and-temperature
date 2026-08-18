"""GLASS-MODIS LST + Land Air Temperature: fetch + prepare.

docs/design/12-glass-modis-rebuild.md: rebuilds GLASS-MODIS on the same
shape as raw MODIS (`src/data/sources/modis/source.py`) instead of the old
per-day-HDF-crawl-then-naive-annual-resample pipeline the former
`GlassSource` (deleted, split into this module + `glass/avhrr.py`) used:

- FETCH targets are `(tile, year)`, not `(year, day[, tile])`. Each target
  downloads that tile-year's daily source files into scratch, combines them
  into one annual multi-band GeoTIFF (LERC_ZSTD-compressed, `raw/` path
  convention) via the shared month-first `composite_annual_stats()`
  compositor, and is auto-pushed to HPC per file (existing
  `transfer_mode=auto` machinery).
- PREPARE mosaics each year's fetched tiles once, then reprojects onto the
  canonical grid tile-by-tile via the shared `run_tiled_prepare` driver
  (`src/data/common/prepare/driver.py`), exactly like `ModisSource.
  _execute_prepare` -- a single `key="all"` target, output is `cell_id`-keyed
  parquet, not a Zarr store.

Registered twice, mirroring `ModisSource`'s `main`/`extended` dual
registration: `glass_modis` (variant "lst", `GLASS06A01`, single daily band
`LST`) and `glass_ta_modis` (variant "ta", `GLASS18A01`, three daily bands
`Ta_min`/`Ta_mean`/`Ta_max` -- the file already provides daily min/mean/max
directly, no derivation needed). One class, `self.variant` derived from
`cfg.source_id`, same pattern as `ModisSource.variant`/the old
`GlassSource.data_source_kind`.

§1's QA-band investigation (against two real downloaded tiles) found neither
product has a QC/QA subdataset -- validity is "not-fill AND within a
physically-plausible value range", read from each file's own
`_FillValue`/`scale_factor`/`valid_range` attrs via
`rxr.open_rasterio(path, masked=True)` (which auto-applies them), plus a
configurable sanity range (`value_min`/`value_max`) mirroring
`ModisSource.lst_min_k`/`lst_max_k`. Ta's physical unit is confirmed
**Kelvin** (`units: 'K'`, `scale_factor: 0.01`, `valid_range: [16000, 37000]`
raw -> 160-370 K physical, `long_name` `Ta_min`/`Ta_mean`/`Ta_max`), checked
against a real downloaded `GLASS18A01` granule via `pyhdf` (2026-08-17).
`heat_stress_k`/`cold_stress_k` remain tuning placeholders (35C/0C in
Kelvin) -- the unit was the open question, not these threshold values,
which are still illustrative defaults (same posture as `modis:`'s own
`qc_max_lst_error_k: 2.0  # pending QC bit-layout verification`).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr

from src.data.common import tiling
from src.data.common.fetch import manifest
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.glass.avhrr import daterange_doy
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection, TransferUnit, is_complete
from src.data.sources import verify

logger = logging.getLogger(__name__)

#: Per-variant constants (docs/design/12-glass-modis-rebuild.md §1/§4):
#: `path_prefix` keys this variant's whole raw/prepared tree (mirroring
#: `GlassSource.path_prefix`, replacing `cfg.data_path`); `band_names` is
#: every daily SDS this variant's HDF granules carry; `mean_band`/
#: `min_band`/`max_band` say which of those feeds the §2 stats helper's
#: `mean_da`/`min_da`/`max_da` (LST has only one band, so all three alias
#: it; Ta already has separate daily min/mean/max bands).
_VARIANT_SPECS: Dict[str, Dict[str, Any]] = {
    "lst": {
        "path_prefix": "glass/LST/MODIS/Daily/1KM/",
        "band_names": ("LST",),
        "mean_band": "LST",
        "min_band": "LST",
        "max_band": "LST",
    },
    "ta": {
        "path_prefix": "glass/Ta/MODIS/",
        "band_names": ("Ta_min", "Ta_mean", "Ta_max"),
        "mean_band": "Ta_mean",
        "min_band": "Ta_min",
        "max_band": "Ta_max",
    },
}


#: The real, physical per-band order GLASS-Ta HDF4-EOS granules carry,
#: confirmed against a downloaded `GLASS18A01` granule via `pyhdf`
#: (2026-08-17) -- NOT `band_names`' own min/mean/max order. Used only as
#: the last-resort positional fallback in `_open_hdf_bands` below, when
#: neither a real `long_name` match nor a name match is available.
_TA_PHYSICAL_BAND_ORDER: Tuple[str, ...] = ("Ta_max", "Ta_mean", "Ta_min")


def _match_band_names(
    path: str, band_names: Tuple[str, ...], long_names: List[Optional[str]], names: List[Optional[str]]
) -> Optional[List[str]]:
    """Maps each of `len(long_names)` physical bands (in file order) to one
    of `band_names`, by `long_name` first, then `.name`, then -- only if
    neither is available/complete for every band -- the known physical
    order (`_TA_PHYSICAL_BAND_ORDER`, only meaningful when `band_names`
    equals that set). Returns `None` (caller warns + skips the file) if no
    strategy produces a complete, unambiguous mapping -- silently guessing
    wrong would mislabel a band undetected."""
    n = len(long_names)
    for candidates, label in ((long_names, "long_name"), (names, "name")):
        if all(c in band_names for c in candidates) and len(set(candidates)) == n:
            return list(candidates)  # type: ignore[arg-type]
    if n == len(_TA_PHYSICAL_BAND_ORDER) and set(band_names) == set(_TA_PHYSICAL_BAND_ORDER):
        logger.warning(
            "GLASS HDF %s: bands matched by known physical order only "
            "(long_name=%s, name=%s didn't give a complete match against "
            "band_names=%s) -- see _open_hdf_bands docstring",
            path, long_names, names, band_names,
        )
        return list(_TA_PHYSICAL_BAND_ORDER)
    return None


def _open_hdf_bands(path: str, band_names: Tuple[str, ...]) -> Dict[str, xr.DataArray]:
    """Open one GLASS daily HDF4 granule, returning `{band_name: DataArray}`.

    A single-SDS file (LST) opens via `rxr.open_rasterio` as one plain
    DataArray directly -- confirmed by today's (pre-rebuild) `GlassSource`
    code, which reads a GLASS-MODIS LST HDF this same way with no
    variable-name lookup at all.

    A multi-SDS file (Ta) does NOT reliably open as a list of per-
    subdataset DataArrays, contrary to this function's first version:
    confirmed in production (2026-08-17 real FETCH run logs) that GDAL's
    HDF4-EOS driver groups same-shape/same-grid SDS (Ta's `Ta_min`/
    `Ta_mean`/`Ta_max`, all `[1200, 1200]` on one `GLASS18A01` grid) into
    ONE multi-band DataArray instead -- `rxr.open_rasterio` returned a
    single (non-list) object with a `band` dim of length 3, which the
    previous list-only code silently treated as "wrong band count for a
    single-SDS file" and returned `{}` for, dropping every day's Ta data
    without raising (`_build_annual_geotiff`'s "Missing expected band(s)
    ... found []" warning, then "No usable daily files" once every day in
    the target failed the same way). Handles both shapes now: a genuine
    list of per-subdataset DataArrays (kept for whatever HDF4 files/GDAL
    versions DO split subdatasets -- not reproducible locally, no working
    HDF4 GDAL plugin here, module docstring), or one DataArray/Dataset
    whose `band` dim (or `data_vars`) has `len(band_names)` entries.

    Per-band identity is resolved via `_match_band_names()`: each SDS's own
    `long_name` attribute is confirmed (via `pyhdf` against a real
    granule) to equal `band_names` verbatim (`Ta_min`/`Ta_mean`/`Ta_max`)
    and is tried first; `.name`/`long_name` values GDAL may expose
    per-band on the grouped case are tried the same way; the physical file
    order (`_TA_PHYSICAL_BAND_ORDER`, confirmed via the same `pyhdf`
    inspection to be Max/Mean/Min -- NOT `band_names`' own order) is the
    last-resort fallback, logged when used.

    Also confirmed from the same real files: LST's fill attribute is the
    standard `_FillValue` (GDAL/rioxarray auto-masks it via `masked=True`),
    but Ta's is a non-standard lowercase `fillvalue` (string-typed `"0"`),
    which `masked=True` may not auto-detect as NoData. Not a correctness
    problem in practice: the caller's `value_min`/`value_max` sanity range
    (confirmed 160-370 K, comfortably above scaled fill=0.0) filters out
    unmasked fill pixels either way -- but if `masked=True` ever silently
    stops being the only defense here, revisit this."""
    opened = rxr.open_rasterio(path, masked=True)

    if isinstance(opened, list):
        sub_arrays = [sd.squeeze("band", drop=True) if "band" in sd.dims else sd for sd in opened]
        long_names = [str(sd.attrs["long_name"]) if sd.attrs.get("long_name") else None for sd in opened]
        names = [str(sd.name) if sd.name else None for sd in opened]
    elif isinstance(opened, xr.Dataset):
        var_names = list(opened.data_vars)
        if len(var_names) != len(band_names):
            return {} if len(band_names) > 1 else {band_names[0]: opened[var_names[0]]}
        sub_arrays = [opened[v] for v in var_names]
        long_names = [str(opened[v].attrs["long_name"]) if opened[v].attrs.get("long_name") else None for v in var_names]
        names = [str(v) for v in var_names]
    elif "band" in opened.dims and opened.sizes["band"] > 1:
        if opened.sizes["band"] != len(band_names):
            return {}
        sub_arrays = [opened.isel(band=i, drop=True) for i in range(opened.sizes["band"])]
        raw_long_name = opened.attrs.get("long_name")
        long_name_seq = list(raw_long_name) if isinstance(raw_long_name, (list, tuple)) else None
        long_names = [str(v) if v else None for v in long_name_seq] if long_name_seq else [None] * len(sub_arrays)
        names = [None] * len(sub_arrays)
    else:
        # Single-band file (LST): whatever GDAL/rioxarray returned IS the
        # one band this variant expects -- no name-matching needed.
        arr = opened.squeeze("band", drop=True) if "band" in opened.dims else opened
        return {band_names[0]: arr} if len(band_names) == 1 else {}

    matched = _match_band_names(path, band_names, long_names, names)
    if matched is None:
        logger.warning(
            "GLASS HDF %s: could not confidently match %d bands to %s "
            "(long_name=%s, name=%s) -- skipping this file",
            path, len(sub_arrays), band_names, long_names, names,
        )
        return {}
    return dict(zip(matched, sub_arrays))


def _composite_glass_annual_stats(
    mean_da: xr.DataArray,
    valid_mask: xr.DataArray,
    *,
    min_da: Optional[xr.DataArray] = None,
    max_da: Optional[xr.DataArray] = None,
    thresholds: Tuple[float, float],
) -> Dict[str, xr.DataArray]:
    """The shared 8-band annual-stats helper both GLASS-MODIS variants call
    (docs/design/12-glass-modis-rebuild.md §2): `mean`/`std`/
    `valid_period_count`/`valid_month_count`/`count_above`/`count_below` via
    the shared month-first `composite_annual_stats()` (fixes the seasonal-
    bias bug the old naive `resample("1YE").mean()` had); `max`/`min`
    computed **directly from daily data**, not month-weighted -- extremes
    shouldn't be diluted by month-averaging.

    `min_da`/`max_da` default to `mean_da` (LST: only one band exists).
    `thresholds` is `(cold_stress_k, heat_stress_k)`, same order
    `composite_annual_stats()` and `ModisSource` both use.

    The same `valid_mask` (built from `mean_da` by the caller) is applied to
    `min_da`/`max_da` too before taking their direct max/min -- a pixel with
    an invalid *mean* reading for a day is not a trustworthy source for that
    day's extreme either.
    """
    from src.data.common.raster.compositing import composite_annual_stats

    if min_da is None:
        min_da = mean_da
    if max_da is None:
        max_da = mean_da

    stats = composite_annual_stats(
        mean_da,
        valid_mask,
        stats=("mean", "std", "valid_period_count", "valid_month_count", "count_above", "count_below"),
        thresholds=thresholds,
    )
    stats["max"] = max_da.where(valid_mask).resample(time="1YE").max()
    stats["min"] = min_da.where(valid_mask).resample(time="1YE").min()
    return stats


class GlassModisSource(DataSource):
    """Registered twice (`registry.register()` calls at module end,
    mirroring `ModisSource`'s `main`/`extended` dual-registration):
    `glass_modis` (variant "lst") and `glass_ta_modis` (variant "ta"). One
    class, `self.variant` derived from `cfg.source_id`; each writes its own
    raw/prepared tree (`self.path_prefix` includes the variant) so they
    never collide."""

    ID = "glass_modis"
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE)

    DATA_SOURCE_NAME = "glass"

    #: bump to force a full reprocess (`run_tiled_prepare`'s `processing_version`)
    PROCESSING_VERSION = "1-tiled"

    #: 8-band shape every GLASS-MODIS variant's FETCH output carries (§2).
    STAT_VARS = ("mean", "std", "max", "min", "count_above", "count_below", "valid_period_count", "valid_month_count")
    #: "std"/the two count vars aren't on the same absolute physical scale as
    #: mean/max/min (a spread, and day-counts respectively) -- excluded from
    #: the GRID-verification range check, mirroring `GlassSource._RANGE_VARS`.
    RANGE_VARS = ("mean", "max", "min")

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        # Which variant: derived from the registered id ("glass_modis" /
        # "glass_ta_modis"), mirroring `ModisSource.variant`/the old
        # `GlassSource.data_source_kind`.
        self.variant = "ta" if "glass_ta_modis" in cfg.source_id.lower() else "lst"
        spec = _VARIANT_SPECS[self.variant]
        self.path_prefix: str = spec["path_prefix"]
        self.band_names: Tuple[str, ...] = spec["band_names"]
        self.mean_band: str = spec["mean_band"]
        self.min_band: str = spec["min_band"]
        self.max_band: str = spec["max_band"]

        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path=self.path_prefix.rstrip("/"))
        super().__init__(ctx, cfg)

        self.base_url: Optional[str] = cfg.raw.get("base_url")
        if not self.base_url:
            raise ValueError("'base_url' is required.")

        day_range = cfg.raw.get("day_range")
        if not day_range or "start" not in day_range or "end" not in day_range:
            raise ValueError("'day_range' with 'start'/'end' [year, day] pairs is required.")
        self.day_range_start: Tuple[int, int] = tuple(day_range["start"])
        self.day_range_end: Tuple[int, int] = tuple(day_range["end"])

        from src.data.sources.modis import tiles as modis_util

        lat_clip_deg = float(cfg.raw.get("lat_clip_deg", 60.0))
        land_tiles = cfg.raw.get("land_tiles")
        land_tiles_set = set(land_tiles) if land_tiles else None
        self.tiles: List[str] = cfg.raw.get("tiles") or modis_util.get_modis_sinusoidal_tiles(
            lat_clip_deg, land_tiles=land_tiles_set
        )

        # Physical-plausibility sanity range for the fill/QC-free valid mask
        # (doc §1), mirroring `ModisSource.lst_min_k`/`lst_max_k`. Kelvin
        # defaults for LST are the existing, long-used `GlassSource.
        # _LST_VALUE_RANGE`; real configs set Ta's own confirmed 160-370 K
        # range explicitly (orchestration/configs/data.yaml) rather than
        # relying on this fallback, which stays LST-shaped.
        self.value_min = float(cfg.raw.get("value_min", 150.0))
        self.value_max = float(cfg.raw.get("value_max", 350.0))
        # count_above/count_below thresholds (§2): GLASS LST/Ta are daytime/
        # blended products (unlike ModisSource's night-only LST_Night_1KM),
        # so its night defaults don't apply as-is -- distinct daytime
        # placeholder defaults, not verified figures (module docstring).
        self.heat_stress_k = float(cfg.raw.get("heat_stress_k", 308.15))
        self.cold_stress_k = float(cfg.raw.get("cold_stress_k", 273.15))

        # One tile-year FETCH target downloads up to ~365 daily HDFs; doing
        # that sequentially (one `requests.get` at a time) was the pre-
        # rebuild `GlassSource`'s actual bottleneck -- the pre-integrated-
        # pipeline `GlassLSTDataSource.download_async` concurrent-download
        # behaviour the user remembers from "the old fetch suite" predates
        # even that, and got factored into `src.data.common.fetch.http
        # .download_with_retries` (its own docstring: "was duplicated...
        # across ... glass/source.py"), just never wired back up after the
        # StepTarget rebuild. Reusing it here, same conservative
        # `limit_per_host=2`/300s timeout that module's docstring documents
        # as glass's own historical tuning (a shared multi-user archive,
        # not a CDN). Default of 5 matches `driver.run_fetch`'s own default.
        self.max_concurrent_downloads = int(cfg.raw.get("max_concurrent_downloads", 5))

        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix=f"glass_modis_{self.variant}_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.tile_size = int(cfg.raw.get("tile_size", tiling.DEFAULT_TILE_SIZE))

        # In-run-only memoization of one (year, day) directory listing across
        # every sibling tile sharing that day, for the lifetime of this
        # source instance (one FETCH run) -- same reasoning as the old
        # GlassSource/new GlassAvhrrSource (docs/design/11-glass-static-fetch.md §4.3).
        self._listing_cache: Dict[Tuple[int, int], List[Tuple[str, str]]] = {}

    def output_root(self, step: PipelineStep, *, namespace: str | None = None, agg: str | None = None) -> str:
        """Same shape as `ModisSource.output_root` (`modis/source.py:193-
        222`): FETCH uses the base-class default; PREPARE/GRID both route
        through the physical GRID tier (there's no separate "annual stats
        zarr" stage anymore -- PREPARE mosaics+reprojects straight from
        FETCH's per-tile-year GeoTIFFs, like `ModisSource`), keyed off
        `self.path_prefix` rather than `cfg.data_path` -- matching today's
        `GlassSource.output_root`.

        `agg` is accepted (so callers like `scripts/migrate_legacy_layout.py`
        that pass `agg=` for every PREPARE call don't blow up) but unused:
        since PREPARE is routed to the GRID tier here, `layout.output_root()`
        never reaches the branch that requires `agg`.

        Unlike `ModisSource`, this does NOT hardcode `grid_id=EASE_GRID_ID`:
        the old `GlassSource` always honored `ctx.grid_id` (a pinned
        regression test covers this, `test_execute_prepare_threads_ctx_
        grid_id_into_target_geobox`), and nothing in the rebuild doc asks to
        drop that flexibility -- so `self.ctx.grid_id` is threaded through
        here and into `_execute_prepare`'s `get_target_geobox()` call below,
        the locally-consistent choice given the ambiguity.
        """
        if step in (PipelineStep.PREPARE, PipelineStep.GRID):
            return layout.output_root(
                self.ctx.data_root,
                self.path_prefix,
                PipelineStep.GRID,
                namespace=namespace,
                grid_id=self.ctx.grid_id,
            )
        return layout.output_root(
            self.ctx.data_root,
            self.path_prefix,
            step,
            namespace=namespace,
            grid_id=self.ctx.grid_id,
        )

    def download(self, file_url: str, output_path: str, session: Any = None) -> None:
        import requests

        s = session or requests.Session()
        r = s.get(file_url, stream=True)
        r.raise_for_status()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # ------------------------------------------------------------------
    # transfer_units -- per-(year, tile) files for FETCH, default for PREPARE
    # (copied from ModisSource.transfer_units() verbatim, doc §4)
    # ------------------------------------------------------------------

    def transfer_units(self, step: PipelineStep) -> List[TransferUnit]:
        self._require_step(step)
        if step is not PipelineStep.FETCH:
            return super().transfer_units(step)

        stage1_root = self.output_root(PipelineStep.FETCH)
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

    # -- FETCH: one (tile, year) target per land tile x year -- doc §4 ------

    def _plan_fetch(self, selection: TargetSelection) -> List[StepTarget]:
        """`(tile, year)` target granularity, `key=f"{year}/{tile}"`,
        deterministic `output_path` -- no `_last_fetch_output_path` hack
        needed (same simplification `ModisSource` already has). Uses
        `resolve_fetch_listing()`/`Completion.PRECOMPUTED` vs `PATH_EXISTS`
        exactly like `ModisSource._plan_fetch` (`modis/source.py:284-328`)."""
        from src.data.common.fetch.manifest import resolve_fetch_listing

        raw_root = self.output_root(PipelineStep.FETCH)
        listing, from_remote = resolve_fetch_listing(self, raw_root, allow_remote=not selection.local_only)

        targets = []
        for tile in self.tiles:
            for year in range(self.day_range_start[0], self.day_range_end[0] + 1):
                if not selection.matches_year(year):
                    continue
                key = f"{year}/{tile}"
                if not selection.matches_key(key):
                    continue
                output_path = os.path.join(raw_root, str(year), f"{tile}.tif")
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

    def _listing_url(self, year: int, day: int) -> str:
        """Day directory (`<year>/<day>/`), same as MODIS's day-subdirectory
        remote layout under the old `GlassSource`."""
        return f"{self.base_url}{year}/{day:03d}/"

    @staticmethod
    def _list_single_directory(url: str) -> List[Tuple[str, str]]:
        """Non-recursive: one GET, parse one directory listing page's links.
        Filters to `.hdf` exactly (doc §1): Ta's directory listing has
        multiple sidecar files per tile/day sharing the same
        `A{year}{day}.{tile}.` token (`.hdf`, `.hdf.xml`, three `.jpg`
        previews) -- without this filter, `_match_in_listing` could
        nondeterministically match a preview image instead of the data
        file."""
        import requests
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin

        res = requests.get(url)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        results = []
        for link in soup.find_all("a"):
            href = link.get("href")
            if not href or href in ("../", "./"):
                continue
            if not href.endswith(".hdf"):
                continue
            results.append((href, urljoin(url, href)))
        return results

    def _listing_for(self, year: int, day: int) -> List[Tuple[str, str]]:
        """Memoized per-`(year, day)` -- one real GET per day, shared across
        every sibling tile target that scope resolves, for the lifetime of
        this source instance (one FETCH run)."""
        cache_key = (year, day)
        if cache_key not in self._listing_cache:
            self._listing_cache[cache_key] = self._list_single_directory(self._listing_url(year, day))
        return self._listing_cache[cache_key]

    @staticmethod
    def _match_in_listing(listing: List[Tuple[str, str]], year: int, day: int, tile: str) -> Optional[Tuple[str, str]]:
        """The one listing entry matching this target's `(year, day, tile)`
        -- filenames always embed `A{year}{day:03d}.{tile}.` verbatim, e.g.
        `GLASS06A01.V01.A2000055.h08v05.2022021.hdf`."""
        token = f"A{year}{day:03d}"
        needle = f".{token}.{tile}."
        for href, url in listing:
            if needle in href:
                return href, url
        return None

    def _resolve_day_urls(self, target: StepTarget, status_dir: str) -> Optional[List[Tuple[int, str, str]]]:
        """Sequential first pass: resolve every day in the target's year to
        a `(day, url, dest_path)` triple via `_listing_for`'s memoized
        listing (shared across every sibling tile target for that day, so
        keeping this sequential doesn't cost extra real GETs beyond the
        first tile processed for each day this run). Returns `None` on a
        transient listing error (already recorded); a day genuinely absent
        from the remote (sensor gap, 404'd day directory) is silently
        skipped, not a failure."""
        import requests

        year, tile = target.meta["year"], target.meta["tile"]
        resolved: List[Tuple[int, str, str]] = []
        for y, day in daterange_doy(self.day_range_start, self.day_range_end):
            if y != year:
                continue
            try:
                listing = self._listing_for(y, day)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    continue  # day directory doesn't exist -- real gap
                manifest.record_failure(status_dir, target.key, f"listing fetch failed for {y}/{day:03d}: {exc}")
                return None
            except requests.RequestException as exc:
                manifest.record_failure(status_dir, target.key, f"listing fetch failed for {y}/{day:03d}: {exc}")
                return None

            match = self._match_in_listing(listing, y, day, tile)
            if match is None:
                continue  # genuinely absent for this tile/day -- gap

            href, url = match
            dest = os.path.join(self.temp_dir, f"{tile}_{y}{day:03d}_{href}")
            resolved.append((day, url, dest))
        return resolved

    async def download_async(self, url: str, dest: str, session: Any) -> None:
        """The async counterpart to `self.download()` -- a separate,
        overridable/mockable method (rather than inlining
        `download_with_retries` into `_download_days_async` directly) so
        tests can monkeypatch the network boundary the same way they
        already monkeypatch the sync `download()`."""
        from src.data.common.fetch.http import download_with_retries

        await download_with_retries(session, url, dest)

    async def _download_days_async(
        self, items: List[Tuple[int, str, str]]
    ) -> List[Tuple[int, str, Optional[str]]]:
        """Concurrent download of every resolved `(day, url, dest)` triple,
        bounded by `self.max_concurrent_downloads` -- same conservative
        connector tuning the pre-rebuild `GlassLSTDataSource.download_async`
        used (module docstring). Returns `(day, dest, error_or_None)` per
        item; a failed item doesn't cancel the others, so one flaky day
        doesn't waste every other already-in-flight download for this
        target."""
        import aiohttp

        semaphore = asyncio.Semaphore(self.max_concurrent_downloads)

        async def _one(day: int, url: str, dest: str, session: aiohttp.ClientSession) -> Tuple[int, str, Optional[str]]:
            async with semaphore:
                try:
                    await self.download_async(url, dest, session)
                    return day, dest, None
                except Exception as exc:
                    return day, dest, str(exc)

        connector = aiohttp.TCPConnector(limit=20, limit_per_host=2)
        timeout = aiohttp.ClientTimeout(total=300, connect=30)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [_one(day, url, dest, session) for day, url, dest in items]
            return list(await asyncio.gather(*tasks))

    def _execute_fetch(self, target: StepTarget) -> bool:
        """For the target's `(tile, year)`: resolve every day's remote URL
        (sequential, memoized listing), download all resolved days
        concurrently (`_download_days_async`), combine into the 8 annual
        stat bands, and write one LERC_ZSTD GeoTIFF."""
        status_dir = self.output_root(PipelineStep.FETCH)
        if not self.cfg.override and is_complete(target):
            return True

        year = target.meta["year"]
        resolved = self._resolve_day_urls(target, status_dir)
        if resolved is None:
            return False  # transient listing error, already recorded

        if not resolved:
            manifest.record_failure(status_dir, target.key, "no daily files found for this tile/year", permanent=True)
            return False

        # All `resolved` dest paths, not just successful ones, must be
        # cleaned up below -- a partially-failed batch still leaves whatever
        # concurrent downloads DID succeed sitting in scratch otherwise.
        all_dest_paths = [dest for _, _, dest in resolved]
        try:
            results = asyncio.run(self._download_days_async(resolved))
            failed = [(day, error) for day, _, error in results if error is not None]
            if failed:
                day, error = failed[0]
                manifest.record_failure(
                    status_dir, target.key,
                    f"download failed for {year}/{day:03d}: {error}"
                    + (f" (+{len(failed) - 1} more)" if len(failed) > 1 else ""),
                )
                return False
            daily_files = [(day, dest) for day, dest, _ in results]

            ok = self._build_annual_geotiff(daily_files, year, target.output_path)
            if ok:
                manifest.clear_failure(status_dir, target.key)
            else:
                manifest.record_failure(status_dir, target.key, "failed to write annual GeoTIFF")
            return ok
        finally:
            # Scratch daily downloads, not the FETCH deliverable -- delete
            # once the tile-year tiff is written (or the attempt failed and
            # will be retried from scratch anyway).
            for path in all_dest_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _build_annual_geotiff(self, daily_files: List[Tuple[int, str]], year: int, output_path: str) -> bool:
        daily_files = sorted(daily_files, key=lambda x: x[0])

        per_band_arrays: Dict[str, list] = {name: [] for name in self.band_names}
        valid_days: List[int] = []
        for day, path in daily_files:
            try:
                bands = _open_hdf_bands(path, self.band_names)
            except Exception:
                logger.warning("Failed to open %s", path, exc_info=True)
                continue
            if not all(name in bands for name in self.band_names):
                logger.warning(
                    "Missing expected band(s) in %s (found %s, need %s)", path, sorted(bands), self.band_names
                )
                continue
            for name in self.band_names:
                per_band_arrays[name].append(bands[name])
            valid_days.append(day)

        if not valid_days:
            logger.error("No usable daily files for %s/%s", year, output_path)
            return False

        time_coord = pd.to_datetime([f"{year}{d:03d}" for d in valid_days], format="%Y%j")
        concatenated = {
            name: xr.concat(arrays, dim="time").assign_coords(time=time_coord)
            for name, arrays in per_band_arrays.items()
        }

        mean_da = concatenated[self.mean_band]
        min_da = concatenated[self.min_band]
        max_da = concatenated[self.max_band]

        valid_mask = mean_da.notnull() & (mean_da >= self.value_min) & (mean_da <= self.value_max)
        stats = _composite_glass_annual_stats(
            mean_da, valid_mask, min_da=min_da, max_da=max_da, thresholds=(self.cold_stress_k, self.heat_stress_k)
        )

        data_vars = {name: arr.squeeze("time", drop=True).astype("float32") for name, arr in stats.items()}
        out_ds = xr.Dataset(data_vars)
        out_ds = out_ds.rio.write_crs(mean_da.rio.crs)
        out_ds.attrs.update({"source_type": "glass_modis", "variant": self.variant, "year": str(year)})
        out_ds = out_ds.compute()

        return self._write_annual_geotiff(out_ds, output_path)

    @staticmethod
    def _write_annual_geotiff(ds: xr.Dataset, output_path: str) -> bool:
        """Same shape as `ModisSource._write_annual_geotiff` (`modis/
        source.py:482-531`) -- `rasterio.open(..., compress="LERC_ZSTD",
        zstd_level=9, tiled=True)`, atomic `.tmp`+`os.replace`."""
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
            ) as dst:
                dst.write(stacked)
                for i, name in enumerate(band_names, start=1):
                    dst.set_band_description(i, name)
                dst.update_tags(**{k: str(v) for k, v in ds.attrs.items()})
            os.replace(tmp_path, output_path)
            return True
        except Exception:
            logger.exception("Error writing GLASS-MODIS annual GeoTIFF to %s", output_path)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False

    # -- PREPARE: mosaic tiles, reproject via the shared SpatialProcessor,
    # exactly like ModisSource (`modis/source.py:539-666`) -----------------

    def _prepare_output_path(self) -> str:
        return layout.grid_store_path(
            self.ctx.data_root,
            self.path_prefix,
            grid_id=self.ctx.grid_id,
            family=f"glass_modis_{self.variant}",
            suffix="",  # cell_id-keyed parquet parts, not a Zarr store
        )

    def _discover_prepare(self, selection: TargetSelection) -> List[StepTarget]:
        """One PREPARE target for the whole source ("all"), matching
        `ModisSource._discover_prepare` -- `_execute_prepare` loops over all
        available years internally via `run_tiled_prepare` rather than one
        StepTarget per year sharing the same `output_path`."""
        stage1_root = self.output_root(PipelineStep.FETCH)
        output_path = self._prepare_output_path()
        years = list(range(self.day_range_start[0], self.day_range_end[0] + 1))

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
                    **verify.verification_meta(
                        self.cfg.raw,
                        expected_vars=self.STAT_VARS,
                        value_range=(self.value_min, self.value_max),
                        range_vars=self.RANGE_VARS,
                    ),
                },
            )
        ]

    def _read_annual_geotiff(self, path: str, year: int) -> xr.Dataset:
        import rasterio

        da_ = rxr.open_rasterio(path, masked=True)
        with rasterio.open(path) as src:
            descriptions = src.descriptions

        time_coord = [pd.Timestamp(f"{year}-12-31")]
        data_vars = {}
        for i, name in enumerate(descriptions):
            if not name:
                continue
            band_da = da_.isel(band=i, drop=True)
            band_da = band_da.expand_dims(time=time_coord, axis=0).expand_dims(band=[1], axis=1)
            data_vars[name] = band_da

        ds = xr.Dataset(data_vars)
        return ds.rio.write_crs(da_.rio.crs)

    def _mosaic_tiles(self, tile_files: List[str], year: int) -> xr.Dataset:
        datasets = [self._read_annual_geotiff(f, year) for f in tile_files]
        return xr.combine_by_coords(datasets, combine_attrs="override")

    def _execute_prepare(self, target: StepTarget) -> bool:
        from src.data.common.geobox import get_target_geobox
        from src.data.common.prepare.driver import run_tiled_prepare
        from src.data.common.raster.spatial import SpatialProcessor
        from src.data.sources.modis import tiles as modis_util
        from src.data.sources.steps import is_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping PREPARE -- already complete: %s", target.output_path)
            return True

        stage1_root = self.output_root(PipelineStep.FETCH)
        years = target.meta["years"]

        # One year's mosaic memoized at a time -- `run_tiled_prepare` iterates
        # (year, tile) year-major, so at most one year's full-mosaic Dataset
        # is ever held in memory (same pattern as GlassAvhrrSource._execute_prepare).
        mosaic_cache: Dict[int, xr.Dataset] = {}

        def year_mosaic(year: int) -> Optional[xr.Dataset]:
            if year not in mosaic_cache:
                mosaic_cache.clear()
                year_dir = os.path.join(stage1_root, str(year))
                tile_files = sorted(
                    os.path.join(year_dir, f) for f in os.listdir(year_dir) if f.endswith(".tif")
                ) if os.path.isdir(year_dir) else []
                if not tile_files:
                    logger.error("No stage-1 tiles for year %d at %s", year, year_dir)
                    return None
                mosaic = self._mosaic_tiles(tile_files, year)
                if mosaic.rio.crs is None:
                    mosaic = mosaic.rio.write_crs(modis_util.SINUSOIDAL_PROJ4)
                mosaic_cache[year] = mosaic
            return mosaic_cache[year]

        def raw_getter(tile, year: int) -> Optional[xr.Dataset]:
            mosaic = year_mosaic(year)
            if mosaic is None:
                return None
            bbox = tile.geobox.pad(32, 32).extent.to_crs(mosaic.rio.crs).boundingbox
            clipped = mosaic.sel(y=slice(bbox.top, bbox.bottom), x=slice(bbox.left, bbox.right))
            if clipped.sizes.get("x", 0) == 0 or clipped.sizes.get("y", 0) == 0:
                return None
            return clipped

        try:
            with self._dask_client() as client:
                target_geobox = get_target_geobox(self.ctx)

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
                        resampling="nearest",
                        dst_nodata=float("nan"),
                        processing_version=self.PROCESSING_VERSION,
                        override=self.cfg.override,
                    )
        except Exception:
            logger.exception("Error in GLASS-MODIS spatial processing for years %s (variant=%s).", years, self.variant)
            return False

    # _dask_client: inherited from DataSource (src/data/sources/base.py) --
    # unlike GlassAvhrrSource, no per-source dashboard-port override needed;
    # this rebuilt pipeline no longer runs its own bespoke chunked-tile
    # reprojection, just the shared SpatialProcessor (like ModisSource).


registry.register(
    "glass_modis",
    __name__,
    GlassModisSource.__name__,
    GlassModisSource.STEPS,
)
registry.register(
    "glass_ta_modis",
    __name__,
    GlassModisSource.__name__,
    GlassModisSource.STEPS,
)

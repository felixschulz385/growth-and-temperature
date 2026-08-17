# 12. Split GLASS AVHRR/MODIS; rebuild GLASS-MODIS on the raw-MODIS tile×year pipeline; add GLASS Land Air Temperature

**Status: planned, not yet implemented.** This doc records the agreed design
so implementation can start in a later session without re-deriving it.
Nothing in `src/data/sources/glass/` has changed yet.

`src/data/sources/glass/source.py` currently implements one `GlassSource`
class shared by `glass_modis` and `glass_avhrr`, branching internally on
`data_source_kind` (see `11-glass-static-fetch.md` for how its FETCH step
reached its current static-target shape). GLASS-MODIS's own pipeline today:
FETCH is a per-day HDF crawl+download, kept forever on disk/HPC; PREPARE
builds a per-`(year, tile)` annual **zarr** with a naive, seasonally-biased
`resample("1YE").mean()` (flagged in `glass/source.py`'s own module
docstring as a known bug "deferred to a separate, labelled follow-on
change" -- this doc is that change), then reprojects via a bespoke
32px-halo/"mode"-resampling tiled zarr path.

This predates `ModisSource` (`src/data/sources/modis/source.py`), which now:
downloads-and-combines per `(tile, year)` directly into one LERC_ZSTD-
compressed multi-band GeoTIFF at FETCH time, using the shared month-first
`composite_annual_stats()` compositor (`src/data/common/raster/
compositing.py`), and reprojects via the shared `SpatialProcessor`
(`src/data/common/raster/spatial.py`) at PREPARE time.

## 0. Goals

1. **Keep AVHRR's behavior unchanged**, but split it into its own class/file
   -- GLASS sources are being separated, not left as one branchy class.
2. **Rebuild GLASS-MODIS** to match raw MODIS's shape: FETCH targets are
   `(tile, year)`; each downloads that tile's daily source files, combines
   them into one annual multi-band GeoTIFF (LERC_ZSTD, `raw/` path
   convention), and is auto-pushed to HPC per file (existing
   `transfer_mode=auto` machinery, extended to the new source id). PREPARE
   mosaics tiles and reprojects via the shared `SpatialProcessor`, like
   `ModisSource`.
3. **Add a new "Land Air Temperature" product** at
   `https://glass.hku.hk/archive/Ta/MODIS/`, `day_range` 055/2000-365/2020
   (identical span to `glass_modis`), sharing the rebuilt pipeline as a
   second variant -- mirroring how `ModisSource` has `main`/`extended`.

## 1. QA-band investigation (done against the live archive)

Two real tiles were downloaded directly from `glass.hku.hk` and inspected
(local GDAL lacks the HDF4 plugin and `pyhdf`/`h5py` aren't installed here,
so inspection was via raw-binary string scanning for embedded HDF4 SDS
field names -- sufficient to answer "is there a QC band", not to read exact
attribute values):

- **LST** (`GLASS06A01`, e.g. `.../LST/MODIS/Daily/1KM/2000/055/
  GLASS06A01.V01.A2000055.h08v05.2022021.hdf`): single SDS,
  `DataFieldName="LST"`, with `_FillValue`/`scale_factor` attrs. **No QC/QA
  subdataset** -- only `LST` appears as an embedded field name, unlike
  satellite MOD11/MOD21's separate `QC_Night` band. Matches what
  `GlassSource._calculate_statistics` already does today (raw-range masking
  only, no QC decode) -- nothing to port from
  `modis/tiles.py::decode_qc_valid_mask`.
- **Ta** (`GLASS18A01`, e.g. `.../Ta/MODIS/2000/055/
  GLASS18A01.V10.A2000055.h03v06.2023300.hdf`): **3 SDS bands** -- `Ta_min`,
  `Ta_mean`, `Ta_max` -- each with its own `fillvalue`/`scale_factor`/
  `valid_range` attrs. Also **no QC/QA band**. A `.hdf.xml` granule-metadata
  sidecar exists per file but carries only granule/tile bounding-box
  metadata, no band-level unit/scale info -- not needed for processing.
- **Conclusion**: no bit-flag QC decoding is needed for either product.
  Validity = not-fill AND within a physically-plausible value range, read
  from each file's own `_FillValue`/`scale_factor`/`valid_range` attrs (via
  `rxr.open_rasterio(path, masked=True)`, which auto-applies them) rather
  than GLASS's current hardcoded raw `[20000, 35000]` magic-number mask --
  this generalizes cleanly to Ta without needing to know its exact
  scale/unit up front.
- **Open item, not blocking**: Ta's physical unit (Celsius vs Kelvin)
  could not be confirmed locally. First implementation step below is to
  confirm this against a real file in an environment with working HDF4
  support (HPC, or `conda install -c conda-forge libgdal-hdf4 pyhdf`
  locally) before hardcoding any Kelvin-range default -- same posture as
  the existing `qc_max_lst_error_k: 2.0  # pending QC bit-layout
  verification` pattern already used elsewhere in this config.
- Also noted for implementation: Ta's directory listing has multiple
  sidecar files per tile/day sharing the same `A{year}{day}.{tile}.` token
  (`.hdf`, `.hdf.xml`, three `.jpg` previews) -- `_match_in_listing` must
  filter to `.hdf` exactly, or it will nondeterministically match a preview
  image instead of the data file.

## 2. Annual stats

Both variants produce the same 8-band shape, computed by one shared helper
both call (given `mean_da`, `valid_mask`, optional `min_da`/`max_da`
defaulting to `mean_da` when the product only has one band, and
`thresholds=(cold_stress_k, heat_stress_k)`):

- `mean`, `std`, `valid_period_count`, `valid_month_count`, `count_above`,
  `count_below` via the shared `composite_annual_stats()` (month-first-
  then-annual -- fixes the seasonal-bias bug the current naive resample
  has).
- `max`, `min` computed **directly from daily data**, not month-weighted --
  extremes shouldn't be diluted by month-averaging; that would defeat the
  point of tracking an extreme.

Dropped vs. today: `median`, `rollmax3`/`rollmin3`, and the separate
monthly zarr.

- **LST**: `mean_da = min_da = max_da` = the single daily `LST` band (only
  one band exists).
- **Ta**: `mean_da = Ta_mean`, `min_da = Ta_min`, `max_da = Ta_max` (the
  file already provides daily min/mean/max directly, so no derivation is
  needed).

`count_above`/`count_below` thresholds must be **daytime-appropriate**:
GLASS LST/Ta, unlike `ModisSource`'s night-only `LST_Night_1KM`, are
daytime/blended products, so `ModisSource`'s night defaults
(`heat_stress_k=298.15` / `cold_stress_k=273.15`) don't apply as-is. Expose
as configurable `heat_stress_k`/`cold_stress_k` (same knob names as
`ModisSource`, for consistency) with distinct daytime placeholder defaults
(`heat_stress_k=308.15` [35C], `cold_stress_k=273.15` [0C]) -- tune once
real values are inspected; this is a placeholder, not a verified figure,
same posture as the QA open item above.

## 3. PREPARE

Adopt the shared `SpatialProcessor` reprojection path -- mosaic tiles,
"nearest" resampling -- exactly like `ModisSource._execute_prepare`, for
**both** LST and Ta. AVHRR is untouched and keeps its current bespoke
`_process_years_chunked`/`_process_year_tiles` (32px halo, "mode"
resampling) path.

## 4. File layout

- **`src/data/sources/glass/avhrr.py`** (new): `GlassAvhrrSource
  (DataSource)`. Move, verbatim, everything AVHRR-only from today's
  `GlassSource`: `daterange_doy()`, `_parse_avhrr_filenames`, the AVHRR
  branch of `_group_daily_files`, `_process_file_group_hpc`/
  `_calculate_statistics` (`VARIABLE_NAME="LST"`, unchanged),
  `_create_annual_zarr_hpc`, `_process_years_chunked`/`_process_year_tiles`/
  `_aggregate_year_files` (EPSG:4326 branch only), and the whole
  crawl/listing/download FETCH machinery (`_listing_url`,
  `_list_single_directory`, `_listing_for`, `_match_in_listing`,
  `_execute_fetch`, `_pending_path`, `_downloaded_path`,
  `_index_existing_fetch_files`). No behavior change -- mechanical
  extraction with the `data_source_kind`/MODIS branches deleted.
- **`src/data/sources/glass/modis.py`** (new): `GlassModisSource
  (DataSource)`, registered twice -- `glass_modis` (variant `"lst"`,
  `VARIABLE_NAME="LST"`, single-band daily files) and `glass_ta_modis`
  (variant `"ta"`, three daily bands `Ta_min`/`Ta_mean`/`Ta_max`) --
  mirroring `ModisSource`'s `main`/`extended` dual-registration (`variant`
  derived from `cfg.source_id`, same as today's `data_source_kind`
  derivation). Contents:
  - Reuses the existing land-tile crawl target-space logic
    (`daterange_doy`, per-`(year,day)` listing memoization, `_listing_for`/
    `_list_single_directory`/`_match_in_listing`, with the `.hdf`-only
    filter from §1).
  - **New FETCH target granularity**: one `StepTarget` per `(tile, year)`,
    `key=f"{year}/{tile}"`, `output_path=os.path.join(raw_root, str(year),
    f"{tile}.tif")` -- deterministic, unlike today's unpredictable-filename
    daily targets, so **no `_last_fetch_output_path` hack is needed** (same
    simplification `ModisSource` already has). Use `resolve_fetch_listing()`
    / `Completion.PRECOMPUTED` vs `PATH_EXISTS` exactly like
    `ModisSource._plan_fetch` (`modis/source.py:284-328`) and today's
    `GlassSource._plan_fetch`.
  - **`_execute_fetch(target)`**: for the target's `(tile, year)`, iterate
    the day range clipped to that year (reuse `daterange_doy` per-year
    slice), fetch each day's listing (memoized), find+download the matching
    `.hdf` into `self.temp_dir` (scratch, not the permanent raw tree --
    daily files are no longer the FETCH deliverable). Open each downloaded
    file with `rxr.open_rasterio(path, masked=True)`, concat along `time`,
    build a valid mask (finite + configurable physical `value_range`,
    mirroring `_LST_VALUE_RANGE`/`ModisSource.lst_min_k`/`lst_max_k`), call
    the §2 stats helper to get the 8 annual bands, stack into one
    `xr.Dataset`, and write via a GLASS-local adaptation of
    `ModisSource._write_annual_geotiff` (`modis/source.py:482-531`) -- same
    `rasterio.open(..., compress="LERC_ZSTD", zstd_level=9, tiled=True)`,
    atomic `.tmp`+`os.replace`. Delete/let-expire the scratch daily
    downloads once the tile-year tiff is written successfully.
  - **`transfer_units()`**: copy `ModisSource.transfer_units()`
    (`modis/source.py:228-262`) verbatim (walks `raw_root/<year>/<tile>.tif`).
  - **PREPARE**: copy `ModisSource`'s `_discover_prepare`/
    `_read_annual_geotiff`/`_mosaic_tiles`/`_execute_prepare`/
    `_prepare_output_path` shape (`modis/source.py:539-666`), parameterized
    by variant for `v2_family`/output zarr filename (e.g.
    `glass_modis_lst` / `glass_modis_ta`, matching the `_grid_output_path()`
    naming convention `GlassSource` already uses today).
  - `output_root()` override: same shape as `ModisSource.output_root`
    (`modis/source.py:193-222`) -- FETCH uses the base-class default,
    PREPARE/GRID uses the `SpatialProcessor`-appropriate grid root, keyed
    off a per-variant path prefix rather than `cfg.data_path`, matching
    today's `GlassSource.output_root`.
  - Shared compositing helper (§2): keep local to `glass/modis.py` unless
    writing it reveals it's trivially generic enough to also benefit
    `ModisSource` later.
- **`src/data/sources/glass/source.py`**: deleted (fully superseded by
  `avhrr.py` + `modis.py`).
- **`src/data/sources/glass/__init__.py`**: update the
  `from .source import GlassSource` export to export
  `GlassAvhrrSource`/`GlassModisSource` from the new modules instead.
- **`src/data/sources/registry.py`**: `_SOURCE_MODULES` -- replace
  `"src.data.sources.glass.source"` with `"src.data.sources.glass.avhrr"`
  and `"src.data.sources.glass.modis"` (`registry.py:150`).
- **`src/data/common/fetch/transfer_mode.py`**: add `"glass_ta_modis"` to
  `AUTO_TRANSFER_DEFAULT_SOURCES` (`transfer_mode.py:27`) so the new
  product gets outstanding-vs-HPC + per-file auto-push for free, same as
  `glass_modis`/`glass_avhrr` already do.

## 5. Config (`orchestration/configs/data.yaml`)

- Leave `glass_avhrr` untouched.
- `glass_modis`: keep `type: "glass_modis"`, `base_url`, `day_range`,
  `land_tiles`; add `heat_stress_k`/`cold_stress_k` (daytime placeholders,
  §2) and a `value_range`/fill-mask config surface mirroring
  `ModisSource`'s `lst_min_k`/`lst_max_k`; update `verification.
  expected_vars` to the new 8-band set (`mean, std, max, min, count_above,
  count_below, valid_period_count, valid_month_count`), dropping
  `median`/`rollmax3`/`rollmin3`.
- New `glass_ta_modis` block: `type: "glass_ta_modis"`,
  `base_url: "https://glass.hku.hk/archive/Ta/MODIS/"`,
  `day_range: {start: [2000, 55], end: [2020, 365]}` (identical to
  `glass_modis`), reuse the same `land_tiles` list (same sinusoidal grid),
  its own `heat_stress_k`/`cold_stress_k`/value-range once Ta's unit is
  confirmed, and a `verification` block matching the same 8-band shape.
- `registry.register()` calls in `glass/modis.py` register both
  `"glass_modis"` and `"glass_ta_modis"` ids to `GlassModisSource`.

## 6. Tests

- `tests/data/sources/glass/test_glass_fetch.py`,
  `test_glass_plan.py`, `test_glass_grid_geobox.py`,
  `test_glass_fetch_remote_check.py`: split along the same AVHRR/MODIS line
  -- AVHRR-specific cases move to a `test_glass_avhrr_*.py` importing
  `GlassAvhrrSource` (same assertions, behavior unchanged); MODIS-specific
  cases get rewritten against `GlassModisSource`'s new `(tile, year)` target
  shape, `.hdf`-only listing match, and the new stats helper (add a focused
  unit test for the shared compositing helper: given synthetic daily
  arrays, assert the 8 output bands' values/shape, and that `max`/`min` are
  NOT month-weighted while `mean`/`std` are).
- `tests/data/common/fetch/test_transfer_mode.py`: add `glass_ta_modis` to
  whatever assertion enumerates `AUTO_TRANSFER_DEFAULT_SOURCES`.
- `tests/cli/data/test_handlers_run_fetch_push.py` /
  `tests/data/sources/test_fetch_protocol.py` / `test_step_contract.py`:
  check these still pass given the removed `_last_fetch_output_path`
  reliance for the MODIS variant (AVHRR keeps needing it, unchanged) --
  likely no change needed, but verify.

## 7. Explicitly out of scope / deferred

- `orchestration/slurm/glass-modis-grid.sh` / `glass-modis-prepare.sh` /
  job definitions in `orchestration/slurm/jobs.yaml`: will need a new
  `glass-modis-ta-*.sh` pair and updated `--source` args once the new
  source id exists -- a small mechanical follow-up pass once the source
  code lands and is validated, not blocking the core implementation.
- Migrating already-downloaded historical `glass_modis` daily HDFs / the
  old annual zarr store on HPC: out of scope here (a data-migration/
  backfill concern, not a code-architecture one) -- flag to the user once
  implementation is done, don't silently delete or migrate existing HPC
  data.

## 8. Verification (once implemented)

1. Unit tests: `pytest tests/data/sources/glass/
   tests/data/common/fetch/test_transfer_mode.py
   tests/data/sources/test_fetch_protocol.py
   tests/data/sources/test_step_contract.py -v`.
2. Dry-run plan: `python -m src.cli.data plan --source glass_modis --step
   fetch` and `--source glass_ta_modis --step fetch` against a small
   `TargetSelection` (a couple of tiles/years) -- confirm `(tile, year)`
   keys, not `(year, day, tile)`.
3. One real end-to-end tile-year FETCH execution (one land tile, one year)
   against the live `glass.hku.hk` archive for both `glass_modis` and
   `glass_ta_modis`, confirming: daily files download, the combined GeoTIFF
   has the expected 8 bands with sane values (spot-check against the raw
   HDF via whatever HDF4-capable tool is available at that point), LERC_ZSTD
   compression is applied, and (if an HPC target is configured) the
   per-file auto-push fires.
4. One PREPARE execution over that same tile/year, confirming the
   `SpatialProcessor` mosaic+reproject path runs end-to-end and produces a
   zarr store analogous to `ModisSource`'s.

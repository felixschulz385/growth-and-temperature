# glass — GLASS (Global LAnd Surface Satellite) LST

Registry: `src/data/sources/glass/source.py`, class `GlassSource`. The class
itself carries `ID = "glass"`, but per its own comment that id is **not
directly registered** — instead, two separate `registry.register(...)` calls
at the bottom of the file register `"glass_modis"` and `"glass_avhrr"`
against the same class (no shared alias grouping, unlike `eog`). Backs
config keys **`glass_modis`, `glass_avhrr`**. Steps:
`STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE, PipelineStep.GRID)` —
identical across both variants, since it's one class. `REQUIRES`: none (not
set; defaults to `()`).

Crawl/download logic is in `src/data/sources/glass/crawler.py`
(`_CrawlerMixin`), mixed into `GlassSource`.

Which variant an instance is (`self.data_source_kind`, `"MODIS"` or
`"AVHRR"`) is derived in `__init__` from `cfg.source_id` (`"avhrr" in
cfg.source_id.lower()` → AVHRR, else MODIS) — the registered id itself, not
`data_path`/`base_url` content. Per the module docstring, MODIS and AVHRR
have genuinely different filename formats, output layouts, and CRS handling
(MODIS: native sinusoidal tiles; AVHRR: one global EPSG:4326 file/day) —
unlike `eog`'s three aliases, which share identical behavior.

`GlassSource.output_root()` **overrides** the base `DataSource.output_root`:
GLASS's output root is keyed by a fixed `path_prefix` constant
(`MODIS_PATH_PREFIX = "glass/LST/MODIS/Daily/1KM/"` or
`AVHRR_PATH_PREFIX = "glass/LST/AVHRR/0.05D/"`), not `cfg.data_path` — so
`data_path` (unset in both config blocks; `cfg.data_path is None` at
construction) exists only as a fallback default (`cfg = dataclasses.replace(
cfg, data_path=self.path_prefix.rstrip("/"))`), matching the old
`GlassPreprocessor.get_hpc_output_path` convention. In practice this means
`data_path` and `path_prefix` are the same string for both variants.

## Config variants

Neither block sets `namespace`. `chunk_size` defaults to
`{"band": 1, "x": 500, "y": 500}`; `file_extensions` defaults to `[".hdf"]`;
`version` defaults to `"v1"`; `grid_cells` (an optional allowlist of
`h##v##` MODIS tiles) defaults to unset (all cells). Both blocks *do* set an
explicit `download:` section.

| variant (config key) | `data_source_kind` | `path_prefix` (= effective `data_path`) | `base_url` | `year_range` | `download:` block | what's distinctive |
|---|---|---|---|---|---|---|
| `glass_modis` | `MODIS` | `glass/LST/MODIS/Daily/1KM/` | `https://glass.hku.hk/archive/LST/MODIS/Daily/1KM/` | `[2000, 2020]` | `batch_size=1000, max_concurrent_downloads=4, tar_max_files=100, tar_max_size_mb=500` | multiple sinusoidal tiles (`h##v##`) per day; PREPARE/GRID keyed by `(year, tile)` |
| `glass_avhrr` | `AVHRR` | `glass/LST/AVHRR/0.05D/` | `https://glass.hku.hk/archive/LST/AVHRR/0.05D/` | `[1982, 2021]` | `batch_size=1000, max_concurrent_downloads=4, tar_max_files=100, tar_max_size_mb=500` | one global 0.05° file per day; PREPARE/GRID keyed by year only |

Both `download:` blocks are identical in value, but each is set
independently per config block (not inherited/shared).

## FETCH

`_plan_fetch()`/`_execute_fetch()` produce one `StepTarget` (`key="all"`,
`Completion.NEVER`) delegating to the shared `run_fetch()` driver
(`src/data/common/fetch/driver.py`), which requires `ctx.ssh_target` to be
configured.

**Discovery**: `_CrawlerMixin.list_remote_files()` — a plain `requests` +
BeautifulSoup recursive crawl of `base_url` (no authentication, unlike
`eog`). Sorts links by `href` before recursing "to process years and days in
order"; a fixed `0.5`s delay before each directory GET. Yields
`(relative_path, file_url)` for files whose extension is in
`file_extensions` (default `[".hdf"]`). `get_all_entrypoints()` additionally
walks the tree to enumerate `{year, day}` entrypoints (4-digit year dirs,
then 3-digit day subdirs if present, else `day=0`); `has_entrypoints = True`
for GLASS (vs. `False` for `eog`) — a `0.3`s delay between year-level
requests. The async crawler (`list_remote_files_async`) uses `aiohttp` with
`limit=5, limit_per_host=2` connections, `0.8`s delay before each directory
fetch, and batches subdirectory crawls (batch size 3, `1.0`s delay between
batches) — "Be more respectful with rate limiting" per the code comment.

**Download**: plain `requests`/`aiohttp` streamed GET
(`download`/`download_async`), the latter via
`src.data.common.fetch.http.download_with_retries` (retry logic lives in
that shared helper, not in `GlassSource` itself) with `aiohttp.ClientTimeout(
total=300, connect=30)` and `TCPConnector(limit=5, limit_per_host=2)` when no
session is passed in.

**Driver-level behavior** (shared, not GLASS-specific — see the `eog` doc
for the general description): batched downloads, atomic `.part`→
`os.replace()` writes, tar-bundled pushes to HPC. Both `glass_modis` and
`glass_avhrr` explicitly set `batch_size=1000, max_concurrent_downloads=4,
tar_max_files=100, tar_max_size_mb=500` in `data.yaml` (vs. `eog`'s
unconfigured driver defaults of `50`/`5`/`100`/`500`).

**Filename parsing** (used at PREPARE time, not FETCH):
`filename_to_entrypoint()` expects the GLASS date-token convention
(`A<year><day-of-year>`, e.g. `A2000055`), used generically for both
variants at the entrypoint level.

**Output path** (`output_root(FETCH)`, keyed by `path_prefix`, no namespace
configured for either variant):
- legacy: `<data_root>/<path_prefix>/raw` → `<data_root>/glass/LST/MODIS/Daily/1KM/raw` or `<data_root>/glass/LST/AVHRR/0.05D/raw`
- v2: `<data_root>/raw/<path_prefix>` → `<data_root>/raw/glass/LST/MODIS/Daily/1KM` or `<data_root>/raw/glass/LST/AVHRR/0.05D`

**File format/naming**: HDF (`.hdf`, per `file_extensions` default).
`GLASS06A01.V01.A2000055.h00v10.2022021.hdf` for MODIS (parsed by
`_parse_modis_filenames`: token `[2]` = `A<year><day>`, token `[3]` =
`h<HH>v<VV>` tile id); `GLASS08B31.V40.A1982001.2021259.hdf` for AVHRR
(parsed by `_parse_avhrr_filenames`: token `[2]` = `A<year><day>`, no tile
token — one global file per day).

## PREPARE

`_plan_prepare()` reads the local ledger's completed-fetch file list,
parses filenames via `_parse_filenames()` (dispatches to
`_parse_modis_filenames`/`_parse_avhrr_filenames` on `data_source_kind`)
into a `(path, year[, h, v])` table, filters by the requested year
selection, and groups into one `StepTarget` per key:
- **MODIS**: grouped by `(year, h, v)`, key `"<year>/h##v##"`; optionally
  pre-filtered to `grid_cells` if that config list is set (unset in both
  checked-in blocks). One target per tile-year, inputs = every daily `.hdf`
  found for that tile-year.
- **AVHRR**: grouped by `year` only, key `"<year>"`, `grid_cell="global"`.
  One target per year, inputs = every daily `.hdf` found for that year.

**Processing** (`_process_file_group_hpc`, "ported verbatim" from the old
`GlassPreprocessor`): opens each day's HDF via
`rioxarray.open_rasterio(decode_coords="all", chunks=self.chunk_size)`,
concatenates along a new `time` dimension (dates from `A<year><day>` via
`pd.to_datetime(..., format="%Y%j")`), then computes annual + monthly
statistics via `_calculate_statistics()`.

**`_calculate_statistics()`** — masks raw LST to `[20000, 35000]`
(physically `200–350K` at the source's own `scale_factor=0.01`) before
computing:
- **annual** (`resample(time="1YE")`): `mean`, `median`, `std`, `max`,
  `min`, `rollmax3`/`rollmin3` (3-day centered rolling mean, then annual
  max/min of that), plus day-count variables `gt30C` (days `>30315`, i.e.
  `>303.15K`), `lt0C` (days `<27315`, i.e. `<273.15K`), `valid_count`
  (unmasked-day count).
- **monthly** (`resample(time="1ME")`): `mean`, `median`, `std`,
  `valid_count` — written to a separate sibling `_monthly.zarr` file, not
  consumed further by GRID.
- Packing: stat variables are `fillna(0).astype(uint16)` with attrs
  `{_FillValue: 0, scale_factor: 0.01, add_offset: 0.0}`; count variables
  (`gt30C`/`lt0C`/`valid_count`) are `fillna(0).astype(uint16)` with no
  scale/offset (raw day counts).

**Caveat, explicitly flagged in the module docstring as a known, deferred
bug**: `_calculate_statistics`'s annual stats are computed via a naive
`resample(time="1YE").mean()` directly from raw daily data, not derived from
its own already-computed `monthly_stats` — `docs/design/07-modis-ingest.md`
§4 flags this exact pattern as "do not copy this pattern" for MODIS's own
compositing. The module docstring states fixing it is "explicitly deferred
to a separate, labelled follow-on change," not done in this migration.

**Output path**: `<output_root(PREPARE)>/...`
- MODIS: `<year>/h##v##.zarr` (+ sibling `_monthly.zarr`)
  - legacy: `<data_root>/<path_prefix>/processed/stage_1/<year>/h##v##.zarr`
  - v2: `<data_root>/prepared/<path_prefix>/<year>/h##v##.zarr`
- AVHRR: `<year>.zarr` (+ sibling `_monthly.zarr`)
  - legacy: `<data_root>/<path_prefix>/processed/stage_1/<year>.zarr`
  - v2: `<data_root>/prepared/<path_prefix>/<year>.zarr`

**Format**: Zarr (annual + monthly stores), Blosc/zstd-compressed
(`clevel=3, shuffle=2`), `consolidated=True`, chunked `{x: 1000, y: 1000,
time: 1}` at write. **Completion**: `Completion.MARKER`.

## GRID

`_plan_grid()` lists all annual `.zarr` files (excluding `_monthly.zarr`)
under the PREPARE output dir matching the requested year selection, and
builds one `StepTarget`:
- MODIS: `key="all_cells"`, `meta.grid_cells` = sorted set of tile ids seen.
- AVHRR: `key="global"`.

Both also record `meta.missing_years` — the gap between the requested
`year_range`'s full span and the years actually found on disk (informational
only; does not block planning).

`_execute_grid()` — **does not use the shared `SpatialProcessor`** (module
docstring: "GLASS's own bespoke tiled-reprojection path predates it and is
ported as-is"). Steps, per variant-independent code path:
1. Creates an empty target zarr sized to the full canonical geobox
   (`_create_empty_target_zarr`) if one doesn't already exist — one time
   coordinate per detected year, all stat variables pre-allocated as
   `uint16` zeros, chunked `(1, 1, 512, 512)`.
2. For each year (`_process_years_chunked`): if MODIS has more than one tile
   for that year, first aggregates all tiles into one combined per-year temp
   zarr (`_aggregate_year_files`, native CRS: MODIS sinusoidal
   `+proj=sinu +lon_0=0 ... +a=6371007.181`, or EPSG:4326 for AVHRR); then
   tile-by-tile reprojects (`_process_year_tiles`, `GeoboxTiles` at
   `tile_size=2048`, `xr_reproject(..., resampling="mode", dst_nodata=np.nan)`)
   and region-writes into the shared output store. Integer stat variables are
   cast to `float32` and masked to `NaN` where `valid_count == 0` before
   reprojection, since `mode`-resampled `NaN` doesn't behave sensibly on an
   integer dtype.

**Output path**: `layout.grid_store_path(..., <legacy_filename>,
grid_id=ctx.grid_id, layout=ctx.layout, v2_family=<family>)`:

| variant | legacy filename | `v2_family` | legacy path | v2 path |
|---|---|---|---|---|
| `glass_modis` | `modis_timeseries_reprojected.zarr` | `glass_modis_lst` | `<data_root>/<path_prefix>/processed/stage_2[_ease6933]/modis_timeseries_reprojected.zarr` | `<data_root>/grid/<grid_id>/glass_modis_lst.zarr` |
| `glass_avhrr` | `avhrr_timeseries_reprojected.zarr` | `glass_avhrr_lst` | `<data_root>/<path_prefix>/processed/stage_2[_ease6933]/avhrr_timeseries_reprojected.zarr` | `<data_root>/grid/<grid_id>/glass_avhrr_lst.zarr` |

**Completion**: `Completion.MARKER`.

**Variables / verification** — `GlassSource._STAT_VARS`,
`GlassSource._RANGE_VARS`, `GlassSource._LST_VALUE_RANGE` are **class
constants shared by both variants** (not per-`data_source_kind`), passed
into `verify.verification_meta()` in `_plan_grid` for both branches
identically; each variant's own `sources.<id>.verification:` block in
`data.yaml` sets the same values explicitly (pinning, not overriding to
something different):

| variable | in `expected_vars` | in `range_vars` (Kelvin check applies) | dtype | fill/nodata | meaning |
|---|---|---|---|---|---|
| `mean` | yes | yes | `uint16` (PREPARE) → `float32` at GRID reprojection | `_FillValue: 0` (PREPARE); `NaN` post-reprojection | annual mean LST, `scale_factor=0.01` (so raw `20000–35000` ⇒ `200–350K`) |
| `median` | yes | yes | same | same | annual median LST |
| `std` | yes | **no** — excluded from the range check | same | same | annual std-dev of LST; not on the same absolute Kelvin scale as the others, so excluded from `_RANGE_VARS` |
| `max` | yes | yes | same | same | annual max LST |
| `min` | yes | yes | same | same | annual min LST |
| `rollmax3` | yes | yes | same | same | annual max of a 3-day centered rolling mean |
| `rollmin3` | yes | yes | same | same | annual min of a 3-day centered rolling mean |

`value_range = [150, 350]` (Kelvin) applies only to the `range_vars` above.
`gt30C`/`lt0C`/`valid_count` (day-count variables `_calculate_statistics`
also writes into the *same* per-year PREPARE dataset as the seven stat
variables above) are **not** in `expected_vars` and so are never checked by
GRID verification — but they are not filtered out of the GRID output
itself: `_create_empty_target_zarr` derives its variable list generically
from `sample_ds.data_vars.keys()`, excluding only the CRS/grid-mapping
names in `_NON_DATA_VAR_NAMES = {"spatial_ref", "crs", "grid_mapping"}`
(`src/data/common/raster/spatial.py`), so on this reading `gt30C`/`lt0C`/
`valid_count` are carried through into the reprojected GRID store
alongside the seven checked stat variables, just uninspected by
verification. (`_STAT_VARS` is used only to build `expected_vars`, never to
restrict which variables GRID actually processes.)

**TODO (needs live data):** confirmation that `gt30C`/`lt0C`/`valid_count`
are actually present as variables in a real reprojected GRID zarr (the code
trace above says they should be, but this hasn't been checked against a
live store); actual year coverage achieved per variant vs. the configured
`year_range`; actual MODIS tile coverage (`grid_cells` unset ⇒ presumably
all global land+ocean tiles, not confirmed); on-disk store sizes; whether
any `missing_years` gaps exist in the current production run.

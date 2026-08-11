# eog — Earth Observation Group nighttime lights (DMSP / VIIRS / DVNL)

Registry: `src/data/sources/eog/source.py`, class `EogSource`. Registered
under `ID = "eog"` with `ALIASES = ("eog_dmsp", "eog_viirs", "eog_dvnl")`
(single `registry.register(EogSource.ID, __name__, EogSource.__name__,
EogSource.STEPS, aliases=EogSource.ALIASES)` call) — one Python class
instantiated three times, once per config block, each producing an
independent `SourceConfig`. Backs config keys **`eog_dmsp`, `eog_viirs`,
`eog_dvnl`**. Steps: `STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE,
PipelineStep.GRID)` — identical across all three variants, since it's one
class. `REQUIRES`: none (not set; defaults to `()`).

Crawl/download logic is in `src/data/sources/eog/crawler.py`
(`_CrawlerMixin`) and `src/data/sources/eog/session.py` (`_SessionMixin`),
both mixed into `EogSource`.

Which variant an instance is (`self.source_type`, one of `dmsp` /
`viirs_annual` / `viirs_dvnl`) is derived in `_derive_source_type()` from
`cfg.source_id` — the literal `sources.<id>:` config block key
(`eog_dmsp`/`eog_viirs`/`eog_dvnl`), matched by substring (`"dmsp"` /
`"dvnl"` / `"viirs"`, checked in that order) — not from `data_path` or
`base_url` content. `source_type` drives PREPARE's output variable name and
GRID's output filename / `v2_family`. A `source_id` matching none of the
three raises `ValueError` at construction rather than silently guessing.

## Config variants

All fields below are read straight from `orchestration/configs/data.yaml`.
None of the three blocks sets `file_extensions`, `resampling`, `namespace`,
or `download:`, so each falls back to `EogSource`'s Python-side defaults
(`file_extensions = [".tif", ".tgz", ".tar.gz", ".gz"]`,
`resampling = "sum"`, `namespace = None`, FETCH driver defaults
`batch_size=50, max_concurrent_downloads=5, tar_max_files=100,
tar_max_size_mb=500`).

| variant (config key) | `source_type` | `data_path` | `base_url` | `year_range` | what's distinctive |
|---|---|---|---|---|---|
| `eog_dmsp` | `dmsp` | `eog/dmsp` | `https://eogdata.mines.edu/wwwdata/dmsp/v4composites_rearrange/` | `[1992, 2013]` | DMSP-OLS: 6-bit digital number (DN), filenames concatenate satellite code + year with no delimiter (e.g. `F182019...`) |
| `eog_viirs` | `viirs_annual` | `eog/viirs` | `https://eogdata.mines.edu/nighttime_light/annual/v21/` | `[2012, 2021]` | VIIRS annual composite v21; continuous radiance, filenames delimit the year normally |
| `eog_dvnl` | `viirs_dvnl` | `eog/dvnl` | `https://eogdata.mines.edu/wwwdata/viirs_products/dvnl/` | `[2013, 2019]` | DVNL product; continuous radiance, filenames delimit the year normally |

## FETCH

`_plan_fetch()`/`_execute_fetch()` produce one `StepTarget`
(`key="all"`, `Completion.NEVER` — always re-planned/re-run, no marker) that
delegates to the shared `run_fetch()` driver
(`src/data/common/fetch/driver.py`), which requires `ctx.ssh_target` (an
HPC/remote transfer target) to be configured — logs a warning and returns
`False` otherwise.

**Discovery**: `_CrawlerMixin.list_remote_files()` — a Selenium-driven
recursive crawl of `base_url` (`session.py::_init_selenium_driver` launches
headless Chrome). Recurses into subdirectories up to `max_depth=8`, with a
randomized `1 + random()` second delay before loading each directory page
("Add small delay to avoid hammering the server", per the code comment).
Parses Apache-style directory listing HTML (`td.indexcolname` links,
falling back to all `<a>` tags), yields `(relative_path, file_url)` pairs
for links whose extension is in `file_extensions`. Requires authentication:
`_SessionMixin._check_and_handle_login()` fills a login form
(`https://eogdata.mines.edu/nighttime_light/login/`, fields `#username`/
`#password`, submit button `#kc-login`) using `EOG_USERNAME`/`EOG_PASSWORD`
env vars — a missing/unset credential pair only logs a warning at
`EogSource.__init__` time, not a hard failure; the actual login attempt
raises `ValueError` if both are still unset when a login form is detected.
Async listing (`list_remote_files_async`) runs the synchronous Selenium
crawl in a thread pool, since Selenium itself isn't async.

**Download**: `download_file()` navigates the driver to the file URL, then
polls the shared Selenium download directory for a new, non-`.tmp`/
`.crdownload` file (5s interval, 300s timeout), and copies the newest
completed file to the requested output path.

**Driver-level behavior** (shared with every FETCH-capable source, not
EOG-specific): downloads happen in batches (`batch_size` pending files at a
time, `max_concurrent_downloads` concurrent), each written to a `.part` file
and atomically `os.replace()`d into place on success; completed local files
are bundled into tar archives (capped at `tar_max_files` files /
`tar_max_size_mb` MB each) and pushed to the HPC target over SSH, tracked in
the source's ledger. None of the three `eog_*` blocks sets a `download:`
config section, so all three use the driver's built-in defaults
(`batch_size=50, max_concurrent_downloads=5, tar_max_files=100,
tar_max_size_mb=500`) rather than a per-source override.

**Output path** (`output_root(FETCH)` = `layout.raw_root()`; no namespace
configured for any of the three variants):
- legacy: `<data_root>/<data_path>/raw`
- v2: `<data_root>/raw/<data_path>`

Only `data_path` differs per variant (`eog/dmsp`, `eog/viirs`, `eog/dvnl`);
the path *shape* is identical across all three.

**File format/naming**: whatever EOG serves at the crawled URL — no
enforced single format; `_select_best_file_for_year()` (used at PREPARE
time) prefers `.tif` over `.tgz` over `.tar.gz` over `.gz` when more than
one candidate file is found for the same year, per `file_extensions`'
default order. DMSP filenames embed satellite + year with no delimiter
(`F<sat><yyyy>...`); VIIRS/DVNL filenames delimit the year with `.`/`_`/`-`.

## PREPARE

`_plan_prepare()` reads the source's local ledger
(`SourceLedger.completed_fetch_files()`, keyed by `data_path`) for the list
of already-fetched relative paths, extracts a year from each via
`_extract_year_from_path()`, and produces one `StepTarget` per year in the
requested selection.

**Year extraction**: DMSP-specific pattern `F\d+(\d{4})` (satellite code
directly followed by year, e.g. `182019` → satellite `F18`, year `2019`)
is tried first; VIIRS/DVNL fall through to the generic delimited/bare
4-digit patterns also used by `acag`/`esacci`/`ntl_harm`. Extracted years
are sanity-clamped to `1990–2040`.

**Best-file selection**: `_select_best_file_for_year()` picks one file per
year, preferring the configured `file_extensions` order when multiple
candidates exist for the same year (falls back to the first candidate if
none of the preferred extensions match).

**Processing** (`_process_data_files`): gunzips `.gz` inputs to a temp file
first (cleaned up after the zarr write); opens via
`rioxarray.open_rasterio(chunks="auto")`; expands a singleton `time`
dimension coordinate at `f"{year}-12-31"`; for `dmsp`, additionally parses
and attaches a `satellite` attribute (e.g. `"F18"`) from the filename.
`_create_annual_zarr()` writes one data variable — named after
`source_type` (`dmsp` / `viirs_annual` / `viirs_dvnl`) — chunked
`{x: 1000, y: 1000}`, Blosc/zstd-compressed (`clevel=3,
shuffle="bitshuffle"`), `zarr_format=3`, `consolidated=False`.

**Output path**: `<output_root(PREPARE)>/<year>.zarr` — identical shape
across all three variants (only `data_path` differs):
- legacy: `<data_root>/<data_path>/processed/stage_1/<year>.zarr`
- v2: `<data_root>/prepared/<data_path>/<year>.zarr`

**Completion**: `Completion.MARKER` (sibling `<output>.complete` file).

**Caveat (historical, from the module docstring)**: the pre-migration
`EOGPreprocessor` called `_extract_year_from_path`/`_select_best_file_for_year`
but never defined either method, so every PREPARE call raised
`AttributeError`, silently caught, always returning `[]` — EOG's PREPARE
stage reportedly never produced output before this migration implemented
both methods for real. Not a currently-open bug, but relevant context for
judging how much production history exists for any given `<year>.zarr`.

## GRID

`_plan_grid()` lists every `<year>.zarr` under the PREPARE output directory
matching the requested year selection, and produces a single `StepTarget`
(`key="all"`) with all of them as inputs. `_execute_grid()` reprojects them
onto the canonical target geobox (`get_target_geobox(ctx)`) via the shared
`SpatialProcessor.process_spatial_standard`, assuming EPSG:4326 if a source
zarr has no CRS written, using `resampling=self.resampling` (default
`"sum"` — area-weighted, chosen because this is a radiance/DN field, not
`SpatialProcessor`'s own default `"nearest"`; not overridden per variant in
`data.yaml`). Output is one combined multi-year zarr per variant.

**Output path**: `layout.grid_store_path(..., f"{source_type}_timeseries_reprojected.zarr",
namespace=None, grid_id=ctx.grid_id, layout=ctx.layout,
v2_family=f"eog_{source_type}")`:
- legacy: `<data_root>/<data_path>/processed/stage_2[_ease6933 if grid_id=="ease6933"]/<source_type>_timeseries_reprojected.zarr`
- v2: `<data_root>/grid/<grid_id>/eog_<source_type>.zarr` (flat, no namespace)

The v2 filename/`v2_family` differs per variant: `eog_dmsp.zarr`,
`eog_viirs_annual.zarr`, `eog_viirs_dvnl.zarr`.

**Completion**: `Completion.MARKER`.

**Variables / verification** (`verify.verification_meta()`, called in
`_plan_grid` with Python-side defaults, overridable via each variant's own
`sources.<id>.verification:` block in `data.yaml` — in practice, all three
blocks below set the identical numbers the code already defaults to for
that variant, so the config isn't *changing* anything today, just pinning
it explicitly):

| variant | data variable | dtype | value_range (code default / `data.yaml`) | notes |
|---|---|---|---|---|
| `eog_dmsp` | `dmsp` | as written by PREPARE (from `rioxarray`, not re-cast at GRID time) | `[0, 63]` | "classic 6-bit DN" per code comment; unit not otherwise documented in code/config |
| `eog_viirs` | `viirs_annual` | as written by PREPARE | `[0, 1000]` | "continuous and can spike much higher over cities/flares" per code comment; unit not documented in code/config |
| `eog_dvnl` | `viirs_dvnl` | as written by PREPARE | `[0, 1000]` | same value_range as VIIRS annual; unit not documented in code/config |

`expected_vars` is a single-element tuple per variant (matching the
variable name above); no `range_vars` is set anywhere for `eog`, so the
range check applies to that one variable unconditionally.

**TODO (needs live data):** actual dtype of the stored raster values
(`rioxarray.open_rasterio` preserves the source file's native dtype —
not pinned/re-cast anywhere in `_process_data_files`/`_create_annual_zarr`);
physical units of the DN/radiance values; actual year coverage achieved per
variant (`year_range` in config is the *configured* span, not confirmed
achieved output); nodata/fill-value convention actually present in files
(no explicit `_FillValue`/nodata handling is set in `EogSource`'s own
PREPARE/GRID code, unlike GLASS's explicit `_FillValue: 0` packing);
on-disk store sizes.

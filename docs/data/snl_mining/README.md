# snl_mining — SNL/S&P Global mining property tables

Registry id `snl_mining` (`src/data/sources/snl_mining/source.py`, class
`SnlMiningSource`). Steps: **prepare, grid** — `fetch` is deliberately absent
(module docstring): acquisition is a manual S&P Capital IQ `.xls` export plus
an OpenAI batch-enrichment loop that isn't automatable, not a gap. `REQUIRES`:
`gadm` prepare + grid, `commodity_prices` prepare.

Two genuinely separate *code paths* live under `src/data/sources/snl_mining/`,
but they share one physical storage tree, `data/raw/snl_mining/`, and — since
both are ultimately "parsed xlsx" — one merged database file:

- **Part A — the pipeline source** (this doc's main subject): turns a
  manually-produced "stage-0" DuckDB into gridded per-pixel mine-density/
  price-shock rasters. Steps: `prepare`, `grid`.
- **Part B — the detail scraper** (`src/data/sources/snl_mining/scraper/`):
  a standalone, interactively-run Selenium tool that scrapes per-mine detail
  pages from the Capital IQ web UI. Not wired into `STEPS`/the `pipeline` CLI
  at all (module docstring: login/browser-session fragility means it still
  needs a human at the keyboard) — invoked only via
  `scripts/debug_snl_mining_scraper.py`. Documented here anyway since it does
  write durable, queryable output.

```
data/raw/snl_mining/
  database.duckdb   # merged: Part A's properties/property_texts/property_llm_years/
                     # source_files/property_work_history_events tables AND Part B's
                     # mines/mine_subsection_*/detail_*/screener_state/scrape_errors tables
  csv/               # optional CSV dump of Part B's detail_* tables (see below), + README.md schema doc
  scraping/          # Part B's downloaded .xlsx/.xls exports (EXPORT_DIR in scraper/config.py)
  imputation/        # Part A's OpenAI batch-enrichment scaffolding (manifest, batch_requests/, batch_outputs/, ...)
  logs/              # Part B's chromedriver per-run logs (centralized here, not scattered per-invocation)
```

Ephemeral, non-durable state (Selenium user-data dirs) lives outside `data/`
entirely, under `scratch_nobackup/snl_mining/browser_profiles/` — see the
repo's gitignored `/scratch_nobackup` convention — so it's never mistaken for
real scraper output.

---

## Part A — pipeline source (prepare, grid)

### Stage 0 — manual input (pre-pipeline, not a `STEPS` member)

Not produced by this codebase's `fetch` (there is none); produced by
`src/data/sources/snl_mining/notebooks/snl_mining_manual_xls_to_duckdb.ipynb`
(a manual S&P `.xls` export) plus `scripts/run_snl_mining_imputation.py`
(`src/data/sources/snl_mining/imputation.py` — an OpenAI batch-enrichment
job, converted from a notebook into a script; see "LLM year-imputation
script" below) and consumed as this source's raw input.

- **Path**: `<output_root(fetch)>/database.duckdb` (`output_root(fetch)` =
  `layout.raw_root()` — legacy: `<data_root>/snl_mining/raw/database.duckdb`;
  v2: `<data_root>/raw/snl_mining/database.duckdb`). Overridable via
  `sources.snl_mining.duckdb_path` in `data.yaml`. Shared with the scraper
  (Part B) — `scraper/config.py`'s `DEFAULT_DB_PATH` points at the same file,
  so PREPARE's `ATTACH ... READ_ONLY` sees both this notebook's tables and
  Part B's `detail_*`/`mines`/etc. tables side by side (no name collisions:
  verified against the live database).
- **Format**: DuckDB database.
- **Tables consumed by PREPARE** (names configurable in `data.yaml`, defaults
  shown):
  | table (config key) | default name | purpose |
  |---|---|---|
  | `properties_table` | `properties` | one row per mine/property (manual side); `latitude`/`longitude`/`actual_start_up_year`/`actual_closure_year` are the *primary* tier of the identity/location/closing-year fusion below, not the sole source anymore |
  | (fixed, not configurable) | `mines` | scraper's own identity table (38,531 rows on real data vs. `properties`' 38,404) — the fusion's backbone `FROM` table (`_fused_mines_from_clause`); a mine present here but not in `properties` still needs an opening-year signal (observed or LLM-imputed) to end up in `active_mines`, so scraper-only records with no year data are correctly excluded, not fabricated |
  | (fixed, not configurable) | `detail_location_map_claims__location` | scraper's `decimal_degrees` free-text field (`"lat, lon"`), fills a mine's location when `properties`' `latitude`/`longitude` is null |
  | (fixed, not configurable) | `detail_discoveries_milestones__milestones` | scraper's `event_type='Actual Closure'` rows (`period` parsed for its leading year, most recent wins if several), fills `closing_year` when `properties.actual_closure_year` is null. No scraped equivalent exists for *opening* year — verified: no `'Actual Startup'` milestone type exists among 49 real event types — so `opening_year` fusion stops at `COALESCE(manual, llm-imputed)`, unchanged from before this fusion existed |
  | `llm_years_table` | `property_llm_years` | LLM-imputed fallback opening/closing years (`llm_opening_year`, `llm_closing_year`), lowest-priority `COALESCE` tier; optional — PREPARE logs a warning and falls back to observed-only years if absent |
  | `work_history_table` | `property_work_history_events` | narrative work-history events (declared in config; not read by `_execute_prepare` directly in the current code path) |
  | (fixed, not configurable) | `detail_reserves_resources` | scraper's `category='Total Reserves & Resources'`, `Contained(...)` rows — auto-derives `commodity_shares` (see below) |
  | (fixed, not configurable) | `mine_property_geometries` | scraper's real per-mine footprint polygons (`geometry_kind='property'`) — builds the `mine_polygons` table backing the `mine_polygon_count` GRID variable (see below) |
  | `commodity_shares_table` | `commodity_shares` | **override, not required**: if a real table exists here (the original "user-owned" contract), it's copied as-is; otherwise `commodity_shares` is auto-derived from `detail_reserves_resources` (`_create_commodity_shares_table`) — `(property_id VARCHAR, commodity VARCHAR, share DOUBLE)`, one row per `(property_id, commodity)`, `commodity` normalized via `src.data.sources.commodities.normalize_commodity(..., source="snl")`, converted to a common tonnes basis (`_CONTAINED_TONNES_PER_UNIT`: oz troy / ct metric carat / lbs avoirdupois / tonnes — verified each of the ~63 real commodity labels uses exactly one of these four units consistently, no per-commodity mixing). Auto-derived coverage on real data: 4,767 mines. If neither an override nor derivable reserves data exists, `mine_priceshock_*` rasterizes as all-`NaN` (warning logged, not an error). |

  **TODO (needs live data):** exact column list/dtypes of `properties`
  beyond the columns referenced by code above — inspect the live stage-0
  DuckDB (`DESCRIBE properties;`) for the full schema (S&P export fields:
  commodity, operator, ownership %, etc. are present but not enumerated in
  code since PREPARE only touches the columns it needs).

**Identity/location/closing-year fusion** (`_fusion_ctes`,
`_fused_mines_from_clause`, `_fused_latitude_expr`/`_fused_longitude_expr`/
`_fused_closing_year_expr`, shared by `_determine_year_bounds` and
`_create_active_mines_table` so the two can't independently drift): verified
against real data that manual `properties.actual_start_up_year`/
`actual_closure_year` are populated for only 15% / 2% of mines — this is
presumably why the LLM-imputation fallback tier existed in the first place.
The scraped `'Actual Closure'` milestone tier alone fills 378 additional
mines' closing year on the current real database that neither manual data
nor (previously) any other source had.

### LLM year-imputation script

`scripts/run_snl_mining_imputation.py` / `src/data/sources/snl_mining/
imputation.py` (converted from `notebooks/snl_mining_openai_enrichment.ipynb`,
which is kept for interactive one-off probing/debugging only). Not a
pipeline `STEPS` member for the same reason `fetch` is absent — a genuinely
async, hours-long external OpenAI Batch API call, and `PipelineStep` only
has `fetch`/`prepare`/`grid`.

- **Text source**: `imputation.load_fused_property_texts()` — prefers the
  scraper's `detail_work_history_events` (concatenated `event_text` in
  `event_sequence` order) over the manual `property_texts.full_work_history`
  field, falling back to the latter only for mines with no scraped text.
  Verified against real data: for the 20,674 mine_ids with both, the scraped
  reconstruction matches the manual narrative (38% byte-identical, the rest
  differing only by re-join whitespace) — but the manual field is capped at
  Excel's ~32,767-char cell limit (203 real mines hit it), while the scraped
  reconstruction is not, so long-history mines get materially more complete
  input text than before.
- **Output**: unchanged target, DuckDB table `property_llm_years` inside
  `database.duckdb` (delete-then-insert keyed by `property_id`), plus a CSV
  export to `csv/property_llm_years.csv`.
- **Usage**: `python scripts/run_snl_mining_imputation.py probe` (one live
  sanity-check request) or `... run [--watch] [--overwrite]` (create/advance
  the batch manifest; `--watch` blocks and polls until the queue drains,
  the default is a single non-blocking pass suitable for periodic
  re-invocation).

### PREPARE

`_execute_prepare()` in `source.py`. Builds a second, derived DuckDB
(`prepared_db_path`) by `ATTACH`ing the stage-0 DB read-only and running a
sequence of `CREATE OR REPLACE TABLE ... AS` spatial queries (DuckDB
`spatial` extension). Independently resumable/retriable without re-touching
stage 0 (module docstring: this is the one place the pipeline migration
genuinely improved resumability over the old combined `stage="spatial"`
preprocessor).

- **Output path**: `<output_root(prepare)>/snl_mining_prepared.duckdb`
  (legacy: `<data_root>/snl_mining/processed/stage_1/...`; v2:
  `<data_root>/prepared/snl_mining/...`). Overridable via
  `sources.snl_mining.prepared_db_path` / `aggregation.prepared_db_path`.
- **Format**: DuckDB database (`Completion.PATH_EXISTS` — single-file,
  atomic-write semantics don't apply the same way as a directory store, so
  completion is a bare existence check).
- **Also requires** (read directly via `layout.output_root()`, not via
  `REQUIRES`-injected paths):
  - `gadm` PREPARE output: `gadm_levelADM_1_simplified.gpkg` /
    `gadm_levelADM_2_simplified.gpkg` under
    `<gadm PREPARE output root>` (namespace `gadm`) — admin polygon
    geometries for the `mine_count_adm1`/`adm2` tables.
  - `commodity_prices` PREPARE output:
    `<commodity_prices PREPARE output root>/commodity_prices.parquet` — a
    `(commodity, year, ln_price_real)` lookup, joined to build
    `mine_priceshock`.
- **Tables written** (all inside `prepared_db_path`; `commodity_shares` is
  also written back to the shared `database.duckdb` + CSV, see below):
  | table | grain | columns |
  |---|---|---|
  | `active_mines` | `(property_id, year)` | `longitude`, `latitude`, `opening_year`, `closing_year`, `point_wgs84`, `point_metric` (in `metric_crs`, default `ESRI:54009` Mollweide), `point_raster` (in the target raster CRS) — one row per mine per year it was active, expanded from `[opening_year, closing_year]` via `LATERAL range()`, clamped to the PREPARE-time-detected/configured year bounds |
  | `commodity_shares` | `(property_id, commodity)` | `share` in `[0, 1]`, summing to `1` per mine — see the fused/auto-derived description above |
  | `mine_priceshock` | `(property_id, year)` | `value = SUM(share * ln_price_real)` over `commodity_shares` joined to the price lookup; `NULL` (not `0`) when no commodity for that mine-year has a price match |
  | `mine_buffers_10km`, `_20km`, `_50km` | `(property_id, year)` | `value` (constant `1`, i.e. presence), `value_priceshock` (from `mine_priceshock`), `geometry_metric` (a `ST_Buffer` circle of the given radius around `point_metric`), `geometry_raster` (same circle reprojected to raster CRS) — one physical table per radius, shared by both the count and price-shock output variables |
  | `mine_polygons` | `(property_id, year)` | `value` (constant `1`), `geometry_raster` — real per-mine footprint polygon (`mine_property_geometries`, `geometry_kind='property'`) reprojected to raster CRS, reused for every active year of that mine (same "static location, one row per active year" shape as `mine_buffers_*`); a plain (not `LEFT`) `JOIN` against `active_mines` means mines without a `property`-kind polygon (~43% on real data) simply don't contribute a row, not a null/zero one |
  | `adm1_year_counts`, `adm2_year_counts` | `(year, adm_code)` | `value` = count of active mines whose point falls inside that `GID_1`/`GID_2` polygon that year, plus `geometry_raster` |

  R-tree spatial indexes (`idx_<table>_rtree`) are built on every
  `geometry_raster` column above (`_create_rtree_indexes`); a best-effort
  `EXPLAIN` check confirms the planner actually uses them
  (`_verify_rtree_queries`, logs only, never fails the stage).

  **Year-bounds detection**: if `sources.snl_mining.year_range` isn't set in
  config, PREPARE infers `[start_year, end_year]` from `MIN`/`MAX` of
  `COALESCE(opening_year, llm_opening_year)` / closing-year, clamped to a
  shared plausible-year sanity window (`src.data.common.years`
  `MIN_PLAUSIBLE_YEAR`/`MAX_PLAUSIBLE_YEAR`) — guards against a garbled
  source value (e.g. `"150"` typo'd for `"1950"`) reaching `to_zarr()`'s
  CF-time auto-region-detection at GRID time and crashing there instead,
  far more cryptically (confirmed on real data per the code comment).
  Excluded rows are logged, not dropped — their own year is still clamped
  into range when building `active_mines`.

  **TODO (needs live data):** actual inferred `[start_year, end_year]` for
  the current stage-0 export, and how many properties get excluded by the
  plausible-year guard.

### GRID

`_execute_grid()`. Two distinct output shapes from one PREPARE run:

**1. Rasterized zarr store** (genuinely per-pixel variables only):
- **Output path**: `layout.grid_store_path(..., v2_family="snl_mining")`.
  - v2: `<data_root>/grid/<grid_id>/snl_mining.zarr`
  - legacy: `<data_root>/snl_mining/processed/stage_2[_ease6933]/snl_mining_timeseries_reprojected.zarr`
    (filename configurable via `aggregation.output_filename`)
- **Format**: Zarr v3 store (`zarr_format=3`, `consolidated=False`), Blosc/
  zstd-compressed, chunked `(1, 1, tile_size, tile_size)` (`tile_size`
  default 2048, configurable). `Completion.MARKER` (sibling `.complete`
  file — directory output).
- **Dims**: `time` (one coordinate per active year, `f"{year}-12-31"`),
  `band` (always `[1]`), plus the target grid's spatial dims (grid CRS from
  `pipeline.grid` in `data.yaml`: `ease6933` or `legacy_4326`).
- **Variables** (`aggregation.output_variables`, default all seven below):

  | variable | dtype | fill/nodata | source table.column | meaning |
  |---|---|---|---|---|
  | `mine_count_10km` | `uint16` | `0` | `mine_buffers_10km.value` | count of active mine buffers (10 km radius) covering the pixel center |
  | `mine_count_20km` | `uint16` | `0` | `mine_buffers_20km.value` | same, 20 km radius |
  | `mine_count_50km` | `uint16` | `0` | `mine_buffers_50km.value` | same, 50 km radius |
  | `mine_priceshock_10km` | `float32` | `NaN` | `mine_buffers_10km.value_priceshock` | sum of `share * ln(real price)` over active, price-matched mine buffers (10 km) covering the pixel center |
  | `mine_priceshock_20km` | `float32` | `NaN` | `mine_buffers_20km.value_priceshock` | same, 20 km |
  | `mine_priceshock_50km` | `float32` | `NaN` | `mine_buffers_50km.value_priceshock` | same, 50 km |
  | `mine_polygon_count` | `uint16` | `0` | `mine_polygons.value` | count of real mine footprint polygons (`geometry_kind='property'`) covering the pixel center — unlike `mine_count_*`, not a fixed radius: the pixel is only counted if it falls inside the mine's actual scraped boundary |

  `mine_count_*` and `mine_priceshock_*` share the same physical
  `mine_buffers_{R}km` table per radius — only `value_column` differs.
  `mine_count_*`'s `0` is a legitimate "no mine nearby" value; a
  `mine_priceshock_*` pixel untouched by any priced buffer is `NaN`, not
  `0`, because `0` is itself a legitimate summed price-shock value (a
  covering mine whose commodities all had `ln(real price) == 0`) —
  deliberately not conflated with "no coverage". `mine_polygon_count` has no
  `radius_km` in its `aggregation.radius_variables` entry — that's the
  discriminator `_execute_prepare` uses to route it through
  `_create_polygon_table` (real geometry) instead of `_create_buffer_table`
  (`ST_Buffer` circles); every other consumer of `buffer_tables` (rasterize,
  fetch, rtree indexing, empty-zarr creation) is already radius-agnostic.

  **Verification** (`src/data/sources/verify.py`, config
  `sources.snl_mining.verification`): `expected_vars` = all seven above;
  `value_range = [0, 200]` applies to the `uint16` count-like family
  (`range_vars`: `mine_count_*` and `mine_polygon_count` — a pixel covered
  by more than one overlapping property polygon is rare but not impossible,
  so the same generous bound is reused); the float32 price-shock family
  only gets the unconditional "sample isn't entirely NaN" check (different
  physical scale — a sum of log-prices, not a count).

**2. Admin-polygon count sidecars** (constant within a `GID_1`/`GID_2`
polygon for a given year, so *not* rasterized — `_export_admin_count_tables`):
- **Output path**: `<same directory as the zarr store above>/mine_count_adm1.parquet`,
  `.../mine_count_adm2.parquet`.
- **Format**: parquet, one file per admin level.
- **Schema**: `(GID_1 | GID_2: int, year: int, mine_count_adm1 | mine_count_adm2: int)`
  — `adm_code` (the GADM string code, e.g. `"USA.5_1"`) is translated to
  gadm's integer id via the `GID_1`/`GID_2` mapping JSON gadm's own GRID
  step writes (`src.data.sources.misc.gadm.gid_mapping_path`); rows whose
  code isn't in that mapping (id `0`) are dropped.
- **Consumption**: merged onto rows during assembly via
  `src.data.assemble.processors.TileProcessor`'s `join_on` mechanism — not
  read by anything in this source itself after being written.

**TODO (needs live data):** actual per-variable value ranges observed
(config only declares the *expected* `[0, 200]` bound for counts), zarr
store size on disk, and confirmation of which `grid_id`/`layout` combination
the current production run actually used.

---

## Part B — detail scraper (standalone, not pipeline-wired)

`src/data/sources/snl_mining/scraper/`, driven via
`scripts/debug_snl_mining_scraper.py <step>` (Selenium, needs a logged-in
Capital IQ session — `orchestration/secrets/spglobal.credentials.json` by
default). Writes into the same `data/raw/snl_mining/database.duckdb` as
Part A (`src/data/sources/snl_mining/scraper/config.py`'s `DEFAULT_DB_PATH`;
deliberately under the gitignored `/data` convention, not `ctx.layout`, since
this tool is standalone and has no `PipelineContext`). Raw downloaded
`.xlsx` exports land under `data/raw/snl_mining/scraping/` (`EXPORT_DIR`).
Chromedriver logs are centralized under `data/raw/snl_mining/logs/`;
ephemeral Selenium browser-profile dirs live outside `data/` entirely, under
`scratch_nobackup/snl_mining/browser_profiles/`.

Four stages (`scraper/stages/names.py::Stage`), each gated by a
`mines.<stage>_completed_at` timestamp column and (for the two `detail_*`
stages) a per-`(mine_id, section_label, subsection_label)` row in
`mine_subsection_stage_status`:

### `ids`

Scrapes the Capital IQ screener result list, paginating via `screener_state`
(tracks `total_pages`/`last_page_done` per `screener_key` so a killed run
resumes mid-pagination). Writes:
- `mines(mine_id PK, id_scraped_at, ...)` — one row per discovered mine id.
- `screener_state(screener_key PK, total_pages, last_page_done, started_at, completed_at)`.

### `detail_exports`

Visits each mine's Capital IQ profile page, discovers every sidebar
subsection, and downloads one `.xlsx` export per subsection via the
"Export" button. Writes:
- `mine_subsections(mine_id, section_label, subsection_label, subsection_href, discovered_at)`
  — every subsection link found on the page, whether or not export
  succeeded.
- `mine_subsection_exports(mine_id, section_label, subsection_label, subsection_href, xls_path, xls_sha256, workbook_title, workbook_subtitle, primary_sheet_name, content_subsection_label, exported_at)`
  — one row per successfully downloaded export file; `workbook_title`/
  `content_subsection_label` are parsed from the exported workbook itself
  (see the `detail_parse` mismatch note below).
- `mine_property_geometries(mine_id, geometry_kind, geometry_wkt, bounds_minx/y, bounds_maxx/y, extracted_at)`
  — property boundary/point geometry scraped from the Property Profile
  page's embedded map, WKT + bounding box. `geometry_kind='property'`
  (22,139 mines on real data, median footprint ~300m across) backs Part A's
  `mine_polygon_count` GRID variable (see "Tables written"/"Variables"
  above); `geometry_kind='linked'` (broader aggregated-claims polygons,
  7,126 mines) is out of scope for now.
- `mine_subsection_stage_status(..., stage_name='detail_exports', status, ...)`.

**Known data-quality issue (real, quantified against the local DB — this is
what drove `detail_regularize`'s content-validation gate, see below):
`subsection_label` (what was requested/clicked) does not reliably match
`content_subsection_label` (what the exported workbook's own title says it
is)** — a page-navigation race between "click subsection link" and "click
Export" during scraping. Example: `Production` requested 7,630 times;
content matched `Capacity & Costs` ~13% of the time instead. Every
subsection type has a `content_subsection_label = NULL` tail in the
thousands (title extraction fails for some page layouts) on top of smaller
cross-type contamination. Downstream stages must not trust
`subsection_label` alone as ground truth for content.

### `detail_parse`

Re-opens each downloaded `.xlsx` (`parsing/xls.py::parse_subsection_xls`)
into a generic structural representation — layout, not semantics: every
cell is `TEXT`, tagged by `block_type` (`text`/`key_value`/`table`) and
`role` (`header`/`data`/`label`/`value`/`note`/`context`). Writes:
- `mine_subsection_blocks(mine_id, section_label, subsection_label, xls_path, xls_sha256, sheet_name, sheet_index, block_index, block_type, block_title, row_start, row_end, header_row_count, workbook_title, workbook_subtitle, primary_sheet_name, content_subsection_label, parsed_at)`
- `mine_subsection_block_cells(mine_id, section_label, subsection_label, xls_path, xls_sha256, sheet_name, sheet_index, block_index, block_type, block_title, row_number, column_index, column_name, cell_ref, cell_role, value, parsed_at)`
  — one row per cell, keyed by raw `(sheet, row, column)` position.
- `mine_subsection_cells` — an older, flatter cell table (`sheet_name,
  row_index, column_name, value`); still populated but superseded by
  `mine_subsection_block_cells` for anything block-structure-aware.
- `mine_subsection_stage_status(..., stage_name='detail_parse', ...)`.

This layer is deliberately generic and is the input `detail_regularize`
re-parses from the source `.xlsx` (not from these already-persisted rows —
see below).

### `detail_regularize`

Added by this project's own recent work (`scraper/regularize/`). Converts
the generic block/cell soup above into one clean, typed table per
real-world subsection type — 27 fixed types, mined from
`SELECT DISTINCT subsection_label FROM mine_subsection_exports` against
real scraped data (`scraper/regularize/registry.py`). A subsection label
outside this fixed list is a hard `unknown_type`, not a silent generic
fallback — extending it requires adding and registering a new regularizer
module (`scraper/regularize/subsections/*.py`).

**Content-validation gate** (runs before every regularizer, given the
`detail_exports` mismatch problem above): if the workbook has a title and
none of the requested type's `expected_title_fragments` appear in it (case-
insensitive), the label doesn't validate. If the workbook has no title at
all (~30-60% of real exports) → status `unverified`, regularized under the
requested label anyway but flagged distinctly. Otherwise (title present and
validates) → `completed`.

**Content-based reclassification** (`_match_by_title` in `registry.py`):
when the requested label doesn't validate — or isn't one of the 27 at all —
the title is checked against *every* registered type's fragments instead of
just the requested one, before giving up. If exactly one *function* among
all 27 matches, the row is regularized under that type instead and flagged
`reclassified` (distinct from `completed`, for auditability). Several types
share a function (e.g. `Ownership`/`Ownership Structure`,
`Capacity & Costs`/`Production`), so matching either isn't ambiguous; a
handful of other fragment overlaps span genuinely different functions (the
`Geology`/`Location, Map & Claims`/`Discoveries & Milestones`/`Drill
Results` cluster; `Modeled Ore Costs`/`Modeled Product Costs`; `Modeled ROM
Costs`/`Modeled Production`) and are left as `content_mismatch` rather than
guessed. This is also what lets `Cost Curve` exports — 100% stale
`Property Profile` content per the code comment, previously a guaranteed
mismatch — recover correctly. Final statuses land in
`mine_subsection_stage_status(stage_name='detail_regularize', status IN ('completed','reclassified','content_mismatch','unverified','unknown_type'))`.

**Output tables**: dynamic schema per table (`storage/regularized.py`,
`CREATE TABLE IF NOT EXISTS ... AS SELECT * FROM df WHERE 0=1`, inferred
from each regularizer's row dicts), delete-then-insert keyed by `mine_id`.
Every row carries `mine_id`, `xls_sha256`, `regularized_at` in addition to
its business fields (injected centrally by `persist_regularized_tables`).
One or more `detail_*` tables per subsection type:

| subsection label(s) | regularizer module | output table(s) |
|---|---|---|
| Property Profile | `property_profile.py` | `detail_property_profile__general`, `__owners`, `__claims_summary`, `__recent_news`, `__contained_reserves`, `__filings`, `__narrative` |
| Location, Map & Claims | `profile_tables.py` | `detail_location_map_claims__location`, `__claims` |
| Geology | `profile_tables.py` | `detail_geology` |
| Work History | `profile_narrative.py` | `detail_work_history_events` |
| Discoveries & Milestones | `profile_tables.py` | `detail_discoveries_milestones__discoveries`, `__milestones` |
| Drill Results | `profile_tables.py` | `detail_drill_results` |
| Development Studies | `profile_tables.py` | `detail_development_studies` |
| Capital Costs | `profile_tables.py` | `detail_capital_costs` |
| Subcontractors | `profile_tables.py` | `detail_subcontractors` |
| Comments | `profile_narrative.py` | `detail_comments__general`, `__bibliography` |
| Ownership, Ownership Structure | `ownership.py` | `detail_ownership__current`, `__former`, `__historical_equity`, `__historical_control`, `__royalty` |
| Capacity & Costs, Production | `capacity_costs.py` | `detail_capacity_costs__production`, `__cost_breakdown`, `__processing_details` |
| Reserves & Resources | `reserves.py` | `detail_reserves_resources` |
| Reserves / Resources & Production Chart | `reserves.py` | `detail_reserves_resources_production_chart` |
| Cash Flow Analysis | `mine_economics.py` | `detail_cash_flow_analysis` |
| Cost Curve | `mine_economics.py` | `detail_cost_curve` — export disabled scraper-side (`_TEMPORARILY_SKIPPED_SUBSECTIONS`); every real row filed under this label is stale content from whatever page was previously open, so `expected_title_fragments` can never match and every row resolves to `content_mismatch` by design |
| Modeled Ore Costs | `mine_economics.py` | `detail_modeled_ore_costs` |
| Modeled Product Costs | `mine_economics.py` | `detail_modeled_product_costs` |
| Modeled Production | `mine_economics.py` | `detail_modeled_production` |
| Modeled ROM Costs | `mine_economics.py` | `detail_modeled_rom_costs` |
| Financings | `financings.py` | `detail_financings` |
| M&A History | `m_a_history.py` | `detail_m_a_history` |
| Documents | `news_events_and_filings.py` | `detail_documents` |
| News | `news_events_and_filings.py` | `detail_news` |
| Events Calendar | `news_events_and_filings.py` | `detail_events_calendar` |

**Column-level schema per table**: not enumerated here — dynamic/inferred
from each regularizer's row dicts. Authoritative sources: the regularizer
function itself (`scraper/regularize/subsections/*.py`); the corresponding
unit test builder in `tests/data/sources/snl_mining/scraper/regularize/test_*.py`;
or, for the full column list + type + current row count of every `detail_*`
table against live data, `data/raw/snl_mining/csv/README.md` (regenerated
alongside the CSV export below).

**Parallelism**: parsing + classification runs across a `ProcessPoolExecutor`
when `max_workers > 1` (default 8, `--workers N` on
`scripts/debug_snl_mining_scraper.py`) — `parsing/xls.py` hand-parses each
workbook's XML (`zipfile` + `xml.etree.ElementTree`, one dataclass built per
cell), which is pure-Python CPU work, not blocking I/O; confirmed
empirically that `ThreadPoolExecutor` gave no real speedup here (GIL
contention on that Python-level parsing), unlike the thread-pooled
push/pull transfer code elsewhere in this repo. `max_workers <= 1` skips
the pool and runs in-process (used by tests, since a monkeypatch can't
reach a spawned worker process). Every DuckDB write (persisting regularized
tables, stage-status upserts, `mark_stage_complete`) stays serialized on
the main process, since a single `DuckDBPyConnection` isn't safe for
concurrent use and never crosses the process boundary. A `tqdm` progress
bar wraps the result-consumption loop, showing live progress over the flat
export-file count.

The worker function itself lives in a dedicated `stages/_regularize_worker.py`
module with a deliberately stdlib-only import graph, not in
`regularize_detail_exports.py` — every one of `max_workers` freshly-spawned
processes re-imports whatever module the worker function lives in from
scratch, and importing *any* `src.data.sources.snl_mining.*` submodule
first runs the package's own `__init__.py`. That `__init__.py` is now lazy
(PEP 562: `SnlMiningSource` is only imported on first attribute access) for
exactly this reason — it used to eagerly import `.source`, which pulls in
the full pipeline stack (`geobox` → `pandas` → `pyarrow`) into every worker
process just to run a few hundred bytes of XML parsing. On a real
332k-row run this was severe enough to look like a hang (worker processes
visibly started, but zero progress) rather than just added latency.
`pool.map(..., chunksize=...)` is also set explicitly (capped at 200) rather
than left at the default of 1, since one full IPC round-trip per row is
significant overhead relative to how little work each row actually does at
that scale.

Row counts per `detail_*` table for the current full scraper run range from
45 (`detail_events_calendar`) to 337,380 (`detail_modeled_product_costs`) —
see `data/raw/snl_mining/csv/README.md` for the exact count per table.

**TODO (needs live data):** the `completed`/`reclassified`/`content_mismatch`/
`unverified`/`unknown_type` status breakdown for the current full scraper
run — query `mine_subsection_stage_status WHERE stage_name='detail_regularize'`
against the live `database.duckdb`.

### Optional CSV export

Not a stage — a post-hoc dump. `storage/regularized.py::
export_regularized_tables_to_csv()`, triggered via
`scripts/debug_snl_mining_scraper.py ... --csv-out data/raw/snl_mining/csv`:
one `<dir>/<name>.csv` per `detail_*` table via DuckDB's native `COPY ... TO`
(no pandas round-trip). The `detail_` prefix is dropped from the CSV
filename (`detail_ownership__current` → `ownership__current.csv`) — the
table name inside the database keeps it. `data/raw/snl_mining/csv/README.md`
documents each CSV's columns/types/row-count and is regenerated by hand
alongside the export (not by the export function itself).

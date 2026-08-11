# snl_mining — SNL/S&P Global mining property tables

Registry id `snl_mining` (`src/data/sources/snl_mining/source.py`, class
`SnlMiningSource`). Steps: **prepare, grid** — `fetch` is deliberately absent
(module docstring): acquisition is a manual S&P Capital IQ `.xls` export plus
an OpenAI batch-enrichment loop that isn't automatable, not a gap. `REQUIRES`:
`gadm` prepare + grid, `commodity_prices` prepare.

Two genuinely separate subsystems live under `src/data/sources/snl_mining/`:

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

---

## Part A — pipeline source (prepare, grid)

### Stage 0 — manual input (pre-pipeline, not a `STEPS` member)

Not produced by this codebase's `fetch` (there is none); produced by
`src/data/sources/snl_mining/notebooks/snl_mining_manual_xls_to_duckdb.ipynb`
(a manual S&P `.xls` export + OpenAI batch-enrichment loop, per the module
docstring) and consumed as this source's raw input.

- **Path**: `<output_root(fetch)>/manual_xls/snl_mining_manual_export.duckdb`
  (`output_root(fetch)` = `layout.raw_root()` — legacy:
  `<data_root>/snl_mining/raw/manual_xls/...`; v2:
  `<data_root>/raw/snl_mining/manual_xls/...`). Overridable via
  `sources.snl_mining.duckdb_path` in `data.yaml`.
- **Format**: DuckDB database.
- **Tables consumed by PREPARE** (names configurable in `data.yaml`, defaults
  shown):
  | table (config key) | default name | purpose |
  |---|---|---|
  | `properties_table` | `properties` | one row per mine/property; must include `property_id`, `latitude`, `longitude`, `actual_start_up_year` (opening year), `actual_closure_year` (closing year, nullable = still active) |
  | `llm_years_table` | `property_llm_years` | LLM-imputed fallback opening/closing years (`llm_opening_year`, `llm_closing_year`), used via `COALESCE` when the observed columns are null; optional — PREPARE logs a warning and falls back to observed-only years if absent |
  | `work_history_table` | `property_work_history_events` | narrative work-history events (declared in config; not read by `_execute_prepare` directly in the current code path) |
  | `commodity_shares_table` | `commodity_shares` | **user-owned**, not produced by the ingestion notebook: `(property_id VARCHAR, commodity VARCHAR, share DOUBLE)`, one row per `(property_id, commodity)`, static across a mine's active years, `commodity` pre-normalized via `src.data.sources.commodities.normalize_commodity(..., source="snl")`. If missing, `mine_priceshock_*` rasterizes as all-`NaN` (warning logged, not an error). |

  **TODO (needs live data):** exact column list/dtypes of `properties`
  beyond the columns referenced by code above — inspect the live stage-0
  DuckDB (`DESCRIBE properties;`) for the full schema (S&P export fields:
  commodity, operator, ownership %, etc. are present but not enumerated in
  code since PREPARE only touches the columns it needs).

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
- **Tables written** (all inside `prepared_db_path`):
  | table | grain | columns |
  |---|---|---|
  | `active_mines` | `(property_id, year)` | `longitude`, `latitude`, `opening_year`, `closing_year`, `point_wgs84`, `point_metric` (in `metric_crs`, default `ESRI:54009` Mollweide), `point_raster` (in the target raster CRS) — one row per mine per year it was active, expanded from `[opening_year, closing_year]` via `LATERAL range()`, clamped to the PREPARE-time-detected/configured year bounds |
  | `mine_priceshock` | `(property_id, year)` | `value = SUM(share * ln_price_real)` over `commodity_shares` joined to the price lookup; `NULL` (not `0`) when no commodity for that mine-year has a price match |
  | `mine_buffers_10km`, `_20km`, `_50km` | `(property_id, year)` | `value` (constant `1`, i.e. presence), `value_priceshock` (from `mine_priceshock`), `geometry_metric` (a `ST_Buffer` circle of the given radius around `point_metric`), `geometry_raster` (same circle reprojected to raster CRS) — one physical table per radius, shared by both the count and price-shock output variables |
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
- **Variables** (`aggregation.output_variables`, default all six below):

  | variable | dtype | fill/nodata | source table.column | meaning |
  |---|---|---|---|---|
  | `mine_count_10km` | `uint16` | `0` | `mine_buffers_10km.value` | count of active mine buffers (10 km radius) covering the pixel center |
  | `mine_count_20km` | `uint16` | `0` | `mine_buffers_20km.value` | same, 20 km radius |
  | `mine_count_50km` | `uint16` | `0` | `mine_buffers_50km.value` | same, 50 km radius |
  | `mine_priceshock_10km` | `float32` | `NaN` | `mine_buffers_10km.value_priceshock` | sum of `share * ln(real price)` over active, price-matched mine buffers (10 km) covering the pixel center |
  | `mine_priceshock_20km` | `float32` | `NaN` | `mine_buffers_20km.value_priceshock` | same, 20 km |
  | `mine_priceshock_50km` | `float32` | `NaN` | `mine_buffers_50km.value_priceshock` | same, 50 km |

  `mine_count_*` and `mine_priceshock_*` share the same physical
  `mine_buffers_{R}km` table per radius — only `value_column` differs.
  `mine_count_*`'s `0` is a legitimate "no mine nearby" value; a
  `mine_priceshock_*` pixel untouched by any priced buffer is `NaN`, not
  `0`, because `0` is itself a legitimate summed price-shock value (a
  covering mine whose commodities all had `ln(real price) == 0`) —
  deliberately not conflated with "no coverage".

  **Verification** (`src/data/sources/verify.py`, config
  `sources.snl_mining.verification`): `expected_vars` = all six above;
  `value_range = [0, 200]` applies only to the `uint16` count family
  (`range_vars`); the float32 price-shock family only gets the
  unconditional "sample isn't entirely NaN" check (different physical
  scale — a sum of log-prices, not a count).

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
default). Entirely separate storage from Part A: writes to its own DuckDB at
`data/snl_mining/scraper/snl_mining_scraper.duckdb` (`src/data/sources/
snl_mining/scraper/config.py`; deliberately under the gitignored `/data`
convention, not `ctx.layout`, to avoid colliding with Part A's pipeline
paths). Raw downloaded `.xlsx` exports land under
`data/snl_mining/scraper/exports/`.

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
  page's embedded map, WKT + bounding box.
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
none of the type's `expected_title_fragments` appear in it (case-
insensitive) → status `content_mismatch`, not regularized. If the workbook
has no title at all (~30-60% of real exports) → status `unverified`,
regularized anyway but flagged distinctly. Otherwise → `completed`. Status
lands in `mine_subsection_stage_status(stage_name='detail_regularize', status IN ('completed','content_mismatch','unverified','unknown_type'))`.

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
function itself (`scraper/regularize/subsections/*.py`) or, for a
concrete example per table shape, the corresponding unit test builder in
`tests/data/sources/snl_mining/scraper/regularize/test_*.py`.

**TODO (needs live data):** row counts per `detail_*` table and the
`completed`/`content_mismatch`/`unverified`/`unknown_type` status
breakdown for the current full scraper run — query
`mine_subsection_stage_status WHERE stage_name='detail_regularize'` against
the live `snl_mining_scraper.duckdb`.

### Optional CSV export

Not a stage — a post-hoc dump. `storage/regularized.py::
export_regularized_tables_to_csv()`, triggered via
`scripts/debug_snl_mining_scraper.py ... --csv-out <dir>`: one
`<dir>/<table_name>.csv` per `detail_*` table via DuckDB's native
`COPY ... TO` (no pandas round-trip).

# snl_mining — stage-0 inputs and the detail scraper

Companion to [`README.md`](README.md). This page documents the parts of `snl_mining` that sit
*before* the pipeline's PREPARE step: the hand-built stage-0 DuckDB, how its sources are fused into
mine identity, location and timing, the LLM year imputation, and the standalone Capital IQ detail
scraper.

> Carried over from the earlier `snl_mining` page (written 2026-08-11/12, when the pipeline still
> had a separate GRID step), with only the step names updated. Row counts and coverage figures
> were measured on the real database at that time; re-check them after a new export or scrape.
> The PREPARE/GRID sections of that page described outputs that no longer exist, so they are
> replaced by [`README.md`](README.md).

Two code paths share one storage tree and one DuckDB file:

- **Part A**: the pipeline side, meaning the stage-0 tables PREPARE reads and the LLM year
  imputation.
- **Part B**: the standalone Selenium detail scraper (`scripts/debug_snl_mining_scraper.py`).
  It is not a pipeline step, but it writes durable tables that PREPARE's fusion reads.

## Storage

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

## Stage 0 — manual input and fusion

Not produced by this codebase's `fetch` (there is none); produced by
`src/data/sources/snl_mining/notebooks/snl_mining_manual_xls_to_duckdb.ipynb`
(a manual S&P `.xls` export) plus `scripts/run_snl_mining_imputation.py`
(`src/data/sources/snl_mining/imputation.py` — an OpenAI batch-enrichment
job, converted from a notebook into a script; see "LLM year-imputation
script" below) and consumed as this source's raw input.

- **Path**: `<output_root(fetch)>/database.duckdb` (`output_root(fetch)` =
  `layout.raw_root()` — `<data_root>/raw/snl_mining/database.duckdb`). Overridable via
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
  | (fixed, not configurable) | `detail_reserves_resources` | scraper's `category='Total Reserves & Resources'`, `Contained(...)` rows — auto-derives `commodity_shares` (see the last row) |
  | (fixed, not configurable) | `mine_property_geometries` | scraper's real per-mine footprint polygons (`geometry_kind='property'`) — builds the `mine_polygons` table backing the `mine_polygon_count` pixel-grid variable ([`README.md`](README.md)) |
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
async, hours-long external OpenAI Batch API call, and FETCH promises
unattended, per-file resumability.

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


## Part B — detail scraper (standalone, not pipeline-wired)

`src/data/sources/snl_mining/scraper/`, driven via
`scripts/debug_snl_mining_scraper.py <step>` (Selenium, needs a logged-in
Capital IQ session — `orchestration/secrets/spglobal.credentials.json` by
default). Writes into the same `data/raw/snl_mining/database.duckdb` as
Part A (`src/data/sources/snl_mining/scraper/config.py`'s `DEFAULT_DB_PATH`;
deliberately under the gitignored `/data` convention, not the pipeline's own
`layout.py`, since this tool is standalone and has no `PipelineContext`). Raw downloaded
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
  `mine_polygon_count` pixel-grid variable (see "Tables written"/"Variables"
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


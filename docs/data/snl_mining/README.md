# snl_mining — S&P Global (SNL) mine properties → mine exposure instruments

| | |
|---|---|
| Config key | `snl_mining` |
| Module | `src/data/sources/snl_mining/source.py` (`SnlMiningSource`); scraper in `snl_mining/scraper/`; LLM year imputation in `snl_mining/imputation.py` |
| Steps | PREPARE only; the input is built by hand, so there is no FETCH |
| Requires | `gadm` PREPARE (admin polygons, `GID_N` id mappings), `commodity_prices` PREPARE |
| In panel | pixel grid → `assembly.sources.snl_mining` |

## What it is

Mine locations, footprints, opening and closing years, and commodity shares for S&P Global
(Capital IQ / SNL) mining properties. It is the basis for the project's mining instruments
(counts of active mines near a cell, and a Berman et al. (2017)-style price-shock exposure).

## Input: the stage-0 DuckDB (manual)

`raw/snl_mining/database.duckdb` is one merged database, filled by three routes:

1. `snl_mining/notebooks/snl_mining_manual_xls_to_duckdb.ipynb`: the manual S&P `.xls` export,
   written to `properties`, `property_texts`, `property_work_history_events`, etc.
2. `scripts/run_snl_mining_imputation.py`: OpenAI batch extraction of opening and closing years
   from work-history text, written to `property_llm_years` (see
   [`notebooks/README.md`](../../../src/data/sources/snl_mining/notebooks/README.md)).
3. The Capital IQ detail scraper (`scripts/debug_snl_mining_scraper.py`, interactive, needs a
   browser session), which adds `detail_*`, `mines` and `mine_property_geometries` (footprint
   polygons).

Each mine's static commodity mix is `commodity_shares` `(property_id, commodity, share)`. A
user-supplied table overrides it. Otherwise it is auto-derived from the scraper's contained-metal
reserves and resources (`detail_reserves_resources`, converted to tonnes), which covered about
4,767 mines on the 2026-08 database. Commodity names are normalised by
`src/data/sources/commodities.py`.

### Fusion

PREPARE fuses the manual and scraped tables into one mine identity, location and timing:

- **Identity:** the scraper's `mines` table is the backbone.
- **Location:** manual coordinates, else the scraped `decimal_degrees`.
- **Closing year:** manual, else the scraped `'Actual Closure'` milestone, else LLM-imputed.
- **Opening year:** manual, else LLM-imputed. There is no scraped opening milestone.

Manual opening and closing years exist for only about **15 % / 2 %** of mines, so most mine
timing comes from the LLM imputation. Details, table by table:
[`inputs-and-scraper.md`](inputs-and-scraper.md).

## Prepared output (PREPARE)

**Phase 1** builds `prepared/snl_mining/misc/snl_mining_prepared.duckdb`:

- An `active_mines` table: one row per (mine, year) with `opening_year ≤ year ≤ closing_year`.
  Opening year is manual, else LLM-imputed. Closing year is manual, else scraped, else
  LLM-imputed; a missing closing year means the mine is still active.
- Buffer tables at 10/20/50 km, built in a metric CRS (`ESRI:54009`).
- Footprint polygon tables.

**Phase 2** rasterizes these onto the EASE grid, per year.

- **Path:** `prepared/snl_mining/crs/ease6933/snl_mining/ix=/iy=/part-<year>.parquet` (annual)

| Column | Meaning | Missing | Panel aggregation |
|---|---|---|---|
| `mine_count_{10,20,50}km` | active mines whose R-km buffer covers the pixel centre (uint16) | 0 | `average` |
| `mine_priceshock_{10,20,50}km` | Σ over those mines of `share × ln(real price)` (float32) | NaN where no price-matched mine | `average` |
| `mine_polygon_count` | active mines whose footprint polygon covers the pixel centre | 0 | `average` |

`mine_count_adm1`/`mine_count_adm2` (mines per admin unit and year) are written as `GID_N`-keyed
parquet sidecars (`_export_admin_count_tables`). They are not in the panel.

## Analysis caveats

- **Footprints cover only part of the sample.** Only active mine-years that have a scraped
  `property`-kind polygon contribute to `mine_polygon_count`; the code notes about 43 % do not. A
  donut that excludes footprint pixels therefore excludes only some of the mine sites. Pair it
  with a distance-based exclusion around mine points.
- **The price shock is not a clean shift-share instrument.** It sums over mines active *that
  year*, so exposure moves with openings and closures, not only with world prices. A
  baseline-fixed-exposure variant would need to be built
  ([`../../analysis/final-analysis-plan.md`](../../analysis/final-analysis-plan.md) §3).
- **Opening and closing years are mostly LLM-imputed** (manual years for about 15 % / 2 % of
  mines). Timing errors blur event studies around openings, and the active-mine counts inherit
  them. A manual-years-only robustness check will be small; report the imputation status next
  to any event study.
- **Price-shock coverage is a subset.** Only mines with commodity shares (about 4,767) and a
  commodity that has a World Bank price series contribute. The rest contribute nothing, not zero
  (see [commodity_prices](../commodity_prices/README.md)).
- **Mine-count identification is rare.** About 5 % of 10 km pixel-years have a mine within 20 km
  (`descriptive_statistics.ipynb`).

## Needs live data

Active mines per year; the share of mine-years with a footprint polygon, by commodity; the share
with a price match.

# commodity_prices — World Bank Pink Sheet commodity prices

- **Registry id:** `commodity_prices`
- **Class:** `CommodityPricesSource` (`src/data/sources/commodity_prices/source.py`)
- **Aliases:** none
- **Steps implemented (`STEPS`):** `FETCH`, `PREPARE` — **no `GRID`**: a tiny (commodity, year) lookup table, not spatial.
- **`REQUIRES`:** none (`()`)
- **Config key in `data.yaml`:** `sources.commodity_prices` (active, not disabled)
  ```yaml
  commodity_prices:
    type: "commodity_prices"
    data_path: "commodity_prices"
    prices_url: "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Annual.xlsx"
    prices_name: "CMO-Historical-Data-Annual.xlsx"
    prices_path: "raw/commodity_prices/auxiliary/CMO-Historical-Data-Annual.xlsx"
  ```
  `namespace` is not set for this source, so no `/<namespace>` path segment applies to any step below.

Source data: the World Bank Commodity Markets "Pink Sheet" CMO Historical Data workbook, "Annual Prices (Real)" sheet — already published in constant 2010 US dollars per the module docstring (confirmed there by inspecting the sheet's own title cell), so no separate CPI-deflation step is applied here.

## FETCH

Uses `ConfiguredFilesFetchMixin` (same pattern as `gadm`/`osm`/`country_classifications`) to download the single configured file (`prices_url`/`prices_name`).

- **Output path**
  - legacy: `<data_root>/commodity_prices/raw/CMO-Historical-Data-Annual.xlsx`
  - v2: `<data_root>/raw/commodity_prices/CMO-Historical-Data-Annual.xlsx`
- **Format:** raw `.xlsx` workbook as downloaded, unchanged.
- **Caveats (from code/`data.yaml` comments):**
  - The World Bank Pink Sheet download URL (`thedocs.worldbank.org/.../related/CMO-Historical-Data-Annual.xlsx`) embeds a content hash + release date and **rotates roughly monthly** — per both the module docstring and the `data.yaml` comment, if FETCH starts 404ing this is expected maintenance, not a bug; the current link should be re-fetched from https://www.worldbank.org/en/research/commodity-markets and `prices_url` bumped.
  - `prices_path` (config key, currently set to `"raw/commodity_prices/auxiliary/CMO-Historical-Data-Annual.xlsx"`, relative to `data_root`) lets PREPARE read an **already-staged local copy directly, bypassing FETCH's own output entirely** — used today for a copy already present under that path. Removing the key makes PREPARE require FETCH's own output instead.
  - `Completion.NEVER`: the FETCH target always re-plans; `run_fetch` decides what's actually missing.
  - Requires `ctx.ssh_target` (an HPC/remote target) configured, else `_execute_fetch` logs a warning and returns `False`.

## PREPARE

Reads the raw workbook (via `prices_path` override if set, else FETCH's own output) with `read_and_normalize_prices()` (`src/data/sources/commodity_prices/prices.py`) and writes the normalized long table to parquet.

`read_and_normalize_prices()`:
- Reads the `"Annual Prices (Real)"` sheet with `header=6` (0-indexed row — row 7 in a 1-indexed spreadsheet view).
- Renames the first column to `year`; coerces it numeric and **drops any row whose `year` cell isn't a plain year value**, rather than relying on a fixed trailing-row count — per the code comment, the sheet has ~11 blank trailer rows after the last data year (2025 at time of writing) that would otherwise need a hardcoded row count that breaks on the next monthly republish.
- Melts to long form (`year`, `wb_column`, `price_real`), maps each `wb_column` to a canonical commodity key via `src.data.sources.commodities.normalize_commodity(..., source="worldbank")`, and **drops any column with no canonical mapping**.
- Coerces `price_real` numeric (the workbook codes missing/not-yet-started series as literal `"…"` rather than a blank cell — coerced to `NaN` and dropped, along with any other non-numeric cell, so nothing propagates a string into `np.log`), then drops non-positive prices.
- Computes `ln_price_real = log(price_real)`.

- **Output path**
  - legacy: `<data_root>/commodity_prices/processed/stage_1/commodity_prices.parquet`
  - v2: `<data_root>/prepared/commodity_prices/commodity_prices.parquet`
- **Format:** single parquet file, one row per (commodity, year), sorted by `(commodity, year)`.
- **Schema**

  | column | dtype | meaning |
  |---|---|---|
  | `commodity` | str | canonical commodity key (`normalize_commodity(..., source="worldbank")`) |
  | `year` | int | calendar year |
  | `price_real` | float | annual real price, constant 2010 USD, per the WB Pink Sheet's own sheet-title cell |
  | `ln_price_real` | float | `log(price_real)` |

- **Caveats:** `Completion.PATH_EXISTS` — skipped if the output file already exists and `cfg.override` is not set. Requires the raw file (FETCH output or `prices_path` override) to exist on disk, else `_plan_prepare` yields no targets. The 2010 base year differs from Berman et al.'s 2005 base year used elsewhere in this codebase; the module docstring argues this is immaterial for a `share * ln(price)` term since a uniform rebasing only shifts each commodity's series by a commodity-specific, time-invariant constant absorbed by any downstream fixed effect — not independently re-verified here.

## Consumers

No `GRID` step. This source is cross-referenced by `snl_mining`'s `PREPARE` step via its own `REQUIRES = (("commodity_prices", PipelineStep.PREPARE),)`, which reads `<commodity_prices PREPARE output>/commodity_prices.parquet` directly (via `layout.output_root(...)`, not a class import) and joins it against per-mine commodity shares to build the `mine_priceshock_*` variables. Not documented further here — see `docs/data/snl_mining/README.md`.

**TODO (needs live data):** actual commodity coverage / row count of a real `commodity_prices.parquet`, and how many World Bank series fail to map via `normalize_commodity` and get dropped, have not been verified against real output.

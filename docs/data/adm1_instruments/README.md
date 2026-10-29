# adm1_instruments — region-level instruments for the DOSE analysis

| | |
|---|---|
| Config key | `adm1_instruments` |
| Module | `src/data/sources/adm1_instruments.py` |
| Steps | PREPARE only · `REQUIRES` snl_mining, commodity_prices, plad, gadm (all PREPARE) |
| In panel | no; `(GID_1, year)` table for the fork's ADM1 analysis (docs/successors/dose-prespec.md §3a) |

## What it is

Three instruments per GADM 4.1 first-level region and year, 2002–2022, derived from other
sources. The rules are in `dose-prespec.md` §3a. The mechanics are in the module docstring.

- **`mine_count`:** SNL mines active in the region. It equals `snl_mining`'s
  `adm1_year_counts`, checked to the row.
- **`price_frozen`:** Σ share × ln(real World Bank price) over the mines active in 2001, so only
  world prices move it. `price_moving` uses each year's active mines instead.
- **`fav_share`:** the share of the year in which the country's leader (PLAD) was born in the
  region. `fav_any` and `fav_share_lag1` are variants.

## Prepared output (PREPARE)

Under `prepared/adm1_instruments/adm/`:

- `adm1_instruments.parquet`: every region × year, zeros where nothing is exposed.
- `adm1_price_exposure.parquet`: `GID_1`, `commodity`, `weight`, the baseline exposure by
  commodity (for leave-one-commodity-out checks).

## Caveats

- **Region aggregates only.** The per-mine assignment is not written out, because mine-level
  data is confidential (CLAUDE.md).
- **Price coverage.** Prices exist for 12 commodities: coal, copper, gold, iron ore, lead, nickel,
  phosphate, platinum, potash, silver, tin and zinc. Bauxite, uranium and diamonds have none.
  Most mines lack `commodity_shares` (714 of 3,857 active in 2001), so they get share 1 on
  their primary commodity.
- **Birthplaces.** 1,238 of 1,248 PLAD leaders with coordinates fall in a simplified GADM
  polygon, 3 more are snapped within 25 km, and 7 are dropped. 97.6 % land in the region PLAD's
  own code names; the rest reflect GADM 4.1 boundary changes (e.g. Bogotá split from
  Cundinamarca).
- Opening and closing years of SNL mines are mostly LLM-imputed (retired conclusion).

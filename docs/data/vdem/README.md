# vdem — V-Dem country-year indices

| | |
|---|---|
| Config key | `vdem` |
| Module | `src/data/sources/vdem.py` |
| Steps | FETCH, PREPARE · no `REQUIRES` |
| In panel | no; country-year table for the pollution–institutions successor (docs/successors/pollution-institutions-prespec.md) |

## What it is

Varieties of Democracy (V-Dem), version 16 (March 2026): expert-coded indices of political
institutions for 202 countries, 1789–2025. The pipeline keeps a few indices:

| Column | Meaning | Scale |
|---|---|---|
| `v2x_polyarchy` | electoral democracy | 0–1 |
| `v2x_libdem` | liberal democracy | 0–1 |
| `v2x_rule` | rule of law | 0–1 |
| `v2x_corr` | political corruption (higher = more corrupt) | 0–1 |
| `v2xcs_ccsi` | core civil society | 0–1 |
| `v2x_regime` | Regimes of the World: 0 closed autocracy, 1 electoral autocracy, 2 electoral democracy, 3 liberal democracy | category |

Add more columns through `variables:` in `data.yaml`.

## Raw data (FETCH)

`raw/vdem/vdem_v16.RData`, ~34 MB (proxy or transfer node). V-Dem's website puts its downloads
behind a form, so FETCH takes the full dataset from the V-Dem Institute's own R data package
(`vdeminstitute/vdemdata` on GitHub). The URL is pinned to the commit of the v16 release.

## Prepared output (PREPARE)

`prepared/vdem/misc/vdem_country_year.parquet`: `country_text_id`, `country_name`, `year` and the
kept indices. Reading `.RData` needs `pyreadr` (in `environment.yml`).

## Caveats

- **`country_text_id` is V-Dem's own code.** It equals ISO 3166 alpha-3, and so GADM `GID_0`, for
  almost every country. Of the 81 DOSE countries, only the Bahamas and `ANT` have no V-Dem series.
- **Institutions change slowly and differ mostly between countries.** A moderation analysis
  rests mainly on cross-country comparisons.
- Indices are posterior means of a measurement model. V-Dem also provides uncertainty intervals,
  which are not kept.

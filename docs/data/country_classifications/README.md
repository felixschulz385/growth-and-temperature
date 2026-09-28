# country_classifications — UNDP HDI and World Bank income groups

| | |
|---|---|
| Config key | `country_classifications` (`data_path: misc`) |
| Module | `src/data/sources/misc/country_classifications.py`; parsers in `misc/hdi.py`, `misc/worldbank.py` |
| Steps | FETCH, PREPARE |
| Requires | `gadm` PREPARE (`GID_0_code_mapping.json`) |
| In panel | yes, joined on `GID_0` |

## What it is

Country-level development classifications:

- UNDP Human Development Index bands (Low < 0.55, Medium < 0.70, High < 0.80, Very High), from
  the HDR 2025 composite-indices time series.
- World Bank income groups (Low, Lower-middle, Upper-middle, High), from the "Country Analytical
  History" sheet.

## Raw data (FETCH)

`raw/misc/country_classifications/`: the HDR `.csv` and the World Bank `.xlsx`.

## Prepared output (PREPARE)

- `prepared/misc/adm/country_classifications/classifications.parquet`: keyed by `iso3`; read
  directly by `src/analysis/subsets/`.
- `prepared/misc/adm/country_classifications/classifications_by_gid0.parquet`: the same table
  keyed by gadm's `GID_0`, which the assembly joins onto rows.

Columns are booleans `HDI_{LO,ME,HI,VH}_{year}` and `WB_{LO,LM,UM,HI}_{year}`, each a snapshot at
1991, 1999 and 2011 (the last observation at or before that year).

## Analysis caveats

The classifications are fixed snapshots, not annual series. Use them to define heterogeneity
groups at baseline (e.g. the 1999 or 2011 status), not as time-varying controls.

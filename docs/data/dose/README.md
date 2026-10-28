# dose — DOSE reported subnational economic output

| | |
|---|---|
| Config key | `dose` |
| Module | `src/data/sources/dose.py` |
| Steps | FETCH, PREPARE · `REQUIRES` gadm (PREPARE) |
| In panel | no; `(GID_1, year)` table for the fork's ADM1 analysis (docs/successors/dose-prespec.md) |

## What it is

Wenz et al. 2023 (*Sci Data* 10:425), v2.9 (Zenodo 13773040, 2024-09-17): gross regional product
and population for 1,661 first-level regions in 83 countries, 1953–2020, compiled from national
statistical offices, with no satellite input. Output is in local currency and USD, current and
2015 prices, and by sector (agriculture, manufacturing, services) where reported. It also carries
DOSE's own annual temperature `T_a` and precipitation `P_a` per region.

## Raw data (FETCH)

`raw/dose/DOSE_V2.9.csv`, ~15 MB (proxy or transfer node).

## Prepared output (PREPARE)

Under `prepared/dose/adm/`:

- `dose_gadm_match.parquet`: one row per DOSE region, with the GADM 4.1 `GID_1` it matched, the
  pipeline's integer `GID_1`, and how it matched (code, name, unmatched, ambiguous).
- `dose_gid1.parquet`: matched region-years, keyed by integer `GID_1` and `year`.

**GADM versions.** DOSE uses GADM 3.6 codes; the pipeline uses 4.1. As of 2026-10-28:

| Match | Regions |
|---|---|
| by code | 1,538 |
| by name, overriding or replacing the code | 58 |
| unmatched (dropped) | 58 |
| ambiguous (dropped) | 8 |

Name matches fix Chile, where `CHL.13_1` is Ñuble in 4.1 and Santiago is `CHL.14_1`, plus Colombia
and Kyiv. The unmatched regions are mostly in the Philippines, Morocco, Sri Lanka, North Macedonia
and Nepal.

## Analysis caveats

- **`T_a` is temperature.** It must never be a control in a regression with temperature as the
  outcome. `P_a` (precipitation) is the weather control.
- **Coverage leans to richer and larger countries.** Years differ by country, and many series end
  before 2020.
- **GRP counts output where it is produced.** A mine's value added enters its region's GRP even
  when the profits leave. A first stage of GRP on mines is therefore partly mechanical.
- Sector columns are often missing.

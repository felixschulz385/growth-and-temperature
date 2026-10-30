# ghs_pop — GHS-POP R2023A population grid

| | |
|---|---|
| Config key | `ghs_pop` |
| Module | `src/data/sources/ghs_pop.py` (`GhsPopSource`) |
| Steps | FETCH, PREPARE · no `REQUIRES` |
| In panel | listed in the assembly since PREPARE finished (2026-10-30); the 10 km panel has not been rebuilt with it |

## What it is

The European Commission JRC's Global Human Settlement Layer population grid, release R2023A.
- **Content:** residential population per cell at 1 km in World Mollweide (ESRI:54009).
- **Epochs:** five-year steps, 1975–2030. 2025 and 2030 are projections.
- **License:** CC BY 4.0.

A Mollweide cell is exactly 1 km², so the value is persons per km².

Used to weight regional means of PM2.5 and NO₂ by population
([`../../successors/pollution-data-options.md`](../../successors/pollution-data-options.md) §3.4).

## Raw data (FETCH)

`raw/ghs_pop/r2023a/`: one ZIP per epoch from the JRC open-data FTP, at
`GHS_POP_GLOBE_R2023A/GHS_POP_E<epoch>_GLOBE_R2023A_54009_1000/V1-0/GHS_POP_E<epoch>_GLOBE_R2023A_54009_1000_V1_0.zip`.
- **Contents:** each ZIP (about 300 MB) holds a Float64 GeoTIFF of 36,082 × 18,000 pixels with
  nodata −200.
- **Epochs:** `year_range` [1990, 2025] selects eight epochs.
- **Route:** FETCH runs on the transfer node (`orchestration/scripts/pollution-fetch.sh`).

## Prepared output (PREPARE)

- **Path:** `prepared/ghs_pop/r2023a/crs/ease6933/pop/ix=/iy=/part-<epoch>.parquet`
- **Column** `pop`: persons per km², read inside the ZIP (`/vsizip/`) and resampled onto the
  1 km EASE grid by `average`.
  - Both grids are equal-area, so averaging a density is area-correct.
  - Persons per EASE cell = `pop` × cell area (about 1.002 km² at 1 km; 100 × that at 10 km
    with `average` aggregation).
- **Epoch years only.** Annual values are interpolated in the analysis, not here.

## Analysis caveats

- **Modelled, not counted.** Census totals are spread over built-up area (GHS-BUILT).
  Population follows buildings, which correlates with night lights and with growth.
- **Epoch interpolation.** Linear or log-linear interpolation between epochs smooths out
  short-run changes. The 2025 epoch is a projection.
- **R2023A, not later releases.** A newer release would change past epochs too. Pin the
  release in any pre-spec.

## Needs live data

National totals against UN WPP as a sanity check; coverage over the EASE grid.

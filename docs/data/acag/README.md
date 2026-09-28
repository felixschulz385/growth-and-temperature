# acag — Surface PM2.5 (WashU ACAG)

| | |
|---|---|
| Config key | `acag` |
| Module | `src/data/sources/acag.py` (`AcagSource`) |
| Steps | FETCH, PREPARE · no `REQUIRES` |
| In panel | yes (`pm25`) |

## What it is

Annual mean surface PM2.5 concentration (µg/m³) from the Washington University Atmospheric
Composition Analysis Group: version V6GL02.04 (`CNNPM25`), 0.01° grid, 1998–2023. The estimates
combine satellite aerosol optical depth, a chemical transport model and ground monitors.

## Raw data (FETCH)

`raw/acag/pm25/GL/Annual/`: one NetCDF per year, downloaded from a Box shared folder using a
hardcoded file inventory (`KNOWN_FILES`). Adding a year is a code change.

## Prepared output (PREPARE)

- **Path:** `prepared/acag/pm25/crs/ease6933/pm25/ix=/iy=/part-<year>.parquet`
- **Column** `pm25` (float32; negative raw values set to NaN): resampled by nearest neighbour
  (0.01° ≈ 1.1 km); aggregated to coarser grids by `average`.

## Analysis caveats

- **This is the only direct measure of the aerosol channel.** Use it as an outcome of the
  treatment or as a baseline heterogeneity dimension. Don't use it as a control: it is a mediator.
- **It may share MODIS's clear-sky problem.** Satellite AOD is retrieved under clear skies only,
  so hazy periods may be under-represented in the same cells where MODIS LST loses observations.
  Check ACAG's documentation for a per-cell data-support measure.
- **The 2000 file is regional.** The inventory's 2000 entry is the Europe file
  (`V6GL02.04.CNNPM25.EU.200001-200012.nc`), not the global one. This is outside the 2002+ Aqua
  window, so it doesn't matter for the main analysis.

## Needs live data

The PM2.5 distribution by region; whether all 26 years prepared cleanly.

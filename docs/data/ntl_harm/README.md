# ntl_harm — Harmonized DMSP–VIIRS nighttime lights

| | |
|---|---|
| Config key | `ntl_harm` |
| Module | `src/data/sources/ntl_harm.py` (`NtlHarmSource`) |
| Steps | FETCH, PREPARE · no `REQUIRES` |
| In panel | yes (`ntl_harm`) |

## What it is

The harmonized global nighttime-lights series of Li et al. (2020), Figshare dataset `9828827`.
DMSP-OLS (1992–2013) is spliced with VIIRS-DNB (2014 on); VIIRS is converted to DMSP-like
digital numbers (DN, 0–63 scale) through a radiometric power function and a Gaussian spatial
blur. The native grid is 30 arc-seconds. Config `year_range` is 1992–2024; the assembly clips to
1992–2022.

## Raw data (FETCH)

`raw/ntl_harm/harmonized/`: one file per year as Figshare lists it (`.tif`, sometimes zip or
gz-wrapped). Year is parsed from the filename.

## Prepared output (PREPARE)

- **Path:** `prepared/ntl_harm/harmonized/crs/ease6933/ntl_harm/ix=/iy=/part-<year>.parquet`
- **Column** `ntl_harm`: resampled onto the 1 km grid by area-weighted `sum` (flux-conserving,
  [`../../design/04-ingest.md`](../../design/04-ingest.md) §1); aggregated to coarser grids by
  `average`.

## Analysis caveats

- **Effective resolution is about 4–7 km** because of the harmonization blur. Design doc 04
  therefore keeps `ntl_harm` for ≥ 5 km long-panel work and makes raw VIIRS (`eog_viirs`) the
  lights input for the 1 km ring model.
- **Sensor splice around 2013.** At DN ≥ 7 the lit share jumps around 2013–2016. At DN ≥ 20 the
  artefact largely disappears, and Li et al. flag values above 20 as more reliable
  (`descriptive_statistics.ipynb` §2).
- **Heavy zero mass.** About 68 % of 10 km pixel-years are exactly 0.
- **Gas flares** are bright and not economic activity. Mask them with `eog_flare`, available
  2012 on only.

## Needs live data

Whether the 2023–2024 files exist in the Figshare release that was fetched.

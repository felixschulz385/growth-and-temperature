# no2 — Global surface NO₂ (Anenberg et al. 2022)

| | |
|---|---|
| Config key | `no2` |
| Module | `src/data/sources/no2.py` (`No2Source`) |
| Steps | FETCH, PREPARE · no `REQUIRES` |
| In panel | listed in the assembly since PREPARE finished (2026-10-30); the 10 km panel has not been rebuilt with it |

## What it is

Annual ground-level NO₂ in ppb, 1 km (30 arc-seconds, WGS84, 60° S–75° N), from Anenberg et
al. (2022), *Lancet Planetary Health* 6(1): e49–e58.

- **Release:** figshare article 12968114, version 4 (DOI 10.6084/m9.figshare.12968114.v4),
  CC BY 4.0.
- **Years:** 1990, 1995, 2000 and every year 2005–2020.
- **Method:**
  - a 2010–2012 land-use regression surface (5,220 monitors in 58 countries) is scaled to
    other years with satellite and reanalysis NO₂ columns, including an in-house OMI level 3
    product;
  - the scaling is adjusted for overpass time and cloud cover;
  - a high bias in rural areas is corrected.

Used in the pollution–institutions follow-up as an outcome that stays near its sources
([`../../successors/pollution-data-options.md`](../../successors/pollution-data-options.md) §3.1).

## Raw data (FETCH)

`raw/no2/anenberg/`: one LZW-compressed Float32 GeoTIFF per year, `<year>_final_1km.tif`
(43,080 × 16,200 pixels, nodata −999), as the figshare API lists them.
- **Size:** 0.29–0.62 GB per year, about 8 GB in all.
- **Route:** FETCH runs on the transfer node (`orchestration/scripts/pollution-fetch.sh`), not
  through the compute nodes' proxy.

## Prepared output (PREPARE)

- **Path:** `prepared/no2/anenberg/crs/ease6933/no2/ix=/iy=/part-<year>.parquet`
- **Column** `no2` (ppb): resampled onto the 1 km EASE grid by area-weighted `average` (a
  concentration). Nodata and negative values are NaN.

## Analysis caveats

- **Where the variation over time comes from.** The base pattern is a fixed 2010–2012 land-use
  regression, so with region or cell fixed effects the variation over time comes from the
  satellite scaling. That runs at the satellite's resolution (roughly 10–25 km), not at 1 km.
- **The base pattern uses roads and built-up land as predictors.** Levels across places partly
  reflect development by construction, but fixed effects remove that fixed part.
- **No level break in 2018–2019** (checked 2026-10-30 on the prepared grid). The 2018–2020
  files are about 50 % larger than the others, so the values carry more detail, but the global
  land mean (1.67, 1.65, 1.64, 1.61 ppb over 2017–2020) and 99th percentile (9.9, 9.7, 9.7,
  9.4) move smoothly.
- **Isolated extreme cells.** Most years from 2005 have a few cells of several hundred ppb
  (max 701 in 2019), against a 99th percentile near 10. Trim or winsorize before using maxima.
- **Gaps:** no 1991–1994 or 1996–1999, and 2001–2004 is missing too.
- **Column, not surface.** The temporal signal comes from tropospheric columns, so changes aloft
  can enter the surface estimate.

## Needs live data

The distribution by region. Coverage: about 116 million valid land cells of 1 km in every year.

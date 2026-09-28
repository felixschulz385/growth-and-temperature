# glass — GLASS air temperature and land surface temperature

| | |
|---|---|
| Config keys | `glass_ta_modis`, `glass_modis`, `glass_avhrr` |
| Modules | `src/data/sources/glass/modis.py` (`glass_modis`, `glass_ta_modis`), `src/data/sources/glass/avhrr.py` (`glass_avhrr`) |
| Steps | FETCH, PREPARE · no `REQUIRES` |
| In panel | `glass_ta_modis` only (columns prefixed `glass_ta_`) |

## What it is

GLASS (Global LAnd Surface Satellite) products from the HKU archive (`glass.hku.hk`):

| Key | Product | Native | Years | Role |
|---|---|---|---|---|
| `glass_ta_modis` | GLASS18A01 near-surface **air** temperature (daily `Ta_min`/`Ta_mean`/`Ta_max`) | 1 km, MODIS sinusoidal tiles | 2000 (day 55) – 2020 | second outcome |
| `glass_modis` | GLASS06A01 LST (daily) | 1 km, MODIS sinusoidal tiles | 2000 (day 55) – 2020 | not used in the panel |
| `glass_avhrr` | GLASS AVHRR LST (daily) | 0.05°, one global file per day | 1992–2020 | pre-MODIS series; candidate for pre-2002 placebos |

GLASS air temperature is *modelled*: it is predicted from remote sensing, reanalysis and station
data. Agreement with MODIS LST is therefore partly mechanical; divergence between the two is the
informative result.

## Raw data (FETCH)

- **MODIS variants:** one target per (tile, year). FETCH downloads that tile-year's daily HDF
  files, composites them month-first to annual statistics, and writes one multi-band GeoTIFF per
  tile-year under `raw/glass/Ta/MODIS/` or `raw/glass/LST/MODIS/Daily/1KM/`. Daily files are
  not kept. Pushed to the HPC per file.
- **AVHRR:** one static target per (year, day), one global file per day under
  `raw/glass/LST/AVHRR/0.05D/<year>/` ([`../../design/11-glass-static-fetch.md`](../../design/11-glass-static-fetch.md)).
- There is no QC band in either product. Validity means not-fill and within the configured
  `value_min`/`value_max` (160–370 K for Ta, 150–350 K for LST). Raw int16 values are scaled by
  0.01 to kelvin.

## Prepared output (PREPARE)

- **MODIS variants:** reprojected like `modis` (per-source-tile overlay, nearest) to
  `prepared/glass/Ta/MODIS/crs/ease6933/glass_modis_ta/` or `.../glass_modis_lst/`.

  | Column (panel name) | Meaning | Panel aggregation |
  |---|---|---|
  | `mean` (`glass_ta_mean`) | annual month-weighted mean of daily `Ta_mean`, K | `average` |
  | `std`, `max`, `min` | spread and extremes (`max`/`min` from `Ta_max`/`Ta_min`), K | `average` |
  | `count_above`, `count_below` | months whose mean is above `heat_stress_k` (308.15 K) or below `cold_stress_k` (273.15 K) | `sum` |
  | `valid_period_count`, `valid_month_count` | valid days and valid months in the year | `sum` |

- **AVHRR:** an annual composite per year, then tiled reprojection to
  `prepared/glass/.../crs/ease6933/glass_avhrr_lst/`. Columns: `mean`, `median`, `std`, `max`,
  `min`, `rollmax3`, `rollmin3`, and the day counts `gt30C`, `lt0C`, `valid_count`. It is
  resampled with `mode`, and the 0.05° (~5 km) source is oversampled onto the 1 km grid.

## Analysis caveats

- GLASS ends in 2020, so a panel that includes it loses 2021–22.
- `heat_stress_k`/`cold_stress_k` are illustrative thresholds, not calibrated ones.
- GLASS's own documentation should confirm whether the Ta product is clear-sky only or gap-filled
  ([`06-open-questions.md`](../../design/06-open-questions.md) §2). This decides whether it shares
  MODIS's clear-sky selection problem.
- `glass_avhrr` is not in the assembly. Using it for pre-2002 placebos means adding it to
  `assembly.sources`.
- **`glass_avhrr`'s annual statistics are naive:** a plain mean over all valid days
  (`_calculate_statistics`), not month-first like MODIS and GLASS-MODIS. Years with seasonally
  uneven cloud cover are biased toward the clear season. This is a known fix, deferred (module
  comment). Fix it before using AVHRR for a placebo, or compare it only with itself.

## Needs live data

Coverage achieved per variant; how closely `glass_ta_mean` tracks `lst_night_mean` (the
regression notebook reports a correlation of about 0.98 at 10 km).

## See also

[`12-glass-modis-rebuild.md`](../../design/12-glass-modis-rebuild.md) (the per-(tile, year)
design and the QA-band investigation), [`11-glass-static-fetch.md`](../../design/11-glass-static-fetch.md).

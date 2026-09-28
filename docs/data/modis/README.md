# modis — MODIS Aqua land surface temperature

| | |
|---|---|
| Config keys | `modis` (primary), `modis_robustness_11a1` (robustness arm); `modis_extended` is commented out |
| Module | `src/data/sources/modis/source.py` (`ModisSource`), QC decoding in `modis/tiles.py` |
| Steps | FETCH, PREPARE · no `REQUIRES` |
| In panel | `modis` → `assembly.sources.modis` |

## What it is

MODIS Aqua land surface temperature from Microsoft Planetary Computer's STAC catalogue.

| Variant | Product | Years | Tiles | Role |
|---|---|---|---|---|
| `modis` | MYD21A2 (8-day, TES emissivity) | 2002–2025 | 280 land tiles within \|φ\| ≤ 60° (`land_tiles` in `data.yaml`) | **primary outcome** |
| `modis_robustness_11a1` | MYD11A1 (daily, split-window) | 2004, 2014, 2023 | 5 biome-representative tiles | checks 21A2 against a daily product |

Why MYD21A2 and not the more common MYD11: its temperature–emissivity separation retrieves
emissivity from the radiances themselves rather than from a land-cover lookup table. The
emissivity, and therefore the LST, is independent of land-cover classification
([`../../design/07-modis-ingest.md`](../../design/07-modis-ingest.md) §1). The 11A1 arm uses the
lookup-table method; its tiles are Amazon (`h12v09`), Sahara (`h18v06`), Central Europe (`h18v04`),
Siberia (`h22v03`) and Australia (`h30v11`).

## Raw data (FETCH)

FETCH is not a plain download. For each (tile, year) it:

1. searches STAC for that tile's Aqua items;
2. loads LST, QC, emissivity and view bands, and applies scale/offset/fill manually (they are not
   applied automatically — [`07a`](../../design/07a-modis-band-reference.md));
3. masks pixels with QC LST error above `qc_max_lst_error_k` (2 K) or outside 150–350 K;
4. composites month-first to annual (`src/data/common/raster/compositing.py`): the mean of each
   month's valid 8-day values, then the mean of the monthly means, so each month counts equally.

- **Path:** `raw/modis/21A2/<year>/<tile>.tif` (and `raw/modis/11A1/...`), one multi-band float32
  GeoTIFF in the sinusoidal projection per tile-year, band descriptions = variable names.
- **Monthly values are not persisted**, only the annual statistics below.
- `transfer_mode=auto`: each tile-year is pushed to the HPC as it's written.

## Prepared output (PREPARE)

Each year's tiles are reprojected onto the EASE grid by nearest neighbour, one source tile at a
time, and overlaid (`src/data/common/prepare/sinusoidal_mosaic.py`,
[`15a`](../../design/15a-modis-prepare-rework.md)). The output grid is always `ease6933`,
regardless of `pipeline.grid`.

- **Path:** `prepared/modis/21A2/crs/ease6933/modis_lst_21a2/ix=/iy=/part-<year>.parquet`
  (`modis_lst_11a1` for the robustness arm)

| Column | Meaning | Panel aggregation |
|---|---|---|
| `lst_night_mean`, `lst_day_mean` | annual month-weighted mean LST, K | `average` |
| `lst_night_median`, `lst_day_median` | annual median, K | `average` |
| `lst_night_sd`, `lst_day_sd` | annual standard deviation, K | `average` |
| `valid_period_count_{night,day}_annual` | valid 8-day periods (21A2) or days (11A1) in the year | `sum` |
| `valid_month_count_{night,day}_annual` | months with at least one valid observation | `sum` |

Day and night are masked separately, from `QC_Day` and `QC_Night`.

## Analysis caveats

- **Clear-sky selection.** A composite contains only clear-sky, QC-passing observations. Haze and
  heavy aerosol can trip the cloud mask, so pollution-heavy periods are under-sampled. The valid
  counts are the only per-pixel measure of this; see the analysis plan, Phase 1b.
- **No monthly detail.** Checking where in the year coverage drops would need FETCH to write
  monthly bands again, which means re-streaming from STAC.
- **Aqua orbit drift** after about 2020 moves the overpass time
  ([`14`](../../design/14-terra-aqua-drift-diagnostic.md)).
- **No extreme-value counts** (heat/cold months) here: 8-day compositing is too coarse for them.
  Use GLASS for those.

## Needs live data

- The share of land pixel-years with a valid annual value, by latitude band.
- Whether the 2002–2025 backfill is complete for all 280 tiles (`data summary --source modis`).

## See also

[`07-modis-ingest.md`](../../design/07-modis-ingest.md) (product choice, compositing),
[`07a-modis-band-reference.md`](../../design/07a-modis-band-reference.md) (per-band scale/offset/fill),
[`07b-modis-outstanding.md`](../../design/07b-modis-outstanding.md) (live checklist).

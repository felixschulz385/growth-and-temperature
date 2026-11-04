# gaia — GAIA v1.0 annual impervious share (Gong et al. 2020)

| | |
|---|---|
| Config key | `gaia` |
| Module | `src/data/sources/gaia.py` (`GaiaSource`) |
| Steps | FETCH only (PREPARE to be added on the HPC) · no `REQUIRES` |
| In panel | no (heat-island successor, [`../../successors/heat-island-prespec.md`](../../successors/heat-island-prespec.md)) |

## What it is

GAIA, Global Artificial Impervious Area (Gong et al. 2020, *Remote Sensing of Environment*
236: 111510, doi:10.1016/j.rse.2019.111510), CC BY 4.0. It is the treatment of the heat-island
successor.

- Each 30 m Landsat pixel holds the year it became impervious. A pixel is impervious if it is
  more than 50 % impervious.
- It covers 1985–2018 and is monotone: a pixel changes once, from pervious to impervious, and
  never back.
- It is read from the Earth Engine copy, `Tsinghua/FROM-GLC/GAIA/v10`, band
  `change_year_index`.

This source stores, for each cell of the canonical 1 km EASE-Grid 2.0 and each year 1985–2018,
**the share of the cell's area that is impervious in that year**.

### Year encoding

`change_year_index` runs from 34 (1985) to 1 (2018). The value of year Y is 2019 − Y.

**Impervious in year Y ⇔ `change_year_index` ≥ 2019 − Y.** Checked 2026-11-04 as follows:

- The catalogue's example ("the impervious surface in 1990 can be revealed as the pixel value
  greater than 29") contradicts its own table, which gives 29 = 1990.
- The asset has exactly the values 1–34, and no 0. Pixels never impervious, and the ocean, are
  masked.
- Under a strict ">", 1985 would be empty, and the value-1 pixels, which exist, would never be
  impervious in any year.
- Value 34 is the largest class in old city cores: 39 % of Beijing's impervious pixels within
  15 km of the centre, against 0.3 % for value 1. It is the stock that was already paved in 1985.
- The 2018 global total agrees with the paper (§ Checks).

## Raw data (FETCH)

Path: `raw/gaia/v10/ix=<row>/iy=<col>/gaia_imp_share_1985_2018.tif`. There is one GeoTIFF for
each of the grid's 98 tiles with land. `ix` is the tile row and `iy` the tile column, as in the
HPC's tile layout.

- **Grid:** EPSG:6933, 1000 m pixels, the tile's 2048 × 2048 window of the canonical grid
  (affine `(1000, 0, −17368000, 0, −1000, 6352000)`), clipped at the grid edge.
- **Bands:** 34, `imp_1985` … `imp_2018`.
- **Encoding:** `uint16`, share × 10,000 (0–10,000), deflate with a horizontal predictor.
- **Nodata:** 65535 is declared but never written. GAIA cannot tell "no data" from "not
  impervious", so every cell has a value, and the ocean is 0.
- **Route:** Earth Engine has no internet path from sciCORE. FETCH runs on a workstation with
  Earth Engine access (the Earth Engine project comes from a git-ignored `*.local.yaml` or
  `EE_PROJECT`). The tiles are then pushed with `data transfer --source gaia --step fetch`.

### How a cell's share is computed

- GAIA is reduced from its native 30 m grid (EPSG:4326) to the 1 km cell, weighting each 30 m
  pixel by the area it shares with the cell. Masked pixels count as not impervious.
- **The share is of the whole cell, water included.** A coastal cell that is half sea and
  fully paved on land has a share of 0.5.
- Each cell aggregates about 1,100 source pixels at the equator and about 2,300 at the grid's
  60° edge. Earth Engine raises an error, rather than subsampling, if a cell needs more inputs
  than allowed (checked: Cairo needs 1,288).
- Checked against an independent mean over each cell's polygon at 30 m for Cairo and Lagos,
  1985, 2000 and 2018: the two differ by at most 3 in 10,000.

## Checks (2026-11-04)

Bulk run on 2026-11-04 (about 2.5 h, 32 Earth Engine worker processes), with the checks made
on the HPC copies:

- **Files:** 98 tiles, 428 MB in all, under `data_nobackup/raw/gaia/v10/` on sciCORE.
- **Range and monotonicity:** no cell above 10,000. In every tile, no cell's share falls from
  one year to the next.
- **Nodata:** no cell is 65535.
- **Global impervious area** (sum of shares × 1 km²):

  | 1985 | 1990 | 2000 | 2010 | 2015 | 2018 |
  |---|---|---|---|---|---|
  | 179,495 | 325,733 | 466,706 | 590,840 | 731,253 | 797,268 km² |

  The paper reports 797,076 km² in 2018, within 0.03 %, although the grid leaves out land
  beyond ±60°.
- **Cities:** the city-centre cell's 1985 / 2000 / 2018 share is:
  - Beijing 0.85 / 0.91 / 0.96;
  - Cairo 0.71 / 0.99 / 1.00;
  - Lagos 0.11 / 0.89 / 0.99;
  - São Paulo 0.48 / 0.86 / 0.95;
  - Paris 0.50 / 0.81 / 0.93.
- **Nothing built:** a mid-Atlantic and a central Saharan cell are 0 in every year.

**Re-running FETCH:** `data transfer` deletes each local tile once its push is verified.
FETCH decides what is missing from local files only, so a re-run fetches every tile again.
To fetch only some tiles, list them under `sources.gaia.tiles` in the local yaml.

## Analysis caveats

- **Densification is invisible.** A 30 m pixel counts as impervious above 50 %, so paving
  inside a pixel that is already impervious does not change the share. Neither does paving
  that stays below 50 % of a pixel.
- **Shares only rise.** The product is monotone by construction, so demolition and greening
  never show.
- **Land cover, not buildings.** Roads, car parks and bare compacted ground count. Building
  height and density do not.
- **Accuracy:** the paper reports overall accuracy above 90 % in its validation years (1985,
  1990, 1995, 2000, 2005, 2010, 2015). Errors are largest in arid regions, where bare soil looks
  like pavement.
- **Pre-1985 stock.** Value 34 means "impervious in 1985 or earlier", so 1985 is a stock, not
  a year of change.

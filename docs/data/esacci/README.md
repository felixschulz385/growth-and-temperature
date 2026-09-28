# esacci — ESA CCI Land Cover

| | |
|---|---|
| Config key | `esacci` |
| Module | `src/data/sources/esacci.py` (`EsacciSource`) |
| Steps | FETCH, PREPARE · no `REQUIRES` |
| In panel | yes (`lccs_class`) |

## What it is

ESA Climate Change Initiative annual land-cover maps (`satellite-land-cover` on the Copernicus
Climate Data Store), 1992–2022, 300 m native. Each pixel has one categorical LCCS class code
(0–220). The ones most relevant here: 10–40 cropland, 50–100 forest, 190 urban, 200–202 bare,
210 water.

## Raw data (FETCH)

One CDS API request per year (versions `v2_0_7cds` and `v2_1_1`; needs `~/.cdsapirc`). Files
land in `raw/esacci/landcover/` as zip-wrapped NetCDF.

## Prepared output (PREPARE)

- **Path:** `prepared/esacci/landcover/crs/ease6933/land_cover/ix=/iy=/part-<year>.parquet`
- **Column** `lccs_class`: resampled onto the 1 km grid by nearest neighbour, nodata 0.
  Aggregated to coarser grids by `mode`.

## Analysis caveats

- **`mode` discards fractions.** At 5–10 km, a cell whose built-up share goes from 5 % to 20 %
  usually keeps the same modal class. Land-cover *change* analysis needs per-class fraction
  columns: the `average` of class indicators, computed at assembly.
- **Nearest neighbour at 1 km also samples.** One 300 m source pixel stands in for about 11. A
  fraction variable needs the class shares computed from the 300 m data.
- **Version break.** CDS serves `v2.0.7cds` for 1992–2015 and `v2.1.1` from 2016. Check for a
  level shift in class shares at 2015/2016.
- LCCS classes are known to change slowly and to under-detect gradual urban expansion.

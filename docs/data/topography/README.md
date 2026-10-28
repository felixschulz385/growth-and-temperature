# topography — EarthEnv-Topography slope

| | |
|---|---|
| Config key | `topography` |
| Module | `src/data/sources/topography.py` |
| Steps | FETCH, PREPARE · no `REQUIRES` |
| In panel | no (not in `assembly.sources`); consumed by `corridors` PREPARE (route cost) |

## What it is

Amatulli et al. 2018 (*Sci Data* 5:180040): terrain variables from GMTED2010. This source uses
`slope_10KMmn_GMTEDmd.tif`, the mean of 250 m slopes (degrees) within each 5-arc-minute (~10 km)
cell, 56°S–84°N.

## Raw data (FETCH)

`raw/topography/slope_10KMmn_GMTEDmd.tif`, ~9 MB (proxy or transfer node). The 1 km product
(`slope_1KMmd_GMTEDmd.tif`, ~700 MB) would be a transfer-node download.

## Prepared output (PREPARE)

Static pixel grid `prepared/topography/crs/ease6933/slope/`, one column `slope_mean_10km`,
nearest-neighbour onto the 1 km grid: each pixel carries its ~10 km cell's mean slope. NaN south
of 56°S.

## Analysis caveats

- At 1 km this is a smoothed ~10 km field, not 1 km slope.
- It could be added to `assembly.sources` as a static covariate; it is not there now.

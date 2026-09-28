# gadm — GADM v4.1 administrative boundaries

| | |
|---|---|
| Config key | `gadm` (`data_path: misc`, `namespace: gadm`) |
| Module | `src/data/sources/misc/gadm.py` (`GadmSource`) |
| Steps | FETCH, PREPARE · no `REQUIRES`; required by `plad`, `country_classifications`, `ecoregions`, `snl_mining` |
| In panel | yes (`GID_0`, `GID_1`, ...) |

## What it is

GADM v4.1 administrative boundaries at every available level (ADM0 = country, ADM1, ADM2, ...),
from the UC Davis `gadm_410-levels.zip`.

## Raw data (FETCH)

`raw/misc/gadm/gadm_410-levels.zip`.

## Prepared output (PREPARE)

Two phases:

1. **Vectors and id mappings** in `prepared/misc/adm/gadm/`: simplified per-level GeoPackages
   (`gadm_level*_simplified.gpkg`), and `GID_N_code_mapping.json` for each level, mapping GADM
   string codes to integer ids. Other sources use these mappings to key their admin tables.
2. **Pixel grid**, static: `prepared/misc/crs/ease6933/country_id/ix=/iy=/part.parquet`, with one
   uint32 column per level (`GID_0`, `GID_1`, ...). 0 means no unit at that level. Polygons are
   rasterized straight onto each tile, and the panel aggregates with `mode`.

## Analysis caveats

- The ids are sequential integers assigned at PREPARE time, not GADM codes. Translate them back
  through the mapping JSONs, and don't compare ids across GADM releases.
- `mode` aggregation at coarse grids assigns border cells to the majority unit.
- `GID_0` defines the country × year fixed effects and the clustering; `GID_1` is the ADM1
  cluster.

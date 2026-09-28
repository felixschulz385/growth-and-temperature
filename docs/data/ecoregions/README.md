# ecoregions — RESOLVE Ecoregions and Biomes (Dinerstein et al. 2017)

| | |
|---|---|
| Config key | `ecoregions` (`data_path: misc`, `namespace: ecoregions`) |
| Module | `src/data/sources/ecoregions/source.py` (`EcoregionsSource`), overlay in `ecoregions/overlay.py` |
| Steps | FETCH, PREPARE |
| Requires | `gadm` PREPARE (only for the `GID_3` dominant-class table) |
| In panel | pixel grid only (`realm_id`, `biome_id`, `eco_id`) |

## What it is

The RESOLVE Ecoregions 2017 layer (Dinerstein et al. 2017): one global polygon layer with 8
biogeographic realms, 14 WWF biomes and 846 ecoregions.

## Raw data (FETCH)

A paginated query against RESOLVE's ArcGIS REST FeatureServer, saved as
`raw/misc/ecoregions/resolve_ecoregions_2017.gpkg`. The module docstring explains why this uses
the REST endpoint (Hub export links expire) and how it handles the service's rate limits.

## Prepared output (PREPARE)

Two independent targets:

- **Pixel grid**, static: `prepared/misc/crs/ease6933/ecoregions/ix=/iy=/part.parquet`, with
  columns `realm_id`, `biome_id`, `eco_id`. These are sequential integer ids, with
  `{var}_code_mapping.json` alongside to map them back to codes; 0 means no polygon. The panel
  aggregates with `mode`.
- **`GID_3` table:** `prepared/misc/adm/ecoregions/dominant_biome_by_gid3.parquet`, the
  area-weighted dominant realm, biome and ecoregion of each GADM level-3 unit, from an exact
  vector overlay. Not in the panel.

## Analysis caveats

- These layers are static, so interacting them with year gives the ecoregion × year and
  biome × year fixed effects and the heterogeneity strata in the analysis plan.
- 846 ecoregions × about 20 years is a large number of fixed-effect cells. Biome × country × year
  is usually the more practical stratum.

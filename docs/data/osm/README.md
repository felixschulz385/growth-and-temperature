# osm — OpenStreetMap land polygons (land mask)

| | |
|---|---|
| Config key | `osm` (`data_path: misc`, `namespace: osm`) |
| Module | `src/data/sources/misc/osm.py` (`OsmSource`) |
| Steps | FETCH, PREPARE · no `REQUIRES` |
| In panel | as the land mask (`assembly.land_mask: true`), not as a column |

## What it is

The OpenStreetMap coastline-derived land polygons (`land-polygons-complete-4326.zip`,
osmdata.openstreetmap.de).

## Raw data (FETCH)

`raw/misc/osm/land-polygons-complete-4326.zip`.

## Prepared output (PREPARE)

- A simplified vector: `prepared/misc/misc/osm/land_polygons_simplified.gpkg`.
- A static pixel grid: `prepared/misc/crs/ease6933/land_mask/ix=/iy=/part.parquet`, column
  `land_mask` (0/1), rasterized straight onto each tile.

The assembly (`src/data/assemble/loaders.py::resolve_land_mask_path`) keeps only land pixels.

## Analysis caveats

Coastal pixels are classed by rasterization at pixel centres, so mixed land/water cells can go
either way. Their LST is also affected by the water fraction.

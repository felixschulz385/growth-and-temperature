# osm — OpenStreetMap land polygons (land/water mask)

Registry id `osm` (`src/data/sources/misc/osm.py`, class `OsmSource`). Steps:
**fetch, prepare, grid**. `REQUIRES`: none. Config key: `osm`. `data_path:
"misc"`, `namespace: "osm"` — one of the three sources split out of the old
combined `misc` source (module docstring references
`docs/design/09-integrated-pipeline.md` §7); despite sharing `data_path` with
`gadm`/`country_classifications`, the `namespace` keeps its raw/prepare/grid
subtree distinct.

## FETCH

Downloads a single configured file via `ConfiguredFilesFetchMixin`
(`src/data/sources/misc/_fetch.py`) — `run_fetch()` requires
`ctx.ssh_target` (an HPC/remote target); with none configured, `_execute_fetch`
logs a warning and returns `False`.

- **URL / filename** (`cfg.raw["url"]` / `["name"]`, overridable in
  `data.yaml`): `https://osmdata.openstreetmap.de/download/land-polygons-complete-4326.zip`
  → `land-polygons-complete-4326.zip`. `data.yaml`'s `osm:` block repeats
  these same defaults verbatim.
- **Output path**:
  - legacy: `<data_root>/misc/raw/osm`
  - v2: `<data_root>/raw/misc/osm`
- **Format**: one zip archive containing an ESRI Shapefile (`.shp` + sidecars)
  of OSM land polygons in EPSG:4326 (per the URL/filename, "land-polygons-complete-4326").
- **Caveats**: no URL-rotation/expiry caveat is documented in code or
  `data.yaml` for this source (contrast `ecoregions`, whose module docstring
  explicitly warns its ArcGIS "Export Data" link expires ~1 hour after
  minting) — this is a static, versionless download URL as far as the code
  is concerned. `download:` tuning in `data.yaml` (`batch_size: 2,
  max_concurrent_downloads: 1, tar_max_files: 2, tar_max_size_mb: 100,
  timeout: 600`) is passed straight through to `run_fetch()`.

## PREPARE

`_execute_prepare()`: extracts the fetched zip to a temp dir, takes the
**first** `.shp` found via `Path(extract_dir).glob("**/*.shp")` (raises
`RuntimeError("No shapefiles found in OSM extract")` if none), reads it with
`geopandas`/`pyogrio`, and simplifies every geometry with
`shapely`'s `simplify(tolerance=simplify_tolerance, preserve_topology=True)`
(`simplify_tolerance` config key, default `0.001`, degrees since the source
CRS is EPSG:4326).

- **Output path**:
  - legacy: `<data_root>/misc/processed/stage_1/osm/land_polygons_simplified.gpkg`
  - v2: `<data_root>/prepared/misc/osm/land_polygons_simplified.gpkg`
- **Format**: single GeoPackage (GPKG), one layer, geometry simplified;
  attribute columns are whatever the source shapefile carried through
  unchanged — not enumerated in code (the function copies the GeoDataFrame
  as-is aside from the geometry column). **TODO (needs live data):** actual
  attribute schema / column list of the OSM land-polygons shapefile.
- **Completion**: `Completion.PATH_EXISTS` (single file).
- **Caveats**: only the first shapefile match is used if the zip ever
  contained more than one `.shp` — glob order is filesystem-dependent, not
  sorted in code.

## GRID

`_execute_grid()`: reads the PREPARE-stage GPKG, unions all polygons into one
`shapely.MultiPolygon`, rasterizes it onto the pipeline's target geobox via
`odc.geo.xr.rasterize()` (a single whole-extent call — not tiled, unlike
`gadm`'s per-tile approach), then writes a one-variable zarr dataset. The
write explicitly applies `write_crs_and_grid_mapping_encoding()` (CRS +
`grid_mapping` link on the data variable) — the module docstring notes this
fixes a real bug class where relying on `rasterize()`'s own georeferencing
alone left `.rio.crs` returning `None` on later reads.

- **Output path** (`layout.grid_store_path(..., v2_family="land_mask")`):
  - legacy: `<data_root>/misc/processed/stage_2[_ease6933 if grid_id="ease6933"]/osm/land_mask.zarr`
  - v2: `<data_root>/grid/<grid_id>/land_mask.zarr` (flat, no namespace)
- **Format**: Zarr store, Blosc/zstd-compressed (`clevel=3, shuffle=bitshuffle`),
  coordinates rounded to 5 decimals. `Completion.MARKER` (sibling completion
  file — directory output).
- **Variables**:

  | variable | dtype | value_range | meaning |
  |---|---|---|---|
  | `land_mask` | `bool` (from `rasterio.features.geometry_mask` via `odc.geo.xr.rasterize`; no dtype override in the zarr encoding) | `[0, 1]` | `1` = land, `0` = water |

  `verification_meta(expected_vars=("land_mask",), value_range=(0, 1))` is
  the code-side default; `data.yaml`'s `osm.verification` block declares the
  identical `expected_vars`/`value_range`, so it doesn't actually override
  anything here — it's there to make the check explicit/discoverable in
  config.
- **Caveats**: the whole land-polygon union is rasterized in one call
  against the full target geobox (no tiling), unlike `gadm`'s
  `GeoboxTiles`-based per-tile rasterization — a potential memory/runtime
  consideration for large geobox extents, though the code has no explicit
  comment flagging this as a known limitation.

**Not independently verifiable from code alone:** actual polygon count /
file size of the OSM download, and the real attribute schema of the raw
shapefile (PREPARE only touches the geometry column, so nothing in code
documents the rest).

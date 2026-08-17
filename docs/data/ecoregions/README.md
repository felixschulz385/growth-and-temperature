# ecoregions — RESOLVE Ecoregions & Biomes (Dinerstein et al. 2017)

- **Registry id:** `ecoregions`
- **Class:** `EcoregionsSource` (`src/data/sources/ecoregions/source.py`)
- **Aliases:** none
- **Steps implemented (`STEPS`):** `FETCH`, `PREPARE`, `GRID`
- **`REQUIRES`:** `("gadm", PipelineStep.PREPARE)`, `("gadm", PipelineStep.GRID)` — resolved by reading `gadm`'s output paths directly via `layout.output_root()` / `src.data.sources.misc.gadm.gid_mapping_path()`, never a class import. Only the second GRID target (the GID_3 dominant-biome table) actually needs gadm's output; the main ecoregions rasterization target needs neither.
- **Config key in `data.yaml`:** `sources.ecoregions` (active, not disabled)
  ```yaml
  ecoregions:
    type: "ecoregions"
    data_path: "misc"
    namespace: "ecoregions"
    url: "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/Resolve_Ecoregions/FeatureServer/0/query?where=1=1&outFields=REALM,BIOME_NUM,BIOME_NAME,ECO_ID,ECO_NAME&geometryPrecision=5&f=geojson"
    name: "resolve_ecoregions_2017.gpkg"
    # page_size: 25  # commented out; download() halves it on any page failure regardless
  ```
  `data_path`/`namespace` default to `"misc"`/`"ecoregions"` in `__init__` if unset, matching the checked-in values above.

Source data: one flat global polygon layer (RESOLVE Ecoregions 2017 / "Biomes and Ecoregions 2017") carrying three attributes of increasing classification complexity per polygon — `REALM` (8 biogeographic realms), `BIOME_NUM`/`BIOME_NAME` (14 WWF biomes), `ECO_ID`/`ECO_NAME` (846 ecoregions; module docstring notes the live service returns 847 features). Rasterized here as `realm_id`/`biome_id`/`eco_id`, structurally modeled on GADM's per-level `GID_N` id-grid pattern.

## FETCH

Pulls from RESOLVE's own ArcGIS REST Feature Service (`.../FeatureServer/0/query`), **not** a static file — per the module docstring, the Esri Hub "Export Data" link 302-redirects to an Azure SAS URL that expires ~1 hour after being minted, unusable as a standing config value; the REST query endpoint is the permanent, versioned API.

`download()` overrides `ConfiguredFilesFetchMixin`'s plain streaming-GET with manual pagination:
- Pages through `resultOffset`/`resultRecordCount=page_size` (`page_size` config, default 25) rather than requesting everything in one response.
- On any page parse failure, halves `page_size` and retries (down to a minimum of 1) rather than trusting one constant to always fit under an apparent upstream response-size cap.
- Sniffs for the service's rate-limit response, which the docstring says is delivered as **HTTP 200 with a JSON `{"error":{"code":429,...}}` body**, not a real 4xx status (so `raise_for_status()` never catches it) — `_rate_limit_wait_seconds()` parses the stated `Retry after N sec` and sleeps before retrying the same page.
- Proactively sleeps 1.1s between successful page requests to stay under the service's confirmed-live "large geometry" query cap of 60/minute, rather than relying solely on 429-triggered backoff.
- Combines all pages into one GeoDataFrame and writes it as a single GeoPackage.

`download_async()` wraps the same synchronous logic in a thread-pool executor (no aiohttp-native form).

- **Output path**
  - `<data_root>/raw/misc/ecoregions/resolve_ecoregions_2017.gpkg`
- **Format:** single GeoPackage (`.gpkg`), all 847 (per docstring) polygon features with fields `REALM, BIOME_NUM, BIOME_NAME, ECO_ID, ECO_NAME` plus geometry (`geometryPrecision=5`, ~1.1 m at the equator).
- **Caveats (from code/docstring):**
  - `Completion.NEVER` — the FETCH target always re-plans; `run_fetch` decides what's actually missing.
  - Requires `ctx.ssh_target` (an HPC/remote target) configured, else `_execute_fetch` logs a warning and returns `False`.
  - `geometryPrecision=5` in the URL trims payload size below PREPARE's own `simplify_tolerance` (default 0.001° ≈ 111 m), so it doesn't lose anything simplification would remove anyway.

## PREPARE

Reads the raw GeoPackage (or extracts a `.shp`/`.gpkg` from it first if it's actually a zip archive), validates that `REALM, BIOME_NUM, BIOME_NAME, ECO_ID, ECO_NAME` are all present (logs an error and fails if any are missing — field names may differ from the assumed RESOLVE 2017 schema), simplifies every polygon's geometry (`shapely`/geopandas `.simplify(tolerance=self.simplify_tolerance, preserve_topology=True)`, default tolerance `0.001`, config-overridable via `simplify_tolerance`), and writes the result back out as a single GeoPackage.

- **Output path**
  - `<data_root>/prepared/misc/ecoregions/ecoregions_simplified.gpkg`
- **Format:** single GeoPackage, one flat layer (no per-level split, unlike GADM), same attribute columns as the raw file plus simplified geometry.
- **Schema:** `REALM` (str), `BIOME_NUM` (numeric), `BIOME_NAME` (str), `ECO_ID` (numeric), `ECO_NAME` (str), `geometry` (simplified polygon).
- **Caveats:** completion is `Completion.PATH_EXISTS` (skipped if the output file already exists and `cfg.override` is not set); requires the raw FETCH file (or an index-file fallback via `layout.index_path`) to exist, else `_plan_prepare` yields no targets.

## GRID

Two distinct GRID targets, both gated on the PREPARE output existing:

**1. `ecoregions_grid` — rasterized id-grid.** Rasterizes each polygon's boundary mask *once* per tile and reuses it to paint all three id-grids (`realm_id`, `biome_id`, `eco_id`) simultaneously — unlike GADM's one-rasterize-call-per-level-per-polygon, all three RESOLVE attributes share the same geometry. Codes are mapped to sequential integer ids per column (`code_to_id[col] = {code: i+1 for i, code in enumerate(sorted(gdf[col].unique()))}`), tiled via `odc.geo.GeoboxTiles` (2048×2048 tiles) with an `sindex`-based per-tile candidate prefilter, and geometries are reprojected once up front to the target geobox's CRS before tiling (guarding against the CRS-mismatch bug the module docstring says GADM's rasterizer hit in commit `f653033`).

- **Output path**
  - `<data_root>/grid/<grid_id>/ecoregions.zarr` (flat; `<grid_id>` is `ease6933` under the checked-in config)
- **Format:** single zarr store, dims `(<y>, <x>)` following the target geobox's own dimension names, `uint32` dtype, chunks `(512, 512)`, Blosc-zstd (level 3, bitshuffle) compression, `Completion.MARKER`. A sidecar `<var>_code_mapping.json` (e.g. `realm_id_code_mapping.json`) is written per variable next to the store, mapping each raw RESOLVE code to its integer id.

  | variable | dtype | meaning | `value_range` (verification) |
  |---|---|---|---|
  | `realm_id` | `uint32` | sequential id per unique `REALM` value; `0` = no polygon at this pixel (coastal/ocean gap) | none declared |
  | `biome_id` | `uint32` | sequential id per unique `BIOME_NUM` value; `0` = no polygon at this pixel | none declared |
  | `eco_id` | `uint32` | sequential id per unique `ECO_ID` value; `0` = no polygon at this pixel | none declared |

  `expected_vars=("realm_id","biome_id","eco_id")` comes from `verify.verification_meta(self.cfg.raw, expected_vars=tuple(CLASS_COLUMNS.values()))`; `value_range` is intentionally omitted — per the code comment there's no fixed upper bound (ids are sized by however many distinct codes exist), same rationale as GADM's own `GID_N` variables. `data.yaml`'s `sources.ecoregions.verification` block is present only as a commented-out example (`expected_vars: ["realm_id","biome_id","eco_id"]`) and does not currently override anything.
- **Caveats:** per-tile rasterization failures are caught and logged, not fatal to the whole run (`_process_ecoregions_tiles`'s inner `try/except`).

**2. `gadm_gid3_dominant` — area-weighted dominant-biome-per-GID_3 table.** Only planned if both gadm's level-3 simplified polygons (PREPARE) and its `GID_3` code-mapping JSON (GRID) already exist on disk; otherwise `_plan_grid` logs an info message and skips it. Computed via `src/data/sources/ecoregions/overlay.py::compute_dominant_classes` — a polygon-polygon intersection (`geopandas.overlay`) + per-group area-argmax, deliberately **not** raster zonal-mode (the module docstring argues exact polygon-area weighting is cheaper and more accurate here than rasterizing both layers and counting pixels, and that `assemble/geometry.py`'s zonal reducers don't support majority/mode anyway). Both inputs are reprojected to the target geobox's CRS (an equal-area CRS) before computing area. Ties in area are broken by the lowest class value, for reproducibility.

- **Output path:** written directly under the PREPARE root (not `grid_store_path()` — it's a small per-GID parquet table, not a `<family>.zarr` pixel-grid store):
  - `<data_root>/prepared/ecoregions/dominant_biome_by_gid3.parquet`
- **Format:** single parquet file, `Completion.PATH_EXISTS`.
- **Schema** (one row per `GID_3` unit that has any intersecting RESOLVE polygon and a valid gadm code mapping):

  | column | dtype | meaning |
  |---|---|---|
  | `GID_3` | int | gadm's integer id for the admin unit (translated from the raw `GID_3` string code via gadm's own code-mapping JSON; rows whose code has no mapping entry are dropped) |
  | `GID_3_code` | str | raw gadm `GID_3` string code |
  | `dominant_realm` | same dtype as `REALM` | area-dominant `REALM` value within the unit |
  | `realm_area_frac` | float | dominant realm's share of the unit's total intersected area |
  | `dominant_biome_num` | same dtype as `BIOME_NUM` | area-dominant `BIOME_NUM` value |
  | `dominant_biome_name` | str | `BIOME_NAME` label for `dominant_biome_num` |
  | `biome_area_frac` | float | dominant biome's share of intersected area |
  | `dominant_eco_id` | same dtype as `ECO_ID` | area-dominant `ECO_ID` value |
  | `dominant_eco_name` | str | `ECO_NAME` label for `dominant_eco_id` |
  | `eco_area_frac` | float | dominant ecoregion's share of intersected area |
  | `n_ecoregions_intersecting` | int | count of distinct `ECO_ID` values intersecting the unit — a heterogeneity/confidence flag |

  `expected_vars=("GID_3", "dominant_biome_num", "biome_area_frac")` (checks only these 3 of the 11 output columns exist) — from `verify.verification_meta(self.cfg.raw, expected_vars=(...))`; no `data.yaml` override is present.
- **Consumption:** designed as a `GID_3`-keyed parquet sidecar for `TileProcessor`'s `join_on` merge mechanism at assembly time (same pattern as `snl_mining`'s admin-count tables), per the module docstring — not consumed by anything inside this source itself after being written.

**TODO (needs live data):** actual polygon/feature count returned by a real FETCH run (docstring says 847, code comment references "846 ecoregions" for `ECO_ID`), realistic `page_size`/runtime under live rate-limiting, and observed `n_ecoregions_intersecting` / area-fraction distributions in the dominant-biome table have not been verified against real output.

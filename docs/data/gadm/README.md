# gadm — GADM v4.1 administrative boundaries (per-level GID id grid)

Registry id `gadm` (`src/data/sources/misc/gadm.py`, class `GadmSource`).
Steps: **fetch, prepare, grid**. `REQUIRES`: none. Config key: `gadm`.
`data_path: "misc"`, `namespace: "gadm"` — second of the three sources split
out of the old combined `misc` source (module docstring,
`docs/design/09-integrated-pipeline.md` §7).

This is the one source whose GRID-stage output is consumed **cross-source**
by others (`country_classifications` via its `REQUIRES`, plus `plad`,
`ecoregions`, `snl_mining` reading the same artefact path directly) — see the
"GID mapping sidecar" section below.

## FETCH

Downloads a single configured file via `ConfiguredFilesFetchMixin`
(`src/data/sources/misc/_fetch.py`); `_execute_fetch` requires `ctx.ssh_target`
(an HPC/remote target), else logs a warning and returns `False`.

- **URL / filename** (`cfg.raw["url"]` / `["name"]`): `https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-levels.zip`
  → `gadm_410-levels.zip`. `data.yaml`'s `gadm:` block repeats these defaults
  verbatim.
- **Output path**:
  - `<data_root>/raw/misc/gadm`
- **Format**: one zip archive containing a single GeoPackage with multiple
  layers, one per administrative level (`ADM_0`, `ADM_1`, ... — GADM v4.1's
  "levels" package format; the exact number of levels present is discovered
  dynamically at PREPARE time via `geopandas.list_layers()`, not fixed in
  code).
- **Caveats**: no URL-rotation/expiry caveat documented in code — a static
  academic-mirror download URL, unlike `ecoregions`' documented ArcGIS
  export-link expiry.

## PREPARE

`_execute_prepare()`: extracts the zip, takes the first `.gpkg` found, lists
every layer via `gpd.list_layers()`, and for each layer simplifies geometry
(`simplify(tolerance=simplify_tolerance, preserve_topology=True)`,
`simplify_tolerance` config key, default `0.001`) before writing it out as
its own GeoPackage file.

- **Output path**: a directory, not a single file.
  - `<data_root>/prepared/misc/gadm/`
  - contains one file per level: `gadm_level{ADM_N}_simplified.gpkg` (e.g.
    `gadm_levelADM_0_simplified.gpkg`, `gadm_levelADM_1_simplified.gpkg`, ...).
- **Format**: GeoPackage per level; attribute columns are GADM's own native
  schema for that layer (`GID_N`, `COUNTRY`/`NAME_N`, etc.) — not enumerated
  in code beyond the `GID_N` id column GRID depends on.
- **Completion**: `Completion.MARKER` (directory output with a variable
  number of files; a fallback glob-based check handles pre-MARKER-policy
  runs so already-completed output isn't silently redone).
- **Caveats**: GRID (below) hard-requires `gadm_levelADM_0_simplified.gpkg`
  to exist; if the source zip's GeoPackage has no `ADM_0` layer, GRID's
  `_plan_grid()` returns no targets.

## GRID

`_execute_grid()` rasterizes **every** ADM level present in PREPARE's output
(not just `ADM_0`/`ADM_1`) into one zarr variable per level, named after that
level's own GADM id column (`ADM_1` file → `GID_1` variable, etc.), using
tiled rasterization (`odc.geo.GeoboxTiles`, `tile_size=2048`) under a Dask
client. Before tiling, every level's GeoDataFrame is reprojected to the
target geobox's CRS via `reproject_for_tile_overlap()` — the module docstring
flags this as a real, previously-shipped bug fix (commit `f653033`): without
it, the per-tile `intersects()` overlap pre-filter compares un-reprojected
(e.g. WGS84 degree) geometries against tile bounds in the target CRS (e.g.
EASE6933 meters), silently producing ~100%-null output with no exception.

Each level's unique `GID_N` string codes are sorted and assigned sequential
integer ids starting at 1 (`{code: i + 1 for i, code in enumerate(sorted(codes))}`);
`0` means "no unit at this level." An empty zarr is created first
(`_create_empty_gadm_zarr`, all-zeros `uint32`), then filled tile-by-tile
(`_process_gadm_tiles`, `to_zarr(..., region="auto", mode="r+")`).

- **Output path — zarr grid** (`layout.grid_store_path(..., family="country_id")`):
  - `<data_root>/grid/<grid_id>/country_id.zarr`
- **Format**: Zarr store, `uint32` data variables, chunked `(512, 512)`,
  Blosc/zstd-compressed. `Completion.MARKER`.
- **Variables** (dynamic — one per ADM level found in PREPARE's output, not
  a fixed set):

  | variable | dtype | fill/nodata | meaning |
  |---|---|---|---|
  | `GID_0`, `GID_1`, `GID_2`, ... (as many as PREPARE produced level files for) | `uint32` | `0` ("no unit at this level") | sequential integer id assigned to that level's `GID_N` polygon, per the `{gid_col}_code_mapping.json` sidecar below |

  **No `verification:` block in `data.yaml` by default** — the code comment
  in `_plan_grid()` and the matching `data.yaml` comment both explain why:
  `expected_vars` is discovered dynamically from whichever level files
  PREPARE produced, and there's no fixed upper bound for `value_range` (`0` =
  no unit, else a sequential id sized by however many polygons that level
  has). `verification_meta(expected_vars=gid_cols)` is still called with the
  dynamically-discovered variable list, so presence/non-degenerate-sample
  checks still run — only the numeric range check is intentionally absent
  unless a deployment adds `sources.gadm.verification.value_range` in
  `data.yaml` explicitly (a commented-out example is left in place there).

- **GID mapping sidecar(s) — cross-source dependency.** For each processed
  level, `_execute_grid()` also writes a JSON file mapping GADM's native
  string `GID_N` code (e.g. `"USA.1_1"`) to the same integer id used in the
  zarr grid:
  - **Path**: same directory as the zarr store above, i.e.
    `<data_root>/grid/<grid_id>/{GID_N}_code_mapping.json`, directly beside
    `country_id.zarr` (the flat GRID directory applies no per-source
    namespace).
  - **Format**: JSON object, `{"<GID_N code>": <int id>, ...}`.
  - **Purpose**: `gid_mapping_path(data_root, grid_id, gid_col)`
    (module-level function in `gadm.py`) computes this exact path for other
    sources to consult without importing `GadmSource` (per
    `docs/design/09-integrated-pipeline.md` §2: cross-source coupling stays
    on artefact paths). It lets a source translate its own native GADM
    string codes into the same integer ids gadm's per-pixel grid uses, so a
    small lookup table can be merged directly onto assembled rows instead of
    being rasterized itself. Confirmed callers: `country_classifications.py`
    (`GID_0`, via its `REQUIRES`), `src/data/sources/plad.py` (configurable
    `GID_N` level), `src/data/sources/ecoregions/source.py` (`GID_3`), and
    `src/data/sources/snl_mining/source.py` (`GID_1`/`GID_2`).
  - **Note**: `gid_mapping_path()` always resolves the underlying zarr path
    with `data_path="misc"`/`namespace="gadm"` hardcoded internally — a
    caller does not need (and should not pass) its own `data_path`.

**Not independently verifiable from code alone:** number of ADM levels /
polygon counts in the currently-fetched GADM v4.1 export, and actual per-level
id-space size (both depend on the live zip's contents).

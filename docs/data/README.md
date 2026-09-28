# Data sources

One page per pipeline source: what the dataset is, where its raw and prepared data land, which
columns it contributes to the assembled panel, and the caveats that matter for the analysis.

Implementation details (how FETCH discovers files, how a raw getter clips and reprojects, retry
logic) are in each module's docstring. They are not repeated here, so these pages don't drift out
of date when the code changes. Facts that can only be confirmed from a real run's output (achieved
coverage, value distributions, sizes) are marked **needs live data**.

Last checked against the code: 2026-09-28.

## Sources

| Page | Config key(s) | Module | Output | In panel | Requires |
|---|---|---|---|---|---|
| [modis](modis/README.md) | `modis`, `modis_robustness_11a1` | `modis/source.py` | annual pixel grid | `modis` only | — |
| [glass](glass/README.md) | `glass_ta_modis`, `glass_modis`, `glass_avhrr` | `glass/modis.py`, `glass/avhrr.py` | annual pixel grid | `glass_ta_modis` only | — |
| [ntl_harm](ntl_harm/README.md) | `ntl_harm` | `ntl_harm.py` | annual pixel grid | yes | — |
| [eog](eog/README.md) | `eog_viirs`, `eog_flare` | `eog/source.py`, `eog/flare.py` | annual pixel grid | yes | — |
| [acag](acag/README.md) | `acag` | `acag.py` | annual pixel grid | yes | — |
| [esacci](esacci/README.md) | `esacci` | `esacci.py` | annual pixel grid | yes | — |
| [snl_mining](snl_mining/README.md) | `snl_mining` | `snl_mining/source.py` | annual pixel grid + ADM tables | yes (pixel grid) | gadm, commodity_prices |
| [commodity_prices](commodity_prices/README.md) | `commodity_prices` | `commodity_prices/source.py` | lookup table | no (feeds snl_mining) | — |
| [plad](plad/README.md) | `plad` | `plad.py` | `(GID_2, year)` table | yes (joined) | gadm |
| [gadm](gadm/README.md) | `gadm` | `misc/gadm.py` | static pixel grid + vectors | yes | — |
| [country_classifications](country_classifications/README.md) | `country_classifications` | `misc/country_classifications.py` | `GID_0` table | yes (joined) | gadm |
| [ecoregions](ecoregions/README.md) | `ecoregions` | `ecoregions/source.py` | static pixel grid + `GID_3` table | yes (pixel grid) | gadm |
| [osm](osm/README.md) | `osm` | `misc/osm.py` | static pixel grid | as land mask | — |
| [berman_mining](berman_mining/README.md) | *(disabled)* | `berman_mining.py` | annual pixel grid | no | — |

Modules are under `src/data/sources/`. Every source declares `STEPS = (FETCH, PREPARE)`, except
`snl_mining` (PREPARE only, because its input is assembled by hand). "In panel" means the source
is listed under `assembly.sources` in `orchestration/configs/data.yaml`.

## How every source stores its data

**Tree** (`src/data/sources/layout.py`), relative to `paths.data_root`:

```
raw/<data_path>[/<namespace>]/...                 FETCH output, as downloaded
prepared/<data_path>/crs/<grid_id>/<family>/      PREPARE pixel grids (grid_id = ease6933)
    ix=<row>/iy=<col>/part-<year>.parquet         one part per (tile, year)
    ix=<row>/iy=<col>/part.parquet                static sources (no year)
prepared/<data_path>/adm[/<namespace>]/...        admin-keyed tables, GADM vectors, id mappings
prepared/<data_path>/misc[/<namespace>]/...       other non-spatial outputs
assembled/grid=<label>/shake=<base|s0|...>/ix=/iy=/*.parquet   the analysis panel
```

**Pixel-grid PREPARE output** (`src/data/common/prepare/driver.py::run_tiled_prepare`,
`src/data/common/raster/spatial.py::SpatialProcessor.process_tile_region`):

- The grid is the canonical 1 km EASE-Grid 2.0 (EPSG:6933), clipped to |φ| ≤ 60°
  ([`../design/01-grid.md`](../design/01-grid.md)), in 2048 × 2048-pixel tiles.
- Each part is a wide parquet table with one row per pixel: `cell_id` (uint32, row-major index
  into the full grid), `year` (annual sources only), then one column per variable.
- **Resampling onto the grid happens here, once.** Methods are per source or per variable:
  `nearest` for LST and categorical data, area-weighted `sum` for radiance, `average` for
  per-pixel statistics. Vector sources are rasterized straight onto each tile, with no resampling.
- **Resumable.** Each (tile, year) unit has a JSON status sidecar under
  `crs/<grid_id>/_status/<family>/`. A `.complete` marker next to the output is written only once
  every unit is complete. Bumping a source's `PROCESSING_VERSION` invalidates every unit.

**FETCH bookkeeping** (`src/data/common/fetch/manifest.py`): each source declares the files it
needs, and FETCH diffs that list against one directory listing (local or HPC). Per-unit failures go
into status sidecars. `data summary` reports complete / outstanding / unavailable counts.

**Assembly** (`src/data/assemble/sql_engine.py`, `python -m src.cli assemble create --grid <label>`)
reads every source in `assembly.sources`:

- For grids coarser than 1 km, it aggregates native pixels into exact N × N blocks with the
  per-variable SQL aggregate set in `resampling` (`average`, `sum`, `mode`, `max`, ...).
- It joins pixel-grid sources on `(pixel_id, year)`, or on `pixel_id` for static sources. A source
  counts as annual when its parts are named `part-<year>.parquet`; the `index_cols` config key is
  informational.
- It merges admin-keyed tables (`join_on`) onto the rows by `GID_N` (and `year`).
- It applies the OSM land mask.

**Verification** (`src/data/sources/verify.py`): each source's `verification:` block in `data.yaml`
(expected variables, value range) is checked on a strided sample by `data summary` and before
assembly. It is a sanity check, not a full scan.

## Adding or changing a source

1. Implement the module under `src/data/sources/` and register it in
   `src/data/sources/registry.py::_SOURCE_MODULES`.
2. Add a `sources.<key>` block to `data.yaml`, plus an `assembly.sources.<key>` entry if it
   belongs in the panel.
3. Add or update its page here. Keep the page to what the dataset *is* and what it contributes;
   put the mechanics in the module docstring.

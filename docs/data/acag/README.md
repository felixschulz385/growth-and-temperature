# acag — ACAG (Atmospheric Composition Analysis Group) PM2.5

- **Registry id:** `acag`
- **Class:** `AcagSource` (`src/data/sources/acag.py`)
- **Aliases:** `acag_pm25`, `pm25`
- **Steps implemented (`STEPS`):** `FETCH`, `PREPARE`, `GRID`
- **`REQUIRES`:** none (default `()`, not overridden)
- **Config key in `data.yaml`:** `sources.acag`
  ```yaml
  acag:
    type: "acag"
    data_path: "acag/pm25"
    year_range: [1998, 2023]
    verification:
      expected_vars: ["pm25"]
      value_range: [0, 500]
  ```
  `namespace` is not set for this source (defaults to `None`), so no `/<namespace>` path segment applies to any step below.

Source data: WashU ACAG global annual PM2.5 surface-concentration grids (V6GL02.04, "CNNPM25"), fetched from a hardcoded Box shared-folder file inventory (`KNOWN_FILES`, one `.nc`/`.nc4` file per year, 1998-2023 in the current inventory).

## FETCH

Downloads whatever `KNOWN_FILES` entries are not yet present, via `run_fetch` against a Box shared-link download URL built per file id (`_file_download_url`). Requires `ctx.ssh_target` (an HPC/remote target) to be configured — `_execute_fetch` logs a warning and returns `False` otherwise. Uses a browser-spoofing `User-Agent`/headers and a fixed 0.2s polite delay between requests (`download_async`).

- **Output path**
  - legacy: `<data_root>/acag/pm25/raw`
  - v2: `<data_root>/raw/acag/pm25`
- **Format:** raw files as downloaded from Box — one NetCDF (`.nc`, occasionally `.nc4`) per year, named like `V6GL02.04.CNNPM25.GL.<YYYY>01-<YYYY>12.nc` (one file in `KNOWN_FILES` uses a `.EU.` region code instead of `.GL.` for year 2000 — `V6GL02.04.CNNPM25.EU.200001-200012.nc`).
- **Caveats (from code):**
  - `KNOWN_FILES` is a hardcoded inventory, not discovered dynamically from Box — adding a new year requires a code change.
  - `Completion.NEVER`: the FETCH target always re-runs; `run_fetch` itself is responsible for only downloading what's missing.

## PREPARE

Builds one annual zarr per year from the selected raw file (`.nc4` preferred over `.nc` if both exist for a year — `_plan_prepare`'s candidate-selection order). Loads via `rioxarray.open_rasterio(..., mask_and_scale=True, driver="HDF5")`, extracts the (first) band, renames dims to `latitude`/`longitude`, rescales the raw pixel-index coordinates (`x*0.01 - 180`, `y*0.01 - 60`), writes `EPSG:4326`, masks negative values (`ds.where(ds >= 0)`) and casts to `float32`. Ensures `time` (`<year>-12-31`) and `band` dims exist, then writes to zarr.

- **Output path**
  - legacy: `<data_root>/acag/pm25/processed/stage_1/<year>.zarr`
  - v2: `<data_root>/prepared/acag/pm25/<year>.zarr`
- **Format:** one zarr store per year, dims `(time=1, band=1, latitude, longitude)`, CRS `EPSG:4326`, chunks `(1, 1, 512, 512)`, Blosc-zstd (level 3, bitshuffle) compression, `zarr_format=3`, `consolidated=False`.
- **Schema**

  | variable | dtype | notes |
  |---|---|---|
  | `pm25` | `float32` | negative raw values masked to NaN before cast |

- **Caveats:** completion is marker-based (`<year>.zarr.complete`); a year is only reprocessed if `cfg.override` is set or the marker is missing. Requires the ledger's `completed_fetch_files()` to know which raw files exist — if `local_index_dir`/the ledger for `acag/pm25` is missing, `_plan_prepare` logs a warning and yields no targets.

## GRID

Reprojects every annual PREPARE zarr onto the pipeline's canonical target geobox (`get_target_geobox(ctx)`) into one multi-year timeseries zarr, via `SpatialProcessor.process_spatial_standard` with `resampling="nearest"` (the function's default — not overridden by this source) and no explicit `dst_nodata`/`packaging_attrs` override.

- **Output path**
  - legacy: `<data_root>/acag/pm25/processed/stage_2[_ease6933]/acag_pm25_timeseries_reprojected.zarr` (the `_ease6933` suffix applies when `ctx.grid_id == "ease6933"`; the checked-in `data.yaml` sets `pipeline.grid: ease6933`, so today's legacy path would be `.../stage_2_ease6933/...`)
  - v2: `<data_root>/grid/<grid_id>/pm25.zarr` (flat directory; `<grid_id>` is `ease6933` under the checked-in config)
- **Format:** single multi-year zarr, dims `(time, band=1, <y>, <x>)` (axis names follow the target geobox's own CRS-dependent dimension names — `latitude`/`longitude` for a geographic grid, `y`/`x` for a projected one such as EASE-Grid 2.0 EPSG:6933), CRS written via `.rio.write_crs()`/`grid_mapping="spatial_ref"`, Blosc-zstd compression, chunks `(1, 1, 512, 512)`.
- **Storage encoding (from `SpatialProcessor.create_empty_target_zarr`, since this source passes no `dst_nodata`/`packaging_attrs` override):** stored as `uint16` with `scale_factor=0.01`, `add_offset=0.0` (packed: `physical = stored * 0.01`), fill/nodata value `65535`.

- **Variables**

  | name | on-disk dtype | physical meaning | nodata/fill | `value_range` (verification) |
  |---|---|---|---|---|
  | `pm25` | `uint16` (packed, `scale_factor=0.01`) | annual-mean surface PM2.5 concentration (µg/m³, per ACAG's own product; not independently confirmed here) | `65535` | `[0, 500]` |

  `expected_vars`/`value_range` come from `verify.verification_meta(self.cfg.raw, expected_vars=("pm25",), value_range=(0, 500))` in `_plan_grid`, and are **not** overridden by `data.yaml`'s `sources.acag.verification` block (same values: `expected_vars: ["pm25"]`, `value_range: [0, 500]`) — the config block just makes the same values explicit/overridable.
- **Caveats:** marker-based completion, same override semantics as PREPARE. `_plan_grid` only includes years whose annual PREPARE zarr already exists on disk (via `_list_annual_zarrs`, scanning the PREPARE output directory) — it does not consult the ledger.

**TODO (needs live data):** actual year coverage achieved in a real run (vs. the 1998-2023 inventory/`year_range`), on-disk file/store sizes, and empirically observed PM2.5 value distribution have not been verified against real output and are not claimed here.

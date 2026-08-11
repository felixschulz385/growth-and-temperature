# ntl_harm — Harmonized DMSP-VIIRS nighttime lights

- **Registry id:** `ntl_harm`
- **Class:** `NtlHarmSource` (`src/data/sources/ntl_harm.py`)
- **Aliases:** `ntlharm`, `harmonized_ntl`
- **Steps implemented (`STEPS`):** `FETCH`, `PREPARE`, `GRID`
- **`REQUIRES`:** none (default `()`, not overridden)
- **Config key in `data.yaml`:** `sources.ntl_harm`
  ```yaml
  ntl_harm:
    type: "ntl_harm"
    base_url: "https://api.figshare.com/v2/articles/9828827"
    data_path: "ntl_harm/harmonized"
    year_range: [1992, 2018]
    file_extensions: [".tif", ".zip", ".tar.gz", ".gz"]
    download:
      batch_size: 50
      max_concurrent_downloads: 1
      tar_max_files: 40
      tar_max_size_mb: 2000
      timeout: 600
    verification:
      expected_vars: ["ntl_harm"]
      value_range: [0, 2000]
  ```
  `namespace` is not set for this source (defaults to `None`), so no `/<namespace>` path segment applies to any step below.

Source data: the harmonized DMSP-VIIRS global nighttime-lights time series, Figshare dataset id `9828827` (`DATASET_ID`), fetched via the Figshare API (`FIGSHARE_API_BASE = "https://api.figshare.com/v2"`).

Two behavioural quirks are called out explicitly in the module docstring as deliberately preserved from the pre-migration code (pinned by `tests/data/preprocess/sources/test_characterization_ntl_harm.py` and `tests/data/sources/ntl_harm/test_ntl_harm_plan.py`):
1. PREPARE targets are emitted in file-discovery/insertion order, not sorted by year.
2. The best-file-for-year preference is `.tif > .zip > .tar.gz > .gz` — the opposite direction from `acag`/`esacci`'s "`.nc4` over `.nc`" preference.

## FETCH

Lists the dataset's file inventory from the Figshare API (`GET {base_url}`, cached in-process for 1h — `_cache_duration = 3600`), filters to filenames matching `file_extensions` (default `[".tif", ".zip", ".tar.gz", ".gz"]`), and downloads whatever is missing via `run_fetch`. Requires `ctx.ssh_target` to be configured, same as `acag`/`esacci`. `download_async` adds a fixed 0.3s polite delay per file.

- **Output path**
  - legacy: `<data_root>/ntl_harm/harmonized/raw`
  - v2: `<data_root>/raw/ntl_harm/harmonized`
- **Format:** raw files as downloaded from Figshare — filenames and container format (`.tif`/`.zip`/`.tar.gz`/`.gz`) are whatever Figshare's own file listing provides; year is parsed from the filename via `_extract_year_from_filename` (tries `\d{4}`, `_\d{4}_`, `\.\d{4}\.` in that order, accepting matches in `[1992, 2030]`).
- **Caveats (from code):** if the Figshare listing request fails (`RequestException`/`JSONDecodeError`), `_get_figshare_files` logs and returns `[]` rather than raising. `Completion.NEVER`: the FETCH target always re-runs.

## PREPARE

Builds one annual zarr per year from the selected raw file (`.tif` preferred, then `.zip`, then `.tar.gz`, then `.gz` — `_select_best_file_for_year`, note the reversed preference vs. `acag`/`esacci`). Handles decompression itself: `.gz` is gunzipped to a sibling path; `.zip` is extracted to a temp dir and the first `.tif` member inside is used (raises if none found). Opens the resulting `.tif` with `rioxarray.open_rasterio`, adds a `time` dim (`<year>-12-31`), and cleans up the temp/extracted file afterward (`_cleanup_file` attr, removed whether the write succeeds or fails).

- **Output path**
  - legacy: `<data_root>/ntl_harm/harmonized/processed/stage_1/<year>.zarr`
  - v2: `<data_root>/prepared/ntl_harm/harmonized/<year>.zarr`
- **Format:** one zarr store per year, variable name `ntl_harm` (`VARIABLE_NAME`), chunked `{"x": 512, "y": 512}`, Blosc-zstd (level 3, bitshuffle) compression. Dataset-level attrs are stamped with `_FillValue=65535, scale_factor=1, add_offset=0.0` (`_create_annual_zarr`) — these are descriptive metadata attrs only; no scaling is actually applied to the array values at this step, and the on-disk dtype is whatever `rioxarray.open_rasterio` read from the source `.tif` (not explicitly cast in code).
- **Schema**

  | variable | dtype | notes |
  |---|---|---|
  | `ntl_harm` | source dtype preserved (not explicitly cast) | harmonized DMSP-VIIRS radiance/DN, per source `.tif` |

- **Caveats:** marker-based completion; same ledger dependency as `acag`/`esacci` (`_plan_prepare` needs the `ntl_harm/harmonized` ledger's `completed_fetch_files()`). Files are grouped by year in **discovery order**, not sorted — see the module-docstring quirk above.

## GRID

Reprojects every annual PREPARE zarr onto the pipeline's canonical target geobox into one multi-year timeseries zarr, via `SpatialProcessor.process_spatial_standard`. Unlike `acag`/`esacci` (which use the function's `"nearest"` default), this source explicitly passes `resampling=self.resampling`, which defaults to `"sum"` (`cfg.raw.get("resampling", "sum")`, not overridden in the checked-in `data.yaml`) — an area-weighted aggregation appropriate for a radiance-like field, per the module docstring's reference to `docs/design/04-ingest.md §1`. No `dst_nodata`/`packaging_attrs` override is passed, so `SpatialProcessor.create_empty_target_zarr`'s defaults apply.

- **Output path**
  - legacy: `<data_root>/ntl_harm/harmonized/processed/stage_2[_ease6933]/ntl_harm_timeseries_reprojected.zarr` (`_ease6933` suffix when `ctx.grid_id == "ease6933"`; the checked-in `data.yaml` sets `pipeline.grid: ease6933`)
  - v2: `<data_root>/grid/<grid_id>/ntl_harm.zarr` (flat directory; `<grid_id>` is `ease6933` under the checked-in config)
- **Format:** single multi-year zarr, dims `(time, band=1, <y>, <x>)` (axis names follow the target geobox's CRS — `latitude`/`longitude` for geographic, `y`/`x` for projected e.g. EASE-Grid 2.0 EPSG:6933), CRS via `.rio.write_crs()`/`grid_mapping="spatial_ref"`, Blosc-zstd compression, chunks `(1, 1, 512, 512)`.
- **Storage encoding (from `SpatialProcessor.create_empty_target_zarr`, since this source passes no `dst_nodata`/`packaging_attrs` override):** stored as `uint16` with `scale_factor=0.01`, `add_offset=0.0` (packed: `physical = stored * 0.01`), fill/nodata value `65535`.

- **Variables**

  | name | on-disk dtype | physical meaning | nodata/fill | `value_range` (verification) |
  |---|---|---|---|---|
  | `ntl_harm` | `uint16` (packed, `scale_factor=0.01`) | harmonized DMSP-VIIRS nighttime-lights radiance, area-summed onto the target grid | `65535` | `[0, 2000]` |

  `expected_vars`/`value_range` come from `verify.verification_meta(self.cfg.raw, expected_vars=("ntl_harm",), value_range=(0, 2000))` in `_plan_grid`, and match `data.yaml`'s `sources.ntl_harm.verification` block (`expected_vars: ["ntl_harm"]`, `value_range: [0, 2000]`) — the config block reiterates rather than overrides the code defaults.
- **Caveats:** marker-based completion; `_plan_grid` only includes years whose annual PREPARE zarr already exists on disk (`_list_annual_zarrs`, not the ledger). Because `resampling="sum"` aggregates (rather than resamples) source pixels into each target cell, values are sensitive to the relative pixel-area ratio between source and target grids — not something verifiable from code alone.

**TODO (needs live data):** actual year coverage achieved in a real run (vs. the 1992-2018 `year_range`), on-disk store sizes, and empirically observed radiance value distribution (vs. the configured `[0, 2000]` verification range) have not been verified against real output and are not claimed here.

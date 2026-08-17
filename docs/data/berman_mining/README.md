# berman_mining — Berman et al. mining conflict / resource-trap data

**Status: disabled.** The entire `sources.berman_mining` block in `orchestration/configs/data.yaml` is commented out (lines ~323-335), with the in-file comment: *"Berman Mining Conflict Data (Manual Download Required) -- disabled, not run by the pipeline. Left commented rather than deleted so it can be re-enabled by uncommenting."* The source class is still registered at import time (`registry.register(...)` runs unconditionally at module load, so it's importable/plannable by the `data` CLI), but with no live `sources.berman_mining` config entry it is not part of any configured `data run` today. Everything below is documented from the code and the commented-out `data.yaml` block, not from a live run.

- **Registry id:** `berman_mining`
- **Class:** `BermanMiningSource` (`src/data/sources/berman_mining.py`)
- **Aliases:** `berman`, `mining_conflict`
- **Steps implemented (`STEPS`):** `FETCH`, `GRID` — **no `PREPARE`**: per the module docstring, like PLAD, mining-point gridding happens directly from the raw `.dta` file in one stage.
- **`REQUIRES`:** none (`()`, not overridden). The module docstring is explicit that this is a deliberate correction of "an earlier, unverified planning assumption": this source only shares the VIIRS-derived geobox *cache location* with `osm`/`gadm` (via `get_or_create_geobox`, which builds the cache independently from a VIIRS download if missing), not a hard dependency on GADM's output.
- **Config key in `data.yaml`:** `sources.berman_mining` — **commented out**:
  ```yaml
  # berman_mining:
  #   type: "berman_mining"
  #   data_path: "berman_mining"
  #
  #   download:
  #     batch_size: 1
  #     max_concurrent_downloads: 1
  #     tar_max_files: 5
  #     tar_max_size_mb: 500
  #     timeout: 60
  ```
  `data_path` defaults to `"berman_mining"` in `__init__` if unset (matching the commented value); `namespace` is not set.

Source data: Berman et al. mining conflict / resource-trap baseline data (`BCRT_baseline.dta`), distributed via ICPSR (openICPSR project 113068, version V1), which requires an authenticated manual download — not a scriptable URL fetch.

**Two real bugs fixed here, not silently ported** (per the module docstring, verified by direct source inspection and pinned in `tests/data/preprocess/sources/test_characterization_berman_mining.py`): the old `BermanMiningPreprocessor` defined `get_hpc_output_path` and `from_config` twice each in the same class body, with the second definitions silently shadowing the first as byte-identical dead code. Not carried forward into this class.

## FETCH

Manual download, not an automated HTTP fetch. `list_remote_files()` returns exactly one entry: `("baseline/BCRT_baseline.dta", "https://www.openicpsr.org/openicpsr/project/113068/version/V1/view")`. `download()` calls `_prompt_for_file_path()`, which:
1. Prints the file description, reference URL, and `DOWNLOAD_INSTRUCTIONS` (visit the ICPSR project page, log in / create an account, navigate to `Data/BCRT_baseline.dta`, download it locally, then provide the local path).
2. Blocks on `input("\nFile path: ")`, re-prompting until an existing file path is given or the user presses Enter to skip (which raises `FileNotFoundError`).
3. Copies the user-provided file to the FETCH output path via `shutil.copy2`.

`download_async()` wraps the same synchronous, interactive `download()` in a thread-pool executor.

- **Output path**
  - `<data_root>/raw/berman_mining`
- **Format:** whatever the user supplies, copied byte-for-byte — expected to be `BCRT_baseline.dta` (Stata `.dta`) under a `baseline/` subfolder, per `MANUAL_FILE`.
- **Caveats (from code):**
  - Requires `ctx.ssh_target` (an HPC/remote target) configured, else `_execute_fetch` logs a warning and returns `False` — same guard as every other source's FETCH, despite this FETCH being interactive/manual rather than a real remote transfer.
  - `Completion.NEVER`: the FETCH target always re-plans; `run_fetch` decides what's actually missing (and would re-trigger the interactive prompt for any missing file).
  - `mining_data_path` (config key) can override where GRID reads the `.dta` file from directly, bypassing the FETCH output path convention; if unset, GRID reads `<FETCH output root>/baseline/BCRT_baseline.dta`.

## PREPARE

Not implemented — not in `STEPS`.

## GRID

`_create_mining_dataset()`: reads the `.dta` file with `pandas.read_stata`, selects columns `nb_mines_a`, `nb_diamond`, indexes by `(latitude, longitude, year)`, converts to an `xarray.Dataset`, writes CRS `EPSG:4326` (`.rio.write_crs(4326)`), and optionally slices to `cfg.year_range` (`target.meta["year_range"]`) if configured.

`_execute_grid()`: casts both variables to `uint8` (`fillna(255)` before the cast), reprojects onto the pipeline's canonical target geobox (`get_target_geobox(ctx)`, **not** GADM-derived — see `REQUIRES` note above) via `odc.geo.xr.xr_reproject(..., resampling="nearest", dst_nodata=255)`, renames the `year` dim to `time` (coordinate `f"{year}-12-31"` per year), adds a constant `band=[1]` dimension, and writes a fresh CRS/grid-mapping encoding before writing to zarr (`zarr_format=3`, `consolidated=False`).

- **Output path** (`layout.grid_store_path(..., family="berman_mining")`; `<grid_id>` depends on `pipeline.grid`, currently `ease6933`)
  - `<data_root>/grid/<grid_id>/berman_mining.zarr` (flat)
- **Format:** single multi-year zarr store, dims `(time, band=1, <y>, <x>)` (axis names follow the target geobox's own dimension names), `uint8` dtype, chunks `(1, 512, 512, 1)` (as declared in the encoding — dimension order not independently re-derived here), Blosc-zstd (level 3, bitshuffle) compression, `fill_value=255`, `Completion.PATH_EXISTS`.

- **Variables**

  | name | dtype | meaning | fill/nodata | `value_range` (verification) |
  |---|---|---|---|---|
  | `nb_mines_a` | `uint8` | per-pixel value from the raw `nb_mines_a` column (mine-count-like variable; exact definition not documented in code beyond the column name) | `255` | `[0, 50]` |
  | `nb_diamond` | `uint8` | per-pixel value from the raw `nb_diamond` column (diamond-mine-related count, per column name) | `255` | `[0, 50]` |

  `expected_vars=("nb_mines_a", "nb_diamond")`, `value_range=(0, 50)` come from `verify.verification_meta(self.cfg.raw, expected_vars=("nb_mines_a", "nb_diamond"), value_range=(0, 50))` in `_plan`. The commented-out `data.yaml` block has no `verification:` sub-block at all, so — if this source were re-enabled by uncommenting it as-is — these Python-side defaults would apply unmodified.
- **Caveats:** `uint8` with `fillna(255)` caps any true per-pixel count at 254 before it would collide with the nodata sentinel — not flagged as a problem in code, just a consequence of the chosen dtype/nodata pairing given the declared `[0, 50]` expected range. `Completion.PATH_EXISTS` (not marker-based) despite being a zarr directory store.

**TODO (needs live data):** exact semantics of `nb_mines_a`/`nb_diamond` beyond their column names (not documented in this code), actual value distribution/coverage of a real gridded run, and confirmation that re-enabling the commented-out config block (with no `verification:` override) is the only change needed to run this source, have not been verified.

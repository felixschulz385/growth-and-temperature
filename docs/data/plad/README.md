# plad — PLAD (Political Leaders and Development) regional favoritism

- **Registry id:** `plad`
- **Class:** `PlaDSource` (`src/data/sources/plad.py`)
- **Aliases:** `harvard_plad`, `harvard`
- **Steps implemented (`STEPS`):** `FETCH`, `GRID` — **no `PREPARE`**: per the module docstring, PLAD's raw `.dta` table already carries GADM's native `gid_1`/`gid_2` string codes directly, so there's no vector-boundary pre-step to build.
- **`REQUIRES`:** `("gadm", PipelineStep.GRID)` — only gadm's GRID-stage `GID_N_code_mapping.json` sidecar is needed (to translate PLAD's native string codes into gadm's integer ids); no GADM polygon geometry is used (module docstring: this changed from requiring gadm's PREPARE once rasterization, and the polygon geometries it needed, was removed — see GRID below).
- **Config key in `data.yaml`:** `sources.plad` (active, not disabled)
  ```yaml
  plad:
    type: "plad"
    doi: "doi:10.7910/DVN/YUS575"
    data_path: "plad"
    file_extensions: ["tab"]
    admin_level: 2
    year_range: [1980, 2022]
    verification:
      expected_vars: ["GID_2", "year", "reg_fav"]
  ```
  `namespace` is not set for this source.

Source data: the Harvard Dataverse "Political Leaders and Development" (PLAD) dataset, DOI `doi:10.7910/DVN/YUS575` by default (config-overridable via `doi`/`base_url`). `admin_level` (config, default `1` in code but set to `2` in `data.yaml`) selects whether the output keys on `GID_1` or `GID_2`; must be `1` or `2` (`ValueError` otherwise).

**Quirk preserved, not fixed:** `output_root()` is overridden for the `GRID` step to hardcode the string `"plad"` (`OUTPUT_PREFIX`) as the output-path prefix, never `self.cfg.data_path` — per the module docstring this mirrors an old-code quirk (`get_hpc_output_path` hardcoded `"plad"`), so even a configured `data_path` override would not change where GRID output lands. `FETCH` is not affected by this override (uses the base class's `output_root()`, which does use `cfg.data_path`) — in practice this makes no difference today since `data_path` already defaults to/is configured as `"plad"`.

## FETCH

Lists files from the Harvard Dataverse API (`GET https://dataverse.harvard.edu/api/datasets/:persistentId?persistentId=<doi>`), filtering to labels ending in one of `file_extensions` (config: `["tab"]`; code default if unset: `[".csv", ".nc", ".tif", ".zip"]`). Downloads via `https://dataverse.harvard.edu/api/access/datafile/<file_id>`, with a fixed 0.5s delay before each request (`download`/`download_async`).

- **Output path**
  - `<data_root>/raw/plad`
- **Format:** raw file(s) as published on Dataverse — a `.dta`/`.tab` table (GRID looks specifically for a file whose name contains `"plad"` and ends in `.dta`, via the local ledger's `completed_fetch_files()`).
- **Caveats (from code):**
  - `Completion.NEVER`: the FETCH target always re-plans; `run_fetch` decides what's actually missing.
  - Requires `ctx.ssh_target` (an HPC/remote target) configured, else `_execute_fetch` logs a warning and returns `False`.
  - `filename_to_entrypoint`/`get_all_entrypoints` are both stubs (return `None`/`[]`) — `has_entrypoints = False`.

## PREPARE

Not implemented — not in `STEPS`.

## GRID

**No longer rasterized** (module docstring): regional favoritism (`reg_fav`) is constant across every pixel of the favored admin unit for a given year — it varies only by `GID_1`/`GID_2` and year, never by pixel location — so GRID writes a tiny `(GID_N, year)`-keyed parquet table of favored units instead of a pixel-grid zarr, and needs no GADM polygon geometry at all.

`_build_reg_fav_table()`: reads the raw PLAD `.dta` file (`pd.read_table`, located via the local ledger by filename), requires a `gid_{admin_level}` column to exist. For each row, expands `[max(startyear, cfg.year_range[0]), min(endyear, cfg.year_range[1])]` into one row per year, translates the raw `gid_N` string code to gadm's integer id via the `GID_N_code_mapping.json` sidecar (rows whose code isn't in the mapping — id `0` — are dropped), sets `reg_fav = True` for every remaining row, and de-duplicates on `(GID_N, year)`.

- **Output path** (via the overridden `output_root()` — see the `OUTPUT_PREFIX` quirk note above; `<grid_id>` depends on `pipeline.grid`, currently `ease6933`):
  - `<data_root>/grid/<grid_id>/plad_adm2_reg_fav.parquet` (flat; filename encodes `admin_level`, e.g. `plad_adm1_reg_fav.parquet` if `admin_level: 1`)
- **Format:** single parquet file, `Completion.PATH_EXISTS`.
- **Schema** (one row per favored admin unit per year it was favored — absence from the table, not a `False` value, means "not favored"):

  | column | dtype | meaning |
  |---|---|---|
  | `GID_2` (or `GID_1` if `admin_level: 1`) | int | gadm's integer id for the favored admin unit |
  | `year` | int | calendar year, clamped to `cfg.year_range` (config `[1980, 2022]`) |
  | `reg_fav` | bool | always `True` in this table — presence-only; downstream assembly is expected to `fillna(False)` for admin units absent from the table (per the module docstring, handled by the assemble config, not baked in here) |

  `expected_vars = (self._gid_column, "year", "reg_fav")` — e.g. `("GID_2", "year", "reg_fav")` at `admin_level=2` — from `verify.verification_meta(self.cfg.raw, expected_vars=(...))`, matching `data.yaml`'s explicit `sources.plad.verification.expected_vars: ["GID_2", "year", "reg_fav"]` (same values; the config block makes them explicit/overridable rather than changing them). No `value_range` — not a numeric measurement table.
- **Caveats:** only planned if the `GID_N_code_mapping.json` sidecar (gadm's GRID output) already exists on disk, else `_plan_grid` yields no targets. `admin_level` must match the `GID_N` column checked by `expected_vars` — a mismatch between `admin_level` and a hand-edited `verification.expected_vars` would silently fail verification rather than error at config-load time.

**TODO (needs live data):** actual row count / year coverage of a real `plad_adm2_reg_fav.parquet`, and how many raw PLAD rows get dropped for having no matching gadm code, have not been verified against real output.

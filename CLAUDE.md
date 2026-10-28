# CLAUDE.md

Working notes for Claude Code in this repository. Keep this file current: update it in the same
commit as any change to the state, layout or rules below.

## State (2026-10-28)

- **The original project is retired** (2026-09-30). The closing record is
  `docs/analysis/project-conclusion.md`. Don't change its results; amendments go in dated notes.
- **The one active line of work is the fork on branch `fork/corridors-dose`.** It tests a
  transit-corridor instrument (bulk mines → ports) and then DOSE regional output at ADM1 as a growth
  measure other than lights. Plan: `docs/successors/fork-corridors-dose.md`. Rules:
  `docs/successors/corridors-prespec.md`. Unscheduled ideas: `docs/successors/ideas.md`.
- **Corridor progress:**
  - done: step 0 (pre-spec), step 1 (`ports`, `topography`), step 2 (routes), step 3 (exposure),
    all as data modules;
  - next: route checks (GRIP roads, maps, balance), then **Gate C1**: first stage of `log_ntl` on
    `z_main_rc` only, port-clustered F ≥ 70. No LST before C1 passes.
- Fork work goes on the branch, not `main`. `main` gets only docs that record the fork.

## Pre-specification discipline

Every analysis block has rules (◇) written down before it runs. Change a rule only through the
change log at the end of the pre-spec, with a date and reason, and never after seeing a result the
change could be fitted to. Checks that load no outcome (routes, snapping, exposure counts) may
change construction rules; log them anyway.

## Environment

- **HPC (sciCORE).** Compute nodes have no internet. Interactive sessions usually run inside a
  SLURM job (check `SLURM_JOB_ID`), so heavy PREPARE steps can run in place or through
  `--slurm`.
- **Downloads.** Small files (< ~100 MB: CSVs, small GeoTIFFs, metadata) may go through the Claude
  proxy (`https_proxy` is set). **Large data goes only through the transfer node:** run FETCH
  there, then `data transfer --direction push` (pattern: `orchestration/scripts/modis-fetch.sh`).
  `src/data/common/fetch/driver.py`'s aiohttp session ignores proxy variables. A source that must
  fetch through the proxy overrides `download_async()` to run a `requests` download in an executor
  (see `ports.py`, `topography.py`).
- **Python:** `miniforge-pypy3/envs/gnt` (the scripts' env) or `envs/src`. `duckreg` is a sibling
  source checkout (`../duckreg`, or `DUCKREG_PATH`); `src/analysis/spec.py` adds it to the path.
- **Data root:** `data_nobackup/` (git-ignored), resolved by `orchestration/configs/data.local.yaml`.

## Data pipeline

- **Every new dataset is a module in `src/data/sources/`, not a script.** Register it in
  `registry.py::_SOURCE_MODULES`, add a `sources.<key>` block to `orchestration/configs/data.yaml`,
  and write `docs/data/<key>/README.md` plus a row in `docs/data/README.md`.
- **CLI:**

  ```bash
  python -m src.cli data run --config orchestration/configs/data.yaml --source <key> --step fetch|prepare [--slurm]
  python -m src.cli data plan|summary ...
  python -m src.cli assemble create --grid 10km
  ```

- **Layout:** `raw/<data_path>/`; `prepared/<data_path>/{crs/ease6933/<family>, adm, misc}/`;
  `assembled/grid=<label>/`. See `docs/data/README.md` § "How every source stores its data".
- **Grid:** 1 km EASE-Grid 2.0 in 2048-pixel tiles. Coarser grids are F × F blocks that **restart
  at each tile origin** (`src/data/assemble/sql_engine.py::_pixel_id_sql`). Anything keyed to the
  10 km panel must use that `pixel_id`, not a regular global 10 km grid.
- **Cross-source inputs** are read by artefact path (`layout.output_root`/`grid_store_path`),
  never by importing another source's class. Ordering goes in `REQUIRES`. A FETCH target is never
  "complete", so a dependency must point at a PREPARE step.
- **Fork sources:**
  - `ports`: PortWatch, FETCH + PREPARE;
  - `topography`: EarthEnv slope, FETCH + PREPARE onto the 1 km grid;
  - `corridors`: PREPARE only. It writes `pixel_id`-keyed tables under
    `prepared/corridors/misc/<variant>/` that join the 10 km panel. Its variants are in
    `data.yaml`.

## Analysis

- **Specification:** `src/analysis/spec.py` (outcome, FE, clusters, sample, `build_panel()`). The
  derived panel is `data_nobackup/_analysis_panel_10km_v2.parquet`.
- **Notebooks:** `output/notebooks/`. The pre-commit hook blocks notebooks with outputs
  (`./.githooks/install.sh` once per clone).

## Docs conventions

- Design docs (`docs/design/NN-*.md`) are never renumbered or renamed. Outdated ones get a status
  banner, and `docs/README.md` has a row for each.
- Source docs describe what a dataset is and its caveats. Mechanics live in module docstrings.
- Commit only when asked. On `main`, branch first for anything that is not documentation.

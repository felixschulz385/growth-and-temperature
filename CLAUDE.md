# CLAUDE.md

Working notes for Claude Code in this repository. Keep this file current: update it in the same
commit as any change to the state, layout or rules below.

## State (2026-10-29)

- **The original project is retired** (2026-09-30). The closing record is
  `docs/analysis/project-conclusion.md`. Don't change its results; amendments go in dated notes.
- **A fork ran 2026-10-28 to 2026-10-29 and is merged into `main` (complete; no active line of
  work).** It tested a transit-corridor instrument (bulk mines → ports), then DOSE regional output
  at ADM1 as a growth measure other than lights. Plan: `docs/successors/fork-corridors-dose.md`.
  Rules and outcomes: `docs/successors/corridors-prespec.md` and `dose-prespec.md`. The
  conclusion's §5 has a dated summary. Unscheduled ideas: `docs/successors/ideas.md`.
- **Corridor result (Part 1): stopped 2026-10-28.** The route check R1 passed: predicted routes
  run along real roads, t = 7.5. Gate C1 failed: first-stage F = 0.08 against 70. The outcomes
  are in `docs/successors/corridors-prespec.md` §8. No LST was loaded.
- **Part 2 (DOSE at ADM1):** the `dose` and `adm1_instruments` data modules are built. DOSE's
  GADM 3.6 codes are matched to 4.1, 1,596 of 1,662 regions. **Gate D1 passed:** 399 exposed
  regions in 48 countries. The design is confirmed (`docs/successors/dose-prespec.md` §3, §3a,
  §3b): three instruments (mine count, mine × world price frozen at 2001, regional favoritism),
  DOSE `P_a` as the precipitation control, so the sample is 2002–2018. **All six first stages
  failed (2026-10-29),** three instruments × steps B (lights) and C (GRP). The largest F is 4.3
  against 10, so there is no IV at ADM1. **The fixed-effects association (descriptive)** of
  night LST is −0.26 K per log point of lights (p < 0.001) and −0.13 K per log point of GRP
  (p = 0.09), the retired result's sign (§4). **Part 2 is complete.**
- **New work** (a successor project) starts on its own branch, with a pre-spec in
  `docs/successors/`. The branch `fork/corridors-dose` is merged and no longer used.

## Confidential mine data (hard rule)

The S&P Global / SNL mine data is licensed and confidential. **Nothing at mine level may appear in
any commit**: file contents, notebook outputs, figures, docs or commit messages.

- **Never commit:** mine ids (`property_id`), names, coordinates or polygons; per-mine
  attributes (commodity, opening and closing years, production); per-mine route tables; maps that
  show mine points, routes from mines, or mine counts per grid cell; tables listing individual
  mines.
- **Allowed:** regression estimates and aggregate counts (number of mines, regions or countries;
  shares), and descriptions at region level ("Shanxi coal goes to Tianjin").
- **Figures that show mines** go to `output/figures/confidential/`, which is git-ignored. Notebooks
  that draw them are committed without outputs.
- **Enforcement:** `.githooks/check-confidential.py` blocks `confidential/` paths, data files and
  figures whose names mention mines, unless they are allow-listed. It can't read content, so this
  rule still applies to everything it misses. Enable the hooks once per clone with
  `git config core.hooksPath .githooks`.
- **Old violations, fixed 2026-10-28 in `afa3c4c` on `main`** and merged here: the mining GIFs
  (mine counts per grid cell) and the outputs of `src/experiments/mines.ipynb`. They stay in
  earlier commits. **Origin is public**, and the author chose not to rewrite history. Never
  rewrite or force-push to purge them without being asked.

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
  - `grip`: GRIP4 road density, FETCH + PREPARE onto the 1 km grid. Used only for route check R1.
  - `dose`: DOSE v2.9 regional output, matched to GADM 4.1 `GID_1` (`prepared/dose/adm/`).
    Its `T_a` column is temperature: never use it as a control. `P_a` (ERA5 precipitation) stops
    in 2018.
  - `corridors`: PREPARE only. It writes `pixel_id`-keyed tables under
    `prepared/corridors/misc/<variant>/` that join the 10 km panel. Its variants are in
    `data.yaml`.
  - `adm1_instruments`: PREPARE only. It writes the `(GID_1, year)` mine count, mine × price and
    favoritism instruments to `prepared/adm1_instruments/adm/`. These are region aggregates
    only, with no per-mine table.

## Analysis

- **Specification:** `src/analysis/spec.py` (outcome, FE, clusters, sample, `build_panel()`). The
  derived panel is `data_nobackup/_analysis_panel_10km_v2.parquet`.
- **Fork analysis:** `src/analysis/corridors.py` builds the corridor panel. It has no LST
  columns until a gate allows them. Notebooks: `05_corridors.ipynb` (R1, C1),
  `05b_corridors_route_map.ipynb`, `06_dose.ipynb` (D1). `src/analysis/dose.py` builds the
  ADM1 region panel, again with no LST until `add_lst()`. Its fits use pyfixest, which takes
  regression weights, rather than duckreg. Notebooks: `06b_dose_step_b.ipynb`, `06c_dose_step_c.ipynb` and
  `06d_dose_twfe.ipynb` (the associations; `add_lst()` joins LST to the region panel).
- **Notebooks:** `output/notebooks/`. The pre-commit hook blocks notebooks with outputs
  (`./.githooks/install.sh` once per clone).

## Docs conventions

- Design docs (`docs/design/NN-*.md`) are never renumbered or renamed. Outdated ones get a status
  banner, and `docs/README.md` has a row for each.
- Source docs describe what a dataset is and its caveats. Mechanics live in module docstrings.
- Commit only when asked. On `main`, branch first for anything that is not documentation.

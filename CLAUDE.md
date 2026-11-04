# CLAUDE.md

Working notes for Claude Code in this repository. Keep this file current: update it in the same
commit as any change to the state, layout or rules below.

## State (2026-11-04)

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
- **Successor project "pollution along the growth path, by institutions": run 2026-10-29 to
  2026-11-04, no active step.** It is on the branch `successor/pollution-institutions` and is
  descriptive: ADM1 PM2.5 (ACAG) on growth, by baseline V-Dem democracy terciles. There are two
  runs, each with its own pre-spec in `docs/successors/`:
  - idea B, with DOSE output (`pollution-institutions-prespec.md`);
  - step 1, with night lights per person, all countries, DMSP years 1998–2012
    (`pollution-lights-prespec.md`).

  The results are in `docs/successors/pollution-results.md`. PM2.5 rises with output below about
  4,000 USD, but only across countries. With DOSE output, institutions make no robust
  difference (Wald p = 0.53). With lights (rerun 2026-11-04 after the antimeridian PREPARE
  fix), the rule finds two high − middle differences: above 12,000 USD (p = 0.018), absent in
  the VIIRS years, and at 1,000–4,000 USD (p = 0.033), which rests on the Dominican Republic
  moving tercile. No turning point survives within-country comparisons. **Step 2, NO₂, ran
  2026-11-04** (`docs/successors/pollution-no2-prespec.md`): transport rule T1 not met (NO₂ rises
  within countries at 1,000–4,000 USD with DOSE output, not with lights); R1 met (step 1's
  difference above 12,000 USD replicates with NO₂, p = 0.017); moderation by the rule only with
  DOSE output (4–12k, high − middle); no downturn. MERRA-2 dust is deferred: it needs an
  Earthdata login.
- **Idea A (surface heat island) has a draft pre-spec** (2026-10-30, not confirmed):
  `docs/successors/heat-island-prespec.md`. It replicates the YCEO city heat islands, then
  estimates LST on within-cell GAIA impervious share at 1 km, 2003–2018. Nothing has run, and
  the data (GAIA, GHSL built-up, the benchmark) are not built. It gets its own branch,
  `successor/heat-island`, once confirmed.
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
  that read mine-level data are committed without outputs. Other notebooks may keep theirs.
- **Enforcement:** `.githooks/check-confidential.py` blocks `confidential/` paths, data files and
  figures whose names mention mines, unless they are allow-listed.
  `.githooks/check-notebook-outputs.py` blocks outputs in notebooks that read mine-level data:
  the `!` lines of `notebook-output-whitelist.txt`, plus any notebook whose code names an
  SNL or Berman source or `property_id`. A notebook that reaches mine data indirectly needs
  a `!` line. It can't read content, so this
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
- **Fork and successor sources:**
  - `ports`: PortWatch, FETCH + PREPARE;
  - `topography`: EarthEnv slope, FETCH + PREPARE onto the 1 km grid;
  - `grip`: GRIP4 road density, FETCH + PREPARE onto the 1 km grid. Used only for route check R1.
  - `dose`: DOSE v2.9 regional output, matched to GADM 4.1 `GID_1` (`prepared/dose/adm/`).
    Its `T_a` column is temperature: never use it as a control. `P_a` (ERA5 precipitation) stops
    in 2018.
  - `corridors`: PREPARE only. It writes `pixel_id`-keyed tables under
    `prepared/corridors/misc/<variant>/` that join the 10 km panel. Its variants are in
    `data.yaml`.
  - `vdem` (successor): V-Dem v16 country-year indices from the V-Dem Institute's R data
    package (`.RData`, read with `pyreadr`).
  - `no2` (successor follow-up): Anenberg et al. surface NO₂, 1 km, figshare (1990, 1995,
    2000, 2005–2020; ~8 GB). FETCH on the transfer node: `orchestration/scripts/pollution-fetch.sh`.
  - `ghs_pop` (successor follow-up): GHS-POP R2023A, 1 km Mollweide, five-year epochs; persons
    per km², epoch years only (interpolate in the analysis). FETCH as for `no2`. Both are
    prepared (2026-10-30) and listed in the assembly, but the 10 km panel has not been rebuilt
    with them, and its year clip drops the 2025 epoch. `src/analysis/pollution_lights.py` sums
    lights and population per region from the 1 km tiles instead.
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
- **NO₂ follow-up:** `src/analysis/pollution_no2.py` aggregates region NO₂ from the 1 km tiles
  (plain and GHS-POP-weighted), builds the two arms and fits the stacked T1. Notebook:
  `09_pollution_no2.ipynb` (G3, all rules and variants, two figures →
  `output/figures/pollution/no2_*`).
- **Lights follow-up:** `src/analysis/pollution_lights.py` builds the region panel from the 1 km
  `ntl_harm` and `ghs_pop` tiles (flare mask, epoch interpolation), calibrates the knots, runs
  G2 and joins region PM2.5 (`region_pm25()`: plain and population-weighted, plus GLASS
  temperature). The fits reuse `pollution.fit_paths(controls=())`. Notebook:
  `08_pollution_lights.ipynb` (G2, main specification, variants 1–8 and H, two figures →
  `output/figures/pollution/pm25_lights_*`).
- **Successor analysis:** `src/analysis/pollution.py` builds the PM2.5 region panel and fits
  the tercile × spline paths (pyfixest), and draws the figure (`plot_paths()`, Nature-style:
  183 mm, 7 pt Nimbus Sans, PDF and PNG). Notebook: `07_pollution.ipynb` (G1, main
  specification, pooled path, variants 1–6, three figures → `output/figures/pollution/`).
- **Notebooks:** `output/notebooks/`. They may keep their outputs, except those that read
  mine-level data (pre-commit hook; `./.githooks/install.sh` once per clone).

## Docs conventions

- Design docs (`docs/design/NN-*.md`) are never renumbered or renamed. Outdated ones get a status
  banner, and `docs/README.md` has a row for each.
- Source docs describe what a dataset is and its caveats. Mechanics live in module docstrings.
- Commit only when asked. On `main`, branch first for anything that is not documentation.

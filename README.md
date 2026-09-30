# Growth and Temperature (GNT)

Does local economic growth change local temperature, separately from the global CO₂-driven
trend? Growth alters the local energy balance through three channels that push in different
directions:

- **land-cover change** (urbanisation, deforestation) and **anthropogenic heat** warm the surface;
- **aerosols** from pollution cool it.

The sign of the net effect is therefore an empirical question.

This repository holds the data pipeline and the analysis for a global 1 km panel of grid cells.

## Research design at a glance

| | |
|---|---|
| Outcome | MODIS Aqua night land surface temperature (MYD21A2), annual, 2002–2022; GLASS near-surface air temperature as a second outcome |
| Treatment | Nighttime lights: harmonized DMSP–VIIRS (`ntl_harm`) and VIIRS annual composites (`eog_viirs`) |
| Instruments | Mining exposure (S&P/SNL mines: active-mine counts and commodity price shocks within 10/20/50 km); regional favoritism (leaders' birth regions, PLAD) |
| Mechanisms | ESA CCI land cover, ACAG surface PM2.5 |
| Model | Pixel and country × biome × year fixed effects, 2SLS at the 10 km cell level; the instrument's spatial reach weights in neighbour-cell spillovers ([`docs/analysis/10km-ring-check.md`](docs/analysis/10km-ring-check.md)). A distance-ring specification up to 30 km is a robustness check |
| Grid | 1 km EASE-Grid 2.0 (EPSG:6933), \|φ\| ≤ 60°; coarser grids by exact block aggregation. Main analysis grid: 10 km |

The analysis plan, including the known threats to identification, is in
[`docs/analysis/final-analysis-plan.md`](docs/analysis/final-analysis-plan.md). Every data source
is described in [`docs/data/`](docs/data/README.md).

## Setup

```bash
conda env create -f environment.yml    # creates the "gnt" env and runs `pip install -e .`
conda activate gnt
./.githooks/install.sh                 # blocks committing notebooks that contain outputs
```

Machine-specific settings go in `orchestration/configs/data.local.yaml` (git-ignored), which
overrides the empty `paths:`/`remote:` blocks in `data.yaml`:

```yaml
paths:
  data_root: "/path/to/data"          # raw/, prepared/, assembled/ live under here
remote:                               # optional: HPC target for pushing fetched data
  ssh_target: "user@host:/path/to/data"
  key_file: "~/.ssh/id_ed25519"
```

Some sources need credentials: EOG in `orchestration/secrets/eog.credentials.json` or
`EOG_USERNAME`/`EOG_PASSWORD`; ESA CCI in `~/.cdsapirc`. The analysis notebooks also expect a
`duckreg` source checkout next to this repo (or at `DUCKREG_PATH`).

## Usage

Everything runs through one CLI, `python -m src.cli`, with three domains.

**`data`**: fetch and prepare one source (see [`docs/data/`](docs/data/README.md)).

```bash
python -m src.cli data list                                  # registered sources
python -m src.cli data summary                               # what's complete / outstanding
python -m src.cli data run --source acag --step fetch
python -m src.cli data run --source acag --step prepare

# on the HPC: submit as SLURM jobs (defaults from orchestration/configs/slurm_jobs.yaml)
python -m src.cli data run --source snl_mining --step prepare --slurm --chain   # includes REQUIRES
python -m src.cli data run --source snl_mining --step prepare --slurm --chain --dry-run
```

**`assemble`**: merge every source in `assembly.sources` into the analysis panel.

```bash
CFG=orchestration/configs/data.yaml    # assemble needs --config explicitly; data defaults to it
python -m src.cli assemble create --config $CFG --grid 10km                # coarser grids aggregate 1 km pixels
python -m src.cli assemble create --config $CFG --grid 10km --shake quad   # + shifted-origin robustness variants
python -m src.cli assemble create --config $CFG --grid 10km --slurm        # submit on the HPC
python -m src.cli assemble update --config $CFG --grid 10km --datasource eog_viirs   # refresh one source
# output: <data_root>/assembled/grid=<label>/shake=<base|s0|...>/ix=/iy=/*.parquet
```

**`analysis`**: batch model runs defined in `orchestration/configs/analysis.xlsx` (git-ignored):
`analysis run | submit | summary | tables | cleanup | subsets`.

Interactive analysis lives in `output/notebooks/`, in the three stages of
[`docs/analysis/final-analysis-plan.md`](docs/analysis/final-analysis-plan.md):
`01_pre_analysis.ipynb`, `02_core_analysis.ipynb` and `03_post_analysis.ipynb`. All three share
the specification in `src/analysis/spec.py`. `descriptive_statistics.ipynb` (exploratory, day
LST, 1992–2022) and `descriptive_overview.ipynb` (one map per assembled column) are companions.

## Repository layout

```
src/
  cli/                 python -m src.cli entry point (data / assemble / analysis)
  data/
    sources/           one module or package per data source, plus registry, layout, verify
    common/            shared machinery: fetch, prepare driver, grid/geobox, neighbourhood engine, HPC push
    assemble/          DuckDB panel assembly (block aggregation, joins, grid-shake)
  analysis/            batch estimation, SLURM submission, table rendering
  viz/                 plotting helpers (coefficient plots, maps)
  experiments/         exploratory notebooks (not maintained)
orchestration/
  configs/             data.yaml (sources + assembly), slurm_jobs.yaml, local overrides
  scripts/             maintenance and validation SLURM scripts (see LOGGING.md)
scripts/               one-off maintenance and diagnostic scripts
tests/                 pytest suite (runs in CI: .github/workflows/tests.yml)
docs/                  analysis plan, data-source pages, design records (docs/README.md)
output/                notebooks, tables, figures, presentations, results website
data/                  local data root (git-ignored)
```

## Tests

```bash
pytest -q
```

## Contact and license

Felix Schulz (felix.schulz@unibas.ch), University of Basel. MIT License, see [LICENSE](LICENSE).

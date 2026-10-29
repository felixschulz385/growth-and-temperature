# Documentation

**The project was retired on 2026-09-30.** Start with
[`analysis/project-conclusion.md`](analysis/project-conclusion.md). The documentation below is kept
as a record. A fork (transit corridors; DOSE regional output at ADM1) ran 2026-10-28 to 2026-10-29
and was merged; it adds no causal result. Its records are in [`successors/`](successors/), and its
outcome is summarised in the conclusion, §5.

| Folder | What's in it |
|---|---|
| [`analysis/`](analysis/) | What was estimated, in which order, why, and what came out |
| [`data/`](data/README.md) | One page per data source: what it is, where it lands, its columns and caveats |
| [`design/`](design/) | Numbered design and decision records for the data pipeline, with incident notes |
| [`successors/`](successors/) | The fork and ideas for successor projects, with their critique |

## Where to start

- **What the project found, and why it stopped:**
  [`analysis/project-conclusion.md`](analysis/project-conclusion.md).
- **The analysis plan, its pre-specified rules and each block's outcome:**
  [`analysis/final-analysis-plan.md`](analysis/final-analysis-plan.md).
- **What a panel column means:** [`data/README.md`](data/README.md), then the source's page.
- **How the pipeline is built:** [`design/00`](design/00-backbone-overview.md) (grid and ring
  model), [`design/09`](design/09-integrated-pipeline.md) (source/step structure), then
  [`data/README.md`](data/README.md) § "How every source stores its data" for the current storage
  and assembly mechanics.
- **Running things:** the root [`README.md`](../README.md); reproducing the analysis:
  [`analysis/project-conclusion.md`](analysis/project-conclusion.md) §6.

## Analysis records

| Doc | Topic | Status |
|---|---|---|
| [project-conclusion](analysis/project-conclusion.md) | Findings, interpretation, decision, successor notes, reproduction | Final |
| [final-analysis-plan](analysis/final-analysis-plan.md) | Fixed specification; pre-analysis and the three stories, with ◇ rules and outcomes | Closed 2026-09-30 |
| [10km-ring-check](analysis/10km-ring-check.md) | Does the main estimate need the distance-ring model? (No) | Record, 2026-09-28 |
| [drift-and-pm25-coverage-checks](analysis/drift-and-pm25-coverage-checks.md) | Aqua night-overpass drift; ACAG PM2.5 clear-sky coverage | Record, 2026-09-28 |

## Successor records

| Doc | Topic | Status |
|---|---|---|
| [fork-corridors-dose](successors/fork-corridors-dose.md) | Transit-corridor instrument; DOSE regional output at ADM1 as a growth measure other than lights | Complete 2026-10-29, merged into `main`: no causal result from either part |
| [corridors-prespec](successors/corridors-prespec.md) | Corridor instrument rules: shock mines, ports, routes, exposure, recentering, Gate C1 | Closed 2026-10-28: R1 passed, Gate C1 failed (F = 0.08) |
| [dose-prespec](successors/dose-prespec.md) | DOSE at ADM1 with three instruments (mine count, mine × price, favoritism): data, Gate D1, steps B and C | Closed 2026-10-29: D1 passed; all six first stages failed (F ≤ 4.3); association −0.26 K per log lights, −0.13 K per log GRP |
| [ideas](successors/ideas.md) | Continuous-urban heat-island panel; pollution along the growth path by institutions; a growth → channels → LST chain | Documented 2026-10-28; not scheduled |

## Design records

Design docs are numbered in the order they were written, and code comments cite them by path.
So they are **never renumbered or renamed**. When a doc is outdated, it gets a status banner under
its title instead of being rewritten, and this table says where the current truth lives.

| Doc | Topic | Status |
|---|---|---|
| [00](design/00-backbone-overview.md) | Backbone redesign: ring model, grid, why | Current (science design) |
| [01](design/01-grid.md) | EASE-Grid 2.0 (EPSG:6933), \|φ\| ≤ 60°, tiling, kernels | Current reference |
| [02](design/02-storage.md) | Storage layout, disc-sum ladder | Current; pixel grids are now parquet (see its update note) |
| [03](design/03-neighbourhood-engine.md) | FFT disc convolution / ring means | Implemented; not yet wired into assembly |
| [04](design/04-ingest.md) | Ingest principles: resampling per variable, compositing, lights choice | Current principles |
| [05](design/05-migration.md) | Backbone rollout plan | Historical |
| [06](design/06-open-questions.md) | Open questions, 2026-07-29 | Session record; live status in 07b and 13 |
| [07](design/07-modis-ingest.md) | MODIS product choice and compositing | Current; step names changed (banner) |
| [07a](design/07a-modis-band-reference.md) | MODIS band scale/offset/fill/QC reference | Current reference |
| [07b](design/07b-modis-outstanding.md) | MODIS checklist | Frozen at retirement (2026-09-30) |
| [08](design/08-hpc-transfer.md) | Generic HPC transfer | Superseded (banner) |
| [09](design/09-integrated-pipeline.md) | Integrated fetch/prepare pipeline, `misc` split | Implemented; GRID step since merged into PREPARE (banner) |
| [10](design/10-fetch-ledger.md) | DuckDB fetch ledger | Superseded: ledger removed 2026-08-14 |
| [11](design/11-glass-static-fetch.md) | GLASS static FETCH targets | Implemented, later restructured by 12 |
| [12](design/12-glass-modis-rebuild.md) | GLASS split and GLASS-MODIS rebuild, GLASS air temperature | Implemented |
| [13](design/13-prepare-memory-parallelism.md) | PREPARE memory and parallelism | Frozen at retirement (2026-09-30) |
| [14](design/14-terra-aqua-drift-diagnostic.md) | Terra/Aqua orbital-drift diagnostic | Never run. The project used Aqua only, so the Terra/Aqua fusion question did not arise; Aqua's night drift was measured in [`analysis/drift-and-pm25-coverage-checks.md`](analysis/drift-and-pm25-coverage-checks.md) §1 |
| [15](design/15-modis-prepare-2002-tile-failures.md) | MODIS PREPARE 2002 tile failures | Superseded by 15a (root-cause record) |
| [15a](design/15a-modis-prepare-rework.md) | Reproject-then-overlay PREPARE | Implemented |

When you add a design doc, give it the next number, put a one-line status under the title once it
ships or is superseded, and add a row here.

## Documentation elsewhere in the repo

- [`orchestration/scripts/LOGGING.md`](../orchestration/scripts/LOGGING.md): SLURM log locations.
- [`.githooks/README.md`](../.githooks/README.md): the notebook-output pre-commit hook.
- [`src/data/sources/snl_mining/notebooks/README.md`](../src/data/sources/snl_mining/notebooks/README.md):
  building the manual mining database.
- [`output/webpage/README.md`](../output/webpage/README.md): the results website.
- Module docstrings under `src/`: implementation details, kept next to the code.

# berman_mining — Berman et al. (2017) mining data (disabled)

| | |
|---|---|
| Config key | `berman_mining`, **commented out** in `data.yaml` |
| Module | `src/data/sources/berman_mining.py` (`BermanMiningSource`) |
| Steps | FETCH (manual), PREPARE · no `REQUIRES` |
| In panel | no |

## What it is

The replication data of Berman, Couttenier, Rohner & Thoenig (2017), *"This Mine Is Mine!"*
(`BCRT_baseline.dta`, openICPSR project 113068). It contains mine counts (`nb_mines_a`,
`nb_diamond`) on a 0.5° × 0.5° grid for Africa.

**Superseded by [snl_mining](../snl_mining/README.md)**, which builds a global mine panel with
metric buffers, footprints and price shocks. The code is kept so this source can be re-enabled by
uncommenting its config block.

## If re-enabled

- **FETCH** is manual. ICPSR requires an authenticated download; the source asks for a local path.
- **PREPARE** writes `prepared/berman_mining/crs/ease6933/berman_mining/ix=/iy=/part-<year>.parquet`
  with uint8 columns `nb_mines_a` and `nb_diamond`. 255 means missing.

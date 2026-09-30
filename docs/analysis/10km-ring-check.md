# Ring Check on the 10 km Panel

Analysis note, 2026-09-28. It records one check that decides whether the main specification
needs the distance-ring model of [`00-backbone-overview.md`](../design/00-backbone-overview.md),
or whether a cell-level regression on a coarse grid is enough and rings can come later as
robustness. The outcome changed Phase 0 of [`final-analysis-plan.md`](final-analysis-plan.md).

**Conclusion:** use the 10 km cell-level 2SLS for the main estimate. Rings, and the 1 km panel
they need, become robustness checks.

## 1. The question

The ring model in [`00`](../design/00-backbone-overview.md) regresses temperature on own-cell
lights plus mean lights in successive annuli. Its quantity of interest is Σ_r β_r. It raises two
concerns:

- **Collinearity.** Lights growth is spatially smooth, so the ring regressors are highly
  correlated even after pixel FE. That makes each β_r imprecise. Σ_r β_r is much less affected,
  because the estimation errors of positively correlated regressors are negatively correlated
  and largely cancel in the sum.
- **Is it needed for the IV at all?** [`00`](../design/00-backbone-overview.md) argues that
  own-cell lights with pixel FE recover only a small fraction of the true effect. That argument
  assumes neighbours' lights vary independently of own-cell lights. For 2SLS with one instrument
  Z, the own-cell estimate converges to

  ```
  RF / π_0  =  Σ_r β_r · (π_r / π_0)
  ```

  where π_r is the effect of Z on lights in ring r. `mine_count_20km` is spatially smooth: one
  mine raises it for every cell within 20 km. If π_r ≈ π_0 across the neighbourhood, own-cell
  2SLS already estimates approximately Σ_r β_r.

The check measures the π_r profile directly, then compares own-cell and neighbourhood-mean 2SLS.

## 2. Setup

- **Panel:** assembled 10 km base grid, Aqua years 2002–2022, rows with non-null
  `lst_night_mean`. That gives 25.25 M pixel-years and 1.20 M pixels.
- **Neighbour terms**, built with dense per-year arrays, not the ring engine:
  - ring 1: the 8 adjacent cells, 10–14 km from the cell centre;
  - ring 2: the 16 cells at Chebyshev distance 2, 20–28 km;
  - disc means: the 3×3 and 5×5 area means of `log1p(ntl_harm)`.
- **Specification:** the notebook's current one. FE `pixel_id + GID_0^year`; instrument
  `mine_count_20km`; clusters at GID_0 and GID_1. The plan's Phase 0 FE (country × biome × year)
  was **not** used, so the numbers are directly comparable to `regression.ipynb` as it stood at the time (since split into three notebooks).
- **Neighbour-lag model:** own-cell and ring-1 lights, instrumented by `mine_count_10km` and
  its ring-1 mean.

## 3. Results

**Collinearity.** Within-pixel correlation of `log1p(ntl_harm)`: own vs ring 1 is 0.81, own vs
ring 2 is 0.73, ring 1 vs ring 2 is 0.91. (Cross-sectional: 0.91 and 0.84.)

**First-stage profile** (effect of `mine_count_20km` on lights):

| Lights in | π | SE GID_0 | SE GID_1 | π / π_own |
|---|---|---|---|---|
| Own cell | 0.0672 | 0.0122 | 0.0085 | 1.00 |
| Ring 1 (10–14 km) | 0.0575 | 0.0110 | 0.0079 | 0.86 |
| Ring 2 (20–28 km) | 0.0358 | 0.0088 | 0.0068 | 0.53 |
| 3×3 disc mean | 0.0586 | 0.0111 | 0.0079 | 0.87 |
| 5×5 disc mean | 0.0440 | 0.0095 | 0.0072 | 0.65 |

First-stage F for the own cell is about 30 with GID_0 clusters and about 62 with GID_1 clusters.

**Reduced form and 2SLS** (K per log point of lights; reduced form per unit of
`mine_count_20km`):

| | Night LST, GID_0 | Night LST, GID_1 | Day LST, GID_0 | Day LST, GID_1 |
|---|---|---|---|---|
| Reduced form | −0.0070 (0.0111) | −0.0070 (0.0119) | −0.0122 (0.0183) | −0.0122 (0.0166) |
| 2SLS, own cell | −0.104 (0.163) | −0.104 (0.179) | −0.182 (0.267) | −0.182 (0.246) |
| 2SLS, 3×3 disc | −0.120 (0.189) | −0.120 (0.205) | −0.209 (0.308) | −0.209 (0.283) |
| 2SLS, 5×5 disc | −0.159 (0.256) | −0.159 (0.273) | −0.278 (0.416) | −0.278 (0.379) |

The day own-cell estimate reproduces the notebook headline (−0.18, SE 0.27).

**Neighbour-lag 2SLS** (night LST, GID_0):
- Estimates: own cell +0.164 (0.030); ring 1 −0.315 (0.205).
- First stage for own lights: own-cell `mine_count_10km` 0.143 (0.029), ring-1 mean 0.104
  (0.051).
- First stage for ring-1 lights: own-cell `mine_count_10km` −0.012 (0.007), ring-1 mean 0.254
  (0.050).

## 4. Interpretation

1. **Own-cell 2SLS is already a neighbourhood estimand.** With π_1/π_0 = 0.86 and
   π_2/π_0 = 0.53, it estimates approximately β_0 + 0.86·β_1 + 0.53·β_2. That is nearly the full
   effect out to about 14 km, and half of it at 20–28 km. The "small fraction" argument in
   [`00`](../design/00-backbone-overview.md) applies to OLS, not to this IV.
2. **The null is in the reduced form, so the choice of lights aggregation cannot change it.**
   With one instrument, every 2SLS row is the same reduced form divided by a different first
   stage. Own-cell, disc and ring treatments rescale the estimate but carry the same evidence.
   For night LST the own-cell 95 % CI is about [−0.42, +0.22] K per log point (GID_0), or
   [−0.45, +0.25] (GID_1).
3. **Separating own from neighbour effects is not credible.** The first-stage matrix is not
   degenerate: each lights term is driven mainly by its own instrument. But the two 2SLS
   coefficients have opposite signs and sum to about the same null (−0.15), with the regressors
   correlated 0.81 within pixel. Neighbour growth cooling a cell by twice what own growth warms
   it is not physically plausible. This is the collinearity pattern, not a finding. No
   Sanderson–Windmeijer F for each regressor was computed.

**What follows for the plan:**
- The main specification is the 10 km cell-level 2SLS, reported with its π profile, with the
  3×3 disc as a companion.
- Rings and the 1 km panel move to robustness, and the ring-engine wiring leaves the critical
  path.
- At 10 km `ntl_harm`'s ~4–7 km blur is smaller than the cell, which resolves inconsistency 6
  in favour of `ntl_harm`.
- The binding constraint is the precision and validity of the reduced form, so the instrument
  work in Phase 3 becomes the priority.

**Caveat on timing.** The grid decision rests on the first-stage profile, which does not involve
the outcome. The reduced forms and 2SLS were seen at the same time. They are null for every
aggregation, so no aggregation was favoured by its result.

## 5. Reproduce

```bash
python scripts/ring_check_build_panel.py scratch_nobackup/fs_profile/panel_nb.parquet   # ~1 min
python scripts/ring_check_regressions.py scratch_nobackup/fs_profile/panel_nb.parquet   # ~40 min, 16 cores
```

The regressions write `panel_nb_results.csv` next to the panel, one row per fit as it finishes.
Run them through `sbatch`, not an interactive session: `/scratch` is local to each node, and a
VSCode disconnect can kill child processes. The run recorded here was SLURM job 24160727.

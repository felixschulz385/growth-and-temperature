# Fork: Transit Corridors and a Growth Measure Other Than Lights

**Status: first thoughts, 2026-10-28. Not started.** The work happens on the branch
`fork/corridors-dose`. This page on `main` records the idea, its critique and the rules to fix
before anything runs. The retired project's result is unchanged:
[`../analysis/project-conclusion.md`](../analysis/project-conclusion.md).

## Why fork

- **The mining bound covers one kind of growth.** Each instrument identifies the effect of the
  growth it induces, with that growth's own mix of land-cover, waste-heat and aerosol changes.
  The channels have opposite signs, so the bound for the mining mix does not carry over to
  other mixes (conclusion §3).
- **A second instrument therefore adds a second bound, not a tighter first one.** Transit
  corridors induce a different kind of growth (trucking, services, market access). The fork's
  first aim is a bound for that kind.
- **Lights are the only growth measure so far.** One log point of lights is a different amount
  of activity in each sector. The fork's second aim is to check the result with a measure built
  from different data.

**What the fork keeps.** It keeps the specification in
[`src/analysis/spec.py`](../../src/analysis/spec.py):
- 10 km EASE-Grid 2.0 cells;
- MYD21A2 night LST, with day LST as the second outcome;
- cell FE + country × biome × year FE, SEs clustered by ADM1;
- Anderson–Rubin sets;
- decision rules (◇) written before each block runs.

## Part 1: a transit-corridor instrument

### Model

Cells on the route from a mine to its market gain growth when the mine is active:

$$Z_{it} = \sum_m w_{im}\, S_{mt}, \qquad
w_{im} = \mathbb{1}[i \in \text{route}(m \to d_m)] \cdot f(\text{distance along the route})$$

- $S_{mt}$ is the mine's shock: active or not, as in the project's instrument. Commodity price ×
  baseline output (the unbuilt baseline-share price shock) is the alternative.
- $d_m$ is the mine's destination: the nearest seaport (World Port Index), or the nearest city
  with more than 500k people in 2000 (GHSL). The rule is fixed before any result.
- **Routes are predicted, not observed.** They are least-cost paths on the 10 km grid, with cost
  from slope (SRTM) and the land mask; straight lines are the robustness check. Today's roads
  (OSM, GRIP) are partly a result of 2002–22 growth, and of the mines themselves.

The first stage and 2SLS follow `spec.py`, with three additions:
- **Exclusions.** Drop cells within 20 km of any mine, so the corridor estimate does not overlap
  the mining one. Drop cells near the destination, which grow for their own reasons.
- **Comparison at equal distance.** Add distance-to-nearest-mine band × year FE, so on-route
  cells are compared with off-route cells as far from a mine.
- **Recentering** (Borusyak and Hull 2023). Exposure is fixed by geography, so on-route cells are
  not random places: they are lower, closer to the coast and to cities. Subtract the expected
  exposure over alternative routes or permuted mine opening years.

Inference clusters by ADM1 (the specification) and by nearest mine; both are reported. The shocks
are mine-level, so the number of mines, not cells, sets the effective sample.

### Critique

- **Power.** The variation still comes from mine-level shocks. A truck route raises lights in a
  cell less than a mine does next door (the mining first stage already falls to 0.51 at ring 2).
  Night LST noise is correlated over tens of km, so it barely averages out along a corridor. A
  rough guess, with assumed inputs: 30× the cells at a quarter of the per-cell first stage gives
  F ≈ 71 × 30/16 ≈ 130, before clustering by mine.
- **Roads change temperature directly.** Haul-road dust, diesel black carbon, paving and clearing
  along roads change LST without going through lights. The donut cannot remove this, because the
  road runs through the treated cells. Two checks: the instrument's reduced form on Hansen forest
  loss and MAIAC AOD, and masking road pixels on the 1 km panel.
- **The compliers are narrow.** They are cells along resource haul routes, a group as specific as
  "cells near mines".
- **Story 1 still applies.** The association is regional, and a corridor IV compares cells with
  nearby off-route cells. It cannot see an effect shared across the region either.

### ◇ Gate C1, before any LST is loaded

Build the exposure and run only the first stage of lights on $Z$, clustered by mine.
- **Proposal, to fix on the branch before the first stage runs:** continue only if F ≥ 70, the
  level that gave the mining bound of ±0.3 K. Below that, the corridor bound would be wider than
  the mining one, so stop.
- For reference: tightening a bound to ±0.1 K needs F ≈ 640 (width scales roughly with 1/√F).

## Part 2: a growth measure other than lights

### What is available

No packaged dataset is global, annual, gridded, reliable for within-cell change, and built
independently of land cover.

| Dataset | What it is | Problem for this design |
|---|---|---|
| Kummu et al. 2025, *Sci Data* 12:178 | GDP per capita PPP, 1990–2022, 43,501 admin-2 units | Within an admin-2 unit, variation comes from the downscaling inputs. Check whether lights or population are among them. |
| DOSE (Wenz et al. 2023, *Sci Data* 10:425) | Reported regional output, 1,661 regions in 83 countries, 1960–2020, GADM-1 | ADM1 only; thin where mines are (Africa) |
| Yeh et al. 2020; Pettersson et al. 2023 (IJCAI) | Asset wealth predicted from Landsat, Africa (Pettersson: 1990–2019, about 3-year steps) | Africa only; changes over time are noisy (Yeh: up to about 50 % of temporal variation, at district level) |
| Meta Relative Wealth Index (Chi et al. 2022) | 2.4 km grid, low- and middle-income countries | One cross-section; unusable with cell FE |
| Jean et al. 2016-style models | Wealth from daytime imagery | See below |

**The models built on daytime imagery share inputs with the outcome.** They learn wealth from
surface reflectance: roofs, roads, vegetation and bare soil. Those same features drive LST, so a
treatment predicted from them builds the land-cover channel into the treatment, and part of the
treatment–LST link becomes mechanical. Lights do not have this problem.

### Choice: DOSE at ADM1 as the first test

DOSE is reported statistics, with no satellite input. It also moves the design to the regional
scale, where Story 1 says the association is.

| | |
|---|---|
| Unit | GADM-1 region × year, 2002–2020 (DOSE ends in 2020) |
| Outcome | area-weighted mean of the cells' night LST; day LST second |
| Treatment | log real GRP per capita (DOSE, 2015 prices) |
| Comparison | region FE + country × year FE |
| Instruments | the region's share of cells within 20 km of an active mine; its corridor exposure share |
| Clusters | region; country as a robustness check |
| Controls | weather (CHIRPS or ERA5-Land precipitation and soil moisture), as conclusion §5 proposes for any regional design |

### ◇ Sequence: one change per step

| Step | Unit | Measure | Instrument | Isolates |
|---|---|---|---|---|
| A | 10 km cell | lights | corridors (Part 1) | the second kind of growth |
| B | ADM1 | lights, aggregated | mines; corridors | the change of unit |
| C | ADM1 | DOSE GRP | mines; corridors | the change of measure |

Each step also reports the fixed-effects association. If B and C give different results, the
measure is responsible. If the project's cell result and B differ, the unit is.

### Critique

- **Coverage.** DOSE's 83 countries lean towards richer and larger ones. ◇ **Gate D1**: count the
  regions with DOSE data for 2002–2020 and any mine or corridor exposure, before any estimate.
  Proposal, to fix on the branch: with fewer than about 150 exposed regions in 15 countries, drop
  the IV at ADM1 and run only the association.
- **GRP counts output where it is produced.** A mine's value added enters regional GRP even when
  the profits leave the region. The mining first stage is then partly mechanical, and the growth
  it scales is output, not local activity. This matters less for corridors.
- **Power.** About 1,661 regions × 19 years, with few regions in many countries and country ×
  year FE. Expect a weak first stage.
- **The units differ.** K per log point of GRP and K per log point of lights are different
  quantities. Compare signs and the overlap of the AR sets, not magnitudes.
- **Averaging.** An ADM1 mean of LST mixes urban and rural, and day and night effects of
  different channels.

**Out of scope for now.** Imagery-based wealth (shares inputs with LST). Kummu et al. 2025 as a
robustness check once its downscaling inputs are known.

## Work list for the branch

| Item | Where it would land |
|---|---|
| World Port Index; city points for 2000 (GHSL); SRTM slope | new sources |
| Least-cost routes and exposure $w_{im}$ | `snl_mining` aggregation, or a new `corridors` step |
| DOSE | new source, joined on GADM-1 |
| ADM1 panel: region means of LST and lights, exposure shares | `assembly`; `spec.build_panel()` variant |
| Weather controls | new source (also a gap in conclusion §5) |
| Notebooks: Gate C1, Part 1, steps B and C | `output/notebooks/` on the branch |

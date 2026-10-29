# Project Conclusion

**Status: retired on 2026-09-30.** This is the closing record of the project. It states what was
found, why the project stops, what the result does and does not claim, and what a successor would
need. The plan, with every pre-specified decision rule and each block's outcome, is in
[`final-analysis-plan.md`](final-analysis-plan.md). The numbers are in the notebooks under
[`output/notebooks/`](../../output/notebooks/).

## The answer

- **No detectable local effect.** Growth, measured by night lights, has no detectable effect on
  night land surface temperature (LST) at the scale of a 10 km cell. The causal evidence covers
  one kind of growth only: growth induced by nearby mining (§3).
- **The causal effect is bounded.** The mining instrument rules out effects beyond about
  **±0.3 K per log point of lights** (95 % Anderson–Rubin set [−0.32, +0.28]). For a cell with
  strong lights growth (p90 over 2002–22, +1.42 log points), that is about ±0.4 K. The bound
  holds for mining-induced growth, with its own mix of land-cover, waste-heat and aerosol
  changes, and not for growth in general.
- **The observed association is regional, not local.** The fixed-effects association
  (−0.030 K per log point) shrinks with every tighter comparison. Between neighbouring cells it is
  a precise zero, so whatever links lights and night LST is shared across tens of kilometres.
- **The channels cannot be separated in this design.** These are land cover, waste heat and
  aerosols (§2.3).

**The bound is the result of record.** It is weaker than the project's aim, a causal and non-zero
local effect. It is also weaker than the fallback proposed in the 2026-09-19 review: a credibly
identified net effect, honestly scoped (§3).

## 1. Question and design

Does local economic growth change local temperature, separately from the global CO₂-driven trend?
Land-cover change and waste heat warm the surface; aerosols cool it. The net sign is an empirical
question.

| | |
|---|---|
| Panel | 10 km EASE-Grid 2.0 cells, \|φ\| ≤ 60°, 2002–2022: 25.3 M cell-years, 1.2 M cells |
| Outcome | MODIS Aqua night LST (MYD21A2), annual mean; day LST as the second outcome |
| Treatment | `log1p(ntl_harm)`: harmonised DMSP–VIIRS night lights, gas flares masked |
| Comparison | cell fixed effects + country × biome × year fixed effects; SEs clustered by ADM1 |
| Instrument | active mines within 20 km (S&P/SNL) |
| Estimand | the effect of growing faster than other cells in the same country-biome-year, at 10 km. The IV estimate is a LATE for cells near mines, and includes most spillovers into neighbouring cells. |

The specification is coded once, in [`src/analysis/spec.py`](../../src/analysis/spec.py) (plan
§2), and every notebook imports it.

The analysis was organised as three stories, each with decision rules (◇) written before the
block ran:

1. Is the association robust?
2. Can the instrument identify the effect?
3. Which channels produce it?

## 2. What was found

### 2.1 The association does not survive tighter comparisons (Story 1)

[`02_baseline.ipynb`](../../output/notebooks/02_baseline.ipynb). Night LST, K per log point of
lights:

| Check | Estimate | Reading |
|---|---|---|
| **B1** fixed-effects association | **−0.030 (0.006)**; country × year FE −0.027 (0.009); day LST −0.015 (0.012) | Sign stable across FE sets. With cell FE alone it is +0.025. |
| **B2** functional form | bins vs DN 0: (0, 7) −0.016 · [7, 20) −0.052 · [20, 30) −0.079 · [30, 50) −0.080 · ≥ 50 −0.051 (n.s.) | Cooling comes from lighting up, not brightening. By day, brightening warms (urban heat island sign). |
| **B3** tighter comparisons | country × biome −0.030 → ADM1 −0.020 → ADM2 −0.010 → **neighbour differences −0.0001 (0.0009)** | **Fails: the association is not local.** |
| **B4** satellite artefacts | snow-prone cells out −0.031 · DMSP era −0.055 · VIIRS era −0.050 · dim lights zeroed −0.017 | Not a snow, sensor or dim-light artefact. |
| **B5** reverse causality | by baseline climate: < 5 °C −0.047 · 5–13 °C −0.062 · 13–22 °C −0.025 · ≥ 22 °C −0.012 (n.s.); breakdown feedback θ\* = −0.0043 | The climate gradient argues against it. A feedback of only this size (θ\*), far below plausible, would produce the association on its own, so B5 cannot exclude it. |

**Why the neighbour-difference zero decides it.** It is precise: its SE is a sixth of B1's.
About half of the lights variation is cell-local (P5: the within-FE r² of a cell's lights with
its neighbours' is 0.48). Even an effect that spilled 85 % into neighbouring cells would leave
about −0.005 here, with t ≈ 5. So the zero means the association is shared between neighbours,
not that it is noisy.

The rule written before the run said a sign flip or |t| < 2 under neighbour differences means
the association "does not survive". It fired. The pre-specified reading is **explained away**,
and what remains is the IV bound.

Other checks held:
- land cover × year: −0.027;
- cell × period FE: −0.047;
- no anticipation in the lead of lights;
- lagged night LST does not predict lights.

### 2.2 The instrument bounds the effect but cannot test it (Story 2)

[`03_iv.ipynb`](../../output/notebooks/03_iv.ipynb) and
[`03b_iv_robustness.ipynb`](../../output/notebooks/03b_iv_robustness.ipynb).

| | Estimate |
|---|---|
| First stage (lights on mines within 20 km) | 0.064 (0.008), F = 71; spatial reach 1.00 / 0.85 / 0.51 in the own cell / ring 1 / ring 2 |
| Reduced form (night LST on mines) | −0.0017 (0.0097), p = 0.86 |
| 2SLS | −0.026 (0.150); **Anderson–Rubin 95 % set [−0.32, +0.28]** |
| Minimum detectable effect (80 % power) | 0.42 K per log point = 14× the fixed-effects estimate |
| I4 donut: cells with no mine within 10 km | F = 11.6; AR set [−2.17, +0.19]. Nothing is masked by the mines' own footprint. |
| I5 radii (pre-specified) | AR sets: 10 km [−0.12, +0.20] · 20 km [−0.32, +0.28] · 50 km [−0.90, +0.50]. The sets overlap, and the bound of record stays at 20 km. |
| R1–R9 robustness (2026-09-28) | Functional form, sensor era, flares, coverage, instrument choice, clusters, FE: all null. The favoritism instrument has no first stage (F ≈ 1.4). |

- **The IV cannot confirm or reject the association.** Its set contains the fixed-effects
  estimate, so the null counts only against effects outside the set. The two also average over
  different kinds of growth: the fixed-effects estimate over all growth in the panel, the IV over
  mining-induced growth. Even with full power, a gap between them could reflect that difference
  rather than bias in the association.
- **A better instrument will not fix this.** Testing an effect of −0.03 needs a first-stage F of
  about 14,000, not 71. Even the tightest radius, 10 km, has a minimum detectable effect 7× too
  large.

### 2.3 The channels cannot be separated (Story 3)

[`04_channels.ipynb`](../../output/notebooks/04_channels.ipynb).

- **M1, PM2.5 accounting.** Adding PM2.5 moves the night coefficient from −0.0300 to −0.0298.
  Lights do not raise PM2.5: γ = −0.098 (0.067). More PM2.5 goes with *warmer* nights:
  δ = +0.0028 (0.0010). PM2.5 therefore accounts for 0.9 % of the association.
- **ACAG PM2.5 is almost entirely regional at 10 km** (P5: within-FE r² with the neighbours'
  mean 0.97). The test can at most exclude a regional, ACAG-measured aerosol explanation.

Each candidate channel has a typical day/night signature. Each acts either on the cell itself or
over a region. The signs are approximate literature expectations:

| Channel | Scale | Day | Night | Fits a negative night effect? | Fits a regional association? |
|---|---|---|---|---|---|
| Built-up / impervious surfaces | local | + | ++ | No | No |
| Waste heat | local | + | + | No | No |
| Aerosols | regional | − | + | No: night is more negative than day | Yes |
| Vegetation clearing | local | + (tropics) | − (temperate, boreal) | Possibly | No |
| Irrigation / greening | local | − | small | Weakly | No |
| Snow-brightened lights (artefact) | regional | − | − | Yes | Yes, but B4 found no snow effect |
| Heat slows growth (reverse causality) | regional | − | − | Yes | Yes, but the B5 gradient runs against it |
| Regional weather or economic shocks | regional | any | any | Yes | Yes |

What the table implies:
- **Local physical channels are ruled out as the source**: they would survive the
  neighbour-difference comparison.
- **The one regional physical channel, aerosols, has the wrong day/night pattern.**
- **The most parsimonious reading left is regional confounding.** This is an interpretation,
  not a test; weather controls would be the test (§5).

A mechanisms section cannot separate the channels in this design, because all three things a
decomposition needs are missing:
1. **A total effect to decompose.** There is no cell-level association (B3), and the causal total
   is only bounded (I2).
2. **Mediators measured at the cell scale.**
   - ACAG PM2.5 is regional.
   - ESA CCI land cover is coarse and changes its classes slowly.
   - Waste heat is unmeasured; lights proxy it, which is circular.
3. **Identified links from lights to each mediator.** Each link has the identification problem
   of Story 2, and there is no instrument per channel.

With better data, the most a mechanisms section can deliver is **accounting, not
identification**. That means lights → mediator associations times external physical
sensitivities (e.g. Duveiller, Hooker and Cescatti 2018), checked against the total.

**The day/night difference is not what blocks this.** It is the one piece of evidence that
already discriminates between channels. Its limit is that two numbers can exclude channels but
cannot split an effect among five or more channels and confounders.

## 3. What the result claims, and what it does not

**It claims:**
- For cells near mines, the local effect of growth on night LST at 10 km is not larger than
  about ±0.3 K per log point of lights.
- The fixed-effects association between lights and night LST is not a cell-level phenomenon.

**It does not claim:**
- that growth has no effect on temperature;
- that the bound applies to other kinds of growth (agriculture, manufacturing, services,
  transport corridors). Each kind changes land cover, waste heat and aerosols in its own mix, and
  the channels have opposite signs. A small net effect for the mining mix therefore limits
  neither its separate channels nor the net effect of another mix. "Per log point of lights" is
  also a different amount of activity in each sector;
- that pollution is not the channel;
- that the regional association is either causal or confounded.

It claims only that this design, at this resolution, cannot detect an effect below the bound.

**Against the review** (*"Reviewing a Research Project: Growth, Night Lights, and Local
Temperature"*, Scientific Reasoning, 2026-09-19):
- **Its measurement concerns were checked where the data allowed.**
  - Selection by clear-sky compositing: coverage does not respond to lights (P3).
  - Day vs night: day and night are estimated separately (B1, B2).
  - Sensor artefacts: B4.
- **Its design responses were run.** The donut ran (I4), and the radius comparison was
  pre-specified (I5). Cell-specific trends are approximated by cell × period FE (B3d).
- **Its fallback is not available.** The review's fallback was "a credibly identified net effect,
  honestly scoped to resource- and favoritism-driven growth with a land-cover footprint". That
  assumed a local effect exists to be scoped. The data say the net effect is not identified
  (Story 2) and there is no local association to scope (Story 1).

## 4. The decision

**Retire the project.** Nothing left in this design can move the result.
- **More checks of the association** would not revive it. It already failed the pre-specified
  comparison.
- **More instruments** cannot reach the power needed.
- **Channel work** has no local effect to decompose.

Two uses of the work remain:
- **Report the bound.** "Growth induced by nearby mining has no large net effect on local night
  temperature", as a short note or as a section of another paper. The notebooks hold everything such a note needs.
- **Start a new project on the regional association.** It would be a different question
  (why do regions whose lights grow have cooler nights?) with a different design (§5). It is not
  a continuation of this one.

## 5. For a successor

**Added 2026-10-28.** A fork (a transit-corridor instrument; regional output from DOSE as a growth
measure other than lights) and three further ideas are recorded in
[`../successors/`](../successors/).

**Added 2026-10-29: the fork's outcome.** The fork is complete and merged into `main`. **It adds
no causal result; the mining bound (§3) stands as the result of record.**

- **Transit corridors** ([`corridors-prespec.md`](../successors/corridors-prespec.md) §8):
  predicted bulk-mine-to-port routes follow real roads, but they do not move lights in the
  cells they cross (first-stage F = 0.08 against a bar of 70).
- **DOSE regional output at ADM1** ([`dose-prespec.md`](../successors/dose-prespec.md) §4). Three
  instruments were tried: the mine count, mine × world price and regional favoritism. None
  moves regional lights or GRP enough to identify anything (largest F = 4.3 against 10).
- **The regional association is there, and it is not specific to lights.** Within a
  country-year, night LST is 0.26 K lower per log point of regional lights (p < 0.001) and
  0.13 K lower per log point of reported GRP (p = 0.09). That is the sign of §2.1. It is
  descriptive: hot years also lower regional output, which alone would produce it.
- **For the regional question below,** this narrows what a successor needs. Reported output is
  usable as a growth measure; mine and favoritism instruments are not strong enough at ADM1; and
  the heat → growth feedback is the confounder to address first.

### The regional question

What a region-level project would need:
- **Weather controls** against the confounding reading: precipitation and soil moisture (CHIRPS
  or ERA5-Land). Weather is not caused by lights, so it is a safe control.
- **ERA5-Land 2 m temperature** to estimate the heat → growth feedback directly. ERA5-Land's land
  surface is static, so it does not respond to local growth. Regressing lights on its temperature
  anomalies estimates the feedback, and the effect of lights then follows under one explicit
  assumption: that simultaneity is the only bias.
- **A region-level unit** (ADM2, or 50 km aggregates) and aerosol data at a scale where it varies:
  MAIAC AOD at 1 km, or ACAG at its native resolution.
- **Independent measurements**, because they are sensor-specific:
  - lights: NASA Black Marble VNP46A4;
  - other treatments: GHSL built-up area, WorldPop population;
  - other outcomes: Landsat LST on a subsample, weather stations.

### Specified in this project but never run

| Area | Check | Specified in |
|---|---|---|
| Association | weather controls; MODIS snow cover instead of the snow proxy; Black Marble; independent treatments and outcomes; the ERA5-Land feedback route; coefficient-stability bounds (needs R² in `duckreg`) | plan §4 |
| IV | I3 (the bound against observed 2002–22 warming); baseline-share price shock; masked 1 km donut; event study at mine openings; shift-share diagnostics; GLASS-AVHRR placebo for 1992–2001; favoritism at ADM2; complier profile; equivalence (TOST) bounds | plan §5 |
| Channels | land-cover fractions and PM2.5 as outcomes of the instrument; accounting with Hansen forest loss, ESA CCI fractions, MAIAC AOD, MODIS NDVI/ET and Duveiller et al. sensitivities; baseline land-cover strata; GLASS vs LST divergence; heterogeneity by biome, aridity, PM2.5 and income; outcomes in neighbouring cells | plan §6 |
| Cross-cutting | MYD11A1 vs MYD21A2 by land-cover class (emissivity bias); coverage-balanced outcome with Lee bounds; panel end 2020 vs 2022; Conley SEs; a specification curve over FE, functional form, sensor era and grid | plan §7 |

### Data and pipeline gaps

| Gap | Where it would land |
|---|---|
| Weather (CHIRPS or ERA5-Land precipitation, soil moisture, 2 m temperature) | new FETCH/PREPARE source |
| Snow cover (MODIS MOD10/MYD10) | new source, or a band of the MODIS stream |
| NASA Black Marble VNP46A4; GHSL built-up; WorldPop population | new sources |
| MAIAC AOD 1 km, Hansen annual forest loss, MODIS NDVI/ET | new sources |
| ESA CCI per-class fractions (now aggregated by `mode`, which hides built-up fractions) | `average` resampling in `assembly.sources.esacci` |
| Baseline-share price shock; distance to the nearest mine (for a masked donut) | `snl_mining` aggregation and PREPARE; assembly |
| Treatment rings to 30 km on the 1 km panel (the engine exists, not wired) | a ring stage in assembly ([`design/03`](../design/03-neighbourhood-engine.md)) |
| `glass_avhrr` in the assembly, after month-first compositing | `data.yaml`; `glass/avhrr.py` |
| Conley SEs; R² | `duckreg` |

### Data caveats a successor must know

- **Aqua night LST must not run past 2022.** The night overpass drifts from 2023 on, and the drift
  differs by biome ([`drift-and-pm25-coverage-checks.md`](drift-and-pm25-coverage-checks.md) §1).
- **ACAG PM2.5 carries almost no cell-local variation at 10 km** (P5).
- **Mining data.** Opening and closing years are mostly LLM-imputed. The price shock is not a
  clean shift-share, because its exposure moves with openings
  ([`../data/snl_mining/README.md`](../data/snl_mining/README.md)).
- **Lights.** `ntl_harm` is continuous, and the dim range (DN < 7) moves with the 2013
  DMSP→VIIRS splice. Dim-only cells carry 34 % of the within-cell lights variation (P6).
- **Every data source** is described in [`../data/README.md`](../data/README.md).

## 6. Reproducing the results

1. **Environment.** `conda env create -f environment.yml` (env `gnt`), and a `duckreg` source
   checkout next to the repository or at `DUCKREG_PATH`.
2. **Panel.** `python -m src.cli assemble create --config orchestration/configs/data.yaml --grid 10km`.
   The analysis panel is derived from it and cached by `spec.build_panel()` at
   `<data root>/_analysis_panel_10km_v2.parquet`: 25.3 M rows, about 2.9 GB, about 2 minutes to
   build.
3. **Notebooks, in order**, executed from `output/notebooks/` (the notebooks find the project
   root from there):
   ```bash
   jupyter nbconvert --to notebook --execute --inplace --allow-errors <notebook>.ipynb \
       --ExecutePreprocessor.timeout=-1
   ```
   On SLURM with 16 CPUs and 128 GB, the runtimes were:
   - `01_pre_analysis`: 42 min;
   - `02_baseline`: 2 h 50 min;
   - `03_iv`: 70 min.
4. **Parallel runs.** `duckreg` locks one database file per data path. Give each concurrent job
   its own file with `DUCKREG_DB=<data root>/.duckreg/<job>.db`.

**Runs of record**, with outputs committed in the notebooks:

| Notebook | Run | SLURM job |
|---|---|---|
| `01_pre_analysis` | 2026-09-30 | 24402428 |
| `02_baseline` | 2026-09-30 | 24402428 |
| `03_iv` | 2026-09-30 | 24407713 |
| `04_channels` | 2026-09-29, not re-run after the split | 24273525 |
| `03b_iv_robustness` | 2026-09-28, carried over from the former `regression.ipynb` | — |

## 7. Timeline

| Date | Step |
|---|---|
| 2025 | Early pipeline and 1 km runs (`output/analysis/legacy/`). A day-LST run with pixel + country × year FE gave a *positive* association, +0.0049 (0.0019). |
| 2026-07 to 2026-09 | Data backbone rebuilt: grid, storage, ingest, MODIS and GLASS pipelines ([`../design/`](../design/)). |
| 2026-09-19 | Scientific Reasoning review of the project. |
| 2026-09-28 | Fixed specification. First results: the IV is null, and controlling for PM2.5 does not move the association. Author and supervisor lean towards stopping. The ring check ([`10km-ring-check.md`](10km-ring-check.md)) and the drift and PM2.5 coverage checks are run. |
| 2026-09-29 | Pre-analysis and core. Both nulls turn out to be absences of power, not negative findings. |
| 2026-09-30 | Analysis reorganised into three stories with pre-specified rules. Story 1 is explained away (B3); the donut and radius checks confirm the bound. **Project retired.** |

**Older outputs were not updated to the final specification.** These are the poster and
presentations, the results website (`output/webpage/`), the LaTeX tables
`output/tables/table_*.tex`, and `output/analysis/legacy/`. They predate it, and their numbers
should not be cited.

## 8. Questions the project answered

| Question | Answer |
|---|---|
| Is the lights–night LST association worth explaining? | Not at the cell level: it vanishes between neighbouring cells (B3). |
| Does the mining IV say anything? | Yes, a bound of about ±0.3 K per log point. It cannot test an effect of the association's size (I2). |
| Is the IV null masked by the mines' own footprint? | No (I4). |
| Does the result depend on the instrument radius? | No: the sets overlap (I5). |
| Is pollution the channel? | Not the sole one (night more negative than day), and ACAG cannot measure a local channel (M1, P5). |
| How large would reverse causality need to be? | Small (θ\* = −0.0043, 12× below a plausible feedback), but the climate gradient points the other way (B5). |
| Do selection by clouds, sensor changes, snow or dim lights drive the association? | No (P3, B4). |
| Can the channels be separated? | Not in this design (§2.3). |
| Does the Aqua orbital drift matter? | Not through 2022; from 2023 it does. |
| Is the ring IV needed? | No. Own-cell 2SLS already includes most spillovers ([`10km-ring-check.md`](10km-ring-check.md)). |

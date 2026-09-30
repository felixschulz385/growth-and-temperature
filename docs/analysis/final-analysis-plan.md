# Final Paper Analysis Plan

Planning note, first written 2026-09-28 and restructured 2026-09-29. It lays out the analysis
flow for the final paper. It combines all sources now in the assembly with the critique from the
Scientific Reasoning review of this project (*"Reviewing a Research Project: Growth, Night
Lights, and Local Temperature"*, 2026-09-19). What each source contains, and its caveats, is in
[`../data/README.md`](../data/README.md).

**Why it was restructured.** The first results were inconclusive. The mining-IV second stage is
null. Controlling for PM2.5 did not move the negative TWFE coefficient, and pollution was
believed to be the only channel that could make that coefficient negative. On that basis, the
author and supervisor lean towards stopping the current path of the project. Before they commit,
the plan now separates the analysis into three stages, each with its own notebook in
`output/notebooks/`:

| Stage | Notebook | Purpose | Status |
|---|---|---|---|
| Pre-analysis (§3) | `01_pre_analysis.ipynb` | Checks the core relies on. None involves the outcome's response to the treatment. | run 2026-09-29 (§1) |
| Core analysis (§4) | `02_core_analysis.ipynb` | The two central tests, and whether each could have detected what it is taken to rule out. | run 2026-09-29 (§1) |
| Post-analysis (§5) | `03_post_analysis.ipynb` | Robustness, instrument extensions, channels and interpretation. | **unresolved** |

All three notebooks take the specification and helpers from `src/analysis/spec.py`, so they
cannot drift apart on the outcome, fixed effects, clusters, instrument or sample.

This is a plan, not a record of decisions made. Every "◇" decision rule says in advance which
result moves the analysis onto which branch, so a later reader can tell what was decided before
the data was seen. The pre- and core-analysis rules added on 2026-09-29 were written before
either notebook ran in its new form. The C4 rules were written after the country × year table
in §1 had been seen, so they are not blind to it.

## 1. Where the project stands

**Current estimates** (the 2026-09-28 run of the former `regression.ipynb` on the fixed
specification of §2; 25.1 M pixel-years; SE clustered by ADM1):

| | pooled OLS | + pixel FE | TWFE | 2SLS |
|---|---:|---:|---:|---:|
| `log1p(ntl_harm)` → night LST | −0.40 (0.29) | +0.025 (0.012) | **−0.030 (0.006)** | −0.026 (0.150) |

First stage +0.064 (0.008), F ≈ 71. Reduced form −0.0017 (0.0097). Day LST: TWFE −0.015 (0.012),
2SLS +0.10 (0.23). GLASS air temperature: TWFE −0.018 (0.004), 2SLS +0.08 (0.09). Every
alternative instrument has a strong first stage and a null 2SLS.

**The PM2.5 control result** (the former `night_lst_mining_pm25.ipynb`, commit `a355ac2`,
exported to `output/tables/night_lst_mining_pm25_10km.*`). It is night LST at 10 km, but with
pixel + country × year FE, country clusters and flares kept:

| | TWFE | TWFE + PM2.5 | 2SLS | RF: LST on Z | RF: PM2.5 on Z |
|---|---:|---:|---:|---:|---:|
| `log1p(ntl_harm)` | −0.0227 (0.0110) | −0.0221 (0.0114) | −0.104 (0.163) | | |
| `pm25` | | **+0.0032** (0.0017) | | | |
| `mine_count_20km` | | | | −0.0070 (0.0111) | **−0.271** (0.137) |

First stage 0.067 (0.012), F ≈ 30. Two things in this table bear on the stop decision:

- **PM2.5 enters with a positive sign.** Within a cell, more PM2.5 goes with *warmer* nights.
- **The implied γ is negative.** From the identity β_short − β_long = γ·δ, lights move ACAG
  PM2.5 by γ ≈ (−0.0227 + 0.0221) / 0.0032 ≈ −0.19 per log point. Mines lower PM2.5 directly.

With γ ≤ 0 and δ > 0, the control has no pathway to make β less negative. "Controlling for
PM2.5 did not move the coefficient" is then the expected result, whether or not pollution
matters. Core step C4 re-ran the table in the §2 specification, with γ and δ reported
explicitly and day LST added (below).

**Pre- and core-analysis results** (2026-09-29, SLURM jobs 24253659 and 24273525; full panel,
§2 specification). The pre-analysis ◇ outcomes:

| | Result | ◇ |
|---|---|---|
| P3 | Night coverage on lights +0.0002 (0.0004) | selection not material |
| P4 | First stage 0.064 (0.008), F = 71; π profile 1.00 / 0.85 / 0.51 | Wald and AR both reported |
| P5 | Within-FE r² of cell PM2.5 with its ring-1 mean 0.97 (lights 0.48) | **C4 speaks to regional aerosol only** |

The core reproduces the "current estimates" above and adds:

| | night LST | day LST |
|---|---:|---:|
| C1 TWFE, country × biome × year | −0.0300 (0.0056) | −0.0147 (0.0123) |
| C1 TWFE, country × year | −0.0265 (0.0086) | −0.0151 (0.0150) |
| C2 2SLS; AR 95 % CI | −0.026 (0.150); [−0.32, +0.28] | |
| C3 MDE (80 % power) | 0.42 K per log point = 14× \|TWFE\| | |
| C4 β_short → β_long | −0.0300 → −0.0298 | −0.0147 → −0.0143 |
| C4 γ (PM2.5 on lights) | −0.098 (0.067) | −0.098 (0.067) |
| C4 δ (LST on PM2.5 given lights) | +0.0028 (0.0010) | +0.0035 (0.0024) |
| C4 share of β_short via PM2.5 | 0.9 % | 2.3 % |

Which ◇ rules fired:

- **C1:** the TWFE sign does not depend on the FE set.
- **C3: the AR CI contains the TWFE estimate**, so the IV says nothing about whether the TWFE
  association is causal. It rules out effects beyond about ±0.3 K per log point, i.e. more than
  about 0.4 K at the p90 of 2002–22 lights growth among ever-lit pixels (+1.42 log points).
- **C4: γ not significantly positive** (the control could not move β), **δ ≥ 0** (the
  aerosol-cooling premise fails for this measure) and **night more negative than day**
  (pollution is unlikely the sole channel). With P5, the test can at most exclude a regional,
  ACAG-measured aerosol explanation.

Both central results are absences of power, not negative findings. They support "these two tools
cannot answer the question at this resolution", not "the lights–temperature link is not causal"
or "pollution is not the channel". The details are in the notebooks' Result cells.

An older 1 km run (`output/analysis/duckreg/modis_logntlharm_pm25/`: day LST, pixel +
country × year FE, 2000–2020) gives a *positive* TWFE lights coefficient (+0.0049, SE 0.0019)
and PM2.5 +0.014 (0.004). The sign of the TWFE association is therefore not settled across
specifications either (C1).

**Inconsistencies:**

| # | Issue | Status |
|---|---|---|
| 1 | The review says the main effects come from night LST; the old headline used day LST. | Resolved: night LST (§2). |
| 2 | The notebook's "extensive/intensive margin" is an NTL threshold split; the review proposes an ESA CCI land-cover split. | Open, post-analysis (needs ESA CCI class fractions). |
| 3 | `NTL_HI` was 7 in the regressions and 20 in the descriptives. | Resolved: 20 everywhere. |
| 4 | The only regional FE was country × year. | Resolved: country × biome × year. |
| 5 | The distance-ring model of [`00`](../design/00-backbone-overview.md) vs own-cell lights. | Resolved: own-cell 2SLS already estimates ≈ β_0 + 0.85·β_1 + 0.51·β_2 in the §2 specification (P4; 0.86 / 0.53 in [`10km-ring-check.md`](10km-ring-check.md)). |
| 6 | Raw VIIRS vs `ntl_harm` ([`04-ingest.md`](../design/04-ingest.md) §1). | Resolved: `ntl_harm` at 10 km, wider than its blur. |
| 7 | The PM2.5-control table uses country × year FE, country clusters and keeps flares; the fixed specification uses country × biome × year FE, ADM1 clusters and drops flares. | Resolved: C4 re-ran it in §2 (`output/tables/core_night_lst_mining_pm25_10km.*`); same reading. |
| 8 | §2 restricts the sample to \|φ\| ≤ 60°; no notebook applies the cut. | Open, post-analysis. |

## 2. Fixed specification

Everything outside this is robustness or exploration. It is coded once, in `src/analysis/spec.py`.

- **Estimand:** the local thermal effect of growing faster than other cells in the same
  country × biome in the same year, measured at the level of a 10 km cell. It includes
  spillovers within the cell and, through the instrument's spatial reach, most spillovers into
  neighbouring cells: own-cell 2SLS estimates Σ_r β_r · π_r/π_0 ≈ β_0 + 0.85·β_1 + 0.51·β_2
  ([`10km-ring-check.md`](10km-ring-check.md)). It is identified as a LATE for mine-proximate
  compliers, and the paper's claims are scoped to that population.
- **Outcome:** `lst_night_mean` (MYD21A2). Day LST is the second outcome wherever aerosols are
  concerned (C4). GLASS air temperature and the valid counts are diagnostics.
- **Treatment:** `log1p(ntl_harm)` in the own cell; the 3×3 disc mean is a companion.
  `NTL_HI = 20` everywhere. Mask `flare_band > 0`.
- **Instrument:** `mine_count_20km`.
- **FE:** `pixel_id + GID_0^biome_id^year`.
- **Inference:** clustered by ADM1, with Anderson–Rubin CIs for every IV. Conley SEs are a
  post-analysis companion (they are not in `duckreg` yet).
- **Sample:** Aqua years 2002–2022, 10 km base grid, non-null night LST, flares dropped.
  - The MODIS night-LST panel must **never extend past 2022**: Aqua's night overpass drifts
    from 2023 on, and the drift differs by biome
    ([`drift-and-pm25-coverage-checks.md`](drift-and-pm25-coverage-checks.md) §1).
  - Whether to end in 2020 instead is still open. The notebooks use 2022, and a 2020 cutoff is
    a post-analysis check.
- **Report with every 2SLS:** the reduced form and the first-stage profile π_r across own
  cell, ring 1 and ring 2. With one instrument, the reduced form is the evidence and the 2SLS
  is its scaling.

## 3. Pre-analysis — `01_pre_analysis.ipynb`

What the core relies on and can be checked without regressing the outcome on the treatment or
the instrument. Nothing here is a reduced form, a TWFE of LST on lights, or a 2SLS.

- **P1. Panel and sample.** Rows, pixels, countries and years. Moments and missing shares of
  night and day LST, lights, the instrument and PM2.5.
- **P2. Treatment series.**
  - The lit share by year at DN 7, 20 and 30. Check that the DMSP→VIIRS artefact is absent
    at `NTL_HI = 20`.
  - The flare mask: how many rows it drops, and how bright they are.
- **P3. Outcome measurement.** MODIS composites only clear-sky observations. If lights, mines
  or haze change how often a cell is observed, the annual LST is a selected mean.
  - Regress within-pixel night coverage (`cov_night`) on lights, on the instrument and on
    PM2.5, with the §2 FE.
  - Regress day coverage (`cov_day`) on PM2.5, to see whether hazy years lose daytime
    observations. That would hide part of the aerosol cooling C4 looks for.
  - ◇ Selection is **material** if a one-log-point rise in lights moves `cov_night` by more than
    0.01 (1 pp of the pixel's best-year coverage). If it does, C1 also reports the TWFE on
    pixel-years with `cov_night ≥ 0.75`.
  - Aqua drift: done ([`drift-and-pm25-coverage-checks.md`](drift-and-pm25-coverage-checks.md)
    §1), not repeated. It is negligible through 2022.
- **P4. Instrument.**
  - Where its variation lives: the share of pixel-years with a mine within 20 km.
  - Balance in levels: night LST, lights and PM2.5 for pixels that ever have a mine vs pixels
    that never do. This shows what pixel FE must absorb.
  - First stage, F and the π_r profile across own cell, ring 1 and ring 2.
  - ◇ If F < 10, the core reports AR inference only. The AR CI is reported either way.
- **P5. PM2.5 as a measure of the mediator.** Controlling for a noisy or over-smoothed mediator
  under-controls. ACAG combines satellite AOD with a chemical transport model that runs at a
  much coarser resolution than 10 km. Part of its year-to-year variation within a cell may
  therefore be regional rather than local.
  - Within-FE local signal: the squared within-FE correlation (r²) between a cell's PM2.5 and
    the mean over its 8 neighbours. It is the product of the two slopes, cell on neighbours and
    neighbours on cell, under the §2 FE. Lights give the benchmark.
  - The clear-sky coverage check is done
    ([`drift-and-pm25-coverage-checks.md`](drift-and-pm25-coverage-checks.md) §2): the
    instrument does not move ACAG's likely data support.
  - ◇ If the within-FE r² of PM2.5 with its ring-1 mean is ≥ 0.9, ACAG carries almost no
    cell-local variation at 10 km. C4 can then speak only to regional aerosol. A null in C4 is
    not evidence against a local pollution channel.

## 4. Core analysis — `02_core_analysis.ipynb`

The two results the stop decision rests on, re-run on the §2 specification, each with the check
that says what it can and cannot rule out. The core does not make the decision. It states which
effect sizes the IV excludes, and whether the PM2.5 test had the power to move the TWFE
coefficient.

### Claim A — "the mining IV is null"

- **C1. The association the IV is compared with.** Estimate the OLS ladder (pooled → pixel FE →
  TWFE) for night and day LST. Estimate TWFE under both country × biome × year and
  country × year FE.
  - ◇ If the TWFE sign differs between the two FE sets, the core reports "the negative TWFE
    coefficient" as specific to the FE set, not as a feature of the data.
- **C2. The IV.** Estimate the reduced form, the 2SLS and the 3×3 disc 2SLS. Report the
  **Anderson–Rubin CI**. With one instrument it comes exactly from three reduced forms: of
  Y, of Y − X and of Y + X on Z (`spec.anderson_rubin_ci`).
- **C3. What the null rules out.**
  - The minimum detectable effect at 80 % power (2.8 × SE).
  - The gap between the 2SLS and TWFE estimates, relative to the 2SLS SE.
  - The economic scale of lights changes: the distribution of within-pixel Δ`log1p(ntl_harm)`
    from 2002 to 2022, and the temperature change each estimate implies at its p50, p90 and p99.
  - ◇ If the AR CI contains the TWFE estimate, the IV says nothing about whether the TWFE
    association is causal. "The IV is insignificant" then counts against effects outside the
    AR CI only. It does not count against an effect of the TWFE size.

### Claim B — "controlling for PM2.5 does not move the TWFE coefficient"

- **C4. PM2.5 channel accounting**, on the pixel-years with PM2.5, separately for night and day
  LST:
  - short: `Y ~ X | FE`;
  - long: `Y ~ X + pm25 | FE`, with δ = the PM2.5 coefficient;
  - auxiliary: `pm25 ~ X | FE`, with γ = the lights coefficient.

  The omitted-variable identity β_short − β_long = γ·δ holds exactly in-sample. The notebook
  checks it numerically. It also reports the reduced form of PM2.5 on the instrument, and the
  2SLS with and without the PM2.5 control on the same sample.

  This uses PM2.5 as a descriptive decomposition of the OLS association: how much of it moves
  with PM2.5. It does not identify a causal direct effect, which is why §5 still says PM2.5
  must not be a control in causal claims.
- **Reading rules**, in order:
  - ◇ **γ not significantly positive:** within a cell, lights do not raise ACAG PM2.5. The
    control *cannot* move β, whatever the true pollution channel. The "no change" result then
    says that ACAG does not register a lights-linked PM2.5 change. It does not say that
    pollution is not the channel. Read this together with P5.
  - ◇ **δ ≥ 0:** within a cell, more PM2.5 goes with warmer (or unchanged) LST, so the
    aerosol-cooling premise fails for this measure and outcome.
  - ◇ **γ > 0 and δ < 0, but |γ·δ| < ¼ |β_short|:** PM2.5, as ACAG measures it, accounts for
    little of the association.
  - ◇ **Day vs night.** Aerosol dimming cools the surface by day. At night aerosols mainly
    affect longwave radiation and tend to warm the surface, so the pollution channel predicts
    a negative coefficient for **day** LST, not night. If the TWFE is more negative at night
    than by day, pollution is an unlikely sole explanation of the night coefficient. The premise
    that pollution is the only channel for a negative sign then needs a second look before it
    supports stopping. Land-cover change can lower night LST, for example.

## 5. Post-analysis — `03_post_analysis.ipynb` (unresolved)

**Status: unresolved.** Which of these steps run, in what order and with which decision rules is
left open until the go/no-go decision. The notebook holds the robustness blocks already run in
the former `regression.ipynb` (2026-09-28), carried over with their outputs and not re-run, and
the list below.

**Already run (carried over, not re-run):**

| Block | Finding |
|---|---|
| R1 NTL functional form | asinh, lit indicators at 20 and 30: all rescale the same null reduced form |
| R2 DMSP→VIIRS break | 2SLS unchanged; the OLS slope is ~40 % weaker in the VIIRS era |
| R3 gas flares | no change |
| R4 coverage-restricted 2SLS | unchanged at `cov_night ≥ 0.75` and `≥ 0.9` (the selection test itself moved to P3) |
| R5 GLASS, VIIRS | GLASS 2SLS null; VIIRS first stage weak (F ≈ 5) (day LST moved to C1) |
| R6 instrument choice | 10/50 km, price shock, over-identified: all F ≥ 22, all 2SLS null |
| R7 inference | ADM1 the most conservative cluster; HC1 about 6× too small |
| R8 FE (2SLS) | country × year: −0.086 (0.174) |
| R9 political favoritism | first stage dead (F ≈ 1.4) |

**Candidate steps, unresolved:**

- **Measurement:**
  - 11A1 vs 21A2 night LST by land-cover class (emissivity bias; 5 tiles × 3 years);
  - the entanglement share of ΔNTL in cells with built-up change;
  - a coverage-balanced outcome with Lee-type bounds;
  - panel end 2020 vs 2022;
  - the \|φ\| ≤ 60° cut.
- **Instruments:**
  - a baseline-share price-shock (exposure frozen at 2002, so only world prices vary);
  - a masked donut outcome;
  - an event study around mine openings (footprint vs ring cells);
  - shift-share diagnostics (Rotemberg weights, share balance, shock-level SEs);
  - a GLASS-AVHRR pre-period placebo for 1992–2001;
  - favoritism at ADM2.
- **Inference and specification:**
  - Conley SEs;
  - a specification curve over FE, functional form, sensor era, grid (5/10/25 km, grid-shake),
    lights aggregation and 1 km rings.
- **Channels and heterogeneity:**
  - land-cover fractions and PM2.5 as outcomes of the instrument;
  - baseline land-cover strata;
  - GLASS vs LST divergence;
  - heterogeneity by biome, aridity, baseline PM2.5 and income;
  - the outcome in neighbouring cells (SUTVA).
- **Interpretation:**
  - a complier profile;
  - TOST equivalence bounds;
  - the reframing rules ("growth with a land-cover footprint").

**Design notes for these steps** (from the review, still valid if they run):

1. **Land-cover strata must be baseline strata.** Growth partly causes conversion, so splitting
   by "converted vs never converted" selects on an outcome of the treatment.
   - Define strata by 2002 built-up fraction and dominant class.
   - Report land-cover change separately, as an outcome of the instrument.
2. **PM2.5 must not be a control in a causal claim.** It is a mediator. C4 uses it as a
   descriptive decomposition only.
3. **`mine_priceshock_*` is not yet a clean shift-share.** Its exposure moves with openings and
   closures. Freeze exposure at baseline before applying Borusyak–Hull–Jaravel or
   Goldsmith-Pinkham diagnostics.
4. **ESA CCI is aggregated by `mode` at coarse grids**, which hides built-up fractions. Add
   per-class fraction variables (`average` of class indicators) first.

| Review lens | Threat | Check | Stage |
|---|---|---|---|
| Measurement | NTL ≈ the land-cover mechanism | entanglement share; baseline strata; land cover as an outcome | post |
| Measurement | Emissivity–land-cover bias in the outcome | 11A1 vs 21A2 | post |
| Measurement | Compositing drops hazy observations | coverage on lights, Z, PM2.5 | **pre (P3)**; bounds post |
| Measurement | Day LST reacts to albedo | day vs night | **core (C1, C4)** |
| Design | Weak or non-local instrument | F, π profile | **pre (P4)** |
| Design | On-site extraction physics breaks exclusion | donut; event study | post |
| Design | Mine locations and openings are not random | baseline shift-share; placebo; share balance | post |
| Causal | A null may hide offsetting channels | AR CI and MDE; PM2.5 accounting | **core (C2–C4)**; mechanism outcomes post |
| Causal | Exchangeability and SUTVA | π profile; neighbouring-cell outcome | pre (P4); post |

## 6. Ordering

```
[§2] FIXED SPEC ─ 10 km · night LST · log1p(ntl_harm), NTL_HI=20, flares dropped · Z = mine_count_20km
     │            FE pixel + ctry×biome×year · ADM1 clusters · 2002–2022        (src/analysis/spec.py)
     ▼
[01] PRE-ANALYSIS ─ no outcome-on-treatment regressions ─────────────────────────────────
     P1 panel · P2 DMSP→VIIRS, flares
     P3 coverage ~ {lights, Z, PM2.5}        ◇ material selection? ──► C1 adds a coverage-restricted TWFE
     P4 Z variation, balance, first stage, π  ◇ F < 10? ──► AR only
     P5 PM2.5 local signal (r² with ring 1)   ◇ r² ≥ 0.9? ──► C4 speaks to regional aerosol only
     ▼
[02] CORE ─────────────────────────────────────────────────────────────────────────────────
     A  C1 OLS ladder, night + day, two FE sets   ◇ sign depends on FE? ──► say so
        C2 RF · 2SLS · disc · AR CI
        C3 MDE · IV−TWFE gap · scale of ΔNTL     ◇ AR CI ∋ TWFE? ──► the IV cannot speak to it
     B  C4 β_short − β_long = γ·δ, night + day; IV with/without PM2.5
          ◇ γ ≤ 0 → no power · δ ≥ 0 → premise fails · |γδ| < ¼|β| → PM2.5 explains little
          ◇ night more negative than day → pollution unlikely the sole channel
     ▼
     GO / NO-GO (author + supervisor)
     ▼
[03] POST-ANALYSIS ─ unresolved (§5)
```

## 7. Data and pipeline gaps

Nothing in the pre- or core analysis is blocked. All of these block post-analysis steps only.

| Gap | Where it lands |
|---|---|
| Baseline-share price-shock variant | new radius variables in `snl_mining.aggregation` (`src/data/sources/snl_mining/source.py`) |
| Masked donut aggregation, and the distance-to-nearest-mine column it needs | assembly (`src/data/assemble/sql_engine.py`); `snl_mining` PREPARE |
| Neighbour-mean columns at panel build (now computed in `spec.build_panel`) | derived columns at assembly, if they are to leave the notebooks |
| Treatment rings to 30 km on the 1 km panel (engine exists, not wired) | a ring stage joined in assembly |
| ESA CCI per-class fractions | `average` resampling in `assembly.sources.esacci` |
| Baseline strata (2002 built-up fraction, 2002 PM2.5 tercile) | derived columns at panel build |
| `glass_avhrr` in `assembly.sources`, after month-first compositing | `data.yaml`; `glass/avhrr.py::_calculate_statistics` |
| ADM2 panel for favoritism | `adm2_1km` data source |
| Conley SEs in `duckreg` | analysis layer |
| 11A1 arm joined onto the 21A2 grid | ad-hoc notebook |
| MODIS monthly bands (only for a seasonal coverage breakdown) | `ModisSource._execute_fetch` + a full re-stream |

## 8. Open questions

1. ~~**Can the ring IV be identified?**~~ **Answered** ([`10km-ring-check.md`](10km-ring-check.md)):
   not needed for the main estimate, and the per-ring split is probably not identified.
2. ~~**Does ACAG share the clear-sky problem?**~~ **Mostly answered**
   ([`drift-and-pm25-coverage-checks.md`](drift-and-pm25-coverage-checks.md) §2): the instrument
   does not move clear-sky coverage. In cloudy regions an ACAG aerosol effect may be a lower
   bound in magnitude.
3. ~~**Terra/Aqua orbital drift.**~~ **Measured**: negligible through 2022, material from 2023.
   Only the 2020-vs-2022 choice remains.
4. ~~**Which grid is the main one?**~~ **Answered:** 10 km, decided on the first-stage profile.
5. **Is pollution the only channel that could make the coefficient negative?** The stop decision
   leans on this premise. Aerosol cooling is a daytime mechanism, and land-cover change can lower
   night LST. C4's day/night contrast is the first test available without new data. A direct
   test of the land-cover channel needs ESA CCI fractions (§7).

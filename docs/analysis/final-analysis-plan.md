# Final Paper Analysis Plan

Planning note, first written 2026-09-28, restructured into stages on 2026-09-29 and into
stories on 2026-09-30. It lays out the analysis flow for the final paper. It combines all
sources now in the assembly with the critique from the Scientific Reasoning review of this
project (*"Reviewing a Research Project: Growth, Night Lights, and Local Temperature"*,
2026-09-19). What each source contains, and its caveats, is in
[`../data/README.md`](../data/README.md).

**Why it was restructured.** The first results were inconclusive. The mining-IV second stage is
null. Controlling for PM2.5 did not move the negative TWFE coefficient, and pollution was
believed to be the only channel that could make that coefficient negative. On that basis, the
author and supervisor leaned towards stopping the current path of the project. The 2026-09-29
pre- and core-analysis showed that both results are absences of power, not negative findings
(§1). The analysis is now organised by the three stories the project could still tell.

## How the analysis is organised

| Order | Notebook | Question | Status |
|---|---|---|---|
| — | `src/analysis/spec.py` (§2) | The fixed specification all notebooks import | fixed |
| 1 | `01_pre_analysis.ipynb` (§3) | What do the stories rely on? No outcome-on-treatment regressions. | P1–P5 run 2026-09-29; P6 run 2026-09-30 |
| 2 | `02_baseline.ipynb` (§4, **Story 1**) | Is the association between lights and night LST robust enough to be worth explaining? | run 2026-09-30: **explained away** (B3) |
| 3 | `03_iv.ipynb` (§5, **Story 2**) | Can the mining instrument identify the effect, and what does its null rule out? | I1–I2 run 2026-09-29; I4–I5 run 2026-09-30 (bound confirmed); other extensions later |
| 3b | `03b_iv_robustness.ipynb` (§5) | Is the IV result robust? (R1–R9) | run 2026-09-28, carried over |
| 4 | `04_channels.ipynb` (§6, **Story 3**) | Which mechanisms could produce the association? | M1 run 2026-09-29; the rest later |

**Read them in order.** Story 1 comes first because the other two build on the association it
establishes: Story 2 asks whether it is causal, Story 3 what produces it. If Story 1 finds the
association is an artefact, Stories 2 and 3 have nothing to explain, and what remains is
Story 2's upper bound.

**Labels.** Blocks are prefixed by stage: P (pre-analysis), B (baseline), I (IV), M (mechanisms),
R (IV robustness, carried over). The 2026-09-29 core notebook `02_core_analysis.ipynb` was split
by story: its C1 is now B1, C2 and C3 are I1 and I2, and C4 is M1. The former
`03_post_analysis.ipynb` is now `03b_iv_robustness.ipynb`. The old claims map as follows:
A "the mining IV is null" → Story 2; B "PM2.5 does not move the coefficient" → Story 3;
C "the association is worth explaining" → Story 1.

All notebooks take the specification and helpers from `src/analysis/spec.py`, so they cannot
drift apart on the outcome, fixed effects, clusters, instrument or sample.

This is a plan, not a record of decisions made. Every "◇" decision rule says in advance which
result moves the analysis onto which branch, so a later reader can tell what was decided before
the data was seen. The pre-analysis and I rules were written on 2026-09-29 before those
notebooks ran in their current form. The M1 rules were written after the country × year PM2.5
table in §1 had been seen, so they are not blind to it. The P6 and B2–B5 rules were written on
2026-09-30, after B1 (then C1) had been seen but before any of B2–B5 ran.

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
matters. M1 re-ran the table in the §2 specification, with γ and δ reported explicitly and day
LST added (below).

**Pre-analysis and core results** (2026-09-29, SLURM jobs 24253659 and 24273525; full panel,
§2 specification). The pre-analysis ◇ outcomes:

| | Result | ◇ |
|---|---|---|
| P3 | Night coverage on lights +0.0002 (0.0004) | selection not material |
| P4 | First stage 0.064 (0.008), F = 71; π profile 1.00 / 0.85 / 0.51 | Wald and AR both reported |
| P5 | Within-FE r² of cell PM2.5 with its ring-1 mean 0.97 (lights 0.48) | **M1 speaks to regional aerosol only** |

The core reproduced the "current estimates" above and added:

| | night LST | day LST |
|---|---:|---:|
| B1 TWFE, country × biome × year | −0.0300 (0.0056) | −0.0147 (0.0123) |
| B1 TWFE, country × year | −0.0265 (0.0086) | −0.0151 (0.0150) |
| I1 2SLS; AR 95 % CI | −0.026 (0.150); [−0.32, +0.28] | |
| I2 MDE (80 % power) | 0.42 K per log point = 14× \|TWFE\| | |
| M1 β_short → β_long | −0.0300 → −0.0298 | −0.0147 → −0.0143 |
| M1 γ (PM2.5 on lights) | −0.098 (0.067) | −0.098 (0.067) |
| M1 δ (LST on PM2.5 given lights) | +0.0028 (0.0010) | +0.0035 (0.0024) |
| M1 share of β_short via PM2.5 | 0.9 % | 2.3 % |

Which ◇ rules fired:

- **B1:** the TWFE sign does not depend on the FE set.
- **I2: the AR CI contains the TWFE estimate**, so the IV says nothing about whether the TWFE
  association is causal. It rules out effects beyond about ±0.3 K per log point, i.e. more than
  about 0.4 K at the p90 of 2002–22 lights growth among ever-lit pixels (+1.42 log points).
- **M1: γ not significantly positive** (the control could not move β), **δ ≥ 0** (the
  aerosol-cooling premise fails for this measure) and **night more negative than day**
  (pollution is unlikely the sole channel). With P5, the test can at most exclude a regional,
  ACAG-measured aerosol explanation.

Both results are absences of power, not negative findings. They support "these two tools cannot
answer the question at this resolution", not "the lights–temperature link is not causal" or
"pollution is not the channel". What each story could still deliver, and what threatens it, is
in §4–§6.

**Story 1 and the IV extensions** (2026-09-30, SLURM jobs 24402428 and 24407713; full panel, §2
specification). Night LST, K per log point:

| | Estimate | ◇ |
|---|---|---|
| P6 | Dim-only pixels carry 33.8 % of the within-pixel SS of lights; 63 % of the < 5 °C band is snow-prone | B4c required with B1 |
| B1 | −0.030 (0.006); country × year −0.027 (0.009) | sign does not depend on the FE set |
| B2a bins | DN (0, 7) −0.016 · [7, 20) −0.052 · [20, 30) −0.079 · [30, 50) −0.080 · ≥ 50 −0.051 (n.s.) | no rule fires: lighting up, not brightening |
| B3 | ADM1 × yr −0.020 · ADM2 × yr −0.010 · land cover × yr −0.027 · cell × period −0.047 · **neighbour differences −0.0001 (0.0009)** | **fires: does not survive tighter comparisons** |
| B4 | snow-prone out −0.031 · DMSP −0.055 · VIIRS −0.050 · dim zeroed −0.017 | no rule fires |
| B5a | < 5 °C −0.047 · 5–13 °C −0.062 · 13–22 °C −0.025 · ≥ 22 °C −0.012 (n.s.) | cold band negative: points away from reverse causality |
| B5b | lead +0.010 (n.s.); lagged night LST → lights +0.0002; lagged day LST → lights −0.0012 (p = 0.015) | small feedback exists (day LST) |
| B5c | θ\* = −0.0043 (0.0008) | \|θ\*\| ≤ 0.05: a plausible feedback could produce β |
| I4 | donut F = 11.6, AR [−2.17, +0.19]; footprint F = 69, AR [−0.24, +0.32] | bound confirmed; nothing masked |
| I5 | AR 10 km [−0.12, +0.20] · 20 km [−0.32, +0.28] · 50 km [−0.90, +0.50] | sets overlap; bound of record stays at 20 km |

**Story 1 is explained away as a cell-level association.** Its sign, shape, artefact checks and
climate gradient hold. But the coefficient shrinks with each tighter comparison and is a precise
zero between neighbouring cells. Whatever links lights and night LST is shared across tens of km.
The design identifies from cell-level variation, so it cannot tell regional confounding from a
regional mechanism (aerosols). **Story 2's upper bound is the result of record**; neither the
donut nor another radius changes it. Story 3 has no cell-level association left to decompose.

An older 1 km run (`output/analysis/duckreg/modis_logntlharm_pm25/`: day LST, pixel +
country × year FE, 2000–2020) gives a *positive* TWFE lights coefficient (+0.0049, SE 0.0019)
and PM2.5 +0.014 (0.004). The sign of the TWFE association is therefore not settled across
specifications either (Story 1).

**Inconsistencies:**

| # | Issue | Status |
|---|---|---|
| 1 | The review says the main effects come from night LST; the old headline used day LST. | Resolved: night LST (§2). |
| 2 | The notebook's "extensive/intensive margin" is an NTL threshold split; the review proposes an ESA CCI land-cover split. | B2b now splits the lights margins; the land-cover split stays in Story 3 (needs ESA CCI class fractions). |
| 3 | `NTL_HI` was 7 in the regressions and 20 in the descriptives. | Resolved: 20 everywhere. |
| 4 | The only regional FE was country × year. | Resolved: country × biome × year. |
| 5 | The distance-ring model of [`00`](../design/00-backbone-overview.md) vs own-cell lights. | Resolved: own-cell 2SLS already estimates ≈ β_0 + 0.85·β_1 + 0.51·β_2 in the §2 specification (P4; 0.86 / 0.53 in [`10km-ring-check.md`](10km-ring-check.md)). |
| 6 | Raw VIIRS vs `ntl_harm` ([`04-ingest.md`](../design/04-ingest.md) §1). | Resolved: `ntl_harm` at 10 km, wider than its blur. |
| 7 | The PM2.5-control table uses country × year FE, country clusters and keeps flares; the fixed specification uses country × biome × year FE, ADM1 clusters and drops flares. | Resolved: M1 re-ran it in §2 (`output/tables/core_night_lst_mining_pm25_10km.*`); same reading. |
| 8 | §2 restricts the sample to \|φ\| ≤ 60°; no notebook applies the cut. | Open (§7). B4a drops \|φ\| ≥ 50° as a snow check. |

## 2. Fixed specification

Everything outside this is robustness or exploration. It is coded once, in `src/analysis/spec.py`.

- **Estimand:** the local thermal effect of growing faster than other cells in the same
  country × biome in the same year, measured at the level of a 10 km cell. It includes
  spillovers within the cell and, through the instrument's spatial reach, most spillovers into
  neighbouring cells: own-cell 2SLS estimates Σ_r β_r · π_r/π_0 ≈ β_0 + 0.85·β_1 + 0.51·β_2
  ([`10km-ring-check.md`](10km-ring-check.md)). It is identified as a LATE for mine-proximate
  compliers, and the paper's claims are scoped to that population.
- **Outcome:** `lst_night_mean` (MYD21A2). Day LST is the second outcome wherever day and night
  make different predictions (B1, B2a, B5a, M1). GLASS air temperature and the valid counts are
  diagnostics.
- **Treatment:** `log1p(ntl_harm)` in the own cell; the 3×3 disc mean is a companion.
  `NTL_HI = 20` everywhere. Mask `flare_band > 0`. B2 tests this functional form.
- **Instrument:** `mine_count_20km`.
- **FE:** `pixel_id + GID_0^biome_id^year`.
- **Inference:** clustered by ADM1, with Anderson–Rubin CIs for every IV. Conley SEs are a
  later companion (§7; they are not in `duckreg` yet).
- **Sample:** Aqua years 2002–2022, 10 km base grid, non-null night LST, flares dropped.
  - The MODIS night-LST panel must **never extend past 2022**: Aqua's night overpass drifts
    from 2023 on, and the drift differs by biome
    ([`drift-and-pm25-coverage-checks.md`](drift-and-pm25-coverage-checks.md) §1).
  - Whether to end in 2020 instead is still open (§7).
- **Report with every 2SLS:** the reduced form and the first-stage profile π_r across own
  cell, ring 1 and ring 2. With one instrument, the reduced form is the evidence and the 2SLS
  is its scaling.

## 3. Pre-analysis — `01_pre_analysis.ipynb`

What the stories rely on and can be checked without regressing the outcome on the treatment or
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
    observations. That would hide part of the aerosol cooling M1 looks for.
  - ◇ Selection is **material** if a one-log-point rise in lights moves `cov_night` by more than
    0.01 (1 pp of the pixel's best-year coverage). If it does, B1 also reports the TWFE on
    pixel-years with `cov_night ≥ 0.75`.
  - Aqua drift: done ([`drift-and-pm25-coverage-checks.md`](drift-and-pm25-coverage-checks.md)
    §1), not repeated. It is negligible through 2022.
- **P4. Instrument.**
  - Where its variation lives: the share of pixel-years with a mine within 20 km.
  - Balance in levels: night LST, lights and PM2.5 for pixels that ever have a mine vs pixels
    that never do. This shows what pixel FE must absorb.
  - First stage, F and the π_r profile across own cell, ring 1 and ring 2.
  - ◇ If F < 10, Story 2 reports AR inference only. The AR CI is reported either way.
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
    cell-local variation at 10 km. M1 can then speak only to regional aerosol. A null in M1 is
    not evidence against a local pollution channel.
- **P6. Where the treatment variation lives, and the strata of Story 1** (added 2026-09-30).
  Story 1 splits the sample and changes the functional form. Before any outcome is looked at,
  P6 checks that each split has lights variation to work with, and how much of the treatment's
  variation sits in the dim range that P2 found sensor-sensitive.
  - **P6a.** Pixel-years by brightness bin (DN 0, (0, 7), [7, 20), [20, 30), [30, 50), ≥ 50;
    `ntl_harm` is continuous), by sensor era. The (0, 7) share is 19 % of pixel-years in the DMSP era and 27 % in the VIIRS era (checked
    2026-09-30), so the dim range moves with the sensor.
  - **P6b.** The share of the within-pixel sum of squares of `log1p(ntl_harm)` contributed by
    pixels that never reach DN 7 (dim flicker only), reach DN 7–19, or reach DN 20 or more.
  - **P6c.** Year-to-year transitions between bins (row shares), as a heatmap: how often cells
    flicker between 0 and the dim range.
  - **P6d.** Sizes of the Story 1 strata: baseline temperature bands (B5a), the snow-prone flag
    (B4a), sensor eras (B4b), baseline land-cover classes (B3c), ADM1 and ADM2 units (B3a–b).
    Per stratum: pixels, pixels whose lights change, within-pixel SD of `log1p(ntl_harm)`.
  - ◇ If pixels that never reach DN 7 contribute ≥ 25 % of the within-pixel sum of squares of
    `log1p(ntl_harm)`, the dim range carries a material part of the treatment variation. B4c
    (dim lights set to zero) is then a required companion of B1, reported next to it in the
    Story 1 summary, not a robustness check.
  - ◇ A B2a bin or a B5a band with fewer than 5,000 pixels whose lights change is reported but
    not read.

## 4. Story 1: the baseline effect — `02_baseline.ipynb`

**Question.** When a cell's lights grow faster than those of similar cells in its
country-biome-year, do its nights get cooler, and does that survive the mundane explanations?

**What it can deliver.** A TWFE association that is robust in sign and shape, or the finding that
it is explained away. Even a robust association is not causal on its own: B5 tests the main
alternative, reverse causality, but cannot exclude it. Story 2 is the causal test.

**What threatens it** (the concerns of 2026-09-30, each mapped to a block):

| # | Concern | Intuition | Block |
|---|---|---|---|
| 1 | Reverse causality | Heat slows growth: cells that get hotter grow less and end up with less light. That gives "more light, cooler" with no effect of lights on temperature. | B5 |
| 2 | Confounding | A wet year cools the surface (evaporation) and raises harvests, incomes and lights. Land clearing, irrigation and agricultural expansion can move both too. | B3 |
| 3 | Satellite artefacts | Snow is cold and brightens lights. Clouds and haze select which nights are observed. The 2013 DMSP→VIIRS switch creates jumps. Errors that move with temperature create a correlation. | B4 (coverage: B1) |
| 4 | Functional form | 66 % of pixel-years are dark and 23 % sit at DN (0, 7). `log1p` mixes lighting up with brightening, its units matter, and it may average opposite effects of dim and bright cells. | B2 |
| 5 | Unexpected sign | Cities warm nights (urban heat island). A negative night effect contradicts that prior. | B2 (does the sign differ by brightness?) and B3–B5 |
| 6 | Unstable sign | Pixel FE alone: +0.025. Negative only once cells are compared within country-biome-year. The older 1 km day-LST run: +0.0049. | B1, B3 |
| 7 | Small size | Even at face value, p90 lights growth implies about −0.04 K over two decades. | Story 2 (I3) |

### Blocks

- **B1. The association** (was C1; re-run here). The OLS ladder (pooled → pixel FE → TWFE) for
  night and day LST; TWFE under country × biome × year and country × year FE; TWFE on
  pixel-years with `cov_night ≥ 0.75`.
  - ◇ If the TWFE sign differs between the two FE sets, "the negative TWFE coefficient" is
    reported as specific to the FE set, not as a feature of the data.
  - Graph: the ladder as a coefficient plot.
- **B2. Functional form.**
  - **B2a. Dose-response bins:** indicators for DN (0, 7), [7, 20), [20, 30), [30, 50) and ≥ 50,
    with DN 0 as reference, §2 FE, night and day.
  - **B2b. Margins:** the lit indicator at DN 20 (lighting up) plus `log(DN/20)` above it
    (brightening among lit cells), in one regression, night and day.
  - **B2c. Transforms:** `log(DN + 0.1)`, `log1p`, `log(DN + 10)`, `asinh` and linear DN, night.
  - **Graph (the functional-form figure):** the B2a bin estimates with 95 % CIs at bin
    mean DN, overlaid with each B2c transform's implied curve β_f · (f(DN) − f(0)) over
    DN 0–60. Where the curves and the bins disagree, the transform imposes a shape the data do
    not have. A second panel shows the day bins; a coefficient plot shows the B2b margins.
  - ◇ If the lit bins (DN ≥ 7) include one significantly negative and one significantly
    positive coefficient, or the B2b margins have opposite signs (both p < 0.05), the log
    coefficient averages opposite effects. It is then not reported as "the" effect: the bins are
    the reported shape, and B3–B5 are re-run per margin before they are read.
  - ◇ If the DN (0, 7) bin is the only significant bin, the association lives in the dim range
    that P2 and P6 show is sensor-sensitive. B4c decides whether it survives.
  - ◇ If `log(DN + 0.1)` and `log(DN + 10)` give opposite signs, `log1p` is driven by the zero
    margin (Chen and Roth 2024): report the bins, not a log slope.
- **B3. Tighter comparisons** (against confounding). Compare cells that are more alike, so
  fewer outside factors differ. Night LST, `log1p`, ADM1 clusters throughout.
  - **B3a.** Pixel + ADM1 × year FE.
  - **B3b.** Pixel + ADM2 × year FE.
  - **B3c.** §2 FE + baseline land cover × year FE (ESA CCI dominant class of the 10 km cell in
    2002), so forest and cropland cells can follow different temperature paths.
  - **B3d.** Cell × period FE (2002–07, 2008–12, 2013–17, 2018–22) + country × biome × year.
    Each cell gets its own level in each five-year period. This is a step-wise stand-in for
    cell-specific trends (which `duckreg` does not fit) and also absorbs each cell's shift at
    the 2013 sensor splice.
  - **B3e.** Neighbour differencing: (Y − Y_ring1) on (X − X_ring1) with the §2 FE. Adjacent
    cells share nearly all weather and local conditions.
  - ◇ If β changes sign, or |t| < 2, under B3b or B3e, the association does not survive tighter
    comparisons. If every B3 estimate is negative with |t| ≥ 2 and within ±50 % of B1
    ([−0.045, −0.015]), it is stable. Anything in between is reported as partly sensitive, per
    variant.
  - Graph: a forest plot of B1 and B3a–e.
- **B4. Satellite artefacts.** Random noise only attenuates β; what matters are errors that move
  with temperature.
  - **B4a. Snow-prone cells dropped:** \|φ\| ≥ 50°, or the boreal forest/taiga or tundra biome.
  - **B4b. Sensor eras separately:** DMSP 2002–12 and VIIRS 2014–22 (the 2013 overlap dropped).
    The former R2 found the era-interacted TWFE slope negative in both (DMSP −0.044, VIIRS
    about −0.028); B4b re-checks it under §2 with separate FE per era.
  - **B4c. Dim lights set to zero:** treatment `log1p(DN · 1[DN ≥ 7])`.
  - Observation selection is covered by the coverage-restricted row of B1.
  - ◇ B4a: if β ≥ 0 or |t| < 2 once snow-prone cells are dropped, snow is a candidate
    explanation. B4b: if the eras differ in sign (both p < 0.05), or one era's estimate is
    positive, the association is sensor-specific. B4c: if β ≥ 0 or |t| < 2, the association is
    carried by dim-range lights.
  - Graph: B4 joins the forest plot.
- **B5. Reverse causality: heat slows growth.**
  - **B5a. Sign by baseline climate.** Growth responds to temperature in a hump shape, with an
    optimum near 13 °C annual mean (Burke, Hsiang and Miguel 2015). Reverse causality predicts
    a negative coefficient in hot cells and a zero or positive one in cold cells. Snow (B4a)
    predicts the opposite in cold cells: negative. One regression per outcome of LST on lights
    interacted with baseline temperature bands (GLASS air temperature, 2002–04 mean: < 5,
    5–13, 13–22, ≥ 22 °C), §2 FE, night and day.
  - **B5b. Timing.** Heat that slows growth lowers *later* lights, and a common driver shows up
    ahead of time. Three §2-FE regressions: (i) the lead placebo LST_t on X_t and X_{t+2};
    (ii) X_t on LST_{t−1} and X_{t−1}; (iii) LST_t on X_{t−1} and LST_{t−1}.
  - **B5c. Breakdown feedback θ\*.** If lights responded to local temperature with θ (log points
    per K) and lights had no effect (β = 0), the TWFE of LST on lights would be
    θ · Var(Ỹ)/Var(X̃), and the reverse regression of lights on LST would recover θ exactly.
    So θ\* is the within-FE coefficient of `log1p(ntl_harm)` on LST, night and day. LST
    measurement noise cannot feed back into growth, so the feedback actually needed is at least
    θ\*: θ\* is the smallest feedback that could produce β on its own.
  - ◇ B5a: if the hot band (≥ 22 °C) is significantly negative and the cold band (< 5 °C) is
    zero or positive, the pattern matches reverse causality. If the cold band is significantly
    negative, it points away from reverse causality, towards snow or another cold-climate
    mechanism.
  - ◇ B5b: a lead coefficient with p < 0.05 and the sign of B1 means a common driver or
    reverse causality is present. Lagged LST lowering lights (p < 0.05) means a heat → lights
    feedback exists.
  - ◇ B5c: the benchmark is fixed before the run at |θ| ≈ 0.05 log points of lights per K.
    At 25 °C the Burke et al. (2015) curve implies about −1.2 pp of GDP growth per +1 °C, and
    lights move about 3× GDP (Henderson, Storeygard and Weil 2012: GDP-on-lights elasticity
    ≈ 0.3), about −0.04, rounded up. If |θ\*| ≤ 0.05, a feedback of plausible size could
    produce β alone, and the association cannot be read as an effect of lights.
  - Graph: B5a coefficients by band, night and day.
- **Summary.** A forest plot of every night-LST β from B1–B4 against the B1 estimate, and a
  table of the rules that fired.

**Reading for the decision.**
- **Explained away:** any of B2–B5 fires against the association (it averages opposite effects,
  lives only in the dim range, fails tighter comparisons, is snow- or sensor-specific, or shows
  a reverse-causality pattern). It is not worth explaining; what remains is Story 2's bound.
- **Robust:** it survives B3–B4 with a stable sign, and B5 shows no reverse-causality pattern.
  The new-data checks below and Story 3 become worth their cost.
- **Robust but not interpretable:** it survives B3–B4, but θ\* is small or the timing tests
  fire. It is then a robust association that needs Story 2, or the ERA5-Land route below, before
  it can be read as an effect of lights.

**Outcome (run 2026-09-30): explained away.** B3 fires. Under neighbour differences the
coefficient is −0.0001 (0.0009), a precise zero; the spillover note below expected a smaller
estimate, not zero with an SE a sixth of B1's. ADM2 × year keeps the sign (−0.010, t = −4.0) at a
third of B1. B1, B2 and B4 fire nothing. B5 is mixed: the climate gradient runs against reverse
causality, but θ\* is small. The numbers are in §1 and in the Result cells of
`02_baseline.ipynb`.

### Known problems, written before the run

- **Multiple testing and forking paths.** Story 1 reports about 30 estimates, so one "significant"
  deviation is expected by chance. The ◇ rules name specific patterns in advance, and the
  summary shows every estimate, not a selection.
- **B2.**
  - Bins are identified from cells that switch bins within the §2 FE. The top bins have few
    switchers, so their CIs will be wide (P6d counts them).
  - The dim bin (DN (0, 7)) and the bin boundary at DN 7 are exposed to the 2013 splice (P2).
  - `log(DN + 0.1)` gives dark cells a value of −2.3, so the 0 ↔ dim margin dominates it by
    construction. That is the point of the comparison, but it makes the estimate noisy.
  - The harmonised VIIRS-era series is converted to a DMSP-like scale by a fitted function, so
    part of any curvature may come from the harmonisation.
- **B3.**
  - ADM2 × year FE have about 0.85 M levels. They absorb much of the lights variation, so SEs
    will grow; the rule reads sign and |t|, not size.
  - Neighbour differencing also removes spillovers. With a first-stage profile of 0.85 in
    ring 1, a causal effect that spreads into neighbours shrinks under B3e. A smaller B3e
    estimate is therefore expected even if the association is real; only the sign and |t| are
    read.
  - The ESA CCI dominant class at 10 km is coarse, and ESA CCI updates classes conservatively.
    B3c absorbs baseline differences, not measured conversion.
  - The B3d periods have five or four years each, so cell × period FE leave little within-group
    variation. The 2013 splice falls on a period boundary on purpose.
- **B4.**
  - The snow-prone flag is crude. It keeps high-altitude snow (Tibet, Andes) and drops the whole
    boreal zone, so a change in β can come from sample composition as much as from snow.
    Snow cover itself needs a new source (below).
  - The eras differ in length, noise and sensor, so a smaller VIIRS estimate is not by itself a
    contradiction; only a sign difference is.
- **B5.**
  - Baseline temperature bands overlap with latitude, snow and income. The cold band separates
    reverse causality from snow only if it has enough variation outside snow-prone cells.
    P6d reports the overlap.
  - GLASS air temperature is 10.7 % missing and partly derived from MODIS LST; as a baseline
    stratifier this is harmless, but pixels without a 2002–04 value drop out of B5a.
  - Burke et al.'s optimum and damage curve are for country-level annual mean air temperature,
    not cell-level LST. The band edges and the θ benchmark are order-of-magnitude guides.
  - Lights are very persistent, so the lead in B5b(i) is correlated with current lights;
    conditioning on X_t is essential. Regressions with a lagged dependent variable and pixel
    FE carry Nickell bias of order 1/T (T ≈ 20).
  - θ\* equals θ only if the sole link is the feedback. With a real β, confounding and LST
    noise, it is a threshold, not an estimate.

### Later, with new data (§9)

- **Weather controls:** precipitation and soil moisture (CHIRPS or ERA5-Land). Weather is not
  caused by lights, so it is a safe control against the wet-year story.
- **Snow cover:** MODIS MOD10/MYD10 instead of the B4a latitude-and-biome proxy.
- **NASA Black Marble (VNP46A4):** a lights product corrected for snow, moonlight and atmosphere,
  VIIRS era only.
- **Independent measurements:** the strongest test against artefacts, because they are
  sensor-specific. The association should hold with a different treatment (GHSL built-up area,
  WorldPop population) and a different outcome (Landsat LST on a subsample; weather stations
  where available).
- **θ from exogenous weather (ERA5-Land 2 m temperature):** ERA5-Land's land surface is static,
  so it does not respond to local growth. Regressing lights on ERA5-Land temperature anomalies
  within the §2 FE (controlling for precipitation) estimates θ, and β then follows from the
  observed second moments. This identifies β without a lights instrument, under one strong,
  explicit assumption: simultaneity is the only bias. Worth doing only if B5 points to reverse
  causality.
- **Coefficient-stability bounds** (Oster 2019) across B3: need R², which `duckreg` does not
  report yet.

Land-cover *change* must not be a control anywhere in Story 1: growth partly causes it. Only
baseline land cover (B3c) is safe.

## 5. Story 2: the IV — `03_iv.ipynb`, `03b_iv_robustness.ipynb`

**Question.** Can the mining instrument identify the effect of lights on night LST, and what
does its null rule out?

**What it can deliver.** Not a test of the TWFE-sized effect (I2), but an **upper bound**: the
AR CI rules out effects beyond about ±0.3 K per log point, about ±0.4 K at p90 lights growth
over 2002–22. "Local growth does not heat or cool its surroundings by much at 10 km" is
defensible now. It is scoped to mine-proximate compliers.

- **I1. The IV** (was C2). Reduced form, 2SLS and the 3×3 disc 2SLS. The **Anderson–Rubin CI**
  comes exactly from three reduced forms: of Y, of Y − X and of Y + X on Z
  (`spec.anderson_rubin_ci`).
- **I2. What the null rules out** (was C3).
  - The minimum detectable effect at 80 % power (2.8 × SE).
  - The gap between the 2SLS and TWFE estimates, relative to the 2SLS SE.
  - The economic scale of lights changes: the distribution of within-pixel Δ`log1p(ntl_harm)`
    from 2002 to 2022, and the temperature change each estimate implies at its p50, p90 and p99.
  - ◇ If the AR CI contains the TWFE estimate, the IV says nothing about whether the TWFE
    association is causal. "The IV is insignificant" then counts against effects outside the
    AR CI only. It does not count against an effect of the TWFE size.
- **I3 (later). The bound against observed warming.** The observed 2002–22 LST change in the
  same cells, next to the bound. Not yet computed.

**Why a new instrument will not fix it.** With one instrument, the 2SLS test is the reduced-form
test, and RF = π·β. At β = −0.030 and π = 0.064 the RF is about 0.002 K, against an RF SE of
0.0097. 80 % power needs a first stage whose t-statistic is about 14× ours, i.e. F ≈ 14,000
instead of 71 at similar noise. No candidate instrument (R6, R9, the extensions below) comes
close on its own.

**What can still raise power.** The reduced-form test has t ≈ π·β / SE_RF, about 0.2 at the TWFE
size against about 2.8 needed. Three things move it, and only the first is plain power:
1. **A larger first stage π** (e.g. the 10 km radius, F ≈ 139): the same β becomes easier to
   see. Roughly a 2× gain.
2. **A larger β in a subsample** (effect heterogeneity): a stronger test of *that sample's*
   effect, not of the average. The full-sample bound already caps the average at about
   0.3 per log point, so a strong local effect can only exist in a small part of the sample.
3. **Unmasking.** If mines also heat or cool cells directly (bare ground, clearing, dust), that
   direct effect can cancel the lights effect in the reduced form. Removing the footprint (I4)
   can then reveal an effect. That is more than power: it would be a finding.

A stronger reaction *inside* the mine footprint is a warning (exclusion fails there), not power.
The radius and donut are therefore fixed below before any of them is estimated; choosing the
specification with the strongest reduced form after the fact would be specification search.

- **I4. Donut (specified 2026-09-30, runs on the current panel).** At 10 km, `mine_count_10km`
  is the average over the cell's 1 km pixel centres of the active mines within 10 km. A cell
  where it is zero in every year 2002–22 has no active mine within 10 km of any part of it.
  - **Donut sample D:** cells with `max(mine_count_10km) = 0` over 2002–22. The instrument stays
    `mine_count_20km`, so only mines 10–20 km away enter. The sample is defined on all years,
    not year by year, so it does not select on the timing of openings.
  - **Footprint sample F:** the complement (a mine within 10 km in at least one year).
  - In both: first stage, F, reduced form, 2SLS and AR CI, night and day LST, §2 FE and clusters.
  - ◇ If the first-stage F in D is below 10, D reports AR inference only (as P4).
  - ◇ **Unmasked:** if D's AR CI excludes zero, an effect appears once the footprint is removed.
    The full-sample null was masked; D becomes Story 2's estimate, compared in sign and size with
    the TWFE.
  - ◇ **Direct mine effect:** if the reduced forms in D and F have opposite signs (both
    p < 0.10), mines affect temperature directly near the footprint. The full-sample 2SLS then
    mixes channels, and the bound is taken from D.
  - ◇ **Bound confirmed:** if D's AR CI contains zero and is bounded, the upper bound holds
    without the footprint cells. It is reported next to the full-sample bound as the more
    credible one, even if wider.
  - Known problems: D drops the most exposed cells, so π and power will probably fall; the
    donut is at 10 km cell resolution with a 10 km buffer, not the masked 1 km donut (later,
    §9); aerosols and dust can travel beyond 10 km, so the donut removes near-field physics
    only; D differs in composition (farther from mines).
- **I5. Radius comparison (pre-specified, not a search).** Mines within 10, 20 and 50 km, and the
  over-identified count + price shock, each with first stage, F, reduced form, 2SLS and AR CI, in
  one table. R6 has the 2SLS; I5 adds the reduced forms and AR CIs in the §2 specification.
  - ◇ The bound of record stays at 20 km, chosen on the P4 π profile before these results. The
    other radii are reported, not selected. The 10 km radius gives the tighter bound but is the
    most exposed to footprint physics; it is read together with I4.
  - ◇ If the AR CIs of different radii do not overlap, the instrument's estimand depends on
    distance (spillovers or exclusion failure), and the bound is reported per radius.
- **I4–I5 outcome (run 2026-09-30).**
  - **I4.** The donut's first stage is weak (F = 11.6), with RF −0.023 (p = 0.10) and AR
    [−2.17, +0.19]. The footprint cells carry the identification (F = 69, AR [−0.24, +0.32]).
    ◇ Bound confirmed; nothing is masked, and no direct mine effect by the rule.
  - **I5.** First-stage F is 139 / 71 / 22 at 10 / 20 / 50 km. The AR sets are [−0.12, +0.20],
    [−0.32, +0.28] and [−0.90, +0.50]; the over-identified Wald CI is [−0.31, +0.28]. ◇ The sets
    overlap, and the bound of record stays at 20 km. The 10 km MDE is 0.22 K per log point, still
    7× the TWFE.

**`03b_iv_robustness.ipynb`** (the former `03_post_analysis.ipynb`; blocks carried over from
the former `regression.ipynb`, run 2026-09-28, not re-run):

| Block | Finding |
|---|---|
| R1 NTL functional form | asinh, lit indicators at 20 and 30: all rescale the same null reduced form |
| R2 DMSP→VIIRS break | 2SLS unchanged; the OLS slope is ~40 % weaker in the VIIRS era |
| R3 gas flares | no change |
| R4 coverage-restricted 2SLS | unchanged at `cov_night ≥ 0.75` and `≥ 0.9` (the selection test itself moved to P3) |
| R5 GLASS, VIIRS | GLASS 2SLS null; VIIRS first stage weak (F ≈ 5) (day LST moved to B1) |
| R6 instrument choice | 10/50 km, price shock, over-identified: all F ≥ 22, all 2SLS null |
| R7 inference | ADM1 the most conservative cluster; HC1 about 6× too small |
| R8 FE (2SLS) | country × year: −0.086 (0.174) |
| R9 political favoritism | first stage dead (F ≈ 1.4) |

**Later steps (unresolved):**
- a baseline-share price-shock (exposure frozen at 2002, so only world prices vary). Its value is
  credibility (world prices are plausibly unrelated to local temperature), not power: the current
  price-shock version already has a larger SE (0.28), and freezing exposure removes the variation
  from openings;
- a masked 1 km donut, refining I4 (needs the distance-to-nearest-mine column, §9);
- an event study around mine openings (footprint vs ring cells);
- shift-share diagnostics (Rotemberg weights, share balance, shock-level SEs);
- a GLASS-AVHRR pre-period placebo for 1992–2001;
- favoritism at ADM2;
- a complier profile;
- TOST equivalence bounds.

**Design note.** `mine_priceshock_*` is not yet a clean shift-share. Its exposure moves with
openings and closures. Freeze exposure at baseline before applying Borusyak–Hull–Jaravel or
Goldsmith-Pinkham diagnostics.

## 6. Story 3: channels — `04_channels.ipynb`

**Question.** Which mechanisms could produce the association of Story 1?

**What it can deliver.** A causal decomposition needs lights → mediator identified (the same
problem as Story 2) *and* mediator → temperature, so it is strictly harder. What can be
convincing is an **accounting**: take mediator → LST from established physics (e.g. the global
day/night LST effects of vegetation transitions in Duveiller, Hooker and Cescatti 2018),
estimate lights → mediator descriptively, and ask whether the product reproduces the TWFE. It
is worth doing only if Story 1 finds a robust association.

- **M1. PM2.5 channel accounting** (was C4), on the pixel-years with PM2.5, separately for
  night and day LST:
  - short: `Y ~ X | FE`;
  - long: `Y ~ X + pm25 | FE`, with δ = the PM2.5 coefficient;
  - auxiliary: `pm25 ~ X | FE`, with γ = the lights coefficient.

  The omitted-variable identity β_short − β_long = γ·δ holds exactly in-sample; the notebook
  checks it. It also reports the reduced form of PM2.5 on the instrument and the 2SLS with and
  without the PM2.5 control on the same sample, and exports the PM2.5 table
  (`output/tables/core_night_lst_mining_pm25_10km.*`). This is a descriptive decomposition of
  the OLS association, not a causal direct effect.
- **M1 reading rules**, in order:
  - ◇ **γ not significantly positive:** within a cell, lights do not raise ACAG PM2.5. The
    control *cannot* move β, whatever the true pollution channel. Read this together with P5.
  - ◇ **δ ≥ 0:** within a cell, more PM2.5 goes with warmer (or unchanged) LST, so the
    aerosol-cooling premise fails for this measure and outcome.
  - ◇ **γ > 0 and δ < 0, but |γ·δ| < ¼ |β_short|:** PM2.5, as ACAG measures it, accounts for
    little of the association.
  - ◇ **Day vs night.** Aerosol dimming cools the surface by day; at night aerosols tend to warm
    it. If the TWFE is more negative at night than by day, pollution is an unlikely sole
    explanation of the night coefficient.

**Which channels can fit.** The sign pattern (night −0.030, day −0.015 n.s.) constrains them:

| Channel | Day | Night | Explains night −? | Data |
|---|---|---|---|---|
| Built-up / impervious expansion | + | ++ | No: wrong sign | ESA CCI fractions; GAIA or GHSL better |
| Anthropogenic heat | + | + | No | lights proxy it (circular) |
| Aerosols | − | + | No (M1) | ACAG too coarse (P5); MAIAC AOD 1 km would fix it |
| Vegetation clearing / agricultural expansion | + (tropics) | − (temperate, boreal) | Possibly | Hansen annual forest loss; ESA CCI fractions |
| Irrigation / greening | − | small | Weakly | MODIS NDVI / ET |
| Snow-brightened lights (artefact) | − | − | Yes | B4a |
| Reverse causality (heat slows growth) | − | − | Yes | B5 |

The explanations that fit a negative night coefficient most easily are an artefact and reverse
causality, which Story 1 tests. Vegetation clearing is the one physical candidate, and only in
some biomes. B2's dose-response shape also bears on this: a negative dim range and a positive
bright range would point to rural clearing and urban heat islands at once.

**Later steps (unresolved):**
- land-cover fractions and PM2.5 as outcomes of the instrument;
- the accounting with Hansen forest loss, ESA CCI fractions, MAIAC AOD, MODIS NDVI/ET and
  Duveiller et al. (2018) sensitivities;
- baseline land-cover strata;
- the entanglement share of ΔNTL in cells with built-up change;
- GLASS vs LST divergence;
- heterogeneity by biome, aridity, baseline PM2.5 and income;
- the outcome in neighbouring cells (SUTVA).

**Design notes** (from the review):
1. **Land-cover strata must be baseline strata.** Growth partly causes conversion, so splitting
   by "converted vs never converted" selects on an outcome of the treatment. Define strata by
   2002 built-up fraction and dominant class; report land-cover change separately, as an
   outcome.
2. **PM2.5 must not be a control in a causal claim.** It is a mediator. M1 uses it as a
   descriptive decomposition only.
3. **ESA CCI is aggregated by `mode` at coarse grids**, which hides built-up fractions. Add
   per-class fraction variables (`average` of class indicators) first.

## 7. Cross-cutting checks, unassigned

- **Measurement:** 11A1 vs 21A2 night LST by land-cover class (emissivity bias; 5 tiles ×
  3 years); a coverage-balanced outcome with Lee-type bounds; panel end 2020 vs 2022; the
  \|φ\| ≤ 60° cut.
- **Inference and specification:** Conley SEs; a specification curve over FE, functional form,
  sensor era, grid (5/10/25 km, grid-shake), lights aggregation and 1 km rings.

| Review lens | Threat | Check | Where |
|---|---|---|---|
| Measurement | NTL ≈ the land-cover mechanism | entanglement share; baseline strata; land cover as an outcome | Story 3 (later) |
| Measurement | Emissivity–land-cover bias in the outcome | 11A1 vs 21A2 | §7 |
| Measurement | Compositing drops hazy observations | coverage on lights, Z, PM2.5 | **P3**; bounds §7 |
| Measurement | Day LST reacts to albedo | day vs night | **B1, B2a, B5a, M1** |
| Measurement | Lights artefacts (snow, splice, dim flicker) | snow-prone, eras, dim-zeroed | **P6, B4** |
| Design | Weak or non-local instrument | F, π profile | **P4** |
| Design | On-site extraction physics breaks exclusion | donut; event study | Story 2 (later) |
| Design | Mine locations and openings are not random | baseline shift-share; placebo; share balance | Story 2 (later) |
| Causal | Reverse causality (heat slows growth) | climate bands, timing, θ\* | **B5** |
| Causal | Confounding | tighter comparisons | **B3** |
| Causal | A null may hide offsetting channels | AR CI and MDE; PM2.5 accounting | **I1–I2, M1**; mechanism outcomes later |
| Causal | Exchangeability and SUTVA | π profile; neighbouring-cell outcome | P4; Story 3 (later) |

## 8. Ordering

```
[§2] FIXED SPEC ─ 10 km · night LST · log1p(ntl_harm), NTL_HI=20, flares dropped · Z = mine_count_20km
     │            FE pixel + ctry×biome×year · ADM1 clusters · 2002–2022        (src/analysis/spec.py)
     ▼
[01] PRE-ANALYSIS ─ no outcome-on-treatment regressions ─────────────────────────────────────
     P1 panel · P2 DMSP→VIIRS, flares
     P3 coverage ~ {lights, Z, PM2.5}        ◇ material selection? ──► B1 adds a coverage-restricted TWFE
     P4 Z variation, balance, first stage, π  ◇ F < 10? ──► AR only
     P5 PM2.5 local signal (r² with ring 1)   ◇ r² ≥ 0.9? ──► M1 speaks to regional aerosol only
     P6 lights variation by bin, flicker, strata  ◇ dim range ≥ 25 % of variation? ──► B4c reported with B1
     ▼
[02] STORY 1 · BASELINE EFFECT ───────────────────────────────────────────────────────────────
     B1 OLS ladder, night + day, two FE sets     ◇ sign depends on FE? ──► say so
     B2 functional form: bins · margins · transforms   ◇ opposite signs / dim only / unit-driven?
     B3 tighter comparisons: ADM1/ADM2 × yr, land cover × yr, cell × period, neighbour diff
                                                  ◇ sign flips or |t| < 2 under ADM2 or neighbour diff?
     B4 artefacts: snow-prone out · DMSP vs VIIRS · dim lights zeroed   ◇ sign lost?
     B5 reverse causality: climate bands · timing · θ*   ◇ hot −, cold ≥ 0? leads? |θ*| ≤ 0.05?
     ▼   explained away ──► the paper is Story 2's bound      ◄ taken 2026-09-30 (B3 fired)
[03] STORY 2 · IV ────────────────────────────────────────────────────────────────────────────
     I1 RF · 2SLS · disc · AR CI
     I2 MDE · IV−TWFE gap · scale of ΔNTL        ◇ AR CI ∋ TWFE? ──► the IV cannot speak to it
     I4 donut (no mine within 10 km) vs footprint   ◇ AR CI excludes 0? ──► unmasked · opposite RF signs? ──► bound from D
     I5 radii 10/20/50 km, over-ID (pre-specified)   ◇ bound of record stays 20 km
     I3 (later) bound vs observed warming
     03b R1–R9 IV robustness (carried over)
     ▼
[04] STORY 3 · CHANNELS ──────────────────────────────────────────────────────────────────────
     M1 β_short − β_long = γ·δ, night + day; IV with/without PM2.5
          ◇ γ ≤ 0 → no power · δ ≥ 0 → premise fails · |γδ| < ¼|β| → PM2.5 explains little
          ◇ night more negative than day → pollution unlikely the sole channel
     later: land-cover accounting, only if Story 1 finds a robust association
```

## 9. Data and pipeline gaps

Nothing in the pre-analysis, B1–B5, I1–I2, I4–I5 or M1 is blocked. All of these block later steps only.

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
| Conley SEs, R² for coefficient-stability bounds in `duckreg` | analysis layer |
| 11A1 arm joined onto the 21A2 grid | ad-hoc notebook |
| MODIS monthly bands (only for a seasonal coverage breakdown) | `ModisSource._execute_fetch` + a full re-stream |
| Weather: precipitation and soil moisture (CHIRPS or ERA5-Land), ERA5-Land 2 m temperature | new FETCH/PREPARE source |
| Snow cover (MODIS MOD10/MYD10) | new source, or a band of the existing MODIS stream |
| NASA Black Marble VNP46A4, VIIRS era | new source |
| Independent treatments: GHSL built-up, WorldPop population | new sources |
| Landsat LST on a subsample | ad-hoc extraction |
| Channel accounting: Hansen annual forest loss, MAIAC AOD 1 km, MODIS NDVI/ET, Duveiller et al. (2018) transition sensitivities | new sources |

The Story 1 derived columns (baseline temperature bands, baseline land cover, latitude, ADM2,
bins, leads and lags, ring-1 LST) are built in `spec.build_panel`.

## 10. Open questions

1. ~~**Can the ring IV be identified?**~~ **Answered** ([`10km-ring-check.md`](10km-ring-check.md)):
   not needed for the main estimate, and the per-ring split is probably not identified.
2. ~~**Does ACAG share the clear-sky problem?**~~ **Mostly answered**
   ([`drift-and-pm25-coverage-checks.md`](drift-and-pm25-coverage-checks.md) §2): the instrument
   does not move clear-sky coverage. In cloudy regions an ACAG aerosol effect may be a lower
   bound in magnitude.
3. ~~**Terra/Aqua orbital drift.**~~ **Measured**: negligible through 2022, material from 2023.
   Only the 2020-vs-2022 choice remains.
4. ~~**Which grid is the main one?**~~ **Answered:** 10 km, decided on the first-stage profile.
5. **Is pollution the only channel that could make the coefficient negative?** **Partly answered
   by M1:** the night coefficient is stronger than the day one, so pollution is unlikely to be
   the sole channel. Which channels could fit is laid out in §6.
6. ~~**Is the TWFE association worth explaining at all?**~~ **Answered (Story 1, 2026-09-30):**
   not at the cell level. It survives functional form, artefacts and the climate-gradient test,
   but vanishes between neighbouring cells (B3).
7. ~~**How large would reverse causality need to be?**~~ **Answered (B5c):** small. θ\* = −0.0043
   log points per K, 12× below the benchmark, so a plausible feedback could produce β. The
   climate gradient (B5a) points the other way.

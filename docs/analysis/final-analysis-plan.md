# Final Paper Analysis Plan

Planning note, 2026-09-28. It lays out the analysis flow for the final paper. It combines all
sources now in the assembly with the critique from the Scientific Reasoning review of this project
(*"Reviewing a Research Project: Growth, Night Lights, and Local Temperature"*, 2026-09-19). No
new estimation was run to write it. The current results cited below come from
`output/notebooks/regression.ipynb` and `output/notebooks/descriptive_statistics.ipynb`.
What each source contains, and its caveats, is in [`../data/README.md`](../data/README.md).

This is a plan, not a record of decisions made. Every "◇" decision rule below says in advance
which result moves the analysis onto which branch. That way a later reader can tell what was
decided before the data was seen and what was decided after.

## 1. Where the project stands

**Current headline** (`regression.ipynb`, 10 km grid, ~25 M pixel-years):

```
lst_day_mean ~ 1 | pixel_id + GID_0^year | (log1p(ntl_harm) ~ mine_count_20km)
```

The 2SLS estimate is −0.18 K per log point (SE 0.27, country-clustered), with first-stage F ≈ 30.
Every instrument variant gives a negative sign (10/20/50 km, price-shock, over-identified), and
none is significant. The magnitude grows with the radius: −0.08, −0.18, −0.47. Regional
favoritism (`reg_fav`) has a dead first stage under pixel FE (F ≈ 0.5) but a reduced form of
+0.09 K (p ≈ 0.04). The notebook summary also finds that TWFE OLS on GLASS air temperature is
−0.020 K (p ≈ 0.02).

**Inconsistencies to fix before anything counts as final:**

| # | Issue | Where |
|---|---|---|
| 1 | The review says the main effects come from **night** LST; the notebook's headline uses **day** LST. | `regression.ipynb` `Y` |
| 2 | The notebook's "extensive/intensive margin" is an **NTL threshold** split. The review proposes an **ESA CCI land-cover** split. These are different objects. | `regression.ipynb` R1 |
| 3 | The NTL threshold is `NTL_HI = 7` in the regression notebook and `20` in the descriptives, which argue for 20 because of the DMSP→VIIRS artefact at 7. | both notebooks |
| 4 | The only regional FE is country × year. The review specifies country **or ecoregion** × year. | `FE` |
| 5 | The estimating equation in [`00-backbone-overview.md`](../design/00-backbone-overview.md) is the **distance-ring** model, whose quantity of interest is Σ_r β_r. The regressions use own-cell lights only. [`00`](../design/00-backbone-overview.md) argues this recovers a small fraction of the true effect when spillovers exist. | `regression.ipynb` vs `00` |
| 6 | [`04-ingest.md`](../design/04-ingest.md) §1 makes raw VIIRS (`eog_viirs`) the lights input for the 1 km ring model, and keeps `ntl_harm` for ≥ 5 km work: the harmonization blur (effective resolution ~4–7 km) is wider than the local channel. The regressions use `ntl_harm`. | `regression.ipynb` vs `04` |

**Assembled inputs not yet used in any regression** (`data.yaml` → `assembly.sources`):

- `acag` PM2.5 (1998–2023): the only direct measure of the aerosol channel. The review does not
  use it.
- `snl_mining`:
  - `mine_polygon_count` (real mine footprints), which enables a donut. Only about 57 % of
    active mine-years have a footprint polygon, so it needs a distance-based complement;
  - opening and closing years, which enable an event study;
  - per-mine commodity shares, which enable a proper shift-share instrument.
- `modis`: `valid_period_count_night_annual` and `valid_month_count_night_annual`, the per-pixel
  coverage of each annual composite. Monthly values are **not** persisted: compositing uses them
  only as an internal weighting step.
- `modis_robustness_11a1`: 5 tiles × 3 years (2004/2014/2023) of daily-product night LST.
- `ecoregions` (`realm_id`, `biome_id`, `eco_id`) and `country_classifications` (HDI, income
  group).
- The neighbourhood engine and ring means ([`03-neighbourhood-engine.md`](../design/03-neighbourhood-engine.md),
  `src/data/assemble/ring_means.py`), and grid-shake (`src/data/assemble/grid_shake.py`).
- `glass_avhrr` (1992–2020): covers 1992–2001, before Aqua, which allows pre-period placebos.
  It is **not** in `assembly.sources` yet.

## 2. Critique → check → data

| Review lens | Threat | Check | Data | Phase |
|---|---|---|---|---|
| Measurement | Treatment (NTL) ≈ the land-cover mechanism | Entanglement share; effect in strata that cannot convert; land cover as an outcome | `esacci` (fractions), `ntl_harm` | 1c, 6a, 6b |
| Measurement | Emissivity–land-cover bias in the outcome | 11A1 vs 21A2 difference by land-cover class | `modis`, `modis_robustness_11a1`, `esacci` | 1a |
| Measurement | The 8-day composite drops hazy observations and smooths out aerosol cooling | Valid counts on NTL, PM2.5 and Z; seasonal breakdown; balanced composite | `modis` valid period/month counts, `acag` | 1b |
| Measurement | Day LST reacts mechanically to albedo change | Day vs night contrast | `modis` day/night | 6c |
| Measurement | GLASS is partly built from land cover and remote sensing | Only divergence from LST is informative | `glass_ta_modis` | 6c |
| Design | On-site extraction physics breaks exclusion | Donut; footprint vs ring event study | `snl_mining` polygons, ring means | 3a, 3c |
| Design | Mine locations are not random (settlement history, pre-trends) | Share balance; pre-period placebo; cell trends | `snl_mining` shares, `glass_avhrr`, `commodity_prices` | 3c, 5 |
| Design | Mine openings are endogenous | Fix shares at baseline so only world prices vary | `snl_mining`, `commodity_prices` | 3a |
| Design | Favoritism flows through land cover; dead first stage | ADM2 re-specification; land cover as an outcome | `plad`, `gadm` | 3d, 6a |
| Causal | One coefficient cannot separate channels of opposite sign | Mechanism first stages; heterogeneity; equivalence bounds | `acag`, `esacci`, `ecoregions`, `country_classifications` | 6, 7 |
| Causal | Exchangeability and SUTVA | Ring model; spillover rings of the outcome | ring means | 4, 6e |

## 3. Four places where the review's own proposals need tightening

1. **The land-cover split conditions on a post-treatment variable.** Growth partly causes
   conversion to built-up. Splitting the sample by "converted vs never converted" therefore
   selects on an outcome of the treatment and biases both sub-sample coefficients. Instead:
   - Define strata from **baseline** characteristics (2002 built-up fraction and dominant class).
     A cell already fully built at baseline can only grow on the intensive margin; a cell with
     no convertible land cannot convert.
   - Report land-cover change separately as an **outcome** of the instrument (Phase 6a).
2. **PM2.5 must not be used as a control.** It is a mediator, so conditioning on it is a bad
   control and does not identify a "direct effect". Use it only as an outcome and as a
   heterogeneity dimension at baseline levels.
3. **`mine_priceshock_*` is not yet a clean shift-share instrument.** It sums
   `share × ln(real price)` over mines *active that year*. Exposure therefore moves with
   openings and closures, which respond to local conditions. Build a variant with **exposure
   fixed at baseline** (mines active by 2002, with their commodity shares), so the only time
   variation comes from world prices. Only then do the Borusyak–Hull–Jaravel and
   Goldsmith-Pinkham diagnostics apply. Keep the count instrument as the secondary one.
4. **ESA CCI is aggregated by `mode` at coarse grids.** The modal class hides built-up fractions,
   so a land-cover split at 5–10 km is almost meaningless. Add **per-class fraction variables**
   (the `average` of class indicators) for built-up (LCCS 190), cropland, forest, bare and water.

## 4. Analysis flow

### Phase 0 — Fix the specification

Write this down before any final run. Everything outside it is robustness or exploration.

- **Estimand:** the local thermal effect of growing faster than other cells in the same
  country × biome in the same year, *including* spillovers up to R_max = 30 km (Σ_r β_r, per
  [`00`](../design/00-backbone-overview.md)). It is identified as a LATE for mine-proximate compliers,
  and the paper's claims are scoped to that population.
- **Outcome:** `lst_night_mean` (MYD21A2). Day LST, GLASS air temperature and the valid counts
  are diagnostics.
- **Treatment:** `log1p(ntl_harm)`, own cell plus ring means. One threshold everywhere:
  `NTL_HI = 20`, with 30 as a robustness check. Mask `flare_band > 0`.
  - ◇ Decide inconsistency 6 before Phase 4. Two options:
    - keep `ntl_harm`, work at ≥ 5 km, and start the rings beyond the blur radius;
    - use `eog_viirs` at 1 km on the 2012–2021 panel. Its first stage was near zero in R5.

    Record the choice and why.
- **FE:** `pixel_id + GID_0^biome_id^year`. Plain `GID_0^year` is a robustness check.
- **Inference:** clustered by ADM1, with Conley spatial SEs as a companion (cutoff ≥ R_max),
  and Anderson–Rubin CIs for every IV. Ignore HC1: R7 already showed it is roughly 7× too small.
- **Sample:** Aqua years 2002–2022, |φ| ≤ 60°, land mask.
- **Grid:** 1 km, restricted to cells within 50 km of any SNL mine, for everything that
  involves footprints, donuts, land-cover fractions or rings. The 10 km full panel is kept for
  iteration and for the Phase 2 benchmark.

**Ring IV.** The ring model has one endogenous term per annulus. Instrument each with the
baseline-share exposure averaged over the same annulus.

◇ If the joint first stage is too weak (effective F < 10 for any ring), collapse the rings into
one area-weighted neighbourhood NTL index, instrumented by the matching instrument index. Report
Σ_r β_r from the reduced-form ring coefficients, and state that the per-ring split is not
identified.

### Phase 1 — Measurement audit (no causal claims)

- **1a. Outcome product.** On the 11A1 arm's 5 tiles × 3 years:
  - Test whether (11A1 − 21A2) night LST varies with land-cover class and fraction. This
    measures the emissivity bias the product choice avoids.
  - Test whether 21A2's valid-period count tracks 11A1's true daily clear-sky counts.
- **1b. Compositing selection.** Regress `valid_period_count_night_annual` on NTL, on PM2.5 and
  on the instrument (reduced form), with the Phase 0 FE. Repeat with
  `valid_month_count_night_annual` (months with any valid observation). That is the only
  seasonal coverage signal in the current output. A month-by-month breakdown would need FETCH
  to persist monthly bands again, which means a full STAC re-stream.
  - ◇ If selection is detected, the primary outcome becomes a coverage-balanced sample
    (`valid_month_count_night_annual == 12`), with Lee-type bounds next to it. Seasonal
    reweighting is possible only after the monthly-band pipeline change.
- **1c. Entanglement, measured before any IV.** What share of within-cell Δ`log1p(ntl_harm)`
  variance occurs in cell-years with a change in built-up fraction, and what share is
  "brightening in place"? This descriptive number decides how much the Phase 6b strata can
  possibly deliver.
- **1d. Treatment series checks:**
  - the DMSP→VIIRS transition at `NTL_HI = 20` (the descriptives suggest the artefact is gone);
  - the flare mask;
  - VIIRS `viirs_annual_cf_cvg` as a coverage diagnostic on the treatment side.

### Phase 2 — Descriptive benchmark

Run the OLS ladder (pooled → pixel FE → TWFE) for night LST, day LST and GLASS, with binscatters.
The existing day-LST version shows the pooled −1.6 K is cross-sectional selection. Redo it for
night LST, and add the own-cell vs ring TWFE comparison to show how much of the neighbourhood
signal pixel FE removes.

### Phase 3 — Build and validate the instruments (gate before any 2SLS)

- **3a. Construction:**
  - **Primary:** baseline-share price-shock (§3.3).
  - **Secondary:** mine count.
  - **Donut variants:** drop cells with `mine_polygon_count > 0` from the *outcome* sample. Also
    drop cells within a fixed distance (e.g. 2 km) of any mine point, because about 43 % of
    active mine-years have no footprint polygon. Build exposure over the 10–20 km and 20–50 km
    annuli.
- **3b. First stages.** Effective F (Montiel-Olea–Pflueger) for every variant, per ring and
  pooled.
  - ◇ If F < 10, use AR-only inference and pool the rings (see Phase 0).
- **3c. Exclusion diagnostics:**
  - **Event study around SNL mine openings** (leads and lags), estimated separately for
    footprint cells and ring cells. Outcomes: NTL, night LST, day LST, PM2.5 and land-cover
    fractions. If LST responds in the footprint but not in the ring, that is extraction physics,
    not spillover. If LST moves in the ring *before* NTL does, that also argues against a clean
    growth channel.
  - **Shift-share diagnostics:**
    - Rotemberg weights by commodity: which commodities drive identification?
    - Share-balance tests of baseline exposure against 2002–05 night-LST trends, biome and
      baseline land-cover fractions.
    - Exposure-robust (shock-level) SEs.
  - **Pre-period placebo:** does exposure to mines that open later predict GLASS AVHRR
    temperature trends over 1992–2001?
  - ◇ If LST responds only in the footprint, or pre-trends show up, the donut specification
    becomes primary and the undonutted estimate is reported as contaminated.
- **3d. Favoritism.** Re-specify at ADM2 with region FE + country × year (Hodler–Raschky style)
  on an ADM2 panel.
  - ◇ If the first stage is still dead, report the reduced form in an appendix only, and drop
    favoritism from the causal claim. The reduced form is itself evidence of a non-lights
    channel.

### Phase 4 — Main estimate

- Ring 2SLS on night LST with the preferred instrument, using the Phase 0 specification, AR CIs
  and Conley SEs. Report Σ_r β_r first, then β_0 and the ring profile.
- **Complier characterization:** how the compliers are distributed across biome, income group,
  baseline built-up fraction and baseline PM2.5. This turns "LATE" into a concrete scope
  statement.

### Phase 5 — Robustness to design threats

- Cell-specific linear trends.
- Biome × year and income × year FE, and plain `GID_0^year`.
- NTL functional form (`log1p`, `asinh`, threshold 20/30).
- Drop 2013–14; sensor-era interaction.
- 11A1 on the subsample it covers.
- Grid resolution (1/5/10 km) plus grid-shake (`--shake quad`) at coarse grids.
- The primary vs balanced outcome from 1b.

Present these as **one specification curve** rather than one table per axis. It shows the whole
distribution of estimates, and it makes the multiple-testing problem visible instead of hiding it.

### Phase 6 — Channels and heterogeneity (the causal-lens critique)

- **6a. Mechanism first stages.** Estimate the effect of the instrument (reduced form) and of
  instrumented NTL on:
  - built-up and cropland fractions (the land-cover channel);
  - PM2.5 (the aerosol channel);
  - valid-observation counts (the compositing channel).

  Their signs and magnitudes give a channel accounting. They do not identify a formal
  decomposition, and the paper should say so.
- **6b. Baseline land-cover strata** (§3.1). The coefficient in cells with no room to convert
  is the closest thing the data offers to "growth without a footprint".
- **6c. Day vs night; GLASS vs LST.** A day coefficient much larger than the night one points
  to a reflectance artefact. Only a *divergence* between GLASS and LST is informative.
- **6d. Heterogeneity** by realm and biome, arid vs humid, baseline PM2.5 terciles, and
  HDI/income group. Large, opposite-signed stratum effects would support the "averages to
  nothing" reading of a null.
- **6e. Spillover profile.** The β_r ring profile from Phase 4, and ring means of the *outcome*
  around treated cells. Warming that leaks into nearby control cells violates SUTVA and biases
  β_0 toward zero.

### Phase 7 — Interpretation and scoping

- Equivalence tests (TOST) and minimum detectable effects on Σ_r β_r: which effect sizes the
  data rules out. This is the honest answer to "no effect vs offsetting effects".
- **Reframing rules** (from the review):
  - ◇ If the effect is null in the no-room-to-convert strata but present where land cover
    converts, the claim becomes "growth with a land-cover footprint", not "economic growth".
  - ◇ If the GLASS and LST results diverge, report both and say which construct each measures.
- **Final claim.** A net effect for resource-driven growth (and favoritism-driven growth, if
  3d survives), among mine-proximate compliers, stated net of the channels in 6a.

## 5. Ordering chart

```
[0] FIX SPEC ─ estimand Σβ_r (R_max 30 km) · night LST (21A2) · log1p NTL, NTL_HI=20, flare mask
    │           ◇ lights input: ntl_harm ≥5 km vs eog_viirs 1 km (inconsistency 6)
    │           FE: pixel + ctry×biome×year · ADM1 + Conley SE · AR CIs
    │           1 km mine-proximate (≤50 km) for main work · 10 km full panel for iteration
    │
    │  ── data gaps to close first (§6) ─────────────────────────────────────────
    │  • baseline-share price-shock   • ESA CCI class fractions   • donut/ring instruments
    │  • glass_avhrr in assembly      • ADM2 panel for reg_fav
    ▼
[1] MEASUREMENT AUDIT ────────────────────────────────────────────────────────
    1a 11A1 vs 21A2 (5 tiles × 3 yrs): emissivity–LC bias, valid-count proxy
    1b valid_count_night ~ NTL / PM2.5 / Z  (+ valid-month counts)
         ◇ selection? ──yes──► outcome := 12-month-coverage sample + bounds
    1c share of ΔNTL in cells with built-up change (entanglement, descriptive)
    1d DMSP→VIIRS at NTL_HI=20 · flares · VIIRS cf_cvg
    ▼
[2] DESCRIPTIVE BENCHMARK ─ OLS ladder × {night, day, GLASS} · own-cell vs ring TWFE
    ▼
[3] INSTRUMENT VALIDATION (gate) ─────────────────────────────────────────────
    3a build: baseline shift-share (primary) · count · donut (10–20, 20–50 km)
    3b first stages, effective F     ◇ F < 10? ──► AR-only, pool rings into one index
    3c event study at openings: footprint vs ring × {NTL, LST n/d, PM2.5, LC}
       shift-share: Rotemberg wts · share balance on 2002–05 trends · shock-level SEs
       GLASS-AVHRR placebo, 1992–2001
         ◇ footprint-only LST response / pre-trends? ──► donut becomes primary
    3d reg_fav at ADM2 (region FE)   ◇ FS dead? ──► appendix reduced form only
    ▼
[4] MAIN ESTIMATE ─ ring 2SLS night LST → Σβ_r, β_0, ring profile · complier profile
    ▼
[5] ROBUSTNESS ─ cell trends · FE variants · NTL form · sensor era · 11A1
    │            grid 1/5/10 km + grid-shake · balanced outcome → specification curve
    ▼
[6] CHANNELS & HETEROGENEITY ─────────────────────────────────────────────────
    6a Z → {built-up/cropland frac, PM2.5, valid counts}   (outcomes, never controls)
    6b effect by BASELINE LC strata (no-room-to-convert ≈ intensive margin)
    6c day vs night · GLASS vs LST divergence
    6d realm/biome · arid/humid · baseline PM2.5 · HDI/income
    6e spillover profile β_r · outcome rings around treated cells (SUTVA)
    ▼
[7] INTERPRETATION ─ TOST / MDE on Σβ_r
                     ◇ effect only where LC converts? ──► "growth with a land-cover footprint"
                     → scoped claim: resource-driven (± favoritism) growth,
                       mine-proximate compliers, net of the channels in 6a
```

## 6. Data and pipeline gaps (block Phases 3–6)

| Gap | Where it lands | Blocks |
|---|---|---|
| Baseline-share price-shock variant (exposure frozen at mines active by 2002; only prices vary) | new radius variables in `snl_mining.aggregation` (`src/data/sources/snl_mining/source.py`) | 3a, 3c, 4 |
| Donut/annulus instrument and treatment rings up to 30 km. The engine ([`03`](../design/03-neighbourhood-engine.md), `src/data/common/neighbourhood/`, `src/data/assemble/ring_means.py`) exists but is **not wired** into PREPARE or assembly; only `scripts/validate_backbone_subset.py` calls it | a ring stage producing disc-sum columns per variable, joined in assembly | 0, 2, 3a, 4, 6e |
| ESA CCI per-class fractions (built-up, cropland, forest, bare, water) | a fraction family next to `lccs_class`; `average` resampling in `assembly.sources.esacci` | 1c, 3c, 6a, 6b |
| Baseline strata (2002 built-up fraction, 2002 PM2.5 tercile) | derived columns at panel build | 6b, 6d |
| `glass_avhrr` in `assembly.sources`, after switching its annual mean to month-first compositing (it is currently a naive mean over valid days, so seasonally biased) | `data.yaml`; `glass/avhrr.py::_calculate_statistics` | 3c placebo |
| ADM2 panel for favoritism | `adm2_1km` data source (`src/analysis/core/config.py`) | 3d |
| Weak-IV tooling (effective F, AR CIs) and Conley SEs with `duckreg` | analysis layer | 0, 3b, 4 |
| 11A1 arm joined onto the 21A2 grid for 1a | ad-hoc notebook, not the main panel | 1a |
| Distance-to-nearest-mine column (mine points, not polygons) for the donut | `snl_mining` PREPARE | 3a |
| MODIS monthly bands (optional, only if 1b needs a seasonal breakdown) | `ModisSource._execute_fetch` + a full re-stream | 1b |

## 7. Open questions

1. **Can the ring IV be identified at all?** Mine exposure is rare (~5 % of 10 km pixel-years
   have a mine within 20 km). One instrument per annulus may be too weak, so the pooled-index
   fallback in Phase 0 may end up as the main result. Phase 3b answers this.
2. **ACAG PM2.5 has the same clear-sky problem.** ACAG V6 draws on satellite AOD, which is
   retrieved only under clear skies. Its own coverage may be thinned in the same hazy years as
   MODIS LST, which would bias 6a toward finding no aerosol response. Check whether ACAG's
   documentation reports a per-cell data-support measure.
3. **Terra/Aqua orbital drift** ([`14`](../design/14-terra-aqua-drift-diagnostic.md)) moves Aqua's night
   overpass time after about 2020–21. Country × biome × year FE absorb the common part.
   Whether the drift differs across latitude bands within a country is still open. Consider
   ending the panel in 2020 as a robustness check.
4. **Which grid is the main one?** 1 km matches [`00`](../design/00-backbone-overview.md) and is required
   for footprints. The IV may still be too noisy at 1 km even on the mine-proximate subsample.
   Decide after 3b, and record the choice so it does not look like picking the grid by result.

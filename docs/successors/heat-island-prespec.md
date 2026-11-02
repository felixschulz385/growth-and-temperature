# Surface Heat Island with a Continuous Urban Measure: Pre-Specification

**Status: draft 2026-10-30, not confirmed. Nothing has run and no LST has been loaded for this
project.** Idea A of [`ideas.md`](ideas.md). It first replicates the published city-level
surface heat island in our pipeline, then estimates how a 1 km cell's LST changes as its
impervious share rises, and compares the two. Several design elements follow Dell, Jones and
Olken (2012, *AEJ: Macro* 4(3)), who answer the reverse question (temperature → growth); §3
names each one. **Descriptive only: paving is chosen, not as good as random, so no causal
claim.**

## 0. Decisions for the author

To confirm before G0.

1. **Main treatment: GAIA** (annual, 30 m, 1985–2018), aggregated to the share of a 1 km cell
   that is impervious in year t. GHSL built-up (5-year epochs) is used for the long differences
   (step 3b) and as a robustness check. GAIA is annual, so it allows lags. It is monotone by
   construction (one year of change per pixel), and it marks a 30 m pixel impervious above
   50 %, so it does not see densification inside an already impervious pixel.
2. **Years: 2003–2018.** Aqua LST starts mid-2002 and GAIA ends in 2018. This also keeps clear
   of the Aqua night-overpass drift after 2022 (conclusion §5).
3. **Sample: study areas, not the globe.** 1 km cells within 30 km of a benchmark urban
   cluster's extent (§2). Cells far from any city have almost no change in impervious share, and
   a global 1 km panel for 16 years would be too large to fit. G0 reports the size.
4. **LST product for the replication (step 1).** (a) MYD21A2, which is in the pipeline, with the
   product difference accepted; or (b) extend the `modis` source to MYD11A2 for the study-area
   tiles, to match the benchmark's product. **Recommended: (b).** The replication is only a clean
   check if it uses the same product, and the MYD11 − MYD21 gap is the emissivity check in §4
   anyway. Steps 2–4 use MYD21A2 in both cases.
5. **No new weather source.** Country × biome × year FE absorb regional weather, as in the
   retired specification. Precipitation enters only as a climatology for the heterogeneity in
   step 4 (WorldClim, small download).

## 1. Question

1. Does our pipeline reproduce the published surface heat island of the world's cities?
2. Within a 1 km cell, how much do day and night LST change as the cell's impervious share
   rises, and how does the change build up over the years after paving?
3. Does the within-cell change, accumulated, reproduce the cross-sectional heat island? If not,
   by how much does it differ, and in which climates?

Question 3 is the contribution. The literature compares cities with their rural surroundings at
one time, or tracks that difference over time. It does not test whether change within a place
adds up to the difference between places.

## 2. Data

| | Source | Status |
|---|---|---|
| Unit | 1 km EASE-Grid 2.0 cells in study areas (§0.3) | 1 km panel not assembled. `assembled/1km.parquet` (2026-02) predates the rebuilt backbone and is not used |
| Outcome | MODIS Aqua LST, day and night, annual mean of valid 8-day composites | `modis` (MYD21A2, 2002–2025) |
| Outcome, step 1 | MYD11A2 for study-area tiles (§0.4) | new: extension of `modis` |
| Treatment | GAIA v1.0: share of 30 m pixels impervious by year t, per 1 km cell | new source `gaia`; FETCH via the transfer node |
| Treatment, step 3b | GHS-BUILT-S R2023A built-up surface share, 5-year epochs | new source `ghs_built` |
| Rings | impervious share in rings of 1–5, 5–10 and 10–30 km around each cell | ring stage of the assembly ([`design/03`](../design/03-neighbourhood-engine.md); engine exists, not wired) |
| Benchmark | YCEO Surface Urban Heat Islands v4 (Chakraborty and Lee 2019): yearly day and night SUHI per urban cluster, 2003–2018, about 10,000 clusters, CC-BY-4.0 | new; small download (path to confirm) |
| Urban extents | the benchmark's cluster extents and rural masks if published with it; otherwise rebuilt from the paper's method (ESA CCI urban class, elevation rule, water excluded) | G1 records which |
| Elevation | GMTED2010 elevation (EarthEnv), for the rural elevation rule | extension of `topography` (now slope only) |
| Heterogeneity | RESOLVE biome; WorldClim 2.1 annual precipitation (bio12); ESA CCI tree + crop share in 2002; baseline lights per person of the ADM1 region; GHS-POP 2000 population of the cluster | `ecoregions`; new (small); `esacci` with `average` resampling; `ntl_harm`; `ghs_pop` |
| Independent outcome | GLASS air temperature (2000–2020) | `glass_ta_modis`. Partly derived from MODIS LST, so semi-independent |
| Retrievals | count of valid 8-day composites per cell-year | must be kept by `modis` PREPARE (checked in G0) |

## 3. Steps and specification

### Step 1. Replication (cross-section)

- For each benchmark cluster and year: SUHI = mean LST over the urban extent − mean LST over the
  rural reference, day and night, with the benchmark's definitions.
- Compared with the benchmark cluster by cluster (G1).
- **The Manoli pattern** (Manoli et al. 2019): across clusters, daytime SUHI against ln
  population and ln precipitation, in our data and in the benchmark's.

### Step 2. Heat islands over time (descriptive)

- Cluster-year panel: SUHI on the urban extent's mean impervious share and the rural
  reference's, with cluster and year FE.
- Information only; no rule depends on it.

### Step 3. Within-cell change (main)

**Main model, a distributed lag in first differences** (Dell et al. eq. 4):

$$\Delta \text{LST}_{it} = \sum_{j=-2}^{5} \rho_j\, \Delta \text{imp}_{i,t-j} + \sum_{r} \sum_{j=0}^{5} \kappa_{rj}\, \Delta \text{ring}_{r,i,t-j} + \delta_{c(i)\,b(i)\,t} + \varepsilon_{it}$$

- imp is the cell's impervious share (0–1); ring_r is the share in ring r.
- Day and night separately.
- First differences rather than levels: both LST and the impervious share trend upward, and a
  regression in levels with cell FE then leans on the common trend (Dell et al., footnote 10,
  after Bond et al. 2010).
- **Reported:** the immediate effect ρ₀; the **cumulative effect θ = Σ_{j=0}^{5} ρ_j** (the
  long-run effect of a unit rise in the stock); the leads ρ₋₂ and ρ₋₁; the ring sums
  Σ_j κ_{rj}.
- FE: country × biome × year (δ). Clusters: ADM1. Weights: none (area).

**Step 3b. Long differences** (Dell et al. §IV): the change in mean LST from 2003–07 to
2014–18 on the change in mean impervious share, with ring changes and country × biome FE. A
second version uses GHSL built-up at the epochs 2005 and 2015. This reaches further toward the
long run than five lags and averages out MODIS noise from year to year.

**Step 3c. Event study.** An event is the first year in which a cell's impervious share rises
by at least 20 percentage points. Control cells are those not yet treated, in the same
country × biome. The window is −3 to +6 years, with the Callaway and Sant'Anna estimator. This
guards against the weighting problems of TWFE with staggered and continuous treatment.

### Step 4. Heterogeneity and functional form

- **Horse race of baseline traits** (Dell et al. Table 2, cols. 4–5): Δimp interacted with all of
  the following at once, each fixed at baseline:
  - climate: precipitation climatology, and biome;
  - baseline vegetation: ESA CCI tree + crop share in 2002;
  - development: baseline lights per person of the ADM1 region;
  - city size: the cluster's GHS-POP population in 2000.
- **Year effects by group:** a variant adds baseline urban class × country × year FE (cells
  above and below 10 % impervious in 2003; Dell et al.'s poor × year FE). A drought year heats
  rural cells more than paved ones, and that must not be credited to paving.
- **Bins of the one-year change** in impervious share, as in the retired B2: 0, (0, 5],
  (5, 15], (15, 30], > 30 percentage points.

### Variants (reported, not selected)

1. + cell-specific linear trends (Dell et al. Table 4, col. 2).
2. ADM1 × year FE instead of country × biome × year.
3. Population weights (GHS-POP), for the exposure that matters for health.
4. Coverage-balanced: cells with at least 40 of 46 valid composites in every year.
5. MYD11A2 outcome on the study-area tiles (if §0.4 (b)).
6. GLASS air temperature as the outcome.
7. Conley standard errors at 50 km. They need support in `duckreg` or pyfixest; until then,
   two-way clusters by ADM1 and country × year.

## 4. ◇ Rules, fixed before any LST is loaded for this project

- **◇ G0, data (loads no LST).**
  - At least 100,000 study-area cells whose impervious share rises by ≥ 10 percentage points
    over 2003–2018, in at least 50 countries.
  - At least 20,000 event cells for step 3c.
  - The panel fits the memory of one compute node, or is split by continent with a stated rule.
  - The count of valid composites exists per cell-year.
  - If a count is missed, the sample rule (§0.3) is revisited through the change log before G1.
- **◇ G1, replication (loads LST, cross-section only).** Passes if all three hold, day and
  night:
  - the Spearman correlation of our per-cluster mean SUHI (2003–2018) with the benchmark's is
    at least 0.7;
  - our mean over clusters is within 0.3 K of the benchmark's mean over the same clusters;
  - the share of clusters with a positive SUHI is within 10 percentage points of the
    benchmark's.

  **If G1 fails,** steps 2–4 do not run until the gap is diagnosed. If the diagnosis points at
  data the retired project also used (MYD21A2 night LST), the conclusion gets a dated
  amendment.
- **◇ G1b, Manoli pattern (information, not a gate).** Signs of the population and precipitation
  slopes, in our data and in the benchmark's.
- **◇ Pre-trends.** If the leads ρ₋₂ and ρ₋₁ are jointly different from zero at 5 %, the main
  claims rest on step 3c (event study) instead of step 3.
- **◇ The main test: does within-cell change add up to the cross-section?** (Dell et al. §V.)
  - For each cluster c: predicted SUHI_c = θ × (mean imp over the urban extent − mean imp over
    the rural reference), in 2018. θ is the cumulative effect of step 3, by day and night.
  - R = mean predicted SUHI / mean replicated SUHI (step 1), over the clusters of G1. The CI
    comes from a cluster bootstrap over ADM1 regions (step 3) and urban clusters (step 1).
  - **Reading:**
    - "the designs agree" if the 95 % CI of R includes 1 and lies inside [0.5, 2];
    - "within-cell change explains less (more) than the cross-section" if the CI lies below
      (above) 1;
    - otherwise "inconclusive".
  - The test is repeated with θ from the long differences (3b). If the two readings differ, the
    conclusion reports both and says the effect builds up for longer than five years.
  - It is also repeated by climate group (precipitation terciles across clusters).
- **◇ Spillovers.** If the cumulative ring-1 effect (1–5 km) is at least half the own-cell θ,
  with the same sign, at 5 %, the conclusion says the rural reference of step 1 is affected by
  paving. The main test is then also reported with the ring effect subtracted from the rural
  reference: predicted SUHI_c = θ × gap − Σκ₁ × (change in ring-1 imp of the rural reference).
- **◇ Heterogeneity.** A trait "carries" the heterogeneity only if its interaction is
  significant at 5 % with all the others in the model. Otherwise: "no trait stands out".
- **◇ Measurement checks** (Dell et al.'s checks against exports and station dropout).
  - **Emissivity:** a claim about θ stands only if θ has the same sign with MYD11A2 (variant 5)
    as with MYD21A2.
  - **Retrievals:** the count of valid composites, regressed on Δimp with the step 3 model. If
    its cumulative effect is significant at 5 % and larger than 5 % of the mean count, θ is
    claimed only where it keeps its sign in variant 4 (coverage-balanced).
  - **Air temperature** (variant 6): information only, since GLASS Ta is partly derived from
    LST.
- **Reading.** Every coefficient is a within-cell association. The paper uses no causal language.

## 5. Outcomes

None yet.

## 6. Already seen (disclosure)

- **Retired project, B1–B3** (10 km cells, log lights, not impervious share):
  - night LST −0.030 K per log point of lights with pixel + country × biome × year FE;
  - by day, brightening goes with warming (the heat-island sign), and the night cooling comes
    from cells lighting up, not from brightening (B2);
  - under neighbour differences the night association is zero (B3): it is shared across tens of
    km, not local.
- **Retired channel table (conclusion §2.3):** built-up surfaces are expected to warm by day
  and more by night. That is the literature's sign, not an estimate.
- **Fork, `06d_dose_twfe.ipynb`:** night LST −0.26 K per log point of lights at ADM1.
- **Published:** the benchmark's global means are about 0.85 K by day and 0.55 K by night
  (Chakraborty and Lee 2019).
- **Not seen:** LST against impervious share at any scale, in any form, in this repository.

## 7. Data to build (none of it loads LST)

1. `gaia` source: FETCH on the transfer node; PREPARE to the 1 km share by year.
2. `ghs_built` source (GHS-BUILT-S R2023A).
3. Benchmark (YCEO SUHI v4) and its cluster extents: small; the download path needs to be
   confirmed (an Earth Engine export needs an account).
4. `topography`: add GMTED2010 elevation. WorldClim bio12: small.
5. `esacci`: `average` resampling for class fractions (conclusion §5, data gaps).
6. `modis`: the MYD11A2 extension (§0.4 (b)), and the count of valid composites.
7. Assembly: a 1 km panel for the study areas, with the ring stage.

## Change log

None.

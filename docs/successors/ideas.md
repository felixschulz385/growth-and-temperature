# Successor Ideas

**Status: documented 2026-10-28. Not scheduled; nothing here is being worked on.** Each idea
records a sketch, the data, a critique and a feasibility call. The fork that is being started is
in [`fork-corridors-dose.md`](fork-corridors-dose.md).

## How the ideas connect

The retired project showed that growth has no single effect on temperature: each kind of growth
changes land cover, waste heat and aerosols in its own mix (conclusion §3). What can carry over
between kinds of growth is the physics of each channel:

$$\Delta \text{LST}_g = \sum_k \theta_k \cdot \frac{\partial X_k}{\partial g}$$

- $\theta_k$ is the temperature response to a unit change in channel $k$.
- $\partial X_k / \partial g$ is how growth of kind $g$ changes that channel.

The three ideas are pieces of this equation. Idea A estimates $\theta$ for impervious surface.
Idea B describes $\partial X / \partial g$ for aerosols, and how institutions change it. Idea C
assembles the whole chain.

## A. Surface heat island with a continuous urban measure

**Question.** How much does a cell's LST change when its impervious share rises, within the cell
over time, by day and night and by climate?

**Design.**
- 1 km cells, annual panel, 2002–2018 (GAIA ends in 2018).
- Treatment: impervious fraction from GAIA (annual, 30 m); GHSL built-up (5-year epochs) and
  ESA CCI urban fractions as robustness checks.
- Outcome: MYD21A2 LST, day and night.
- Estimator: continuous-treatment difference-in-differences (Callaway, Goodman-Bacon and
  Sant'Anna 2024), or an event study around increases in the impervious share. Plain TWFE is
  biased here (next section).
- Spillovers: distance rings or neighbour differences around each cell.
- Heterogeneity by climate and background vegetation (cf. Manoli et al. 2019).

**Critique.**
- **Novelty.** The literature is crowded: Peng et al. 2012, Clinton and Gong 2013, the
  Chakraborty and Lee global surface heat-island dataset, Manoli et al. 2019. The contribution is
  change within a cell against the usual urban-vs-rural comparison of places. The paper has to
  show the two give different answers.
- **The treatment almost never reverses.** GAIA impervious area does not decrease. Plain TWFE then
  compares cells with already-treated cells and can give wrong-signed weights.
- **Rural comparison cells are affected too.** The retired project found the lights–LST link is
  shared across tens of km, so cells near cities are not clean controls.
- **Retrieval bias.** Paving changes surface emissivity, which biases MODIS LST. MYD21 handles
  this better than MYD11; the project's MYD11 vs MYD21 check by land-cover class applies.
- **Confounded with what comes along.** Urban expansion goes with irrigation and greening around
  cities, which cool the surface.

**Feasibility: high.** The 1 km panel, MODIS LST and ESA CCI exist in the pipeline. GAIA and GHSL
would be new sources. This is the most likely of the ideas to yield a result, and it supplies
$\theta_{\text{impervious}}$ for idea C.

## B. Pollution along the growth path, by strength of institutions

**Question.** How does PM2.5 change as regions grow richer, and does the path differ with
political institutions? This is descriptive, not causal.

**Design.**
- GADM-1 region × year, 1998–2020.
- Outcome: ACAG PM2.5 at its native resolution, aggregated to regions.
- Growth: log GRP per capita from DOSE (see the fork), in bins or a spline rather than a fixed
  polynomial.
- Institutions: V-Dem or WGI, interacted with the growth bins.
- Comparison: region FE + year FE, with country trends as a variant.

**Critique.**
- **The literature is crowded and contested.** Environmental Kuznets curve results are fragile
  (Stern 2004: nonstationarity, sensitivity to functional form). Institutions as a moderator are
  well studied (Dasgupta et al. 2002; Bernauer and Koubi 2009).
- **Institutions vary between countries and change slowly.** The moderating effect then comes
  mostly from cross-country comparisons, which country-level confounders easily explain.
  Subnational data help with growth, not with institutions.
- **PM2.5 includes natural sources.** Dust and wildfire smoke dominate in dry regions. Check
  whether ACAG's dust- and sea-salt-free variant covers the panel years. Source attribution
  exists only for single years (McDuffie et al. 2021).
- **DOSE coverage** leans towards richer and larger countries, which narrows the institutional
  variation.
- **Framing.** As a descriptive atlas of pollution paths by type of institution it is publishable;
  it should not claim to be causal.

**Feasibility: high on data** (ACAG is in the pipeline; DOSE and V-Dem are easy to add).
**Medium on contribution.**

## C. A structural chain: growth → channels → temperature

**Model.** This follows the structure of Qiu et al. 2025 (*Nature*, doi:10.1038/s41586-025-09611-w).
That paper chains separately estimated models (climate → wildfire smoke PM2.5 → mortality) and
carries uncertainty through every link. Here:

1. **Stage 1, growth → channels.** For each kind of growth $g$ (one instrument each: mines,
   corridors, ...), estimate how it moves each channel $X_k$:
   - impervious share (GAIA or GHSL);
   - vegetation (Hansen forest loss, MODIS NDVI);
   - aerosols (MAIAC AOD);
   - waste heat (no panel exists; see the critique).
2. **Stage 2, channels → LST.** Take $\theta_k$ from physical studies (Duveiller, Hooker and
   Cescatti 2018 for vegetation; idea A for impervious surface), separately by biome and for day
   and night. Panel estimates are a check, not the main source.
3. **Composition.** $\Delta \text{LST}_g = \sum_k \theta_k \, \partial X_k / \partial g$.
   Uncertainty comes from bootstrapping stage 1 and drawing $\theta$ from its published ranges.
4. **Validation.** The composed effect must lie inside the instrument's reduced-form AR set (for
   mines, ±0.3 K per log point).

**Why it might work where the IV did not.** The binding problem was power: the IV divides a noisy
LST reduced form by a weak lights first stage. Growth moves the channels much more strongly than
it moves LST (land cover at mines changes visibly), so stage 1 has strong first stages. The
physical θ are precise. The composed effect could be far tighter than the reduced-form bound.

**Critique.**
- **Completeness.** The composed effect includes only the modelled channels. Waste heat has no
  annual gridded panel: existing anthropogenic heat flux datasets are static or built from energy
  statistics and lights. A missing channel biases the composition without warning, and the only
  guard is the reduced-form check, which is wide (±0.3 K).
- **Stage 2 is hard to identify from the panel.** Channels move together with growth. A regression
  of LST on all channels at once is confounded by the missing channels and by regional weather,
  which is the Story 1 problem again. External θ avoid that, but then assume that physics from
  other sites and scales transfers.
- **Additivity.** Channels interact: the aerosol effect depends on the surface, and the sign of
  clearing depends on the climate zone. θ by biome helps; interaction terms would need data the
  physics literature rarely gives.
- **Scale.** Aerosols act regionally, the other channels locally. The chain needs two scales,
  which a cell-level stage 1 cannot provide.
- **The analogy with Qiu et al. is partial.** Their driver (climate) is exogenous, and their middle
  variable (smoke PM2.5) is well measured. Here, growth is exogenous only through an instrument,
  and the middle variables are poorly measured at 10 km.
- **Assumptions.** As a mediation analysis it needs sequential ignorability (Imai et al. 2010):
  nothing that affects both channel and LST may be left out.

**Feasibility: medium.** It depends on idea A and on channel data the pipeline lacks (Hansen forest
loss, MAIAC AOD, GAIA/GHSL; listed as gaps in conclusion §5). It extends the conclusion's channel
accounting (§2.3) with two additions: kind of growth as an explicit dimension, and external θ.

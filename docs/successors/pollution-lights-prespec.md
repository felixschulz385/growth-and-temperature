# Pollution Along the Growth Path, by Institutions, with Night Lights: Pre-Specification

**Status: confirmed 2026-10-30 (drafted 2026-10-29), as drafted. G2 and G2b passed. The
main specification and all variants ran (2026-10-30, §5). Summary with idea B in
[`pollution-results.md`](pollution-results.md).** Step 1 of [`pollution-data-options.md`](pollution-data-options.md) §4. It repeats
[`pollution-institutions-prespec.md`](pollution-institutions-prespec.md) (idea B), but measures
growth with night lights instead of DOSE output, so that it covers every country. **Descriptive
only, no causal claim.**

## 0. Decisions for the author

**Confirmed as drafted, 2026-10-30.**

1. **Growth measure: lights per person** (needs `ghs_pop` prepared), rather than lights per area,
   which needs no new data. Per person is the analogue of output per person, and it places a
   dense poor city correctly against a sparse rich region. With region FE, lights per area
   equals total lights, which mixes in population growth. Per area becomes variant 7.
2. **No weather controls in the main specification.** Outside DOSE the panel has GLASS air
   temperature (2000–2020) but no precipitation. A temperature-only control is partial, and
   tercile × year FE absorb common shocks. GLASS temperature becomes variant 8. Precipitation
   would need a new source (ERA5 or CHIRPS).
3. **Haze guard** (§4): claims must hold in regions with low baseline PM2.5, where aerosols dim
   lights least.
4. **Knots calibrated to DOSE** (§3), so that the segments mean roughly the same incomes as in
   idea B.

## 1. Question

The same as idea B. Here the sample includes poor democracies and rich autocracies, which DOSE
lacks (idea B, G1: no high-democracy region below 1,000 USD).

## 2. Data

| | Source | Module or column |
|---|---|---|
| Unit | GADM 4.1 first-level regions (GID_1) in countries with V-Dem data | `gadm` (panel `GID_1`) |
| Outcome | ACAG V6GL02.04 PM2.5: region mean over its 10 km cells (area-weighted), as in idea B | panel `pm25` |
| Growth | ln(lights per person): Σ `ntl_harm` over the region's cells ÷ population (Σ GHS-POP density × cell area, interpolated log-linearly between epochs) | panel `ntl_harm`; `ghs_pop` |
| Institutions | V-Dem `v2x_polyarchy` (main), `v2x_rule` (variant 3), baseline 1998–2000 | `vdem` |
| Dust | the region's share of cells in RESOLVE biome 13 | panel `biome_id` |

- **Years:** 1998–2012, the DMSP years (changed 2026-10-30, change log; first 1998–2022). ACAG
  starts in 1998. The VIIRS years 2014–2022 are variant 6, and 2013, the splice year, is in
  neither.
- **Regions:** at least 5 cells of 10 km with both PM2.5 and lights (≥ about 500 km²), so
  small islands and city regions do not add noise from a cell or two.
- **Region-years with zero lights** are dropped. G2 reports how many and where.
- **Gas flares:** 1 km cells within 5 km of an upstream flare in any survey year (EOG
  `flare_band` > 0, 2012–2021) are left out of the lights and the population sums in every year
  (added 2026-10-30, change log).
- **Lights:** `ntl_harm` in DMSP-like digital numbers, spliced to VIIRS in 2013/14. Year FE
  absorb a global level shift at the splice, but not one that differs with brightness (variant
  6).

## 3. Specification

- **Outcome:** ln PM2.5.
- **Path:** a piecewise-linear spline in ln(lights per person) with three knots, interacted
  with the democracy tercile.
- **Knots, calibrated without PM2.5:**
  - on the main sample's DOSE-matched region-years (1998–2012), regress ln(lights per person) on ln(DOSE output per
    person) with year FE (pooled OLS);
  - the knots are the predicted ln(lights per person) at 1,000, 4,000 and 12,000 USD, at the
    mean year;
  - if that regression's within-year R² is below 0.5, the knots are instead the sample
    quartiles of ln(lights per person) (region-years, each region weighted equally);
  - G2 reports which rule applied and the knot values.
- **Institutions:** terciles of the baseline `v2x_polyarchy` across this sample's countries,
  each country counted once. The cutoffs are recomputed for the larger set of countries.
- **Model:** as idea B, without weather:
  $$\ln \text{PM}_{rt} = \sum_{g}\sum_{s} \beta_{sg}\, \text{spline}_s(\ln L_{rt}/N_{rt})\, \mathbb{1}[g(r)=g] + \alpha_r + \delta_{g t} + \varepsilon_{rt}$$
- **Clusters:** country. **Weights:** none.
- **Pooled path:** one spline, with region + year FE.

### Variants (reported, not selected)

1. + country-specific linear trends.
2. Country × year FE instead of tercile × year FE. Close to what has already been seen (§6).
3. Rule of law (`v2x_rule` terciles).
4. Without dusty regions (desert share > 0.5).
5. Population-weighted PM2.5 (GHS-POP weights within the region), and population regression
   weights.
6. VIIRS years only (2014–2022), with knots and terciles set by the same rules on those years
   (changed 2026-10-30; first "DMSP years only, 1998–2013").
7. ln(lights per area) instead of per person.
8. + GLASS air temperature (2000–2012 in the main sample).
9. **H, low-haze regions:** regions in the lower two terciles of baseline (1998–2000) mean
   PM2.5, with cutoffs across regions. This is the haze guard of §4.

**Information, not a rule:** the DOSE overlap. On DOSE-matched regions, the segment slopes
estimated with lights are put next to those estimated with DOSE output (the idea-B main
specification). The gap shows how far the lights path stands in for an output path.

## 4. ◇ Rules, fixed before any PM2.5 regression

- **◇ G2, coverage (loads no PM2.5).**
  - Each tercile needs at least 15 countries and 150 regions.
  - A tercile × segment cell is **supported** if at least 50 regions have income in it.
  - The comparisons are the adjacent supported terciles within each segment, as in idea B's
    change log.
  - If a tercile misses the country or region count, only the pooled path is estimated.
- **◇ G2b, purpose (loads no PM2.5).** Lights are used to fill the overlap gap. If the high
  tercile still has fewer than 50 regions in the lowest segment, the conclusion says so: lights
  did not solve the overlap problem. The comparisons are then still made on supported cells.
- **◇ Moderation.** "Institutions change the path" is claimed only if all three hold:
  - the joint Wald test of the supported comparisons rejects at 5 % with country clusters, in
    the main specification;
  - each comparison that differs at 5 % keeps its sign in variants 1 and 2;
  - each such comparison also keeps its sign in variant H (haze guard).

  Otherwise: "no robust difference between institution groups".
- **◇ Turning point.** A segment's slope counts as a downturn only if it is negative at 5 % in
  the main specification **and** in variant H. A negative slope only in the main specification
  could be haze: aerosols dim measured lights most where PM2.5 is highest, which pushes slopes
  down.
- **Haze bias, stated in advance.** Measured ln lights = true ln lights − k·aerosol with k > 0.
  So when PM2.5 is high, measured lights are too low, and the bias in a slope of ln PM2.5 on
  ln lights is negative. It is larger in groups and periods with more PM2.5 variation. A
  positive slope is therefore conservative, and a negative one is suspect.
- **Reading.** Every coefficient is descriptive.

## 5. Outcomes

### Variants 3–8 (2026-10-30; `08_pollution_lights.ipynb`)

Variants are reported, not selected (§3), so the rule outcomes of the main specification
stand. Construction follows the change log entry for variants 3–8. Figures:
`output/figures/pollution/pm25_lights_paths` (pooled path, paths by tercile, variants 1–2) and
`pm25_lights_variants` (variants 3–8).

- **Knots: two variants fall back to quartiles.** The DOSE calibration's within-year R² is
  0.386 in the VIIRS years (variant 6) and 0.288 for lights per area (variant 7). So, by §3,
  their segments are the quartiles of their own measure, not the USD ranges of the main
  specification. Their "4–12k" and "> 12k" cells are the third and fourth quartiles.
- **The comparison that carries the main result** (above 12,000 USD, high − middle; main
  +0.133, p = 0.008) and **the two downturn cells**, by variant:

  | Variant | N | Wald p (q) | high − middle > 12k | low 4–12k | middle > 12k |
  |---|---|---|---|---|---|
  | main | 41,274 | 0.032 (8) | +0.133 (p 0.008) | −0.109 (0.018) | −0.086 (0.009) |
  | 3 rule of law | 41,274 | 0.009 (8) | +0.159 (0.008) | −0.108 (0.006) | −0.133 (0.007) |
  | 4 desert share ≤ 0.5 | 37,243 | 0.010 (8) | +0.144 (0.004) | −0.128 (0.016) | −0.096 (0.004) |
  | 5 population-weighted | 41,274 | < 0.001 (8) | +0.403 (0.073) | −0.275 (0.011) | −0.059 (0.021) |
  | 6 VIIRS 2014–2022 (quartile knots) | 25,343 | 0.131 (7) | +0.024 (0.446) | +0.039 (0.193) | −0.019 (0.456) |
  | 7 lights per area (quartile knots) | 41,274 | 0.013 (8) | +0.174 (0.028) | +0.009 (0.783) | −0.109 (0.061) |
  | 8 + GLASS temperature, 2000–2012 | 35,689 | 0.053 (8) | +0.125 (0.002) | −0.052 (0.093) | −0.044 (0.019) |

- **The high − middle difference above 12,000 USD** is positive in every variant.
  - It is significant at 5 % in variants 3, 4, 7 and 8, as well as in 1, 2 and H.
  - It is not significant in 5 (p = 0.073).
  - It is gone in 6, the VIIRS years: +0.024 (p = 0.45).
- **The two downturns** hold in variants 3, 4 and 5. In 8, middle above 12k holds
  (p = 0.019) and low at 4–12k weakens (p = 0.093). Both disappear in 6. In 7, low at 4–12k
  is +0.009.
- **3, rule of law:** the joint test rejects (p = 0.009), with 3 comparisons at 5 %: high −
  middle at 1–4k (+0.119), at 4–12k (+0.181) and above 12k (+0.159).
- **5, population-weighted PM2.5 and population weights:** the joint test rejects
  (p < 0.001). The rejection comes from the lower segments: high − middle +0.33 below 1,000
  USD (p < 0.001) and +0.44 at 1–4k (p = 0.014).
  - The weights concentrate on a few countries: IND has 45 % of the high tercile's weight,
    and CHN 56 % of the low tercile's. As in idea B, this compares India with the rest more
    than institution groups.
  - Information: population-weighted PM2.5 alone gives Wald p = 0.043 (above 12k +0.135,
    p = 0.009); population weights alone give p < 0.001 (above 12k +0.363, p = 0.10).
- **6, VIIRS years:**
  - The sample is 25,343 region-years in 2,827 regions and 167 countries.
  - The terciles are unchanged: the same countries are eligible.
  - The high tercile has 47 regions in the lowest quartile, so that cell is unsupported and 7
    comparisons remain.
  - No comparison except high − middle in the second quartile (+0.072, p = 0.026) differs
    at 5 %, and the joint test does not reject.
  - This is the only check on another sensor and period, and it does not reproduce the main
    result. Its knots, sensor and years all differ, so it does not say which of them matters.
- **7, lights per area:** the joint test rejects (p = 0.013), with 4 comparisons at 5 %.
  Above the top quartile, middle − low is −0.200 (p = 0.017) and high − middle +0.174.
- **8, + GLASS air temperature (2000–2012):** the sample is 35,689 region-years in 2,804
  regions. The joint test misses 5 % narrowly (p = 0.053), but the top-segment difference
  holds (+0.125, p = 0.002).
- **Reading.** Over the DMSP years, the one difference is stable across groupings, samples and
  controls. High-democracy regions' PM2.5 does not fall with lights above about 12,000 USD,
  while the middle tercile's does. Country trends (variant 1) take away the joint test but not
  that comparison. The VIIRS years do not show it. The downturns rest on the main sample's
  knots and the DMSP years.

### Main specification, pooled path, variants 1, 2 and H (2026-10-30; `08_pollution_lights.ipynb`)

This is the first run that loads PM2.5. Every row of the G2 sample has PM2.5, so N = 41,274
region-years (2,818 regions, 167 countries; pyfixest drops 13 singletons). All 12 cells are
supported, so the 8 comparisons are those of G2.

- **Slopes** (elasticity of PM2.5 with respect to lights per person; country clusters, 167):

  | Tercile | < 1k USD | 1–4k | 4–12k | > 12k |
  |---|---|---|---|---|
  | low | 0.017 (0.010) | −0.030 (0.041) | **−0.109** (0.046) | **−0.064** (0.029) |
  | middle | −0.004 (0.014) | −0.009 (0.026) | −0.045 (0.048) | **−0.086** (0.033) |
  | high | 0.055 (0.032) | 0.150 (0.082) | 0.011 (0.034) | 0.046 (0.037) |

  Bold: p < 0.05. Every high-tercile slope is positive, and none is significant.
- **◇ Moderation: by the rule, "institutions change the path".**
  - The joint Wald test of the 8 comparisons rejects: F = 2.17 (8, 166), p = 0.032.
  - One comparison differs at 5 %: above 12,000 USD, high − middle = +0.133 (0.049),
    p = 0.008. It keeps its sign in variant 1 (+0.087, p = 0.003), variant 2 (+0.042,
    p = 0.013) and variant H (+0.116, p = 0.018).
  - **What it says:** above 12,000 USD, PM2.5 falls with lights in the middle tercile and does
    not fall in the high tercile. That is the opposite of "democracies clean up sooner".
  - **How much it rests on (information, not a rule):** the claim is one comparison of eight.
    A Bonferroni bar (0.05 / 8 = 0.006) would not pass it. With country trends (variant 1)
    the joint test does not reject (p = 0.078); the rule does not require it to. The other
    comparisons that are large in the main specification (1–4k, high − middle +0.159,
    p = 0.065) shrink to +0.01 and +0.03 in variants 1 and 2.
  - Region clusters (information): F = 11.5, p < 0.001.
- **◇ Turning point: two cells, both in the upper segments.** These slopes are negative at 5 %
  in the main specification and in variant H:
  - low tercile at 4,000–12,000 USD: −0.109 (p = 0.018), in H −0.201 (p < 0.001);
  - middle tercile above 12,000 USD: −0.086 (p = 0.009), in H −0.079 (p = 0.006).

  The low tercile above 12,000 USD (−0.064, p = 0.027) is not supported in H (33 regions), so
  it does not count. **Information:** with country trends or country × year FE the two
  downturns shrink to −0.04 and −0.03 (low; p = 0.22 and 0.13) and −0.005 and −0.011 (middle;
  p = 0.78 and 0.07). The rule does not require them. The low tercile's top segments are the
  oil producers of G2 (IRN, LBY, SAU, KAZ, OMN) and SRB.
- **Pooled path** (region + year FE, country clusters): 0.019 (0.010; p = 0.056), 0.019
  (0.020), −0.042 (0.031), −0.021 (0.028). No segment differs from zero at 5 %, so the
  pooled path is flat, with no turning point. Variant H gives the same picture.
  - It is much flatter than idea B's output path (0.205 and 0.174 below 4,000 USD).
  - Units do not explain it. With the calibration slope of 0.73, the elasticity in output
    terms would be 0.73 times the lights one, which is flatter still. Possible reasons,
    stated as information: year-to-year noise in lights within a region attenuates the slopes,
    and the sample differs (all countries, 1998–2012, no weather controls).
- **Variant 1** (+ country trends): Wald p = 0.078. One slope is significant: high above
  12,000 USD, +0.082 (p < 0.001).
- **Variant 2** (country × year FE): Wald p = 0.023. One slope is significant: high above
  12,000 USD, +0.030 (p = 0.049). Middle above 12,000 USD is −0.011 (p = 0.070).
- **Variant H** (the haze guard): baseline PM2.5 cutoffs are 11.9 and 18.5 µg/m³. That leaves
  27,389 region-years, 1,846 regions and 133 countries. Two cells lose support: high below
  1,000 USD (49 regions) and low above 12,000 USD (33). The joint test of the 6 remaining
  comparisons rejects: F = 4.85 (6, 132), p < 0.001.
- **Information: the DOSE overlap.** On the 18,567 DOSE-matched region-years (1,484
  regions), the lights slopes are put next to idea B's main specification:
  - they agree in some cells (low below 1,000 USD: 0.104 against 0.118; middle at 1–4k: 0.057
    against 0.063; middle above 12k: −0.044 against −0.045);
  - they disagree in others (low at 1–4k: −0.025 against 0.058; high at 4–12k: 0.018 against
    0.118);
  - so lights do not reproduce the output path cell by cell.

### G2 and G2b, rerun on the DMSP years with the flare mask (2026-10-30; `08_pollution_lights.ipynb`; no PM2.5 loaded)

This is the run that counts. It follows the 2026-10-30 change log.

- **Sample:** 41,287 region-years, 2,818 regions, 167 countries, 1998–2012.
  - **Dropped region-years:**
    - 6,540 below 5 cells (436 regions);
    - 1,620 without population (108 regions);
    - 1,283 with zero lights (238 regions), mostly UGA 322, BDI 93, BTN 80, AFG 80, CAF 77,
      LBR 70.
  - **No V-Dem baseline:** 44 countries and territories (152 regions).
- **Flare mask:** it removes 3.7 % of all lights. The highest country shares are BHR 0.53,
  GAB 0.46, KWT 0.33, NGA 0.27, IRQ 0.27, COG 0.25.
- **Knots: the DOSE calibration applies.** Within-year R² 0.533 (≥ 0.5); slope 0.73;
  18,567 region-years in 1,484 regions; mean year 2005. The knots are −4.022, −3.013 and
  −2.213 in ln lights per person, which is 1,000, 4,000 and 12,000 USD by construction.
- **◇ G2 passes.**
  - Coverage by tercile: low 56 countries and 817 regions, middle 55 and 1,213, high 56 and
    788.
  - All 12 cells are supported, so all 8 adjacent comparisons are made.

  | Tercile | seg0 | seg1 | seg2 | seg3 |
  |---|---|---|---|---|
  | low | 515 | 228 | 198 | 146 |
  | middle | 649 | 508 | 450 | 214 |
  | high | 89 | 192 | 343 | 460 |

- **◇ G2b passes.** The high tercile has 89 regions in the lowest segment, against 0 with DOSE
  output in idea B (G1).
  - 54 of the 89 are in three countries: IND 26, MNG 14, SEN 14.
  - Then PAN 7, BOL 6, NAM 4, and 10 more countries with 1–3 regions each.
  - The cell exists, but a few countries carry it.
- **Information: the low tercile's top segment** has 21 countries, led by IRN 23, SRB 21,
  LBY 21, SAU 13, KAZ 11 and OMN 11.

### First G2 run, 1998–2022 without a flare mask (2026-10-30; superseded, kept as the record)


- **Sample:** 69,455 region-years, 2,827 regions, 167 countries, 1998–2022. The panel has
  3,429 regions in 235 GADM countries.
  - **Dropped region-years:**
    - 10,900 below 5 cells (436 regions);
    - 2,700 without population (108 regions);
    - 1,470 with zero lights (258 regions), mostly UGA 324, BDI 93, LBR 88, SLB 86, BTN 82,
      AFG 82, CAF 77, TCD 69.
  - **No V-Dem baseline:** 44 countries and territories (152 regions), such as SSD, XKO,
    PRI, FJI and the disputed Z0x areas.
- **Knots: the quartile fallback applies.** The DOSE calibration has a within-year R² of
  0.481, below 0.5. Its other figures: slope 0.70; 26,904 region-years in 1,527 regions;
  mean year 2008.
  - The knots are the quartiles of ln lights per person: −4.826, −3.218 and −2.244.
  - **Information only:** by the calibration they correspond to about 340, 3,370 and 13,520
    USD.
- **◇ G2 passes.**
  - Each tercile has at least 15 countries and 150 regions: low 56 countries and 822
    regions, middle 55 and 1,217, high 56 and 788.
  - All 12 tercile × segment cells are supported, so all 8 adjacent comparisons are made.

  | Tercile | seg0 | seg1 | seg2 | seg3 |
  |---|---|---|---|---|
  | low | 446 | 541 | 408 | 291 |
  | middle | 547 | 913 | 723 | 485 |
  | high | 60 | 228 | 525 | 589 |

- **◇ G2b passes, narrowly.** The high tercile has 60 regions in the lowest segment (≥ 50).
  - 42 of the 60 are in three countries: IND 16, SEN 14, MNG 12.
  - So lights add poor democracies, but few countries carry that cell.
- **Information: the low tercile's top segment** spans 46 countries and is led by oil
  producers (IRN 29, LBY 21, IRQ 14, KAZ 14, SAU 13, OMN 11) besides SRB 23 and TUN 15. Gas
  flares are lit but are not economic activity (`docs/data/ntl_harm/README.md`).
- **Information: the sensor splice does not act as a common shift.** These are median log
  changes of region lights, by 2012 brightness quintile:

  | 2012 brightness | 2013 vs mean(2012, 2014) | 2012 → 2014 | 2016 → 2017 |
  |---|---|---|---|
  | dimmest | +2.27 | −0.52 | +0.65 |
  | 2 | +1.21 | −0.38 | +0.48 |
  | 3 | +0.57 | −0.31 | +0.41 |
  | 4 | +0.30 | −0.26 | +0.31 |
  | brightest | +0.13 | −0.25 | +0.26 |

  Global lights (Σ DN) are 368 M in 2012, 505 M in 2013, 297 M in 2014 and 381 M in 2017.
  Year FE (§2) cannot absorb jumps that differ by brightness. Variant 6 (1998–2013) also
  contains the 2013 spike.

## 6. Already seen (disclosure)

- **Idea B (2026-10-29):** the same outcome and terciles on DOSE regions, with output per
  person. The path rises below about 4,000 USD and flattens. There is no robust difference by
  institutions where the terciles overlap. With country × year FE, no supported slope is
  positive.
- **Retired project, conclusion §2.3 M1:** at 10 km cells, PM2.5 on log(1 + lights) with pixel
  + country × biome × year FE: γ = −0.098 (0.067). That is lights within country-years, close to
  variant 2 here, so variant 2 is not new evidence.
- **Retired project, P5:** ACAG PM2.5 is almost entirely regional at 10 km (within-FE r² with
  the neighbours' mean 0.97). This is why the unit is the region.
- **`04_channels.ipynb`:** the reduced form of PM2.5 on mine counts.
- **Not seen:** PM2.5 on lights across countries within a tercile (the main specification), at
  ADM1, in any country set.

## Change log

- **2026-10-30, confirmation.** The author confirmed the four decisions of §0 as drafted, before
  G2 and before any regression with PM2.5. No rule changed.
- **2026-10-30, construction (before G2; loads no PM2.5).** Lights and population are summed per
  region from the 1 km tiles (land cells only), over the 1 km cells of the region's 10 km panel
  cells (`src/analysis/pollution_lights.py`), not from the panel's 10 km means. The panel
  averages lights over a coastal cell's land part, so a sum built from it would scale them by a
  full cell. Its year clip (1992–2022) also drops the 2025 GHS-POP epoch, which 2021–22 need for
  the interpolation. The region-size rule (≥ 5 cells with PM2.5 and lights) applies per
  region-year. The countries for the terciles are those with V-Dem's baseline index and at least
  one region-year that passes the size rule with positive lights and population.
- **2026-10-30, after G2 and before any PM2.5 (author's decision; loads no PM2.5).** G2 showed
  that the sensor splice is not a common shift (§5): in 2013 the dimmest fifth of regions gains
  +2.27 log points against the mean of 2012 and 2014, and the brightest fifth +0.13. There are
  further brightness-dependent steps into 2014 and 2017. Year FE cannot absorb that. It also
  showed that the low tercile's brightest segment is led by oil producers.
  - **Years:** the main sample becomes the DMSP years 1998–2012. Variant 6 becomes the VIIRS
    years 2014–2022, with knots and terciles set by the same rules on those years. 2013 is in
    neither.
  - **Flares:** a static flare mask. 1 km cells with `flare_band` > 0 in any year 2012–2021
    (within 5 km of an upstream flare) are left out of the lights and population sums in all
    years. The survey starts in 2012, so the mask is static. Refinery and industrial flares are
    not in the survey.
  - **Knot calibration:** on the main sample's DOSE-matched region-years (1998–2012), not
    1998–2018.
  - **Terciles:** across the countries of the 1998–2012 sample.
  - **G2 and G2b are rerun** on the new sample. The first run stays in §5 as a record.

- **2026-10-30, after the main specification and before variants 3–8 (construction only; the
  author said go).** §3 leaves three details of the variants open. They are fixed here, before
  any of these variants runs:
  - **Variant 5:** a 10 km cell's population weight is its GHS-POP population summed over its
    land 1 km cells, flare cells included (the PM2.5 mean covers them too). It is interpolated
    **linearly** between epochs, so cells with zero population stay valid. The regression
    weight is the region population of the main specification. Both run together, as written;
    each one alone is reported as information.
  - **Variant 6:** G2's counts are reported on the VIIRS sample as information. The knots and
    terciles follow §3 on 2014–2022.
  - **Variant 7:** the knots come from the same rule as in §3, applied to ln lights per area:
    the DOSE calibration, or the quartile fallback if its within-year R² is below 0.5.
  - **Variant 8:** GLASS air temperature is the region mean over its 10 km cells. The sample
    is 2000–2012, the region-years with temperature.

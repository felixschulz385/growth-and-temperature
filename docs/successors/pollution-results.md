# Pollution Along the Growth Path, by Institutions: Results

**Status: written 2026-10-30; step 1 updated 2026-11-04** after its rerun on corrected inputs
(a PREPARE bug at the antimeridian; `pollution-lights-prespec.md`, change log). This covers both runs of the successor project on the branch
`successor/pollution-institutions`:

- **idea B**, with DOSE regional output as the growth measure, 2026-10-29;
- **step 1**, with night lights per person as the growth measure, 2026-10-30.

Both are **descriptive**: neither growth nor institutions is instrumented. The rules, change logs
and full tables are in the two pre-specs:
[`pollution-institutions-prespec.md`](pollution-institutions-prespec.md) (idea B) and
[`pollution-lights-prespec.md`](pollution-lights-prespec.md) (step 1). This page summarises them
and adds nothing new.

## Summary

1. **PM2.5 rises with output only at low incomes, and only across countries.**
   - With DOSE output, the pooled path rises below about 4,000 USD per person (elasticities
     0.21 and 0.17) and flattens above.
   - With country trends or country × year FE, no positive slope survives. So the rise comes
     from countries whose output and PM2.5 move together, not from faster-growing regions
     within a country.
   - With lights, the pooled path is flat everywhere: no segment differs from zero at 5 %.
2. **With DOSE output, institutions make no difference where they can be compared.**
   - Strong democracies are almost never poor in DOSE, so only the 4,000–12,000 USD range has
     all three democracy terciles.
   - There, the joint test of the tercile differences gives p = 0.53.
3. **With lights, the pre-specified rule finds two differences, and both are fragile.**
   - Above about 12,000 USD, PM2.5 falls as lights rise in the middle democracy tercile
     (−0.079), but not in the high tercile (+0.043). The difference is +0.121 (p = 0.018), and
     the joint test of all 8 comparisons gives p = 0.043.
   - It keeps its sign in every variant over the DMSP years 1998–2012.
   - It is absent in the VIIRS years 2014–2022 (+0.017, p = 0.59).
   - The second, high − middle at 1,000–4,000 USD (+0.139, p = 0.033), comes from one country:
     the Dominican Republic, which moved into the high tercile in the rerun. With it in the
     middle tercile, p = 0.068.
   - Neither passes a Bonferroni bar over the eight comparisons.
   - Their direction is the opposite of "democracies clean up sooner".
4. **No turning point holds up.**
   - With DOSE output, no slope is negative at 5 %.
   - With lights, two cells pass the pre-specified rule, which requires a negative slope in
     the main specification and in low-haze regions. They are the low tercile at 4,000–12,000
     USD and the middle tercile above 12,000 USD.
   - Both vanish with country trends, with country × year FE and in the VIIRS years.
5. **Population weighting compares countries, not institutions.** In both runs, the weights put
   most of a tercile on one country: China, India or the United States. So the variants that
   weight by population mostly compare those countries.

## 1. Question and design

How does fine particulate matter (PM2.5) change as a region grows richer, and does the path
differ between countries with stronger and weaker democratic institutions?

- **Unit:** GADM 4.1 first-level regions (ADM1), region × year.
- **Outcome:** ln PM2.5, the region's mean over its 10 km cells (ACAG V6GL02.04).
- **Path:** a piecewise-linear spline in ln growth with knots at 1,000, 4,000 and 12,000 USD
  per person, interacted with the democracy tercile.
- **Institutions:** terciles of each country's baseline (1998–2000) V-Dem electoral democracy
  index (`v2x_polyarchy`), each country counted once. Rule of law (`v2x_rule`) is a variant.
- **Fixed effects:** region FE and tercile × year FE. Errors are clustered by country.
- **Supported cells:** a tercile × segment cell is compared only if at least 50 regions have
  income in it. The comparisons are between adjacent terciles within a segment.
- **Moderation rule:** "institutions change the path" is claimed only if the joint Wald test
  rejects at 5 %, and each comparison that differs at 5 % keeps its sign in the check
  variants. The check variants are country trends and country × year FE, plus low-haze regions
  for lights.

## 2. The two runs

| | Idea B: DOSE output | Step 1: night lights per person |
|---|---|---|
| Growth | DOSE `grp_pc_lcu2015_usd` (2015 USD per person) | Σ `ntl_harm` ÷ GHS-POP population, from the 1 km tiles; gas-flare cells masked |
| Knots | 1,000 / 4,000 / 12,000 USD | ln lights at those incomes, by a calibration on DOSE regions (within-year R² 0.54) |
| Years | 1998–2018 | 1998–2012, the DMSP years. The DMSP–VIIRS splice moves dim regions far more than bright ones, so 2014–2022 is a variant |
| Controls | DOSE precipitation and temperature | none (air temperature is a variant) |
| Sample | 27,512 region-years, 1,564 regions, 77 countries | 41,557 region-years, 2,840 regions, 169 countries |
| Overlap | fails: no high-tercile region below 1,000 USD; 5 comparisons | all 12 cells supported; 8 comparisons |
| Notebook | `07_pollution.ipynb` | `08_pollution_lights.ipynb` |

**Lights fill the overlap gap only partly.** The high tercile has 103 regions below 1,000 USD,
against none with DOSE output. But 67 of them are in India, Mongolia, Senegal and the Dominican
Republic, so a difference in that cell is mainly a comparison with those four countries.

## 3. Results with DOSE output (idea B)

Elasticities of PM2.5 with respect to output per person, country clusters. Cells in brackets
are unsupported.

| Segment (2015 USD) | low | middle | high |
|---|---|---|---|
| < 1,000 | 0.118 (0.043) | 0.208 (0.170) | — |
| 1,000–4,000 | 0.058 (0.023) | 0.063 (0.055) | [0.710, 18 regions] |
| 4,000–12,000 | −0.009 (0.059) | −0.051 (0.072) | 0.118 (0.078) |
| > 12,000 | [−0.022, 43 regions] | −0.045 (0.054) | 0.034 (0.050) |

- **Moderation:** no robust difference. The joint Wald test gives p = 0.53, and no single
  comparison differs at 5 %. With region clusters it would reject (p < 0.001), so any
  difference rests on treating regions as independent.
- **Turning point:** none.
- **Variants:**
  - rule of law (p = 0.35), no dusty regions (p = 0.61) and PM2.5 in levels (p = 0.77) agree;
  - only population weights reject, and those weights are China, India and the United States;
  - the 10 countries that change tercile are too few to read.

## 4. Results with night lights (step 1)

Elasticities of PM2.5 with respect to lights per person, country clusters. All cells are
supported. Bold: p < 0.05.

| Segment (≈ 2015 USD) | low | middle | high |
|---|---|---|---|
| < 1,000 | 0.017 (0.010) | −0.005 (0.015) | 0.052 (0.026) |
| 1,000–4,000 | −0.030 (0.042) | −0.011 (0.027) | **0.128** (0.059) |
| 4,000–12,000 | **−0.110** (0.046) | −0.042 (0.047) | 0.004 (0.032) |
| > 12,000 | **−0.064** (0.028) | **−0.079** (0.034) | 0.043 (0.038) |

- **Moderation, by the rule: "institutions change the path".** Two high − middle comparisons
  differ at 5 % and keep their sign in variants 1, 2 and H.
  - **Above 12,000 USD.** It keeps its sign in every variant and is significant at 5 % in all
    but two:

    | Variant | Joint test p | High − middle above 12k |
    |---|---|---|
    | main | 0.043 | +0.121 (p 0.018) |
    | 1 + country trends | 0.120 | +0.077 (0.007) |
    | 2 country × year FE | 0.013 | +0.041 (0.015) |
    | H low-haze regions | < 0.001 | +0.102 (0.041) |
    | 3 rule of law | 0.008 | +0.159 (0.007) |
    | 4 no dusty regions | 0.016 | +0.133 (0.009) |
    | 5 population-weighted | < 0.001 | +0.402 (0.072) |
    | 6 VIIRS years 2014–2022 | 0.100 | +0.017 (0.594) |
    | 7 lights per area | 0.016 | +0.164 (0.039) |
    | 8 + air temperature | 0.079 | +0.114 (0.005) |

  - **1,000–4,000 USD:** +0.139 (p = 0.033); in variants 1 and 2 only +0.03 and +0.02. The
    Dominican Republic has 27 of the high tercile's 219 regions in this segment. It moved from
    the middle tercile to the high one in the rerun, because Fiji and Vanuatu entered the
    tercile ranking. With it in the middle tercile (information), the difference is +0.155
    (p = 0.068), as in the first run, and the one above 12k is unchanged (+0.126, p = 0.013).
  - **First run (2026-10-30, superseded):** the same picture with one comparison, above 12k
    +0.133 (p = 0.008), joint test p = 0.032. Its inputs lacked part of New Zealand, Fiji and
    Vanuatu.

  In variants 6 and 7 the DOSE calibration's R² is below 0.5 (0.39 and 0.29), so their knots
  are quartiles. Their top segment is the top quartile, not "above 12,000 USD".
- **Turning point, by the rule:** two cells, the low tercile at 4,000–12,000 USD and the middle
  tercile above 12,000 USD. Negative slopes are the suspect direction here: aerosols dim
  measured lights most where PM2.5 is high, which biases slopes down. That is why the rule also
  requires them in low-haze regions. Both cells shrink to near zero with country trends or
  country × year FE, and disappear in the VIIRS years.
- **Pooled path:** 0.018, 0.020, −0.042 and −0.018 by segment; none differs from zero at 5 %.
- **Lights and output paths differ.** On the DOSE regions, the lights slopes match the output
  slopes in some cells, such as the low tercile below 1,000 USD (0.104 against 0.118). In other
  cells they differ, such as the high tercile at 4,000–12,000 USD (0.015 against 0.118). Lights
  stand in for output only partly.

## 5. What the results support

- **Supported:**
  - at low incomes, PM2.5 and output rise together across countries, not across regions within
    a country;
  - above about 4,000 USD the path is flat;
  - no data set here shows strong democracies with a lower or earlier-turning PM2.5 path.
- **Recorded, but not a finding to build on:** by the pre-specified rule, institutions change
  the lights path. That rests on one rich-segment comparison: the middle tercile's PM2.5 falls
  and the high tercile's does not. It appears only in the DMSP years and not in the VIIRS
  years. The rule's second comparison (1,000–4,000 USD) rests on one country's tercile.
- **Not supported:** an environmental Kuznets turning point in any group that survives
  within-country comparisons.

## 6. Limitations

- **Descriptive.** The paths condition on region FE. They are not effects of growth or of
  institutions, and countries differ in much besides institutions.
- **PM2.5 is regional.** ACAG PM2.5 is smooth at 10 km, and it travels across borders. A region's
  PM2.5 partly reflects its neighbours' output, which may be why within-country slopes vanish.
  NO₂ stays closer to its sources (next steps).
- **Dust.** ACAG V6GL02.04 includes mineral dust. Dropping desert regions changes little, but
  there is no dust-free series.
- **Lights measure output with error.**
  - The DMSP–VIIRS splice is brightness-dependent.
  - Year-to-year noise attenuates slopes.
  - Haze dims lights where PM2.5 is high.
  - Gas flares are masked, but refinery and industrial flares are not in the survey.
- **Few countries carry some cells.** India, Mongolia and Senegal carry the poor
  high-democracy cell. Oil producers (Iran, Libya, Saudi Arabia, Kazakhstan, Oman) and Serbia
  lead the rich low-democracy cell.
- **Fixed terciles.** Institutions are fixed at their 1998–2000 baseline. Too few countries
  change tercile to study changes.

## 7. Next steps (not scheduled)

From [`pollution-data-options.md`](pollution-data-options.md) §4:

- **NO₂ as a second outcome** (Anenberg et al., 1 km). The `no2` module is prepared, but not yet
  in the 10 km panel. It would test whether within-country slopes vanish because PM2.5 is
  transported. It needs its own pre-spec.
- **MERRA-2 dust share.** Deferred: it needs an Earthdata login.
- **GHS-POP** is built and already used in step 1, for lights per person and population
  weights.

## 8. Reproduction

- **Data modules:**
  - `vdem` (V-Dem v16);
  - `ghs_pop` (GHS-POP R2023A);
  - `no2` (Anenberg NO₂, prepared, unused);
  - `dose`, `acag`, `ntl_harm`, `eog` and `gadm` from earlier work.
  - Pages are in [`docs/data/`](../data/README.md). FETCH for `no2` and `ghs_pop` runs on the
    transfer node (`orchestration/scripts/pollution-fetch.sh`).
- **Code:**
  - `src/analysis/pollution.py`: the idea B region panel, the tercile × spline fits and the
    figures;
  - `src/analysis/pollution_lights.py`: the lights region panel from the 1 km tiles, the knot
    calibration, G2, region PM2.5 and the variants.
- **Notebooks:** `output/notebooks/07_pollution.ipynb` (idea B) and `08_pollution_lights.ipynb`
  (step 1). Each runs top to bottom in the `gnt` environment.
- **Caches** (`data_nobackup/`, git-ignored, rebuilt on first use):
  - `_pollution_region_panel_10km.parquet`;
  - `_pollution_lights_cell_sums_10km.parquet`;
  - `_pollution_lights_region_panel_10km.parquet`;
  - `_pollution_lights_region_pm25_10km.parquet`.
- **Figures** (`output/figures/pollution/`, PDF and PNG):
  - idea B: `pm25_paths`, `pm25_variants`, `pm25_movers`;
  - step 1: `pm25_lights_paths`, `pm25_lights_variants`.

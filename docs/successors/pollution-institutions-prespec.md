# Pollution Along the Growth Path, by Institutions: Pre-Specification

**Status: confirmed by the author 2026-10-29, before G1 or any regression with PM2.5 ran. Run
2026-10-29 (§5); summary with the lights follow-up in [`pollution-results.md`](pollution-results.md).**
This is idea B of [`ideas.md`](ideas.md), now a successor project on the branch
`successor/pollution-institutions`. Its critique there applies. **It is descriptive and makes no
causal claim:** neither growth nor institutions is instrumented.

## 1. Question

How does fine particulate matter (PM2.5) change as a region's output per person rises, and does
that path differ between countries with stronger and weaker democratic institutions?

## 2. Data

| | Source | Module |
|---|---|---|
| Unit | DOSE regions matched to GADM 4.1 (1,596 regions in 81 countries) | `dose` |
| Outcome | ACAG V6GL02.04 PM2.5: mean over the region's cells in the assembled 10 km panel, each cell the mean of its 1 km pixels (area-weighted on the equal-area grid) | `acag` → assembly → `src/analysis/pollution.py` |
| Growth | DOSE `grp_pc_lcu2015_usd`: real output per person at 2015 prices, converted at the 2015 exchange rate. Changes within a country are real growth, and levels compare across countries. | `dose` |
| Institutions | V-Dem v16 electoral democracy index `v2x_polyarchy` (main); rule of law `v2x_rule` (second) | `vdem` |
| Weather | DOSE precipitation `P_a` and temperature `T_a` (ERA5). Temperature is allowed here, because the outcome is PM2.5, not temperature. | `dose` |
| Dust | each region's share of 10 km cells in RESOLVE biome 13 (deserts and xeric shrublands) | `ecoregions` → assembly |

- **Years:** 1998–2018. ACAG starts in 1998. DOSE's `P_a` and `T_a` stop in 2018, the same
  constraint as the fork (dose-prespec §3b).
- **Country match:** V-Dem's `country_text_id` to GADM `GID_0`. The Bahamas has no V-Dem series
  and drops.
- **Not available:** ACAG offers no dust-free PM2.5 for V6GL02.04, and the pipeline has no
  population grid. So the outcome is area-weighted, not population-weighted.
- **Why the 10 km panel and not the 1 km grid.** Checked 2026-10-29 against a 1 km build, looking
  only at PM2.5 itself. Region means agree: correlation 0.999 in levels and 0.994 in yearly log
  changes, median absolute difference 0.3 %. Differences are larger only in small regions,
  where border cells go whole to one region: 117 regions differ by more than 5 % on average. The
  10 km route keeps 1,573 DOSE regions against 1,576 at 1 km.

## 3. Specification

- **Outcome:** ln PM2.5. The level (µg/m³) is a robustness check.
- **Path:** a piecewise-linear spline in ln output per person, with knots at 2015 USD 1,000, 4,000
  and 12,000. These are roughly the World Bank's income thresholds, fixed before seeing any data.
  Each segment's slope is the elasticity of PM2.5 with respect to output in that income range.
- **Institutions:** each country's mean `v2x_polyarchy` over 1998–2000, fixed over the panel.
  - **Groups (author, 2026-10-29): terciles** of that baseline score across the sample's
    countries, each country counted once: low, middle, high. Terciles rather than the 0.5
    cutoff, because DOSE leans towards richer countries and a fixed cutoff could leave one group
    nearly empty.
  - **The spline is interacted with the tercile,** so each tercile has its own path.
  - **Fixing institutions at baseline** keeps the groups from responding to growth or pollution.
- **Model:**

  $$\ln \text{PM}_{rt} = \sum_{g}\sum_{s} \beta_{sg}\, \text{spline}_s(\ln y_{rt})\, \mathbb{1}[g(r)=g] + \gamma' w_{rt} + \alpha_r + \delta_{g t} + \varepsilon_{rt}$$

  - $w_{rt}$: precipitation and temperature;
  - $\alpha_r$: region FE;
  - $\delta_{gt}$: year FE by tercile, so the terciles may have different global trends.
- **Clusters:** country, because institutions are assigned at country level (about 79 clusters).
  Region clusters are reported for information.
- **Weights:** none. Each region counts once. Population weights from DOSE `pop` are a robustness
  check.

### Variants (reported, not selected)

1. **+ country-specific linear trends.**
2. **Country × year FE instead of group × year FE.** This compares regions within the same
   country-year, so national policy and national shocks drop out. The paths are then about
   regions growing faster than their country.
3. **Rule of law** (`v2x_rule`, baseline 1998–2000, terciles of countries) instead of electoral
   democracy.
4. **Without dusty regions:** drop regions with a desert share above 0.5.
5. **PM2.5 in levels; population weights.**
6. **Within-country changes (information):** countries whose yearly `v2x_polyarchy` moves into
   another tercile (cutoffs fixed at baseline) within 1998–2018. Their regions' slope before and
   after the move. This is a small set of countries, so it is descriptive only.

## 4. ◇ Rules, fixed before any PM2.5 regression

- **◇ G1, coverage (loads no PM2.5).** Each tercile needs at least 15 countries and 150 regions
  in the sample, and in each tercile at least 50 regions must have income in each of the four
  spline segments (over all their years). If a tercile falls short, only the pooled path (no
  interaction) is estimated, and the moderation question is reported as unanswerable with DOSE.
- **◇ Moderation.** "Institutions change the path" is claimed only if both hold:
  - the Wald test that all three terciles share the same segment slopes rejects at 5 % with
    country clusters, in the main specification;
  - in each segment where the high and low terciles differ at 5 %, the difference has the same
    sign in variants 1 and 2.

  Otherwise: "no robust difference between institution groups".
- **◇ Turning point.** A downturn in PM2.5 (an environmental Kuznets curve) is reported only if a
  segment's slope is negative at 5 % with country clusters. A falling slope that is still positive
  is described as flattening, not as a turning point.
- **Reading.** Every coefficient is descriptive: a path conditional on region FE, not the effect
  of growth or of institutions. Countries differ in many ways besides institutions, which
  variant 2 only partly addresses.

## 5. Outcomes

**◇ G1: fails on the income-range condition** (2026-10-29, `src/analysis/pollution.py`; no
PM2.5 loaded).

- **Sample:** 27,512 region-years, 1,564 regions in 77 countries, 1998–2018.
- **Country and region counts pass:** each tercile has 25–26 countries and 415–597 regions.
- **Regions with income in each spline segment** (2015 USD per person, any year):

| Democracy tercile | Countries | Regions | < 1,000 | 1,000–4,000 | 4,000–12,000 | > 12,000 |
|---|---|---|---|---|---|---|
| low | 26 | 552 | 175 | 441 | 221 | **43** |
| middle | 25 | 597 | 83 | 342 | 346 | 154 |
| high | 26 | 415 | **0** | **18** | 133 | 344 |

- **Members.**
  - **Low:** ALB, ARE, AZE, BLR, CHN, EGY, ETH, GEO, IDN, IRN, KAZ, KEN, KGZ, LAO, MAR, MOZ,
    MYS, NGA, PAK, PER, RUS, SRB, TZA, UKR, UZB, VNM.
  - **Middle:** ARG, BGR, BIH, BOL, CAN, COL, ECU, GBR, GTM, HND, HRV, IND, JPN, KOR, LTU, LVA,
    MEX, MNG, PAN, PRY, ROU, SVK, THA, TUR, ZAF.
  - **High:** AUS, AUT, BEL, BRA, CHE, CHL, CZE, DEU, DNK, ESP, EST, FIN, FRA, GRC, HUN, IRL,
    ITA, NLD, NOR, NZL, POL, PRT, SVN, SWE, URY, USA.
- **Reading:** strong institutions and poverty barely overlap in DOSE. No high-tercile region is
  below 1,000 USD, and few low-tercile regions are above 12,000 USD. Only 4,000–12,000 USD has
  support in all three terciles.
- **By the rule,** only the pooled path would be estimated. The author chose a logged change
  instead (change log, 2026-10-29): compare terciles only on common support.

**Main specification** (2026-10-29, `output/notebooks/07_pollution.ipynb`; rules as amended by
the change log). G1 re-run first, with the same counts as above. 27,512 region-years, 77
countries. Slopes are elasticities of PM2.5 with respect to output per person, with country
clusters. Unsupported cells are shown in brackets and are not interpreted.

| Segment (2015 USD) | low | middle | high |
|---|---|---|---|
| < 1,000 | 0.118 (0.043) | 0.208 (0.170) | — |
| 1,000–4,000 | 0.058 (0.023) | 0.063 (0.055) | [0.710, 18 regions] |
| 4,000–12,000 | −0.009 (0.059) | −0.051 (0.072) | 0.118 (0.078) |
| > 12,000 | [−0.022, 43 regions] | −0.045 (0.054) | 0.034 (0.050) |

- **◇ Moderation: no robust difference between institution groups.**
  - The joint Wald test of the five supported comparisons gives F(5, 76) = 0.83, p = 0.53.
  - No single comparison differs at 5 %; the smallest p is 0.12 (high − middle at
    4,000–12,000). So the sign check in variants 1 and 2 has nothing to check. For the record,
    their Wald p values are 0.64 and 0.57.
  - Region clusters (information): F = 6.8, p < 0.001. The difference rests on treating
    regions as independent, not on countries.
- **◇ Turning point: none.** No supported slope is negative at 5 %. Only the low tercile's two
  lower segments are significant, and both are positive (p = 0.008 and 0.014).
- **Pooled path** (region + year FE, country clusters): 0.205 (0.044), 0.174 (0.043),
  0.051 (0.058), −0.083 (0.052; p = 0.12). PM2.5 rises with output below 4,000 USD and flattens
  above it. The top segment is negative but not significant, so it is flattening, not a turning
  point.
- **Variants 1 and 2 shrink the path.** With country trends or country × year FE, no supported
  slope is positive at 5 %. The middle tercile at 1,000–4,000 USD is −0.04 (p = 0.004 and
  0.017). So the positive slopes at low income in the main specification come from countries
  whose output and PM2.5 move together. Within a country-year, faster-growing regions are not
  more polluted.
- **Figure:** `output/figures/pollution/pm25_paths.{pdf,png}`. It shows the pooled path, the
  paths by tercile over their supported ranges, and the segment slopes in the main specification
  and variants 1–2.
- **Reading.** In DOSE regions, PM2.5 rises with output at low incomes and flattens at higher
  incomes. The data show no difference by baseline democracy where the terciles overlap. All
  of this is descriptive.

**Variants 3–6** (2026-10-29, `07_pollution.ipynb`). They are reported, not selected. The
◇ moderation rule rests on the main specification and variants 1–2, so its verdict stands.
Variants 3–5 recompute the supported cells, and every one keeps the same five comparisons.

| Variant | Region-years | Wald F (5 restrictions) | p |
|---|---|---|---|
| 3 rule of law (`v2x_rule` terciles) | 27,512 | 1.14 | 0.35 |
| 4 desert share ≤ 0.5 (105 regions drop) | 25,749 | 0.71 | 0.61 |
| 5a PM2.5 in levels | 27,512 | 0.51 | 0.77 |
| 5b ln PM2.5, population weights | 26,630 | 9.57 | < 0.001 |

- **3, rule of law.** No comparison differs at 5 %. The closest is high − middle above 12,000 USD,
  +0.25 (p = 0.056), where the high tercile's slope is 0.25 (p = 0.048). The other slopes are close
  to the main specification. 57 of the 77 countries have the same tercile under both indices.
- **4, without dusty regions.** The slopes barely move.
- **5a, levels.** The slopes are µg/m³ per log point of output. Only the low tercile at
  1,000–4,000 USD differs from zero at 5 %: 1.8 (p = 0.048).
- **5b, population weights: the only variant that rejects.**
  - Three comparisons differ at 5 %: middle − low below 1,000 USD, +0.36 (p = 0.030), and at
    1,000–4,000 USD, +0.21 (p = 0.049); high − middle at 4,000–12,000 USD, +0.60 (p = 0.032).
  - The weights concentrate on single countries: China holds 62 % of the low tercile's
    weight, India 62 % of the middle tercile's, and the United States 35 % of the high
    tercile's. So this check compares mostly China, India and the United States, not institution
    groups. The steep middle-tercile slopes below 4,000 USD (0.36 and 0.26) are mostly India's.
- **◇ Turning point:** none in variants 3–5. No supported slope is negative at 5 %.
- **6, countries that change tercile (information).** 10 lasting movers with 308 regions
  (5,276 region-years).
  - **Up (4 countries):** the slope is −0.14 before the move (p = 0.009) and −0.05 after
    (p = 0.27). After − before is +0.09 (p = 0.003).
  - **Down (6 countries):** the slope is +0.01 before and −0.03 after. After − before is
    −0.04 (p = 0.36).
  - With 10 country clusters, four of them up-movers, these p values are not reliable.
  - If more democracy flattened the path, the up-movers' slope would fall after the move;
    instead it rises towards zero. Four countries are no evidence either way.
- **Figures:** `output/figures/pollution/pm25_variants.{pdf,png}` (paths by rule-of-law
  tercile, paths in levels, slopes across variants 3–5) and `pm25_movers.{pdf,png}` (democracy
  trajectories with the baseline boundaries; movers' slopes before and after).
- **Reading.** Variants 3–5 support the main result: where the terciles overlap, there is no
  difference by institutions. The exception is population weighting, which turns the
  comparison into one of a few large countries.

## Change log

- **2026-10-29, after G1 and before any PM2.5 was loaded (author's decision).** G1 failed on
  support, not on country or region counts. The rules now compare terciles only where they
  overlap.
  - **Supported cell:** a tercile × segment with at least 50 regions.
  - **Comparisons, all between supported cells of the same segment:**
    - below 1,000 USD: low vs middle;
    - 1,000–4,000 USD: low vs middle;
    - 4,000–12,000 USD: all three equal (two restrictions);
    - above 12,000 USD: middle vs high.

    That makes five restrictions. Variants with other groupings (rule of law) or samples
    (without dusty regions) recompute the supported cells by the same 50-region rule.
  - **◇ Moderation, replacing §4's version.** The joint Wald test of these five restrictions
    rejects at 5 % with country clusters, in the main specification. And every comparison that
    differs at 5 % in the main specification keeps its sign in variants 1 and 2.
  - **◇ Turning point:** judged only in supported cells.
  - **The model still contains every tercile × segment term.** Unsupported cells are estimated
    but neither interpreted nor compared.
  - **The pooled path** (one spline, region + year FE) is reported as well.
  - **Why this is not fitted to a result:** G1 loads no PM2.5, and the change depends only on
    which income ranges each tercile reaches.
- **2026-10-29, before variants 3–6 were run with PM2.5 (construction only; the mover counts use
  V-Dem alone).** §3 does not define variant 6 fully, and variant 5 lists two checks. The rules
  are now:
  - **Variant 5** runs as two separate checks: PM2.5 in levels (µg/m³, unweighted), and ln PM2.5
    weighted by DOSE `pop`. The weighted check drops the 878 region-years without `pop`.
  - **Variant 6, yearly tercile:** the yearly `v2x_polyarchy` is cut at the baseline tercile
    boundaries (0.502 and 0.839).
  - **Variant 6, a move:** a country moves only if it is in a tercile other than its baseline
    one in every year from the move year through 2018, with at least three years on each side
    (move year 2001–2016). Countries that cross a cutoff and come back are not movers: 29
    countries leave their tercile in some year, often for one year near a cutoff.
  - **Variant 6, movers:** 10 countries.
    - **Up:** ALB 2005, GEO 2012, NGA 2012, PER 2001 (69 regions).
    - **Down:** BRA 2016, HND 2009, HUN 2010, POL 2016, THA 2006, TUR 2013 (239 regions).
  - **Variant 6, model:** on the movers' regions only, ln PM2.5 on ln output split into before
    and after the move, by direction (four slopes), plus a post-move dummy by direction, `P_a`,
    `T_a`, region FE and direction × year FE. Country clusters (10). The slope is linear rather
    than a spline, because there are too few countries per segment. The difference after − before
    is reported for each direction, as information only.

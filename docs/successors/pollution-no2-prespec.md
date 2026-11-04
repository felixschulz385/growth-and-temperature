# Pollution Along the Growth Path, with NO₂: Pre-Specification

**Status: confirmed 2026-10-30, as drafted. G3 passed, and all fits ran (2026-11-04, §5):
T1 not met, R1 met, moderation by the rule in the DOSE arm only, no turning point.** This is step 2 of [`pollution-data-options.md`](pollution-data-options.md) §4. It
repeats the two PM2.5 runs, idea B with DOSE output
([`pollution-institutions-prespec.md`](pollution-institutions-prespec.md)) and step 1 with night
lights ([`pollution-lights-prespec.md`](pollution-lights-prespec.md)), with surface NO₂ as the
outcome. Results so far: [`pollution-results.md`](pollution-results.md). **Descriptive only, no
causal claim.**

## 0. Decisions for the author

**Confirmed as drafted, 2026-10-30:** the recommended option in each case.

1. **Main arm: DOSE output, 2005–2018** (recommended), with the lights arm as the check. The
   alternative is to make lights the main arm.
   - DOSE measures output, not a proxy for it, and keeps the weather controls.
   - It gives 14 NO₂ years, against 8 for the lights arm's DMSP years 2005–2012.
   - Lights cover more countries, but in step 1 the lights path matched the output path in
     only some cells.
2. **Multiplicity:** the transport rule (◇ T1) uses Holm-adjusted p values across its four
   segments (recommended), not a raw 5 % per segment. Step 1's one result was a single
   comparison among eight. A new rule should not repeat that weakness.
3. **Implausible cells:** 1 km cells above 100 ppb are set to missing before region means
   (recommended). Annual mean surface NO₂ does not reach 100 ppb anywhere. The source page
   reports isolated cells of several hundred ppb against a 99th percentile near 10. In a small
   region (500 cells of 1 km), one 700 ppb cell would add 1.4 ppb to a mean of about 2.
4. **Lights arm reuses step 1's knots and terciles** (recommended) rather than recalibrating on
   2005–2012. The knots come from a calibration that loads no outcome. Reusing them keeps the
   segments' incomes and the tercile members identical to step 1, so the two outcomes can be
   compared directly. The DOSE arm likewise reuses idea B's terciles.
5. **A replication test of step 1's result** (◇ R1, recommended). Step 1 found one difference,
   above 12,000 USD between the high and middle terciles. Testing that one comparison for NO₂
   is the only confirmatory question here.

## 1. Question

1. **Transport.** With PM2.5, the path disappears within countries: with country × year FE, no
   positive slope survives. Is that because PM2.5 travels across regional borders? NO₂ lasts
   hours and stays near traffic, power plants and industry. If NO₂ shows a within-country path
   where PM2.5 does not, transport explains the national-only result.
2. **Institutions.** Does the NO₂ path differ by baseline democracy, as idea B and step 1 asked
   for PM2.5?

## 2. Data

| | Source | Module or column |
|---|---|---|
| Unit | GADM 4.1 first-level regions (GID_1) | `gadm` (panel `GID_1`) |
| Outcome | Anenberg et al. (2022) surface NO₂, ppb: region mean over the land 1 km cells of the region's 10 km panel cells, each valid cell weighted equally (equal-area grid) | `no2` (1 km tiles) |
| Comparison outcome | ACAG PM2.5, the region mean of idea B and step 1 | panel `pm25` |
| Growth, DOSE arm | DOSE `grp_pc_lcu2015_usd` | `dose` |
| Growth, lights arm | ln lights per person, as in step 1 (flare mask, GHS-POP) | `ntl_harm`, `ghs_pop` |
| Institutions | baseline (1998–2000) `v2x_polyarchy` terciles: idea B's for the DOSE arm, step 1's for the lights arm | `vdem` |
| Weather (DOSE arm) | DOSE `P_a`, `T_a` | `dose` |

- **Years:**
  - DOSE arm: 2005–2018. NO₂ is annual from 2005; DOSE `P_a` and `T_a` stop in 2018.
  - Lights arm: 2005–2012, the DMSP years. VIIRS 2014–2020 is variant 6.
  - **NO₂ before 2005 is not used.** Its 1990, 1995 and 2000 values are scaled with another
    source (reanalysis, before OMI), and the gaps between them are 5 years.
- **Regions:**
  - The DOSE arm uses idea B's rule.
  - The lights arm uses step 1's: at least 5 10 km cells with PM2.5 and lights.
  - Both arms also need at least 500 valid 1 km NO₂ cells.
  - Region-years must have both NO₂ and PM2.5, so the two outcomes share a sample.
- **Flares:** the flare mask applies to lights and population only, as in step 1. Flares emit
  NO₂, and a region's NO₂ mean covers all its cells, as its PM2.5 mean does.
- **What the region FE leave.** Anenberg's base pattern is a fixed 2010–2012 land-use
  regression that uses roads and built-up land as predictors. Region FE absorb it. The
  variation over time comes from OMI column scaling at about 10–25 km, which is fine for ADM1
  means.

## 3. Specification

- **Outcome:** ln NO₂.
- **Model:** as in each PM2.5 run, with the same spline, knots, terciles and FE:
  - **DOSE arm:** region FE, tercile × year FE, `P_a` and `T_a`. Knots at 1,000, 4,000 and
    12,000 USD.
  - **Lights arm:** region FE and tercile × year FE, no weather. Step 1's knots (−4.022,
    −3.013 and −2.213 in ln lights per person).
  - **Both:** clustered by country, no weights. A cell is supported if at least 50 regions have
    growth in it, and the comparisons are between adjacent supported terciles.
- **Pooled paths:** one spline. In the main specification it has region + year FE. For ◇ T1
  it has region + country × year FE.
- **PM2.5 on the same sample.** Each arm also fits ln PM2.5 on the NO₂ sample. This is needed
  for ◇ T1, and the sample differs from the PM2.5 runs in its years.
- **Stacked fit for ◇ T1.** ln NO₂ and ln PM2.5 are stacked, each region-year twice. Each
  outcome gets its own spline slopes, region × outcome FE and country × year × outcome FE.
  Errors are clustered by country, so the covariance between outcomes is estimated. The
  difference NO₂ − PM2.5 in each segment's slope is then a single coefficient test.

### Variants (reported, not selected)

1. + country-specific linear trends.
2. Country × year FE instead of tercile × year FE.
3. Rule of law (`v2x_rule` terciles).
4. Population-weighted NO₂ (GHS-POP weights at 1 km, linear between epochs) with population
   regression weights. The PM2.5 runs' dust variant is dropped: dust contains no NO₂.
5. NO₂ in levels (ppb).
6. Lights arm only: VIIRS years 2014–2020, with knots and terciles set by step 1's rules on
   those years.
7. Lights arm only: ln lights per area.
8. Lights arm only: + GLASS air temperature.
9. **H, lights arm only:** low-haze regions, the lower two terciles of baseline (1998–2000)
   PM2.5, as in step 1. Haze dims measured lights wherever aerosols are high, whatever the
   outcome.

## 4. ◇ Rules, fixed before any NO₂ regression

- **◇ G3, coverage (loads no NO₂ values; it counts valid cells).**
  - For each arm: the sample (region-years, regions, countries) and the number of regions
    that the NO₂ cell rule drops, and where.
  - The supported tercile × segment cells and the comparisons.
  - If a tercile has fewer than 15 countries or 150 regions, that arm estimates only its
    pooled paths.
- **◇ T1, transport.** In the DOSE arm, fit the stacked pooled paths with region + country ×
  year FE. For each of the four segments, test whether NO₂ − PM2.5 > 0, with Holm-adjusted p
  values across the four segments.
  - **"NO₂ has a within-country path that PM2.5 lacks"** is claimed only if two things hold.
    - Some segment's NO₂ − PM2.5 is positive with a Holm-adjusted p below 0.05, and its NO₂
      slope is positive at 5 %.
    - The same segment's NO₂ − PM2.5 is positive in the lights arm, at any p. If the lights
      arm lacks that segment, the condition is not met.
  - Otherwise: "NO₂ does not show a within-country path that PM2.5 lacks".
  - A within-country NO₂ path is necessary for the transport explanation, but not sufficient.
    The write-up says so.
- **◇ R1, replicating step 1.** In the lights arm, main specification, NO₂ above 12,000 USD:
  high − middle. "Step 1's difference replicates with NO₂" is claimed only if it is positive at
  5 %. The same comparison for PM2.5 on the NO₂ sample is reported as information.
- **◇ Moderation.** The same rule as idea B and step 1, in each arm:
  - the joint Wald test of the supported comparisons rejects at 5 %;
  - each comparison that differs at 5 % keeps its sign in variants 1 and 2, and in the lights
    arm also in variant H.

  Each arm reports its own verdict. The conclusion leads with the DOSE arm (§0.1).
- **◇ Turning point.** A tercile × segment slope counts as a downturn only if it is negative
  at 5 % in the main specification. In the lights arm, it must also be negative in variant H.
- **Haze bias, stated in advance (lights arm).** Aerosols dim measured lights most where
  PM2.5 is high. NO₂ and PM2.5 share sources, so slopes of ln NO₂ on ln lights are biased down,
  as with PM2.5. A positive slope is conservative, and a negative one is suspect.
- **Reading.** Every coefficient is descriptive.

## 5. Outcomes

### NO₂ fits: main specification, ◇ T1, ◇ R1, ◇ moderation, ◇ turning point, variants (2026-11-04; `09_pollution_no2.ipynb`)

The first NO₂ load. Construction follows the change log (2026-11-04, before any NO₂
regression). Figures: `output/figures/pollution/no2_paths` (DOSE arm: pooled path, paths by
tercile, variants 1–2) and `no2_transport` (◇ T1, both arms). Region NO₂ has a median of 2.8 ppb in the DOSE arm and 2.0 in the lights arm
(1st–99th percentile about 0.2–11).

- **◇ T1, transport: "NO₂ does not show a within-country path that PM2.5 lacks".** Stacked
  pooled paths with region + country × year FE:

  | Segment | DOSE: NO₂ | DOSE: PM2.5 | DOSE: NO₂ − PM2.5 (Holm p) | lights: NO₂ − PM2.5 |
  |---|---|---|---|---|
  | < 1,000 | 0.041 (p 0.040) | −0.005 | +0.046 (0.26) | −0.003 |
  | 1,000–4,000 | **0.068** (0.003) | −0.023 | **+0.091 (0.003)** | −0.002 |
  | 4,000–12,000 | 0.021 (0.25) | −0.028 | +0.049 (0.069) | +0.000 |
  | > 12,000 | −0.017 (0.51) | +0.011 | −0.028 (0.85) | −0.012 |

  - The DOSE arm meets the first condition at 1,000–4,000 USD: NO₂ rises with output within
    countries (0.068) where PM2.5 does not (−0.023), and the difference passes Holm.
  - The lights arm does not confirm it: there, NO₂ − PM2.5 at 1,000–4,000 USD is −0.002, and no
    within-country NO₂ slope differs from zero (all p > 0.26). So the rule's claim fails.
  - A within-country NO₂ path would be necessary for the transport explanation, not
    sufficient.
- **◇ R1: "step 1's difference replicates with NO₂".** Lights arm, above 12,000 USD, high −
  middle: +0.095 (0.040), p = 0.017. PM2.5 on the same rows (information): +0.091, p = 0.064.
  - Information: it keeps its sign and 5 % in variants H (+0.104), 3, 4, 5 and 8. It is
    +0.006 with country trends (p = 0.81), +0.048 with country × year FE (p = 0.090), +0.026 in
    the VIIRS years (p = 0.19) and −0.016 with lights per area. As for PM2.5, it is a DMSP-years
    comparison across countries.
- **◇ Moderation, DOSE arm (the conclusion leads with it): by the rule, "institutions change
  the path".**
  - Joint test of the 4 comparisons: F = 8.08 (4, 75), p < 0.001. For PM2.5 on the same rows
    (information): p = 0.92.
  - One comparison at 5 %: 4,000–12,000 USD, high − middle +0.212 (0.089), p = 0.020. It keeps
    its sign in variant 1 (+0.001, p = 0.98) and variant 2 (+0.022). In variant 1 it is zero
    in all but sign.
  - Slopes at 4,000–12,000 USD: high 0.345 (p < 0.001), middle 0.133 (p = 0.058), low −0.009.
    High-democracy regions' NO₂ rises fastest with output, the opposite of "democracies clean
    up sooner".
- **◇ Moderation, lights arm: "no robust difference".** The joint test rejects (F = 2.13 (8,
  163), p = 0.036), with two comparisons at 5 %: high − middle below 1,000 USD (+0.054,
  p = 0.029) and above 12,000 USD (+0.095). Both keep their sign in variants 1 and 2. The first
  has no counterpart in variant H, where the high tercile has 39 regions below 1,000 USD, so the
  rule is not met.
- **◇ Turning point: none in either arm.** No supported slope is negative at 5 % in the DOSE
  arm. In the lights arm, middle above 12,000 USD is −0.065 (p = 0.028) in the main
  specification but not negative at 5 % in H.
- **Pooled paths** (region + year FE):
  - DOSE arm: 0.122 (p = 0.032), 0.192 (p < 0.001), 0.144 (p = 0.005), −0.042 (0.25). NO₂
    rises with output up to about 12,000 USD.
  - Lights arm: 0.006, 0.041 (p = 0.014), 0.025, −0.039 (p = 0.060).
- **Variants** (joint test p; high − middle above 12,000 USD):

  | Variant | DOSE arm | lights arm |
  |---|---|---|
  | main | < 0.001; +0.153 (p 0.057) | 0.036; +0.095 (0.017) |
  | PM2.5, same rows (information) | 0.92; +0.028 (0.62) | 0.009; +0.091 (0.064) |
  | 1 + country trends | 0.18; +0.059 (0.11) | 0.15; +0.006 (0.81) |
  | 2 country × year FE | 0.28; +0.030 (0.52) | 0.57; +0.048 (0.090) |
  | H low haze | — | 0.006; +0.104 (0.022) |
  | 3 rule of law | < 0.001; +0.329 (0.006) | 0.001; +0.105 (0.008) |
  | 4 population-weighted NO₂ + weights | < 0.001; +0.377 (0.044) | < 0.001; +0.321 (0.008) |
  | 5 NO₂ in levels | < 0.001; +0.678 ppb (0.049) | 0.010; +0.414 ppb (0.003) |
  | 6 VIIRS 2014–2020 (quartile knots) | — | 0.14; +0.026 (0.19) |
  | 7 lights per area (quartile knots) | — | 0.006; −0.016 (0.76) |
  | 8 + GLASS air temperature | — | 0.044; +0.095 (0.022) |

  The tercile differences disappear with country trends or country × year FE in both arms:
  they compare countries.
- **Reading.**
  - In the DOSE arm, NO₂ rises with output up to about 12,000 USD (pooled path), and its
    tercile slopes are mostly larger than PM2.5's on the same rows. Within countries (country ×
    year FE) only the 1,000–4,000 USD rise remains, and lights do not reproduce it.
  - High-democracy countries' NO₂ rises at least as fast as the middle tercile's wherever they
    can be compared. No group shows a downturn.

### ◇ G3, coverage (2026-11-04, on the fixed tiles; `09_pollution_no2.ipynb`; no NO₂ values loaded)

**G3 passes in both arms.** Every tercile clears 15 countries and 150 regions, so each arm
estimates its tercile paths.

- **DOSE arm (2005–2018):** 18,841 region-years, 1,507 regions, 76 countries. The cell rule
  (< 500 valid 1 km cells) drops 452 region-years in 37 regions (Switzerland 7, no other
  country more than 2). Four cells are unsupported: high below 1,000 USD (0 regions) and at
  1–4k (10), middle below 1,000 (34), low above 12k (40). That leaves four comparisons:
  middle − low at 1–4k and 4–12k, high − middle at 4–12k and above 12k.
- **Lights arm (2005–2012, step 1's rerun knots and terciles):** 21,858 region-years, 2,772
  regions, 164 countries. The cell rule drops 428 region-years in 56 regions, mostly on small
  islands (Cape Verde 7, Mauritius 5, Trinidad and Tobago 3). All 12 cells are
  supported (the high tercile has 73 regions below 1,000 USD: India 21, Senegal 14, Mongolia
  12, Dominican Republic 7).
- **The first G3 run (2026-10-30) is superseded:** its cell counts predate the `no2` fix, and
  its lights arm used step 1's old panel (change log, 2026-11-04).

## 6. Already seen (disclosure)

- **PM2.5 with both growth measures** ([`pollution-results.md`](pollution-results.md)).
  - With country × year FE, no PM2.5 slope is positive at 5 % with DOSE output.
  - With lights, only the high tercile above 12,000 USD is (+0.030).
  - In both, the pooled path is flat above about 4,000 USD.
  - The PM2.5 counterparts in this pre-spec use shorter samples (2005–2018 and 2005–2012), so
    they are new but close to what has been seen.
- **NO₂ itself** (`docs/data/no2/README.md`, 2026-10-30): the global land mean and 99th
  percentile for 2017–2020, and the cell maximum, from the break check. No NO₂ value has been
  aggregated to a region or linked to growth.
- **Not seen:** any NO₂ path, region mean or regression.

## Change log

- **2026-10-30, confirmation.** The author confirmed the draft and the five decisions of §0 as
  drafted, before G3 and before any NO₂ value was aggregated. No rule changed.
- **2026-10-30, after the first G3 run, before any NO₂ value was aggregated (construction
  fix).** G3 showed that New Zealand has no valid NO₂ cell. The cause is a PREPARE bug.
  - A tile that crosses the antimeridian (the first and last tile columns) got an inverted
    clip box, so PREPARE NaN-filled it.
  - It hits `no2` fully in those tiles, and `ghs_pop` partly: half of New Zealand, Fiji,
    Vanuatu and New Caledonia. It also hits `topography` and `grip`, which only the closed
    corridor work used.
  - `ntl_harm`, `acag` and `eog` flares are complete there.
  - The fix is `clip_to_tile()` in `src/data/common/raster/spatial.py`, used by all seven
    tiled sources. The 14 wrapping tiles were re-prepared for `no2` and `ghs_pop` (SLURM jobs
    29146068 and 29146069). The old outputs are in
    `data_nobackup/_old/prepare_antimeridian_2026-10-30/`.
  - Finding the cause meant reading 30 raw NO₂ values at six test points (New Zealand,
    Hawaii, Alaska, Zurich). None was linked to a region or to growth.
  - **G3 is rerun after the fix; the first run is superseded.** Step 1 used `ghs_pop`, so its
    sample and population are checked too (`pollution-lights-prespec.md`).
- **2026-11-02, scan of every tiled source in the wrapping tiles (loads no outcome).** The
  entry above was wrong about the flares, and it missed VIIRS.
  - `eog_viirs` has the bug too, but its tiles are zero-filled, not NaN: in all 14 wrapping
    tiles, every column is 0, including the cloud-free count. Re-prepared with the fix
    (SLURM job 29344146, a full rerun).
  - `eog_flare` uses its own lon/lat box to pick the flare points, and that box wraps the same
    way. It dropped 6–25 flares a year near 180° (the New Zealand tile now gets 764 flare
    cells in 2018, not 0). Fixed in `src/data/sources/eog/flare.py`; the wrapping tiles are
    re-prepared (job 29344151). Step 1's flare mask therefore missed New Zealand's flares.
  - `topography` and `grip` are empty in every wrapping tile. They are not re-prepared: only
    the closed corridor work used them.
  - Complete and plausible there: `ntl_harm`, `acag`, `esacci`, GLASS air temperature, MODIS
    LST and `country_id`. The `no2` tiles east of 180° (Samoa, Tonga, Fiji's Lau islands)
    stay empty: the raw file has no data there.
- **2026-11-04, G3 had not been rerun after the fix (loads no NO₂ values).** The cell-count
  cache (`_pollution_no2_cell_counts_10km.parquet`) was written at 16:08 on 2026-10-30. The
  `no2` re-prepare (job 29146068) ran from 16:22 to 16:39, so the G3 in the notebook committed
  in `bd9dc20` counted the broken wrapping tiles. The lights arm also used step 1's old panel,
  which lacked part of `ghs_pop` there and the flares near 180°
  (`pollution-lights-prespec.md`, change log, 2026-11-04). Both caches were moved to
  `data_nobackup/_old/pollution_lights_antimeridian_2026-11-04/`, and G3 is rerun on the
  fixed tiles. The lights arm keeps decision 4 (step 1's knots and terciles), now taken from
  step 1's rerun.
- **2026-11-04, after G3 and before any NO₂ regression (construction; no NO₂ value has been
  aggregated).** The author said go. §3 and §4 leave these details open; they are fixed here:
  - **Region NO₂:** the mean of the valid land 1 km cells (≤ 100 ppb) over the region's 10 km
    panel cells, each cell weighted equally. **Variant 4:** each 1 km cell weighted by its
    GHS-POP density in the same 1 km cell, linear between the 2005–2020 epochs (zero where
    missing). The regression weight is idea B's DOSE `pop` in the DOSE arm and step 1's
    region population in the lights arm.
  - **◇ T1:** the stacked fit interacts the DOSE arm's `P_a` and `T_a` with the outcome. "NO₂ −
    PM2.5 > 0" is a one-sided t test (G − 1 df, as for the comparisons), Holm-adjusted across
    the four segments. "Its NO₂ slope is positive at 5 %" means positive with a two-sided
    p below 0.05, as in the other rules. Two-sided p values of the differences are reported as
    information.
  - **◇ Turning point, lights arm:** "also negative in variant H" means negative at 5 %, as in
    step 1's rule.
  - **Variant H:** step 1's low-haze regions (baseline 1998–2000 PM2.5, cutoffs across step 1's
    sample). The NO₂ sample starts in 2005 and has no baseline years of its own.
  - **Variants 6 and 7:** step 1's variant-6 knots and terciles (set on 2014–2022), on the rows
    of 2014–2020; step 1's variant-7 knots (ln lights per area, 1998–2012). This follows
    decision 4.
  - **Variant 8:** GLASS air temperature as in step 1, on the rows that have it.
  - **PM2.5 on the NO₂ sample:** each arm's main specification is also fitted for ln PM2.5, as
    information and for ◇ T1 and ◇ R1.

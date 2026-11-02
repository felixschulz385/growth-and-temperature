# Pollution Along the Growth Path, with NO₂: Pre-Specification

**Status: confirmed 2026-10-30, as drafted. No NO₂ value has been linked to a region or to
growth.** This is step 2 of [`pollution-data-options.md`](pollution-data-options.md) §4. It
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

None yet.

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

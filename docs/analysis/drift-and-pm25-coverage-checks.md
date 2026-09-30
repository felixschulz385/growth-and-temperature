# Aqua Drift and PM2.5 Coverage Checks

Analysis note, 2026-09-28. It records two checks run for the analysis plan
([`final-analysis-plan.md`](final-analysis-plan.md)):

1. **Aqua orbital drift** — the Phase 0 panel-end rule (◇): does the drift in Aqua's night
   overpass time bias night LST differently across places before the panel ends?
2. **Clear-sky coverage of ACAG PM2.5** — open question 2: could ACAG's reliance on
   clear-sky satellite retrievals bias the Phase 6a aerosol test?

## 1. Aqua night-overpass drift

**Design** (`scripts/aqua_drift_diagnostic.py`, `scripts/aqua_drift_summary.py`). This is
narrower than [`14`](../design/14-terra-aqua-drift-diagnostic.md), which asks a Terra/Aqua
fusion question.
- **Source data:** MYD21A2 `View_Time_Night` (local solar observation time) and night LST,
  read with the FETCH machinery (STAC search, tile-pinned load, QC mask, month-first
  annual compositing).
- **Coverage:** the 5 robustness tiles (Amazon h12v09, Sahara h18v06, Central Europe h18v04,
  Siberia h22v03, Australia h30v11). Aqua for 2003, 2008, 2013, 2016 and 2018–2025; Terra for
  2018–2019.
- **Drift:** a pixel's view time minus its 2018–19 mean.
- **Cooling rate:** each pixel's pre-drift rate, (Terra LST − Aqua LST) / (Aqua time − Terra
  time), from the ~22:30 and ~01:30 overpasses.
- **Implied LST bias:** −rate × drift.
- **Summary:** medians by tile, year and 2° latitude band.

The 2025 composite is incomplete: 28 of 46 eight-day periods.

**Results.** Median drift in overpass time vs 2018–19 (hours; positive = later):

| Tile | 2003–2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| Amazon | −0.08 to +0.02 | 0.00 | +0.07 | +0.28 | +0.65 | +1.11 |
| Central Europe | −0.07 to +0.01 | 0.00 | +0.06 | +0.26 | +0.64 | +1.12 |
| Sahara | −0.06 to +0.01 | 0.00 | +0.06 | +0.24 | +0.61 | +1.10 |
| Siberia | −0.07 to +0.03 | +0.01 | +0.07 | +0.25 | +0.61 | +1.13 |
| Australia | −0.07 to 0.00 | −0.02 | +0.04 | +0.24 | +0.59 | +1.09 |

The night overpass drifts **later**; doc 14's "earlier" does not describe Aqua's night
observation. Median night cooling rates at ~01:30 are 0.24 K/h (Amazon), 0.38 (Central
Europe), 0.39 (Siberia), 0.68 (Sahara) and 0.80 (Australia).

Implied LST bias:

| | 2022 | 2023 | 2025 |
|---|---|---|---|
| Median by tile | −0.01 to −0.04 K | −0.06 to −0.19 K | −0.26 to −0.88 K |
| Gap between tiles | 0.03 K | 0.12 K | 0.62 K |
| Within-tile spread across 2° bands | ≤ 0.015 K | ≤ 0.04 K | up to 0.17 K |
| Pixel spread within a band (p90 − p10) | 0.06–0.08 K | 0.08–0.15 K | 0.27–0.56 K |

In pre-drift years (2003–2020), the within-band pixel spread is 0.02–0.10 K and the gap
between tiles is at most 0.05 K.

**Interpretation.**
- **Through 2022, negligible.** The drift and its heterogeneity are about the size of
  pre-drift year-to-year noise. To bias the IV, a bias this small would also need to
  correlate with mine exposure, and it applies to 1 of 21 years.
- **From 2023 on, material.** The drift depends on biome, because deserts cool fastest, and
  by 2025 the gap between tiles (0.6 K) exceeds the effects the paper tries to detect.
- **The rule as written is not decisive.** The ◇ rule says "if the post-2020 shift differs
  across tiles or biomes → end in 2020". It did not set a threshold, and strictly the 2022
  shift does differ by 0.03 K. Choosing between 2020 and 2022 is a judgement call, recorded
  in the plan.
- **Hard limit.** Any MODIS night-LST panel must **not extend past 2022**. ACAG runs to 2023
  and the MODIS config to 2025.

## 2. Clear-sky coverage of ACAG PM2.5

**Why an indirect test.** ACAG's annual NetCDF (V6GL02.04) carries only `PM25`: no
data-support or uncertainty field. Aqua MODIS daytime LST shares the sensor, the 13:30
overpass and the cloud screening of the AOD retrievals ACAG draws on, so its valid counts
proxy clear-sky retrieval availability.

**Design** (`scripts/acag_coverage_check.py`):
- **Panel:** 10 km, 2002–2022, flares dropped, 25.2 M pixel-years.
- **FE and clusters:** pixel + country × biome × year FE; GID_1 clusters.
- **`cov_day`:** this year's valid pixel-months divided by the pixel's best year.
- **`dens_day`:** valid 8-day periods per valid pixel-month, i.e. clear-sky frequency.
- **Terciles:** of the pixel's 2002–06 mean `dens_day`.

**Results:**

| Test | Estimate (SE) | p |
|---|---|---|
| T2 `cov_day` ~ `mine_count_20km` | −0.0005 (0.0006) | 0.41 |
| T2 `dens_day` ~ `mine_count_20km` | −0.0016 (0.0020) | 0.43 |
| T1 `pm25` ~ `cov_day` | −0.053 (0.086) | 0.54 |
| T1 `pm25` ~ `dens_day` | +0.334 (0.087) | < 0.001 |
| T3 `pm25` ~ `mine_count_20km`, all | −0.128 (0.067) | 0.056 |
| cloudiest tercile | −0.121 (0.101) | 0.23 |
| middle tercile | −0.127 (0.082) | 0.12 |
| clearest tercile | −0.297 (0.114) | 0.009 |

Mean PM2.5 is 17.1, 16.2 and 23.9 µg/m³ from the cloudiest to the clearest tercile.

**Interpretation.**
- **The instrument does not move clear-sky coverage (T2).** A coverage artefact in ACAG is
  therefore orthogonal to Z and cannot bias the Phase 6a reduced form of PM2.5 on Z. This is
  the part that matters for the analysis.
- **ACAG does co-move with clear-sky frequency within a pixel (T1), but little.** A typical
  year-to-year swing of ~0.3 periods per month is worth ~0.1 µg/m³, against a mean of 19. The
  test cannot tell a retrieval artefact from real meteorology: dry, stagnant years hold more
  particulate.
- **The mine effect on PM2.5 does not differ significantly by cloudiness (T3).** The gap
  between the clearest and cloudiest terciles is −0.18, with SE ≈ 0.15. It is largest in the
  clearest tercile. That fits some damping of ACAG's signal where retrievals are rare, but
  that tercile is also the drylands (highest PM2.5, dust). In cloudy regions, read an
  ACAG-based aerosol effect as a possible lower bound in magnitude.
- **Side finding for Phase 6a.** Mine exposure goes with slightly *lower* PM2.5 (p ≈ 0.06),
  the opposite of the growth → pollution → cooling story. It needs a proper look in
  Phase 6a, not a conclusion here.

## 3. Reproduce

```bash
# drift: needs outbound access to Planetary Computer; ~30 min for 70 tile-years, 2 in parallel
for t in h12v09 h18v06 h18v04 h22v03 h30v11; do
  python scripts/aqua_drift_diagnostic.py --tile $t --out scratch_nobackup/drift
done
python scripts/aqua_drift_summary.py scratch_nobackup/drift

# ACAG coverage: ~15 min on 16 cores
python scripts/acag_coverage_check.py scratch_nobackup/acag_coverage
```

The drift download skips tile-years that are already on disk. From Claude Code sessions,
run it on the session node: SLURM jobs inherit the session's local proxy variables, which
don't work on other nodes.

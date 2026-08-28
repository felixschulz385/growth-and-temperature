# 14 — Terra/Aqua Platform-Bias & Orbital-Drift Diagnostic

Handoff note for whoever picks this up next (agent or human) — this environment (local sandbox,
this session) has no `pystac_client`/`planetary_computer`/`odc-stac` installed and likely no
egress to Planetary Computer's blob storage for actual pixel reads, so **this diagnostic needs to
run somewhere with real STAC+COG network access** — the scicore HPC environment (same one the
oversized-tile fix/cleanup was verified against, `docs/design/13-prepare-memory-parallelism.md`)
is the natural place, following this repo's usual pattern (`scripts/*.py` + `--slurm` default, see
`scripts/find_and_remove_oversized_modis_tiles.py` for the exact convention to mirror: dry-run and
`--slurm` both default on, self-resubmits via `sbatch --wrap=...` built from
`orchestration/configs/slurm_jobs.yaml`'s `cluster:` block, no committed `.sh` file).

## Background (context a fresh agent won't have)

This surfaced from a chain of fixes/discussion in one session, 2026-08-26:

1. Fixed a real MODIS FETCH bug (`ModisSource._tile_bbox_4326`/`_load_tile_year`,
   `src/data/sources/modis/source.py`) that wrote wildly oversized `.tif` tiles (up to ~40,000 km
   wide instead of the true ~1,112 km MODIS tile) — root cause was a naive antimeridian-unsafe bbox
   calc feeding an unconstrained `odc.stac.load`, compounded by the STAC bbox search legitimately
   also matching neighbouring h/v tiles' items (`_search_items` now filters those out via
   `modis:horizontal-tile`/`modis:vertical-tile` STAC properties). Verified fixed against real HPC
   data (`2005/h09v02.tif` now comes out exactly 1200x1200px / 1111.95x1111.95 km).
2. In discussing whether to add multi-platform (Terra+Aqua) fusion to increase temporal sampling
   density (currently FETCH defaults to `platform: aqua` only, filtering Terra out — the
   `modis-21A2-061` STAC collection already contains both), surfaced that **Terra and Aqua have
   both been drifting to progressively earlier equatorial crossing times since their final
   inclination maneuvers in 2020/2021** (real, current, ongoing — not a fixed calibration offset;
   see NASA's own drift page and the NSIDC data announcement, both linked below). This means a
   platform's "night" observation local solar time is not stable across the mission, which risks
   an artificial temperature trend purely from sampling-time drift, independent of real warming —
   a real concern for this project's actual use case (temperature-vs-growth panel analysis).
3. Searched for prior art: Collection 6/6.1 MxD21 (this pipeline's product) has documented ground-
   validation mean biases of **+0.34 K (Terra) / -0.67 K (Aqua), daytime** — better than legacy
   MxD11's -1.77/-2.63 K, confirming C6.1's recalibration genuinely helped, but no night-specific
   numbers were found and no ready-made *annual*, drift-corrected, Terra+Aqua-merged product
   exists off the shelf (closest are monthly CMG products with no drift correction, or research-
   grade diurnal-temperature-cycle normalization methods that aren't distributed products). NASA's
   own stated long-term direction is VIIRS continuity, not retroactively fixing MODIS for this.
4. Discussed how downstream country×year fixed effects interact with all this: they absorb any
   bias uniform across a whole country in a given year (including a country-specific manifestation
   of the drift, since it's the fully-interacted term, not additive country FE + year FE), but do
   **not** absorb within-country-year heterogeneity (if the drift's LST impact or platform mix
   varies by sub-national land cover/climate) or measurement noise from sparse/low-
   `valid_period_count` pixels (that's variance, not a level bias — needs weighting, not FEs). Also
   flagged: country×year FE is only useful here if the regression's unit of observation still has
   within-country-year identifying variation (pixel/grid-cell level) — if temperature enters the
   regression already aggregated to country-year, the FE would span the same space and leave
   nothing to identify off.
5. Landed on a prioritized next-steps list, of which **this diagnostic is #1** (the "Run 1" this
   doc is a handoff for) — its result gates the fusion-pooling decision (blindly-pooled composite
   vs. platform-kept-separate) and clarifies how urgent the within-country-year heterogeneity gap
   actually is.

## What "Run 1" means, concretely

**Goal**: measure whether Terra-minus-Aqua LST divergence for *this pipeline's own tiles* is (a) a
roughly constant offset across years (→ correctable with a single empirical bias term, cheap), or
(b) growing over time, especially post-2020/2021 (→ confirms live drift, argues for keeping
platform explicit downstream rather than pre-correcting).

**Method**:

1. Pick a handful of representative tiles spanning different biomes/latitudes — reuse the 5 tiles
   already chosen for `modis_robustness_11a1` (`orchestration/configs/data.yaml`,
   `docs/design/07b-modis-outstanding.md`'s "tiles/years" entry): `h12v09` (Amazon rainforest),
   `h18v06` (Sahara desert), `h18v04` (Central Europe temperate), `h22v03` (Siberian boreal/taiga),
   `h30v11` (Australian outback). These were already deliberately chosen for biome diversity — no
   need to re-derive a tile list.
2. Pick years spanning pre/post the 2020/2021 drift onset — e.g. `2005, 2012, 2018, 2021, 2023,
   2025` (adjust freely; the point is bracketing 2020/2021, not these exact values).
3. For each (tile, year): call `ModisSource._search_items(tile, year)` **twice**, once with
   `self.platform` temporarily forced to `"terra"` and once `"aqua"` (or refactor `_search_items`
   to take an explicit platform override arg rather than mutating `self.platform` — cleaner, do
   whichever is less invasive), then `_load_tile_year(items, tile)`, `decode_qc_valid_mask(...)`,
   and `composite_annual_stats(..., stats=("mean",))` — i.e. reuse the exact existing FETCH machinery
   rather than reimplementing compositing logic. This gives one Terra annual-mean array and one
   Aqua annual-mean array per (tile, year), both already QC-masked and month-first-weighted.
4. Compute the per-pixel difference (Terra - Aqua) where both have valid data that year; report
   mean/std/percentiles of the difference, plus the count of jointly-valid pixels (the comparison
   is only meaningful where both platforms actually have data).
5. Report as a small table: (tile, year) x (mean_diff_K, std_diff_K, n_joint_valid_px). Look for:
   is mean_diff roughly flat across years (→ static bias) or trending in magnitude, especially
   after 2020/2021 (→ drift signature)? Does it vary by tile/biome (→ within-country-year
   heterogeneity risk is real, not hypothetical)?

**Where this differs from a naive "just diff the two rasters" approach**: reusing
`_search_items`/`_load_tile_year`/`decode_qc_valid_mask`/`composite_annual_stats` directly (rather
than hand-rolling a parallel pixel pipeline) means this diagnostic automatically inherits every
correctness fix already made this session (`geobox=` pin, tile-identity STAC filter, antimeridian-
safe bbox, month-first weighting) — don't reimplement any of that.

## Practical notes for whoever runs this

- Needs real network access to `https://planetarycomputer.microsoft.com` and its backing blob
  storage for actual COG pixel reads (not just STAC metadata) — confirmed this session that STAC
  *metadata* queries work fine from a normal web-fetch-capable environment, but actual
  `odc.stac.load()` pixel reads are a different, heavier network path. Verify from wherever this
  runs before committing to the full tile/year matrix.
- `ModisSource.__init__` requires a `PipelineContext`/`SourceConfig` the same way every other
  script in this repo constructs one — see `scripts/find_and_remove_oversized_modis_tiles.py`'s
  `build_source()` or `scripts/migrate_legacy_layout.py`'s, for the exact pattern (uses
  `get_source_config(config, source_key)` + `PipelineContext(data_root=...)`).
- This is read-only against Planetary Computer (no writes, no changes to already-fetched local
  files) — safe to run without the dry-run ceremony the cleanup script needed, but still worth
  putting behind `--slurm` default given the network I/O volume across 5 tiles x 6 years x 2
  platforms = 60 loads.
- Keep the output as a plain table/CSV, not a fitted model — the point of this step is descriptive
  (does the gap exist, is it growing), not yet a correction. Fitting a bias-correction model is a
  later, separate decision gated by what this shows.

## Sources (from this session's web research, for whoever wants to go deeper)

- [MODIS/Aqua LST/Emissivity 8-Day L3 Global 1km SIN Grid V061 (MYD21A2)](https://www.earthdata.nasa.gov/data/catalog/lpcloud-myd21a2-061)
- [MODIS Land Surface Temperature and Emissivity (MOD21) product page](https://modis.gsfc.nasa.gov/data/dataprod/mod21.php)
- [Collection-6 MOD21 ATBD/User Guide (NASA)](https://modis.gsfc.nasa.gov/data/user_guide/atbd_mod21_userguide.pdf)
- [Terra Orbital Drift Information (NASA Terra mission site)](https://terra.nasa.gov/about/terras-orbit-changes/terra-orbital-drift-information)
- [Ongoing changes in Terra and Aqua orbits impacting MODIS snow and sea ice products (NSIDC)](https://nsidc.org/data/user-resources/data-announcements/ongoing-changes-terra-and-aqua-orbits-impacting-modis-snow-and-sea-ice-products)
- [Normalization of the temporal effect on the MODIS LST product using random forest regression](https://www.sciencedirect.com/science/article/abs/pii/S0924271619301042)
- [Generation of a time-consistent land surface temperature product from MODIS data](https://www.sciencedirect.com/science/article/abs/pii/S0034425713003064)
- [Improved estimates of monthly LST from MODIS using a diurnal temperature cycle (DTC) model](https://www.sciencedirect.com/science/article/abs/pii/S0924271620302161)

## Next steps after this diagnostic (from the same session, for context)

1. **(this doc)** Run the Terra/Aqua divergence check.
2. Add `platform_fraction` (Terra share of valid periods) as a FETCH/PREPARE output column
   regardless of the fusion decision — cheap now, unlocks every downstream option (covariate for
   FE interaction, empirical cross-calibration input, or just a diagnostic).
3. Keep using `valid_period_count`/`valid_month_count` (already produced) as regression
   weights/coverage filters downstream — addresses sparse-pixel measurement noise, which FEs don't.
4. Pin down the regression's unit of observation (pixel/grid-cell vs. country-year-aggregated)
   before finalizing anything above — gates whether country×year FE has identifying variation left
   and whether `platform_fraction` belongs in the raster pipeline or an analysis-time join.
5. Decide fusion pooling strategy (blindly pooled vs. platform-kept-separate/averaged) only after
   step 1 has an answer — not urgent, MODIS FETCH is already correct without it.

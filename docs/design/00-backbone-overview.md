# 00 — Backbone Overview

## What changes, and why

The econometric design behind this repository changed. The estimating equation is now a
distance-ring specification:

```
T_bt = μ_b + η_{c(b)t} + Σ_{r=0..R} β_r · L^(r)_bt + u_bt
```

where `μ_b` is a pixel fixed effect, `η_ct` a country×year fixed effect, and `L^(r)_bt` is the
**mean radiance over the annulus `[d_{r-1}, d_r)`** around cell `b`. The quantity of interest is
`Σ_r β_r`, not `β_0`. With pixel fixed effects and only own-cell lights, an estimator on a 1 km
grid with a 5 km spillover range recovers roughly `(h/ρ)² ≈ 4%` of the true effect — neighbours are
differenced away along with the confounder. The ring terms exist to put the neighbourhood signal
back.

Two things follow, and they are the two things this backbone redesigns:

**1. The grid is wrong.** `src/data/assemble/constants.py` sets `DEFAULT_CRS = 4326` — geographic
lat/lon. In EPSG:4326 a "1 km" cell is not 1 km on the ground, and its ground size varies with
latitude, so a disc of pixels is not a disc on the ground. The distortion is systematic in
latitude, which correlates with almost every variable of interest in this study. Ring means require
a grid that is **metric and equal-area**. See [`01-grid.md`](01-grid.md) for the full argument and
the CRS decision (EPSG:6933 / EASE-Grid 2.0, not MODIS Sinusoidal, not per-tile local CRS).

**2. There is no neighbourhood computation.** A grep for `convolve`, `annulus`, `ring_`,
`neighborhood`/`neighbourhood`, `disc_kernel`, `FFT` across `src/` returns nothing. Ring means are
disc-mean differences, and disc means are convolutions — `O(N log N)` via FFT, as opposed to
enumerating ~2,000 neighbours per cell for hundreds of millions of cells, which is intractable. This
is genuinely new capability, not an extension of anything that exists today. See
[`03-neighbourhood-engine.md`](03-neighbourhood-engine.md).

Everything else in this set of documents — storage layout, ingest changes, tabularization rules,
migration sequencing — exists to support those two changes without breaking the pipeline that
currently works.

## Scope and what these documents are not

This is a **design plan**, produced without writing implementation code. It settles decisions with
reasons, and records rejected alternatives, so the reasoning is reconstructable without this
conversation. Two things it deliberately does not do:

- It does not touch `src/analysis/` beyond stating the tabular interface that stage produces and
  consumes (`pixel_id` + `year` + country id + ring-mean columns + provenance). Within-cell and
  country×year fixed-effect absorption is an estimation-time concern, not a backbone concern —
  see [`05-migration.md`](05-migration.md) and the tabularization section of
  [`04-ingest.md`](04-ingest.md) for why the existing `demean_columns`/`assemble demean` machinery is
  dead code, not a pattern to preserve.
- It does not fully resolve the MODIS LST product decision (11A1 vs 11A2 vs 21A2), STAC host
  availability, or band-level scale/QC handling. [`04-ingest.md`](04-ingest.md) states the ingest
  *principles* (resampling method per variable, annual compositing at ingest, valid-observation
  counts as a diagnostic) and defers the rest to [`07-modis-ingest.md`](07-modis-ingest.md)/
  [`07a-modis-band-reference.md`](07a-modis-band-reference.md), which resolve it (recommendation:
  MYD21A2 primary, not 11A1 — see `07` §1 for why).
- It also did not originally anticipate a need to move processed/aggregated data over SSH — MODIS
  ingest's streaming requirement surfaced that need, and [`08-hpc-transfer.md`](08-hpc-transfer.md)
  specifies it as a generic capability, not a MODIS-only one.

## Confirmed parameters (settled with the user before writing)

| Parameter | Value | Why |
|---|---|---|
| Target resolution | 1 km | Matches native LST resolution (both GLASS and MODIS candidates are 1 km-native); the README's 500 m figure would be false precision relative to the coarsest scientifically meaningful input. |
| R_max (ring ladder top) | 30 km | ~6× the 5 km spillover range used as the motivating example in the research brief — a deliberate margin past where the effect should be negligible. |
| Disk budget | 5–20 TB free | The backbone stores (analysis-ready grids + disc-sum/count ladder) are estimated at ~5.0 TB uncompressed / ~1.9 TB compressed under these parameters — comfortably inside budget, see [`02-storage.md`](02-storage.md) for the worked arithmetic. |
| Latitude clip | \|φ\| ≤ 60° | Caps kernel anisotropy at 3:1 under EPSG:6933; independently justified because VIIRS DNB is unusable at high summer latitude and thermal LST also degrades there. Accepted as a scope restriction. |

## Order of work

The dependency-ordered task list lives in [`05-migration.md`](05-migration.md). At a glance: grid
construction → latitude-band kernel registry → convolution engine → per-source resampling fixes →
Zarr disc-ladder store → assembly integration → validation on a subset → cutover. The existing
EPSG:4326 pipeline keeps running throughout; `DEFAULT_CRS` is the one deliberate in-place flip, and
it happens last, not first.

## Document map

1. [`01-grid.md`](01-grid.md) — canonical GeoBox, CRS decision and rejected alternatives, tiling,
   `pixel_id`, latitude-band kernel registry, clipping, antimeridian/pole handling.
2. [`02-storage.md`](02-storage.md) — Zarr layout, chunking, dtypes/compression, disc-ladder schema,
   sizing table.
3. [`03-neighbourhood-engine.md`](03-neighbourhood-engine.md) — the new convolution subsystem.
4. [`04-ingest.md`](04-ingest.md) — per-source acquisition/regridding changes, resampling per
   variable, annual compositing, STAC evaluation.
5. [`05-migration.md`](05-migration.md) — additive rollout, cutover criteria, task list.
6. [`06-open-questions.md`](06-open-questions.md) — everything not resolvable from the repo, with
   the specific check that would resolve it.
7. [`07-modis-ingest.md`](07-modis-ingest.md) — MODIS LST product decision (MYD21A2 primary,
   MYD11A1 robustness arm), architecture, compositing, correctness details, operational design.
   [`07a-modis-band-reference.md`](07a-modis-band-reference.md) — verified per-band scale/offset/
   fill/QC reference.
8. [`08-hpc-transfer.md`](08-hpc-transfer.md) — generic capability for pushing a preprocess stage's
   local output to the HPC over SSH, built around the existing `HPCClient`; used by MODIS's stage
   "annual" so it can run wherever internet egress is available.

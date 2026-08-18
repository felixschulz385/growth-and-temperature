# 02 — Storage (Zarr)

## 1. Exactly two write points

**Decision: persist Zarr at exactly two points** — (1) analysis-ready grids, after regridding onto
the canonical GeoBox *and* after temporal compositing to annual, and (2) the nested disc sum/count
store (`S_d`, `N_d`), after convolution. Everything else (halo-padded arrays, intermediate
reprojections, ring means derived from `S_d`/`N_d`) stays in memory / Dask graphs, never persisted.

**Update — point 1 is now parquet for the shared-driver sources.** `acag`, `esacci`, `ntl_harm`, and
`eog` (every source going through `src.data.common.prepare.driver.run_tiled_prepare` /
`SpatialProcessor.process_tile_region`) now write `cell_id`-keyed parquet parts
(`ix=<row>/iy=<col>/part-<year>.parquet`, one self-contained file per (tile, year) unit) instead of a
Zarr region write into a pre-allocated skeleton — no shared store to bootstrap, no region-write
chunk-alignment constraint. Written as a wide table (`cell_id`, `year`, one column per data variable)
so a later stage can widen it with more columns (e.g. convolution output) without changing its shape.
SCAFFOLDING: no convolution/ring-mean logic runs here, this is exactly the reprojected variable
value(s). `glass/modis.py` and `modis/source.py` (the other point-1 producers, via
`SpatialProcessor.write_year_to_zarr`/`process_spatial_standard`, whole-extent-per-year, multi-band)
are **unchanged** — still Zarr, deferred because their output shape and the CRS-bug regression tests
guarding `write_year_to_zarr` (`tests/data/common/raster/test_spatial_crs_preservation.py`) are
meaningfully different from the single/few-variable tiled case. `gadm.py`/`ecoregions/source.py`/
`snl_mining/source.py`/`glass/avhrr.py`/`osm.py`/`berman_mining.py`'s bespoke hand-rolled Zarr tiling
is also unchanged. Point 2 (the disc-ladder store) is unaffected — see this document's earlier
addendum on `write_disc_tile_parquet`.

Not yet done, left for follow-up: `src/data/sources/verify.py`'s `_run_verification` dispatches any
directory path to `_verify_zarr` (`xr.open_zarr`), so verification of these four sources' PREPARE
output now fails (caught, reported as a verification failure, not a crash) until it gains a
parquet-parts-directory check; `src/data/assemble/loaders.py` still reads Zarr and does not yet
consume this parquet output.

**Why not fewer than two.** The convolution engine needs a *common* grid — same CRS, resolution, and
tiling — across every input before it can share one kernel registry and one halo-read pattern
(`GeoBox.buffered(R_max)`). That makes a canonical-grid intermediate structurally required before
convolution can happen at all; you cannot go directly from raw per-source native-grid rasters to disc
sums.

**Why not more than two.** Nothing else in the pipeline needs a persisted intermediate:
- Halo-padded arrays are a read-time slice (`tile_geobox.buffered(R_max)`), not an artefact anyone
  reuses — persisting them would duplicate the analysis-ready store with padding baked in, for no
  benefit.
- Ring means (`L^(r) = (S_{d_r} − S_{d_{r-1}}) / (N_{d_r} − N_{d_{r-1}})`) are a division of
  differences, cheap to compute at query/tabularization time from `S_d`/`N_d` (§4) — persisting them
  would mean committing to one fixed ring partition forever, and would forfeit the ability to test a
  different ladder without rerunning the expensive convolution.

Each of the two chosen points corresponds to a genuinely different array shape/grid requirement
(point 1: one array per pixel per year; point 2: one array per pixel per year per radius) — that's
what makes them the right two, not an arbitrary pair.

**Scaffolding addendum:** a parquet-format sibling writer for point 2's per-tile output now exists,
`src/data/common/neighbourhood/store.py::write_disc_tile_parquet` — raw `S_d`/`N_d` per pixel per
radius, keyed on the new global `cell_id` (ease6933-only, [`01-grid.md`](01-grid.md) §5a), sorted and
written as `ix=<row>/iy=<col>/part-<year>.parquet` instead of a Zarr region write. It is explicitly
**not** production-wired (no per-source PREPARE CLI calls it; only unit tests and
`scripts/validate_backbone_subset.py`'s optional `--parquet-out` flag exercise it) and does not change
the "exactly two Zarr write points" decision above — it's an additional, parallel output format for
point 2, not a replacement.

## 2. One store per variable family — tied to the CRS decision

**Decision: one Zarr store per variable family** (e.g. `lights.zarr`, `lst.zarr`, `ndvi.zarr`,
`disc_sums/lights.zarr`), sharing the canonical GeoBox/CRS/chunking across all of them — **not** one
store per tile.

This follows directly from rejecting per-tile local-AEQD in [`01-grid.md`](01-grid.md) §1: because
every variable lives on the same GeoBox, a single Zarr array chunked by tile is a coherent,
mosaicked, randomly-sliceable object — exactly what the convolution engine's halo reads need (a read
spanning a tile boundary resolves to one coordinate space, no CRS reconciliation). Had the CRS been
per-tile, one-store-per-variable-family would have been *impossible* — a single Zarr array cannot
span multiple CRSs. **This is the same decision as the CRS decision, made once, with consequences in
two documents.**

## 3. Chunking, padding, and parallel writes

**Chunk shape = tile shape**: 2048×2048 spatial, chunk size 1 along the time/year dimension, matching
`GeoboxTiles` exactly — so each pipeline tile maps to one Zarr chunk, and `to_zarr(..., region=...)`
writes from different tile-workers land in disjoint chunks with no cross-worker contention. Pattern:
initialize the store once with the correct shape/dtype/chunking via `to_zarr(..., compute=False)`,
then have each tile-worker write its own region. This is a **new pattern for this repo's assembly
stage** — today's stage_3 output is parquet, and Zarr is used only for intermediate stage_2 arrays —
so it should be validated on a small subset early, not assumed to behave identically to the existing
parquet-per-tile writer.

**Padding adequacy**: at `C=2048`, R_max = 30 km needs a 30 px halo. The existing
`DEFAULT_TILE_PADDING=64` already covers this with 2× margin for its original (resampling-edge)
purpose, but the convolution engine's own halo reads should use `GeoBox.buffered(R_max)` computed
directly from R_max, not reuse the fixed 64 px constant — see [`01-grid.md`](01-grid.md) §4 for why
the two padding needs are decoupled.

## 4. Disc-ladder schema: store `S_d`/`N_d`, not ring means

**Decision: recommended radii ladder `[1, 2, 3, 4, 5, 7, 10, 14, 20, 30] km`** (10 rings). Spacing
is dense near the origin (1 km steps out to 5 km) and coarsens beyond that (7 → 10 → 14 → 20 → 30).
Why: the research context states the own-cell estimator recovers only ~4% of the true effect at
1 km/5 km-spillover, meaning most of the identifying variation sits in the *near* rings — fine
resolution there captures the steepest part of the spillover-decay curve, while coarser spacing
further out is adequate once annulus areas are already large and marginal information per additional
ring is lower. The top of the ladder tracks R_max (30 km, confirmed) — if R_max changes, the ladder
top should move with it; the near-origin spacing should not, since it isn't R_max-dependent.

**Decision: store cumulative disc sums `S_d` and disc valid-counts `N_d` at each radius — never ring
means, and never even per-annulus sums directly.** Reasons:
1. A ring mean is a division of a *difference* of two cumulative quantities. Storing the cumulative
   `S_d`/`N_d` lets any future ring boundary (a different ladder, a robustness check) be constructed
   at read time by differencing — with zero recomputation of the expensive raster-space convolution.
2. Sums are what FFT convolution produces natively (`disc sum = field ⊛ disc_kernel`); a mean
   requires an extra division by the count, and that division is exactly the step that must be
   deferred to read/tabularization time so missing data is handled correctly rather than baked in as
   a lossy raster-time average (§ neighbourhood engine, [`03-neighbourhood-engine.md`](03-neighbourhood-engine.md)).
3. Storing means directly would discard the count information needed both to know how much of an
   annulus was actually observed (coastal cells, cloud/missing-data gaps, cross-border masking, the
   60° clip edge all reduce `N_d` below the full geometric annulus pixel count) and to support
   thinning/robustness weighting at the row level later ([`04-ingest.md`](04-ingest.md) §
   tabularization) — both need `N_d` as a first-class, queryable output.

## 5. Sharing `N_d` across variables with the same validity mask

**Decision: share `N_d` arrays across every convolved variable that uses the same validity mask**,
rather than storing a separate `N_d` per variable — `N_d` depends only on the mask, never on the
variable's values.

Illustrative saving: if a mask family is shared by 6 convolved variables (lights + 4 mediators + mine
density) at 10 radii, storing `N_d` once instead of once per variable is a **6× reduction** on that
mask family's `N_d` volume. In the worked example below (§6), unshared `N_d` for that group would run
to ~1.7 TB uncompressed; shared, it's ~0.56 TB — **roughly 1.1 TB saved**, before compression is even
applied.

## 6. Dtypes, compression, and a worked sizing table

**Dtypes:**

| Array | Dtype | Why |
|---|---|---|
| Analysis-ready continuous fields (LST, NDVI, radiance, albedo, PM2.5) | `float32` | No scientific justification for `float64` — remote-sensing measurement error dwarfs float32 rounding. |
| `S_d` (disc sums) | `float32` | Sum of float32 inputs; same reasoning. |
| `N_d` (disc counts) | `uint16` | Max disc pixel count at R=30 km / 1 km resolution ≈ π·30² ≈ 2,827 px — `uint8` (max 255) is not enough, `uint16` (max 65,535) fits with large margin, `uint32` is unnecessary. |
| Validity/land masks | `uint8` | Boolean-ish; leaves room for a small enum of invalidity reasons if useful later. |
| Country-id raster (from existing GADM output) | `uint16` | A few hundred countries; fits easily. |

**Compression: Zarr `blosc` codec, `zstd`, level ~5, shuffle enabled** as a starting default for both
smooth geophysical fields and highly spatially-autocorrelated integer counts. **Flag: the exact ratio
should be measured on real data early, not assumed** — the table below uses illustrative ratios
(2.5× for float32 fields, 5× for uint16 counts) purely so the sizing arithmetic is checkable; they
are not measurements.

**Worked example** (assumptions stated explicitly so it can be checked against the confirmed 5–20 TB
budget): 1 km resolution, |φ|≤60° clip (441.2M px/slice, from [`01-grid.md`](01-grid.md)), 32 years
(1992–2023), 9 analysis-ready variables (LST, harmonized lights, 4 mediators [NDVI/tree-cover/
built-up/albedo], PM2.5, land-validity mask, country-id), 6 convolved variables (lights + 4 mediators
+ mine density) at the 10-radius ladder above, 2 mask variants for `S_d`/`N_d` (unmasked +
own-country-masked, applied only to lights per [`03-neighbourhood-engine.md`](03-neighbourhood-engine.md) §5).

| Component | Arithmetic | Uncompressed | Compressed (assumed) |
|---|---|---|---|
| Analysis-ready grids | 9 vars × 32 yr × 441.2M px × 4 B | 508 GB | ~200 GB |
| `S_d` (disc sums) | 7 combos (6 vars×1 mask + 1 extra mask on lights) × 10 radii × 32 yr × 441.2M px × 4 B | 3.95 TB | ~1.6 TB |
| `N_d` (disc counts, shared) | 2 mask variants × 10 radii × 32 yr × 441.2M px × 2 B | 0.56 TB | ~0.11 TB |
| **Total (backbone stores only)** | | **~5.0 TB** | **~1.9 TB** |

This excludes: existing raw/native-grid preprocess stage_2 zarrs (unchanged in kind), the final
tabular parquet (row count depends on land fraction and thinning — [`04-ingest.md`](04-ingest.md)),
and any robustness-check re-runs with alternative ladders/R_max. **Comfortably inside the confirmed
5–20 TB budget** ([`00-backbone-overview.md`](00-backbone-overview.md)), but treat this table as
order-of-magnitude — the real number depends on measured compression ratios and the final variable
list, both to be checked before the expensive stage runs.

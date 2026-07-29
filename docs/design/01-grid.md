# 01 — Grid & CRS

## 1. CRS decision: EPSG:6933, clipped to |φ| ≤ 60°

**Decision: a single global GeoBox in EPSG:6933** (WGS 84 / NSIDC EASE-Grid 2.0 Global, Lambert
cylindrical equal-area, standard parallel 30°), clipped to |φ| ≤ 60°.

**Why equal-area and metric matters.** Ring means are area averages over `[d_{r-1}, d_r)` km from a
cell. That's only a well-defined operation if grid cells have (approximately) equal ground area and
"km" in pixel space means the same thing everywhere. EPSG:4326 (the current `DEFAULT_CRS`) fails
both: cell ground-area shrinks with `cos(φ)` toward the poles, and a "disc of pixels" is an ellipse
on the ground whose eccentricity is systematic in latitude — which correlates with nearly every
covariate in this study (development, climate, land cover).

**Why EPSG:6933 specifically.** It is equal-area, and — unlike sinusoidal or conic projections — it
has **zero shear**: distortion is a closed-form function of latitude alone,

```
scale_EW = cos(φ_s)/cos(φ)      scale_NS = cos(φ)/cos(φ_s)      product = 1
```

with `φ_s = 30°` the standard parallel. Because the anisotropy is 1-dimensional (depends only on
row/latitude, not on column/longitude), it is correctable with a **latitude-banded elliptical
kernel** (§4) instead of a per-pixel or per-tile correction — a real engineering simplification that
follows directly from picking this specific projection.

### Rejected: MODIS Sinusoidal

Equal-area, but **sheared**: meridians tilt by `arctan((λ−λ₀)·sin φ)`, reaching ~48° at (45°N, 90°E)
under MODIS's global `λ₀ = 0`. A disc of pixels becomes a skewed parallelogram whose shape depends on
*both* latitude and longitude — correcting it would need a 2-D (lat×lon) kernel registry, orders of
magnitude larger than the 1-D registry EPSG:6933 needs, for a study that is explicitly global (not
single-continent, where the shear near one central meridian is tolerable). Rejected outright.

### Evaluated and rejected: per-tile local azimuthal equidistant (AEQD)

**Pro:** discs are exact circles in pixel space in every tile — no elliptical-kernel correction
needed at all.

**Con, decisive:** AEQD's projection origin moves with the tile, so every tile lives in a different
CRS. That breaks the one-canonical-GeoBox model this repo already depends on throughout the
assembly pipeline (`GeoboxTiles(target_geobox, (tile_size, tile_size))`, one shared `target_geobox`,
one `pixel_id` numbering scheme). Under per-tile AEQD:

- a single Zarr array can no longer span tiles (different CRS per tile ⇒ different `x`/`y`
  coordinate systems ⇒ no single coherent array) — this forces **per-tile stores**, not
  one-store-per-variable-family (see [`02-storage.md`](02-storage.md)),
- halo reads for convolution are inherently cross-tile, so every halo read becomes a
  reprojection/mosaic operation instead of a slice,
- there is no single canonical `pixel_id` scheme, and
- the same ground point has different pixel coordinates in each of its neighbouring tiles' CRS,
  doubling reprojection cost at every tile boundary.

**Verdict:** reject. The latitude-banded elliptical kernel under EPSG:6933 is a modest, one-time,
quantified engineering cost (build a kernel registry) traded against keeping the
one-store-per-variable-family model intact. For a system that already leans on that model
everywhere, that is the correct trade. **This decision and the storage-layout decision in
[`02-storage.md`](02-storage.md) are the same decision, made once here.**

### Latitude clip: |φ| ≤ 60°

Two independent justifications:
1. Caps the elliptical anisotropy ratio `cos²(φ_s)/cos²(φ)` at 3:1 (0.75 at the equator, 1.0 at 30°,
   3.0 at 60°), bounding both the kernel registry's size and the worst-case convolution error.
2. VIIRS DNB is unusable at high latitude in summer (no astronomical darkness), and thermal LST
   retrieval quality also degrades there — so above 60° both the regressor and the outcome are
   compromised independent of any grid concern.

This is a real sample restriction (excludes most of Scandinavia, Alaska, northern Canada, Siberia,
Antarctica) and was confirmed accepted by the user as a scope decision; see
[`06-open-questions.md`](06-open-questions.md) for the note that the exact land-area/country impact
has not been quantified here and is a research-team sign-off item, not an engineering check.

## 2. Canonical GeoBox: extent, resolution, construction

**Decision: compute the extent programmatically from the installed PROJ database at build/init
time** (`pyproj.Transformer.from_crs("EPSG:4326", "EPSG:6933")` to find the projected coordinates of
the geographic bounds, then `GeoBox.from_bbox(bbox, crs="EPSG:6933", resolution=...)`), rather than
hand-copying published constants.

Why: a direct check against the installed PROJ database (pyproj) found the antimeridian (x) extent
agrees with NSIDC's published EASE2_G constant to millimetre precision (17,367,530.45 m computed vs.
17,367,530.44 m published), but the **polar (y) extent disagrees by ~0.38%** (7,342,230.14 m
computed via ellipsoidal transform at φ=90° vs. 7,314,540.83 m published by NSIDC). This is likely
because NSIDC's published raster grid is defined by an integer row/column count times a rounded cell
size rather than the literal ellipsoidal pole-to-pole distance — but the discrepancy was not
resolved from a primary source and should not be silently picked one way. See
[`06-open-questions.md`](06-open-questions.md) item 1. Computing the extent from the installed PROJ
database at build time, rather than hardcoding either number, also means the grid stays
self-consistent with whatever geodesy library version is actually pinned — which matters more once
[`05-migration.md`](05-migration.md)'s dependency-pinning fix is in place, since an unpinned
PROJ/pyproj today means this exact extent could silently shift between machines.

**At |φ| ≤ 60°, resolution 1 km:**

```
x ∈ [−17,367,530.45, +17,367,530.45] m   (full width  34,735,060.9 m)
y ∈ [ −6,351,420.00,  +6,351,420.00] m   (full height 12,702,840.0 m, at φ=60°)
```

Grid dimensions: width ≈ ⌈34,735,061 / 1000⌉ = **34,735 px**, height ≈ ⌈12,702,840 / 1000⌉ =
**12,703 px**. Total ≈ **441.2 million pixels** per full time-slice (dense, including ocean).

**Resolution: 1 km, confirmed with the user.** The README states 500 m; both LST candidates (GLASS,
MODIS MYD21/11A1) are 1 km-native, so 1 km avoids fabricating precision the outcome variable can't
support. See [`00-backbone-overview.md`](00-backbone-overview.md) for the confirmed-parameters table.

## 3. Tiling scheme: tile size vs. halo overhead

**Decision: keep `DEFAULT_TILE_SIZE = 2048`.** For a core tile of `C×C` pixels padded by `R` pixels
on each side, the extra-work fraction from the halo is:

```
overhead = ((C+2R)² − C²) / C² = 4R/C + 4R²/C²
```

| Tile size C (px) | R=10 km | R=20 km | R=30 km | R=50 km |
|---|---|---|---|---|
| 256  | 16.2% | 33.7% | 52.4% | 93.4% |
| 512  | 8.0%  | 16.2% | 24.8% | 42.9% |
| 1024 | 3.9%  | 8.0%  | 12.1% | 20.5% |
| **2048** | **2.0%** | **3.9%** | **5.9%** | **10.0%** |
| 4096 | 1.0%  | 2.0%  | 3.0%  | 4.9%  |

At R_max = 30 km (the confirmed parameter), `2048` gives 5.9% overhead — a good balance. `4096`
would nearly halve it again (3.0%) but at 4× the per-tile memory footprint for concurrently
processed tiles and coarser Dask/SLURM scheduling granularity; `512` or below push overhead into a
wasteful 25–93% range. **`2048` is the right choice, unchanged from today** — this is a case where
the existing constant survives the CRS migration unmodified.

## 4. Padding: two distinct concerns, decoupled

Today, `DEFAULT_TILE_PADDING = 64` (px) is used only to avoid resampling edge artifacts before
reprojection (`tile_geobox.pad(DEFAULT_TILE_PADDING, DEFAULT_TILE_PADDING)`, used in
`processors.py` and `geometry.py`) — it has nothing to do with neighbourhood computation, because no
neighbourhood computation exists yet. `GeoBox.buffered()` is not called anywhere in `src/`.

**Decision: keep `DEFAULT_TILE_PADDING = 64` for its existing resampling-edge purpose, and introduce
a separate, R_max-derived halo for the convolution engine**, computed as
`halo_px = ceil(R_max_km / resolution_km)` and read via `tile_geobox.buffered(R_max_m)` (in
projected metres, since the canonical grid is now metric — this is the first real use of
`.buffered()` in this codebase). At R_max = 30 km / 1 km resolution, the convolution halo is 30 px —
comfortably inside the existing 64 px constant's margin, but the two should not share one constant:
a future change to R_max must not silently under-pad reprojection, or over/under-pad convolution,
because they happened to be tied together by coincidence today.

## 5. `pixel_id`: does the existing 16/16/32 layout survive?

**Decision: yes, unchanged.** The existing scheme (`src/data/assemble/constants.py`):

```
pixel_id = (ix << 48) | (iy << 32) | local_pixel   # [ix:16 | iy:16 | local_pixel:32]
```

At tile_size = 2048 and 1 km resolution, the clipped grid needs `⌈34735/2048⌉ = 17` tiles in x and
`⌈12703/2048⌉ = 7` tiles in y — **119 tiles total**, both trivially inside the 16-bit (65,536)
per-axis budget. Even at the README's 500 m figure, tile counts only double (≈34 × 13–15) — still
far under the limit. `local_pixel` at `2048×2048 = 4,194,304` values is inside the 32-bit budget with
enormous margin (exhausted only at tile_size ≥ 65,536, an implausible choice). **This is a checked
item, not a redesign** — worth stating explicitly since the migration brief flagged it as something
to verify.

## 6. Latitude-band elliptical kernel registry

**Decision: precompute one 2-D convolution kernel per narrow latitude band**, applied by selecting
the appropriate band's kernel per output row during convolution (§ neighbourhood engine,
[`03-neighbourhood-engine.md`](03-neighbourhood-engine.md)).

**Band width.** Bands are chosen non-uniformly so the anisotropy ratio `cos²(φ_s)/cos²(φ)` changes
by less than ~2% within a band. The ratio's rate of change in latitude is
`d(ln ratio)/dφ = 2 tan(φ)`, which is ~0 near the equator (bands can be wide there) and largest near
the 60° clip edge (bands must be narrow there). Solving for a per-band width of
`Δφ ≈ ln(1.02) / (2 tan φ)` gives, illustratively, ~1° bands near 30° and ~0.33° bands near 60°;
integrating from the equator to 60° gives on the order of 100–150 bands total (both hemispheres) —
**this count is illustrative, not a spec**; compute exact band edges at implementation time from the
actual tolerance chosen, and cache the result (see below).

**Kernel construction.** For each radius `d` in the disc ladder ([`02-storage.md`](02-storage.md))
and each band, the kernel footprint is an ellipse in pixel space with semi-axes derived from
`scale_EW(φ_band)` and `scale_NS(φ_band)` — a ground-circle of radius `d` maps to a pixel-space
ellipse via those factors. **Flag explicitly:** the precise direction of the correction (which axis
compresses vs. expands at a given latitude) must be derived and verified numerically at
implementation time — e.g. by rasterizing a known-radius geographic circle at several test
latitudes and confirming the disc-mean recovers the true circle mean — rather than trusting a sign
convention asserted in this document without a runnable check.

**Registry design.** A small module-level object,
`EllipticalKernelRegistry(bands, radii, resolution_m) -> Dict[(band_idx, radius), np.ndarray]`,
built once per (canonical grid, disc ladder) pair and cached to disk — following the existing
`viirs_geobox.pkl` pattern in `src/data/common/geobox/geobox.py` — since it's expensive to build and
depends only on grid + ladder, never on data.

**Tile/band boundary handling.** A tile straddling a band boundary needs different kernels for
different output rows within the same tile. Implement as an outer loop over the bands intersecting a
tile's (padded) extent, restricting each convolution's *output* rows to that band while reading the
full padded input once — a standard pattern for spatially-varying-kernel convolution, still
`O(N log N)` per band-slice. Flag this as the piece of the engine most likely to have surprising
performance characteristics, worth prototyping and benchmarking early rather than assuming it's free.

## 7. Antimeridian and pole edge cases

**Decision: mostly moot for the raster engine itself under a clipped, single global EASE-Grid 2.0
GeoBox; the existing wrapdateline pattern remains only for vector inputs pre-reprojection.**

In a single finite EASE-Grid 2.0 raster, `x = ±17,367,530 m` is a hard raster edge, not a coordinate
discontinuity the way `±180°` is in geographic CRS — within-raster convolution near the antimeridian
is ordinary edge-of-array handling (pad with nodata), with no wraparound concept to implement.

The antimeridian problem only resurfaces for **inputs still expressed in EPSG:4326 before
reprojection onto the canonical grid** — vector geometries (GADM boundaries, mine points) crossing
±180°. For those, keep using the existing cascading fallback already in
`src/data/common/geobox/geobox.py` (`_patched_grid_intersect`): direct EPSG:4326 intersection →
`wrapdateline=True` → source footprint directly → world bounds `(-180, -90, 180, 90)`. That logic
operates at the reprojection-into-target-grid step, which is unchanged in kind by the CRS switch —
only the target CRS changes, from 4326 to 6933. No new antimeridian logic needs to be invented; the
existing pattern's job simply shrinks, since raster-space data no longer carries the discontinuity at
all once on the canonical grid.

Poles are irrelevant once clipped to |φ| ≤ 60°.

**Note on `geobox_patch.py`:** confirmed to be dead code — a byte-identical vendored copy of the
installed odc-geo 0.5.0's `geobox.py`, not imported anywhere in `src/`. The actual monkeypatches
(including the wrapdateline fallback above) live in `geobox.py`, not `geobox_patch.py`. Flagged for
deletion in [`05-migration.md`](05-migration.md) rather than left as unexplained 1,607 lines of dead
code next to the grid logic this design extends.

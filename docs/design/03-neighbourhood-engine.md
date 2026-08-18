# 03 — Neighbourhood (Convolution) Engine

This is the principal new capability in the redesign. A grep for `convolve`, `annulus`, `ring_`,
`neighborhood`/`neighbourhood`, `disc_kernel`, `FFT` across `src/` returns nothing — there is no
prior art in this codebase to extend, only prior art to *cite* (§6).

## 1. Module placement

**Decision: new module `src/data/common/neighbourhood/`**, sibling to `src/data/common/geobox/`, not
nested inside it. Reasoning: `geobox/`'s job is defining and patching the grid itself (CRS, tiling,
footprint intersection); the convolution engine is a distinct concern — kernel registry, halo reads,
FFT convolution, disc-ladder orchestration — that *depends on* a GeoBox but isn't part of
constructing one. Keeping them separate mirrors the existing split between "what is the grid"
(`geobox/`) and "what runs on the grid" (the rest of `src/data/`).

Suggested internal layout (illustrative, not binding):

```
src/data/common/neighbourhood/
  __init__.py     # public API
  kernels.py       # latitude-band elliptical kernel registry (built in 01-grid.md §6)
  convolve.py       # FFT convolution + halo-read orchestration
  discs.py          # disc-ladder orchestration -> (S_d, N_d)
```

## 2. Public interface

**Core function:** `convolve_discs(variable, mask, ladder, kernel_registry) -> (S_d, N_d)`

- **Inputs:** a variable `DataArray` on the canonical GeoBox (single time-slice or a batch over
  years), a validity mask `DataArray` (same grid), the disc-radius ladder
  ([`02-storage.md`](02-storage.md) §4), and the precomputed latitude-band kernel registry
  ([`01-grid.md`](01-grid.md) §6, built once and passed in, never rebuilt per call).
- **Outputs:** `S_d` and `N_d` DataArrays, one per radius, same grid/tiling as the input.

**Core guarantee, designed in from the start, not optional plumbing:** the engine convolves the
variable **and** the validity mask together, in the same call:

```
S_d = convolve(where(mask, variable, 0), kernel_d)
N_d = convolve(mask.astype(numeric), kernel_d)
```

**Why this is the single most important correctness property of the engine.** If the variable array
were convolved with NaNs treated as/replaced by zero *without* simultaneously tracking how many
valid cells contributed, every disc mean near any missing-data boundary — coastlines, cloud gaps,
sensor swath edges, the new |φ|≤60° clip edge itself — would be biased toward zero in proportion to
the missing fraction. That bias is spatially structured: correlated with coastal proximity, cloud
climatology, and latitude — exactly the kind of confound this entire redesign exists to eliminate.
Convolving the mask alongside the variable, and dividing only at read/tabularization time, makes the
missing-data handling exact instead of approximate.

**Output writers:** `store.py::write_disc_tile` (Zarr region write, production path for point 2 of
[`02-storage.md`](02-storage.md) §1) and `store.py::write_disc_tile_parquet` (scaffolding-only,
`cell_id`-keyed parquet sibling — see `02-storage.md`'s addendum and [`01-grid.md`](01-grid.md) §5a).
Neither is called from a pipeline CLI step yet — still no CLI verb (§ below).

## 3. Halo reads via `GeoBox.buffered(R_max)`

`GeoBox.buffered(xbuff, ybuff=None)` exists in the installed odc-geo API and is not currently called
anywhere in `src/`. **Decision: use it to compute each tile's padded read-window**
(`tile_geobox.buffered(R_max_m)`, in projected metres since the canonical grid is metric) before
convolution — this is the first real exercise of `.buffered()` in this codebase, replacing the ad hoc
fixed-pixel `tile_geobox.pad(...)` pattern for this specific purpose (see
[`01-grid.md`](01-grid.md) §4 for why the two padding needs — resampling-edge vs. convolution-halo —
are decoupled rather than sharing one constant). Worth a dedicated smoke test comparing its output
against a hand-computed bounding box before wiring it into the full pipeline.

## 4. Four consumers, one code path

The four named consumers — lights (regressor), mediators (NDVI/tree-cover/built-up/albedo),
rasterized mine points/counts, and the validity mask itself — all call the **same**
`convolve_discs(...)` function. This is not DRY-ness for its own sake: it's what makes "mediators
ride the same grid/neighbourhood treatment as the regressor, never as baseline controls" (the
estimand requirement from [`00-backbone-overview.md`](00-backbone-overview.md)) structurally true
rather than a convention that four separately-implemented call sites could drift apart on.

Concretely: convolving the mask itself (the 4th consumer) is the degenerate case
`convolve_discs(mask, mask, ladder, registry)` — `S_d` and `N_d` become the same array. This
degenerate case is also a natural unit test for the engine.

## 5. Cross-border ring decision

**Question: do rings cross national borders?** By default, yes — a ring straddling a border mixes
two country×year fixed effects (`η_{c(b)t}`).

**Decision: compute two variants — unmasked and own-country-masked — but apply the own-country-masked
variant only to lights (the regressor), not to every mediator.** Reasoning: this ties to the storage
argument in [`02-storage.md`](02-storage.md) §6 — applying it to all 6 convolved variables would
nearly double `S_d` storage for a robustness check whose scientific purpose (testing whether
cross-border neighbour lights still identify the effect, given that country×year FE is defined at the
country level) is specifically about the regressor's spillover, not mediator decomposition. If the
research team later needs masked mediator variants too, that's a cheap incremental addition to the
same shared code path (§4), not a redesign.

**Implementation:** `convolve_discs(variable, mask & (country_raster == own_country), ladder,
registry)` — one extra convolution against a country raster, using the **existing** GADM country-id
raster this repo already produces (`misc/processed/stage_2/gadm/countries_grid.zarr`, built by
`misc.py`'s GADM rasterization step). No new ingest work is needed for this — only a new consumer of
an existing store.

## 6. Prior art already in this repo

`src/data/preprocess/sources/snl_mining.py` already computes distance-based buffers (10/20/50 km) in
a metric equal-area CRS (`ESRI:54009`, World Mollweide, config-driven via `data.yaml`'s
`aggregation.metric_crs`) — confirming this repo has already established the idea that distance
computations need a metric CRS, just not generalized. It is **not** FFT-convolution-based: it's a
geometry-buffer/rasterize operation via DuckDB spatial functions on sparse point geometries, which is
appropriate for mine points but not for dense per-pixel means of continuous fields. **Cite it as
precedent for the idea** ("like `snl_mining`'s metric buffer computation, but raster-convolution-based
and shared across all four consumers instead of bespoke per-source geometry SQL"), not as code to
generalize directly.

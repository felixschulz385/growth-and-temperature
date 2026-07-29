# 05 — Migration

## 1. Additive rollout, nothing broken immediately

The new canonical EPSG:6933 GeoBox, the neighbourhood engine module, and the new Zarr disc-ladder
stores are all **additive** — they live alongside the existing EPSG:4326
`viirs_geobox.pkl`-derived grid and the existing parquet-based assembly pipeline. Nothing about
today's assembly output format needs to change to introduce them.

Concretely:
- Per-source preprocessing needed for the ring model (lights, mediators, LST) is **re-run onto the
  new grid as a parallel output path**, e.g. `<source>/processed/stage_2_ease6933/...` alongside
  today's `<source>/processed/stage_2/...`, leaving existing EPSG:4326 outputs untouched for any
  other consumer that may still depend on them.
- The lights resampling change (`nearest` → area-weighted sum, [`04-ingest.md`](04-ingest.md) §1) is
  scoped to the new grid path if implemented as a per-variable override rather than a shared-default
  flip — the existing EPSG:4326 lights output, if anything still consumes it, is unaffected.
- Recommend a config-level switch (e.g. `grid: ease6933` vs. `grid: legacy_4326` in
  `orchestration/configs/data.yaml`) so both pipelines are runnable simultaneously by configuration,
  not by branching code.

## 2. What changes in place, and when

**`DEFAULT_CRS = 4326` → EPSG:6933 is the one deliberate in-place change**, and it should be the
*last* thing flipped — only once the new grid, the neighbourhood engine, and the re-run preprocessing
have been validated end-to-end on a subset (one region, one year). Flipping it earlier would silently
change the canonical grid for every existing consumer of `get_or_create_geobox()` before the new
pipeline is proven.

**Cleanup items to call out explicitly, not left silent:**
- **Delete `src/data/common/geobox/geobox_patch.py`** (1,607 lines), confirmed dead — a
  byte-identical vendored copy of the installed odc-geo 0.5.0's `geobox.py`, not imported anywhere in
  `src/`. If there's a reason to keep it (e.g. as a rollback fallback for a future odc-geo upgrade),
  state that reason explicitly rather than leaving unexplained dead code next to the grid logic this
  migration extends.
- **Remove the dead `demean_columns`/`assemble demean` config and CLI wiring** — confirmed to raise
  `ModuleNotFoundError` today (`src/data/assemble/demean.py` doesn't exist). This is cleanup, not a
  breaking change, since nothing currently works through that path. See
  [`04-ingest.md`](04-ingest.md) §6 for why it should not be resurrected as-is.
- Same treatment for the confirmed-unconsumed `spark`/`filter_land_only` keys in
  `orchestration/configs/data.yaml`'s `processing:` block — dead config, not read anywhere in
  `src/data/assemble/*.py`.

## 3. Prerequisite: pin the environment

**No `pyproject.toml`, `environment.yml`, `requirements.txt`, `setup.py`, `poetry.lock`, or
`conda*.yml` exists anywhere in the repository**, despite the README's install instructions
referencing `conda env create -f environment.yml` as if it exists. The locally installed
`odc.geo.__version__ == "0.5.0"` is not pinned anywhere; `orchestration/slurm/*.sh` scripts simply
`conda activate src`, assuming a pre-existing, out-of-repo environment.

**Recommend this be fixed as a prerequisite of this migration, not deferred.** This is not general
hygiene — it's specific to this work: the entire CRS/GeoBox design in
[`01-grid.md`](01-grid.md) depends on exact odc-geo/pyproj/PROJ behavior (`GeoBox.from_bbox`,
`.buffered()`, the EPSG:6933 transform itself). The discovered ~0.38% NSIDC-vs-PROJ extent
discrepancy ([`01-grid.md`](01-grid.md) §2) is a concrete demonstration of why: an unpinned
environment means the canonical grid's exact extent — and the convolution engine's exact halo
geometry — could silently shift between machines or across time as these libraries are upgraded,
with no record of what actually produced a given Zarr store.

## 4. What runs in parallel, and for how long

Keep both grids' preprocessing pipelines runnable simultaneously (via the `grid:` config switch, §1)
until at least one **full validation cycle** has run and been reviewed: ingest → convolve →
tabularize → a sanity regression reproducing an expected coefficient pattern, on a bounded subset.
Only after that should the legacy EPSG:4326 path be deprecated — not necessarily deleted immediately,
since other analyses in the repo may still depend on it; that call belongs to whoever owns those
analyses, out of scope for this backbone document.

## 5. Dependency-ordered task list (effort sizing is rough — order matters more than the estimates)

1. **Pin the environment** (§3) — `pyproject.toml` or equivalent recording exact `odc-geo`, `pyproj`,
   `rioxarray`, `xarray`, `zarr` versions. *Small, but blocking* — everything downstream depends on
   a known-good geodesy stack.
2. **Canonical EPSG:6933 GeoBox construction** ([`01-grid.md`](01-grid.md) §2) — extent computed
   programmatically from the pinned PROJ database, resolution 1 km, |φ|≤60° clip, cached the way
   `viirs_geobox.pkl` is today. *Small.*
3. **Latitude-band elliptical kernel registry** ([`01-grid.md`](01-grid.md) §6) — band-edge solver,
   kernel construction, numerical verification against known-radius test circles, disk cache. *Medium
   — the ellipse-axis-orientation derivation needs a runnable check, not an assumption.*
4. **Neighbourhood/convolution engine** ([`03-neighbourhood-engine.md`](03-neighbourhood-engine.md))
   — `convolve_discs` core function, halo reads via `GeoBox.buffered(R_max)`, band-boundary handling
   within tiles. *Large — this is the principal new subsystem, and the band-boundary convolution
   pattern is flagged as the piece most likely to have surprising performance characteristics; budget
   time for benchmarking, not just implementation.*
5. **Per-source resampling fixes** ([`04-ingest.md`](04-ingest.md) §1) — per-variable resampling
   override in `SpatialProcessor`, lights switched to area-weighted sum, trace whether `eog.py`
   actually routes through the shared path. *Small–medium.*
6. **Zarr disc-ladder store** ([`02-storage.md`](02-storage.md)) — analysis-ready grid stores, the
   `S_d`/`N_d` store on the 10-radius ladder, chunk-aligned `to_zarr(region=...)` parallel writes
   (a new pattern for this repo's assembly stage). *Medium.*
7. **Assembly integration** — wire the neighbourhood engine's `S_d`/`N_d` output into tabularization,
   `pixel_id` construction against the new grid's tiling, the handoff interface to `src/analysis/`
   ([`04-ingest.md`](04-ingest.md) §6). *Medium.*
8. **Validation subset run** (§4) — one region, one year, full pipeline, sanity check against expected
   coefficient behavior. *Small effort, but a hard gate — nothing in step 9 should start before this
   passes.*
9. **Cutover**: flip `DEFAULT_CRS`, remove dead code (`geobox_patch.py`, `demean_columns`/`assemble
   demean` wiring, §2), deprecate the legacy path per the criteria in §4.

Steps 1–3 can proceed in parallel with early work on step 4's interface design, but step 4's actual
convolution implementation depends on steps 2 and 3 being settled. Steps 5 and 6 can proceed in
parallel with each other once step 1 is done, since they don't depend on each other. The MODIS LST
ingest itself (companion `PLAN_PROMPT_modis_ingest.md` → `docs/design/07-modis-ingest.md`) is a
separate, parallel-track deliverable that consumes the grid decisions from step 2 but is not on this
task list's critical path.

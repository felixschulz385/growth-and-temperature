# 15 — MODIS PREPARE 2002: tile-grid combine bug + separate reproject failures

> **SUPERSEDED (2026-08-28).** Both Part A and Part B are resolved by the
> reproject-then-overlay rearchitecture in `modis_prepare_rework.md` (repo
> root): `src/data/common/prepare/sinusoidal_mosaic.py` reprojects each
> fetched sinusoidal source tile individually onto the output tile's geobox
> and overlays first-wins onto a georegistered NaN canvas, with the driver
> called `reproject=False`. That deletes `_regrid_to_sinusoidal` (the interim
> Part A fix), `_trim_edge_overlap`, `xr.combine_by_coords` and `sel_bbox`
> from the MODIS/GLASS-MODIS PREPARE path. `PROCESSING_VERSION` bumped
> `1-tiled` -> `2-tiled`. This note is kept for the root-cause analysis
> below.

Handoff note. Investigated 2026-08-27 against real HPC data
(`schulz0022@transfer12.scicore.unibas.ch`,
`data_nobackup/raw/modis/21A2/2002/`, 280 source tiles). A production
`data run --step prepare` for `modis_lst_21a2` / year 2002 left 11 units
non-`complete` in
`data_nobackup/prepared/modis/21A2/crs/ease6933/_status/modis_lst_21a2/_status/`.
They split into **two unrelated problems**.

---

## Part A — `combine_by_coords` "duplicate values" / "not monotonic"  ✅ FIX WRITTEN, NOT COMMITTED

### Affected units
- `2002/0003_0002` — `cannot reindex or align along dimension 'y' … duplicate values`
- `2002/0003_0007` — same, `y`
- `2002/0003_0008` — `Resulting object does not have monotonic global indexes along dimension x`

### Root cause
`ModisSource._execute_prepare.raw_getter` (and the GLASS-MODIS mirror)
snapped each source tile's coords with `np.round(coord / res) * res` before
`xr.combine_by_coords`. That snap is unsound:

- A `vNN` tile whose north edge lies on the sinusoidal origin (`v09`'s
  north edge is `y = 0`) has every pixel centre at a *half-integer*
  multiple of the resolution. `np.round` is round-half-to-even, so with
  sub-µm FP wobble in the row spacing it tie-breaks inconsistently and
  folds ~1 row in 5 onto its neighbour's label — **232 of 1200 rows of the
  real `h04v09` (2002) collapse to duplicate `y` labels within that single
  tile**, before any cross-tile merge. `_trim_edge_overlap` can't help
  (the collision is intra-tile).
- The same mechanism on `x` produces a non-monotonic merged axis instead
  of exact duplicates (`0003_0008`).

The three source tiles for `0003_0002` (`h04v09/h04v10/h05v10`) are
actually mutually co-registered to ~1e-7 m already — the drift is purely
FP, introduced by per-tile `geobox=` pinning at FETCH time.

### Fix (implemented locally — review before commit)
New helper `_regrid_to_sinusoidal(ds, path)` replacing the round-snap in
the multi-tile branch of `raw_getter`, in **both**:
- `src/data/sources/modis/source.py`
- `src/data/sources/glass/modis.py`

Behaviour:
1. Parse `(h, v)` from the `hHHvVV` stem of the tile path.
2. Rebuild both axes as `origin + (arange(n) + 0.5) * res`, with
   `origin`/`res` from `modis_util.tile_bounds_m(h, v)` and
   `TILE_SIZE_M / n` — the LP DAAC global sinusoidal grid definition
   ("compute, don't hardcode", per `modis/tiles.py`).
3. **Guard:** only regrid when the stored coords already sit within ½ px
   of that lattice (the genuine sub-pixel-drift case). Otherwise fall back
   to the old per-axis rounding — keeps the synthetic-fixture tests
   (`hHHvVV`-named 8×8 EPSG:4326 tiles) working and never moves real data
   by a whole pixel.

Stale explanatory comments above `year_tile_index` in both files updated.

### Tests
- New regression test
  `tests/data/sources/modis/test_modis_prepare.py::test_regrid_to_sinusoidal_fixes_round_half_to_even_collapse_at_grid_origin`
  — reproduces the y=0-edge collapse, asserts the regrid yields unique +
  monotonic coords and a clean merge.
- `tests/data/sources/{modis,glass} + tests/data/common/prepare` → 112 pass.

### Verified on HPC (real 2002 tiles, replaying raw_getter's combine)
| unit | source tiles | result |
|---|---|---|
| `2002/0003_0002` | h04v09/h04v10/h05v10 | merged 2400×2400, y & x monotonic, no dups ✅ |
| `2002/0003_0007` | 6 tiles | ✅ |
| `2002/0003_0008` | 9 tiles | ✅ (x now monotonic) |
| `2002/0001_0009` | 12 tiles | combine ✅ (fails later — see Part B) |

### TODO for Part A
- [ ] Review the diff (`git diff src/data/sources/modis/source.py src/data/sources/glass/modis.py tests/…`).
- [ ] Decide: keep the rounding fallback, or make non-sinusoidal input a hard error (only fixtures hit it).
- [ ] Commit on `rework`.
- [ ] **HPC working tree** (`…/growth-and-temperature`, branch `rework`, was clean at `8d9af2b`) currently has the two patched source files scp'd in uncommitted, for the verification run. After committing, `git checkout` them there and `git pull`, or just let the commit propagate.
- [ ] Re-run PREPARE for 2002 and confirm `0003_0002/0003_0007/0003_0008` reach `complete`.

---

## Part B — downstream `process_tile_region` / `xr_reproject` failures  ❌ NOT STARTED

These fail *after* `raw_getter` returns (or in `raw_getter`'s NaN-fill
fallback). **Not touched by Part A.** The driver only records the generic
string `"process_tile_region failed"` in the status sidecar; the real
tracebacks (from the SLURM run log, 2026-08-27) are below. Four distinct
signatures:

### B1 — `Can not reproject non-georegistered array`
- **Units:** `2002/0002_0002`, and almost certainly `2002/0006_0000`,
  `0006_0001`, `0006_0002`, `0006_0010`, `0006_0014` (all have **0
  overlapping source tiles** — confirmed).
- **Traceback:** `spatial.py:483 xr_reproject` → `_xr_reproject_ds` →
  `ValueError("Can not reproject non-georegistered array.")`.
- **Cause:** the "no overlapping tiles" branch of `raw_getter`
  (`modis/source.py` ~line 957, mirrored in `glass/modis.py`) returns
  `xr.Dataset({var: ((dim_y, dim_x), np.full(tile.geobox.shape, nan))})`
  with **no x/y coords and no CRS**. `process_tile_region` then calls
  `xr_reproject` on it, which needs a georegistered array.
- **Fix direction:** either (a) attach `tile.geobox`'s coords + CRS to the
  NaN-fill dataset so it's a valid identity-reproject, or (b) have the
  driver / `process_tile_region` detect an already-on-`tile.geobox`
  all-NaN result and region-write it straight through without reprojecting.
  (a) is smaller and local to `raw_getter`. Same fix needed in both
  sources. Note the other "NaN-fill" sources (ecoregions/gadm/snl_mining
  `_rasterize_tile`) — check how they avoid this; the convention comment in
  `raw_getter` cites them.
- [ ] Decide (a) vs (b); implement in `modis/source.py` + `glass/modis.py`.
- [ ] Test: a canonical tile with no source coverage for a year → NaN tile written, unit `complete`.

### B2 — `IllegalArgumentException: Points of LinearRing do not form a closed linestring`
- **Unit:** `2002/0005_0002` — single source tile `h11v12` (far-south, high-v).
- **Traceback:** `spatial.py:483 xr_reproject` → `_xr_reproject_da` →
  `dask_rio_reproject` → `gbt_dst.grid_intersect(gbt_src)` →
  our `_patched_grid_intersect` (`src/data/common/geobox/geobox.py:114`) →
  `geobox.tiles` → `range_from_bbox` → `_gbox.project(bbox.polygon)` →
  `box()` → `polygon()` → `ValueError: … LinearRing … not … closed`.
- **Hypothesis:** the clipped `h11v12` source region (after `sel_bbox` to
  this tile's bbox) degenerates — zero-width/zero-height, or a
  bbox whose sinusoidal→EASE projected footprint collapses to <4 distinct
  points, so `box(*bbox)` builds an invalid ring. `0005_0002` is a
  bottom-of-grid tile (row 5 of 7); `h11v12` barely clips it.
- **Where to look:** `sel_bbox` result dims for this unit; the bbox handed
  to `_patched_grid_intersect`; whether `_patched_grid_intersect` (ours)
  should guard against a degenerate `src_footprint`.
- [ ] Reproduce: run `raw_getter(tile(5,2), 2002)` on HPC, inspect
  `clipped.sizes` and coords before it's returned.
- [ ] Guard: `raw_getter` should return the B1-style NaN tile (once B1 is
  fixed properly) when `clipped` has an empty/1-px axis, instead of
  handing a degenerate array to reproject. There's already a
  `clipped.sizes.get("x",0)==0 or ...==0` check — extend it to `< 2`, and
  make sure the fallback it returns is georegistered (ties into B1).

### B3 — `affine.TransformNotInvertibleError: Cannot invert degenerate transform`
- **Unit:** `2002/0001_0009` — 12 source tiles
  (`h18–h21 × v05–v07`); combine now succeeds (Part A) but the merged →
  `sel_bbox`-clipped `source_ds` has a degenerate transform.
- **Traceback:** `spatial.py:483 xr_reproject` → `_xr_reproject_ds` →
  `assert isinstance(src.odc, ODCExtensionDs)` triggers `_locate_geo_info`
  → `_extract_transform` → `approx_equal_affine` → `~a * b` → invert of a
  degenerate affine (a scale term is 0 → size-1 axis).
- **Hypothesis:** same family as B2 — `sel_bbox` yields a 1-row or 1-col
  strip for this tile, so the derived affine has `sy == 0` (or `sx == 0`)
  and can't be inverted. `0001_0009` is row 1 (near north edge of the
  EASE grid) × col 9.
- **Fix direction:** same guard as B2 — reject `clipped` with any axis
  `< 2` in `raw_getter` and return a georegistered NaN tile. Fixing B1
  properly (georegistered fallback) + the `< 2` guard likely closes B2 and
  B3 together.
- [ ] Confirm `clipped.sizes` for `tile(1,9)/2002` is degenerate.

### B4 — (subsumed) `2002/0003_0008` "not monotonic x"
- Already fixed by **Part A** (`_regrid_to_sinusoidal`). Listed here only
  because it shared the `process_tile_region`-adjacent status string in
  the sidecar; its real failure was in `raw_getter`'s combine. No separate
  work.

### Cross-cutting for Part B
- B1/B2/B3 all reduce to: **`raw_getter` must always return a fully
  georegistered dataset on `tile.geobox` (coords + CRS), and must treat a
  degenerate clip (any axis < 2 px) as "no coverage" rather than passing
  it downstream.** One helper — "build NaN tile on `tile.geobox`" used by
  every early-return — plus one `< 2` guard, in both `modis/source.py` and
  `glass/modis.py`, probably clears all of Part B.
- [ ] Design that shared fallback helper; wire all early-returns through it.
- [ ] Add regression tests: (i) year with zero source tiles for a tile,
  (ii) tile clipped to a 1-px sliver by its only source tile.
- [ ] Re-run 2002 PREPARE end-to-end; all 11 units → `complete`.

---

## Quick reference — reproducing on HPC
- SSH: `ssh -i ~/.ssh/id_ed25519_scicore schulz0022@transfer12.scicore.unibas.ch`
  (transfer node drops idle connections fast — use `-o ServerAliveInterval=15`
  and run work via `nohup … &` + poll the logfile).
- Python: `/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/src/bin/python`
- Project: `/scicore/home/meiera/schulz0022/projects/growth-and-temperature`
- Canonical geobox: `data_nobackup/canonical_geobox.pkl`
  (`get_or_create_canonical_geobox`), grid `Shape2d(x=34736, y=12704)`
  EPSG:6933, tile grid `Shape2d(x=17, y=7)` at `tile_size=2048`.
- Unit id `YYYY/ROW_COL` → `tiling.iter_tiles(target_geobox, tile_size=2048)`
  with `Tile.row/.col`.

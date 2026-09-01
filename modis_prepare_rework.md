# MODIS / GLASS-MODIS PREPARE — reproject-then-overlay rearchitecture

## Context

A production `data run --step prepare` for `modis_lst_21a2` / year 2002 left 11 of 119
output-tile units non-`complete`, across **four distinct failure modes**
(`docs/design/15-modis-prepare-2002-tile-failures.md`):

| unit(s) | error | cause |
|---|---|---|
| `2002/0003_0002`, `0003_0007` | `cannot reindex or align along dimension 'y' … duplicate values` | `xr.combine_by_coords` of MODIS sinusoidal source tiles needs bit-equal coordinate labels; `np.round(coord/res)*res` pixel-snap collapses ~232/1200 rows of a tile whose north edge is the sinusoidal origin (round-half-to-even on half-integer coords) |
| `2002/0003_0008` | `Resulting object does not have monotonic global indexes along dimension x` | same mechanism on x |
| `2002/0002_0002`, `0006_0000/01/02/10/14` | `Can not reproject non-georegistered array` | the "no overlapping source tiles" branch returns `xr.Dataset({var: ((dim_y,dim_x), np.full(...))})` — **no x/y coords, no CRS**; `xr_reproject` in `process_tile_region` rejects it |
| `2002/0005_0002` | `IllegalArgumentException: Points of LinearRing do not form a closed linestring` | `sel_bbox` clips a far-south single-tile input to a near-degenerate sliver; the **dask** `xr_reproject` path calls `GeoboxTiles.grid_intersect` (repo-monkeypatched, `src/data/common/geobox/geobox.py:67`) whose footprint intersection throws on degenerate geometry |
| `2002/0001_0009` | `affine.TransformNotInvertibleError: Cannot invert degenerate transform` | `sel_bbox` leaves a 1-px strip → derived affine has a zero scale term |

The interim fix (`_regrid_to_sinusoidal`, currently **uncommitted** in the working tree and
scp'd onto the HPC checkout) only patches the first two rows. The root cause is common to all
four: the architecture **mosaics every overlapping source tile in the MODIS sinusoidal CRS,
clips, then reprojects the mosaic onto the output tile** — which forces `combine_by_coords`
(bit-exact coords) and hands malformed datasets to `xr_reproject`.

**This plan replaces it with: `.compute()` each overlapping source tile, `xr_reproject` it
independently onto the output tile's own `tile.geobox`, then overlay first-wins onto a
georegistered NaN canvas.** After reprojection every contribution is on the identical target
lattice by construction — no coordinate matching, no `combine_by_coords`,
`_trim_edge_overlap`, `_regrid_to_sinusoidal`, or `sel_bbox`. All four failure modes are
eliminated structurally and ~150 lines of fragile coordinate code are deleted.

### Verified against installed `odc.geo` 0.5.0 (this session)

- `xr_reproject(src, how)` with `how` a `GeoBox` → output grid **is** that GeoBox verbatim
  (`_xr_interop.py:904`).
- **numpy-backed src → `xr_reproject` makes 0 `grid_intersect` calls** (uses `rio_reproject`
  directly, `_xr_interop.py:987`); **dask-backed src → 1 call** into the monkeypatched
  `_patched_grid_intersect` (B2's traceback). ⇒ `.compute()` each source tile before
  reprojecting to bypass B2 entirely.
- `rep = xr_reproject(src, tile.geobox, dst_nodata=np.nan)` returns a dataset whose `y`/`x`
  coords are **identical** to `xr_zeros(tile.geobox)` — same length, values, and descending
  order. ⇒ `canvas.combine_first(rep)` aligns with **no reindex and no re-sort**, `geobox`
  identity preserved. (Verified empirically; `combine_first` only re-sorts when indexes
  differ, e.g. against a *sub*-window — so warp onto the **whole** `tile.geobox`, never a
  crop.)
- `xr.Dataset({v: xr.full_like(xr_zeros(geobox, "float32"), np.nan) for v in vars})` is
  georegistered (`.odc.geobox == geobox`), float32, dims auto-named per CRS (`y`/`x` for
  EASE6933, `latitude`/`longitude` for legacy 4326). This is both the overlay base and the
  whole return value when there is no coverage — fixes B1.
- `process_tile_region(reproject=False)` (`spatial.py:485-504`) never inspects source coords
  or CRS — only that each var is exactly `tile.geobox.shape`. Precedent:
  `test_driver.py::test_run_tiled_prepare_reproject_false_uses_raw_getter_output_as_is`.

---

## Design

### 1. Move reprojection out of the driver, into `raw_getter`

Both `_execute_prepare`s call `run_tiled_prepare(...)` **without** `reproject=` (defaults
`True`), so `process_tile_region` (`spatial.py:421`) runs `xr_reproject(source_ds,
tile.geobox)` — the call where B1/B2/B3 crash. Change both calls to **`reproject=False`** and
**drop `resampling=` / `dst_nodata=`** from them (dead under `reproject=False`).
`process_tile_region` then only tabulates (`encode_cell_ids` + flatten + `part-<year>.parquet`).
`reproject=False` is already used by `gadm`/`osm`/`ecoregions`/`snl_mining`/`berman_mining`.

`raw_getter` now returns a dataset **already on `tile.geobox`** — exact shape, native
(y-descending) order, georegistered.

### 2. New `raw_getter` flow (no sub-window — warp each whole source tile onto the whole tile)

Keep unchanged (cheap and correct):
- `year_tile_index(year)` — per-year `[(path, rasterio_bounds)]` header index + `source_crs`
  from the first tif. Keep `sorted(os.listdir(...))` → deterministic first-wins overlay.
- `read_source_tile(path, year)` + bounded LRU — **but add `.compute()`** (see §3) and set
  cache size **8** (materialized arrays now).
- the doc-13 antimeridian-wrap **bbox clamp**: `tile.geobox.pad(32,32).extent` clamped into
  `target_geobox.extent.boundingbox` with `margin = 2*abs(target_geobox.resolution.x)`,
  `box(...)`, `.to_crs(source_crs).boundingbox`, unclamped fallback if the margin degenerates.
  Still the mechanism that picks *which* source tiles matter (prevents the 104/282 edge-tile
  over-match). The 32-px pad only widens *selection* so an edge output pixel whose nearest
  source pixel sits just past a tile's exact extent still finds that tile.
- the `overlapping` header-bounds intersection filter.

Replace the `len(datasets)==1 / else combine_by_coords` block **and** the `sel_bbox` clip
**and** both non-georegistered NaN-fill returns with:

```
raw_getter(tile, year):
    index, source_crs = year_tile_index(year)
    if not index:
        return None                                  # genuine "FETCH not done" -> retryable
    bbox   = clamped_padded_extent(tile).to_crs(source_crs).boundingbox
    paths  = [p for p, b in index                    # sorted order
              if b.right >= bbox.left and b.left <= bbox.right
              and b.top  >= bbox.bottom and b.bottom <= bbox.top]
    vars_  = list(read_source_tile(index[0][0], year).data_vars)     # cheap, via LRU
    base   = xr_zeros(tile.geobox, "float32")                        # georegistered
    canvas = xr.Dataset({v: xr.full_like(base, np.nan) for v in vars_})
    for p in paths:
        src = read_source_tile(p, year)              # numpy (computed), ALL bands
        try:
            rep = xr_reproject(src, tile.geobox, resampling="nearest", dst_nodata=np.nan)
        except Exception:
            logger.exception("%s: reproject failed for %s onto tile %s; skipping", LABEL, p, tile.id)
            continue
        canvas = canvas.combine_first(rep)           # first non-null wins (canvas starts all-NaN)
    return canvas                                    # ALWAYS exactly tile.geobox, georegistered
```

**Only one return path** other than `index == []`:
- `paths == []` → loop skipped → pristine NaN canvas → **B1 gone**, unit recorded `complete`
  (matches the old NaN-fill intent, not a retryable failure).
- a source tile whose warp raises or yields all-NaN contributes nothing via `combine_first` →
  **B2/B3 gone**. No `footprint()` / `overlap_roi` / manual affine math — the earlier
  sub-window idea is dropped because `GeoBox.footprint()` on a high-`v`/antimeridian tile is
  exactly the degenerate ring that caused B2, and a numpy warp onto ≤2048² is already
  destination-bounded (GDAL restricts to the real overlap internally).
- **first-wins** matches the current order-dependent `_trim_edge_overlap` claim; on this grid
  source tiles only share a 0–1 px seam carrying the same physical pixel, so first-vs-mean is
  sub-pixel cosmetic. First-wins is cheaper and deterministic.
- **`nearest` needs no destination halo.** Known limitation to document: a future
  bilinear/cubic/`sum` kernel would want source tiles mosaicked *before* warp at the
  inter-tile seam; all current MODIS/GLASS-MODIS vars are continuous physical quantities so
  `nearest` is correct and no `mode`/`sum` path is needed.

### 3. `.compute()` source tiles eagerly

In `read_source_tile`, after the squeeze, `ds = ds.compute()` before caching. Rationale:
- removes the dask `_xr_reproject_da` → `dask_rio_reproject` → `grid_intersect` →
  `_patched_grid_intersect` stack that is B2's entire traceback;
- one MODIS source tile = 1200²·(1 band for main `lst_night_mean` … ~7 for extended)·4 B ≈
  6–40 MB; `.compute()` is sub-second;
- worst case = a pole output tile ≈ 37 source tiles, consumed **one at a time** in the loop:
  peak ≈ LRU(8) + 1 in-flight compute + 1 full-`tile.geobox` warp output/var (2048²·4 B =
  16 MB) — far below the doc-13 OOM regime (whole-year mosaics);
- serial driver loop → no cross-unit multiplier.

Cache size 8 (down from 16) covers a 3×3 output-tile neighbourhood for cross-tile reuse; pole
units thrash and re-read next unit (I/O only, few pole tiles). Keep it a parameter.

### 4. Deletions

| symbol | files | note |
|---|---|---|
| `_regrid_to_sinusoidal` | `modis/source.py:127-172`, `glass/modis.py:109-136` | uncommitted — drop; also revert on HPC checkout |
| `_trim_edge_overlap` | `modis/source.py:81-124`, `glass/modis.py:72-106` | committed (`1801056`) — remove fn + its test |
| `xr.combine_by_coords` block | both `raw_getter`s (`modis/source.py:1034-1045`, `glass/modis.py:1037-1048`) | replaced by §2 |
| `sel_bbox(...)` call + degenerate-clip guard | both `raw_getter`s (`modis/source.py:1049-1056`, `glass/modis.py:1052-1059`) | `sel_bbox` **stays defined** in `spatial.py:58-81` (acag/esacci/eog/ntl_harm use it); drop its import from the two source files |
| non-georegistered NaN-fill returns (2/file) | both `raw_getter`s | replaced by the georegistered `canvas` |

Keep `from odc.geo.geom import box` in both `_execute_prepare` (clamp still uses it). Refresh
the stale comments above `year_tile_index` (`modis/source.py:887-901`, `glass/modis.py:922-928`)
and the "stays dask-backed … only `.compute()`s after clipping" claim in both
`_read_annual_geotiff` docstrings (`modis/source.py:843-849`, `glass/modis.py:879-885`).

### 5. `PROCESSING_VERSION` bump (forces full reprocess)

- `ModisSource.PROCESSING_VERSION` `"1-tiled"` → `"2-tiled"` (`modis/source.py:244`)
- `GlassModisSource.PROCESSING_VERSION` `"1-tiled"` → `"2-tiled"` (`glass/modis.py:335`)

`run_tiled_prepare` reprocesses any unit whose stored `processing_version` differs
(`driver.py:164-167`); parquet parts overwrite in place; marker mtime bumps once. No manual
status-file deletion needed. Covered by `test_driver.py::test_processing_version_bump_forces_reprocessing`.

### 6. Shared-helper extraction (recommended)

The two raw-getter closure families (`year_tile_index` + `read_source_tile` + `raw_getter` +
clamp, ~110 lines) are verbatim copies — Part A had to be fixed twice because of it. Extract to
**`src/data/common/prepare/sinusoidal_mosaic.py`**:

```python
"""raw_getter factory for sources that mosaic a fixed sinusoidal source-tile grid
(MODIS 21A2/11A1, GLASS-MODIS) onto the shared output tile grid.

Builds the raw_getter(tile, year) that run_tiled_prepare(..., reproject=False) calls per
output unit: pick source tiles overlapping the output tile (header-only bbox test), read +
.compute() each, xr_reproject each onto tile.geobox (numpy -> rio_reproject: no dask, no
grid_intersect, no monkeypatch), overlay first-wins onto a NaN canvas. Return value is ALWAYS
exactly tile.geobox (georegistered, float32, one var per source band) -- no non-georegistered
early-return, so process_tile_region never sees a malformed source (docs/design/15 B1/B2/B3)."""

from collections import OrderedDict
from typing import Any, Callable, Optional
import logging, numpy as np, xarray as xr
from odc.geo.geom import box
from odc.geo.xr import xr_reproject, xr_zeros

logger = logging.getLogger(__name__)

def build_sinusoidal_mosaic_raw_getter(
    *,
    stage1_root: str,
    target_geobox,
    read_annual_geotiff: Callable[[str, int], xr.Dataset],   # source's own staticmethod
    log_label: str = "sinusoidal-mosaic",
    source_tile_cache_size: int = 8,
    pad_pixels: int = 32,
) -> "Callable[[Any, Optional[int]], Optional[xr.Dataset]]":
    tile_index_cache: dict[int, tuple[list[tuple[str, Any]], Any]] = {}
    source_tile_cache: "OrderedDict[str, xr.Dataset]" = OrderedDict()

    def year_tile_index(year): ...          # unchanged: one rasterio header-open per .tif
    def clamped_extent(tile):  ...          # unchanged doc-13 antimeridian clamp
    def read_source_tile(path, year):
        # LRU; read_annual_geotiff -> squeeze time/band -> .compute() -> cache (cap size)
        ...
    def raw_getter(tile, year):
        # the §2 body
        ...
    return raw_getter
```

- `read_annual_geotiff` is the only real divergence (MODIS skips `_monthly_` bands, GLASS skips
  only empty names) — leave both `_read_annual_geotiff` staticmethods where they are, pass one.
- `tile_size` is **not** a parameter — the driver owns tiling; the raw-getter only sees a
  `tile` with a ready `tile.geobox`.
- The helper must **not** close over a dask client. Each `_execute_prepare` keeps its own
  wiring (resolve geobox, build `SpatialProcessor`, `with self._dask_client()`,
  `run_tiled_prepare(..., reproject=False, processing_version="2-tiled")`); only the closure
  family moves.

MODIS caller: `target_geobox = get_or_create_canonical_geobox(<data_root>/canonical_geobox.pkl)`,
`read_annual_geotiff=ModisSource._read_annual_geotiff`, `log_label="MODIS"`.
GLASS caller: `target_geobox = get_target_geobox(self.ctx)` (grid_id-aware, may be legacy
EPSG:4326), `read_annual_geotiff=GlassModisSource._read_annual_geotiff`, `log_label="GLASS-MODIS"`.

If extraction feels too risky in one change, fall back to identical in-place edits in both
files and keep the factory as a follow-up — but the factory is the clean end state.

---

## Bug resolution mapping

| bug | fixed by |
|---|---|
| A1/A2 (`duplicate values` / `not monotonic`) | no `combine_by_coords` — each source tile reprojected independently onto the shared output lattice |
| B1 (`non-georegistered array`) | `raw_getter` always returns the `xr_zeros(tile.geobox)`-based canvas; `reproject=False` means `process_tile_region` never calls `xr_reproject` anyway |
| B2 (`LinearRing not closed`) | source tiles `.compute()`d → numpy → `xr_reproject` skips `grid_intersect`/`_patched_grid_intersect`; a failing warp is caught + skipped |
| B3 (`Cannot invert degenerate transform`) | no `sel_bbox` sliver; the warp target is always the full `tile.geobox` |

---

## Files to modify

- `src/data/common/prepare/sinusoidal_mosaic.py` — **new** (§2, §6).
- `src/data/sources/modis/source.py` — delete `:81-124` + `:127-172`; rewrite `_execute_prepare`
  `:869-1083` to use the factory + `reproject=False` (drop `resampling=`/`dst_nodata=`); bump
  `PROCESSING_VERSION` `:244`; refresh comments `:843-849`, `:887-901`.
- `src/data/sources/glass/modis.py` — delete `:72-106` + `:109-136`; rewrite `_execute_prepare`
  `:904-1085`; bump `PROCESSING_VERSION` `:335`; refresh comments `:879-885`, `:922-928`.
- `src/data/common/raster/spatial.py` — **docstring only** on `process_tile_region` `:421-472`
  (note MODIS/GLASS-MODIS now arrive `reproject=False`). No code change; `sel_bbox` stays.
- `src/data/common/prepare/driver.py` — no change.
- `docs/design/15-modis-prepare-2002-tile-failures.md` — mark Part A/B superseded; link this plan.

---

## Test plan

### `tests/data/sources/modis/test_modis_prepare.py`
- **DELETE** `_regrid_to_sinusoidal`, `_trim_edge_overlap` from the import (line 18 → keep only
  `ModisSource`); **DELETE** `test_trim_edge_overlap_resolves_real_multi_pixel_boundary_overlap`
  (`:167`) and `test_regrid_to_sinusoidal_fixes_round_half_to_even_collapse_at_grid_origin`
  (`:198`).
- **KEEP** `test_execute_prepare_writes_real_tiled_parquet_output` (`:40`),
  `test_execute_prepare_reuses_one_years_mosaic_at_a_time` (`:245`, rename `…_reuses_one_year_index…`),
  `test_clamped_bbox_avoids_antimeridian_wrap_at_real_grid_corner_tile` (`:113`, unchanged).
- **ADAPT** `test_execute_prepare_handles_slightly_misaligned_adjacent_tiles` (`:71`) → keep the
  two-overlapping-fixtures scenario + `np.all(np.isfinite(...))`; rewrite docstring ("each
  source tile is reprojected onto the output grid then overlaid, so sub-pixel misalignment
  between source tiles cannot collide"); also assert the overlap region == the first
  (lexicographic) tile's value.
- **ADD**:
  - `…_tile_with_no_source_coverage_writes_nan_tile_and_completes` — source `.tif` covers only
    part of the fake grid → ≥1 output tile has zero overlapping source tiles; assert 4 parts,
    marker, `is True`, that tile's column all-NaN, its status `complete`. **(B1)**
  - `…_source_tile_barely_clips_output_tile_writes_nan_not_error` — sub-pixel overlap → no
    exception, part written, unit `complete`. **(B2/B3)**
  - `…_overlapping_source_tiles_first_wins` — two constant-value overlapping fixtures → overlap
    pixels == first tile's value.

### `tests/data/sources/glass/test_glass_modis_prepare.py`
- **KEEP** `test_execute_prepare_threads_ctx_grid_id_and_writes_real_tiled_parquet` (`:45`),
  `test_clamped_bbox_avoids_antimeridian_wrap_…` (`:116`).
- **ADAPT** `test_execute_prepare_handles_slightly_misaligned_adjacent_tiles` (`:79`) — as MODIS.
- **ADD** GLASS B1 + B2 equivalents (8-band `_write_tile_tif`), and
  `…_legacy_4326_grid_produces_georegistered_nan_canvas` — default `grid_id="legacy_4326"` +
  partial coverage → `complete` + NaN tile (canvas dims `latitude`/`longitude`).

### `tests/data/common/prepare/test_driver.py`
- **KEEP** `test_run_tiled_prepare_reproject_false_uses_raw_getter_output_as_is` (`:195`) — now
  the primary contract for these sources.
- **ADD** `…_reproject_false_accepts_georegistered_nan_canvas` (raw_getter returns
  `xr.full_like(xr_zeros(tile.geobox,"float32"), np.nan)` per var → `h*w` rows, NaN round-trips)
  and `…_reproject_false_never_calls_xr_reproject` (monkeypatch `spatial.xr_reproject` to raise
  → run still succeeds).

### `tests/data/common/raster/test_process_tile_region.py`
- **ADD** `…_reproject_false_tabulates_georegistered_nan_canvas`.

### NEW `tests/data/common/prepare/test_sinusoidal_mosaic.py`
Parametrised over a small fake EASE6933 geobox and a small fake EPSG:4326 geobox:
- `test_no_overlap_returns_nan_canvas_on_tile_geobox` (shape == `tile.geobox.shape`,
  `.odc.geobox` set, one var/band, all NaN)
- `test_single_source_tile_reprojects_onto_output_grid` — **genuine** small sinusoidal fixture
  (`GeoBox.from_bbox(modis_util.tile_bounds_m(h,v), crs=modis_util.SINUSOIDAL_PROJ4, resolution=…)`)
  → constant lands on the EASE canvas where covered, NaN elsewhere (only CI coverage of the
  real cross-CRS path; `_write_tile_tif` fixtures are same-CRS)
- `test_multiple_overlapping_tiles_first_wins`
- `test_source_tiles_are_materialised_not_dask` (`canvas.chunks is None`; spy the `.compute()`)
- `test_lru_never_exceeds_configured_size`
- `test_reproject_failure_on_one_source_tile_is_skipped_not_raised`
- `test_empty_year_index_returns_none`
- `test_ragged_tile_shape_preserved`

### Fixtures reused unchanged
`_make_source`, `_write_tile_tif(path, value, band_names, bounds=(-1,-1,1,1), size=8)` (EPSG:4326
`hHHvVV.tif`), canonical-geobox monkeypatch
(`GeoBox.from_bbox((-1,-1,1,1), crs="EPSG:4326", resolution=0.5)` + `source.tile_size = 2`). The
synthetic fixtures no longer need to be real sinusoidal (no `tile_bounds_m` parsing).

### Run
```
pytest tests/data/sources/modis/test_modis_prepare.py \
       tests/data/sources/glass/test_glass_modis_prepare.py \
       tests/data/common/prepare/ tests/data/common/raster/ -q
```

---

## Rollout & HPC verification

1. **One commit on `rework`**: add `sinusoidal_mosaic.py`; both source edits (deletions +
   `_execute_prepare` rewrite + version bump + comment refresh); `spatial.py` docstring; tests.
2. **HPC checkout hygiene**: the HPC tree
   (`schulz0022@transfer12.scicore.unibas.ch:~/projects/growth-and-temperature`, branch
   `rework`, was clean at `8d9af2b`) has the two `_regrid_to_sinusoidal`-patched source files
   scp'd in uncommitted. `git -C <path> checkout -- src/data/sources/modis/source.py
   src/data/sources/glass/modis.py`, then `git pull`. The interim fix is **superseded** — its
   whole `combine_by_coords` path is deleted — discard, don't reconcile.
3. **Cache-bust** is automatic via the `PROCESSING_VERSION` bump.
4. **HPC verification** before the full rerun (env python
   `/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/src/bin/python`; SSH note: `transfer12`
   drops idle connections — `nohup … &` + poll a logfile, `-o ServerAliveInterval=15`):
   - **All 11 failing 2002 units** reach `complete`, each part exactly `h*w` rows:
     `0002_0002 0006_0000 0006_0001 0006_0002 0006_0010 0006_0014` (B1),
     `0005_0002` (B2, `h11v12`), `0001_0009` (B3, 12 tiles),
     `0003_0002 0003_0007 0003_0008` (Part A). Scope with `override=True` for 2002 or delete
     just those units' status sidecars.
   - **Pole tile**: a `row=0` or `row=6` tile with ~30+ overlapping source tiles — time it,
     watch RSS (expect a few hundred MB); non-empty over land, NaN elsewhere.
   - **Antimeridian / grid-edge**: `0000_0000` (col 0) and a `col=16` tile — assert `paths`
     stays `< 30` (the `test_clamped_bbox…` bound, not 104) and the warp is not world-wrapped.
   - **Ragged last row/col**: canonical grid `Shape2d(x=34736, y=12704)` at `tile_size=2048`
     → `17×7` tiles; last col width `34736 − 16·2048 = 1968`, last row height
     `12704 − 6·2048 = 416`. Run one `row=6` and one `col=16` unit; assert
     `canvas.shape == tile.geobox.shape` (not 2048²) and the `values.size == h*w` guard passes.
   - **GLASS-MODIS**: no-coverage + a normal tile, on both `ease6933` and a `legacy_4326`
     `grid_id`.
   - **Regression**: for a non-2002 year that previously succeeded, diff interior-tile pixel
     values old (`"1-tiled"`) vs new — expect bit-identical for tiles fully inside one source
     tile; differences only at source-tile seams (nearest).
5. Full 2002 rerun → all 119 units `complete`, marker written. Then full multi-year reprocess
   of `modis` / `modis_extended` / `glass_modis` / `glass_ta_modis` (all cache-busted).
6. Commit; update `docs/design/15`.

---

## Risks & edge cases

| case | handling |
|---|---|
| Ragged last row/col (`tile.geobox.shape != (2048,2048)`) | Free: canvas = `xr_zeros(tile.geobox)`, warp target = `tile.geobox`, `process_tile_region` uses `tile.geobox` for `encode_cell_ids` + the shape assert. Explicit test added. |
| Pole tiles (~37 source tiles) | Sequential read→compute→warp→`combine_first`→free. Peak ≈ LRU(8) + 1 in-flight + 1 warp output/var. `combine_first` over ~37 datasets ≈ seconds; per-var `np.where(np.isnan(acc), new, acc)` accumulator is the fallback if it profiles hot. |
| Antimeridian source tiles (`h00`/`h35`) | `footprint`/`grid_intersect`/`_patched_grid_intersect` never entered (numpy src). Only antimeridian touch left is `extent.to_crs(source_crs)` in the clamp — already guarded by the retained doc-13 margin clamp + its test. |
| `GeoBox.from_bbox` FP drift at FETCH (`modis/source.py:620-624`, original root cause) | Now irrelevant — each source tile reprojected by its own georeferencing; sub-pixel drift → reprojected footprints differ by `<1` dst px, resolved by first-wins. No coordinate-label equality anywhere. |
| GLASS native-grid tiles (no FETCH `geobox=` pin) | New flow uses each tif's own transform/CRS as authoritative — strictly better than the old `_regrid_to_sinusoidal` guard that bailed for these. |
| GLASS legacy EPSG:4326 target grid | Canvas dims → `latitude`/`longitude`; `xr_reproject(src, tile.geobox)` outputs on `tile.geobox`; `process_tile_region` is dim-name-agnostic. Test added. |
| `combine_first` re-sort trap | Safe **only** because every warp targets the same `tile.geobox` object → object-identical coords. Never warp to CRS+resolution or a sub-window and `combine_first`. |
| `dst_nodata=NaN` + dtypes | All MODIS/GLASS-MODIS PREPARE vars are float32 physical quantities → NaN nodata correct, parquet preserves NaN. |
| categorical vs continuous | All continuous → `nearest` correct; no `mode`/`sum` path needed. |
| LRU now holds computed arrays | 0 → ~50–320 MB, bounded; `source_tile_cache_size` param, default 8. |
| Determinism | `year_tile_index` keeps `sorted(os.listdir())` → stable `hHHvVV.tif` order → reproducible first-wins. |
| `index == []` | `raw_getter` returns `None` → retryable failure — correct ("FETCH not done", distinct from "tile outside coverage"). |
| Same-CRS test fixtures | `_write_tile_tif` + `fake_geobox` are both EPSG:4326 → near-identity warp. Genuine sinusoidal→EASE covered by the new `test_sinusoidal_mosaic.py` fixture + HPC step 4. |
| Extraction risk | If `sinusoidal_mosaic.py` destabilizes, fall back to identical in-place edits in both source files; factory as follow-up. |

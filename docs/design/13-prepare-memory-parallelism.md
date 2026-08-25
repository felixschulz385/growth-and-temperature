# 13 — PREPARE Memory & Parallelism

Live checklist, same convention as [`07b-modis-outstanding.md`](07b-modis-outstanding.md): update in
place as items get resolved, don't treat this as a session-scoped writeup.

Triggered by a real SLURM OOM kill on a MODIS PREPARE job (2026-08-24, `StepId=21266004.batch`, 4
dask workers at 38GiB each configured, driver process killed anyway). Root-causing that surfaced two
patterns worth checking across every PREPARE-capable source, not just MODIS:

1. **Eager whole-domain load before per-tile clip** — a source's per-unit read materializes far more
   than one output tile's extent needs before `SpatialProcessor.process_tile_region` clips/reprojects
   it down. MODIS's version of this was worst-in-class: `xr.combine_by_coords`-ing *every* fetched
   tile for a year into one in-memory mosaic, repeated per canonical PREPARE tile.
2. **Dask client created but never actually driving distributed compute** — `self._dask_client()`
   spins up a `LocalCluster` and reserves ~90% of the SLURM job's memory across its workers
   (`src/cli/data/slurm.py:94`, `src/data/common/dask/client.py:88`), but
   `run_tiled_prepare`'s per-unit loop (`src/data/common/prepare/driver.py:141`) is single-process and
   serial — nothing ever gets `client.submit`/`.map`'d to it, so the reserved workers sit idle for the
   whole run.

## Resolved

- [x] **MODIS / GLASS-MODIS eager whole-year mosaic OOM — fixed (2026-08-24).**
      `_execute_prepare` in [`modis/source.py`](../../src/data/sources/modis/source.py) and
      [`glass/modis.py`](../../src/data/sources/glass/modis.py) used to build one full-year mosaic
      from every fetched source tile before clipping to each canonical PREPARE tile. Replaced with a
      per-year index of `(file, bounds)` built from a cheap `rasterio.open()` header read (no pixel
      data touched), so only the handful of source tiles that actually overlap a given canonical
      tile's bbox are ever read.
- [x] **MODIS / GLASS-MODIS idle dask cluster — fixed (2026-08-24).** New
      [`modis/parallel_prepare.py`](../../src/data/sources/modis/parallel_prepare.py):
      `run_tiled_prepare_dask()` reuses `run_tiled_prepare`'s own status/locking/marker primitives but
      submits each `(tile, year)` unit to the dask client via `client.submit`/`as_completed` instead of
      looping serially, so the already-reserved workers do real work. Scoped to the two MODIS-family
      sources only — `src/data/common/prepare/driver.py`'s shared serial loop is untouched, since
      several other sources' `raw_getter` closures (DuckDB connections, GeoDataFrames) aren't safe to
      hand to a separate worker process.

## Resolved (continued)

- [x] **esacci / eog / ntl_harm / acag eager whole-domain load — fixed (2026-08-24).** Each used to
      eagerly `.load()` (or, for acag, open with no `chunks=` at all) the whole year's global raster
      before any per-tile clip. Fixed by adding `sel_bbox()`
      ([`spatial.py`](../../src/data/common/raster/spatial.py)) — a shared helper that slices a
      dataset to a bbox without assuming which direction either coordinate runs (a plain
      `ds.sel(y=slice(top, bottom))` silently returns empty if the axis happens to run the other
      way) — and calling it in each source's `raw_getter` before `.compute()`, matching
      `glass_avhrr`'s already-correct pattern. Two real correctness wrinkles surfaced along the way:
      - `eog`/`ntl_harm`'s `.gz`/`.zip`-wrapped sources decompress to a temp file/dir that the old
        code deleted immediately after `_load_year` returned, relying on the eager `.load()` to have
        already read everything. Made lazy, that delete would race a later `.compute()` reading from
        the now-deleted file. Fixed by having `_load_year` return `(dataset, cleanup_path)` and
        moving cleanup to the caller's year-cache eviction (and a `finally` after the whole PREPARE
        run, so the last year's temp file doesn't leak) — deleted only once every tile for that year
        has been computed.
      - `esacci`/`eog`/`ntl_harm`/`acag`'s `raw_getter` used to always return the whole (unclipped)
        raster, so a target tile falling outside the source's actual coverage still got a real
        (all-nodata-after-reproject) result. A `sel_bbox()` clip can come back empty for that same
        case, so each source NaN-fills on `tile.geobox` instead (matching MODIS/AVHRR's existing
        convention) — except `esacci`, which is a genuinely global product with no legitimate gaps,
        so an empty clip there is logged as an error instead.
      New/extended tests: `tests/data/common/raster/test_sel_bbox.py` (axis-direction-agnostic
      slicing), `tests/data/sources/esacci/test_esacci_prepare.py`,
      `tests/data/sources/eog/test_eog_prepare.py` (including the `.gz` temp-file-survives-to-compute
      case), `tests/data/sources/acag/test_acag_prepare.py`, plus a new zip-wrapped case added to
      `tests/data/sources/ntl_harm/test_ntl_harm_plan.py`.

## Resolved (continued, 2)

- [x] **esacci / eog / ntl_harm / acag idle dask cluster — fixed (2026-08-25).** New
      [`raster_year_parallel.py`](../../src/data/common/prepare/raster_year_parallel.py) —
      `run_tiled_prepare_dask_year_major()`, the "one whole-year source file, clipped per tile" analog
      of `modis/parallel_prepare.py`'s per-unit dask dispatch. Reuses `run_tiled_prepare`'s own
      status/locking/marker primitives; each `(tile, year)` unit is submitted to the dask client, and
      each source's own `_load_*`/`_load_nc_as_dataset` became a plain `@staticmethod` (qualified-
      name-referenced, not a closure or `functools.partial`, so it pickles by reference and every
      worker's deserialized copy hits the same per-worker `functools.lru_cache` — a `partial`-bound
      copy would reconstruct by value and silently defeat that cache). Instance state a `load_year_fn`
      needs (EOG's `source_type`) travels separately via a `load_year_extra` argument instead of being
      bound into the function. Two more correctness wrinkles specific to running this across separate
      *processes* now, not just lazily:
      - `esacci`'s `clip_lon180()` antimeridian handling (previously applied inline in `raw_getter`)
        is now a `clip_antimeridian=True` dispatcher flag, applied inside the worker task.
      - `eog`/`ntl_harm`'s `.gz` decompression used to write to a fixed `local_file[:-3]` path — safe
        with one process, but multiple worker *processes* touching the same year concurrently could
        race writing that same path. Now decompresses to a fresh, PID-unique `tempfile.mkstemp()` path
        per worker instead (never deleted — see the new module's docstring for why: no safe point to
        delete under concurrent in-worker access, bounded by years-touched × worker-count, left for
        job-end scratch cleanup). `ntl_harm`'s `.zip` branch already extracted to a fresh
        `tempfile.mkdtemp()` per call, so no equivalent race existed there.
      Updated tests: `test_esacci_prepare.py`, `test_eog_prepare.py`, `test_acag_prepare.py`,
      `test_ntl_harm_plan.py`'s two `_execute_prepare` tests, and `test_acag_grid_geobox.py`'s spy
      target — all now exercise (or, for the geobox test, monkeypatch) the new dispatcher instead of
      `run_tiled_prepare`, and use a real (small, local) Dask client instead of a `nullcontext` stub
      where `client.submit` is actually invoked.
- [x] **gadm.py:272, ecoregions/source.py:503 — fixed (2026-08-25).** Removed the `self._dask_client()`
      call entirely from both — it was used only for `dashboard_link` logging, and `SpatialProcessor`
      was never constructed with `dask_client=` either; the actual per-tile work (`_rasterize_tile`) is
      pure shapely/numpy and never touched dask. No behavior change, just stops reserving SLURM job
      memory for a cluster that structurally could never be used in this code path.

## Resolved (continued, 3)

- [x] **GDAL's own native block cache, unbounded per worker process — fixed (2026-08-25).** Surfaced
      on the first real production MODIS PREPARE run after `parallel_prepare.py` went live: all 4
      workers repeatedly climbed to ~34GB RSS ("Unmanaged memory... 30GB", worker paused, then the
      nanny killing/restarting it for exceeding 95% of its 38GiB budget), over and over, throughout
      the run — much more than the LRU caches' `maxsize` alone would suggest (a handful of ~50MB MODIS
      tiles). Root cause: `ModisSource`/`GlassModisSource._read_annual_geotiff` opened each source
      GeoTIFF via `rxr.open_rasterio(path, masked=True)` (no `chunks=`) and never closed the underlying
      rasterio/GDAL file handle — it stayed open for as long as `_read_source_tile_cached`'s per-worker
      LRU entry lived. GDAL's block cache defaults to 5% of the *node's total* RAM, computed
      independently by each worker process and never bounded anywhere in this repo (confirmed via
      `grep -rn "GDAL_CACHEMAX\|rasterio.Env" src/` — zero hits) — across the thousands of sequential
      opens one long-lived worker does over a full run, that native (non-Python-tracked) cache
      accumulates and shows up exactly as "unmanaged memory" in worker logs, not a Python-heap leak.
      Fixed two ways: (1) `_read_annual_geotiff` now calls `.load()` on the constructed Dataset and
      explicitly `.close()`s the source `DataArray` before returning, so an LRU eviction actually
      releases the GDAL handle instead of just dropping a Python reference to a still-open one; (2)
      both `modis/parallel_prepare.py` and `raster_year_parallel.py` now set
      `os.environ.setdefault("GDAL_CACHEMAX", "512")` at module-import time (once per worker process,
      before any raster is opened) as a hard ceiling independent of node RAM size — applied to the
      esacci/eog/ntl_harm/acag dispatcher too, since they open rasters/netCDFs from worker processes
      the same way and would very plausibly hit the same failure once run at real production scale.

## Confirmed fine / not applicable

- [x] **glass_avhrr** (`src/data/sources/glass/avhrr.py:704-721`) — `year_ds()` opens via
      `xr.open_zarr` (dask-chunked), `raw_getter` does `.sel(y=..., x=...).compute()` — clips before
      computing, so only the small per-tile+halo region ever materializes. Already the target shape
      for the pattern-1 fixes above.
- [x] **snl_mining, osm** — neither calls `self._dask_client()` anywhere in their PREPARE path, and
      neither builds a full-domain pixel array (`_rasterize_tile` is vector-polygon rasterization
      directly onto one output tile's geobox). No memory or idle-cluster risk to fix.

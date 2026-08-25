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
- [x] **MODIS / GLASS-MODIS idle dask cluster — fixed (2026-08-24), superseded (2026-08-25).** New
      `modis/parallel_prepare.py`: `run_tiled_prepare_dask()` reused `run_tiled_prepare`'s own
      status/locking/marker primitives but submitted each `(tile, year)` unit to the dask client via
      `client.submit`/`as_completed` instead of looping serially. **Superseded** by the "Resolved
      (continued, 4)" entry below — this per-unit task-dispatch design proved unstable on a real
      production run (see that entry's GDAL-cache investigation and the run that ultimately crashed
      with `KilledWorker` after a task died on all 4 workers) and `parallel_prepare.py` has been
      deleted. Left here for history; superseded, not currently in the tree.

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

- [x] **esacci / eog / ntl_harm / acag idle dask cluster — fixed (2026-08-25), superseded
      (2026-08-25).** New `raster_year_parallel.py`: `run_tiled_prepare_dask_year_major()`, the "one
      whole-year source file, clipped per tile" analog of `modis/parallel_prepare.py`'s per-unit dask
      dispatch, submitting each `(tile, year)` unit to the dask client via `client.submit`. **Superseded
      the same day** by the "Resolved (continued, 5)" entry below, once MODIS's real production failure
      (see "Resolved (continued, 3)" and "(continued, 4)") showed this exact per-unit task-dispatch
      shape to be unstable at scale — these four sources hadn't hit that failure themselves yet only
      because none had been run at comparable scale. `raster_year_parallel.py` is deleted; these four
      sources now use the same lazy-per-year-Dataset + shared serial driver pattern as MODIS/
      `glass_avhrr`. Left here for history.
- [x] **gadm.py:272, ecoregions/source.py:503 — fixed (2026-08-25).** Removed the `self._dask_client()`
      call entirely from both — it was used only for `dashboard_link` logging, and `SpatialProcessor`
      was never constructed with `dask_client=` either; the actual per-tile work (`_rasterize_tile`) is
      pure shapely/numpy and never touched dask. No behavior change, just stops reserving SLURM job
      memory for a cluster that structurally could never be used in this code path.

## Resolved (continued, 3)

- [x] **GDAL's own native block cache, unbounded per worker process — attempted fix (2026-08-25),
      did not resolve the underlying instability.** Surfaced on the first real production MODIS
      PREPARE run after `parallel_prepare.py` went live: all 4 workers repeatedly climbed to ~34GB
      RSS ("Unmanaged memory... 30GB", worker paused, then the nanny killing/restarting it for
      exceeding 95% of its 38GiB budget), over and over, throughout the run. Hypothesized GDAL's
      unbounded native block cache (5% of node RAM, never configured anywhere in this repo) as the
      cause and shipped `os.environ.setdefault("GDAL_CACHEMAX", "512")` plus explicit dataset closing
      in `_read_annual_geotiff`. **A second production run showed the identical ~30.4-30.6GB plateau
      at the same point in the run, unaffected by the fix, and this run additionally crashed outright**
      (`distributed.scheduler.KilledWorker`: one task retried on all 4 workers, all of which died
      running it). This disproved the GDAL-cache hypothesis as the (sole) cause — see the entry below
      for the actual fix. The `GDAL_CACHEMAX` bound in `raster_year_parallel.py` (esacci/eog/ntl_harm/
      acag) is left in place as cheap insurance even though it wasn't the MODIS root cause.

## Resolved (continued, 4)

- [x] **MODIS / GLASS-MODIS per-unit `client.submit` task dispatch — replaced (2026-08-25) with a
      lazy dask-array + shared serial driver design.** Root cause of the real production instability
      above: dispatching one `client.submit(_process_unit, ...)` task per `(tile, year)` unit (2856
      of them) meant every worker process independently opened, cached, and `xr.combine_by_coords`'d
      its own subset of MODIS source tiles, redundantly and unpredictably, with no way to bound total
      cross-worker memory even after closing individual file handles. Replaced `parallel_prepare.py`'s
      task-dispatch model entirely with the same pattern `glass_avhrr` already used successfully:
      - `_read_annual_geotiff` now opens each source tile with `chunks=True` (dask-backed, no pixel
        data read yet) instead of eagerly.
      - `_execute_prepare` builds one **lazy** per-year mosaic via `xr.combine_by_coords` over those
        dask-backed tiles — cheap regardless of tile count, since combining lazy arrays only builds a
        task graph, no I/O happens yet.
      - `raw_getter` clips that lazy mosaic to one output tile's bbox via `sel_bbox()`, then calls
        `.compute()` — *this* is the only point real pixel data is read, and only for the small
        clipped region. With the dask client active as the default scheduler, that `.compute()` call
        distributes the underlying chunk reads across the (still-running) worker pool automatically
        and releases them immediately after.
      - The per-unit loop itself is back to `run_tiled_prepare`'s shared **serial** driver — reprojection
        and the parquet write happen one tile at a time in the main process, against regular (not
        worker) process memory, not fanned out via `client.submit`.
      This means the Dask cluster's role shrinks to "distributed chunk reader" rather than "task
      executor holding a whole per-unit pipeline in worker memory" — it can be sized much smaller
      relative to the main process's own memory than the 90%-of-node-memory split `slurm.py` currently
      uses, since the bulk of any single moment's live data is one tile's worth, held in the main
      process, not accumulated per-worker across hundreds of tasks. `modis/parallel_prepare.py` is
      deleted; `run_tiled_prepare_dask()`/`_process_unit`/the per-worker LRU caches no longer exist for
      MODIS. Verified against the full existing test suite (`tests/data/sources/modis/`,
      `tests/data/sources/glass/`, real small local Dask clients, same expected output values) — no
      test changes were needed, since `_execute_prepare`'s external behavior is unchanged.
      Applied to esacci/eog/ntl_harm/acag too the same day — see "Resolved (continued, 5)" below.

## Resolved (continued, 5)

- [x] **esacci / eog / ntl_harm / acag per-unit `client.submit` task dispatch — replaced (2026-08-25)
      with the same lazy-Dataset + shared serial driver design as MODIS.** Same reasoning as the MODIS
      rework above, applied preemptively (no production failure observed for these four yet, but the
      per-unit dispatch shape is identical to what just broke MODIS) and simpler to land: each of these
      four already had exactly one whole-year source file per year, not many tiles to mosaic, so no
      `xr.combine_by_coords` step was even needed — just reverting each `_execute_prepare` back to a
      lazy per-year `Dataset` (`chunks="auto"`, from `raster_year_parallel.py`'s pattern-1 fix — see
      "Resolved (continued)" above, which already got this part right) plus `run_tiled_prepare`'s shared
      serial driver, exactly matching `glass_avhrr`/MODIS's shape.
      - `esacci`: `_load_nc_as_dataset` reverted to a plain `(file_path, year, temp_dir)` signature,
        called via `EsacciSource._load_nc_as_dataset(...)` (class-level, not `self.`, to avoid an
        instance-method auto-binding pitfall a monkeypatched test lambda hit — see below); `raw_getter`
        clips via `sel_bbox()` + `clip_lon180()` inline again, same as before the dispatcher existed.
      - `acag`: same shape, no cleanup complexity (never had a `.gz`/`.zip` case).
      - `eog`/`ntl_harm`: `_load_year` reverted to an instance method returning `(dataset,
        cleanup_path_or_None)`; `_execute_prepare`'s `year_ds()`/`evict_cache()` closures own deleting
        the `.gz`/`.zip` temp file/dir once a year's cache entry is evicted — safe again now that only
        one process ever touches a given year's decompressed file, so the PID-unique-temp-path
        workaround from the dispatcher version is no longer needed (back to the fixed
        `local_file[:-3]` path, immediately cleaned up).
      - **Pitfall hit while reverting**: calling a `@staticmethod` via `self._load_nc_as_dataset(...)`
        instead of `ClassName._load_nc_as_dataset(...)` works fine for the real staticmethod, but a
        test's `monkeypatch.setattr(Cls, "_load_nc_as_dataset", lambda ...)` replaces it with a *plain*
        function — accessed via `self.`, Python's descriptor protocol then auto-binds `self` as the
        lambda's first positional argument, shifting every other argument by one and producing bizarre
        downstream errors (a file path parsed as if it were a `pandas.Timestamp` year string). Fixed by
        calling the classmethod-style reference (`EsacciSource._load_nc_as_dataset(...)`,
        `AcagSource._load_nc_as_dataset(...)`) everywhere, which sidesteps instance-binding entirely.
      `raster_year_parallel.py` is deleted. Updated tests: reverted `test_esacci_prepare.py`'s and
      `test_acag_prepare.py`'s docstrings/monkeypatch signatures, `test_acag_grid_geobox.py`'s spy
      target back to `driver.run_tiled_prepare`, `test_eog_prepare.py`'s and `test_ntl_harm_plan.py`'s
      `.gz`/`.zip` tests back to asserting the temp file/dir *is* cleaned up (rather than deliberately
      leaked) — full `tests/data` suite (732 passed) confirms no behavior regressions.

**Net result of "Resolved (continued, 3)" through "(continued, 5)"**: every tiled-raster PREPARE
source (MODIS, GLASS-MODIS, glass_avhrr, esacci, eog, ntl_harm, acag) now shares one architecture —
open lazily, clip via `sel_bbox()`, `.compute()` through the active Dask client (which distributes
just that clip's chunk reads across the worker pool), then reproject/write serially in the main
process. The Dask cluster's role everywhere is "distributed chunk reader," never "task executor
holding a whole per-unit pipeline in worker memory" — which is what makes it safe to size much
smaller relative to main-process memory than the current 90%-of-node-memory `slurm.py` split assumes.
Revisiting that split (or at least MODIS's own SLURM invocation) is a reasonable follow-up, not done
here.

## Confirmed fine / not applicable

- [x] **glass_avhrr** (`src/data/sources/glass/avhrr.py:704-721`) — `year_ds()` opens via
      `xr.open_zarr` (dask-chunked), `raw_getter` does `.sel(y=..., x=...).compute()` — clips before
      computing, so only the small per-tile+halo region ever materializes. Already the target shape
      for the pattern-1 fixes above.
- [x] **snl_mining, osm** — neither calls `self._dask_client()` anywhere in their PREPARE path, and
      neither builds a full-domain pixel array (`_rasterize_tile` is vector-polygon rasterization
      directly onto one output tile's geobox). No memory or idle-cluster risk to fix.

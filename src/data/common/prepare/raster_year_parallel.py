"""Dask-distributed PREPARE loop for tiled raster sources whose PREPARE step
reads one whole-year source file (esacci, eog, ntl_harm, acag) and clips it
per output tile via `sel_bbox()` before reprojecting -- as opposed to
MODIS's per-(tile, year) source-tile files
(`src.data.sources.modis.parallel_prepare`) or the vector-rasterization
sources (gadm, ecoregions, snl_mining, osm), which stay on
`run_tiled_prepare`'s shared serial loop since their `raw_getter` closures
hold state (DuckDB connections, GeoDataFrames) unsafe to hand to a separate
worker process.

Reuses `run_tiled_prepare`'s own status-tracking/locking/marker primitives
(`driver.py`) so a run behaves identically from the outside -- only the loop
body differs: outstanding units are submitted concurrently to the
already-running Dask client instead of processed one at a time in this
process (docs/design/13-prepare-memory-parallelism.md).

A source's `load_year_fn(path, year, temp_dir, extra) -> Dataset | None`
must be a plain, qualified-name-referenced function (e.g.
`SomeSource._load_year`, a `@staticmethod` -- not a closure or
`functools.partial`, see `_process_unit`'s docstring for why) -- it runs
inside a Dask worker process, once per (tile, year) unit, behind a bounded
per-worker `functools.lru_cache` so repeated units for the same year on the
same worker don't re-open/re-decompress the source file. *extra* carries
whatever source-instance state `load_year_fn` needs (e.g. EOG's
`source_type`) without binding it into the function object itself; pass
`None` when unneeded.

**Accepted trade-off for `.gz`/`.zip`-wrapped sources** (eog, ntl_harm):
each worker that touches a given year decompresses it once (not once per
tile, thanks to the cache), but the decompressed temp file/dir is never
explicitly deleted here -- there is no safe point to delete it without
risking a race against another in-flight task on the same worker still
reading it (2 threads/worker), and `functools.lru_cache` gives no eviction
hook to hang a delete off. Left for the OS/job's own scratch-dir cleanup at
process/job end. Bounded by (distinct years touched) x (worker count), not
by tile count -- a small, fixed cost, not the unbounded-with-run-size
problem this whole document started from.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import xarray as xr

from src.data.common import lockfile, statusfile
from src.data.common.fetch.manifest import DEFAULT_MAX_ATTEMPTS, STATUS_UNAVAILABLE, clear_failure, record_failure
from src.data.common.prepare.driver import STATUS_COMPLETE, status_dir_for, tile_units
from src.data.common.raster.spatial import SpatialProcessor, sel_bbox
from src.data.sources.steps import mark_complete, marker_path

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=4)
def _load_year_cached(
    load_year_fn: Callable[[str, int, str, Any], Optional["xr.Dataset"]],
    path: str,
    year: int,
    temp_dir: str,
    load_year_extra: Any,
) -> Optional["xr.Dataset"]:
    return load_year_fn(path, year, temp_dir, load_year_extra)


def _process_unit(
    load_year_fn: Callable[[str, int, str, Any], Optional["xr.Dataset"]],
    load_year_extra: Any,
    source_path: str,
    y_dim: str,
    x_dim: str,
    variable_name: str,
    nan_fill_on_empty: bool,
    clip_antimeridian: bool,
    tile,
    year: int,
    output_path: str,
    full_width: int,
    hpc_root: str,
    temp_dir: str,
    target_geobox,
    resampling: str,
    dst_nodata: Optional[float],
) -> "tuple[str, bool, str]":
    """Runs inside a Dask worker process -- must not close over anything
    from the submitting process besides its plain (picklable) arguments.
    Returns `(unit_id, ok, error)` rather than raising/logging into the
    submitting process's status machinery directly, since only the
    submitting process holds the status-dir lock.

    *load_year_fn* must be a plain, qualified-name-referenced function (e.g.
    `SomeSource._load_year`, a `@staticmethod`) -- not a closure or
    `functools.partial` -- so it pickles by reference and every worker's
    deserialized copy is the *same* function object; `_load_year_cached`'s
    `lru_cache` keys on it by identity, so a by-value-reconstructed copy
    would silently defeat the cache (still correct, just re-opens/
    re-decompresses the source file for every single unit instead of once
    per worker per year). Any source-instance state `load_year_fn` needs
    (e.g. EOG's `source_type`) travels separately via *load_year_extra*
    instead of being bound into the function itself, for the same reason.
    """
    unit_id = f"{year}/{tile.id}"
    try:
        ds = _load_year_cached(load_year_fn, source_path, year, temp_dir, load_year_extra)
        if ds is None:
            return unit_id, False, f"raw file failed to load for year {year} ({source_path})"

        extent = tile.geobox.pad(32, 32).extent
        if clip_antimeridian:
            from odc.geo.geom import clip_lon180

            extent = clip_lon180(extent)
        bbox = extent.to_crs(ds.rio.crs).boundingbox
        clipped = sel_bbox(ds, bbox, y_dim=y_dim, x_dim=x_dim)
        if clipped.sizes.get(x_dim, 0) == 0 or clipped.sizes.get(y_dim, 0) == 0:
            if not nan_fill_on_empty:
                return unit_id, False, "tile falls outside source coverage"
            # Legitimate tile state (e.g. poleward of the source's own
            # extent), not a failure -- NaN-fill on tile.geobox instead,
            # same convention as MODIS/AVHRR's own raw_getter.
            dim_y, dim_x = tile.geobox.dims
            source_ds = xr.Dataset(
                {variable_name: ((dim_y, dim_x), np.full(tile.geobox.shape, np.nan, dtype=np.float32))}
            )
        else:
            source_ds = clipped.compute()

        processor = SpatialProcessor(hpc_root=hpc_root, temp_dir=temp_dir, target_geobox=target_geobox)
        ok = processor.process_tile_region(
            source_ds, output_path, tile, year, full_width, resampling=resampling, dst_nodata=dst_nodata,
        )
        return (unit_id, True, "") if ok else (unit_id, False, "process_tile_region failed")
    except Exception as exc:  # noqa: BLE001 -- reported to the submitting process, not raised in the worker
        logger.exception("Error processing PREPARE unit %s", unit_id)
        return unit_id, False, str(exc)


def run_tiled_prepare_dask_year_major(
    *,
    client,
    load_year_fn: Callable[[str, int, str, Any], Optional["xr.Dataset"]],
    load_year_extra: Any = None,
    raw_files_resolved: Dict[int, str],
    y_dim: str = "y",
    x_dim: str = "x",
    variable_name: str,
    nan_fill_on_empty: bool = True,
    clip_antimeridian: bool = False,
    output_path: str,
    years: List[int],
    target_geobox,
    hpc_root: str,
    temp_dir: str,
    tile_size: int,
    resampling: str = "nearest",
    dst_nodata: Optional[float] = None,
    processing_version: str = "1",
    override: bool = False,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> bool:
    """`run_tiled_prepare`'s per-unit loop, parallelized across the given
    Dask `client`'s worker processes instead of running serially in this
    one. Same resumability/marker semantics -- see `driver.py`'s
    docstring -- only the loop body differs. *raw_files_resolved* must
    already be fully-resolved local paths (not relative-to-raw-root), since
    resolving them may need instance state that isn't safe to pickle into a
    worker task."""
    from distributed import as_completed

    status_dir = status_dir_for(output_path)
    lock_path = os.path.join(status_dir, "prepare.lock")

    units = tile_units(years, target_geobox, tile_size=tile_size)
    full_width = target_geobox.shape.x
    total_units = len(units)

    try:
        with lockfile.held(lock_path):
            all_ok = True
            any_processed = False
            processed = 0
            skipped = 0
            failed = 0
            run_started = time.monotonic()
            logger.info(
                "PREPARE %s: %d unit(s) to check (tile_size=%d, dask-parallel)",
                output_path, total_units, tile_size,
            )

            outstanding = []
            for unit in units:
                unit_status_path = statusfile.status_path(status_dir, unit.unit_id)
                existing = statusfile.read(unit_status_path)
                if not override and existing is not None:
                    if existing.get("status") == STATUS_UNAVAILABLE:
                        all_ok = False
                        failed += 1
                        continue
                    if (
                        existing.get("status") == STATUS_COMPLETE
                        and existing.get("processing_version") == processing_version
                    ):
                        skipped += 1
                        continue
                outstanding.append(unit)

            if outstanding:
                any_processed = True
                futures = [
                    client.submit(
                        _process_unit,
                        load_year_fn,
                        load_year_extra,
                        raw_files_resolved[unit.year],
                        y_dim,
                        x_dim,
                        variable_name,
                        nan_fill_on_empty,
                        clip_antimeridian,
                        unit.tile,
                        unit.year,
                        output_path,
                        full_width,
                        hpc_root,
                        temp_dir,
                        target_geobox,
                        resampling,
                        dst_nodata,
                        pure=False,
                    )
                    for unit in outstanding
                ]
                for future in as_completed(futures):
                    unit_id, ok, error = future.result()
                    unit_status_path = statusfile.status_path(status_dir, unit_id)
                    if ok:
                        clear_failure(status_dir, unit_id)
                        statusfile.write(
                            unit_status_path, {"status": STATUS_COMPLETE, "processing_version": processing_version}
                        )
                        processed += 1
                        logger.info("PREPARE %s: unit %s done", output_path, unit_id)
                    else:
                        record_failure(status_dir, unit_id, error, max_attempts=max_attempts)
                        all_ok = False
                        failed += 1
                        logger.error("PREPARE %s: unit %s failed -- %s", output_path, unit_id, error)

            logger.info(
                "PREPARE %s: finished -- %d processed, %d already complete, %d failed/unavailable "
                "(%d total unit(s)), %.1fs elapsed",
                output_path, processed, skipped, failed, total_units, time.monotonic() - run_started,
            )

            if all_ok and (any_processed or not os.path.exists(marker_path(output_path))):
                mark_complete(output_path)
            return all_ok
    except lockfile.LockHeldError:
        logger.warning("PREPARE already running for %s -- skipping this invocation", output_path)
        return False

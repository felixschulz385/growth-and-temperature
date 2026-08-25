"""Dask-distributed PREPARE loop for the MODIS-shaped sources (ModisSource,
GlassModisSource). `src.data.common.prepare.driver.run_tiled_prepare`'s own
per-unit loop is intentionally serial and shared by ~8 sources, several of
whose `raw_getter` closures hold state that isn't safe to hand to a separate
worker process (snl_mining's DuckDB connection, gadm/ecoregions' GeoDataFrame
closures, osm's land polygons) -- so this module does not touch that shared
driver. It exists only for the two MODIS-family sources, whose per-unit read
is a pure function of `(stage1_root, tile, year)`: a handful of GeoTIFFs on
shared disk, safe to read from any worker process.

Reuses `run_tiled_prepare`'s own status-tracking/locking/marker primitives
(`driver.py`) so a MODIS PREPARE run behaves identically from the outside
(resumable, `data summary --by-tile`-visible, same marker semantics) --
only the loop body differs: outstanding units are submitted concurrently to
the already-running Dask client instead of processed one at a time in this
process. This also means the Dask cluster every MODIS PREPARE run already
spins up (and reserves most of the SLURM job's memory for) does real work,
instead of sitting idle while a single-process loop does everything.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import xarray as xr

from src.data.common import lockfile, statusfile
from src.data.common.fetch.manifest import DEFAULT_MAX_ATTEMPTS, STATUS_UNAVAILABLE, clear_failure, record_failure
from src.data.common.prepare.driver import STATUS_COMPLETE, status_dir_for, tile_units
from src.data.common.raster.spatial import SpatialProcessor
from src.data.sources.steps import mark_complete, marker_path

logger = logging.getLogger(__name__)


# functools.lru_cache-decorated module-level functions keep their cache in
# each worker process's own memory, persisting across every unit that
# worker executes for the life of the cluster -- the parallel counterpart
# of the bounded per-run caches the single-process loop used to keep as
# closures. Bounded (unlike loading a whole year's tiles at once) so memory
# never grows with how many years/tiles a worker ends up processing.
@functools.lru_cache(maxsize=8)
def _year_tile_index(stage1_root: str, year: int) -> Tuple[Tuple[Tuple[str, Any], ...], Any]:
    import rasterio

    year_dir = os.path.join(stage1_root, str(year))
    index = []
    crs = None
    if os.path.isdir(year_dir):
        for f in sorted(os.listdir(year_dir)):
            if not f.endswith(".tif"):
                continue
            path = os.path.join(year_dir, f)
            with rasterio.open(path) as src:
                index.append((path, src.bounds))
                if crs is None:
                    crs = src.crs
    return tuple(index), crs


@functools.lru_cache(maxsize=16)
def _read_source_tile_cached(read_tile_fn: Callable[[str, int], "xr.Dataset"], path: str, year: int) -> "xr.Dataset":
    return read_tile_fn(path, year)


def _process_unit(
    read_tile_fn: Callable[[str, int], "xr.Dataset"],
    stage1_root: str,
    tile,
    year: int,
    output_path: str,
    full_width: int,
    hpc_root: str,
    temp_dir: str,
    target_geobox,
    resampling: str,
    dst_nodata: Optional[float],
) -> Tuple[str, bool, str]:
    """Runs inside a Dask worker process -- must not close over anything
    from the submitting process besides its plain (picklable) arguments.
    Returns `(unit_id, ok, error)` rather than raising/logging into the
    submitting process's status machinery directly, since only the
    submitting process holds the status-dir lock."""
    unit_id = f"{year}/{tile.id}"
    try:
        index, source_crs = _year_tile_index(stage1_root, year)
        if not index:
            return unit_id, False, f"No stage-1 tiles for year {year} at {os.path.join(stage1_root, str(year))}"

        bbox = tile.geobox.pad(32, 32).extent.to_crs(source_crs).boundingbox
        overlapping = [
            path
            for path, bounds in index
            if bounds.right >= bbox.left
            and bounds.left <= bbox.right
            and bounds.top >= bbox.bottom
            and bounds.bottom <= bbox.top
        ]

        dim_y, dim_x = tile.geobox.dims
        if not overlapping:
            # Tile falls outside this year's fetched coverage -- legitimate
            # tile state, not a failure (same convention as
            # ecoregions/gadm/snl_mining's _rasterize_tile). Var names come
            # from one arbitrary already-fetched tile, not a full mosaic.
            sample = _read_source_tile_cached(read_tile_fn, index[0][0], year)
            source_ds = xr.Dataset(
                {
                    var: ((dim_y, dim_x), np.full(tile.geobox.shape, np.nan, dtype=np.float32))
                    for var in sample.data_vars
                }
            )
        else:
            datasets = [_read_source_tile_cached(read_tile_fn, path, year) for path in overlapping]
            mosaic = (
                datasets[0] if len(datasets) == 1 else xr.combine_by_coords(datasets, combine_attrs="override")
            )
            if mosaic.rio.crs is None:
                mosaic = mosaic.rio.write_crs(source_crs)
            clipped = mosaic.sel(y=slice(bbox.top, bbox.bottom), x=slice(bbox.left, bbox.right))
            if clipped.sizes.get("x", 0) == 0 or clipped.sizes.get("y", 0) == 0:
                source_ds = xr.Dataset(
                    {
                        var: ((dim_y, dim_x), np.full(tile.geobox.shape, np.nan, dtype=np.float32))
                        for var in mosaic.data_vars
                    }
                )
            else:
                source_ds = clipped

        processor = SpatialProcessor(hpc_root=hpc_root, temp_dir=temp_dir, target_geobox=target_geobox)
        ok = processor.process_tile_region(
            source_ds, output_path, tile, year, full_width, resampling=resampling, dst_nodata=dst_nodata,
        )
        return (unit_id, True, "") if ok else (unit_id, False, "process_tile_region failed")
    except Exception as exc:  # noqa: BLE001 -- reported to the submitting process, not raised in the worker
        logger.exception("Error processing PREPARE unit %s", unit_id)
        return unit_id, False, str(exc)


def run_tiled_prepare_dask(
    *,
    client,
    read_tile_fn: Callable[[str, int], "xr.Dataset"],
    stage1_root: str,
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
    docstring -- only the loop body differs."""
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
                        read_tile_fn,
                        stage1_root,
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

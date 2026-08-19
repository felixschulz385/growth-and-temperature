"""Generic per-(tile, year) PREPARE execution loop. A source declares its required
output shape once (`years`, `variables`, `raw_getter`) and this drives the
same bootstrap-then-fill-tiles sequence every tiled raster source needs:
create the empty, chunk-aligned output zarr if missing, then for every
(year, tile) unit not already recorded `complete` in its status sidecar,
call the source's own `raw_getter` (which owns halo/overshoot handling --
this driver knows nothing about a source's raw input layout) and region-write
the result via `SpatialProcessor.process_tile_region`.

Single-worker-per-(source, stage) is the only concurrency model this needs
(confirmed for FETCH already, same answer for PREPARE) -- `lockfile.held()`
exists to catch an accidental double-invocation, not to arbitrate real
parallel workers, so the whole run holds one lock for its duration rather
than one lock per unit.

Completion is two-layered, same idea as FETCH's manifest buckets: each
unit's own status sidecar (`complete`/`retrying`/`unavailable`, via
`src.data.common.fetch.manifest`'s already-generic `record_failure`/
`clear_failure` -- nothing FETCH-specific about that pair beyond the module
they live in) tracks per-unit progress across resumed runs; the output's
`Completion.MARKER` sibling (`steps.mark_complete`) is only written once
every declared unit for this output is `complete`, so a partially-filled
zarr is never read as finished by a downstream consumer.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from src.data.common import lockfile, statusfile, tiling
from src.data.common.fetch.manifest import DEFAULT_MAX_ATTEMPTS, STATUS_UNAVAILABLE, clear_failure, record_failure
from src.data.sources.steps import mark_complete

logger = logging.getLogger(__name__)

STATUS_COMPLETE = "complete"


@dataclass(frozen=True)
class TileUnit:
    """One (tile, year) unit of PREPARE work -- or one (tile,) unit for a
    static, year-independent source (`year=None`, e.g. gadm/ecoregions/osm:
    one value per pixel, no temporal dimension at all)."""

    tile: tiling.Tile
    year: Optional[int] = None

    @property
    def unit_id(self) -> str:
        return self.tile.id if self.year is None else f"{self.year}/{self.tile.id}"


def status_dir_for(output_path: str) -> str:
    """`_status/<output-basename>/` sibling to *output_path* -- same
    sibling-not-inside convention `verify.manifest_path()` uses for its own
    cache, for the same reason: a unit status file living inside the zarr
    store directory would make writing it look like a chunk-tree edit to the
    store's own fingerprint."""
    trimmed = output_path.rstrip(os.sep)
    return os.path.join(os.path.dirname(trimmed), statusfile.STATUS_SUBDIR, os.path.basename(trimmed))


def tile_units(
    years: Optional[Sequence[int]], target_geobox, tile_size: int = tiling.DEFAULT_TILE_SIZE
) -> list[TileUnit]:
    """`years=None` (or empty) declares a static, year-independent output --
    one unit per tile, `year=None` -- instead of one unit per (tile, year)."""
    tiles = list(tiling.iter_tiles(target_geobox, tile_size=tile_size))
    if not years:
        return [TileUnit(tile=t, year=None) for t in tiles]
    return [TileUnit(tile=t, year=y) for y in years for t in tiles]


def run_tiled_prepare(
    *,
    output_path: str,
    years: Optional[Sequence[int]] = None,
    variables: Sequence[str] = (),
    target_geobox,
    processor: Any,
    raw_getter: "Callable[[tiling.Tile, Optional[int]], Optional[Any]]",
    tile_size: int = tiling.DEFAULT_TILE_SIZE,
    reproject: bool = True,
    preprocess_func: "Optional[Callable[[Any], Any]]" = None,
    dst_nodata: Optional[float] = None,
    resampling: str = "nearest",
    processing_version: str = "1",
    override: bool = False,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> bool:
    """Run PREPARE for one tiled output. Returns True only if every declared
    unit ends the run `complete` (a unit stuck `unavailable` after
    *max_attempts* keeps this False forever until an operator intervenes --
    unlike FETCH, a permanently-missing tile is not an acceptable steady
    state for a raster output a downstream reader expects to be gap-free).

    *years=None* (default) declares a static, year-independent output -- one
    unit per tile instead of one per (tile, year), for sources with no
    temporal dimension at all (e.g. gadm/ecoregions' admin-boundary grids,
    osm's land mask). Pass an explicit year list for temporal outputs.

    *raw_getter(tile, year)* returns the source-local input dataset already
    covering `tile.geobox`'s extent plus whatever halo that source's own
    reprojection needs (this driver applies no buffering itself), or `None`/
    raises if that unit's input isn't available yet -- either way the unit is
    recorded as a failure and retried on the next call, exactly like a FETCH
    download failure. *year* is `None` when `years=None`. `raw_getter` is
    also where any per-source pre-tile setup belongs (loading vector layers,
    building an id-mapping dict, a DuckDB feature pre-pass, memoizing a
    per-year composite across tile calls, ...) -- this driver only calls
    `raw_getter(tile, year)` per unit, so a source closes over whatever
    shared state it needs before calling `run_tiled_prepare`.

    *reproject=True* (default) reprojects `raw_getter`'s return value onto
    `tile.geobox` via `xr_reproject` (the raster-resampling case). Pass
    `reproject=False` for a source whose `raw_getter` already rasterizes/
    produces its output directly on `tile.geobox` (e.g. vector polygon
    rasterization) -- the dataset is used as-is, no resampling applied.

    *processing_version* is a source-controlled cache-buster: bump it when a
    raw-getter or its processing logic changes in a way that must invalidate
    every unit's `complete` status, forcing a full reprocess. `override=True`
    forces every unit regardless of status, same meaning as every other
    step's `cfg.override`.

    Output is `cell_id`-keyed parquet, one self-contained part per unit
    (`processor.process_tile_region`), not a Zarr store -- so there is no
    shared skeleton to bootstrap before the tile loop (each unit's own
    `os.makedirs` creates whatever directories it needs).
    """
    status_dir = status_dir_for(output_path)
    lock_path = os.path.join(status_dir, "prepare.lock")

    units = tile_units(years, target_geobox, tile_size=tile_size)
    full_width = target_geobox.shape.x
    total_units = len(units)

    try:
        with lockfile.held(lock_path):
            all_ok = True
            processed = 0
            skipped = 0
            failed = 0
            run_started = time.monotonic()
            logger.info("PREPARE %s: %d unit(s) to check (tile_size=%d)", output_path, total_units, tile_size)

            for i, unit in enumerate(units, start=1):
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

                unit_started = time.monotonic()
                try:
                    source_ds = raw_getter(unit.tile, unit.year)
                except Exception as exc:  # noqa: BLE001 -- one unit's failure must not abort the whole run
                    logger.exception("raw_getter failed for unit %s", unit.unit_id)
                    record_failure(status_dir, unit.unit_id, str(exc), max_attempts=max_attempts)
                    all_ok = False
                    failed += 1
                    continue

                if source_ds is None:
                    record_failure(
                        status_dir, unit.unit_id, "raw_getter returned no data", max_attempts=max_attempts
                    )
                    all_ok = False
                    failed += 1
                    continue

                ok = processor.process_tile_region(
                    source_ds,
                    output_path,
                    unit.tile,
                    unit.year,
                    full_width,
                    reproject=reproject,
                    preprocess_func=preprocess_func,
                    dst_nodata=dst_nodata,
                    resampling=resampling,
                )
                if ok:
                    clear_failure(status_dir, unit.unit_id)
                    statusfile.write(
                        unit_status_path, {"status": STATUS_COMPLETE, "processing_version": processing_version}
                    )
                    processed += 1
                    logger.info(
                        "PREPARE %s: [%d/%d] unit %s done in %.1fs",
                        output_path, i, total_units, unit.unit_id, time.monotonic() - unit_started,
                    )
                else:
                    record_failure(status_dir, unit.unit_id, "process_tile_region failed", max_attempts=max_attempts)
                    all_ok = False
                    failed += 1

            logger.info(
                "PREPARE %s: finished -- %d processed, %d already complete, %d failed/unavailable "
                "(%d total unit(s)), %.1fs elapsed",
                output_path, processed, skipped, failed, total_units, time.monotonic() - run_started,
            )

            if all_ok:
                mark_complete(output_path)
            return all_ok
    except lockfile.LockHeldError:
        logger.warning("PREPARE already running for %s -- skipping this invocation", output_path)
        return False


def prepare_status(
    output_path: str,
    years: Optional[Sequence[int]],
    target_geobox,
    tile_size: int = tiling.DEFAULT_TILE_SIZE,
) -> dict[str, int]:
    """Per-unit status counts for `data summary --by-tile` -- complete/
    outstanding/unavailable across every declared (tile, year) unit, without
    running anything."""
    status_dir = status_dir_for(output_path)
    units = tile_units(years, target_geobox, tile_size=tile_size)
    counts = {STATUS_COMPLETE: 0, "outstanding": 0, STATUS_UNAVAILABLE: 0}
    for unit in units:
        existing = statusfile.read(statusfile.status_path(status_dir, unit.unit_id))
        if existing is None:
            counts["outstanding"] += 1
        elif existing.get("status") == STATUS_UNAVAILABLE:
            counts[STATUS_UNAVAILABLE] += 1
        elif existing.get("status") == STATUS_COMPLETE:
            counts[STATUS_COMPLETE] += 1
        else:
            counts["outstanding"] += 1
    return counts

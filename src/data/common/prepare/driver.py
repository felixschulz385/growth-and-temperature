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
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from src.data.common import lockfile, statusfile, tiling
from src.data.common.fetch.manifest import DEFAULT_MAX_ATTEMPTS, STATUS_UNAVAILABLE, clear_failure, record_failure
from src.data.sources.steps import mark_complete

logger = logging.getLogger(__name__)

STATUS_COMPLETE = "complete"


@dataclass(frozen=True)
class TileUnit:
    """One (tile, year) unit of PREPARE work."""

    tile: tiling.Tile
    year: int

    @property
    def unit_id(self) -> str:
        return f"{self.year}/{self.tile.id}"


def status_dir_for(output_path: str) -> str:
    """`_status/<output-basename>/` sibling to *output_path* -- same
    sibling-not-inside convention `verify.manifest_path()` uses for its own
    cache, for the same reason: a unit status file living inside the zarr
    store directory would make writing it look like a chunk-tree edit to the
    store's own fingerprint."""
    trimmed = output_path.rstrip(os.sep)
    return os.path.join(os.path.dirname(trimmed), statusfile.STATUS_SUBDIR, os.path.basename(trimmed))


def tile_units(years: Sequence[int], target_geobox, tile_size: int = tiling.DEFAULT_TILE_SIZE) -> list[TileUnit]:
    tiles = list(tiling.iter_tiles(target_geobox, tile_size=tile_size))
    return [TileUnit(tile=t, year=y) for y in years for t in tiles]


def run_tiled_prepare(
    *,
    output_path: str,
    years: Sequence[int],
    variables: Sequence[str],
    target_geobox,
    processor: Any,
    raw_getter: "Callable[[tiling.Tile, int], Optional[Any]]",
    target_dims: tuple[str, str],
    tile_size: int = tiling.DEFAULT_TILE_SIZE,
    preprocess_func: "Optional[Callable[[Any], Any]]" = None,
    dst_nodata: Optional[float] = None,
    resampling: str = "nearest",
    dtype: str = "float32",
    packaging_attrs: Optional[dict] = None,
    sample_attrs: Optional[dict] = None,
    processing_version: str = "1",
    override: bool = False,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> bool:
    """Run PREPARE for one tiled output. Returns True only if every declared
    (tile, year) unit ends the run `complete` (a unit stuck `unavailable`
    after *max_attempts* keeps this False forever until an operator
    intervenes -- unlike FETCH, a permanently-missing tile is not an
    acceptable steady state for a raster output a downstream reader expects
    to be gap-free).

    *raw_getter(tile, year)* returns the source-local input dataset already
    covering `tile.geobox`'s extent plus whatever halo that source's own
    reprojection needs (this driver applies no buffering itself), or `None`/
    raises if that unit's input isn't available yet -- either way the unit is
    recorded as a failure and retried on the next call, exactly like a FETCH
    download failure.

    *processing_version* is a source-controlled cache-buster: bump it when a
    raw-getter or its processing logic changes in a way that must invalidate
    every unit's `complete` status, forcing a full reprocess. `override=True`
    forces every unit regardless of status, same meaning as every other
    step's `cfg.override`.
    """
    status_dir = status_dir_for(output_path)
    lock_path = os.path.join(status_dir, "prepare.lock")

    units = tile_units(years, target_geobox, tile_size=tile_size)

    try:
        with lockfile.held(lock_path):
            if not os.path.exists(output_path):
                if not processor.create_empty_target_zarr(
                    output_path,
                    target_geobox,
                    list(years),
                    list(variables),
                    sample_attrs=sample_attrs,
                    dst_nodata=dst_nodata,
                    packaging_attrs=packaging_attrs,
                    dtype=dtype,
                    chunk_size=(tile_size, tile_size),
                ):
                    logger.error("Failed to bootstrap empty output zarr at %s", output_path)
                    return False

            all_ok = True
            for unit in units:
                unit_status_path = statusfile.status_path(status_dir, unit.unit_id)
                existing = statusfile.read(unit_status_path)

                if not override and existing is not None:
                    if existing.get("status") == STATUS_UNAVAILABLE:
                        all_ok = False
                        continue
                    if (
                        existing.get("status") == STATUS_COMPLETE
                        and existing.get("processing_version") == processing_version
                    ):
                        continue

                try:
                    source_ds = raw_getter(unit.tile, unit.year)
                except Exception as exc:  # noqa: BLE001 -- one unit's failure must not abort the whole run
                    logger.exception("raw_getter failed for unit %s", unit.unit_id)
                    record_failure(status_dir, unit.unit_id, str(exc), max_attempts=max_attempts)
                    all_ok = False
                    continue

                if source_ds is None:
                    record_failure(
                        status_dir, unit.unit_id, "raw_getter returned no data", max_attempts=max_attempts
                    )
                    all_ok = False
                    continue

                ok = processor.process_tile_region(
                    source_ds,
                    output_path,
                    unit.tile,
                    target_dims,
                    preprocess_func=preprocess_func,
                    dst_nodata=dst_nodata,
                    resampling=resampling,
                )
                if ok:
                    clear_failure(status_dir, unit.unit_id)
                    statusfile.write(
                        unit_status_path, {"status": STATUS_COMPLETE, "processing_version": processing_version}
                    )
                else:
                    record_failure(status_dir, unit.unit_id, "process_tile_region failed", max_attempts=max_attempts)
                    all_ok = False

            if all_ok:
                mark_complete(output_path)
            return all_ok
    except lockfile.LockHeldError:
        logger.warning("PREPARE already running for %s -- skipping this invocation", output_path)
        return False


def prepare_status(
    output_path: str, years: Sequence[int], target_geobox, tile_size: int = tiling.DEFAULT_TILE_SIZE
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

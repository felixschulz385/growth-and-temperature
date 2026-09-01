"""`raw_getter` factory for PREPARE sources that mosaic a fixed sinusoidal
source-tile grid (MODIS 21A2/11A1, GLASS-MODIS) onto the shared output tile
grid.

FETCH writes one annual multi-band GeoTIFF per ``hHHvVV`` sinusoidal tile.
This builds the ``raw_getter(tile, year)`` that
``run_tiled_prepare(..., reproject=False)`` calls once per output-tile unit:

1. pick the source tiles whose (header-only) bounds intersect the output
   tile's padded, antimeridian-clamped bbox -- reprojected into the source
   CRS (cheap: no pixels read);
2. read + ``.compute()`` each (small: one 1200x1200 tile per band), then
   ``xr_reproject`` it **individually** onto ``tile.geobox`` -- a numpy
   source takes ``rio_reproject`` directly, so this never touches the dask
   ``grid_intersect`` path (and its monkeypatched, degenerate-geometry-prone
   ``_patched_grid_intersect``);
3. overlay first-non-null-wins onto a georegistered all-NaN canvas built on
   ``tile.geobox``.

The return value is **always** exactly ``tile.geobox`` -- correct shape,
native (y-descending) order, georegistered, float32, one variable per source
band -- with no non-georegistered early-return path, so
``process_tile_region`` (which runs with ``reproject=False`` for these
sources and only tabulates) never sees a malformed dataset. This replaces the
old "mosaic every overlapping source tile in the sinusoidal CRS via
``xr.combine_by_coords``, clip, then reproject the mosaic" flow, which needed
bit-exact coordinate labels (``combine_by_coords`` "duplicate values" /
"not monotonic") and handed non-georegistered NaN-fill / degenerate-clip
datasets to ``xr_reproject`` (docs/design/15-modis-prepare-2002-tile-failures.md).

Because each source tile is reprojected by its own georeferencing, the
sub-pixel FETCH-time grid drift between adjacent tiles
(``GeoBox.from_bbox`` per-tile pin, docs/design/13) no longer matters: two
reprojected footprints differ by less than one destination pixel and the
overlay resolves the seam. No coordinate-label equality is required anywhere.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from odc.geo.geom import box
from odc.geo.xr import xr_reproject, xr_zeros

logger = logging.getLogger(__name__)

#: Bounded LRU of materialised per-source-tile Datasets. Neighbouring output
#: tiles frequently need the same handful of source tiles; 8 slots cover a
#: 3x3 output-tile neighbourhood. Entries are now `.compute()`d numpy arrays
#: (one ~1200x1200 tile per band, a few MB), so this is a few tens of MB at
#: most -- pole output tiles that overlap many source tiles simply re-read on
#: the next unit (I/O only).
DEFAULT_SOURCE_TILE_CACHE_SIZE = 8

#: Halo (in output-grid pixels) added to the output tile's bbox purely to
#: widen *source-tile selection* -- an edge output pixel whose nearest source
#: pixel sits just past a source tile's exact extent still finds that tile.
#: No destination-side halo is needed for nearest-neighbour resampling.
DEFAULT_PAD_PIXELS = 32


def build_sinusoidal_mosaic_raw_getter(
    *,
    stage1_root: str,
    target_geobox: Any,
    read_annual_geotiff: Callable[[str, int], xr.Dataset],
    log_label: str = "sinusoidal-mosaic",
    resampling: str = "nearest",
    source_tile_cache_size: int = DEFAULT_SOURCE_TILE_CACHE_SIZE,
    pad_pixels: int = DEFAULT_PAD_PIXELS,
) -> "Callable[[Any, Optional[int]], Optional[xr.Dataset]]":
    """Build the per-unit ``raw_getter(tile, year)`` (see module docstring).

    *stage1_root* is the source's FETCH output root (``.../<year>/<tile>.tif``).
    *target_geobox* is the fully-resolved output grid (the caller owns the
    ``ease6933`` vs legacy choice). *read_annual_geotiff* is the source's own
    single-file reader (e.g. ``ModisSource._read_annual_geotiff``) -- it may
    return size-1 ``time``/``band`` dims, which are squeezed here.
    """
    tile_index_cache: Dict[int, Tuple[List[Tuple[str, Any]], Any]] = {}
    source_tile_cache: "OrderedDict[str, xr.Dataset]" = OrderedDict()

    def year_tile_index(year: int) -> Tuple[List[Tuple[str, Any]], Any]:
        """Per-year ``[(path, rasterio_bounds), ...]`` (sorted by filename ->
        deterministic first-wins overlay) plus that year's shared source CRS,
        from header-only ``rasterio.open()`` reads (no pixels decoded)."""
        if year not in tile_index_cache:
            import rasterio

            logger.debug("%s PREPARE: building tile index for year %s", log_label, year)
            year_dir = os.path.join(stage1_root, str(year))
            index: List[Tuple[str, Any]] = []
            crs = None
            if os.path.isdir(year_dir):
                for fname in sorted(os.listdir(year_dir)):
                    if not fname.endswith(".tif"):
                        continue
                    path = os.path.join(year_dir, fname)
                    with rasterio.open(path) as src:
                        index.append((path, src.bounds))
                        if crs is None:
                            crs = src.crs
            tile_index_cache[year] = (index, crs)
            logger.debug("%s PREPARE: year %s index built, %d tile(s)", log_label, year, len(index))
        return tile_index_cache[year]

    def read_source_tile(path: str, year: int) -> xr.Dataset:
        """LRU-cached, **materialised** single source tile (squeezed of the
        size-1 time/band dims the reader adds). Eager ``.compute()`` keeps
        the subsequent ``xr_reproject`` on the numpy fast path -- no dask,
        so no ``grid_intersect`` / ``_patched_grid_intersect``."""
        if path in source_tile_cache:
            source_tile_cache.move_to_end(path)
            return source_tile_cache[path]
        logger.debug("%s PREPARE: reading source tile %s", log_label, path)
        ds = read_annual_geotiff(path, year)
        squeeze_dims = [d for d in ("time", "band") if d in ds.dims]
        if squeeze_dims:
            ds = ds.squeeze(squeeze_dims, drop=True)
        ds = ds.compute()
        source_tile_cache[path] = ds
        if len(source_tile_cache) > source_tile_cache_size:
            source_tile_cache.popitem(last=False)
        return ds

    def _selection_bbox(tile, source_crs):
        """The output tile's padded bbox, clamped to the canonical grid's
        own valid extent (antimeridian-wrap guard for grid-edge tiles --
        `GeoBox.from_bbox` pixel-snaps the grid's edges slightly outside the
        valid domain of a periodic CRS like EASE6933, and reprojecting an
        out-of-domain coordinate silently wraps to the far side of the
        world: an unclamped edge tile matched 104/282 fetched tiles instead
        of ~14, docs/design/13), reprojected into the source CRS. A no-op
        for every interior tile."""
        grid_bbox = target_geobox.extent.boundingbox
        margin = 2 * abs(target_geobox.resolution.x)
        padded = tile.geobox.pad(pad_pixels, pad_pixels).extent.boundingbox
        left = max(padded.left, grid_bbox.left + margin)
        bottom = max(padded.bottom, grid_bbox.bottom + margin)
        right = min(padded.right, grid_bbox.right - margin)
        top = min(padded.top, grid_bbox.top - margin)
        if left < right and bottom < top:
            extent = box(left, bottom, right, top, crs=tile.geobox.crs)
        else:
            # Margin would degenerate the bbox (target grid smaller than the
            # margin -- only synthetic tiny-grid tests). Clamp is a
            # defensive no-op for interior tiles anyway.
            extent = tile.geobox.pad(pad_pixels, pad_pixels).extent
        return extent.to_crs(source_crs).boundingbox

    def raw_getter(tile: Any, year: Optional[int]) -> Optional[xr.Dataset]:
        index, source_crs = year_tile_index(year)
        if not index:
            # Genuine "FETCH not done for this year" -- retryable, distinct
            # from "this tile is outside coverage" (handled below).
            logger.error("%s PREPARE: no stage-1 tiles for year %s at %s", log_label, year, stage1_root)
            return None

        bbox = _selection_bbox(tile, source_crs)
        paths = [
            path
            for path, bounds in index
            if bounds.right >= bbox.left
            and bounds.left <= bbox.right
            and bounds.top >= bbox.bottom
            and bounds.bottom <= bbox.top
        ]

        var_names = list(read_source_tile(index[0][0], year).data_vars)
        base = xr_zeros(tile.geobox, "float32")
        canvas = xr.Dataset({v: xr.full_like(base, np.nan) for v in var_names})

        if not paths:
            # Outside this year's fetched coverage -- a legitimate tile
            # state, not a failure. The pristine NaN canvas is already on
            # tile.geobox and georegistered.
            logger.debug("%s PREPARE: tile %s year %s -- no source coverage", log_label, tile.id, year)
            return canvas

        logger.debug(
            "%s PREPARE: tile %s year %s overlaps %d source tile(s)", log_label, tile.id, year, len(paths)
        )
        for path in paths:
            src = read_source_tile(path, year)
            try:
                rep = xr_reproject(src, tile.geobox, resampling=resampling, dst_nodata=np.nan)
            except Exception:
                # A single source tile whose footprint degenerates in the
                # target CRS (grazing overlap at a grid/antimeridian edge)
                # contributes nothing -- skip it rather than fail the unit.
                logger.exception(
                    "%s PREPARE: reproject failed for %s onto tile %s; skipping", log_label, path, tile.id
                )
                continue
            # First non-null wins: `canvas` starts all-NaN, so paths[0]
            # fills its whole footprint and later tiles only fill pixels
            # still NaN. Every `rep` is on `tile.geobox` verbatim, so this
            # aligns with no reindex and no re-sort (identical coords).
            canvas = canvas.combine_first(rep)

        return canvas

    # Exposed for introspection/tests (the per-run memo + bounded LRU).
    raw_getter.tile_index_cache = tile_index_cache
    raw_getter.source_tile_cache = source_tile_cache
    return raw_getter

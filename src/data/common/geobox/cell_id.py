"""Global row-major cell id for the canonical EPSG:6933 EASE grid.

A NEW, separate id scheme from the legacy `pixel_id` (tile-packed
`[ix:16|iy:16|local_pixel:32]`, `src/data/assemble/constants.py`) -- `pixel_id`
stays completely unchanged for the EPSG:4326 legacy grid path
(`ctx.grid_id == LEGACY_GRID_ID`). `cell_id` is used only when
`ctx.grid_id == EASE_GRID_ID`.

Encoding: `cell_id = row * W + col`, where `(row, col)` are 0-based indices
into the FULL canonical grid (not tile-relative) and `W` is the canonical
grid's pixel width. `W`/`H` are never memoized in this module -- every
function that needs them takes them as an explicit parameter, sourced by the
caller from `canonical_ease_geobox()`/`get_or_create_canonical_geobox()`
(`src/data/common/geobox/canonical.py`). Memoizing here would risk silently
desyncing from the actual grid if resolution/lat-clip constants ever change
elsewhere.

Being a flat row-major index (not a tile-packed one like `pixel_id`), it
decomposes with plain integer arithmetic -- directly expressible in SQL/
DuckDB, no decode step needed:

    row, col        = divmod(cell_id, W)
    tile_row, tile_col = row // tile_size, col // tile_size   # -> ix, iy
    coarse_row, coarse_col = row // factor, col // factor      # any factor
    shaken_row, shaken_col = (row + dr) // factor, (col + dc) // factor

Tile-index mapping: `tile_row` corresponds to the assemble pipeline's `ix`
partition key and `Tile.row`; `tile_col` corresponds to `iy` and `Tile.col`.
Verified end-to-end: `src/data/assemble/tiles.py::get_available_tiles` binds
`ix` to `GeoboxTiles.shape[0]` (the row/y axis) and `iy` to `shape[1]`
(col/x axis), consumed unchanged through `workflow.py::_process_all_tiles`
into `processors.py::_get_output_path`'s `ix=.../iy=...` directories -- so
`ix` is the row index and `iy` the col index throughout the real pipeline,
not the intuitive-sounding opposite. (Note: `scripts/validate_backbone_
subset.py`, a validation-only script, uses the opposite ix=col/iy=row
convention -- a pre-existing inconsistency in that script, not reconciled
here.)

Dtype: uint32. The canonical grid is ~34,735 x 12,703 px (~441.2M pixels),
comfortably inside uint32's ~4.29B range even under a hypothetical future
resolution bump -- and halving the width of the dominant per-row column
across every assembled parquet file is a real IO/storage saving. `cell_id`
is a semantically distinct column from `pixel_id`; there is no requirement
that the two schemes share a bit width.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

CELL_ID_DTYPE = np.uint32


def encode_cell_ids(row0: int, col0: int, tile_geobox, full_width: int) -> np.ndarray:
    """`cell_id` matrix for one tile.

    Args:
        row0: Global row (y) pixel offset of the tile's top-left corner, e.g.
            `Tile.y_slice.start` -- not `tile.row * tile_size`, since
            `GeoboxTiles` already handles the ragged final row/col and this
            must match its actual pixel offsets.
        col0: Global col (x) pixel offset of the tile's top-left corner, e.g.
            `Tile.x_slice.start`.
        tile_geobox: The tile's own geobox (gives the output shape).
        full_width: Pixel width `W` of the FULL canonical grid.

    Returns:
        `(h, w)` uint32 array, row-major, matching `tile_geobox.shape`.
    """
    h, w = tile_geobox.shape
    rows = row0 + np.arange(h, dtype=np.int64)[:, None]
    cols = col0 + np.arange(w, dtype=np.int64)[None, :]
    ids = rows * np.int64(full_width) + cols
    max_id = int(ids.max()) if ids.size else 0
    dtype_max = int(np.iinfo(CELL_ID_DTYPE).max)
    if max_id > dtype_max:
        raise ValueError(
            f"cell_id {max_id} exceeds {CELL_ID_DTYPE.__name__} range ({dtype_max}); "
            "the canonical grid has grown too large for this dtype."
        )
    return ids.astype(CELL_ID_DTYPE)


def make_cell_ids(row0: int, col0: int, tile_geobox, full_width: int) -> xr.DataArray:
    """`cell_id` as an `xr.DataArray` on `tile_geobox`'s own y/x coords.

    The canonical-grid analogue of `src/data/assemble/utils.py::make_pixel_ids`
    / `src/data/assemble/ring_means.py::make_canonical_pixel_ids`, for a caller
    building a per-tile dataset to merge into extracted tile output before
    tabularization.
    """
    ids = encode_cell_ids(row0, col0, tile_geobox, full_width)
    dim_y, dim_x = tile_geobox.dims
    da = xr.DataArray(
        ids,
        dims=(dim_y, dim_x),
        coords={
            dim_y: tile_geobox.coords[dim_y].values,
            dim_x: tile_geobox.coords[dim_x].values,
        },
        name="cell_id",
    )
    return da


def decode_cell_id(cell_id, full_width: int) -> tuple:
    """`(row, col) = divmod(cell_id, full_width)`. Accepts scalars or arrays."""
    cell_id = np.asarray(cell_id, dtype=np.int64)
    row, col = np.divmod(cell_id, np.int64(full_width))
    return row, col


def cell_tile_indices(row, col, tile_size: int) -> tuple:
    """`(tile_row, tile_col) = (row // tile_size, col // tile_size)`.

    `tile_row` maps to the assemble pipeline's `ix` partition key and
    `Tile.row`; `tile_col` maps to `iy` and `Tile.col` -- see module
    docstring for the verified mapping.
    """
    row = np.asarray(row)
    col = np.asarray(col)
    return row // tile_size, col // tile_size


def coarsen_cell_id(row, col, factor: int) -> tuple:
    """`(row // factor, col // factor)` -- arbitrary-factor coarse cell
    coordinate, for derived low-resolution columns (e.g. 5km, 50km)."""
    row = np.asarray(row)
    col = np.asarray(col)
    return row // factor, col // factor


def shaken_cell_id(row, col, dr: int, dc: int, factor: int) -> tuple:
    """`((row + dr) // factor, (col + dc) // factor)` -- grid-shake offset
    pattern, mirrors `src/data/assemble/grid_shake.py`'s existing shake
    semantics but expressed on `cell_id`'s row/col."""
    row = np.asarray(row)
    col = np.asarray(col)
    return (row + dr) // factor, (col + dc) // factor

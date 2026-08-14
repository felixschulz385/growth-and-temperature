"""The one shared 2048x2048 output tile grid every raster PREPARE step uses.

Plan 2 (docs/design successor to the ledger) replaces GRID's old
whole-extent-per-year zarr writes with per-(tile, year) region writes, so
every source needs the *same* tile grid -- if `acag` and `esacci` diced the
shared target geobox (`src.data.common.geobox.target.get_target_geobox`)
into tiles independently, two sources' tile #17 would not necessarily cover
the same pixels, breaking any cross-source per-tile assembly. One function,
called with the same `target_geobox`/`tile_size` everywhere, is the fix.

Built on `odc.geo.GeoboxTiles` rather than hand-rolled index math: it already
handles the ragged edge case (a geobox whose extent isn't an exact multiple
of `tile_size` gets a final row/column of narrower tiles, not an error or a
silently-clipped grid) and gives back real per-tile `GeoBox`es for the
reprojection call each source's raw-getter/region-write needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from odc.geo import GeoboxTiles

#: Every source's PREPARE step must pass this (or an explicit override) to
#: `iter_tiles`/`build_tile_grid` -- see module docstring for why a
#: per-source default would silently break cross-source tile alignment.
DEFAULT_TILE_SIZE = 2048


@dataclass(frozen=True)
class Tile:
    """One cell of the shared output tile grid.

    `row`/`col` are `GeoboxTiles` index order, i.e. `(y, x)` -- the same
    order `y_slice`/`x_slice` and `geobox` follow, so a caller building an
    xarray `region=` dict can write `{dim_y: tile.y_slice, dim_x: tile.x_slice}`
    directly without a transposition step.
    """

    row: int
    col: int
    geobox: object
    y_slice: slice
    x_slice: slice

    @property
    def id(self) -> str:
        """Stable, filesystem/status-file-safe unit id: `"<row>_<col>"`."""
        return f"{self.row:04d}_{self.col:04d}"


def build_tile_grid(target_geobox, tile_size: int = DEFAULT_TILE_SIZE) -> GeoboxTiles:
    return GeoboxTiles(target_geobox, (tile_size, tile_size))


def grid_shape(target_geobox, tile_size: int = DEFAULT_TILE_SIZE) -> tuple[int, int]:
    """`(n_rows, n_cols)` -- i.e. `(ny_tiles, nx_tiles)` -- for the shared grid."""
    shape = build_tile_grid(target_geobox, tile_size).shape
    return shape.y, shape.x


def iter_tiles(target_geobox, tile_size: int = DEFAULT_TILE_SIZE) -> Iterator[Tile]:
    """Every tile of the shared grid covering `target_geobox`, row-major."""
    tiles = build_tile_grid(target_geobox, tile_size)
    n_rows, n_cols = tiles.shape.y, tiles.shape.x
    for row in range(n_rows):
        for col in range(n_cols):
            idx = (row, col)
            y_slice, x_slice = tiles.roi[idx]
            yield Tile(row=row, col=col, geobox=tiles[idx], y_slice=y_slice, x_slice=x_slice)


def tile_by_id(target_geobox, tile_id: str, tile_size: int = DEFAULT_TILE_SIZE) -> Tile:
    """Reconstruct one `Tile` from its `id` without iterating the whole grid --
    used by the PREPARE execution loop to resume a single outstanding unit."""
    row_str, col_str = tile_id.split("_")
    row, col = int(row_str), int(col_str)
    tiles = build_tile_grid(target_geobox, tile_size)
    idx = (row, col)
    y_slice, x_slice = tiles.roi[idx]
    return Tile(row=row, col=col, geobox=tiles[idx], y_slice=y_slice, x_slice=x_slice)

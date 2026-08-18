"""Tests for the global row-major cell_id scheme (EASE6933 grid only)."""

import numpy as np
import pytest
from odc.geo.geobox import GeoBox

from src.data.assemble.tiles import get_available_tiles
from src.data.common.geobox.cell_id import (
    CELL_ID_DTYPE,
    cell_tile_indices,
    coarsen_cell_id,
    decode_cell_id,
    encode_cell_ids,
    make_cell_ids,
    shaken_cell_id,
)
from src.data.common.tiling import iter_tiles


def _make_geobox(width_m=20_000, height_m=10_000, resolution=1000.0):
    return GeoBox.from_bbox((0, 0, width_m, height_m), crs="EPSG:6933", resolution=resolution)


def test_encode_cell_ids_row_major_within_tile():
    gbox = _make_geobox(width_m=5_000, height_m=3_000)  # 5x3 px
    ids = encode_cell_ids(row0=0, col0=0, tile_geobox=gbox, full_width=5)
    expected = np.arange(15, dtype=CELL_ID_DTYPE).reshape(3, 5)
    np.testing.assert_array_equal(ids, expected)
    assert ids.dtype == CELL_ID_DTYPE


def test_encode_cell_ids_offset_tile_matches_global_index():
    # A 2x2 tile starting at global (row0=1, col0=2) in a full grid of width 5
    full_width = 5
    gbox = _make_geobox(width_m=2_000, height_m=2_000)  # 2x2 px
    ids = encode_cell_ids(row0=1, col0=2, tile_geobox=gbox, full_width=full_width)
    # global row 1: cols 2,3 -> ids 7,8 ; global row 2: cols 2,3 -> ids 12,13
    expected = np.array([[7, 8], [12, 13]], dtype=CELL_ID_DTYPE)
    np.testing.assert_array_equal(ids, expected)


def test_encode_cell_ids_raises_on_uint32_overflow():
    gbox = _make_geobox(width_m=1_000, height_m=1_000)  # 1x1 px
    huge_width = 2**32  # row0*huge_width alone overflows uint32
    with pytest.raises(ValueError):
        encode_cell_ids(row0=1, col0=0, tile_geobox=gbox, full_width=huge_width)


def test_decode_cell_id_round_trips_encode():
    full_width = 17
    gbox = _make_geobox(width_m=6_000, height_m=4_000)  # 6x4 px
    ids = encode_cell_ids(row0=3, col0=5, tile_geobox=gbox, full_width=full_width)
    row, col = decode_cell_id(ids, full_width)
    expected_row = 3 + np.arange(4)[:, None] * np.ones((1, 6), dtype=int)
    expected_col = 5 + np.ones((4, 1), dtype=int) * np.arange(6)[None, :]
    np.testing.assert_array_equal(row, expected_row)
    np.testing.assert_array_equal(col, expected_col)


def test_make_cell_ids_returns_named_dataarray_on_tile_coords():
    gbox = _make_geobox(width_m=3_000, height_m=3_000)
    da = make_cell_ids(row0=0, col0=0, tile_geobox=gbox, full_width=3)
    assert da.name == "cell_id"
    assert da.shape == (3, 3)


def test_coarsen_cell_id_arbitrary_factor():
    row = np.array([0, 4, 5, 9])
    col = np.array([0, 4, 5, 9])
    crow, ccol = coarsen_cell_id(row, col, factor=5)
    np.testing.assert_array_equal(crow, [0, 0, 1, 1])
    np.testing.assert_array_equal(ccol, [0, 0, 1, 1])


def test_shaken_cell_id_offsets_before_dividing():
    row = np.array([4])
    col = np.array([4])
    # without shake: 4 // 5 == 0 ; with a +1 shake: (4+1)//5 == 1
    crow, ccol = shaken_cell_id(row, col, dr=1, dc=1, factor=5)
    np.testing.assert_array_equal(crow, [1])
    np.testing.assert_array_equal(ccol, [1])


def test_cell_tile_indices_matches_production_tiler():
    """Cross-check cell_tile_indices' (tile_row, tile_col) -> (ix, iy) mapping
    against the real tile enumeration used by the assemble pipeline
    (src/data/assemble/tiles.py::get_available_tiles + processors.py's
    ix=/iy= output partitioning), not just isolated arithmetic."""
    tile_size = 4
    gbox = _make_geobox(width_m=12_000, height_m=8_000, resolution=1000.0)  # 12x8 px -> 3x2 tiles
    full_width = gbox.shape.x

    available = get_available_tiles({"processing": {"tile_size": tile_size}}, gbox)
    assert set(available) == {(ix, iy) for ix in range(2) for iy in range(3)}

    for tile in iter_tiles(gbox, tile_size=tile_size):
        row0, col0 = tile.y_slice.start, tile.x_slice.start
        tile_row, tile_col = cell_tile_indices(row0, col0, tile_size)
        assert (int(tile_row), int(tile_col)) == (tile.row, tile.col)
        # and this (tile.row, tile.col) is exactly the (ix, iy) pair used for
        # this tile's real output path today
        assert (tile.row, tile.col) in available

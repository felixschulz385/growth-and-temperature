"""tiling.py grid math: geobox -> N chunk-aligned tiles covering the full
extent with no gaps/overlaps, plus round-tripping a tile through its id."""

from odc.geo.geobox import GeoBox

from src.data.common import tiling


def _geobox(width, height, tile_size):
    # Deliberately not an exact multiple of tile_size, so the ragged final
    # row/column is exercised by every test using this fixture.
    return GeoBox.from_bbox((0, 0, width * 0.1, height * 0.1), crs="EPSG:4326", resolution=0.1)


def test_grid_shape_matches_ceil_division():
    gb = _geobox(200, 100, 32)
    assert tiling.grid_shape(gb, tile_size=32) == (4, 7)  # (n_rows=ceil(100/32), n_cols=ceil(200/32))


def test_iter_tiles_covers_full_extent_with_no_gaps_or_overlaps():
    gb = _geobox(200, 100, 32)
    tiles = list(tiling.iter_tiles(gb, tile_size=32))

    n_rows, n_cols = tiling.grid_shape(gb, tile_size=32)
    assert len(tiles) == n_rows * n_cols

    covered = set()
    for t in tiles:
        ys = range(t.y_slice.start, t.y_slice.stop)
        xs = range(t.x_slice.start, t.x_slice.stop)
        for y in ys:
            for x in xs:
                key = (y, x)
                assert key not in covered, "overlapping tile region"
                covered.add(key)

    ny, nx = gb.shape.y, gb.shape.x
    assert covered == {(y, x) for y in range(ny) for x in range(nx)}


def test_ragged_edge_tiles_are_narrower_not_clipped_or_erroring():
    gb = _geobox(200, 100, 32)  # 200/32 -> last col width 8, 100/32 -> last row height 4
    tiles = {t.id: t for t in tiling.iter_tiles(gb, tile_size=32)}

    last_col_tile = tiles["0000_0006"]  # row 0, col 6 (7th column, 0-indexed)
    assert last_col_tile.x_slice.stop - last_col_tile.x_slice.start == 8

    last_row_tile = tiles["0003_0000"]  # row 3 (4th row, 0-indexed), col 0
    assert last_row_tile.y_slice.stop - last_row_tile.y_slice.start == 4


def test_tile_ids_are_unique_and_stable():
    gb = _geobox(200, 100, 32)
    tiles = list(tiling.iter_tiles(gb, tile_size=32))
    ids = [t.id for t in tiles]
    assert len(ids) == len(set(ids))


def test_tile_by_id_reconstructs_the_same_tile_as_iteration():
    gb = _geobox(200, 100, 32)
    tiles = list(tiling.iter_tiles(gb, tile_size=32))
    sample = tiles[5]

    rebuilt = tiling.tile_by_id(gb, sample.id, tile_size=32)

    assert rebuilt.row == sample.row
    assert rebuilt.col == sample.col
    assert rebuilt.y_slice == sample.y_slice
    assert rebuilt.x_slice == sample.x_slice
    assert rebuilt.geobox.affine == sample.geobox.affine


def test_default_tile_size_is_2048():
    assert tiling.DEFAULT_TILE_SIZE == 2048

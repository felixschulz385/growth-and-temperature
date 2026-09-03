"""`sql_engine._pixel_id_sql` must reproduce the tile-packed `pixel_id`
`src/analysis` consumes -- i.e. `utils.make_pixel_ids` for the native grid, and
`make_pixel_ids(ix, iy, tile_geobox.zoom_to(resolution))` for a coarsened one,
including the ragged final tile row/column -- and round-trip through
`utils.decode_pixel_id`.
"""

import duckdb
import numpy as np
import pytest
from odc.geo.geobox import GeoBox

from src.data.assemble import sql_engine as se
from src.data.common import tiling
from src.data.assemble.utils import decode_pixel_id, make_pixel_ids

EASE = "EPSG:6933"
RES = 1000.0
W, H, TS = 20, 16, 8   # -> 2 tile-rows x 3 tile-cols, last col tile is width 4


def _geobox():
    gb = GeoBox.from_bbox((0.0, 0.0, W * RES, H * RES), crs=EASE, resolution=RES)
    assert (gb.shape[1], gb.shape[0]) == (W, H)
    return gb


def _sql_pixel_ids(cell_ids, g, factor, dr, dc):
    expr = se._pixel_id_sql("cid::BIGINT", g, factor, dr, dc)
    con = duckdb.connect()
    con.execute("CREATE TABLE t(cid BIGINT)")
    con.executemany("INSERT INTO t VALUES (?)", [(int(x),) for x in cell_ids])
    return {int(c): int(p) for c, p in con.execute(f"SELECT cid, {expr} FROM t").fetchall()}


@pytest.mark.parametrize(
    "factor,tsize",
    [
        (1, 8), (2, 8), (4, 8),   # factor divides the tile size
        (3, 8), (5, 8), (5, 7),   # factor does NOT divide the tile size (real: F=5, TS=2048)
        (10, 7),
    ],
)
def test_pixel_id_sql_matches_make_pixel_ids_every_tile(factor, tsize):
    gb = GeoBox.from_bbox((0.0, 0.0, W * RES, H * RES), crs=EASE, resolution=RES)
    TS = tsize
    g = se.GridFacts(W=W, H=H, TS=TS, F=factor, DR=0, DC=0)
    grid = tiling.build_tile_grid(gb, TS)
    n_rows, n_cols = tiling.grid_shape(gb, TS)

    for ix in range(n_rows):
        for iy in range(n_cols):
            tile = grid[ix, iy]
            zoomed = tile if factor == 1 else tile.zoom_to(resolution=RES * factor)
            ref = make_pixel_ids(ix, iy, zoomed)["pixel_id"].values  # (ch, cw)
            ch, cw = ref.shape

            r0, c0 = ix * TS, iy * TS
            th = min(TS, H - r0)
            tw = min(TS, W - c0)
            cell_ids = [(r0 + rr) * W + (c0 + cc) for rr in range(th) for cc in range(tw)]
            got = _sql_pixel_ids(cell_ids, g, factor, 0, 0)

            for rr in range(th):
                for cc in range(tw):
                    cid = (r0 + rr) * W + (c0 + cc)
                    crl, ccl = rr // factor, cc // factor
                    assert got[cid] == int(ref[crl, ccl]), (
                        f"factor={factor} tile=({ix},{iy}) native=({rr},{cc})"
                    )


def test_pixel_id_sql_round_trips_through_decode():
    g = se.GridFacts(W=W, H=H, TS=TS, F=2, DR=0, DC=0)
    cell_ids = [0, 1, W + 1, 8, 8 * W + 16, H * W - 1]
    got = _sql_pixel_ids(cell_ids, g, 2, 0, 0)
    for cid, pid in got.items():
        nrow, ncol = divmod(cid, W)
        ix, iy, _local = decode_pixel_id(np.uint64(pid))
        assert (ix, iy) == (nrow // TS, ncol // TS)


def test_grid_shake_offset_changes_pixel_id_assignment():
    base = se.GridFacts(W=W, H=H, TS=TS, F=4, DR=0, DC=0)
    shaken = se.GridFacts(W=W, H=H, TS=TS, F=4, DR=2, DC=2)
    cells = [r * W + c for r in range(H) for c in range(W)]
    b = _sql_pixel_ids(cells, base, 4, 0, 0)
    s = _sql_pixel_ids(cells, shaken, 4, 2, 2)
    # a non-trivial fraction of cells land in a different coarse block
    changed = sum(b[c] != s[c] for c in cells)
    assert 0 < changed < len(cells)


@pytest.mark.parametrize("factor,tsize,dr,dc", [(4, 8, 1, 0), (4, 8, 0, 3), (5, 7, 2, 3), (5, 8, 3, 1)])
def test_grid_shake_matches_shifted_block_index_within_tile(factor, tsize, dr, dc):
    """`_pixel_id_sql`'s shake is exactly `(nrl + dr) // factor` on tile-local
    indices (cell_id.shaken_cell_id's convention), with the coarse tile width
    widened to `ceil((tw + dc) / factor)`."""
    import math
    g = se.GridFacts(W=W, H=H, TS=tsize, F=factor, DR=dr, DC=dc)
    for ix in range(math.ceil(H / tsize)):
        for iy in range(math.ceil(W / tsize)):
            r0, c0 = ix * tsize, iy * tsize
            th, tw = min(tsize, H - r0), min(tsize, W - c0)
            cw = math.ceil((tw + dc) / factor)
            cells = [(r0 + rr) * W + (c0 + cc) for rr in range(th) for cc in range(tw)]
            got = _sql_pixel_ids(cells, g, factor, dr, dc)
            for rr in range(th):
                for cc in range(tw):
                    exp_local = ((rr + dr) // factor) * cw + ((cc + dc) // factor)
                    exp = (ix << 48) | (iy << 32) | exp_local
                    assert got[(r0 + rr) * W + (c0 + cc)] == exp


def test_grid_facts_build_shake_is_integer_native_pixels():
    g = se.GridFacts.build(5000.0, (0.5, 0.5), 2048)
    assert (g.F, g.DR, g.DC) == (5, 2, 2)   # round(0.5 * 5) == 2
    assert se.GridFacts.build(None, (0.5, 0.5), 2048).DR == 0  # native grid: no shake

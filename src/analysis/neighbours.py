"""
Neighbour means on a coarse assembled grid (docs/analysis/10km-ring-check.md).

For each cell-year, the mean of a variable over

  ring 1 (``<var>_nb1``): the 8 queen neighbours, and
  ring 2 (``<var>_nb2``): the 16 cells at Chebyshev distance 2,

computed from dense per-year arrays. Neighbour means use every cell with a
finite value, so pass the full grid, not an estimation sample.

Coarse ``pixel_id`` indexing restarts at each native tile
(``src/data/assemble/sql_engine.py::_pixel_id_sql``): global coarse row/col =
tile index * ceil(TS / F) + local offset, and the ragged last tile column is
narrower. The partial cell at each tile edge is treated as an ordinary
neighbour. Only the base (unshaken) grid is supported.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.ndimage import convolve

from src.data.assemble.constants import PIXEL_ID_IX_SHIFT, PIXEL_ID_IY_SHIFT

_RING1 = np.ones((3, 3), np.float32)
_RING1[1, 1] = 0
_RING2 = np.ones((5, 5), np.float32)
_RING2[1:4, 1:4] = 0


def coarse_row_col(pixel_id: np.ndarray, factor: int, W: int, TS: int = 2048):
    """Global (row, col) on the base coarse grid for tile-packed ``pixel_id``s."""
    pid = pixel_id.astype(np.uint64)
    ix = (pid >> np.uint64(PIXEL_ID_IX_SHIFT)) & np.uint64(0xFFFF)
    iy = (pid >> np.uint64(PIXEL_ID_IY_SHIFT)) & np.uint64(0xFFFF)
    loc = (pid & np.uint64(0xFFFFFFFF)).astype(np.int64)
    ix, iy = ix.astype(np.int64), iy.astype(np.int64)
    cw_full = -(-TS // factor)
    tile_w = np.minimum(TS, W - iy * TS)             # native width, ragged last column
    cw = -(-tile_w // factor)
    return ix * cw_full + loc // cw, iy * cw_full + loc % cw


def add_neighbour_means(
    df: pd.DataFrame,
    variables: Iterable[str],
    factor: int,
    W: int,
    TS: int = 2048,
) -> pd.DataFrame:
    """Add ``<var>_nb1`` / ``<var>_nb2`` columns to a ``pixel_id``/``year`` frame.

    *factor* is the coarsening factor (10 for the 10 km grid) and *W* the
    canonical grid width in native cells (``GridFacts.build(...).W``).
    """
    variables = list(variables)
    r, c = coarse_row_col(df["pixel_id"].to_numpy(), factor, W, TS)
    shape = (int(r.max()) + 3, int(c.max()) + 3)
    out = {f"{v}_{k}": np.full(len(df), np.nan, np.float32)
           for v in variables for k in ("nb1", "nb2")}
    for _, idx in df.groupby("year").indices.items():
        ri, ci = r[idx], c[idx]
        for v in variables:
            x = df[v].to_numpy()[idx].astype(np.float32)
            ok = np.isfinite(x)
            val = np.zeros(shape, np.float32)
            has = np.zeros(shape, np.float32)
            val[ri[ok], ci[ok]] = x[ok]
            has[ri[ok], ci[ok]] = 1
            for k, kern in (("nb1", _RING1), ("nb2", _RING2)):
                s = convolve(val, kern, mode="constant")
                n = convolve(has, kern, mode="constant")
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[f"{v}_{k}"][idx] = (s / n)[ri, ci]
    return df.assign(**out)

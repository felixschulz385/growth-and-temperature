"""The panel merge: every per-source block-aggregate CTE is FULL OUTER JOINed on
``(pixel_id, year)`` (annual) / ``(pixel_id)`` (static), so no row is dropped,
NaN nodata never poisons an aggregate, and static values broadcast across years.
"""

import numpy as np
import pytest

from tests.data.assemble.conftest import grid, run_create, write_tiled_source

W, H = 20, 16
GF1 = dict(F=1)  # native grid -> one panel row per (kept cell, year)


def _annual(root, name, value_fn, years=(2000, 2001), keep=None):
    return write_tiled_source(root, name, W=W, H=H, years=list(years), value_fn=value_fn, keep=keep)


def test_full_outer_join_keeps_rows_present_in_only_one_source(tmp_path):
    root = str(tmp_path / "g")
    a = _annual(root, "a", lambda r, c, y: {"a_v": 1.0}, keep={(0, 0), (0, 1)})
    b = _annual(root, "b", lambda r, c, y: {"b_v": 2.0}, keep={(0, 1), (0, 2)})
    datasets = {
        "a": {"path": a, "index_cols": ["pixel_id", "year"], "resampling": "average"},
        "b": {"path": b, "index_cols": ["pixel_id", "year"], "resampling": "average"},
    }
    df = run_create(grid(**GF1), datasets, str(tmp_path / "out"))

    # cells (0,0) a-only, (0,1) both, (0,2) b-only -> 3 pixels x 2 years = 6 rows
    assert len(df) == 6
    a_only = df[df.b_v.isna()]
    b_only = df[df.a_v.isna()]
    assert len(a_only) == 2 and set(a_only.a_v) == {1.0}
    assert len(b_only) == 2 and set(b_only.b_v) == {2.0}


def test_nan_nodata_does_not_poison_average_or_sum(tmp_path):
    root = str(tmp_path / "g")

    def vf(r, c, y):
        # half the native cells in each 2x2 block carry NaN nodata
        return {"v": (np.nan if c % 2 else float(r + c))}

    src = _annual(root, "m", vf, years=(2000,))
    datasets = {"m": {"path": src, "index_cols": ["pixel_id", "year"],
                      "resampling": {"default": "average", "v": "average"}}}
    df = run_create(grid(F=2), datasets, str(tmp_path / "out"))
    # NaN rows became NULL and were ignored, not propagated
    assert df["v"].notna().all()
    # block (rows 0-1, cols 0-1): non-null values are (0,0)=0 and (1,0)=1 -> avg 0.5
    assert df.sort_values(["pixel_id"]).iloc[0]["v"] == pytest.approx(0.5)


def test_static_source_broadcasts_over_every_annual_year(tmp_path):
    root = str(tmp_path / "g")
    ann = _annual(root, "ann", lambda r, c, y: {"t": float(y - 2000)}, years=(2000, 2001, 2002))
    stat = write_tiled_source(root, "stat", W=W, H=H, years=None,
                              value_fn=lambda r, c, y: {"s": 9.0})
    datasets = {
        "ann": {"path": ann, "index_cols": ["pixel_id", "year"], "resampling": "average"},
        "stat": {"path": stat, "index_cols": ["pixel_id"], "resampling": "average"},
    }
    df = run_create(grid(F=2), datasets, str(tmp_path / "out"))
    g0 = df[df.pixel_id == df.pixel_id.min()]
    assert sorted(g0["year"]) == [2000, 2001, 2002]
    assert set(g0["s"]) == {9.0}          # same static value each year
    assert sorted(g0["t"]) == [0.0, 1.0, 2.0]


def test_all_null_block_is_dropped_per_source(tmp_path):
    root = str(tmp_path / "g")
    # source only present on a couple of cells -> most blocks are entirely absent
    src = _annual(root, "sp", lambda r, c, y: {"x": 1.0}, years=(2000,), keep={(0, 0), (4, 4)})
    datasets = {"sp": {"path": src, "index_cols": ["pixel_id", "year"], "resampling": "sum"}}
    df = run_create(grid(F=2), datasets, str(tmp_path / "out"))
    assert len(df) == 2  # exactly the two blocks that had data
    assert set(df["x"]) == {1.0}

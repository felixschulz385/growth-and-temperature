"""Tests for TileProcessor's DuckDB-based merge engine (_duckdb_join /
_merge_dataframes / _merge_update_table), which replaced pandas `pd.merge`
calls. These assert equivalence with the pre-swap pandas merge semantics:
row order, NaN-key handling, and dtype contract."""

import numpy as np
import pandas as pd
import pytest

from src.data.assemble.processors import TileProcessor


def _processor(tmp_path):
    config = {
        "output_path": str(tmp_path / "out"),
        "datasets": {"a": {"path": "does-not-exist.zarr", "index_cols": ["pixel_id"]}},
        "processing": {},
    }
    return TileProcessor(config, str(tmp_path / "out"))


def test_merge_dataframes_outer_join_matches_pandas_values_and_order(tmp_path):
    processor = _processor(tmp_path)
    combined = pd.DataFrame({"pixel_id": [1, 2, 3], "a": [10, 20, 30]})
    df = pd.DataFrame({"pixel_id": [2, 3, 4], "b": [200, 300, 400]})

    expected = pd.merge(combined, df, on=["pixel_id"], how="outer")
    result = processor._merge_dataframes(combined, df, "b", ix=0, iy=0)

    # left-originated rows (pixel_id 1,2,3) keep their original relative
    # order, matching pd.merge's row-order guarantee
    assert list(result["pixel_id"]) == list(expected["pixel_id"])
    pd.testing.assert_frame_equal(
        result.sort_values("pixel_id").reset_index(drop=True)[["pixel_id", "a", "b"]],
        expected.sort_values("pixel_id").reset_index(drop=True)[["pixel_id", "a", "b"]],
        check_dtype=False,
    )


def test_merge_dataframes_handles_nan_merge_keys_like_pandas_outer_merge(tmp_path):
    processor = _processor(tmp_path)
    combined = pd.DataFrame({"pixel_id": [1.0, np.nan, 3.0], "a": [10, 20, 30]})
    df = pd.DataFrame({"pixel_id": [1.0, np.nan, 4.0], "b": [100, 200, 400]})

    expected = pd.merge(combined, df, on=["pixel_id"], how="outer")
    result = processor._merge_dataframes(combined, df, "b", ix=0, iy=0)

    assert len(result) == len(expected)
    # NaN-keyed rows from both sides must survive as unmatched rows, not be dropped
    assert result["a"].isna().sum() == expected["a"].isna().sum()
    assert result["b"].isna().sum() == expected["b"].isna().sum()


def test_merge_dataframes_no_common_columns_returns_combined_unchanged(tmp_path):
    processor = _processor(tmp_path)
    combined = pd.DataFrame({"pixel_id": [1, 2], "a": [10, 20]})
    df = pd.DataFrame({"other_key": [1, 2], "b": [100, 200]})

    result = processor._merge_dataframes(combined, df, "b", ix=0, iy=0)
    pd.testing.assert_frame_equal(result, combined)


def test_merge_dataframes_three_way_chain_composes_correctly(tmp_path):
    processor = _processor(tmp_path)
    a = pd.DataFrame({"pixel_id": [1, 2, 3], "a": [1, 2, 3]})
    b = pd.DataFrame({"pixel_id": [2, 3, 4], "b": [20, 30, 40]})
    c = pd.DataFrame({"pixel_id": [3, 4, 5], "c": [300, 400, 500]})

    combined = processor._merge_dataframes(a, b, "b", ix=0, iy=0)
    combined = processor._merge_dataframes(combined, c, "c", ix=0, iy=0)

    row = combined[combined["pixel_id"] == 3].iloc[0]
    assert row["a"] == 3
    assert row["b"] == 30
    assert row["c"] == 300
    assert set(combined["pixel_id"]) == {1, 2, 3, 4, 5}


def test_merge_update_table_left_join_matches_pandas(tmp_path):
    processor = _processor(tmp_path)
    existing = pd.DataFrame({"pixel_id": [1, 2, 3], "a": [10, 20, 30]})
    update = pd.DataFrame({"pixel_id": [2, 3], "a": [200, 300]})

    # _merge_update_table drops existing's overlapping data columns (here
    # "a") before left-joining update's values in -- so a row absent from
    # `update` (pixel_id=1) gets NaN, not its old value. This mirrors the
    # pre-swap pandas behavior exactly (that drop-then-merge logic is
    # unchanged; only the join engine underneath it was swapped).
    result = processor._merge_update_table(existing, update, ["pixel_id"], context="test")

    assert list(result["pixel_id"]) == [1, 2, 3]
    assert pd.isna(result["a"].iloc[0])
    assert list(result["a"].iloc[1:]) == [200, 300]


def test_merge_update_table_dtype_preserved_when_no_nulls_introduced(tmp_path):
    processor = _processor(tmp_path)
    existing = pd.DataFrame({"pixel_id": [1, 2], "a": pd.array([10, 20], dtype="int64")})
    update = pd.DataFrame({"pixel_id": [1, 2], "a": pd.array([100, 200], dtype="int64")})

    result = processor._merge_update_table(existing, update, ["pixel_id"], context="test")
    assert result["a"].dtype == np.int64

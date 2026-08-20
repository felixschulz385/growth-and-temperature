from __future__ import annotations

import numpy as np
import pytest

from src.viz.grid import bbox_to_row_col, load_matrix


def test_bbox_to_row_col_returns_valid_range(canonical_geobox):
    row0, row1, col0, col1 = bbox_to_row_col((10.0, 45.0, 10.5, 45.3), geobox=canonical_geobox)
    height, width = canonical_geobox.shape
    assert 0 <= row0 < row1 <= height
    assert 0 <= col0 < col1 <= width


def test_bbox_to_row_col_rejects_inverted_bbox(canonical_geobox):
    with pytest.raises(ValueError):
        bbox_to_row_col((10.5, 45.0, 10.0, 45.3), geobox=canonical_geobox)


def test_bbox_to_row_col_rejects_bbox_outside_lat_clip(canonical_geobox):
    # Grid is clipped to |lat| <= 60 deg (docs/design/01-grid.md).
    with pytest.raises(ValueError):
        bbox_to_row_col((10.0, 70.0, 10.5, 71.0), geobox=canonical_geobox)


def test_load_matrix_excludes_decoy_tile(prepare_tree):
    results = load_matrix(
        prepare_tree["root"],
        prepare_tree["variable"],
        bbox=prepare_tree["bbox"],
        years=prepare_tree["years"],
    )
    assert set(results) == set(prepare_tree["years"])
    for matrix_result in results.values():
        # every pixel came from the real region, none from the opposite-corner
        # decoy tile -- if pruning failed, the matrix shape/extent would be
        # dominated by the decoy's location instead.
        row0, row1 = prepare_tree["row_range"]
        col0, col1 = prepare_tree["col_range"]
        expected_rows = row1 - row0
        expected_cols = col1 - col0
        assert matrix_result.matrix.shape == (expected_rows, expected_cols)
        assert np.isfinite(matrix_result.matrix).any()


def test_load_matrix_downsamples_to_max_pixels(prepare_tree):
    row0, row1 = prepare_tree["row_range"]
    col0, col1 = prepare_tree["col_range"]
    full_size = max(row1 - row0, col1 - col0)

    results = load_matrix(
        prepare_tree["root"],
        prepare_tree["variable"],
        bbox=prepare_tree["bbox"],
        years=[prepare_tree["years"][0]],
        max_pixels=max(1, full_size // 3),
    )
    matrix = next(iter(results.values())).matrix
    assert max(matrix.shape) <= max(1, full_size // 3) + 1


def test_load_matrix_rejects_unknown_variable(prepare_tree):
    with pytest.raises(ValueError):
        load_matrix(
            prepare_tree["root"],
            "does_not_exist",
            bbox=prepare_tree["bbox"],
            years=prepare_tree["years"],
        )


def test_load_matrix_rejects_bad_agg(prepare_tree):
    with pytest.raises(ValueError):
        load_matrix(
            prepare_tree["root"],
            prepare_tree["variable"],
            bbox=prepare_tree["bbox"],
            agg="not_a_real_agg",
        )

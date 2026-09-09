from __future__ import annotations

import numpy as np
import pytest

from src.viz.stats import binscatter, corr_matrix, histogram


def test_histogram_matches_numpy_for_identity(panel_parquet, panel_frame):
    h = histogram(panel_parquet, "x", bins=20)
    np_counts, np_edges = np.histogram(panel_frame["x"].to_numpy(), bins=20)
    assert np.allclose(h.edges, np_edges)
    assert np.array_equal(h.counts.astype(int), np_counts)
    assert h.counts.sum() == h.n - h.n_missing


def test_histogram_counts_missing(panel_parquet, panel_frame):
    h = histogram(panel_parquet, "ntl", bins=15, transform="log1p")
    assert h.n_missing == int(panel_frame["ntl"].isna().sum())
    assert h.counts.sum() == h.n - h.n_missing
    assert h.transform == "log1p"


def test_histogram_groupby(panel_parquet, panel_frame):
    h = histogram(panel_parquet, "y", bins=10, groupby="country")
    assert h.counts.shape == (panel_frame["country"].nunique(), 10)
    assert h.counts.sum() == h.n - h.n_missing
    assert h.groups == sorted(panel_frame["country"].unique())


def test_histogram_weighted_bin_totals(panel_parquet, panel_frame):
    h = histogram(panel_parquet, "x", bins=12, weights="ntl")
    expected = panel_frame["ntl"].sum()  # NaN weights drop out
    assert h.counts.sum() == pytest.approx(expected, rel=1e-9)


def test_histogram_rejects_unknown_column(panel_parquet):
    with pytest.raises(ValueError):
        histogram(panel_parquet, "does_not_exist")


def test_histogram_rejects_unknown_transform(panel_parquet):
    with pytest.raises(ValueError):
        histogram(panel_parquet, "x", transform="sqrt")


def test_binscatter_recovers_slope(panel_parquet):
    bs = binscatter(panel_parquet, y="y", x="x", bins=25)
    assert bs.slope == pytest.approx(2.0, abs=0.15)
    assert bs.x_mean.shape == (25,)
    assert bs.y_se is not None and np.all(bs.y_se >= 0)
    assert bs.n.sum() == pytest.approx(800, abs=1)


def test_binscatter_groupby_shapes(panel_parquet, panel_frame):
    n_groups = panel_frame["country"].nunique()
    bs = binscatter(panel_parquet, y="y", x="x", bins=10, groupby="country")
    assert bs.x_mean.shape == (n_groups, 10)
    assert set(bs.slopes) == set(panel_frame["country"].unique())


def test_binscatter_weighted_has_no_se(panel_parquet):
    bs = binscatter(panel_parquet, y="y", x="x", bins=10, weights="ntl")
    assert bs.y_se is None


def test_corr_matrix_matches_numpy(panel_parquet, panel_frame):
    cols = ["x", "y", "treat"]
    cm = corr_matrix(panel_parquet, cols)
    expected = np.corrcoef(panel_frame[cols].to_numpy().T)
    assert np.allclose(cm.matrix, expected, atol=1e-9)
    assert cm.labels == cols
    assert np.all(np.diag(cm.n) == len(panel_frame))


def test_corr_matrix_spearman_runs(panel_parquet):
    cm = corr_matrix(panel_parquet, ["x", "y", "ntl"], method="spearman")
    assert cm.method == "spearman"
    assert np.allclose(np.diag(cm.matrix), 1.0)
    assert cm.matrix[0, 1] == pytest.approx(cm.matrix[1, 0])


def test_corr_matrix_from_dataframe_source(panel_frame):
    cm = corr_matrix(panel_frame, ["x", "y"])
    assert cm.matrix.shape == (2, 2)


def test_stats_accept_shared_connection(panel_parquet):
    import duckdb

    con = duckdb.connect()
    h = histogram(panel_parquet, "x", bins=10, con=con)
    bs = binscatter(panel_parquet, y="y", x="x", bins=10, con=con)
    assert h.counts.sum() > 0 and bs.slope is not None
    con.execute("SELECT 1")  # not closed by the stat calls
    con.close()

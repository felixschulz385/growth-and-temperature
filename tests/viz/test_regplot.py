from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.viz import stats as viz_stats
from src.viz.models import FittedModel
from src.viz.regplot import (
    plot_binscatter,
    plot_coefficients,
    plot_corr_matrix,
    plot_distribution,
    plot_residuals,
)

_RESULTS_JSON = {
    "analysis_metadata": {"analysis_type": "duckreg", "formula": "y ~ x", "fixed_effects": "unit"},
    "coefficients": {
        "names": ["x", "treat"],
        "estimates": [2.0, 0.5],
        "std_errors": [0.1, 0.2],
        "p_values": [0.0, 0.01],
    },
}


def _fig_ok(fig):
    try:
        assert fig.__class__.__name__ == "Figure"
        assert len(fig.axes) >= 1
    finally:
        plt.close(fig)


def test_plot_distribution_hist(panel_parquet):
    _fig_ok(plot_distribution(panel_parquet, "x"))


def test_plot_distribution_ecdf_with_overlay(panel_parquet):
    _fig_ok(
        plot_distribution(
            panel_parquet, "ntl", kind="ecdf", transform="log1p", overlay_transform="identity"
        )
    )


def test_plot_distribution_logy_and_vline(panel_parquet):
    fig = plot_distribution(panel_parquet, "ntl", transform="identity", logy=True, vlines=1.0)
    try:
        ax = fig.axes[0]
        assert ax.get_yscale() == "log"
        assert any(line.get_linestyle() == "--" for line in ax.get_lines())
    finally:
        plt.close(fig)


def test_plot_distribution_groupby_has_legend(panel_parquet):
    fig = plot_distribution(panel_parquet, "y", groupby="country")
    try:
        assert fig.axes[0].get_legend() is not None
    finally:
        plt.close(fig)


def test_plot_binscatter(panel_parquet):
    _fig_ok(plot_binscatter(panel_parquet, y="y", x="x"))


def test_plot_binscatter_groupby(panel_parquet):
    _fig_ok(plot_binscatter(panel_parquet, y="y", x="x", groupby="country"))


def test_plot_corr_matrix(panel_parquet):
    _fig_ok(plot_corr_matrix(panel_parquet, ["x", "y", "treat", "ntl"]))


def test_plot_coefficients_single(panel_parquet):
    _fig_ok(plot_coefficients(_RESULTS_JSON))


def test_plot_coefficients_multiple_has_legend():
    fig = plot_coefficients([_RESULTS_JSON, _RESULTS_JSON], labels=["a", "b"])
    try:
        leg = fig.axes[0].get_legend()
        assert leg is not None and len(leg.get_texts()) == 2
    finally:
        plt.close(fig)


def test_plot_coefficients_drop_intercept():
    with_int = {
        "analysis_metadata": {"analysis_type": "duckreg"},
        "coefficients": {"names": ["Intercept", "x"], "estimates": [1.0, 2.0], "std_errors": [0.1, 0.1]},
    }
    fig = plot_coefficients(with_int)
    try:
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert "x" in labels and "Intercept" not in labels
    finally:
        plt.close(fig)


def test_precomputed_dataclass_does_not_touch_duckdb(panel_parquet, monkeypatch):
    hist = viz_stats.histogram(panel_parquet, "x", bins=10)
    bs = viz_stats.binscatter(panel_parquet, y="y", x="x", bins=10)
    cm = viz_stats.corr_matrix(panel_parquet, ["x", "y"])

    def _boom(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("resolve_source called for a precomputed dataclass")

    monkeypatch.setattr(viz_stats, "resolve_source", _boom)
    _fig_ok(plot_distribution(hist))
    _fig_ok(plot_binscatter(bs))
    _fig_ok(plot_corr_matrix(cm))


def test_plot_residuals_requires_residuals():
    fm = FittedModel(coef_table=__import__("pandas").DataFrame({"term": ["x"], "estimate": [1.0]}))
    with pytest.raises(ValueError):
        plot_residuals(fm)


def test_plot_residuals_pyfixest(panel_frame):
    pf = pytest.importorskip("pyfixest")
    fit = pf.feols("y ~ x + treat | unit + year", data=panel_frame)
    _fig_ok(plot_residuals(fit, kind="vs_fitted"))
    _fig_ok(plot_residuals(fit, kind="qq"))

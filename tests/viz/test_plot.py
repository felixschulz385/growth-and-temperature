from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from src.viz.plot import plot_grid


def test_plot_grid_single_year(prepare_tree):
    fig = plot_grid(
        prepare_tree["root"],
        prepare_tree["variable"],
        bbox=prepare_tree["bbox"],
        years=[prepare_tree["years"][0]],
    )
    try:
        assert len(fig.axes) >= 1
    finally:
        plt.close(fig)


def test_plot_grid_facet_multiple_years(prepare_tree):
    fig = plot_grid(
        prepare_tree["root"],
        prepare_tree["variable"],
        bbox=prepare_tree["bbox"],
        years=prepare_tree["years"],
        facet=True,
    )
    try:
        # one imshow axes per year plus a colorbar axes
        n_years = len(prepare_tree["years"])
        assert sum(1 for ax in fig.axes if ax.images) == n_years
    finally:
        plt.close(fig)


def test_plot_grid_multiple_years_without_facet_raises(prepare_tree):
    with pytest.raises(ValueError):
        plot_grid(
            prepare_tree["root"],
            prepare_tree["variable"],
            bbox=prepare_tree["bbox"],
            years=prepare_tree["years"],
            facet=False,
        )

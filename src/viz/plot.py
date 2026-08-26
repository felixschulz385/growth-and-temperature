"""Matplotlib rendering for EASE6933 `cell_id` PREPARE parquet grids.

Renders `Figure`s only -- callers decide whether to `savefig`, `show`, or
embed in a notebook (no file I/O in this module).
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from src.viz.grid import load_matrix


def plot_grid(
    input_dir: str,
    variable: str,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    years: list[int] | None = None,
    facet: bool = False,
    max_pixels: int = 800,
    agg: str = "mean",
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
):
    """Plot one variable from a PREPARE `cell_id` parquet tree.

    - `years=[Y]` (or a single resolved year with `years=None, facet=False`):
      one panel.
    - `years=[...]` with 2+ years, or `years=None, facet=True` (all years
      present in the queried region): one panel per year, faceted, sharing
      one colorbar.

    Returns the `matplotlib.figure.Figure`.
    """
    results = load_matrix(
        input_dir,
        variable,
        bbox=bbox,
        years=years,
        max_pixels=max_pixels,
        agg=agg,
    )

    if not results:
        raise ValueError("No data found for the given input_dir/variable/bbox/years")

    resolved_years = sorted(results)

    if len(resolved_years) > 1 and not facet:
        raise ValueError(
            f"Query resolved {len(resolved_years)} years ({resolved_years}); "
            "pass facet=True to plot them all, or years=[<one year>] to pick one."
        )

    finite = [r.matrix[~np.isnan(r.matrix)] for r in results.values()]
    finite = [f for f in finite if f.size]
    if not finite:
        raise ValueError("No non-NaN pixels found for the given input_dir/variable/bbox/years")
    vmin = float(min(f.min() for f in finite))
    vmax = float(max(f.max() for f in finite))

    n = len(resolved_years)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (4 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    im = None
    for ax, year in zip(axes_flat, resolved_years):
        result = results[year]
        im = ax.imshow(
            result.matrix,
            extent=result.extent_lonlat,
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_title(str(year))
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")

    for ax in axes_flat[n:]:
        ax.axis("off")

    if im is not None:
        fig.colorbar(im, ax=axes_flat[:n].tolist(), label=variable, shrink=0.8)

    return fig

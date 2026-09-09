"""Shared near-square small-multiples layout.

Extracted from ``plot_grid``'s year-facet loop so the regression renderers
(``src.viz.regplot``) get the same behaviour for free.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import matplotlib.pyplot as plt


def facet_grid(
    items: Sequence,
    draw: Callable,
    *,
    figsize: tuple[float, float] | None = None,
    panel_size: tuple[float, float] = (4.0, 3.5),
    sharex: bool = False,
    sharey: bool = False,
):
    """Lay ``items`` out on a near-square grid of axes, call ``draw(ax, item)``
    for each, switch the leftover axes off, and return ``(fig, used_axes)`` where
    ``used_axes`` is the list of axes that actually got an item.
    """
    n = len(items)
    if n == 0:
        raise ValueError("facet_grid: no items to plot")

    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    panel_w, panel_h = panel_size

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize or (panel_w * ncols, panel_h * nrows),
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
    )
    axes_flat = list(axes.flatten())
    for ax, item in zip(axes_flat, items):
        draw(ax, item)
    for ax in axes_flat[n:]:
        ax.axis("off")
    return fig, axes_flat[:n]

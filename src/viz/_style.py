"""Seaborn-like matplotlib styling for the regression-plot renderers.

Pure matplotlib by default -- seaborn is imported lazily and only when a caller
passes ``use_seaborn=True`` *and* it is installed, so importing ``src.viz`` never
requires seaborn.
"""

from __future__ import annotations

import contextlib

import matplotlib as mpl

# Muted, colour-blind-friendly cycle (seaborn "deep"-ish), kept local so the
# look does not depend on seaborn being importable.
_PALETTE = [
    "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3",
    "#937860", "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd",
]


def palette(n: int) -> list[str]:
    """``n`` hex colours, cycling the base palette when ``n`` exceeds its length."""
    if n <= 0:
        return []
    reps = (n + len(_PALETTE) - 1) // len(_PALETTE)
    return (_PALETTE * reps)[:n]


def apply(ax) -> None:
    """Despine top/right, add a light horizontal grid, soften the ticks."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, labelsize="small")


@contextlib.contextmanager
def style_context(use_seaborn: bool = False):
    """Wrap a renderer body. Applies seaborn's ``whitegrid`` theme when asked and
    available; otherwise a light ``rcParams`` tweak. Either way the palette above
    is the active property cycle.
    """
    rc = {
        "axes.prop_cycle": mpl.cycler(color=_PALETTE),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
    }
    if use_seaborn:
        try:
            import seaborn as sns

            with sns.axes_style("whitegrid"), mpl.rc_context(rc):
                yield
            return
        except ImportError:
            pass
    with mpl.rc_context(rc):
        yield

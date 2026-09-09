"""Matplotlib renderers for regression-analysis distribution plots.

Every function returns a ``matplotlib.figure.Figure`` and does no file I/O
(callers decide whether to ``savefig`` / ``show``), matching ``src.viz.plot``.

Each renderer takes **either** a precomputed dataclass from ``src.viz.stats`` /
a ``FittedModel`` **or** a raw data source plus column arguments, in which case
it computes the statistic first.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.viz import _style
from src.viz.models import FittedModel, as_fitted_model
from src.viz.stats import BinScatter, CorrMatrix, Histogram, binscatter, corr_matrix, histogram


def _new_ax(ax, figsize):
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize or (7, 4.5))
    return fig, ax


# ---------------------------------------------------------------------------
# distribution
# ---------------------------------------------------------------------------


def plot_distribution(
    data_or_source,
    col: str | None = None,
    *,
    kind: str = "hist",
    transform: str = "identity",
    offset: float = 0.01,
    overlay_transform: str | None = None,
    groupby: str | None = None,
    bins: int = 50,
    value_range=None,
    weights: str | None = None,
    logy: bool = False,
    vlines=None,
    ax=None,
    figsize=None,
    con=None,
    table: str | None = None,
    use_seaborn: bool = False,
):
    """Histogram (``kind="hist"``) or binned ECDF (``kind="ecdf"``) of a column.

    ``overlay_transform`` draws a second faint histogram of the same column under
    another transform (e.g. raw vs ``log1p``) -- only when ``data_or_source`` is a
    source, not a precomputed :class:`Histogram`.

    ``logy=True`` puts the count axis on a log scale (``kind="hist"`` only) -- the
    usual way to read a zero-inflated, heavy-tailed column without transforming
    it. ``vlines`` marks one or more x positions (e.g. a threshold).
    """
    if kind not in ("hist", "ecdf"):
        raise ValueError("kind must be 'hist' or 'ecdf'")

    hists: list[Histogram] = []
    if isinstance(data_or_source, Histogram):
        hists.append(data_or_source)
    else:
        if col is None:
            raise ValueError("col is required when passing a data source")
        hists.append(
            histogram(
                data_or_source, col, bins=bins, value_range=value_range,
                transform=transform, offset=offset, groupby=groupby,
                weights=weights, con=con, table=table,
            )
        )
        if overlay_transform:
            hists.append(
                histogram(
                    data_or_source, col, bins=bins, transform=overlay_transform,
                    offset=offset, weights=weights, con=con, table=table,
                )
            )

    with _style.style_context(use_seaborn):
        fig, ax = _new_ax(ax, figsize)
        for depth, hist in enumerate(hists):
            _draw_hist(ax, hist, kind=kind, alpha=1.0 if depth == 0 else 0.35, logy=logy)
        if logy and kind == "hist":
            ax.set_yscale("log")
        if vlines is not None:
            for xv in (vlines if isinstance(vlines, (list, tuple)) else [vlines]):
                ax.axvline(float(xv), color="0.35", ls="--", lw=1)
        ax.set_xlabel(_label(hists[0]))
        ax.set_ylabel("share" if kind == "ecdf" else ("weight" if weights else "count"))
        title = hists[0].col
        if hists[0].n_missing:
            title += f"  (n={hists[0].n - hists[0].n_missing}, {hists[0].n_missing} missing)"
        ax.set_title(title)
        _style.apply(ax)
        if hists[0].groups is not None:
            ax.legend(title=hists[0].group, fontsize="small")
    return fig


def _label(hist: Histogram) -> str:
    return hist.col if hist.transform == "identity" else f"{hist.transform}({hist.col})"


def _draw_hist(ax, hist: Histogram, *, kind: str, alpha: float, logy: bool = False) -> None:
    counts = hist.counts if hist.counts.ndim == 2 else hist.counts[None, :]
    labels = hist.groups if hist.groups is not None else [None]
    colors = _style.palette(len(labels))
    width = np.diff(hist.edges)
    for row, label, color in zip(counts, labels, colors):
        if kind == "ecdf":
            total = row.sum()
            y = np.cumsum(row) / total if total else np.zeros_like(row)
            ax.step(hist.edges[1:], y, where="post", label=label, color=color, alpha=alpha)
        else:
            heights = np.where(row > 0, row, np.nan) if logy else row
            ax.bar(
                hist.edges[:-1], heights, width=width, align="edge",
                label=label, color=color, alpha=alpha * 0.85, edgecolor="none",
            )


# ---------------------------------------------------------------------------
# binscatter
# ---------------------------------------------------------------------------


def plot_binscatter(
    data_or_source,
    y: str | None = None,
    x: str | None = None,
    *,
    bins: int = 20,
    x_transform: str = "identity",
    y_transform: str = "identity",
    offset: float = 0.01,
    groupby: str | None = None,
    weights: str | None = None,
    fit_line: bool = True,
    ci: bool = True,
    ax=None,
    figsize=None,
    con=None,
    table: str | None = None,
    use_seaborn: bool = False,
):
    """Quantile-binned mean of ``y`` against ``x`` with per-bin SE whiskers and an
    OLS line through the binned points.
    """
    if isinstance(data_or_source, BinScatter):
        bs = data_or_source
    else:
        if y is None or x is None:
            raise ValueError("y and x are required when passing a data source")
        bs = binscatter(
            data_or_source, y, x, bins=bins, x_transform=x_transform,
            y_transform=y_transform, offset=offset, groupby=groupby,
            weights=weights, fit_line=fit_line, con=con, table=table,
        )

    grouped = bs.x_mean.ndim == 2
    series = (
        list(zip(bs.groups, bs.x_mean, bs.y_mean, _se_rows(bs)))
        if grouped
        else [(None, bs.x_mean, bs.y_mean, bs.y_se)]
    )
    colors = _style.palette(len(series))

    with _style.style_context(use_seaborn):
        fig, ax = _new_ax(ax, figsize)
        for (label, xm, ym, se), color in zip(series, colors):
            ok = np.isfinite(xm) & np.isfinite(ym)
            if ci and se is not None:
                ax.errorbar(
                    xm[ok], ym[ok], yerr=se[ok], fmt="o", ms=4, color=color,
                    ecolor=color, elinewidth=0.8, capsize=2, label=label,
                )
            else:
                ax.plot(xm[ok], ym[ok], "o", ms=4, color=color, label=label)
            slope, intercept = _line_for(bs, label)
            if fit_line and slope is not None and ok.any():
                xs = np.array([xm[ok].min(), xm[ok].max()])
                ax.plot(xs, intercept + slope * xs, "-", lw=1.3, color=color, alpha=0.9)
        ax.set_xlabel(f"{x_transform}({bs.x})" if x_transform != "identity" else bs.x)
        ax.set_ylabel(f"{y_transform}({bs.y})" if y_transform != "identity" else bs.y)
        ax.set_title(f"{bs.y} vs {bs.x}  ({_nbins(bs)} bins)")
        _style.apply(ax)
        if grouped:
            ax.legend(title=bs.group, fontsize="small")
    return fig


def _se_rows(bs: BinScatter):
    if bs.y_se is None:
        return [None] * len(bs.groups)
    return list(bs.y_se)


def _line_for(bs: BinScatter, label):
    if bs.slopes is not None:
        return bs.slopes.get(label, (None, None))
    return (bs.slope, bs.intercept)


def _nbins(bs: BinScatter) -> int:
    return bs.x_mean.shape[-1]


# ---------------------------------------------------------------------------
# correlation matrix
# ---------------------------------------------------------------------------


def plot_corr_matrix(
    data_or_source,
    cols: list[str] | None = None,
    *,
    method: str = "pearson",
    weights: str | None = None,
    annot: bool = True,
    ax=None,
    figsize=None,
    con=None,
    table: str | None = None,
    use_seaborn: bool = False,
):
    """Heatmap of a pairwise correlation matrix (collinearity screen)."""
    if isinstance(data_or_source, CorrMatrix):
        cm = data_or_source
    else:
        if not cols:
            raise ValueError("cols is required when passing a data source")
        cm = corr_matrix(
            data_or_source, cols, method=method, weights=weights, con=con, table=table
        )

    k = len(cm.labels)
    with _style.style_context(use_seaborn):
        fig, ax = _new_ax(ax, figsize or (1.1 * k + 2, 1.1 * k + 1.5))
        im = ax.imshow(cm.matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(k), cm.labels, rotation=45, ha="right", fontsize="small")
        ax.set_yticks(range(k), cm.labels, fontsize="small")
        ax.set_title(f"{cm.method} correlation")
        if annot:
            for i in range(k):
                for j in range(k):
                    val = cm.matrix[i, j]
                    if np.isfinite(val):
                        ax.text(
                            j, i, f"{val:.2f}", ha="center", va="center",
                            color="white" if abs(val) > 0.5 else "black", fontsize="small",
                        )
        fig.colorbar(im, ax=ax, shrink=0.8, label="r")
    return fig


# ---------------------------------------------------------------------------
# coefficient plot
# ---------------------------------------------------------------------------


def plot_coefficients(
    models,
    *,
    terms: list[str] | None = None,
    labels: list[str] | None = None,
    drop_intercept: bool = True,
    ax=None,
    figsize=None,
    use_seaborn: bool = False,
):
    """Dot-and-whisker plot of one or several fitted models / results-JSON files.

    ``models`` is a :class:`FittedModel`, a pyfixest fit, a results-JSON path or
    dict, or a list mixing any of those.
    """
    if isinstance(models, (FittedModel, dict, str)) or not _is_sequence(models):
        models = [models]
    fitted = [as_fitted_model(m) for m in models]
    if labels:
        for fm, lab in zip(fitted, labels):
            fm.label = lab
    names = [fm.label or f"model {i + 1}" for i, fm in enumerate(fitted)]

    all_terms = terms or _term_union(fitted, drop_intercept)
    y_base = np.arange(len(all_terms))[::-1]
    n_models = len(fitted)
    offsets = np.linspace(-0.28, 0.28, n_models) if n_models > 1 else [0.0]
    colors = _style.palette(n_models)

    with _style.style_context(use_seaborn):
        fig, ax = _new_ax(ax, figsize or (7, 0.6 * len(all_terms) + 1.5))
        ax.axvline(0.0, color="0.6", lw=1, zorder=1)
        for fm, name, dy, color in zip(fitted, names, offsets, colors):
            tbl = fm.coef_table.set_index("term")
            # A duplicated term name would make tbl.loc[term] a DataFrame and
            # feed ax.errorbar a ragged list; keep the first row per term.
            tbl = tbl[~tbl.index.duplicated(keep="first")]
            xs, ys, lo, hi = [], [], [], []
            for ti, term in enumerate(all_terms):
                if term not in tbl.index:
                    continue
                row = tbl.loc[term]
                xs.append(row["estimate"])
                ys.append(y_base[ti] + dy)
                lo.append(row["estimate"] - row["ci_low"])
                hi.append(row["ci_high"] - row["estimate"])
            if not xs:
                continue
            ax.errorbar(
                xs, ys, xerr=[lo, hi], fmt="o", ms=5, color=color, ecolor=color,
                elinewidth=1.1, capsize=2, label=name,
            )
        ax.set_yticks(y_base, all_terms)
        ax.set_xlabel("coefficient (95% CI)")
        _style.apply(ax)
        ax.grid(True, axis="x", color="0.9", linewidth=0.6)
        if n_models > 1:
            ax.legend(fontsize="small")
    return fig


def _is_sequence(obj) -> bool:
    return isinstance(obj, (list, tuple))


def _term_union(fitted, drop_intercept: bool) -> list[str]:
    seen: list[str] = []
    for fm in fitted:
        for term in fm.coef_table["term"].tolist():
            if drop_intercept and str(term).lower() in ("intercept", "(intercept)", "const"):
                continue
            if term not in seen:
                seen.append(term)
    return seen


# ---------------------------------------------------------------------------
# residual diagnostics
# ---------------------------------------------------------------------------


def plot_residuals(
    model,
    *,
    kind: str = "vs_fitted",
    ax=None,
    figsize=None,
    use_seaborn: bool = False,
):
    """Residuals-vs-fitted (``kind="vs_fitted"``) or normal QQ (``kind="qq"``).

    Requires row-level residuals, i.e. a pyfixest-backed :class:`FittedModel`.
    """
    if kind not in ("vs_fitted", "qq"):
        raise ValueError("kind must be 'vs_fitted' or 'qq'")
    fm = as_fitted_model(model)
    resid = fm.residuals

    with _style.style_context(use_seaborn):
        fig, ax = _new_ax(ax, figsize or (6.5, 4.5))
        if kind == "vs_fitted":
            within = bool(fm.fe_dims)
            ax.scatter(fm.fitted, resid, s=10, alpha=0.5, color=_style.palette(1)[0])
            ax.axhline(0.0, color="0.6", lw=1)
            ax.set_xlabel("fitted (within FE)" if within else "fitted")
            ax.set_ylabel("residual")
            ax.set_title("residuals vs fitted")
        else:
            from scipy import stats as _sps

            _sps.probplot(resid, dist="norm", plot=ax)
            ax.set_title("normal QQ plot")
            ax.get_lines()[0].set(marker="o", ms=4, alpha=0.5)
        _style.apply(ax)
    return fig

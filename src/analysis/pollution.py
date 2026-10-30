"""
The pollution-institutions successor's region panel
(docs/successors/pollution-institutions-prespec.md), on top of ``spec.py``.

``build_region_panel()`` averages the assembled 10 km panel's PM2.5 over GADM
4.1 first-level regions (every cell, not only the cells with night LST: this
project's outcome is PM2.5) and joins DOSE output, population and weather and
V-Dem's baseline institution terciles. Fits use pyfixest, like ``dose.py``.
"""
from __future__ import annotations

import json
import os

import duckdb
import numpy as np
import pandas as pd

from src.analysis.spec import DATA, GRID, SRC, THREADS

DOSE = f"{DATA}/prepared/dose/adm/dose_gid1.parquet"
VDEM = f"{DATA}/prepared/vdem/misc/vdem_country_year.parquet"
GID1_MAP = f"{DATA}/prepared/misc/adm/gadm/GID_1_code_mapping.json"
REGION_PANEL = f"{DATA}/_pollution_region_panel_{GRID}.parquet"

YEARS = (1998, 2018)              # ACAG starts 1998; DOSE P_a and T_a stop 2018
BASELINE = (1998, 2000)           # institutions fixed at their 1998-2000 mean
KNOTS_USD = (1000, 4000, 12000)   # spline knots, 2015 USD per person
DESERT_BIOME = 13                 # RESOLVE deserts and xeric shrublands
INSTITUTIONS = {"democracy": "v2x_polyarchy", "rule of law": "v2x_rule"}
FE_MAIN = "GID_1 + tercile^year"
CL_CTY = {"CRV1": "GID_0"}


def _con():
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute("SET enable_progress_bar = false")
    return con


def spline_terms(log_y: pd.Series, knots=KNOTS_USD) -> pd.DataFrame:
    """Piecewise-linear basis in ln output: ``seg0`` is ln y capped at the
    first knot; ``seg<k>`` the part of ln y between knot k and k+1."""
    cuts = [-np.inf, *np.log(knots), np.inf]
    out = {}
    for k in range(len(cuts) - 1):
        lo, hi = cuts[k], cuts[k + 1]
        if k == 0:
            out["seg0"] = np.minimum(log_y, hi)
        else:
            out[f"seg{k}"] = np.clip(log_y - lo, 0, hi - lo)
    return pd.DataFrame(out, index=log_y.index)


def segment(log_y: pd.Series, knots=KNOTS_USD) -> pd.Series:
    """Which spline segment (0-3) each value of ln output falls in."""
    return pd.Series(np.searchsorted(np.log(knots), log_y), index=log_y.index)


def baseline_terciles(vdem: pd.DataFrame, countries, col: str) -> pd.Series:
    """Terciles (0 low, 1 middle, 2 high) of each country's baseline mean of
    *col*, across *countries*, each country counted once."""
    base = (vdem[vdem.year.between(*BASELINE) & vdem.country_text_id.isin(countries)]
            .groupby("country_text_id")[col].mean().dropna())
    return pd.qcut(base, 3, labels=False).astype("int8")


def build_region_panel(path: str = REGION_PANEL) -> pd.DataFrame:
    """Region x year, 1998-2018, DOSE-matched regions: PM2.5, desert share,
    output, population, weather, spline terms and institution terciles. Cached."""
    if os.path.exists(path):
        return pd.read_parquet(path)
    with open(GID1_MAP) as f:
        codes = pd.DataFrame([(v, k) for k, v in json.load(f).items()], columns=["gid1_id", "GID_1"])
    con = _con()
    con.register("codes", codes)
    pm = con.execute(f"""
        SELECT c.GID_1, p.year, avg(p.pm25) AS pm25, count(*) AS n_cells,
               avg((p.biome_id = {DESERT_BIOME})::INT) AS desert_share
        FROM read_parquet('{SRC}', union_by_name = true, hive_partitioning = true) p
        JOIN codes c ON c.gid1_id = p.GID_1
        WHERE p.pm25 IS NOT NULL AND p.year BETWEEN {YEARS[0]} AND {YEARS[1]}
        GROUP BY 1, 2
    """).df()
    dose = con.execute(f"""
        SELECT GID_1_gadm41 AS GID_1, year, GID_0, grp_pc_lcu2015_usd AS y_usd, pop, P_a, T_a
        FROM '{DOSE}' WHERE year BETWEEN {YEARS[0]} AND {YEARS[1]}
    """).df()
    con.close()
    # a region's desert share is static: its mean over the years
    pm["desert_share"] = pm.groupby("GID_1").desert_share.transform("mean")
    out = dose.merge(pm, on=["GID_1", "year"], how="inner")
    out["log_pm25"] = np.log(out.pm25)
    out["log_y"] = np.log(out.y_usd)
    out = out.join(spline_terms(out.log_y.where(out.y_usd > 0)))
    out["segment"] = segment(out.log_y.fillna(-np.inf)).where(out.y_usd > 0)

    vdem = pd.read_parquet(VDEM)
    # the sample's countries: any region-year with output and weather
    eligible = out.loc[out.y_usd.gt(0) & out.P_a.notna() & out.T_a.notna(), "GID_0"].unique()
    for name, col in INSTITUTIONS.items():
        key = "tercile" if name == "democracy" else "tercile_rule"
        terc = baseline_terciles(vdem, eligible, col)
        out[key] = out.GID_0.map(terc)
    out = out.merge(vdem[["country_text_id", "year", "v2x_polyarchy"]]
                    .rename(columns={"country_text_id": "GID_0"}), on=["GID_0", "year"], how="left")
    out.to_parquet(path, index=False)
    return out


def sample(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of the main specification: output, weather and a democracy tercile."""
    m = df.y_usd.gt(0) & df.pm25.gt(0) & df.P_a.notna() & df.T_a.notna() & df.tercile.notna()
    return df[m].copy()


# -- estimation (prespec §3, §4 and the 2026-10-29 change log) -----------------

MIN_REGIONS = 50                  # a supported tercile x segment cell
SEGMENTS = ("seg0", "seg1", "seg2", "seg3")
SEGMENT_LABELS = ("< 1,000", "1,000-4,000", "4,000-12,000", "> 12,000")


def supported_cells(df: pd.DataFrame, group: str = "tercile") -> pd.DataFrame:
    """Regions with income in each segment, per group, and whether the cell
    is supported (>= MIN_REGIONS regions)."""
    n = df.groupby([group, "segment"]).GID_1.nunique().unstack(fill_value=0)
    n = n.reindex(columns=range(len(SEGMENTS)), fill_value=0)
    n.columns = list(SEGMENTS)
    return n


def comparisons(n: pd.DataFrame) -> list[tuple[str, int, int]]:
    """The supported comparisons: (segment, group a, group b) for adjacent
    supported groups in each segment (two restrictions when all three are)."""
    out = []
    for s in SEGMENTS:
        ok = [int(g) for g in sorted(n.index) if n.loc[g, s] >= MIN_REGIONS]
        out += [(s, a, b) for a, b in zip(ok, ok[1:])]
    return out


def add_interactions(df: pd.DataFrame, group: str = "tercile") -> pd.DataFrame:
    df = df.copy()
    for s in SEGMENTS:
        for g in sorted(df[group].dropna().unique()):
            df[f"{s}_g{int(g)}"] = df[s] * (df[group] == g)
    return df


def add_country_trends(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Variant 1: one linear trend per country, as regressors."""
    df = df.copy()
    t = df.year - YEARS[0]
    cols = []
    for c in sorted(df.GID_0.unique()):
        col = f"trend_{c}"
        df[col] = t * (df.GID_0 == c)
        cols.append(col)
    return df, cols


def fit_paths(df: pd.DataFrame, group: str = "tercile", y: str = "log_pm25", fe: str | None = None,
              extra: list[str] | None = None, weights: str | None = None, vcov=CL_CTY,
              controls=("P_a", "T_a")) -> dict:
    """Group-specific spline paths. Returns the slope table (with support),
    the supported comparisons and their joint Wald test (F with G - 1 df).
    *controls* are the weather terms (none in the lights follow-up)."""
    import pyfixest as pf
    from scipy import stats

    d = add_interactions(df, group)
    fe = fe or f"GID_1 + {group}^year"
    groups = sorted(d[group].dropna().unique().astype(int))
    terms = [f"{s}_g{g}" for s in SEGMENTS for g in groups]
    rhs = " + ".join(terms + list(controls) + (extra or []))
    m = pf.feols(f"{y} ~ {rhs} | {fe}", data=d, vcov=vcov, weights=weights)
    tidy = m.tidy()
    n = supported_cells(d, group)
    rows = []
    for s, lab in zip(SEGMENTS, SEGMENT_LABELS):
        for g in groups:
            t = f"{s}_g{g}"
            r = tidy.loc[t] if t in tidy.index else None
            rows.append(dict(segment=lab, group=g, regions=int(n.loc[g, s]),
                             supported=bool(n.loc[g, s] >= MIN_REGIONS),
                             est=None if r is None else r["Estimate"],
                             se=None if r is None else r["Std. Error"],
                             p=None if r is None else r["Pr(>|t|)"]))
    slopes = pd.DataFrame(rows)

    # joint Wald test of the supported comparisons
    names = list(m._coefnames)
    b = np.asarray(m.coef().reindex(names))
    V = np.asarray(m._vcov)
    comps = [(s, a, bb) for s, a, bb in comparisons(n) if f"{s}_g{a}" in names and f"{s}_g{bb}" in names]
    R = np.zeros((len(comps), len(names)))
    diffs = []
    for i, (s, a, bb) in enumerate(comps):
        R[i, names.index(f"{s}_g{bb}")] = 1
        R[i, names.index(f"{s}_g{a}")] = -1
        est = float(R[i] @ b)
        se = float(np.sqrt(R[i] @ V @ R[i]))
        diffs.append(dict(segment=SEGMENT_LABELS[SEGMENTS.index(s)], comparison=f"{bb} - {a}",
                          diff=est, se=se, p=float(2 * stats.t.sf(abs(est / se), df=d.GID_0.nunique() - 1))))
    Rb = R @ b
    W = float(Rb @ np.linalg.solve(R @ V @ R.T, Rb))
    q, G = len(comps), d.GID_0.nunique()
    F = W / q
    return dict(model=m, slopes=slopes, diffs=pd.DataFrame(diffs),
                wald=dict(F=F, q=q, df2=G - 1, p=float(stats.f.sf(F, q, G - 1))), n=int(m._N))


def fit_pooled(df: pd.DataFrame, y: str = "log_pm25", vcov=CL_CTY, controls=("P_a", "T_a")) -> dict:
    """The pooled path: one spline, region + year FE. Returns the model and
    its slope table."""
    import pyfixest as pf

    m = pf.feols(f"{y} ~ {' + '.join([*SEGMENTS, *controls])} | GID_1 + year", data=df, vcov=vcov)
    t = m.tidy().loc[list(SEGMENTS), ["Estimate", "Std. Error", "Pr(>|t|)"]]
    t.index = SEGMENT_LABELS
    return dict(model=m, slopes=t)



# -- variant 6: countries that change tercile (change log, 2026-10-29) ---------

MOVE_WINDOW = (2001, 2016)        # a move leaves >= 3 years on each side


def yearly_terciles(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Country x year ``v2x_polyarchy`` and its tercile at the baseline
    boundaries, for the countries of *df*; and the two boundaries."""
    vdem = pd.read_parquet(VDEM)
    cty = df.GID_0.unique()
    base = (vdem[vdem.year.between(*BASELINE) & vdem.country_text_id.isin(cty)]
            .groupby("country_text_id").v2x_polyarchy.mean())
    cuts = pd.qcut(base, 3, labels=False, retbins=True)[1][1:3]
    v = (vdem.loc[vdem.country_text_id.isin(cty) & vdem.year.between(*YEARS),
                  ["country_text_id", "year", "v2x_polyarchy"]]
         .rename(columns={"country_text_id": "GID_0"}).sort_values(["GID_0", "year"]))
    v["t_year"] = np.searchsorted(cuts, v.v2x_polyarchy, side="right")
    v["t_base"] = v.GID_0.map(baseline_terciles(vdem, cty, "v2x_polyarchy"))
    return v, cuts


def movers(v: pd.DataFrame) -> pd.DataFrame:
    """Countries outside their baseline tercile in every year from the move
    year through the panel's end, the move year within MOVE_WINDOW."""
    rows = []
    for c, g in v.groupby("GID_0"):
        off = (g.t_year != g.t_base).to_numpy()[::-1]
        k = len(off) if off.all() else int(np.argmin(off))   # length of the final off-baseline run
        if k and MOVE_WINDOW[0] <= g.year.iloc[-k] <= MOVE_WINDOW[1]:
            rows.append(dict(GID_0=c, move_year=int(g.year.iloc[-k]), base=int(g.t_base.iloc[0]),
                             to=int(g.t_year.iloc[-1])))
    m = pd.DataFrame(rows)
    m["direction"] = np.where(m.to > m.base, "up", "down")
    return m


def fit_movers(df: pd.DataFrame, mv: pd.DataFrame, vcov=CL_CTY) -> dict:
    """Movers' regions only: linear slopes in ln output before and after the
    move, by direction, with a post-move dummy, region FE and direction x
    year FE. Returns the slope table and after - before per direction."""
    import pyfixest as pf
    from scipy import stats

    d = df.merge(mv[["GID_0", "move_year", "direction"]], on="GID_0")
    post = (d.year >= d.move_year).astype(int)
    terms = []
    for k in ("up", "down"):
        on = (d.direction == k).astype(int)
        d[f"before_{k}"], d[f"after_{k}"], d[f"post_{k}"] = d.log_y * on * (1 - post), d.log_y * on * post, on * post
        terms += [f"before_{k}", f"after_{k}", f"post_{k}"]
    m = pf.feols(f"log_pm25 ~ {' + '.join(terms)} + P_a + T_a | GID_1 + direction^year", data=d, vcov=vcov)
    tidy, names, V = m.tidy(), list(m._coefnames), np.asarray(m._vcov)
    G = d.GID_0.nunique()
    rows = []
    for k in ("up", "down"):
        sub = d[d.direction == k]
        for t in ("before", "after"):
            r = tidy.loc[f"{t}_{k}"]
            rows.append(dict(direction=k, period=t, est=r["Estimate"], se=r["Std. Error"], p=r["Pr(>|t|)"],
                             countries=sub.GID_0.nunique(), regions=sub.GID_1.nunique()))
        a, b = names.index(f"after_{k}"), names.index(f"before_{k}")
        diff = m.coef()[f"after_{k}"] - m.coef()[f"before_{k}"]
        se = float(np.sqrt(V[a, a] + V[b, b] - 2 * V[a, b]))
        rows.append(dict(direction=k, period="after - before", est=diff, se=se,
                         p=float(2 * stats.t.sf(abs(diff / se), df=G - 1))))
    return dict(model=m, slopes=pd.DataFrame(rows), n=int(m._N), countries=G)

# -- figures --------------------------------------------------------------------

REF_USD = 4000                    # paths are drawn relative to this income
GROUP_LABELS = ("low democracy", "middle", "high democracy")
GROUP_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")   # categorical slots 1-3, validated all-pairs
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
FIGURES = "output/figures/pollution"


def path_curve(model, terms: list[str], log_y: np.ndarray, ref_usd: float = REF_USD):
    """ln PM2.5 along the spline relative to *ref_usd*, and its standard error:
    a linear combination of the segment slopes *terms* (in SEGMENTS order).
    Terms dropped as collinear with the FE are skipped; their basis is constant
    over a group's supported range, so they do not move the path there."""
    basis = spline_terms(pd.Series(log_y)) - spline_terms(pd.Series([np.log(ref_usd)])).iloc[0]
    names = list(model._coefnames)
    keep = [k for k, t in enumerate(terms) if t in names]
    idx = [names.index(terms[k]) for k in keep]
    A = basis.to_numpy()[:, keep]
    b = model.coef().reindex(names).to_numpy()[idx]
    V = np.asarray(model._vcov)[np.ix_(idx, idx)]
    return A @ b, np.sqrt(np.einsum("ij,jk,ik->i", A, V, A))


def supported_range(df: pd.DataFrame, g: int, n: pd.DataFrame, group: str = "tercile",
                    q=(0.01, 0.99)) -> tuple[float, float]:
    """ln output range drawn for group *g*: its supported segments, trimmed to
    the group's 1st-99th percentile of ln output."""
    cuts = [-np.inf, *np.log(KNOTS_USD), np.inf]
    ok = [k for k, s in enumerate(SEGMENTS) if n.loc[g, s] >= MIN_REGIONS]
    lo, hi = df.loc[df[group] == g, "log_y"].quantile(q)
    return max(lo, cuts[ok[0]]), min(hi, cuts[ok[-1] + 1])


def nature_style():
    """Matplotlib defaults close to Nature's figure guide: 7 pt sans serif,
    thin axes, outward ticks, editable text in vector output."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5,
        "xtick.direction": "out", "ytick.direction": "out", "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "lines.linewidth": 1.0, "legend.frameon": False,
        "mathtext.fontset": "custom", "mathtext.rm": "Nimbus Sans", "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 600,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def _usd_axis(ax, lo: float, hi: float):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    ax.set_xscale("log")
    ax.set_xlim(np.exp(lo), np.exp(hi))
    ticks = [t for t in (300, 1000, 4000, 12000, 50000) if np.exp(lo) <= t <= np.exp(hi)]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    for k in KNOTS_USD:
        ax.axvline(k, color=INK_3, lw=0.4, ls=(0, (1, 2)), zorder=0)


def _group_paths(ax, ax2, df, model, lo, hi, group="tercile", labels=GROUP_LABELS, pct=True):
    """Paths by group on their supported ranges, relative to REF_USD, with 95 %
    bands; below, each group's income distribution (5-95 %, IQR, median)."""
    n = supported_cells(df, group)
    for g in sorted(int(g) for g in df[group].unique()):
        rlo, rhi = supported_range(df, g, n, group)
        x = np.linspace(max(rlo, lo), min(rhi, hi), 200)
        f, se = path_curve(model, [f"{s}_g{g}" for s in SEGMENTS], x)
        y = (lambda v: 100 * (np.exp(v) - 1)) if pct else (lambda v: v)
        c = GROUP_COLORS[g]
        ax.fill_between(np.exp(x), y(f - 1.96 * se), y(f + 1.96 * se), color=c, alpha=0.15, lw=0)
        ax.plot(np.exp(x), y(f), color=c, lw=1.2, solid_capstyle="round")
        # low: below its left end (its right end is where the paths meet); middle: below
        # its right end; high: above its right end
        end, off, ha, va = [(0, (2, -8), "left", "top"), (-1, (0, -5), "right", "top"),
                            (-1, (0, 4), "right", "bottom")][g]
        ax.annotate(labels[g], (np.exp(x[end]), y(f[end])), xytext=off, textcoords="offset points",
                    ha=ha, va=va, fontsize=6, color=INK)
        q = np.exp(df.loc[df[group] == g, "log_y"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_numpy())
        yb = 2 - g
        ax2.plot([q[0], q[4]], [yb, yb], color=c, lw=0.8, solid_capstyle="round")
        ax2.plot([q[1], q[3]], [yb, yb], color=c, lw=2.6, solid_capstyle="butt")
        ax2.plot(q[2], yb, "o", ms=3.2, mfc="white", mec=c, mew=0.8)
    ax2.set_yticks([2, 1, 0], ["low", "middle", "high"])
    ax2.set_ylim(-0.7, 2.7)
    ax2.tick_params(axis="y", length=0)
    ax2.spines["left"].set_visible(False)


def _path_axes(ax, ax2, lo, hi):
    ax.axhline(0, color=INK_3, lw=0.5, zorder=0)
    _usd_axis(ax, lo, hi), _usd_axis(ax2, lo, hi)
    ax.tick_params(labelbottom=False)
    ax2.set_xlabel("Output per person (2015 USD, log scale)")


def _slope_panel(ax, specs: dict[str, tuple], groups):
    """Segment slopes with 95 % intervals: *specs* maps a legend label to
    (result of fit_paths, marker, filled). Unsupported cells are left out."""
    from matplotlib.lines import Line2D

    width, ns = 0.8, len(specs)
    step = width / 3 / (ns + 1)
    for i, s_lab in enumerate(SEGMENT_LABELS):
        for j, g in enumerate(groups):
            shown = 0
            for k, (res, m, filled) in enumerate(specs.values()):
                r = res["slopes"].query("segment == @s_lab and group == @g").iloc[0]
                xpos = i + (j - 1) * width / 3 + (k - (ns - 1) / 2) * step
                if not r.supported:
                    continue
                shown += 1
                c = GROUP_COLORS[g]
                ax.plot([xpos, xpos], [r.est - 1.96 * r.se, r.est + 1.96 * r.se], color=c, lw=0.8)
                ax.plot(xpos, r.est, m, ms=3.6, mec=c, mew=0.8, mfc=c if filled else "white")
            if not shown:
                ax.text(i + (j - 1) * width / 3, 0, "n < 50", rotation=90, ha="center", va="center",
                        fontsize=5, color=INK_3)
        if i:
            ax.axvline(i - 0.5, color=INK_3, lw=0.4, zorder=0)
    ax.axhline(0, color=INK_3, lw=0.5, zorder=0)
    ax.set_xticks(range(len(SEGMENTS)), [f"{s} USD" for s in SEGMENT_LABELS])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(SEGMENTS) - 0.5)
    ax.set_ylabel("Elasticity of PM$_{2.5}$\nwith respect to output")
    h = [Line2D([], [], ls="", marker=m, ms=3.6, mec=INK, mew=0.8, mfc=INK if filled else "white", label=lab)
         for lab, (_, m, filled) in specs.items()]
    h += [Line2D([], [], color=GROUP_COLORS[g], lw=1.2, label=["low", "middle", "high"][g]) for g in groups]
    ax.legend(handles=h, ncol=len(h), loc="lower left", bbox_to_anchor=(0, 1.02), handletextpad=0.4,
              columnspacing=1.2, borderaxespad=0)


def _path_figure(height_mm: float = 128):
    """Two path panels with income strips on top, one wide panel below."""
    import matplotlib.pyplot as plt

    nature_style()
    fig = plt.figure(figsize=(183 / 25.4, height_mm / 25.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.42, wspace=0.3,
                          left=0.07, right=0.985, top=0.965, bottom=0.08)
    top = [gs[0, k].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.08) for k in range(2)]
    ax_a, ax_a2 = fig.add_subplot(top[0][0]), fig.add_subplot(top[0][1])
    ax_b, ax_b2 = fig.add_subplot(top[1][0]), fig.add_subplot(top[1][1])
    return fig, ax_a, ax_a2, ax_b, ax_b2, fig.add_subplot(gs[1, :])


def _letters(axes, xs):
    """Bold lowercase panel letters; *xs* are offsets in axes fractions."""
    for ax, x, letter in zip(axes, xs, "abcdef"):
        ax.text(x, 1.04, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def _save(fig, path):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(f"{path}.{ext}")
    return fig


def plot_paths(df: pd.DataFrame, pooled: dict, main: dict, variants: dict[str, dict], path: str | None = None):
    """Figure: (a) pooled path, (b) paths by democracy tercile on their
    supported ranges, each with a strip of where the region-years lie, and
    (c) segment elasticities by tercile in the main specification and
    variants 1-2. Paths are % PM2.5 relative to REF_USD with 95 % bands
    (country clusters)."""
    fig, ax_a, ax_a2, ax_b, ax_b2, ax_c = _path_figure()
    lo, hi = df.log_y.quantile([0.005, 0.995])

    x = np.linspace(*df.log_y.quantile([0.01, 0.99]), 200)
    f, se = path_curve(pooled["model"], list(SEGMENTS), x)
    to_pct = lambda v: 100 * (np.exp(v) - 1)
    ax_a.fill_between(np.exp(x), to_pct(f - 1.96 * se), to_pct(f + 1.96 * se), color=INK_2, alpha=0.18, lw=0)
    ax_a.plot(np.exp(x), to_pct(f), color=INK, lw=1.2)
    ax_a2.hist(np.exp(df.log_y), bins=np.exp(np.linspace(lo, hi, 60)), color="#c3c2b7", lw=0)
    ax_a2.set_ylabel("Region-\nyears", labelpad=2)
    ax_a2.set_yticks([])
    ax_a.set_title("All regions", loc="left", pad=3)

    _group_paths(ax_b, ax_b2, df, main["model"], lo, hi)
    ax_b.set_title("By democracy tercile (supported income ranges)", loc="left", pad=3)
    for ax, ax2 in ((ax_a, ax_a2), (ax_b, ax_b2)):
        _path_axes(ax, ax2, lo, hi)
    ax_a2.spines["left"].set_visible(True)
    ylim = [f(ax.get_ylim()[k] for ax in (ax_a, ax_b)) for k, f in enumerate((min, max))]
    ax_a.set_ylim(ylim), ax_b.set_ylim(ylim)
    ax_a.set_ylabel(f"PM$_{{2.5}}$ relative to {REF_USD:,} USD (%)")
    ax_b.tick_params(labelleft=False)

    names = ["Main: region + tercile × year FE", "+ country linear trends", "Region + country × year FE"]
    _slope_panel(ax_c, dict(zip(names, [(main, "o", True), (variants["+ country trends"], "^", False),
                                        (variants["country × year FE"], "s", False)])), [0, 1, 2])
    _letters((ax_a, ax_b, ax_c), (-0.1, -0.1, -0.045))
    return _save(fig, path)


def plot_robustness(df: pd.DataFrame, main: dict, rule: dict, nodust: dict, popw: dict, levels: dict,
                    path: str | None = None):
    """Figure for variants 3-5: (a) paths by rule-of-law tercile, (b) paths by
    democracy tercile with PM2.5 in levels (ug/m3), and (c) segment
    elasticities in the main specification, by rule-of-law tercile, without
    dusty regions and with population weights."""
    fig, ax_a, ax_a2, ax_b, ax_b2, ax_c = _path_figure()
    lo, hi = df.log_y.quantile([0.005, 0.995])

    _group_paths(ax_a, ax_a2, df, rule["model"], lo, hi, group="tercile_rule",
                 labels=("low rule of law", "middle", "high rule of law"))
    ax_a.set_title("By rule-of-law tercile (variant 3)", loc="left", pad=3)
    ax_a.set_ylabel(f"PM$_{{2.5}}$ relative to {REF_USD:,} USD (%)")
    _group_paths(ax_b, ax_b2, df, levels["model"], lo, hi, pct=False)
    ax_b.set_title("By democracy tercile, PM$_{2.5}$ in levels (variant 5)", loc="left", pad=3)
    ax_b.set_ylabel(f"PM$_{{2.5}}$ relative to {REF_USD:,} USD (µg m$^{{-3}}$)")
    for ax, ax2 in ((ax_a, ax_a2), (ax_b, ax_b2)):
        _path_axes(ax, ax2, lo, hi)

    _slope_panel(ax_c, {"Main (democracy terciles)": (main, "o", True),
                        "Rule-of-law terciles": (rule, "D", False),
                        "Desert share ≤ 0.5": (nodust, "^", False),
                        "Population weights": (popw, "s", False)}, [0, 1, 2])
    _letters((ax_a, ax_b, ax_c), (-0.12, -0.12, -0.045))
    return _save(fig, path)


def plot_movers(v: pd.DataFrame, cuts, mv: pd.DataFrame, res: dict, path: str | None = None):
    """Figure for variant 6: (a) yearly v2x_polyarchy with the baseline tercile
    boundaries, movers highlighted and their move year marked; (b) movers'
    slopes before and after the move, by direction (country clusters)."""
    import matplotlib.pyplot as plt

    nature_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(183 / 25.4, 70 / 25.4), width_ratios=[2.2, 1],
                                     gridspec_kw=dict(wspace=0.35, left=0.07, right=0.97, top=0.86, bottom=0.15))
    colors = dict(up=GROUP_COLORS[0], down=GROUP_COLORS[1])
    moving = mv.set_index("GID_0")
    ax_a.axvspan(BASELINE[0], BASELINE[1], color="#ecebe8", lw=0, zorder=0)
    for c, g in v.groupby("GID_0"):
        if c not in moving.index:
            ax_a.plot(g.year, g.v2x_polyarchy, color="#d6d5d0", lw=0.5, zorder=1)
    ends = []
    for c, g in v[v.GID_0.isin(moving.index)].groupby("GID_0"):
        r = moving.loc[c]
        col = colors[r.direction]
        ax_a.plot(g.year, g.v2x_polyarchy, color=col, lw=1.0, zorder=3)
        y0 = g.loc[g.year == r.move_year, "v2x_polyarchy"].iloc[0]
        ax_a.plot(r.move_year, y0, "o", ms=3.2, mfc="white", mec=col, mew=0.8, zorder=4)
        ends.append((g.v2x_polyarchy.iloc[-1], c))
    # spread the end labels so they do not overlap
    ends.sort()
    placed = []
    for yv, c in ends:
        yl = max(yv, placed[-1] + 0.035) if placed else yv
        placed.append(yl)
        ax_a.annotate(c, (YEARS[1], yv), xytext=(YEARS[1] + 0.6, yl), textcoords="data", va="center",
                      fontsize=5.5, color=INK, arrowprops=dict(arrowstyle="-", lw=0.3, color=INK_3))
    for cut in cuts:
        ax_a.axhline(cut, color=INK_2, lw=0.6, ls=(0, (3, 2)), zorder=2)
    ax_a.text((BASELINE[0] + BASELINE[1]) / 2, 0.02, "baseline", fontsize=5.5, color=INK_2, ha="center")
    ax_a.set_xlim(YEARS[0], YEARS[1] + 2.4)
    ax_a.set_ylim(0, 1)
    ax_a.set_xticks(range(YEARS[0], YEARS[1] + 1, 4))
    ax_a.set_xlabel("Year")
    ax_a.set_ylabel("Electoral democracy (V-Dem v2x_polyarchy)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=colors[k], lw=1.0, label=f"moved {k}") for k in ("up", "down")]
    h += [Line2D([], [], color="#d6d5d0", lw=0.8, label="other countries"),
          Line2D([], [], color=INK_2, lw=0.6, ls=(0, (3, 2)), label="baseline tercile boundaries"),
          Line2D([], [], ls="", marker="o", ms=3.2, mfc="white", mec=INK, mew=0.8, label="move year")]
    ax_a.legend(handles=h, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0, columnspacing=1.2)

    t = res["slopes"]
    for i, k in enumerate(("up", "down")):
        b = t.query("direction == @k and period == 'before'").iloc[0]
        a = t.query("direction == @k and period == 'after'").iloc[0]
        d = t.query("direction == @k and period == 'after - before'").iloc[0]
        col = colors[k]
        for dx, r, filled in ((-0.15, b, False), (0.15, a, True)):
            ax_b.plot([i + dx] * 2, [r.est - 1.96 * r.se, r.est + 1.96 * r.se], color=col, lw=0.8)
            ax_b.plot(i + dx, r.est, "o", ms=3.6, mec=col, mew=0.8, mfc=col if filled else "white")
        ax_b.plot([i - 0.15, i + 0.15], [b.est, a.est], color=col, lw=0.5, ls=":")
        ax_b.text(i, 0.99, f"after − before {d.est:+.2f}\np = {d.p:.3f}" if d.p < 0.01 else
                  f"after − before {d.est:+.2f}\np = {d.p:.2f}", transform=ax_b.get_xaxis_transform(),
                  ha="center", va="top", fontsize=5.5)
    top = (t.est + 1.96 * t.se).max()
    ax_b.set_ylim(top=top + 0.35 * (top - (t.est - 1.96 * t.se).min()))
    ax_b.axhline(0, color=INK_3, lw=0.5, zorder=0)
    n_up, n_down = (int(t.query("direction == @k").countries.dropna().iloc[0]) for k in ("up", "down"))
    ax_b.set_xticks([0, 1], [f"Moved up\n({n_up} countries)", f"Moved down\n({n_down} countries)"])
    ax_b.tick_params(axis="x", length=0)
    ax_b.set_xlim(-0.6, 1.6)
    ax_b.set_ylabel("Elasticity of PM$_{2.5}$ with respect to output")
    h = [Line2D([], [], ls="", marker="o", ms=3.6, mec=INK, mew=0.8, mfc=f, label=l)
         for f, l in (("white", "before move"), (INK, "after move"))]
    ax_b.legend(handles=h, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0)
    _letters((ax_a, ax_b), (-0.08, -0.3))
    return _save(fig, path)

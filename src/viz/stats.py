"""DuckDB statistic builders for regression-analysis plots.

Each function pushes the aggregation into SQL and returns a small tidy
dataclass that carries everything a renderer in ``src.viz.regplot`` needs, so a
figure can be redrawn without re-touching the data. No matplotlib import here --
this is the query half, mirroring ``src.viz.grid``.

Column names and transforms reach SQL, so callers only get to name a column
(validated as a bare identifier) and pick a transform from a fixed table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.viz.source import columns, quote_ident, resolve_source

# name -> (SQL expr template using {c} for the quoted column, {o} for the offset)
_TRANSFORMS: dict[str, str] = {
    "identity": "{c}",
    "log": "ln({c})",
    "log1p": "ln({c} + {o})",
    "asinh": "asinh({c})",
}


def _expr(col: str, transform: str, offset: float) -> str:
    try:
        template = _TRANSFORMS[transform]
    except KeyError:
        raise ValueError(
            f"transform must be one of {sorted(_TRANSFORMS)}, got {transform!r}"
        ) from None
    return template.format(c=quote_ident(col), o=float(offset))


def _require(cols: list[str], *names: str | None) -> None:
    missing = [n for n in names if n is not None and n not in cols]
    if missing:
        raise ValueError(f"column(s) {missing} not found; available: {sorted(cols)}")


# ---------------------------------------------------------------------------
# histogram
# ---------------------------------------------------------------------------


@dataclass
class Histogram:
    """A binned univariate distribution, optionally split by ``group``."""

    edges: np.ndarray  # (nbins + 1,)
    counts: np.ndarray  # (nbins,) or (n_groups, nbins); float (weighted-capable)
    col: str
    transform: str
    n: int  # rows scanned
    n_missing: int  # rows with a NULL transformed value
    group: str | None = None
    groups: list | None = None  # group labels, parallel to axis 0 of a 2-D ``counts``

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    def to_frame(self) -> pd.DataFrame:
        counts = self.counts if self.counts.ndim == 2 else self.counts[None, :]
        labels = self.groups if self.groups is not None else [None]
        rows = []
        for label, row in zip(labels, counts):
            for lo, hi, c in zip(self.edges[:-1], self.edges[1:], row):
                rows.append({"group": label, "bin_low": lo, "bin_high": hi, "count": c})
        return pd.DataFrame(rows)


def histogram(
    source,
    col: str,
    *,
    bins: int = 50,
    value_range: tuple[float, float] | None = None,
    transform: str = "identity",
    offset: float = 0.01,
    groupby: str | None = None,
    weights: str | None = None,
    con=None,
    table: str | None = None,
) -> Histogram:
    """Uniform-width histogram of ``transform(col)``.

    ``weights`` (a column name) makes each bin the summed weight rather than the
    row count. With no ``value_range`` the bin span is the data min/max and the
    bin counts reproduce ``numpy.histogram``; an explicit range drops
    out-of-range rows (numpy semantics).
    """
    if bins < 1:
        raise ValueError("bins must be >= 1")

    with resolve_source(source, con=con, table=table) as rs:
        cols = columns(rs)
        _require(cols, col, groupby, weights)
        vexpr = _expr(col, transform, offset)
        gsel = f", {quote_ident(groupby)} AS grp" if groupby else ""
        wsel = f", {quote_ident(weights)} AS w" if weights else ""
        base = f"SELECT {vexpr} AS v{gsel}{wsel} FROM {rs.relation}"

        summary = rs.con.execute(
            f"SELECT count(*), count(v), min(v), max(v) FROM ({base})"
        ).fetchone()
        n_total, n_nonnull, lo, hi = summary
        n_total = int(n_total or 0)
        n_nonnull = int(n_nonnull or 0)

        if value_range is not None:
            lo, hi = float(value_range[0]), float(value_range[1])
        elif lo is None or hi is None:
            lo, hi = 0.0, 1.0
        lo, hi = float(lo), float(hi)
        if not hi > lo:
            hi = lo + 1.0

        edges = np.linspace(lo, hi, bins + 1)
        binwidth = (hi - lo) / bins

        where = "v IS NOT NULL"
        if weights:
            # A bin whose rows all carry a NULL weight would otherwise yield
            # sum(w) = NULL -> NaN, poisoning counts.sum() and the plot.
            where += " AND w IS NOT NULL"
        if value_range is not None:
            where += f" AND v BETWEEN {lo} AND {hi}"
        agg = "sum(w)" if weights else "count(*)"
        grp_cols = "grp, " if groupby else ""
        query = f"""
            SELECT {grp_cols}
                   LEAST({bins - 1}, GREATEST(0, floor((v - {lo}) / {binwidth})))::BIGINT AS bin,
                   {agg} AS c
            FROM ({base})
            WHERE {where}
            GROUP BY {grp_cols}bin
        """
        df = rs.con.execute(query).fetchdf()

    n_missing = n_total - n_nonnull
    if groupby:
        labels = sorted(df["grp"].dropna().unique().tolist())
        counts = np.zeros((len(labels), bins), dtype=float)
        index = {label: i for i, label in enumerate(labels)}
        for _, r in df.iterrows():
            if pd.isna(r["grp"]):
                continue
            counts[index[r["grp"]], int(r["bin"])] = float(r["c"]) if pd.notna(r["c"]) else 0.0
        return Histogram(edges, counts, col, transform, n_total, n_missing, groupby, labels)

    counts = np.zeros(bins, dtype=float)
    for _, r in df.iterrows():
        counts[int(r["bin"])] = float(r["c"]) if pd.notna(r["c"]) else 0.0
    return Histogram(edges, counts, col, transform, n_total, n_missing)


# ---------------------------------------------------------------------------
# binscatter
# ---------------------------------------------------------------------------


@dataclass
class BinScatter:
    """Quantile-binned means of ``y`` against ``x`` (optionally per ``group``)."""

    x_mean: np.ndarray
    y_mean: np.ndarray
    n: np.ndarray
    y_se: np.ndarray | None
    x: str
    y: str
    slope: float | None = None
    intercept: float | None = None
    group: str | None = None
    groups: list | None = None  # parallel to axis 0 when arrays are 2-D
    slopes: dict | None = None  # per-group (slope, intercept)

    def to_frame(self) -> pd.DataFrame:
        x = self.x_mean if self.x_mean.ndim == 2 else self.x_mean[None, :]
        y = self.y_mean if self.y_mean.ndim == 2 else self.y_mean[None, :]
        n = self.n if self.n.ndim == 2 else self.n[None, :]
        se = self.y_se
        se = (se if (se is not None and se.ndim == 2) else (None if se is None else se[None, :]))
        labels = self.groups if self.groups is not None else [None]
        rows = []
        for gi, label in enumerate(labels):
            for bi in range(x.shape[1]):
                rows.append(
                    {
                        "group": label,
                        "x_mean": x[gi, bi],
                        "y_mean": y[gi, bi],
                        "n": n[gi, bi],
                        "y_se": (se[gi, bi] if se is not None else np.nan),
                    }
                )
        return pd.DataFrame(rows)


def _ols_line(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return None, None
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    return float(slope), float(intercept)


def binscatter(
    source,
    y: str,
    x: str,
    *,
    bins: int = 20,
    x_transform: str = "identity",
    y_transform: str = "identity",
    offset: float = 0.01,
    groupby: str | None = None,
    weights: str | None = None,
    fit_line: bool = True,
    con=None,
    table: str | None = None,
) -> BinScatter:
    """Split ``x`` into ``bins`` equal-count bins and return the per-bin mean of
    ``x`` and ``y``. ``y_se`` is the per-bin standard error of the mean, or
    ``None`` when ``weights`` is given (weighted SEs are out of scope for now).
    """
    if bins < 2:
        raise ValueError("bins must be >= 2")

    with resolve_source(source, con=con, table=table) as rs:
        cols = columns(rs)
        _require(cols, x, y, groupby, weights)
        xexpr = _expr(x, x_transform, offset)
        yexpr = _expr(y, y_transform, offset)
        gsel = f", {quote_ident(groupby)} AS grp" if groupby else ""
        wsel = f", {quote_ident(weights)} AS w" if weights else ""
        where = f"{xexpr} IS NOT NULL AND {yexpr} IS NOT NULL"
        if groupby:
            where += f" AND {quote_ident(groupby)} IS NOT NULL"
        if weights:
            where += f" AND {quote_ident(weights)} IS NOT NULL"

        base = (
            f"SELECT {xexpr} AS x, {yexpr} AS y{gsel}{wsel} "
            f"FROM {rs.relation} WHERE {where}"
        )
        partition = "PARTITION BY grp " if groupby else ""
        binned = f"SELECT *, ntile({bins}) OVER ({partition}ORDER BY x) AS b FROM ({base})"

        if weights:
            xm, ym = "sum(w * x) / sum(w)", "sum(w * y) / sum(w)"
        else:
            xm, ym = "avg(x)", "avg(y)"
        grp_cols = "grp, " if groupby else ""
        query = f"""
            SELECT {grp_cols}b,
                   {xm} AS x_mean,
                   {ym} AS y_mean,
                   count(*) AS n,
                   stddev_samp(y) AS y_sd
            FROM ({binned})
            GROUP BY {grp_cols}b
            ORDER BY {grp_cols}b
        """
        df = rs.con.execute(query).fetchdf()

    def _pack(sub: pd.DataFrame):
        xa = sub["x_mean"].to_numpy(dtype=float)
        ya = sub["y_mean"].to_numpy(dtype=float)
        na = sub["n"].to_numpy(dtype=float)
        se = None if weights else (sub["y_sd"].to_numpy(dtype=float) / np.sqrt(na))
        return xa, ya, na, se

    if groupby:
        labels = sorted(df["grp"].dropna().unique().tolist())
        packed = [_pack(df[df["grp"] == label]) for label in labels]
        width = max((p[0].size for p in packed), default=0)

        def _pad(a):
            out = np.full(width, np.nan)
            out[: a.size] = a
            return out

        x_mean = np.vstack([_pad(p[0]) for p in packed]) if packed else np.empty((0, 0))
        y_mean = np.vstack([_pad(p[1]) for p in packed]) if packed else np.empty((0, 0))
        n = np.vstack([_pad(p[2]) for p in packed]) if packed else np.empty((0, 0))
        y_se = (
            None if weights
            else (np.vstack([_pad(p[3]) for p in packed]) if packed else np.empty((0, 0)))
        )
        slopes = {}
        if fit_line:
            for label, p in zip(labels, packed):
                slopes[label] = _ols_line(p[0], p[1])
        return BinScatter(
            x_mean, y_mean, n, y_se, x, y,
            group=groupby, groups=labels, slopes=slopes or None,
        )

    xa, ya, na, se = _pack(df)
    slope, intercept = _ols_line(xa, ya) if fit_line else (None, None)
    return BinScatter(xa, ya, na, se, x, y, slope=slope, intercept=intercept)


# ---------------------------------------------------------------------------
# correlation matrix
# ---------------------------------------------------------------------------


@dataclass
class CorrMatrix:
    """Symmetric pairwise correlation matrix plus pairwise non-null counts."""

    labels: list[str]
    matrix: np.ndarray  # (k, k)
    n: np.ndarray  # (k, k) int
    method: str = "pearson"

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.matrix, index=self.labels, columns=self.labels)


def _weighted_corr_expr(qa: str, qb: str, qw: str) -> str:
    """Weighted Pearson r over rows where ``qa``/``qb``/``qw`` are all non-null,
    as one FILTER-guarded aggregate expression (so every pair shares a single
    table scan)."""
    both = f"({qa} IS NOT NULL AND {qb} IS NOT NULL AND {qw} IS NOT NULL)"
    f = f" FILTER (WHERE {both})"
    sw = f"sum({qw}){f}"
    swa, swb = f"sum({qw}*{qa}){f}", f"sum({qw}*{qb}){f}"
    swaa, swbb = f"sum({qw}*{qa}*{qa}){f}", f"sum({qw}*{qb}*{qb}){f}"
    swab = f"sum({qw}*{qa}*{qb}){f}"
    cov = f"({swab}/{sw} - ({swa}/{sw})*({swb}/{sw}))"
    va = f"({swaa}/{sw} - pow({swa}/{sw}, 2))"
    vb = f"({swbb}/{sw} - pow({swb}/{sw}, 2))"
    return f"({cov} / (sqrt({va}) * sqrt({vb})))"


def _pair_spearman(con, relation, qa: str, qb: str):
    """Spearman r + pairwise-complete count for one column pair. Rank is taken
    *within* the pair's complete-case rows (matches ``pandas.DataFrame.corr``),
    so this stays per-pair rather than folding into the single-pass SELECT."""
    inner = (
        f"SELECT rank() OVER (ORDER BY {qa}) AS ra, rank() OVER (ORDER BY {qb}) AS rb "
        f"FROM {relation} WHERE {qa} IS NOT NULL AND {qb} IS NOT NULL"
    )
    return con.execute(f"SELECT corr(ra, rb), count(*) FROM ({inner})").fetchone()


def corr_matrix(
    source,
    cols: list[str],
    *,
    method: str = "pearson",
    weights: str | None = None,
    con=None,
    table: str | None = None,
) -> CorrMatrix:
    """Pairwise correlation of ``cols``. ``method`` is ``"pearson"`` or
    ``"spearman"`` (rank-transform then Pearson); ``weights`` applies only to
    Pearson.

    Pearson (weighted or not) is computed in a single table scan -- one SELECT
    carrying every pair's ``corr()`` / weighted expression plus its
    pairwise-complete count. Spearman stays one query per pair (each pair
    re-ranks within its own complete-case rows).
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")
    if len(cols) < 2:
        raise ValueError("corr_matrix needs at least two columns")

    k = len(cols)
    matrix = np.eye(k)
    n = np.zeros((k, k), dtype=np.int64)

    with resolve_source(source, con=con, table=table) as rs:
        available = columns(rs)
        _require(available, *cols, weights)
        q = [quote_ident(c) for c in cols]

        if method == "spearman":
            for i in range(k):
                n[i, i] = int(
                    rs.con.execute(f"SELECT count({q[i]}) FROM {rs.relation}").fetchone()[0] or 0
                )
                for j in range(i + 1, k):
                    r, nij = _pair_spearman(rs.con, rs.relation, q[i], q[j])
                    matrix[i, j] = matrix[j, i] = float(r) if r is not None else np.nan
                    n[i, j] = n[j, i] = int(nij or 0)
            return CorrMatrix(list(cols), matrix, n, method)

        qw = quote_ident(weights) if weights else None
        select_parts: list[str] = []
        for i in range(k):
            select_parts.append(f"count({q[i]}) AS n_{i}_{i}")
            for j in range(i + 1, k):
                both = f"{q[i]} IS NOT NULL AND {q[j]} IS NOT NULL"
                if qw is not None:
                    select_parts.append(f"{_weighted_corr_expr(q[i], q[j], qw)} AS c_{i}_{j}")
                    select_parts.append(
                        f"count(*) FILTER (WHERE {both} AND {qw} IS NOT NULL) AS n_{i}_{j}"
                    )
                else:
                    select_parts.append(f"corr({q[i]}, {q[j]}) AS c_{i}_{j}")
                    select_parts.append(f"count(*) FILTER (WHERE {both}) AS n_{i}_{j}")

        row = rs.con.execute(
            f"SELECT {', '.join(select_parts)} FROM {rs.relation}"
        ).fetchdf().iloc[0]

    for i in range(k):
        n[i, i] = int(row[f"n_{i}_{i}"] or 0)
        for j in range(i + 1, k):
            v = row[f"c_{i}_{j}"]
            matrix[i, j] = matrix[j, i] = float(v) if pd.notna(v) else np.nan
            nij = row[f"n_{i}_{j}"]
            n[i, j] = n[j, i] = int(nij) if pd.notna(nij) else 0

    return CorrMatrix(list(cols), matrix, n, method)

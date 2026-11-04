"""
Step 2 of the pollution follow-up (docs/successors/pollution-no2-prespec.md):
surface NO2 (Anenberg et al. 2022) as a second outcome, on DOSE output (main
arm, 2005-2018) and on night lights per person (check arm, 2005-2012).

A region's NO2 is the mean over the land 1 km cells of its 10 km panel cells,
from the 1 km ``no2`` tiles keyed to the panel's ``pixel_id`` (as
``pollution_lights`` does for lights). Cells above MAX_PPB are set to missing:
no annual mean reaches that.

``build_no2_counts()`` counts valid cells only, for G3: it reads NO2 values
only to test them against NaN and MAX_PPB, and aggregates none.
``build_no2_sums()`` aggregates them (after G3), plain and weighted by
GHS-POP at 1 km (variant 4); the fits follow §3-§4 and the change log of
2026-11-04.
"""
from __future__ import annotations

import glob
import json
import os

import duckdb
import numpy as np
import pandas as pd

import src.analysis.pollution as P
import src.analysis.pollution_lights as pl
from src.analysis.spec import DATA, GRID, SRC, THREADS

NO2 = f"{DATA}/prepared/no2/anenberg/crs/ease6933/no2"
NO2_COUNTS = f"{DATA}/_pollution_no2_cell_counts_{GRID}.parquet"
NO2_SUMS = f"{DATA}/_pollution_no2_cell_sums_{GRID}.parquet"

NO2_YEARS = (2005, 2020)          # annual from 2005; earlier years use another scaling
DOSE_YEARS = (2005, 2018)         # main arm: DOSE P_a and T_a stop 2018
LIGHTS_YEARS = (2005, 2012)       # check arm: DMSP years
MAX_PPB = 100.0                   # 1 km cells above this are missing
MIN_NO2_CELLS = 500               # valid 1 km NO2 cells per region-year
STEP1_KNOTS = (-4.017, -3.020, -2.231)    # step 1 rerun, 2026-11-04
POP_EPOCHS = (2005, 2010, 2015, 2020)     # GHS-POP epochs spanning NO2_YEARS
WEATHER = ("P_a", "T_a")                  # DOSE arm controls


def _con():
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute("SET enable_progress_bar = false")
    return con


def build_no2_counts(path: str = NO2_COUNTS) -> pd.DataFrame:
    """Per 10 km ``pixel_id`` and year (NO2_YEARS): land 1 km cells with a
    valid NO2 value (``n_valid``, at most MAX_PPB) and above MAX_PPB
    (``n_high``). No NO2 value is aggregated. One tile at a time."""
    if os.path.exists(path):
        return pd.read_parquet(path)
    from src.data.assemble.sql_engine import GridFacts, _pixel_id_sql

    g = GridFacts.build(int(GRID.removesuffix("km")) * 1000, (0.0, 0.0))
    pid = _pixel_id_sql("n.cell_id::BIGINT", g, g.F, g.DR, g.DC)
    con = _con()
    parts = []
    for tile in sorted(glob.glob(f"{pl.LAND}/ix=*/iy=*")):
        rel = os.path.relpath(tile, pl.LAND)
        files = [f for f in glob.glob(f"{NO2}/{rel}/part-*.parquet")
                 if NO2_YEARS[0] <= int(f[-12:-8]) <= NO2_YEARS[1]]
        if not files:
            continue
        parts.append(con.execute(f"""
            SELECT {pid} AS pixel_id, n.year,
                   count(*) FILTER (WHERE n.no2 <= {MAX_PPB}) AS n_valid,
                   count(*) FILTER (WHERE n.no2 > {MAX_PPB}) AS n_high
            FROM read_parquet({files!r}) n
            WHERE n.cell_id IN (SELECT cell_id FROM read_parquet('{tile}/*.parquet') WHERE land_mask)
              AND n.no2 IS NOT NULL AND NOT isnan(n.no2)
            GROUP BY 1, 2
        """).df())
    con.close()
    out = pd.concat(parts, ignore_index=True)
    out["pixel_id"] = out.pixel_id.astype("uint64")
    out.to_parquet(path, index=False)
    return out


def _pixel_regions() -> pd.DataFrame:
    """Each panel pixel's GID_1 code (static)."""
    with open(P.GID1_MAP) as f:
        codes = pd.DataFrame([(v, k) for k, v in json.load(f).items()], columns=["gid1_id", "GID_1"])
    con = _con()
    con.register("codes", codes)
    out = con.execute(f"""
        SELECT p.pixel_id, any_value(c.GID_1) AS GID_1
        FROM read_parquet('{SRC}', union_by_name = true, hive_partitioning = true) p
        JOIN codes c ON c.gid1_id = p.GID_1
        GROUP BY 1
    """).df()
    con.close()
    out["pixel_id"] = out.pixel_id.astype("uint64")
    return out


def region_no2_counts() -> pd.DataFrame:
    """Region x year: valid 1 km NO2 cells and cells above MAX_PPB."""
    c = build_no2_counts().merge(_pixel_regions(), on="pixel_id")
    return c.groupby(["GID_1", "year"], as_index=False)[["n_valid", "n_high"]].sum()


# -- the two arms (§2) ---------------------------------------------------------

def dose_arm(counts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Idea B's sample in DOSE_YEARS with its terciles and segments, and the
    rows it keeps and drops by the NO2 cell rule. No NO2 values."""
    s = P.sample(P.build_region_panel())
    s = s[s.year.between(*DOSE_YEARS)].merge(counts, on=["GID_1", "year"], how="left")
    keep = s.n_valid.fillna(0).ge(MIN_NO2_CELLS)
    return s[keep].copy(), s[~keep].copy()


def lights_arm(counts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Step 1's sample in LIGHTS_YEARS with its terciles and its knots (from
    the step 1 calibration, reproduced), the dropped rows, and the knots."""
    s = pl.sample(pl.build_lights_panel())
    knots = pl.calibrate_knots(s)["knots"]
    assert np.allclose(knots, STEP1_KNOTS, atol=5e-4), knots
    s = s[s.year.between(*LIGHTS_YEARS)].merge(counts, on=["GID_1", "year"], how="left")
    s["segment"] = pl.segment(s.log_lpc, knots)
    keep = s.n_valid.fillna(0).ge(MIN_NO2_CELLS)
    return s[keep].copy(), s[~keep].copy(), knots


def g3(df: pd.DataFrame) -> dict:
    """◇ G3: coverage by tercile, supported cells and comparisons. Loads no NO2."""
    by = df.groupby("tercile").agg(countries=("GID_0", "nunique"), regions=("GID_1", "nunique"))
    by["passes"] = by.countries.ge(pl.MIN_COUNTRIES) & by.regions.ge(pl.MIN_TERCILE_REGIONS)
    n = P.supported_cells(df)
    return dict(by_tercile=by, regions=n, comparisons=P.comparisons(n), passes=bool(by.passes.all()))


# -- region NO2 (§2; change log 2026-11-04) -----------------------------------

def _pop_weight_sql(col: str = "n.year") -> str:
    """GHS-POP density at a 1 km cell in year *col*, linear between epochs."""
    w = []
    for y in range(NO2_YEARS[0], NO2_YEARS[1] + 1):
        lo, hi = max(e for e in POP_EPOCHS if e <= y), min(e for e in POP_EPOCHS if e >= y)
        a = 0.0 if hi == lo else (y - lo) / (hi - lo)
        w.append(f"WHEN {y} THEN {1 - a} * coalesce(p.p{lo}, 0) + {a} * coalesce(p.p{hi}, 0)")
    return f"CASE {col} {' '.join(w)} END"


def build_no2_sums(path: str = NO2_SUMS) -> pd.DataFrame:
    """Per 10 km ``pixel_id`` and year (NO2_YEARS), over valid land 1 km cells
    (at most MAX_PPB): their count, Σ NO2, Σ w NO2 and Σ w, where w is the
    cell's GHS-POP density (variant 4). One tile at a time. Cached."""
    if os.path.exists(path):
        return pd.read_parquet(path)
    from src.data.assemble.sql_engine import GridFacts, _pixel_id_sql

    g = GridFacts.build(int(GRID.removesuffix("km")) * 1000, (0.0, 0.0))
    pid = _pixel_id_sql("n.cell_id::BIGINT", g, g.F, g.DR, g.DC)
    w = _pop_weight_sql()
    epochs = ", ".join(f"max(CASE WHEN year = {e} THEN pop END) AS p{e}" for e in POP_EPOCHS)
    con = _con()
    parts = []
    for tile in sorted(glob.glob(f"{pl.LAND}/ix=*/iy=*")):
        rel = os.path.relpath(tile, pl.LAND)
        files = [f for f in glob.glob(f"{NO2}/{rel}/part-*.parquet")
                 if NO2_YEARS[0] <= int(f[-12:-8]) <= NO2_YEARS[1]]
        if not files:
            continue
        pops = [f for f in glob.glob(f"{pl.POP}/{rel}/part-*.parquet") if int(f[-12:-8]) in POP_EPOCHS]
        pop = (f"SELECT cell_id, {epochs} FROM read_parquet({pops!r}) WHERE NOT isnan(pop) GROUP BY 1"
               if pops else "SELECT NULL::UINTEGER AS cell_id, " + ", ".join(f"NULL::FLOAT AS p{e}" for e in POP_EPOCHS))
        parts.append(con.execute(f"""
            WITH p AS ({pop})
            SELECT {pid} AS pixel_id, n.year, count(*) AS n_valid,
                   sum(n.no2::DOUBLE) AS s_no2,
                   sum(n.no2::DOUBLE * {w}) AS s_wno2,
                   sum({w}) AS s_w
            FROM read_parquet({files!r}) n LEFT JOIN p ON p.cell_id = n.cell_id
            WHERE n.cell_id IN (SELECT cell_id FROM read_parquet('{tile}/*.parquet') WHERE land_mask)
              AND n.no2 IS NOT NULL AND NOT isnan(n.no2) AND n.no2 <= {MAX_PPB}
            GROUP BY 1, 2
        """).df())
    con.close()
    out = pd.concat(parts, ignore_index=True)
    out["pixel_id"] = out.pixel_id.astype("uint64")
    out.to_parquet(path, index=False)
    return out


def region_no2() -> pd.DataFrame:
    """Region x year: mean NO2 over valid 1 km cells (``no2``) and the
    GHS-POP-weighted mean (``no2_popw``), ppb."""
    c = build_no2_sums().merge(_pixel_regions(), on="pixel_id")
    r = c.groupby(["GID_1", "year"], as_index=False)[["n_valid", "s_no2", "s_wno2", "s_w"]].sum()
    r["no2"] = r.s_no2 / r.n_valid
    r["no2_popw"] = r.s_wno2 / r.s_w.where(r.s_w > 0)
    return r[["GID_1", "year", "no2", "no2_popw"]]


def _add_no2(df: pd.DataFrame, no2: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(no2, on=["GID_1", "year"], how="inner")
    out = out[out.no2.gt(0)].copy()
    out["log_no2"] = np.log(out.no2)
    out["log_no2_popw"] = np.log(out.no2_popw.where(out.no2_popw > 0))
    return out


def dose_sample(counts: pd.DataFrame, no2: pd.DataFrame) -> pd.DataFrame:
    """DOSE arm rows (G3's) with ln NO2 and ln PM2.5."""
    return _add_no2(dose_arm(counts)[0], no2)


def lights_sample(counts: pd.DataFrame, no2: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Lights arm rows (G3's) with the spline at step 1's knots, ln PM2.5
    (step 1's region PM2.5), GLASS temperature and ln NO2; and the knots."""
    s, _, knots = lights_arm(counts)
    return _add_no2(pl.add_pm25(s, knots), no2), knots


def viirs_sample(counts: pd.DataFrame, no2: pd.DataFrame) -> pd.DataFrame:
    """Variant 6: step 1's variant-6 rows, knots and terciles (set on
    2014-2022), in 2014-2020 and with the NO2 cell rule."""
    d = pl.build_lights_panel()
    s = pl.sample(d, years=pl.VIIRS_YEARS)
    k = pl.calibrate_knots(s)["knots"]
    s = s[s.year.le(NO2_YEARS[1])].merge(counts, on=["GID_1", "year"], how="left")
    s = s[s.n_valid.fillna(0).ge(MIN_NO2_CELLS)]
    return _add_no2(pl.add_pm25(s, k), no2)


def low_haze_regions() -> pd.Index:
    """Variant H: step 1's low-haze regions (baseline PM2.5 cutoffs across
    step 1's main sample)."""
    s = pl.sample(pl.build_lights_panel())
    h, _ = pl.low_haze(pl.add_pm25(s, pl.calibrate_knots(s)["knots"]))
    return pd.Index(h.GID_1.unique())


def respline(df: pd.DataFrame, knots, x: str) -> pd.DataFrame:
    """*df* with the spline basis and segments in *x* at *knots* (ln)."""
    out = df.drop(columns=[*P.SEGMENTS, "segment"]).join(pl.spline_terms(df[x], knots=np.exp(knots)))
    out["segment"] = pl.segment(out[x], knots)
    return out


# -- ◇ T1: stacked NO2 and PM2.5 pooled paths (§3, §4) -------------------------

def fit_stacked(df: pd.DataFrame, controls=(), fe: str = "country_year") -> dict:
    """ln NO2 and ln PM2.5 stacked, each with its own spline slopes, region x
    outcome FE and (country or tercile) x year x outcome FE, clusters by
    country. Per segment: both slopes and NO2 - PM2.5 with a one-sided p
    (> 0), Holm-adjusted across segments, and the two-sided p."""
    import pyfixest as pf
    from scipy import stats

    keep = ["GID_1", "GID_0", "year", "tercile", *P.SEGMENTS, *controls]
    parts = []
    for o, y in (("no2", "log_no2"), ("pm25", "log_pm25")):
        d = df[keep + [y]].rename(columns={y: "y"}).assign(o=o)
        for v in (*P.SEGMENTS, *controls):
            for oo in ("no2", "pm25"):
                d[f"{v}_{oo}"] = d[v] * (oo == o)
        parts.append(d)
    st = pd.concat(parts, ignore_index=True)
    st["reg_o"] = st.GID_1 + "|" + st.o
    by = st.GID_0 if fe == "country_year" else st.tercile.astype(int).astype(str)
    st["yr_o"] = by + "|" + st.year.astype(str) + "|" + st.o
    x = [f"{v}_{o}" for o in ("no2", "pm25") for v in (*P.SEGMENTS, *controls)]
    m = pf.feols(f"y ~ {' + '.join(x)} | reg_o + yr_o", data=st, vcov=P.CL_CTY)
    names = list(m._coefnames)
    b, V = np.asarray(m.coef().reindex(names)), np.asarray(m._vcov)
    G = df.GID_0.nunique()
    rows = []
    for s, lab in zip(P.SEGMENTS, P.SEGMENT_LABELS):
        r = np.zeros(len(names))
        r[names.index(f"{s}_no2")], r[names.index(f"{s}_pm25")] = 1, -1
        d_, se = float(r @ b), float(np.sqrt(r @ V @ r))
        t_ = d_ / se
        i = names.index(f"{s}_no2")
        bn, sn = b[i], np.sqrt(V[i, i])
        rows.append(dict(segment=lab, no2=bn, p_no2=float(2 * stats.t.sf(abs(bn / sn), G - 1)),
                         pm25=b[names.index(f"{s}_pm25")], diff=d_, se=se,
                         p_one=float(stats.t.sf(t_, G - 1)), p_two=float(2 * stats.t.sf(abs(t_), G - 1))))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p_one.to_numpy())
    return dict(model=m, table=out, n=int(m._N))


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-adjusted p values."""
    o = np.argsort(p)
    k = len(p)
    adj = np.maximum.accumulate([(k - i) * p[j] for i, j in enumerate(o)])
    out = np.empty(k)
    out[o] = np.minimum(adj, 1.0)
    return out


def t1(dose: dict, lights: dict) -> dict:
    """◇ T1 from the two arms' stacked fits with country x year FE."""
    d, l = dose["table"].set_index("segment"), lights["table"].set_index("segment")
    hits = [s for s in d.index
            if d.loc[s, "diff"] > 0 and d.loc[s, "p_holm"] < 0.05
            and d.loc[s, "no2"] > 0 and d.loc[s, "p_no2"] < 0.05
            and s in l.index and l.loc[s, "diff"] > 0]
    return dict(segments=hits, claim=bool(hits))


# -- ◇ moderation, ◇ R1, ◇ turning point (§4) ---------------------------------

def moderation(main: dict, checks: dict[str, dict]) -> dict:
    """The joint test rejects at 5 % and each comparison at 5 % keeps its
    sign in every check variant (1, 2 and, in the lights arm, H)."""
    key = ["segment", "comparison"]
    sig = main["diffs"][main["diffs"].p < 0.05]
    signs = sig[key + ["diff", "p"]]
    for name, r in checks.items():
        signs = signs.merge(r["diffs"][key + ["diff"]].rename(columns={"diff": name}), on=key, how="left")
    keeps = all(np.sign(row["diff"]) == np.sign(row[n]) for _, row in signs.iterrows() for n in checks)
    rejects = main["wald"]["p"] < 0.05
    return dict(rejects=rejects, significant=signs, keeps=keeps, claim=bool(rejects and len(sig) and keeps))


def r1(main: dict) -> dict:
    """◇ R1: lights arm, main, above 12,000 USD, high - middle."""
    d = main["diffs"].set_index(["segment", "comparison"])
    k = ("> 12,000", "2 - 1")
    if k not in d.index:
        return dict(diff=np.nan, p=np.nan, claim=False)
    r = d.loc[k]
    return dict(diff=float(r["diff"]), p=float(r.p), claim=bool(r["diff"] > 0 and r.p < 0.05))


def downturns(main: dict, h: dict | None = None) -> list[tuple[str, int]]:
    """◇ Turning point: supported slopes negative at 5 % in *main* (and in
    *h*, the lights arm's variant H)."""
    def neg(r):
        s = r["slopes"]
        s = s[s.supported & s.est.lt(0) & s.p.lt(0.05)]
        return set(zip(s.segment, s.group))
    out = neg(main) if h is None else neg(main) & neg(h)
    return sorted(out)


# -- figures -------------------------------------------------------------------

def plot_transport(dose: dict, lights: dict, path: str | None = None):
    """Figure for ◇ T1: within-country (region + country x year FE) segment
    elasticities of NO2 and PM2.5 from the stacked fits, (a) DOSE arm and
    (b) lights arm, with 95 % intervals (country clusters) and NO2 - PM2.5
    (Holm-adjusted one-sided p) above each segment."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    P.nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 62 / 25.4), sharey=True,
                             gridspec_kw=dict(wspace=0.08, left=0.075, right=0.99, top=0.86, bottom=0.1))
    marks = {"no2": ("o", True, P.INK), "pm25": ("s", False, P.INK_2)}
    for ax, r, title in ((axes[0], dose, "DOSE output, 2005–2018"),
                         (axes[1], lights, "Night lights per person, 2005–2012")):
        m, t = r["model"], r["table"]
        names = list(m._coefnames)
        b, V = m.coef().reindex(names).to_numpy(), np.asarray(m._vcov)
        for i, s in enumerate(P.SEGMENTS):
            for k, (o, (mk, filled, c)) in enumerate(marks.items()):
                j = names.index(f"{s}_{o}")
                x, est, se = i + (k - 0.5) * 0.22, b[j], np.sqrt(V[j, j])
                ax.plot([x, x], [est - 1.96 * se, est + 1.96 * se], color=c, lw=0.8)
                ax.plot(x, est, mk, ms=3.6, mec=c, mew=0.8, mfc=c if filled else "white")
            row = t.iloc[i]
            p = row.p_holm
            ax.text(i, 1.0, f"Δ {row['diff']:+.3f}\n" + ("p < 0.001" if p < 0.001 else f"p = {p:.3f}"),
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.5,
                    color=P.INK if p < 0.05 else P.INK_3)
            if i:
                ax.axvline(i - 0.5, color=P.INK_3, lw=0.4, zorder=0)
        ax.axhline(0, color=P.INK_3, lw=0.5, zorder=0)
        ax.set_xticks(range(len(P.SEGMENTS)), [f"{s} USD" for s in P.SEGMENT_LABELS])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.5, len(P.SEGMENTS) - 0.5)
        ax.set_title(title, loc="left", pad=3)
    axes[0].set_ylabel("Within-country elasticity\nwith respect to growth")
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo, hi + 0.35 * (hi - lo))
    h = [Line2D([], [], ls="", marker=mk, ms=3.6, mec=c, mew=0.8, mfc=c if f else "white", label=lab)
         for lab, (mk, f, c) in zip(("NO$_2$", "PM$_{2.5}$"), marks.values())]
    fig.legend(handles=h, ncol=2, loc="lower right", bbox_to_anchor=(0.99, 0.86), handletextpad=0.3,
               columnspacing=1.2, borderaxespad=0)
    P._letters(axes, (-0.13, -0.04))
    return P._save(fig, path)

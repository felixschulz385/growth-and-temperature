"""
The fixed specification and shared helpers for the analysis notebooks
(``output/notebooks/01_pre_analysis.ipynb``, ``02_baseline.ipynb``, ``03_iv.ipynb``,
``03b_iv_robustness.ipynb``, ``04_channels.ipynb``; ``docs/analysis/final-analysis-plan.md`` §2).

Keeping the constants here means the notebooks cannot drift apart on the
outcome, fixed effects, clusters, instrument or sample.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT = Path(__file__).resolve().parents[2]

# duckreg is a source checkout, not pip-installed: a sibling of this repo unless
# DUCKREG_PATH says otherwise.
DUCKREG_PATH = os.environ.get("DUCKREG_PATH", str(PROJECT.parent / "duckreg"))
if DUCKREG_PATH not in sys.path:
    sys.path.insert(0, DUCKREG_PATH)
from duckreg import duckreg  # noqa: E402
import duckreg.estimators.base as _dre_base  # noqa: E402

# Silence DuckDB's progress-bar widget on the default connection (the notebooks'
# duckdb.sql queries) and, through the connection-init hook, on duckreg's own.
duckdb.default_connection().execute("SET enable_progress_bar = false")
_dre_orig_init = _dre_base.DuckEstimator._init_connection


def _dre_quiet_init(self):
    _dre_orig_init(self)
    try:
        self.conn.execute("SET enable_progress_bar = false")
    except Exception:
        pass


_dre_base.DuckEstimator._init_connection = _dre_quiet_init

# Data root: a local mirror at <repo>/data, else the HPC's data_nobackup/
# (orchestration/configs/data.local.yaml).
DATA = next(d for d in (PROJECT / "data", PROJECT / "data_nobackup") if (d / "assembled").exists())
GRID = "10km"
SRC = f"{DATA}/assembled/grid={GRID}/shake=base/ix=*/iy=*/*.parquet"
# derived cache, built by build_panel(); v2 adds the Story 1 columns (same rows as v1)
PANEL = f"{DATA}/_analysis_panel_{GRID}_v2.parquet"

YEARS = (2002, 2022)   # Aqua years; night LST must not pass 2022 (drift check)
NTL_HI = 20            # lit threshold; DN 7 carries the DMSP->VIIRS artefact
COV_MIN = 0.75         # min within-pixel night-LST coverage (share of the pixel's best year)

# Story 1 (plan §4). Brightness bins (ntl_harm is continuous; reference DN 0), baseline
# temperature bands (GLASS air temperature, 2002-04 mean, deg C) and the B3d periods.
NTL_BINS = {"bin_lt7": "ntl_harm > 0 AND ntl_harm < 7", "bin_7_20": "ntl_harm >= 7 AND ntl_harm < 20",
            "bin_20_30": "ntl_harm >= 20 AND ntl_harm < 30", "bin_30_50": "ntl_harm >= 30 AND ntl_harm < 50",
            "bin_ge50": "ntl_harm >= 50"}
T_BANDS = {"tb_lt5": (None, 5), "tb_5_13": (5, 13), "tb_13_22": (13, 22), "tb_ge22": (22, None)}
PERIODS = ((2002, 2007), (2008, 2012), (2013, 2017), (2018, 2022))   # 2013 splice on a boundary
SNOW_LAT = 50          # |lat| at or above which a cell counts as snow-prone (B4a)
SNOW_BIOMES = (6, 11)  # RESOLVE biome numbers: boreal forests/taiga, tundra

Y = "lst_night_mean"
Y_DAY = "lst_day_mean"
XLOG = "log_ntl"                            # = log1p(ntl_harm)
Z = "mine_count_20km"
M = "pm25"                                  # ACAG PM2.5, the aerosol mediator
FE = "pixel_id + GID_0^biome_id^year"       # pixel + country x biome x year
FE_CTY = "pixel_id + GID_0^year"            # pixel + country x year
CL = {"CRV1": "GID_1"}                      # cluster by first-level admin region
BASE = "is_flare = 0"                       # sample restriction applied to every fit

# compression=5 rounds continuous strata to 5 sig-figs before grouping (the
# src.analysis production default): ~5x faster demeaning, estimates unchanged
# to 3 decimals vs exact compression.
THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
DR = dict(fitter="auto", compression=5, threads=THREADS, memory_limit="24GB")
# duckreg's working database defaults to one file per data path, which a second
# concurrent job cannot lock; DUCKREG_DB gives a job its own file.
if os.environ.get("DUCKREG_DB"):
    DR["db_name"] = os.environ["DUCKREG_DB"]


def build_panel(path: str = PANEL) -> str:
    """Build the derived regression panel once: the columns the notebooks touch plus
    every transform, one flat zstd parquet. Neighbour means (ring 1 = 8 adjacent
    cells, ring 2 = the next 16) are computed on every cell with lights, then rows
    are restricted to non-null night LST. ``union_by_name`` guards against per-part
    schema drift in the assembled mirror."""
    if os.path.exists(path):
        return path
    from src.analysis.neighbours import add_neighbour_means
    from src.data.assemble.sql_engine import GridFacts

    t = time.time()
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute("SET enable_progress_bar = false")

    bins = ",\n          ".join(f"({cond})::INT AS {b}" for b, cond in NTL_BINS.items())
    period = "CASE " + " ".join(f"WHEN year <= {b} THEN {i}" for i, (_, b) in enumerate(PERIODS)) + " END"
    # lags and leads only across consecutive years
    w = "OVER (PARTITION BY pixel_id ORDER BY year)"
    lag = lambda e, k=1: f"CASE WHEN lag(year, {k}) {w} = year - {k} THEN lag({e}, {k}) {w} END"
    lead = lambda e, k=1: f"CASE WHEN lead(year, {k}) {w} = year + {k} THEN lead({e}, {k}) {w} END"
    df = con.execute(f"""
        SELECT pixel_id, year, GID_0, GID_1, GID_2, biome_id,
          lst_night_mean, lst_day_mean, glass_ta_mean, pm25,
          ntl_harm, viirs_annual_median AS viirs,
          mine_count_10km, mine_count_20km, mine_count_50km, mine_priceshock_20km,
          valid_month_count_night_annual AS vm_night, flare_band, reg_fav,
          -- valid counts are summed over the ~100 native 1 km cells, so coverage is
          -- measured within pixel: this year's valid pixel-months / the pixel's best year
          valid_month_count_night_annual
            / nullif(max(valid_month_count_night_annual) OVER (PARTITION BY pixel_id), 0) AS cov_night,
          valid_month_count_day_annual
            / nullif(max(valid_month_count_day_annual) OVER (PARTITION BY pixel_id), 0)   AS cov_day,
          ln(ntl_harm + 1.0)                                    AS log_ntl,
          asinh(ntl_harm)                                       AS asinh_ntl,
          (ntl_harm >= {NTL_HI})::INT                           AS lit_hi,
          (ntl_harm >= 30)::INT                                 AS lit_30,
          CASE WHEN ntl_harm >= {NTL_HI} THEN ln(ntl_harm) END  AS log_ntl_int,
          ln(greatest(viirs, 0) + 1.0)                          AS log_viirs,
          CASE WHEN year <= 2012 THEN 'dmsp'
               WHEN year  = 2013 THEN 'overlap' ELSE 'viirs' END AS era,
          (coalesce(flare_band, 0) > 0)::INT                    AS is_flare,
          -- outcome shifted by -/+ the treatment: the Anderson-Rubin CI needs the
          -- reduced form of (Y - b * X) on Z at b = +1 and b = -1
          lst_night_mean - ln(ntl_harm + 1.0)                   AS ar_m1,
          lst_night_mean + ln(ntl_harm + 1.0)                   AS ar_p1,
          -- Story 1 (plan §4): functional form (B2, B4c)
          {bins},
          ln(ntl_harm + 0.1)                                    AS log_ntl_c01,
          ln(ntl_harm + 10.0)                                   AS log_ntl_c10,
          CASE WHEN ntl_harm >= {NTL_HI} THEN ln(ntl_harm / {NTL_HI}) ELSE 0 END AS log_ntl_above,
          ln(1.0 + CASE WHEN ntl_harm >= 7 THEN ntl_harm ELSE 0 END) AS log_ntl_d7,
          -- baseline strata (B3c, B5a) and periods (B3d)
          max(CASE WHEN year = {YEARS[0]} THEN lccs_class END) OVER (PARTITION BY pixel_id) AS lc_base,
          avg(CASE WHEN year BETWEEN {YEARS[0]} AND {YEARS[0] + 2} THEN glass_ta_mean END)
            OVER (PARTITION BY pixel_id) - 273.15              AS ta_base_c,
          {period}                                              AS period,
          -- timing (B5b)
          {lead('ln(ntl_harm + 1.0)', 2)}                        AS log_ntl_lead2,
          {lag('ln(ntl_harm + 1.0)')}                            AS log_ntl_lag1,
          {lag('lst_night_mean')}                                AS lst_night_lag1,
          {lag('lst_day_mean')}                                  AS lst_day_lag1
        FROM read_parquet('{SRC}', union_by_name = true, hive_partitioning = true)
        WHERE year BETWEEN {YEARS[0]} AND {YEARS[1]} AND ntl_harm IS NOT NULL
    """).fetchdf()
    con.close()
    g = GridFacts.build(10000.0, (0, 0))
    df = add_neighbour_means(df, [XLOG, M, Y, Y_DAY], g.F, g.W, g.TS)
    df["log_ntl_disc1"] = (df.log_ntl + 8 * df.log_ntl_nb1) / 9     # 3x3 area mean
    # neighbour differences (B3e): own cell minus its 8 adjacent cells
    df["d_log_ntl"] = df.log_ntl - df.log_ntl_nb1
    df["d_lst_night"] = df[Y] - df[f"{Y}_nb1"]
    df["d_lst_day"] = df[Y_DAY] - df[f"{Y_DAY}_nb1"]
    # latitude of the cell centre and the snow-prone flag (B4a)
    df["lat"] = cell_latitude(df.pixel_id.to_numpy(), g)
    df["snow_prone"] = ((df.lat.abs() >= SNOW_LAT) | df.biome_id.isin(SNOW_BIOMES)).astype("int8")
    # baseline temperature band and the lights x band interactions (B5a)
    df["t_band"] = pd.Series(pd.NA, index=df.index, dtype="string")
    for b, (lo, hi) in T_BANDS.items():
        m = df.ta_base_c.notna()
        if lo is not None:
            m &= df.ta_base_c >= lo
        if hi is not None:
            m &= df.ta_base_c < hi
        df.loc[m, "t_band"] = b
        df[f"x_{b}"] = np.where(m, df.log_ntl, 0.0)
    df = df[df[Y].notna()]
    df.to_parquet(path, index=False, compression="zstd")
    del df
    print(f"built {path} in {time.time() - t:.0f}s ({os.path.getsize(path) / 1e6:.0f} MB)")
    return path


def cell_latitude(pixel_id: np.ndarray, g) -> np.ndarray:
    """Latitude (deg) of the centre of each coarse base-grid cell on the canonical
    EPSG:6933 grid (docs/design/01-grid.md)."""
    from pyproj import Transformer
    from src.analysis.neighbours import coarse_row_col
    from src.data.common.geobox.canonical import canonical_ease_geobox

    a = canonical_ease_geobox().affine
    r, _ = coarse_row_col(pixel_id, g.F, g.W, g.TS)
    rows = np.unique(r)
    y = a.f + a.e * (rows * g.F + g.F / 2)          # native rows r*F .. r*F+F-1
    _, lat = Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True).transform(np.zeros_like(y), y)
    return np.asarray(lat, np.float32)[np.searchsorted(rows, r)]


def fit(fml, se=CL, subset=None, data=PANEL, base=True):
    """duckreg with the notebook defaults: full panel, flare mask (BASE), cluster by GID_1."""
    parts = ([BASE] if base else []) + ([subset] if subset else [])
    sub = " AND ".join(f"({p})" for p in parts) or None
    return duckreg(fml, data=data, se_method=se, subset=sub, **DR)


def row(m, term):
    """One coefficient row from a fitted model as a dict."""
    td = m.tidy().set_index("variable")
    if term not in td.index:                       # IV endog term is renamed by duckreg
        term = next((i for i in td.index if term in i and i != "Intercept"), term)
    r = td.loc[term]
    return dict(term=term, est=r["estimate"], se=r["std_error"],
                p=r["p_value"], lo=r["ci_lower"], hi=r["ci_upper"])


def reg_table(models, labels, term):
    """Compact estimate / SE table across models for one term."""
    recs = []
    for m, lab in zip(models, labels):
        rr = row(m, term)
        recs.append({"spec": lab, "estimate": round(rr["est"], 4),
                     "std_error": round(rr["se"], 4), "p": round(rr["p"], 4)})
    return pd.DataFrame(recs).set_index("spec")


def first_stage_F(endog, instr, subset=None, fe=FE):
    """Cluster-robust first-stage F for a single excluded instrument, as (b/se)^2
    from an explicit first-stage regression."""
    r = row(fit(f"{endog} ~ {instr} | {fe}", subset=subset), instr)
    return float((r["est"] / r["se"]) ** 2)


def anderson_rubin_ci(rf, fs, rf_m1, rf_p1, instr=Z, alpha=0.05):
    """Anderson-Rubin confidence set for one endogenous regressor and one instrument.

    The AR test of b0 regresses (Y - b0 X) on Z. Its coefficient is RF - b0 * FS,
    and its cluster-robust variance is exactly quadratic in b0, so three reduced
    forms pin it down: *rf* (b0 = 0), *rf_m1* (outcome Y - X, b0 = 1) and *rf_p1*
    (outcome Y + X, b0 = -1), all fitted on the same sample with the same FE and
    clusters. *fs* is the first stage X ~ Z. Returns (lo, hi, kind, check), where
    kind is "bounded", "union of two rays" or "whole line", and check is the
    rf_m1 coefficient minus RF + FS (should be ~0).
    """
    r0, f, r1, rm = (row(m, instr) for m in (rf, fs, rf_m1, rf_p1))
    a = r0["se"] ** 2
    B = (r1["se"] ** 2 - rm["se"] ** 2) / 2
    C = (r1["se"] ** 2 + rm["se"] ** 2) / 2 - a
    k2 = norm.ppf(1 - alpha / 2) ** 2
    # (RF - b FS)^2 <= k2 (a + B b + C b^2)  <=>  qa b^2 + qb b + qc <= 0
    qa = f["est"] ** 2 - k2 * C
    qb = -2 * r0["est"] * f["est"] - k2 * B
    qc = r0["est"] ** 2 - k2 * a
    disc = qb ** 2 - 4 * qa * qc
    check = r1["est"] - (r0["est"] - f["est"])
    if disc < 0:
        return (-np.inf, np.inf, "whole line", check)
    lo, hi = sorted(((-qb - np.sqrt(disc)) / (2 * qa), (-qb + np.sqrt(disc)) / (2 * qa)))
    return (lo, hi, "bounded" if qa > 0 else "union of two rays", check)


def coefplot(models, labels, term, title):
    """Dot-and-whisker of one term across duckreg models (via src.viz)."""
    import src.viz as viz
    fig = viz.plot_coefficients(
        [viz.from_duckreg(m, label=lab) for m, lab in zip(models, labels)],
        terms=[term],
    )
    fig.axes[0].set_title(title)
    fig.tight_layout()
    return fig


def forest(rows: pd.DataFrame, title: str, ref: float | None = None, xlabel: str = ""):
    """Horizontal dot-and-whisker plot of estimates from different specifications.

    *rows* has columns ``label``, ``est``, ``lo``, ``hi`` and optionally ``group``
    (groups are separated by a gap and labelled on the left). *ref* draws a dashed
    reference line, e.g. the baseline estimate.
    """
    import matplotlib.pyplot as plt

    rows = rows.reset_index(drop=True)
    groups = rows["group"] if "group" in rows else pd.Series([""] * len(rows))
    ypos, y, prev = [], 0.0, None
    for g in groups:
        if prev is not None and g != prev:
            y += 0.6
        ypos.append(y)
        y, prev = y + 1, g
    ypos = np.array(ypos)
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * y + 1.0))
    ax.axvline(0, color="0.6", lw=0.8)
    if ref is not None:
        ax.axvline(ref, color="C3", lw=0.8, ls="--")
    ax.errorbar(rows.est, ypos, xerr=[rows.est - rows.lo, rows.hi - rows.est],
                fmt="o", ms=4, color="C0", ecolor="C0", elinewidth=1.2, capsize=0)
    ax.set_yticks(ypos, [f"{g}: {l}" if g else l for g, l in zip(groups, rows.label)])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig

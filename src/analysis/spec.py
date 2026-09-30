"""
The fixed specification and shared helpers for the three analysis notebooks
(``output/notebooks/01_pre_analysis.ipynb``, ``02_core_analysis.ipynb``,
``03_post_analysis.ipynb``; ``docs/analysis/final-analysis-plan.md`` §2).

Keeping the constants here means the three notebooks cannot drift apart on the
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
PANEL = f"{DATA}/_analysis_panel_{GRID}.parquet"   # derived cache, built by build_panel()

YEARS = (2002, 2022)   # Aqua years; night LST must not pass 2022 (drift check)
NTL_HI = 20            # lit threshold; DN 7 carries the DMSP->VIIRS artefact
COV_MIN = 0.75         # min within-pixel night-LST coverage (share of the pixel's best year)

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
    df = con.execute(f"""
        SELECT pixel_id, year, GID_0, GID_1, biome_id,
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
          lst_night_mean + ln(ntl_harm + 1.0)                   AS ar_p1
        FROM read_parquet('{SRC}', union_by_name = true, hive_partitioning = true)
        WHERE year BETWEEN {YEARS[0]} AND {YEARS[1]} AND ntl_harm IS NOT NULL
    """).fetchdf()
    con.close()
    g = GridFacts.build(10000.0, (0, 0))
    df = add_neighbour_means(df, [XLOG, M], g.F, g.W, g.TS)
    df["log_ntl_disc1"] = (df.log_ntl + 8 * df.log_ntl_nb1) / 9     # 3x3 area mean
    df = df[df[Y].notna()]
    df.to_parquet(path, index=False, compression="zstd")
    del df
    print(f"built {path} in {time.time() - t:.0f}s ({os.path.getsize(path) / 1e6:.0f} MB)")
    return path


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

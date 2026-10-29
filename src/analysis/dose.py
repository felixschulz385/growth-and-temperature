"""
The DOSE fork's region panel (docs/successors/dose-prespec.md §3, §3a), on top
of the fixed specification in ``spec.py``.

``build_region_panel()`` aggregates the retired 10 km panel (its rows are the
cell-years with night LST, ``BASE`` applied) to GADM 4.1 first-level regions
and joins DOSE output and precipitation and the ``adm1_instruments`` tables.
It has **no temperature column**: only lights and row counts are read from the
cell panel, so the per-step first stages (prespec §3) cannot touch the outcome.
``add_lst()`` joins the region means of night and day LST separately, for the
fixed-effects associations §3 always reports.

The region-level fits use pyfixest, not duckreg: the panel is small and needs
regression weights (each region's number of panel cells), which duckreg lacks.
"""
from __future__ import annotations

import json
import os

import duckdb
import numpy as np
import pandas as pd

from src.analysis.spec import BASE, DATA, GRID, PANEL, THREADS

DOSE = f"{DATA}/prepared/dose/adm/dose_gid1.parquet"
INSTR = f"{DATA}/prepared/adm1_instruments/adm"
GID1_MAP = f"{DATA}/prepared/misc/adm/gadm/GID_1_code_mapping.json"
REGION_PANEL = f"{DATA}/_dose_region_panel_{GRID}.parquet"

YEARS = (2002, 2020)          # DOSE ends in 2020
D1_MIN_YEARS = 15             # prespec §2: output in at least 15 of the 19 years
COV_MIN = 0.5                 # prespec §3: half the region's panel cells have night LST
FE_R = "GID_1 + GID_0^year"   # region + country x year
CTRL = "P_a"                  # DOSE precipitation (ERA5); never T_a
CL_R = {"CRV1": "GID_1"}
CL_CTY = {"CRV1": "GID_0"}
W = "n_cells"
# prespec §3a: the three instruments, each used alone
INSTRUMENTS = {"mine count": "mine_count", "mine x price": "price_frozen", "favoritism": "fav_share"}


def _con():
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute("SET enable_progress_bar = false")
    return con


def build_region_panel(path: str = REGION_PANEL) -> pd.DataFrame:
    """Region x year, 2002-2020, every DOSE-matched region: lights over the
    region's panel cells, coverage, DOSE output and precipitation, the
    instruments, and the D1 coverage flag. Cached."""
    if os.path.exists(path):
        return pd.read_parquet(path)
    with open(GID1_MAP) as f:
        id_to_code = pd.DataFrame([(v, k) for k, v in json.load(f).items()], columns=["gid1_id", "GID_1"])
    con = _con()
    con.register("gid1", id_to_code)
    # only pixel_id, year, GID_1 and ntl_harm are read: no LST
    df = con.execute(f"""
        WITH cells AS (
            SELECT pixel_id, year, GID_1 AS gid1_id, ntl_harm FROM '{PANEL}'
            WHERE {BASE} AND GID_1 IS NOT NULL AND year BETWEEN {YEARS[0]} AND {YEARS[1]}
        ),
        n AS (SELECT gid1_id, count(DISTINCT pixel_id) AS n_cells FROM cells GROUP BY 1),
        ry AS (SELECT gid1_id, year, count(*) AS n_rows, avg(ntl_harm) AS ntl_mean FROM cells GROUP BY 1, 2)
        SELECT g.GID_1, ry.year, n.n_cells, ry.n_rows, ry.n_rows / n.n_cells AS cov,
               ln(1 + ry.ntl_mean) AS log_ntl_r
        FROM ry JOIN n USING (gid1_id) JOIN gid1 g USING (gid1_id)
    """).df()
    dose = con.execute(f"""
        SELECT GID_1_gadm41 AS GID_1, year, GID_0, grp_pc_lcu_2015, pop, P_a,
               count(*) FILTER (WHERE grp_pc_lcu_2015 IS NOT NULL AND pop IS NOT NULL)
                 OVER (PARTITION BY GID_1_gadm41) AS n_years
        FROM '{DOSE}' WHERE year BETWEEN {YEARS[0]} AND {YEARS[1]}
    """).df()
    con.close()
    inst = pd.read_parquet(f"{INSTR}/adm1_instruments.parquet").drop(columns="GID_0")
    out = (dose.merge(df, on=["GID_1", "year"], how="left")
               .merge(inst, on=["GID_1", "year"], how="left"))
    out["d1_covered"] = (out.n_years >= D1_MIN_YEARS).astype("int8")
    out["log_grp"] = np.log(out.grp_pc_lcu_2015)
    out["log1p_mine_count"] = np.log1p(out.mine_count)
    out.to_parquet(path, index=False)
    return out


def step_sample(df: pd.DataFrame) -> pd.DataFrame:
    """The rows steps B and C share (prespec §3, §3b): D1-covered regions,
    region-years with output, precipitation and at least half the region's
    panel cells observed. B and C use the same rows, so B -> C changes only
    the growth measure."""
    m = ((df.d1_covered == 1) & df.log_grp.notna() & df[CTRL].notna()
         & df.log_ntl_r.notna() & (df["cov"] >= COV_MIN))
    return df[m].copy()


def price_leave_one_out(df: pd.DataFrame, commodity: str) -> pd.Series:
    """``price_frozen`` without one commodity (prespec §3a, information)."""
    prices = pd.read_parquet(f"{DATA}/prepared/commodity_prices/misc/commodity_prices.parquet")
    exp = pd.read_parquet(f"{INSTR}/adm1_price_exposure.parquet")
    part = (exp[exp.commodity == commodity].merge(prices, on="commodity")
            .assign(v=lambda d: d.weight * d.ln_price_real).groupby(["GID_1", "year"]).v.sum())
    drop = df.set_index(["GID_1", "year"]).index.map(part).fillna(0).to_numpy()
    return df.price_frozen - drop


def first_stage(df: pd.DataFrame, endog: str, instr: str, vcov=CL_R, weights: str | None = W) -> dict:
    """First stage of *endog* on one instrument with the precipitation control,
    region + country x year FE; F = t^2 of the single instrument."""
    import pyfixest as pf

    m = pf.feols(f"{endog} ~ {instr} + {CTRL} | {FE_R}", data=df, vcov=vcov, weights=weights)
    t = m.tidy().loc[instr]
    return dict(endog=endog, instrument=instr, est=t["Estimate"], se=t["Std. Error"],
                F=float((t["Estimate"] / t["Std. Error"]) ** 2), n=int(m._N),
                regions=df.loc[df[instr].notna(), "GID_1"].nunique())


def exposed_regions(df: pd.DataFrame, col: str) -> tuple[int, int]:
    """Regions (and countries) in *df* whose *col* changes over the years
    (prespec §3a: reported before any first stage)."""
    rng = df.groupby("GID_1")[col].agg(lambda s: s.max() - s.min())
    ids = rng[rng > 0].index
    return len(ids), df.loc[df.GID_1.isin(ids), "GID_0"].nunique()


def add_lst(df: pd.DataFrame) -> pd.DataFrame:
    """Join each region-year's mean night and day LST over the same panel
    cells as ``log_ntl_r`` (prespec §3: plain means over the cells)."""
    with open(GID1_MAP) as f:
        id_to_code = pd.DataFrame([(v, k) for k, v in json.load(f).items()], columns=["gid1_id", "GID_1"])
    con = _con()
    con.register("gid1", id_to_code)
    lst = con.execute(f"""
        SELECT g.GID_1, c.year, avg(c.lst_night_mean) AS lst_night_r, avg(c.lst_day_mean) AS lst_day_r
        FROM '{PANEL}' c JOIN gid1 g ON g.gid1_id = c.GID_1
        WHERE {BASE} AND c.year BETWEEN {YEARS[0]} AND {YEARS[1]}
        GROUP BY 1, 2
    """).df()
    con.close()
    return df.merge(lst, on=["GID_1", "year"], how="left")


def association(df: pd.DataFrame, y: str, x: str, vcov=CL_R, weights: str | None = W) -> dict:
    """Fixed-effects association of *y* on *x* (prespec §3: always reported),
    with the precipitation control, region + country x year FE."""
    import pyfixest as pf

    d = df[df[y].notna()]
    m = pf.feols(f"{y} ~ {x} + {CTRL} | {FE_R}", data=d, vcov=vcov, weights=weights)
    t = m.tidy().loc[x]
    return dict(y=y, x=x, est=t["Estimate"], se=t["Std. Error"], p=t["Pr(>|t|)"],
                lo=t["2.5%"], hi=t["97.5%"], n=int(m._N), regions=d.GID_1.nunique())

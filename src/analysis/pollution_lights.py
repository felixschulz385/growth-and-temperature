"""
Step 1 of the pollution follow-up (docs/successors/pollution-lights-prespec.md):
idea B with night lights per person as the growth measure, all countries.

``build_lights_panel()`` sums ``ntl_harm`` and GHS-POP over each GADM 4.1
first-level region from the 1 km tiles, keyed to the assembled 10 km panel's
``pixel_id`` (so a region is exactly the 10 km cells it has in the panel) and
its land mask. The assembled panel is not used for these two sums: it would
average lights over a coastal cell's land part and scale them by a full cell,
and its year clip (1992-2022) drops the 2025 GHS-POP epoch that 2021-22 need.
1 km cells within 5 km of an upstream gas flare in any survey year (EOG
``flare_band`` > 0, 2012-2021) are left out of both sums in every year: a
region is its flare-free part (2026-10-30 change log).

The main sample is the DMSP years, 1998-2012: the DMSP-VIIRS splice moves dim
regions' lights far more than bright ones' (2013 spike, 2014 and 2017 steps),
so year FE cannot absorb it. The VIIRS years 2014-2022 are variant 6.

The panel is built without PM2.5 values: G2 and G2b load none. The only
PM2.5 term is a count of the 10 km cells that have it (the region size rule).
``add_pm25()`` joins the region mean PM2.5 for the main specification (§3).
"""
from __future__ import annotations

import glob
import json
import os

import duckdb
import numpy as np
import pandas as pd

from src.analysis.pollution import (BASELINE, DESERT_BIOME, DOSE, GID1_MAP, INSTITUTIONS, KNOTS_USD,
                                    MIN_REGIONS, VDEM, baseline_terciles, comparisons, spline_terms)
from src.analysis.spec import DATA, GRID, SRC, THREADS
from src.data.assemble.sql_engine import GridFacts, _pixel_id_sql
from src.data.common.geobox.canonical import canonical_ease_geobox

GID0_MAP = f"{DATA}/prepared/misc/adm/gadm/GID_0_code_mapping.json"
NTL = f"{DATA}/prepared/ntl_harm/harmonized/crs/ease6933/ntl_harm"
POP = f"{DATA}/prepared/ghs_pop/r2023a/crs/ease6933/pop"
LAND = f"{DATA}/prepared/misc/crs/ease6933/land_mask"
FLARE = f"{DATA}/prepared/eog/flare/crs/ease6933/eog_flare"
CELL_SUMS = f"{DATA}/_pollution_lights_cell_sums_{GRID}.parquet"
LIGHTS_PANEL = f"{DATA}/_pollution_lights_region_panel_{GRID}.parquet"
REGION_PM25 = f"{DATA}/_pollution_lights_region_pm25_{GRID}.parquet"

YEARS = (1998, 2022)              # ACAG starts 1998; ntl_harm ends 2022 in the panel
MAIN_YEARS = (1998, 2012)         # DMSP; 2013 is the splice year
VIIRS_YEARS = (2014, 2022)        # variant 6
EPOCHS = (1995, 2000, 2005, 2010, 2015, 2020, 2025)
MIN_CELLS = 5                     # 10 km cells with PM2.5 and lights
MIN_R2 = 0.5                      # within-year R2 of the knot calibration
MIN_COUNTRIES, MIN_TERCILE_REGIONS = 15, 150


def _con():
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute("SET enable_progress_bar = false")
    return con


def _cell_km2() -> float:
    res = canonical_ease_geobox().resolution
    return abs(res.x * res.y) / 1e6


def build_cell_sums(path: str = CELL_SUMS) -> pd.DataFrame:
    """Per 10 km ``pixel_id``: Σ ``ntl_harm`` by year (YEARS) and population
    (Σ density x cell area) by epoch, over land 1 km cells, split by whether
    the cell is near a flare (``flare``). One tile at a time."""
    if os.path.exists(path):
        return pd.read_parquet(path)
    g = GridFacts.build(int(GRID.removesuffix("km")) * 1000, (0.0, 0.0))
    pid = _pixel_id_sql("n.cell_id::BIGINT", g, g.F, g.DR, g.DC)
    km2 = _cell_km2()
    con = _con()
    parts = []
    for tile in sorted(glob.glob(f"{LAND}/ix=*/iy=*")):
        rel = os.path.relpath(tile, LAND)
        land = f"read_parquet('{tile}/*.parquet')"
        con.execute(f"""CREATE OR REPLACE TEMP TABLE flare AS
            SELECT DISTINCT cell_id FROM read_parquet('{FLARE}/{rel}/*.parquet') WHERE flare_band > 0""")
        for var, src, years, agg in (
                ("lights", NTL, YEARS, "sum(n.ntl_harm::DOUBLE)"),
                ("pop", POP, (EPOCHS[0], EPOCHS[-1]), f"sum(n.pop::DOUBLE) * {km2}")):
            files = [f for f in glob.glob(f"{src}/{rel}/part-*.parquet")
                     if years[0] <= int(f[-12:-8]) <= years[1]]
            if not files:
                continue
            col = "ntl_harm" if var == "lights" else "pop"
            parts.append(con.execute(f"""
                SELECT {pid} AS pixel_id, n.year, '{var}' AS var,
                       n.cell_id IN (SELECT cell_id FROM flare) AS flare, {agg} AS value
                FROM read_parquet({files!r}) n
                WHERE n.cell_id IN (SELECT cell_id FROM {land} WHERE land_mask)
                  AND NOT isnan(n.{col}::DOUBLE)
                GROUP BY 1, 2, 4
            """).df())
    con.close()
    out = pd.concat(parts, ignore_index=True)
    out["pixel_id"] = out.pixel_id.astype("uint64")
    out.to_parquet(path, index=False)
    return out


def _pixels() -> pd.DataFrame:
    """Each panel pixel's region and country codes, and per year whether it
    has PM2.5 and lights (a count, no PM2.5 value) and is desert."""
    con = _con()
    df = con.execute(f"""
        SELECT pixel_id, year, any_value(GID_0) AS gid0_id, any_value(GID_1) AS gid1_id,
               bool_or(pm25 IS NOT NULL AND ntl_harm IS NOT NULL) AS has_both,
               bool_or(biome_id = {DESERT_BIOME}) AS desert
        FROM read_parquet('{SRC}', union_by_name = true, hive_partitioning = true)
        WHERE GID_1 IS NOT NULL AND year BETWEEN {YEARS[0]} AND {YEARS[1]}
        GROUP BY 1, 2
    """).df()
    con.close()
    return df


def _interpolate_pop(pop: pd.DataFrame) -> pd.DataFrame:
    """Region population for every year in YEARS, log-linear between epochs."""
    wide = pop.pivot(index="GID_1", columns="year", values="pop").reindex(columns=EPOCHS)
    logp = np.log(wide.where(wide > 0))
    years = np.arange(YEARS[0], YEARS[1] + 1)
    out = {}
    for y in years:
        lo = max(e for e in EPOCHS if e <= y)
        hi = min(e for e in EPOCHS if e >= y)
        w = 0.0 if hi == lo else (y - lo) / (hi - lo)
        out[y] = np.exp((1 - w) * logp[lo] + w * logp[hi])
    return (pd.DataFrame(out).rename_axis(columns="year").stack(dropna=False)
            .rename("pop").reset_index())


def build_lights_panel(path: str = LIGHTS_PANEL) -> pd.DataFrame:
    """Region x year, 1998-2022: Σ lights (flare-free, and the flare cells'
    for information), population, lights per person, region size, desert
    share, DOSE output where matched, V-Dem terciles. No PM2.5. Cached."""
    if os.path.exists(path):
        return pd.read_parquet(path)
    sums = build_cell_sums()
    px = _pixels()
    with open(GID1_MAP) as f:
        gid1 = {v: k for k, v in json.load(f).items()}
    with open(GID0_MAP) as f:
        gid0 = {v: k for k, v in json.load(f).items()}
    static = px.groupby("pixel_id")[["gid0_id", "gid1_id"]].first()
    sums = sums.join(static, on="pixel_id", how="inner")
    sums["var"] = sums["var"].where(~sums.flare, sums["var"] + "_flare")
    reg = (sums.groupby(["gid1_id", "var", "year"]).value.sum().unstack("var").reset_index())
    lights = reg.loc[reg.year.between(*YEARS), ["gid1_id", "year", "lights", "lights_flare"]]
    lights["lights_flare"] = lights.lights_flare.fillna(0.0)
    pop = reg.loc[reg.year.isin(EPOCHS), ["gid1_id", "year", "pop"]].rename(columns={"gid1_id": "GID_1"})
    pop = _interpolate_pop(pop).rename(columns={"GID_1": "gid1_id"})
    size = (px.groupby(["gid1_id", "year"])
            .agg(n_cells=("has_both", "sum"), desert_share=("desert", "mean")).reset_index())
    size["desert_share"] = size.groupby("gid1_id").desert_share.transform("mean")
    country = static.drop_duplicates("gid1_id").set_index("gid1_id").gid0_id

    out = size.merge(lights, on=["gid1_id", "year"], how="left").merge(pop, on=["gid1_id", "year"], how="left")
    out["GID_1"] = out.gid1_id.map(gid1)
    out["GID_0"] = out.gid1_id.map(country).map(gid0)
    out["log_lpc"] = np.log(out.lights / out["pop"])
    out["log_lpa"] = np.log(out.lights / out.n_cells)

    con = _con()
    dose = con.execute(f"""
        SELECT GID_1_gadm41 AS GID_1, year, grp_pc_lcu2015_usd AS y_usd
        FROM '{DOSE}' WHERE year BETWEEN {YEARS[0]} AND {YEARS[1]}
    """).df()
    con.close()
    out = out.merge(dose, on=["GID_1", "year"], how="left")

    out = _set_terciles(out, MAIN_YEARS)
    out = out.drop(columns=["gid1_id"])
    out.to_parquet(path, index=False)
    return out


def _set_terciles(df: pd.DataFrame, years) -> pd.DataFrame:
    """``tercile`` and ``tercile_rule`` across the countries with V-Dem's
    baseline index and a region-year in *years* that passes the size rule
    with positive lights and population."""
    vdem = pd.read_parquet(VDEM)
    has_vdem = vdem.loc[vdem.year.between(*BASELINE) & vdem.v2x_polyarchy.notna(), "country_text_id"]
    eligible = df.loc[df.year.between(*years) & df.n_cells.ge(MIN_CELLS) & df.lights.gt(0)
                      & df["pop"].gt(0) & df.GID_0.isin(has_vdem), "GID_0"].unique()
    df = df.copy()
    for name, col in INSTITUTIONS.items():
        key = "tercile" if name == "democracy" else "tercile_rule"
        df[key] = df.GID_0.map(baseline_terciles(vdem, eligible, col))
    return df


def sample(df: pd.DataFrame, years=MAIN_YEARS) -> pd.DataFrame:
    """Rows of the main specification (§2): the DMSP years, region size,
    positive lights and population, a democracy tercile. For other *years*
    (variant 6) the terciles are recomputed across that sample's countries."""
    if tuple(years) != MAIN_YEARS:
        df = _set_terciles(df, years)
    m = (df.year.between(*years) & df.n_cells.ge(MIN_CELLS) & df.lights.gt(0)
         & df["pop"].gt(0) & df.tercile.notna())
    return df[m].copy()


# -- knots (§3) ----------------------------------------------------------------

def calibrate_knots(df: pd.DataFrame, x: str = "log_lpc") -> dict:
    """Pooled OLS of ln lights per person (*x*; ``log_lpa`` for variant 7) on
    ln DOSE output per person with year FE, on the DOSE-matched region-years of
    *df* (the main sample: 1998-2012); knots = predicted *x* at KNOTS_USD at the
    mean year. Falls back to quartiles if the within-year R2 is below MIN_R2."""
    import pyfixest as pf
    d = df[df.y_usd.gt(0)].copy()
    d["log_y"] = np.log(d.y_usd)
    m = pf.feols(f"{x} ~ log_y | year", data=d)
    b = float(m.coef()["log_y"])
    r2 = float(m._r2_within)
    # year FE at the mean year: the FE of the year closest to the mean
    fe = m.fixef()["C(year)"]
    fe = {int(float(k)): v for k, v in fe.items()}
    y_mean = int(round(d.year.mean()))
    knots = [fe[y_mean] + b * np.log(k) for k in KNOTS_USD]
    rule = "DOSE calibration"
    if r2 < MIN_R2:
        w = 1 / df.groupby("GID_1").GID_1.transform("size")
        knots = [_wquantile(df[x], w, q) for q in (0.25, 0.5, 0.75)]
        rule = "quartiles (R2 below 0.5)"
    return dict(rule=rule, knots=np.array(knots), slope=b, r2_within=r2,
                n=int(m._N), regions=int(d.GID_1.nunique()), mean_year=y_mean)


def _wquantile(x: pd.Series, w: pd.Series, q: float) -> float:
    o = np.argsort(x.to_numpy())
    xs, ws = x.to_numpy()[o], w.to_numpy()[o]
    c = np.cumsum(ws) / ws.sum()
    return float(xs[np.searchsorted(c, q)])


def segment(log_l: pd.Series, knots) -> pd.Series:
    return pd.Series(np.searchsorted(knots, log_l), index=log_l.index)


# -- G2, G2b (§4) --------------------------------------------------------------

def g2(df: pd.DataFrame, knots, x: str = "log_lpc") -> dict:
    """Coverage by tercile; supported tercile x segment cells; the adjacent
    comparisons; G2b (high tercile in the lowest segment). Loads no PM2.5."""
    d = df.assign(segment=segment(df[x], knots))
    by = d.groupby("tercile").agg(countries=("GID_0", "nunique"), regions=("GID_1", "nunique"))
    by["passes"] = by.countries.ge(MIN_COUNTRIES) & by.regions.ge(MIN_TERCILE_REGIONS)
    n = d.groupby(["tercile", "segment"]).GID_1.nunique().unstack(fill_value=0)
    n = n.reindex(index=[0, 1, 2], columns=range(len(knots) + 1), fill_value=0)
    n.columns = [f"seg{k}" for k in n.columns]
    return dict(by_tercile=by, regions=n, supported=n.ge(MIN_REGIONS),
                comparisons=comparisons(n), g2_passes=bool(by.passes.all()),
                g2b_high_seg0=int(n.loc[2, "seg0"]),
                g2b_passes=bool(n.loc[2, "seg0"] >= MIN_REGIONS))


# -- main specification (§3): loads PM2.5 --------------------------------------

def region_pm25(path: str = REGION_PM25) -> pd.DataFrame:
    """Region x year, 1998-2022: mean PM2.5 over the region's 10 km panel cells
    that have it (equal-area cells), as idea B; the same mean weighted by each
    cell's GHS-POP population (variant 5; linear between epochs, all land 1 km
    cells, flare cells included); mean GLASS air temperature (variant 8). Cached."""
    if os.path.exists(path):
        return pd.read_parquet(path)
    with open(GID1_MAP) as f:
        codes = pd.DataFrame([(v, k) for k, v in json.load(f).items()], columns=["gid1_id", "GID_1"])
    sums = build_cell_sums()
    pop = (sums[sums["var"] == "pop"].groupby(["pixel_id", "year"]).value.sum()
           .unstack("year").reindex(columns=EPOCHS).fillna(0.0))
    pop.columns = [f"p{e}" for e in EPOCHS]
    pop = pop.reset_index()
    w = []
    for y in range(YEARS[0], YEARS[1] + 1):
        lo, hi = max(e for e in EPOCHS if e <= y), min(e for e in EPOCHS if e >= y)
        a = 0.0 if hi == lo else (y - lo) / (hi - lo)
        w.append(f"WHEN {y} THEN {1 - a} * w.p{lo} + {a} * w.p{hi}")
    con = _con()
    con.register("codes", codes)
    con.register("pop", pop)
    out = con.execute(f"""
        WITH cells AS (
            SELECT pixel_id, year, any_value(GID_1) AS gid1_id, avg(pm25) AS pm25,
                   avg(glass_ta_mean) AS ta
            FROM read_parquet('{SRC}', union_by_name = true, hive_partitioning = true)
            WHERE GID_1 IS NOT NULL AND year BETWEEN {YEARS[0]} AND {YEARS[1]}
            GROUP BY 1, 2
        ), weighted AS (
            SELECT x.*, CASE x.year {' '.join(w)} END AS w
            FROM cells x LEFT JOIN pop w ON w.pixel_id = x.pixel_id
        )
        SELECT c.GID_1, x.year, avg(x.pm25) AS pm25,
               sum(x.pm25 * x.w) / nullif(sum(CASE WHEN x.pm25 IS NOT NULL THEN x.w END), 0) AS pm25_popw,
               avg(x.ta) AS glass_ta
        FROM weighted x JOIN codes c ON c.gid1_id = x.gid1_id
        GROUP BY 1, 2
        HAVING count(x.pm25) > 0
    """).df()
    con.close()
    out.to_parquet(path, index=False)
    return out


def add_pm25(df: pd.DataFrame, knots, x: str = "log_lpc") -> pd.DataFrame:
    """*df* (a sample) with ln PM2.5 (and ln population-weighted PM2.5), the
    spline basis ``seg0``-``seg3`` in *x* at *knots* (ln values), and
    ``segment``. Rows without positive PM2.5 drop."""
    out = df.merge(region_pm25(), on=["GID_1", "year"], how="inner")
    out = out[out.pm25.gt(0)].copy()
    out["log_pm25"] = np.log(out.pm25)
    out["log_pm25_popw"] = np.log(out.pm25_popw.where(out.pm25_popw > 0))
    out = out.join(spline_terms(out[x], knots=np.exp(knots)))
    out["segment"] = segment(out[x], knots)
    return out


def usd_equivalent(log_l: pd.Series, k: dict) -> pd.Series:
    """ln lights per person as ln output by the knot calibration *k*
    (``calibrate_knots``): the knots map to ln KNOTS_USD. Figures only."""
    return (log_l - k["knots"][0]) / k["slope"] + np.log(KNOTS_USD[0])


def low_haze(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Variant H: regions in the lower two terciles of their baseline
    (1998-2000) mean PM2.5, cutoffs across *df*'s regions, each counted once.
    Regions without a baseline year drop. Returns the rows and the cutoffs."""
    base = df[df.year.between(*BASELINE)].groupby("GID_1").pm25.mean()
    cuts = base.quantile([1 / 3, 2 / 3]).to_numpy()
    keep = base.index[base <= cuts[1]]
    return df[df.GID_1.isin(keep)].copy(), cuts

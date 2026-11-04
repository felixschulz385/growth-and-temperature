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

NO2_YEARS = (2005, 2020)          # annual from 2005; earlier years use another scaling
DOSE_YEARS = (2005, 2018)         # main arm: DOSE P_a and T_a stop 2018
LIGHTS_YEARS = (2005, 2012)       # check arm: DMSP years
MAX_PPB = 100.0                   # 1 km cells above this are missing
MIN_NO2_CELLS = 500               # valid 1 km NO2 cells per region-year
STEP1_KNOTS = (-4.017, -3.020, -2.231)    # step 1 rerun, 2026-11-04


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

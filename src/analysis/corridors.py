"""
The corridor fork's analysis inputs (docs/successors/corridors-prespec.md),
on top of the fixed specification in ``spec.py``.

Two derived files, both cached under the data root:

- ``cells_xs(variant)``: one row per 10 km cell with the corridor flags,
  on-route indicator, slope, land share, distance to the nearest used port and
  GRIP4 road density (route check R1, prespec §6a).
- ``build_panel(variant)``: the retired panel's rows (same sample: non-null
  night LST, 2002-2022) with the treatment, FE and cluster columns plus the
  corridor exposure -- and **no temperature column**, so Gate C1 (prespec §7)
  cannot touch the outcome. LST is added only after C1 passes.
"""
from __future__ import annotations

import os
import time

import duckdb
import numpy as np
import pandas as pd

from src.analysis.spec import DATA, FE, GRID, PANEL, THREADS

CORR = f"{DATA}/prepared/corridors/misc"
FE_C = f"{FE} + mine_dist_band^year"        # prespec §6: + distance band x year
CL_PORT = {"CRV1": "port_code"}             # prespec §6: port catchment (main)
ZC = "z_main_rc"                            # prespec §5: recentered instrument
MAIN_KM = (20, 300)                         # prespec §4: main bands

# Panel columns kept for C1: treatment, FE, clusters, sample flags. No LST.
_PANEL_COLS = ("pixel_id", "year", "GID_0", "GID_1", "biome_id", "ntl_harm", "log_ntl",
               "is_flare", "mine_count_20km", "log_ntl_lead2", "log_ntl_lag1")


def _con():
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={THREADS}")
    con.execute("SET enable_progress_bar = false")
    return con


def road_density(path: str = f"{DATA}/_corridors_roads_{GRID}.parquet") -> str:
    """GRIP4 road density per analysis cell (mean over its 1 km pixels), cached."""
    if os.path.exists(path):
        return path
    from src.data.assemble.sql_engine import GridFacts, _pixel_id_sql

    g = GridFacts.build(10000.0, (0, 0))
    pid = _pixel_id_sql("cell_id", g, g.F, 0, 0)
    con = _con()
    con.execute(f"""
        COPY (SELECT {pid} AS pixel_id,
                     avg(grip_tp1_m_km2 + grip_tp2_m_km2) AS road_hp,
                     avg(grip_total_m_km2) AS road_total
              FROM read_parquet('{DATA}/prepared/grip/crs/ease6933/road_density/ix=*/iy=*/part.parquet')
              WHERE NOT isnan(grip_total_m_km2) GROUP BY 1)
        TO '{path}' (FORMAT parquet, COMPRESSION zstd)""")
    con.close()
    return path


def cells_xs(variant: str = "main") -> pd.DataFrame:
    """One row per cell: corridor flags, on-route indicator, R1 covariates."""
    from src.data.sources.corridors.routing import PointIndex

    d = f"{CORR}/{variant}"
    con = _con()
    df = con.execute(f"""
        WITH on_route AS (SELECT DISTINCT pixel_id FROM '{d}/routes.parquet'
                          WHERE dist_km >= {MAIN_KM[0]} AND dist_km < {MAIN_KM[1]})
        SELECT c.pixel_id, c.GID_0, c.lon, c.lat, c.land_share, c.slope,
               f.port_id, f.mine_dist_km, f.mine_dist_band, f.near_mine, f.near_port,
               (o.pixel_id IS NOT NULL)::INT AS on_route,
               r.road_hp, r.road_total
        FROM '{CORR}/cells_10km.parquet' c
        JOIN '{d}/cell_flags.parquet' f USING (pixel_id)
        LEFT JOIN on_route o USING (pixel_id)
        LEFT JOIN '{road_density()}' r USING (pixel_id)
    """).df()
    ports = con.execute(f"SELECT lon, lat FROM '{d}/ports.parquet' WHERE used").df()
    con.close()
    _, df["port_dist_km"] = PointIndex(ports.lon.to_numpy(), ports.lat.to_numpy()).nearest(df.lon, df.lat)
    df["log_road_hp"] = np.log1p(df.road_hp)
    df["port_code"] = df.port_id.astype("category").cat.codes.astype("int64")
    return df


def build_panel(variant: str = "main") -> str:
    """The C1 panel: retired panel rows x corridor exposure, no LST (module docstring)."""
    path = f"{DATA}/_analysis_panel_{GRID}_corridors_{variant}.parquet"
    if os.path.exists(path):
        return path
    t = time.time()
    d = f"{CORR}/{variant}"
    xs = cells_xs(variant)[["pixel_id", "port_code", "mine_dist_band", "near_mine", "near_port", "on_route"]]
    con = _con()
    con.register("xs", xs)
    cols = ", ".join(f"p.{c}" for c in _PANEL_COLS)
    con.execute(f"""
        COPY (SELECT {cols}, xs.port_code, xs.mine_dist_band, xs.near_mine::INT AS near_mine,
                     xs.near_port::INT AS near_port, xs.on_route,
                     coalesce(e.z_20_100, 0) AS z_20_100, coalesce(e.z_100_300, 0) AS z_100_300,
                     coalesce(e.z_300_plus, 0) AS z_300_plus, coalesce(e.z_main, 0) AS z_main,
                     coalesce(e.mu_main, 0) AS mu_main, coalesce(e.z_main_rc, 0) AS z_main_rc,
                     (coalesce(e.z_main, 0) > 0)::INT AS z_any
              FROM read_parquet('{PANEL}') p
              JOIN xs USING (pixel_id)
              LEFT JOIN '{d}/exposure.parquet' e USING (pixel_id, year))
        TO '{path}' (FORMAT parquet, COMPRESSION zstd)""")
    con.close()
    print(f"built {path} in {time.time() - t:.0f}s ({os.path.getsize(path) / 1e6:.0f} MB)")
    return path


#: prespec §6 sample: outside 20 km of any mine and 30 km of a used port
SAMPLE = "near_mine = 0 AND near_port = 0"

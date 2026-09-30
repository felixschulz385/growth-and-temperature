"""
Build the 10 km neighbour panel for the ring check (docs/analysis/10km-ring-check.md).

Reads the assembled 10 km base grid (Aqua years 2002-2022) and adds, per
cell-year, ring-1 and ring-2 neighbour means (src/analysis/neighbours.py) of
log1p(ntl_harm), mine_count_20km and mine_count_10km:

  *_nb1   ring 1: the 8 queen neighbours (centre distance 10-14 km)
  *_nb2   ring 2: the 16 cells at Chebyshev distance 2 (20-28 km)

plus the 3x3 and 5x5 area means of lights (log_ntl_disc1, log_ntl_disc2).
Neighbour means use every cell with data; rows are then restricted to
non-null lst_night_mean.

Usage:
    python scripts/ring_check_build_panel.py OUT.parquet
"""
import sys
import time
from pathlib import Path

import duckdb

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.analysis.neighbours import add_neighbour_means  # noqa: E402
from src.data.assemble.sql_engine import GridFacts  # noqa: E402

SRC = f"{PROJECT}/data_nobackup/assembled/grid=10km/shake=base/ix=*/iy=*/*.parquet"


def main(out: str) -> None:
    t0 = time.time()
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    df = con.execute(f"""
      SELECT pixel_id, year, GID_0, GID_1, biome_id,
             lst_night_mean, lst_day_mean,
             ln(ntl_harm + 1.0) AS log_ntl, mine_count_20km, mine_count_10km
      FROM read_parquet('{SRC}', hive_partitioning = true, union_by_name = true)
      WHERE year BETWEEN 2002 AND 2022 AND ntl_harm IS NOT NULL
    """).fetchdf()
    print(f"loaded {len(df):,} rows in {time.time()-t0:.0f}s", flush=True)

    g = GridFacts.build(10000.0, (0, 0))
    df = add_neighbour_means(df, ["log_ntl", "mine_count_20km", "mine_count_10km"], g.F, g.W, g.TS)
    df["log_ntl_disc1"] = (df.log_ntl + 8 * df.log_ntl_nb1) / 9               # 3x3 area mean
    df["log_ntl_disc2"] = (9 * df.log_ntl_disc1 + 16 * df.log_ntl_nb2) / 25   # 5x5 area mean
    df = df[df.lst_night_mean.notna() & df.log_ntl_nb2.notna()]
    df.to_parquet(out, index=False)
    print(f"sample {len(df):,} rows, {df.pixel_id.nunique():,} pixels; "
          f"saved {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

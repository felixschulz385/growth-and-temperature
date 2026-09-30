"""
Clear-sky coverage check for ACAG PM2.5 (analysis plan §7, open question 2).

ACAG's annual PM2.5 draws on satellite AOD, retrieved only under clear daytime
skies, and the annual files carry no per-cell data-support field. Aqua MODIS
daytime LST shares the sensor, the 13:30 overpass and the cloud screening, so
its valid counts proxy how often AOD could be retrieved in a cell-year:

  cov_day   valid pixel-months / the pixel's best year (within-pixel coverage;
            raw counts are sums over the ~100 native 1 km cells at 10 km)
  dens_day  valid 8-day periods per valid pixel-month (clear-sky frequency
            within observed months, independent of the land-pixel count)

Tests, all with pixel + country x biome x year FE and GID_1 clusters:

  T1  Does ACAG track sampling?   pm25 ~ cov_day, pm25 ~ dens_day
  T2  Does the instrument move sampling?   cov_day ~ Z, dens_day ~ Z
      (if not, a coverage artefact in ACAG is orthogonal to Z and cannot
      bias the Phase 6a reduced form of PM2.5 on Z)
  T3  Reduced form pm25 ~ Z, overall and by baseline (2002-06) clear-sky
      density tercile (attenuation where skies are cloudiest would point to
      coverage-driven damping of the aerosol response)

Usage:
    python scripts/acag_coverage_check.py OUT_DIR
"""
import os
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get("DUCKREG_PATH", str(PROJECT.parent / "duckreg")))
from duckreg import duckreg  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
SRC = f"{PROJECT}/data_nobackup/assembled/grid=10km/shake=base/ix=*/iy=*/*.parquet"
PANEL = str(OUT / "acag_panel.parquet")
RES = str(OUT / "acag_results.csv")
THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
DR = dict(fitter="auto", compression=5, threads=THREADS, memory_limit="96GB")
FE = "pixel_id + GID_0^biome_id^year"
CL = {"CRV1": "GID_1"}
Z = "mine_count_20km"

if not os.path.exists(PANEL):
    t = time.time()
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={THREADS}; SET enable_progress_bar = false")
    con.execute(f"""
      COPY (
        WITH b AS (
          SELECT pixel_id, year, GID_0, GID_1, biome_id, pm25, {Z},
                 valid_month_count_day_annual AS vm, valid_period_count_day_annual AS vp
          FROM read_parquet('{SRC}', hive_partitioning = true, union_by_name = true)
          WHERE year BETWEEN 2002 AND 2022 AND pm25 IS NOT NULL
            AND coalesce(flare_band, 0) = 0
        ), c AS (
          SELECT *, vm / nullif(max(vm) OVER (PARTITION BY pixel_id), 0) AS cov_day,
                    vp / nullif(vm, 0)                                   AS dens_day
          FROM b
        )
        SELECT c.*, ntile(3) OVER (ORDER BY base.dens0) AS dens_tercile
        FROM c JOIN (
          SELECT pixel_id, avg(dens_day) AS dens0 FROM c
          WHERE year BETWEEN 2002 AND 2006 GROUP BY pixel_id HAVING avg(dens_day) IS NOT NULL
        ) base USING (pixel_id)
      ) TO '{PANEL}' (FORMAT parquet, COMPRESSION zstd)
    """)
    print(f"built panel in {time.time()-t:.0f}s", flush=True)

print(duckdb.sql(f"""
  SELECT count(*) n, count(DISTINCT pixel_id) px, avg(pm25) pm25, avg(cov_day) cov, avg(dens_day) dens,
         corr(pm25, dens_day) raw_corr_pm_dens
  FROM read_parquet('{PANEL}')""").df().round(3).to_string(), flush=True)
print(duckdb.sql(f"""
  SELECT dens_tercile, count(DISTINCT pixel_id) px, avg(dens_day) dens, avg(pm25) pm25,
         avg(({Z} > 0)::INT) share_exposed
  FROM read_parquet('{PANEL}') GROUP BY 1 ORDER BY 1""").df().round(3).to_string(), flush=True)

done = set(pd.read_csv(RES).spec) if os.path.exists(RES) else set()


def run(spec, formula, term, subset=None):
    if spec in done:
        return
    t = time.time()
    td = duckreg(formula, data=PANEL, se_method=CL, subset=subset, **DR).tidy().set_index("variable")
    r = td.loc[term]
    row = dict(spec=spec, term=term, est=r["estimate"], se=r["std_error"], p=r["p_value"])
    pd.DataFrame([row]).to_csv(RES, mode="a", header=not os.path.exists(RES), index=False)
    print(f"{spec}: {term}={row['est']:.4f} ({row['se']:.4f}) p={row['p']:.3f}  [{time.time()-t:.0f}s]", flush=True)


run("T2 cov_day ~ Z", f"cov_day ~ {Z} | {FE}", Z)
run("T2 dens_day ~ Z", f"dens_day ~ {Z} | {FE}", Z)
run("T3 pm25 ~ Z", f"pm25 ~ {Z} | {FE}", Z)
run("T1 pm25 ~ cov_day", f"pm25 ~ cov_day | {FE}", "cov_day")
run("T1 pm25 ~ dens_day", f"pm25 ~ dens_day | {FE}", "dens_day")
for k in (1, 2, 3):
    run(f"T3 pm25 ~ Z | dens tercile {k}", f"pm25 ~ {Z} | {FE}", Z, subset=f"dens_tercile = {k}")
print("ALL DONE", flush=True)

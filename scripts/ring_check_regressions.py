"""
Regressions for the 10 km ring check (docs/analysis/10km-ring-check.md).

On the panel from scripts/ring_check_build_panel.py, with pixel + country x
year FE and clusters at GID_0 and GID_1:

  1. first-stage profile: mine_count_20km on own, ring-1, ring-2 and disc lights
  2. reduced form and 2SLS of night/day LST on own-cell vs 3x3 / 5x5 disc lights
  3. neighbour-lag 2SLS (own + ring-1 lights, instrumented by own and ring-1
     mine_count_10km) with its two first stages, GID_0 clusters only

Each fit appends one row to <panel>_results.csv as soon as it finishes; a
rerun skips fits already in the file.

Usage (duckreg source checkout as a sibling of this repo, or DUCKREG_PATH):
    python scripts/ring_check_regressions.py PANEL.parquet
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get("DUCKREG_PATH", str(PROJECT.parent / "duckreg")))
from duckreg import duckreg  # noqa: E402

P = sys.argv[1]
OUT = P.replace(".parquet", "_results.csv")
THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
DR = dict(fitter="auto", compression=5, threads=THREADS, memory_limit="96GB")
FE = "pixel_id + GID_0^year"
SUB = "log_ntl_nb1 IS NOT NULL"
Z = "mine_count_20km"
ZL = "mine_count_10km + mine_count_10km_nb1"
CLS = {"GID_0": {"CRV1": "GID_0"}, "GID_1": {"CRV1": "GID_1"}}

done = set()
if os.path.exists(OUT):
    d = pd.read_csv(OUT)
    done = set(zip(d.spec, d.term, d.cluster))


def run(spec, formula, terms, cl):
    if all((spec, t, cl) in done for t in terms):
        return
    t0 = time.time()
    td = duckreg(formula, data=P, se_method=CLS[cl], subset=SUB, **DR).tidy().set_index("variable")
    rows = []
    for t in terms:
        # duckreg renames the IV endogenous term; match on substring
        k = t if t in td.index else next(i for i in td.index if t in i and i != "Intercept")
        r = td.loc[k]
        rows.append(dict(spec=spec, term=t, cluster=cl, est=r["estimate"], se=r["std_error"],
                         p=r["p_value"], F=(r["estimate"] / r["std_error"]) ** 2))
    pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    print(f"{spec} [{cl}] {time.time()-t0:.0f}s: " +
          ", ".join(f"{r['term']}={r['est']:.4f} ({r['se']:.4f})" for r in rows), flush=True)


jobs = []
for cl in ["GID_0", "GID_1"]:
    for dv in ["log_ntl", "log_ntl_nb1", "log_ntl_nb2", "log_ntl_disc1", "log_ntl_disc2"]:
        jobs.append((f"FS {dv}", f"{dv} ~ {Z} | {FE}", [Z], cl))
    for y in ["lst_night_mean", "lst_day_mean"]:
        jobs.append((f"RF {y}", f"{y} ~ {Z} | {FE}", [Z], cl))
        for x in ["log_ntl", "log_ntl_disc1", "log_ntl_disc2"]:
            jobs.append((f"2SLS {y} on {x}", f"{y} ~ 1 | {FE} | ({x} ~ {Z})", [x], cl))
    if cl == "GID_0":
        jobs.append(("NL 2SLS lst_night_mean",
                     f"lst_night_mean ~ 1 | {FE} | (log_ntl + log_ntl_nb1 ~ {ZL})",
                     ["log_ntl", "log_ntl_nb1"], cl))
        for dv in ["log_ntl", "log_ntl_nb1"]:
            jobs.append((f"NL FS {dv}", f"{dv} ~ {ZL} | {FE}",
                         ["mine_count_10km", "mine_count_10km_nb1"], cl))

for j in jobs:
    run(*j)
print("ALL DONE", flush=True)

"""
Summarise the Aqua night-overpass drift arrays from scripts/aqua_drift_diagnostic.py.

Per pixel:
  drift_h    Aqua night view time minus its 2018-19 mean (hours; negative =
             earlier overpass)
  rate_kph   night cooling rate from the pre-drift Terra (~22:30) vs Aqua
             (~01:30) pair, (LST_terra - LST_aqua) / (t_aqua - t_terra), mean
             of 2018 and 2019 (K per hour, positive = cooling)
  bias_k     implied LST bias of the drift = -rate_kph * drift_h (an earlier
             overpass sees a warmer surface)
  dlst_k     observed Aqua LST minus its 2018-19 mean (drift plus weather)

and reports medians and 10th/90th percentiles by tile x year x 2-degree
latitude band, plus the spread of the band medians within each tile-year.
Only the *heterogeneity* of bias_k within a country-biome-year matters for the
panel; its common part is absorbed by the fixed effects.

Usage:
    python scripts/aqua_drift_summary.py scratch_nobackup/drift
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

D = sys.argv[1]
BASE_YEARS = (2018, 2019)


def load(tile, platform, year):
    p = os.path.join(D, f"{tile}_{platform}_{year}.npz")
    return dict(np.load(p)) if os.path.exists(p) else None


def signed_hours(vt):
    """Local solar hours on a continuous scale around midnight (22:30 -> -1.5)."""
    return np.where(vt > 12, vt - 24, vt)


rows = []
tiles = sorted({os.path.basename(p).split("_")[0] for p in glob.glob(os.path.join(D, "*.npz"))})
for tile in tiles:
    base_a = [load(tile, "aqua", y) for y in BASE_YEARS]
    base_t = [load(tile, "terra", y) for y in BASE_YEARS]
    if any(x is None for x in base_a):
        print(f"{tile}: missing Aqua base years, skipped")
        continue
    lat = base_a[0]["lat"]
    lat2d = np.broadcast_to(lat[:, None], base_a[0]["lst"].shape)
    vt0 = np.nanmean([signed_hours(b["view_time"]) for b in base_a], axis=0)
    lst0 = np.nanmean([b["lst"] for b in base_a], axis=0)
    rate = np.full(vt0.shape, np.nan, "float32")
    if all(x is not None for x in base_t):
        rates = []
        for a, t in zip(base_a, base_t):
            gap = signed_hours(a["view_time"]) - signed_hours(t["view_time"])
            with np.errstate(invalid="ignore", divide="ignore"):
                rates.append(np.where(gap > 0.5, (t["lst"] - a["lst"]) / gap, np.nan))
        rate = np.nanmean(rates, axis=0)
    band = (np.floor(lat2d / 2) * 2).astype(int)

    for f in sorted(glob.glob(os.path.join(D, f"{tile}_aqua_*.npz"))):
        year = int(f.rsplit("_", 1)[1][:4])
        a = load(tile, "aqua", year)
        drift = signed_hours(a["view_time"]) - vt0
        bias = -rate * drift
        dlst = a["lst"] - lst0
        df = pd.DataFrame({"band": band.ravel(), "drift_h": drift.ravel(), "rate_kph": rate.ravel(),
                           "bias_k": bias.ravel(), "dlst_k": dlst.ravel()}).dropna(subset=["drift_h"])
        for b, g in df.groupby("band"):
            if len(g) < 500:
                continue
            q = g.quantile([0.1, 0.5, 0.9])
            rows.append(dict(tile=tile, year=year, lat_band=f"{b}..{b+2}", n_px=len(g),
                             drift_h=q.drift_h[0.5], drift_p10=q.drift_h[0.1], drift_p90=q.drift_h[0.9],
                             rate_kph=q.rate_kph[0.5],
                             bias_k=q.bias_k[0.5], bias_p10=q.bias_k[0.1], bias_p90=q.bias_k[0.9],
                             dlst_k=q.dlst_k[0.5]))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(D, "drift_summary.csv"), index=False)
pd.set_option("display.width", 220)

print("\nMedian drift in overpass time (hours, vs 2018-19), by tile x year (median over bands):")
print(res.pivot_table(index="tile", columns="year", values="drift_h", aggfunc="median").round(2).to_string())
print("\nMedian night cooling rate (K/h, 2018-19 Terra vs Aqua), by tile:")
print(res[res.year == 2019].groupby("tile").rate_kph.median().round(2).to_string())
print("\nMedian implied LST bias (K), by tile x year:")
print(res.pivot_table(index="tile", columns="year", values="bias_k", aggfunc="median").round(3).to_string())
print("\nWithin-tile spread of band-median implied bias (max - min across 2-deg bands, K):")
spread = res.groupby(["tile", "year"]).bias_k.agg(lambda s: s.max() - s.min()).unstack()
print(spread.round(3).to_string())
print("\nWithin-band pixel spread of implied bias (median over bands of p90 - p10, K):")
print(res.assign(w=res.bias_p90 - res.bias_p10).pivot_table(index="tile", columns="year", values="w",
      aggfunc="median").round(3).to_string())

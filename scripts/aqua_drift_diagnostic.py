"""
Aqua night-overpass drift diagnostic (docs/design/14-terra-aqua-drift-diagnostic.md,
narrowed to the analysis plan's Phase 0 panel-end rule).

Aqua's orbit has drifted since its final inclination maneuver (2020/21), so the
local solar time of the MYD21A2 night observation shifts. Pixel and country x
biome x year FE absorb the common part; what could bias the panel is drift whose
LST impact differs *within* a country-biome-year (by latitude or land cover).

Per tile, this script writes one .npz per (platform, year) holding the
month-first annual means (same STAC search, tile-pinned load, QC mask and
compositing as FETCH) of

  view_time   View_Time_Night, local solar hours
  lst         LST_Night_1KM, K
  n_months    months with a valid night observation

on the tile's 1200x1200 sinusoidal grid, plus `lat` per row. Aqua is loaded for
a year ladder bracketing 2020/21; Terra only for pre-drift years, where
(Terra LST - Aqua LST) / (Aqua time - Terra time) gives each pixel's night
cooling rate (K/h). scripts/aqua_drift_summary.py turns these into the implied
LST bias of Aqua's drift. Read-only against Planetary Computer.

Usage (one tile per SLURM array task):
    python scripts/aqua_drift_diagnostic.py --tile h18v04 --out scratch_nobackup/drift
"""
from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.cli.config import load_config_with_env_vars  # noqa: E402
from src.config.runtime import get_paths_config  # noqa: E402
from src.data.common.raster.compositing import composite_annual_stats  # noqa: E402
from src.data.pipeline.config import get_source_config  # noqa: E402
from src.data.pipeline.context import PipelineContext  # noqa: E402
from src.data.sources import registry  # noqa: E402
from src.data.sources.modis import tiles as modis_util  # noqa: E402

logger = logging.getLogger("aqua_drift")

AQUA_YEARS = (2003, 2008, 2013, 2016, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)
TERRA_YEARS = (2018, 2019)          # pre-drift: cooling-rate reference only
BANDS = ("lst", "qc", "view_time")  # load only what the diagnostic needs


def build_source(config_path: str):
    config = load_config_with_env_vars(config_path)
    data_root = get_paths_config(config).get("data_root")
    cls = registry.load("modis")
    src = cls(PipelineContext(data_root=data_root), get_source_config(config, "modis"))
    src.band_spec = copy.deepcopy(src.band_spec)
    src.band_spec["assets"] = {k: v for k, v in src.band_spec["assets"].items() if k in BANDS}
    return src


def row_latitudes(tile: str) -> np.ndarray:
    """Latitude (deg) of each of the tile's 1200 pixel rows, top to bottom."""
    v = int(tile[4:6])
    _, y0, _, y1 = modis_util.tile_bounds_m(0, v)
    res = (y1 - y0) / 1200
    y = y1 - (np.arange(1200) + 0.5) * res
    return np.degrees(y / modis_util.SPHERE_RADIUS_M).astype("float32")


def tile_year(src, tile: str, year: int, platform: str) -> dict | None:
    src.platform = platform
    items = src._search_items(tile, year)
    if not items:
        logger.warning("no STAC items: %s %s %d", platform, tile, year)
        return None
    ds = src._load_tile_year(items, tile)
    if ds is None or not set(BANDS) <= set(ds.data_vars):
        logger.warning("missing bands: %s %s %d", platform, tile, year)
        return None
    valid = modis_util.decode_qc_valid_mask(
        ds["qc"], src.qc_max_lst_error_k, product=src.product,
        lst=ds["lst"], min_lst_k=src.lst_min_k, max_lst_k=src.lst_max_k,
    )
    lst = composite_annual_stats(ds["lst"], valid, stats=("mean", "valid_month_count"))
    # View time is only meaningful where the LST observation is valid, so it is
    # composited under the same mask.
    vt = composite_annual_stats(ds["view_time"], valid, stats=("mean",))
    out = {
        "lst": lst["mean"].squeeze("time", drop=True),
        "n_months": lst["valid_month_count"].squeeze("time", drop=True),
        "view_time": vt["mean"].squeeze("time", drop=True),
    }
    return {k: np.asarray(v.compute().values, dtype="float32") for k, v in out.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tile", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=str(PROJECT / "orchestration/configs/data.yaml"))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    os.makedirs(args.out, exist_ok=True)
    src = build_source(args.config)
    lat = row_latitudes(args.tile)
    jobs = [("aqua", y) for y in AQUA_YEARS] + [("terra", y) for y in TERRA_YEARS]
    for platform, year in jobs:
        path = os.path.join(args.out, f"{args.tile}_{platform}_{year}.npz")
        if os.path.exists(path):
            continue
        t0 = time.time()
        try:
            res = tile_year(src, args.tile, year, platform)
        except Exception:
            logger.exception("failed: %s %s %d", platform, args.tile, year)
            continue
        if res is None:
            continue
        np.savez_compressed(path, lat=lat, **res)
        ok = np.isfinite(res["view_time"])
        logger.info("%s %s %d: %.0fs, %d valid px, mean view time %.2f h, mean LST %.1f K",
                    args.tile, platform, year, time.time() - t0, ok.sum(),
                    np.nanmean(res["view_time"]), np.nanmean(res["lst"]))


if __name__ == "__main__":
    main()

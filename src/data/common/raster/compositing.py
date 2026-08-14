"""
Shared month-first-then-annual temporal compositing helper.

docs/design/04-ingest.md §4: a naive temporal mean over days/periods with
different cloud/missing patterns is biased toward whatever conditions are
over-represented in the missing-data pattern (e.g. dry-season conditions in
seasonally-cloudy regions). The fix is month-first-then-annual compositing:
average to monthly first, then average the monthly means to annual, so each
*month* gets equal weight regardless of how many valid observations it had.

`GlassPreprocessor._calculate_statistics` (src/data/preprocess/sources/
glass.py:739-806) is the prior art for the xarray/Dask resample mechanics
this generalizes -- but its own `annual_stats` is computed directly from
daily data (`rechunked[VAR].resample(time="1YE").mean()`), which is exactly
the naive equal-weighted annual mean this module exists to avoid. Do not
imitate that shortcut; imitate its resample-call shape and its valid-count
pattern (`mask.resample(time="1YE").sum()`), both reproduced here.
"""

import logging
from typing import Tuple

import xarray as xr

logger = logging.getLogger(__name__)


def composite_annual_stats(
    data: xr.DataArray,
    valid_mask: xr.DataArray,
    *,
    stats: Tuple[str, ...] = ("mean",),
    thresholds: "tuple[float, float] | None" = None,
    time_dim: str = "time",
    monthly_freq: str = "1ME",
    annual_freq: str = "1YE",
) -> dict:
    """Annual summary stats from a daily-or-periodic DataArray, month-first.

    Same bias-avoidance rationale as this module always had (see git history
    for the prior `composite_to_annual`, superseded by this function, sole
    caller `src.data.sources.modis.source`): a naive mean/median/etc. over
    raw periods is biased toward whichever season has more valid
    observations (missing-data patterns aren't uniform across the year).
    The fix is the same here -- an intermediate `monthly_mean` (mean of
    valid periods within each month, NaN for a month with none) is what
    every requested annual stat is actually computed from, so each *month*
    gets equal weight regardless of how many raw periods it had. No monthly
    values are returned -- `monthly_mean` is purely an internal weighting
    step, not an output.

    Args:
        data: Values indexed along `time_dim`.
        valid_mask: Boolean/0-1 mask, same shape as `data`.
        stats: Which annual stats to compute -- any of "mean", "median",
            "std", "valid_period_count" (raw valid periods across the year,
            an observation-*density* diagnostic, not a reliability measure
            for the month-weighted stats above), "valid_month_count" (count
            of months that actually contributed -- the correctly-
            denominated reliability diagnostic a thin, few-month composite
            needs `valid_period_count` alone can't show), "count_above"/
            "count_below" (count of months whose monthly mean is above/
            below `thresholds`, required together with `thresholds`).
        thresholds: `(low, high)` for "count_below"/"count_above".

    Returns: dict keyed by the names in `stats`.
    """
    masked = data.where(valid_mask)
    monthly_mean = masked.resample({time_dim: monthly_freq}).mean()
    annual = monthly_mean.resample({time_dim: annual_freq})

    result: dict = {}
    if "mean" in stats:
        result["mean"] = annual.mean()
    if "median" in stats:
        result["median"] = annual.median()
    if "std" in stats:
        result["std"] = annual.std()
    if "valid_period_count" in stats:
        result["valid_period_count"] = valid_mask.resample({time_dim: annual_freq}).sum()
    if "valid_month_count" in stats:
        result["valid_month_count"] = monthly_mean.notnull().resample({time_dim: annual_freq}).sum()
    if "count_above" in stats or "count_below" in stats:
        low, high = thresholds
        if "count_above" in stats:
            result["count_above"] = (monthly_mean > high).resample({time_dim: annual_freq}).sum()
        if "count_below" in stats:
            result["count_below"] = (monthly_mean < low).resample({time_dim: annual_freq}).sum()
    return result

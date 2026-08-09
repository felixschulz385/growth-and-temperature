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


def composite_to_annual(
    data: xr.DataArray,
    valid_mask: xr.DataArray,
    time_dim: str = "time",
    monthly_freq: str = "1ME",
    annual_freq: str = "1YE",
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Composite a daily-or-periodic DataArray to annual, month-first.

    Args:
        data: Values indexed along `time_dim` (daily observations, or
            already-periodic composites such as MODIS's 8-day products --
            see docs/design/07-modis-ingest.md §4 for the composite-of-
            composites case, which changes the weighting inputs, not this
            function's logic).
        valid_mask: Boolean/0-1 mask, same shape as `data`, True/1 where an
            observation is valid and should contribute to the mean.
        time_dim: Name of the time dimension.
        monthly_freq: Resample frequency for the monthly step.
        annual_freq: Resample frequency for the annual step.

    Returns:
        (annual_mean, monthly_mean, monthly_valid_count, annual_valid_count,
        annual_valid_month_count)

        - `monthly_mean`: mean of valid values within each month; NaN for a
          month with zero valid observations (excluded, not zero-filled).
        - `annual_mean`: mean of `monthly_mean` over the year -- since
          `xarray`'s `.mean()` skips NaN by default, months with no valid
          observations are excluded from the denominator automatically,
          implementing "mean over available months, not over 12"
          (docs/design/07-modis-ingest.md §4) without extra code.
        - `monthly_valid_count`: count of valid observations/periods
          contributing to each month, the required diagnostic output per
          docs/design/04-ingest.md §4.
        - `annual_valid_count`: count of valid observations/periods across
          the whole year -- an observation-*density* diagnostic, not a
          reliability measure for `annual_mean`: it does not reflect how
          those observations were distributed across months, since
          `annual_mean` is a mean of monthly means, not of raw observations.
        - `annual_valid_month_count`: count of months that actually
          contributed to `annual_mean`'s own averaging (i.e. months where
          `monthly_mean` isn't NaN) -- the correctly-denominated
          reliability diagnostic. A year with e.g. 300 valid daily
          observations concentrated in only 2 months (10 months entirely
          cloud-covered) has a large `annual_valid_count` (~300) but a
          `annual_valid_month_count` of 2, correctly flagging `annual_mean`
          as a thin, 2-month composite that `annual_valid_count` alone
          would misrepresent as well-supported.
    """
    masked = data.where(valid_mask)

    monthly_mean = masked.resample({time_dim: monthly_freq}).mean()
    monthly_valid_count = valid_mask.resample({time_dim: monthly_freq}).sum()

    annual_mean = monthly_mean.resample({time_dim: annual_freq}).mean()
    annual_valid_count = valid_mask.resample({time_dim: annual_freq}).sum()
    annual_valid_month_count = monthly_mean.notnull().resample({time_dim: annual_freq}).sum()

    return annual_mean, monthly_mean, monthly_valid_count, annual_valid_count, annual_valid_month_count

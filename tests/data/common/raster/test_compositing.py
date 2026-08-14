"""composite_annual_stats()'s month-first-then-annual compositing and its
valid-observation-count diagnostics (docs/design/04-ingest.md §4).
"""

import numpy as np
import pandas as pd
import xarray as xr

from src.data.common.raster.compositing import composite_annual_stats


def _daily_data(values_by_month, fill=10.0):
    """One year of daily data/mask, `values_by_month`: dict of
    month -> (value, n_valid_days) or None for "no valid days"."""
    days = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    data = xr.DataArray(np.full(len(days), fill), dims=("time",), coords={"time": days})
    mask = xr.DataArray(np.zeros(len(days), dtype=bool), dims=("time",), coords={"time": days})

    for month, spec in values_by_month.items():
        month_mask = days.month == month
        if spec is None:
            continue
        value, n_valid = spec
        idx = np.where(month_mask)[0][:n_valid]
        data.values[idx] = value
        mask.values[idx] = True
    return data, mask


def test_valid_month_count_reflects_months_contributing_not_raw_days():
    """The concrete scenario the fix targets: valid daily observations
    concentrated in only 2 fully-observed months (10 months entirely
    cloud-covered). valid_period_count (raw day total) is large;
    valid_month_count (months that actually fed the mean) must correctly
    read 2, not be conflated with the raw day count."""
    values_by_month = {1: (5.0, 31), 2: (5.0, 29)}  # full Jan + Feb 2020 (leap year)
    data, mask = _daily_data(values_by_month)

    result = composite_annual_stats(data, mask, stats=("valid_period_count", "valid_month_count"))

    assert int(result["valid_period_count"].item()) == 60
    assert int(result["valid_month_count"].item()) == 2


def test_mean_and_valid_month_count_match_number_of_contributing_months():
    values_by_month = {1: (5.0, 10), 3: (7.0, 5), 6: (9.0, 20)}
    data, mask = _daily_data(values_by_month)

    result = composite_annual_stats(data, mask, stats=("mean", "valid_month_count"))

    assert int(result["valid_month_count"].item()) == 3
    # Annual mean averages the 3 monthly means, all valid at 5/7/9 -> mean 7.
    assert result["mean"].item() == 7.0


def test_valid_month_count_zero_when_no_valid_observations():
    data, mask = _daily_data({})
    result = composite_annual_stats(data, mask, stats=("valid_period_count", "valid_month_count"))
    assert int(result["valid_period_count"].item()) == 0
    assert int(result["valid_month_count"].item()) == 0


def test_median_and_std_computed_from_monthly_means_not_raw_days():
    # One low month (2 days) and one high month (28 days) -- a naive
    # raw-day median/std would be dominated by the 28-day month; the
    # month-weighted version treats both months equally.
    values_by_month = {1: (0.0, 2), 2: (10.0, 28)}
    data, mask = _daily_data(values_by_month)

    result = composite_annual_stats(data, mask, stats=("median", "std"))

    assert result["median"].item() == 5.0  # median of [0, 10], not raw days
    assert result["std"].item() == 5.0  # std of [0, 10]


def test_count_above_and_below_thresholds_count_months_not_days():
    values_by_month = {1: (300.0, 31), 2: (250.0, 5), 3: (280.0, 10)}
    data, mask = _daily_data(values_by_month)

    result = composite_annual_stats(data, mask, stats=("count_above", "count_below"), thresholds=(273.15, 298.15))

    assert int(result["count_above"].item()) == 1  # January's monthly mean (300) only
    assert int(result["count_below"].item()) == 1  # February's monthly mean (250) only
    # March (280) is between the thresholds -- counted in neither.

"""composite_to_annual()'s month-first-then-annual compositing and its
valid-observation-count diagnostics (docs/design/04-ingest.md §4).
"""

import numpy as np
import pandas as pd
import xarray as xr

from src.data.common.raster.compositing import composite_to_annual


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


def test_annual_valid_month_count_reflects_months_contributing_not_raw_days():
    """The concrete scenario the fix targets: valid daily observations
    concentrated in only 2 fully-observed months (10 months entirely
    cloud-covered). annual_valid_count (raw day total) is large;
    annual_valid_month_count (months that actually fed annual_mean) must
    correctly read 2, not be conflated with the raw day count."""
    values_by_month = {1: (5.0, 31), 2: (5.0, 29)}  # full Jan + Feb 2020 (leap year)
    data, mask = _daily_data(values_by_month)

    _, _, _, annual_valid_count, annual_valid_month_count = composite_to_annual(data, mask)

    assert int(annual_valid_count.item()) == 60
    assert int(annual_valid_month_count.item()) == 2


def test_annual_valid_month_count_matches_number_of_non_nan_monthly_means():
    values_by_month = {1: (5.0, 10), 3: (7.0, 5), 6: (9.0, 20)}
    data, mask = _daily_data(values_by_month)

    annual_mean, monthly_mean, _, _, annual_valid_month_count = composite_to_annual(data, mask)

    assert int(monthly_mean.notnull().sum().item()) == 3
    assert int(annual_valid_month_count.item()) == 3
    # Annual mean averages the 3 monthly means, all valid at 5/7/9 -> mean 7.
    assert annual_mean.item() == 7.0


def test_annual_valid_month_count_zero_when_no_valid_observations():
    data, mask = _daily_data({})
    _, _, _, annual_valid_count, annual_valid_month_count = composite_to_annual(data, mask)
    assert int(annual_valid_count.item()) == 0
    assert int(annual_valid_month_count.item()) == 0

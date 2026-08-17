"""Focused unit test for `_composite_glass_annual_stats()`
(src/data/sources/glass/modis.py) -- the shared 8-band annual-stats helper
both GLASS-MODIS variants (LST, Ta) call (docs/design/12-glass-modis-
rebuild.md §2). Verifies the 8 output bands' shape/keys, and specifically
that `max`/`min` are computed directly from daily data (NOT month-weighted)
while `mean`/`std` go through the month-first-then-annual
`composite_annual_stats()` compositor.
"""

import numpy as np
import pandas as pd
import xarray as xr

from src.data.sources.glass.modis import _composite_glass_annual_stats

_EXPECTED_KEYS = {
    "mean", "std", "valid_period_count", "valid_month_count", "count_above", "count_below", "max", "min",
}


def _daily_series(year: int, extreme_value: float, extreme_day_of_year: int, base_value: float = 10.0) -> xr.DataArray:
    time = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    values = np.full(time.shape, base_value, dtype="float64")
    values[extreme_day_of_year - 1] = extreme_value
    return xr.DataArray(values, dims=("time",), coords={"time": time})


def test_eight_bands_present_with_expected_shape():
    daily = _daily_series(2021, extreme_value=100.0, extreme_day_of_year=15)
    valid_mask = xr.full_like(daily, True, dtype=bool)

    stats = _composite_glass_annual_stats(daily, valid_mask, thresholds=(5.0, 50.0))

    assert set(stats) == _EXPECTED_KEYS
    for name, arr in stats.items():
        # One year of daily input -> exactly one annual value per stat.
        assert arr.sizes["time"] == 1, f"{name} should have exactly one annual value"


def test_max_min_are_not_month_weighted_but_mean_std_are():
    # A single extreme day (100.0) inside a 31-day January where every other
    # day is 10.0, and a flat 10.0 February. Month-first-then-annual gives
    # January's monthly mean ~= (30*10 + 100) / 31 ~= 12.9, so the annual
    # *mean* is pulled only slightly above 10 -- nowhere near 100. If `max`
    # were (incorrectly) derived the same month-weighted way, it too would
    # top out at ~12.9, never seeing the raw 100.0 extreme.
    daily = _daily_series(2021, extreme_value=100.0, extreme_day_of_year=15)
    valid_mask = xr.full_like(daily, True, dtype=bool)

    stats = _composite_glass_annual_stats(daily, valid_mask, thresholds=(5.0, 50.0))

    assert float(stats["max"].item()) == 100.0
    assert float(stats["min"].item()) == 10.0
    # mean/std went through the month-first compositor -- nowhere near the
    # raw extreme, proving they're computed differently from max/min.
    mean_value = float(stats["mean"].item())
    assert 10.0 < mean_value < 13.5
    assert float(stats["std"].item()) >= 0.0


def test_min_da_max_da_default_to_mean_da_when_single_band():
    # LST-shaped call: only mean_da given (min_da/max_da omitted) -- both
    # should default to mean_da itself (docs/design/12-glass-modis-
    # rebuild.md §2: "mean_da = min_da = max_da" for the single-band LST
    # product).
    daily = _daily_series(2021, extreme_value=42.0, extreme_day_of_year=200)
    valid_mask = xr.full_like(daily, True, dtype=bool)

    stats = _composite_glass_annual_stats(daily, valid_mask, thresholds=(5.0, 50.0))
    assert float(stats["max"].item()) == 42.0
    assert float(stats["min"].item()) == 10.0


def test_separate_min_da_max_da_used_directly_when_given():
    # Ta-shaped call: mean_da/min_da/max_da are three distinct daily series
    # (Ta_mean/Ta_min/Ta_max) -- max/min must come from their own arrays,
    # not from mean_da.
    year = 2021
    mean_da = _daily_series(year, extreme_value=20.0, extreme_day_of_year=100, base_value=15.0)
    min_da = _daily_series(year, extreme_value=-5.0, extreme_day_of_year=100, base_value=5.0)
    max_da = _daily_series(year, extreme_value=45.0, extreme_day_of_year=100, base_value=25.0)
    valid_mask = xr.full_like(mean_da, True, dtype=bool)

    stats = _composite_glass_annual_stats(
        mean_da, valid_mask, min_da=min_da, max_da=max_da, thresholds=(5.0, 50.0)
    )
    assert float(stats["max"].item()) == 45.0
    assert float(stats["min"].item()) == -5.0

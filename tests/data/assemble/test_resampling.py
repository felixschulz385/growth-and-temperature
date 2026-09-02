"""Per-variable resampling: `resolve_resampling` (name -> method) and
`_reproject_per_variable` (one store, method-grouped downsampling pass).
"""

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from src.data.assemble.constants import DEFAULT_CRS, DEFAULT_RESAMPLING_METHOD
from src.data.assemble.processors import _reproject_per_variable
from src.data.assemble.utils import resolve_resampling


# --- resolve_resampling ------------------------------------------------------


def test_none_config_is_default_for_every_var():
    assert resolve_resampling(None, ["a", "b"]) == {
        "a": DEFAULT_RESAMPLING_METHOD,
        "b": DEFAULT_RESAMPLING_METHOD,
    }


def test_string_config_applies_to_every_var():
    assert resolve_resampling("average", ["a", "b"]) == {"a": "average", "b": "average"}


def test_map_config_default_plus_glob_first_match_wins():
    cfg = {
        "default": "average",
        "valid_*count*": "sum",
        "*_sd": "nearest",
    }
    got = resolve_resampling(
        cfg,
        ["lst_day_mean", "lst_day_sd", "valid_period_count_annual", "valid_month_count_day_annual"],
    )
    assert got == {
        "lst_day_mean": "average",
        "lst_day_sd": "nearest",
        "valid_period_count_annual": "sum",
        "valid_month_count_day_annual": "sum",
    }


def test_map_config_without_default_falls_back_to_module_default():
    got = resolve_resampling({"*_count": "sum"}, ["x_count", "y_mean"])
    assert got == {"x_count": "sum", "y_mean": DEFAULT_RESAMPLING_METHOD}


def test_unknown_method_raises_even_with_no_var_names():
    with pytest.raises(ValueError, match="Unknown resampling method 'avg'"):
        resolve_resampling({"default": "avg"}, [])
    with pytest.raises(ValueError, match="Unknown resampling method 'summe'"):
        resolve_resampling({"default": "average", "*_count": "summe"}, [])
    with pytest.raises(ValueError, match="Unknown resampling method"):
        resolve_resampling("bogus", ["a"])


def test_non_str_non_mapping_config_raises():
    with pytest.raises(ValueError, match="must be a method string or a mapping"):
        resolve_resampling(["average"], ["a"])


# --- _reproject_per_variable -----------------------------------------------


def _ds_of_ones(geobox, var_names):
    h, w = geobox.shape
    data = np.ones((h, w), dtype="float32")
    ds = xr.Dataset(
        {v: (("latitude", "longitude"), data.copy()) for v in var_names},
        coords={
            "latitude": geobox.coords["latitude"].values,
            "longitude": geobox.coords["longitude"].values,
        },
    )
    return ds.odc.assign_crs(DEFAULT_CRS)


def test_single_method_still_reprojects_all_vars_together():
    src = GeoBox.from_bbox((0, 0, 0.16, 0.16), crs="EPSG:4326", resolution=0.01)  # 16x16
    tgt = GeoBox.from_bbox((0, 0, 0.16, 0.16), crs="EPSG:4326", resolution=0.04)  # 4x4
    out = _reproject_per_variable(_ds_of_ones(src, ["mean", "other"]), tgt, "average")
    assert out.sizes["latitude"] == 4 and out.sizes["longitude"] == 4
    assert np.allclose(out["mean"].values, 1.0)
    assert np.allclose(out["other"].values, 1.0)


def test_map_splits_average_and_sum_in_one_pass():
    src = GeoBox.from_bbox((0, 0, 0.16, 0.16), crs="EPSG:4326", resolution=0.01)  # 16x16
    tgt = GeoBox.from_bbox((0, 0, 0.16, 0.16), crs="EPSG:4326", resolution=0.04)  # 4x4 (16:1)

    cfg = {"default": "average", "*_count": "sum"}
    out = _reproject_per_variable(_ds_of_ones(src, ["lst_mean", "obs_count"]), tgt, cfg)

    # every 4km cell aggregates a 4x4 block of 1s
    assert np.allclose(out["lst_mean"].values, 1.0)      # area-weighted mean
    assert np.allclose(out["obs_count"].values, 16.0)    # summed
    assert set(out.data_vars) == {"lst_mean", "obs_count"}

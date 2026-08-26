"""sel_bbox() must slice correctly regardless of which direction each
coordinate runs -- see its docstring (src/data/common/raster/spatial.py)."""

from types import SimpleNamespace

import numpy as np
import xarray as xr

from src.data.common.raster.spatial import sel_bbox


def _bbox(left, bottom, right, top):
    return SimpleNamespace(left=left, bottom=bottom, right=right, top=top)


def test_sel_bbox_descending_y_ascending_x():
    y = np.array([10, 8, 6, 4, 2, 0])
    x = np.array([0, 2, 4, 6, 8, 10])
    ds = xr.Dataset({"v": (("y", "x"), np.arange(36).reshape(6, 6))}, coords={"y": y, "x": x})

    clipped = sel_bbox(ds, _bbox(left=3, bottom=1, right=7, top=9), y_dim="y", x_dim="x")

    assert list(clipped["y"].values) == [8, 6, 4, 2]
    assert list(clipped["x"].values) == [4, 6]


def test_sel_bbox_ascending_y_descending_x():
    y = np.array([0, 2, 4, 6, 8, 10])
    x = np.array([10, 8, 6, 4, 2, 0])
    ds = xr.Dataset({"v": (("y", "x"), np.arange(36).reshape(6, 6))}, coords={"y": y, "x": x})

    clipped = sel_bbox(ds, _bbox(left=3, bottom=1, right=7, top=9), y_dim="y", x_dim="x")

    assert list(clipped["y"].values) == [2, 4, 6, 8]
    assert list(clipped["x"].values) == [6, 4]


def test_sel_bbox_respects_custom_dim_names():
    lat = np.array([10, 5, 0])
    lon = np.array([0, 5, 10])
    ds = xr.Dataset(
        {"v": (("latitude", "longitude"), np.arange(9).reshape(3, 3))},
        coords={"latitude": lat, "longitude": lon},
    )

    clipped = sel_bbox(ds, _bbox(left=1, bottom=1, right=9, top=9), y_dim="latitude", x_dim="longitude")

    assert list(clipped["latitude"].values) == [5]
    assert list(clipped["longitude"].values) == [5]

"""`utils` spatial-dim / pixel-id helpers are CRS-agnostic: they key off the
geobox's own dim names, so they work on the canonical EASE 6933 grid (`y`/`x`)
as well as the legacy EPSG:4326 grid (`latitude`/`longitude`). Still used by
`src/data/assemble/ring_means.py`.
"""

import numpy as np
import xarray as xr
from odc.geo.geobox import GeoBox

from src.data.assemble.utils import (
    dataset_spatial_dims,
    decode_pixel_id,
    geobox_spatial_dims,
    make_pixel_ids,
)

EASE = "EPSG:6933"


def _ease_geobox(n=16, res=1000.0):
    return GeoBox.from_bbox((0.0, 0.0, n * res, n * res), crs=EASE, resolution=res)


def test_geobox_spatial_dims_by_crs():
    assert geobox_spatial_dims(GeoBox.from_bbox((0, 0, 1, 1), crs="EPSG:4326", resolution=0.1)) == (
        "latitude",
        "longitude",
    )
    assert geobox_spatial_dims(_ease_geobox()) == ("y", "x")


def test_dataset_spatial_dims_recognizes_both_and_none():
    yx = xr.Dataset({"v": (("y", "x"), np.zeros((2, 2)))}, coords={"y": [1, 0], "x": [0, 1]})
    ll = xr.Dataset(
        {"v": (("latitude", "longitude"), np.zeros((2, 2)))},
        coords={"latitude": [1, 0], "longitude": [0, 1]},
    )
    other = xr.Dataset({"v": (("a", "b"), np.zeros((2, 2)))})
    assert dataset_spatial_dims(yx) == ("y", "x")
    assert dataset_spatial_dims(ll) == ("latitude", "longitude")
    assert dataset_spatial_dims(other) is None


def test_make_pixel_ids_on_ease_geobox_uses_y_x_and_ease_crs():
    gb = _ease_geobox(n=8)
    ds = make_pixel_ids(3, 5, gb)

    assert set(ds["pixel_id"].dims) == {"y", "x"}
    assert ds.odc.crs == EASE
    assert ds["pixel_id"].shape == (8, 8)
    ix, iy, local = decode_pixel_id(np.uint64(ds["pixel_id"].values[0, 0]))
    assert (ix, iy, local) == (3, 5, 0)

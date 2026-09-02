"""The assemble tile engine keys spatial dim names off the run's geobox, not a
hardcoded latitude/longitude -- so it works on the canonical EASE 6933 grid
(dims `y`/`x`) as well as the legacy EPSG:4326 grid (`latitude`/`longitude`).
"""

import numpy as np
import xarray as xr
from odc.geo.geobox import GeoBox

from src.data.assemble.processors import TileProcessor
from src.data.assemble.utils import (
    dataset_spatial_dims,
    decode_pixel_id,
    geobox_spatial_dims,
    make_pixel_ids,
)

EASE = "EPSG:6933"


def _ease_geobox(n=16, res=1000.0):
    return GeoBox.from_bbox((0.0, 0.0, n * res, n * res), crs=EASE, resolution=res)


# --- dim-name helpers -------------------------------------------------------


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


# --- make_pixel_ids on a projected grid -----------------------------------


def test_make_pixel_ids_on_ease_geobox_uses_y_x_and_ease_crs():
    gb = _ease_geobox(n=8)
    ds = make_pixel_ids(3, 5, gb)

    assert set(ds["pixel_id"].dims) == {"y", "x"}
    assert ds.odc.crs == EASE
    assert ds["pixel_id"].shape == (8, 8)
    # round-trips through the packed encoding
    ix, iy, local = decode_pixel_id(np.uint64(ds["pixel_id"].values[0, 0]))
    assert (ix, iy, local) == (3, 5, 0)


# --- _extract_dataset_tile end-to-end on EASE ----------------------------------


def _processor(target_geobox, resolution):
    cfg = {"output_path": "unused", "datasets": {}, "processing": {"resolution": resolution}}
    return TileProcessor(cfg, "unused", target_geobox=target_geobox)


def test_extract_dataset_tile_runs_on_ease_grid_and_drops_y_x_columns():
    native = _ease_geobox(n=16, res=1000.0)
    proc = _processor(native, resolution=4000.0)  # 16 km tile -> 4 km cells

    padded, zoomed = proc._create_tile_geoboxes(native)
    pixel_id_ds = make_pixel_ids(0, 0, zoomed)

    h, w = padded.shape
    src = xr.Dataset(
        {"lst_mean": (("y", "x"), np.ones((h, w), dtype="float32"))},
        coords={
            "y": padded.coords["y"].values,
            "x": padded.coords["x"].values,
        },
    ).odc.assign_crs(EASE)

    df = proc._extract_dataset_tile(
        src, {"resampling": "average"}, ix=0, iy=0,
        padded_tile_geobox=padded, target_geobox_zoomed=zoomed, pixel_id_ds=pixel_id_ds,
    )

    assert df is not None and not df.empty
    assert "pixel_id" in df.columns
    assert "lst_mean" in df.columns
    assert not ({"y", "x", "latitude", "longitude"} & set(df.columns))
    assert np.allclose(df["lst_mean"].to_numpy(), 1.0)

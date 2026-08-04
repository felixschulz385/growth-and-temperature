"""Tests for the grid-shake / block-reduce downsampling robustness check."""

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from src.data.assemble.constants import DEFAULT_CRS
from src.data.assemble.config import validate_assembly_config
from src.data.assemble.grid_shake import (
    DEFAULT_GRID_SHAKE_PRESETS,
    normalize_grid_shake_offsets,
    shift_geobox_origin,
)
from src.data.assemble.processors import TileProcessor


def _make_geobox(shape=16, resolution=0.01):
    return GeoBox.from_bbox((0, 0, shape * resolution, shape * resolution), crs="EPSG:4326", resolution=resolution)


# --- shift_geobox_origin ---------------------------------------------------


def test_shift_geobox_origin_preserves_shape_resolution_crs():
    geobox = _make_geobox()
    shifted = shift_geobox_origin(geobox, 0.5, 0.25)

    assert shifted.shape == geobox.shape
    assert shifted.resolution == geobox.resolution
    assert shifted.crs == geobox.crs


def test_shift_geobox_origin_translates_by_pixel_fraction():
    geobox = _make_geobox(resolution=0.01)
    shifted = shift_geobox_origin(geobox, 0.5, 0.25)

    # affine.c/.f are the origin (x, y) offsets
    assert shifted.affine.c == pytest.approx(geobox.affine.c + 0.5 * geobox.resolution.x)
    assert shifted.affine.f == pytest.approx(geobox.affine.f + 0.25 * geobox.resolution.y)


def test_shift_geobox_origin_zero_offset_is_identity():
    geobox = _make_geobox()
    shifted = shift_geobox_origin(geobox, 0.0, 0.0)
    assert shifted.affine == geobox.affine


# --- normalize_grid_shake_offsets ------------------------------------------


def test_normalize_grid_shake_offsets_none_or_empty():
    assert normalize_grid_shake_offsets(None) == []
    assert normalize_grid_shake_offsets({}) == []
    assert normalize_grid_shake_offsets([]) == []


def test_normalize_grid_shake_offsets_preset():
    specs = normalize_grid_shake_offsets("quad")
    expected = DEFAULT_GRID_SHAKE_PRESETS["quad"]
    assert [(dx, dy) for _, dx, dy in specs] == expected
    assert [label for label, _, _ in specs] == ["0", "1", "2"]


def test_normalize_grid_shake_offsets_dict_form():
    specs = normalize_grid_shake_offsets({"offsets": [(0.5, 0.0)]})
    assert specs == [("0", 0.5, 0.0)]


def test_normalize_grid_shake_offsets_raw_list_form():
    specs = normalize_grid_shake_offsets([(0.25, 0.25), (0.75, 0.75)])
    assert specs == [("0", 0.25, 0.25), ("1", 0.75, 0.75)]


def test_normalize_grid_shake_offsets_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown grid_shake preset"):
        normalize_grid_shake_offsets("not-a-real-preset")


def test_normalize_grid_shake_offsets_out_of_range_raises():
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        normalize_grid_shake_offsets([(1.0, 0.0)])
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        normalize_grid_shake_offsets([(-0.1, 0.0)])


def test_normalize_grid_shake_offsets_dict_without_offsets_raises():
    with pytest.raises(ValueError, match="offsets"):
        normalize_grid_shake_offsets({"foo": "bar"})


# --- config validation -------------------------------------------------


def _base_assembly_config(**processing_overrides):
    return {
        "output_path": "/tmp/does-not-matter",
        "datasets": {
            "src": {"path": "/tmp/does-not-exist.zarr"},
        },
        "processing": {"resolution": 0.04, **processing_overrides},
    }


def test_validate_assembly_config_accepts_valid_grid_shake():
    errors = validate_assembly_config(_base_assembly_config(grid_shake="quad"))
    assert not any("grid_shake" in e for e in errors)


def test_validate_assembly_config_rejects_bad_grid_shake():
    errors = validate_assembly_config(_base_assembly_config(grid_shake="nonsense"))
    assert any("grid_shake" in e for e in errors)


def test_validate_assembly_config_rejects_grid_shake_with_geometry_partition():
    config = _base_assembly_config(grid_shake="quad", spatial_partition="geometry")
    config["geometry_source"] = {"path": "/tmp/x.gpkg", "id_column": "id"}
    config["geometry_aggregator"] = "pkg.module:function"
    errors = validate_assembly_config(config)
    assert any("spatial_partition='geometry'" in e for e in errors)


# --- TileProcessor integration ----------------------------------------------


def _make_processor(target_geobox, resolution, grid_shake="quad"):
    assembly_config = {
        "output_path": "unused",
        "datasets": {},
        "processing": {"resolution": resolution, "grid_shake": grid_shake},
    }
    return TileProcessor(assembly_config, "unused", target_geobox=target_geobox)


def _make_padded_source_dataset(padded_geobox):
    """A spatially-varying field covering the padded tile extent, so downsampled
    output cells have real (non-NaN) values regardless of grid-shake origin shift."""
    h, w = padded_geobox.shape
    rows = np.arange(h).reshape(h, 1)
    cols = np.arange(w).reshape(1, w)
    values = (rows * 7 + cols * 3).astype("float32")
    lat = padded_geobox.coords["latitude"].values
    lon = padded_geobox.coords["longitude"].values
    da = xr.DataArray(values, dims=("latitude", "longitude"), coords={"latitude": lat, "longitude": lon}, name="test_var")
    ds = xr.Dataset({"test_var": da})
    return ds.odc.assign_crs(DEFAULT_CRS)


def test_grid_shake_active_only_when_downsampling():
    native_res = 0.01
    tile_geobox = _make_geobox(shape=16, resolution=native_res)

    coarser = _make_processor(tile_geobox, resolution=0.04)
    assert coarser.grid_shake_active is True

    same_res = _make_processor(tile_geobox, resolution=native_res)
    assert same_res.grid_shake_active is False

    finer = _make_processor(tile_geobox, resolution=0.005)
    assert finer.grid_shake_active is False

    no_config = _make_processor(tile_geobox, resolution=0.04, grid_shake=None)
    assert no_config.grid_shake_active is False


def test_extract_dataset_tile_produces_shake_columns():
    native_res = 0.01
    target_res = 0.04
    tile_geobox = _make_geobox(shape=16, resolution=native_res)

    processor = _make_processor(tile_geobox, resolution=target_res)
    padded_tile_geobox, target_geobox_zoomed = processor._create_tile_geoboxes(tile_geobox)
    shaken_geoboxes = processor._get_shaken_geoboxes(target_geobox_zoomed)
    assert [label for label, _ in shaken_geoboxes] == ["0", "1", "2"]

    ds = _make_padded_source_dataset(padded_tile_geobox)

    df = processor._extract_dataset_tile(
        ds,
        {"resampling": "average"},
        ix=0,
        iy=0,
        padded_tile_geobox=padded_tile_geobox,
        target_geobox_zoomed=target_geobox_zoomed,
        pixel_id_ds=None,
        include_pixel_id=False,
        shaken_geoboxes=shaken_geoboxes,
    )

    assert df is not None
    for label in ("0", "1", "2"):
        assert f"test_var__shake_{label}" in df.columns

    base_rows = len(df)
    for label in ("0", "1", "2"):
        assert len(df[f"test_var__shake_{label}"].dropna()) > 0
        assert len(df) == base_rows

    # At least one shake variant must differ from the unshaken column somewhere --
    # otherwise the origin shift silently had no effect.
    differs = False
    for label in ("0", "1", "2"):
        shake_col = df[f"test_var__shake_{label}"]
        if not np.allclose(shake_col.values, df["test_var"].values, equal_nan=True):
            differs = True
    assert differs


def test_extract_dataset_tile_no_shake_columns_when_inactive():
    native_res = 0.01
    tile_geobox = _make_geobox(shape=16, resolution=native_res)

    processor = _make_processor(tile_geobox, resolution=native_res)  # not downsampling
    padded_tile_geobox, target_geobox_zoomed = processor._create_tile_geoboxes(tile_geobox)
    shaken_geoboxes = processor._get_shaken_geoboxes(target_geobox_zoomed)
    assert shaken_geoboxes == []

    ds = _make_padded_source_dataset(padded_tile_geobox)
    df = processor._extract_dataset_tile(
        ds,
        {"resampling": "average"},
        ix=0,
        iy=0,
        padded_tile_geobox=padded_tile_geobox,
        target_geobox_zoomed=target_geobox_zoomed,
        pixel_id_ds=None,
        include_pixel_id=False,
        shaken_geoboxes=shaken_geoboxes,
    )

    assert df is not None
    assert not any(col.startswith("test_var__shake_") for col in df.columns)

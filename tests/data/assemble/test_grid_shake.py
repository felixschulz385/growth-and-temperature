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
    resolve_shake_selection,
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


# --- resolve_shake_selection ---------------------------------------------------


def test_resolve_shake_selection_none_is_base_only():
    assert resolve_shake_selection(None) == [("base", 0.0, 0.0)]
    assert resolve_shake_selection("none") == [("base", 0.0, 0.0)]


def test_resolve_shake_selection_preset_is_base_plus_offsets():
    sel = resolve_shake_selection("quad")
    assert [label for label, _, _ in sel] == ["base", "s0", "s1", "s2"]
    assert [(dx, dy) for _, dx, dy in sel[1:]] == DEFAULT_GRID_SHAKE_PRESETS["quad"]


def test_resolve_shake_selection_single_offset_label_is_that_partition_only():
    assert resolve_shake_selection("s1") == [("s1", 0.0, 0.5)]


def test_resolve_shake_selection_unknown_raises():
    with pytest.raises(ValueError, match="Unknown --shake value"):
        resolve_shake_selection("wobble")
    with pytest.raises(ValueError, match="out of range"):
        resolve_shake_selection("s9")


# --- TileProcessor integration ----------------------------------------------


def _make_processor(target_geobox, resolution):
    assembly_config = {
        "output_path": "unused",
        "datasets": {},
        "processing": {"resolution": resolution},
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


def test_extract_dataset_tile_has_no_shake_columns():
    """Grid-shake is now a whole-run origin shift, not a per-column operation --
    a single pass produces exactly the source's own columns."""
    native_res = 0.01
    tile_geobox = _make_geobox(shape=16, resolution=native_res)

    processor = _make_processor(tile_geobox, resolution=0.04)
    padded_tile_geobox, target_geobox_zoomed = processor._create_tile_geoboxes(tile_geobox)
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
    )

    assert df is not None
    assert "test_var" in df.columns
    assert not any("__shake_" in col for col in df.columns)


def test_origin_shift_changes_downsampled_values():
    """A shifted target grid must actually produce different coarse-cell values --
    this is what a shake=s* sibling table is."""
    native_res = 0.01
    target_res = 0.04
    tile_geobox = _make_geobox(shape=16, resolution=native_res)

    processor = _make_processor(tile_geobox, resolution=target_res)
    padded_tile_geobox, target_geobox_zoomed = processor._create_tile_geoboxes(tile_geobox)
    ds = _make_padded_source_dataset(padded_tile_geobox)

    common = dict(
        dataset_config={"resampling": "average"},
        ix=0,
        iy=0,
        padded_tile_geobox=padded_tile_geobox,
        pixel_id_ds=None,
        include_pixel_id=False,
    )
    base_df = processor._extract_dataset_tile(ds=ds, target_geobox_zoomed=target_geobox_zoomed, **common)
    shaken_df = processor._extract_dataset_tile(
        ds=ds,
        target_geobox_zoomed=shift_geobox_origin(target_geobox_zoomed, 0.5, 0.5),
        **common,
    )

    assert base_df is not None and shaken_df is not None
    assert len(base_df) == len(shaken_df)  # same schema / row count
    assert not np.allclose(
        base_df["test_var"].values, shaken_df["test_var"].values, equal_nan=True
    )

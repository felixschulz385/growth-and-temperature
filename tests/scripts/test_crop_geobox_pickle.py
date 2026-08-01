"""Tests for scripts/crop_geobox_pickle.py."""

import pickle
import subprocess
import sys
from pathlib import Path

import pytest
from odc.geo.geobox import GeoBox

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from crop_geobox_pickle import crop_geobox  # noqa: E402


def _full_geobox(shape=(33601, 86401)):
    ny, nx = shape
    return GeoBox.from_bbox((-180, -90, 180, 90), crs="EPSG:4326", shape=(ny, nx))


def test_crop_geobox_default_centers_on_pixel_midpoint():
    full = _full_geobox()
    ny, nx = full.shape
    window = crop_geobox(full, 300)
    assert window.shape.x == 300
    assert window.shape.y == 300
    assert window.crs == full.crs
    assert window.resolution == full.resolution


def test_crop_geobox_preserves_crs_and_resolution():
    full = _full_geobox()
    window = crop_geobox(full, 200)
    assert window.crs == full.crs
    assert window.resolution.x == pytest.approx(full.resolution.x)
    assert window.resolution.y == pytest.approx(full.resolution.y)


def test_crop_geobox_rejects_oversized_window():
    full = _full_geobox(shape=(100, 100))
    with pytest.raises(ValueError, match="exceeds full geobox shape"):
        crop_geobox(full, 200)


def test_crop_geobox_with_lon_lat_targets_correct_region():
    full = _full_geobox()
    # (lon=10, lat=50) is central Europe -- window should be centered there,
    # not at the grid's pixel midpoint (lon=0, lat=0).
    window = crop_geobox(full, 100, lon=10.0, lat=50.0)
    center_lon, center_lat = window.affine * (window.shape.x / 2, window.shape.y / 2)
    assert center_lon == pytest.approx(10.0, abs=0.1)
    assert center_lat == pytest.approx(50.0, abs=0.1)


def test_crop_geobox_lon_lat_out_of_bounds_raises():
    full = _full_geobox()
    with pytest.raises(ValueError, match="outside the geobox"):
        crop_geobox(full, 100, lon=500.0, lat=500.0)


def test_crop_geobox_clamps_at_grid_edge_without_shrinking():
    full = _full_geobox()
    # (lon=-180, lat=90) is the top-left corner -- a naive centered window
    # would go out of bounds; clamping should shift, not shrink.
    window = crop_geobox(full, 100, lon=-179.99, lat=89.99)
    assert window.shape.x == 100
    assert window.shape.y == 100


def test_cli_round_trip(tmp_path):
    full = _full_geobox()
    input_pkl = tmp_path / "full.pkl"
    output_pkl = tmp_path / "cropped.pkl"
    with open(input_pkl, "wb") as f:
        pickle.dump(full, f)

    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "crop_geobox_pickle.py"), str(input_pkl), str(output_pkl), "--window-px", "150"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "cropped target geobox" in result.stdout

    with open(output_pkl, "rb") as f:
        window = pickle.load(f)
    assert window.shape.x == 150
    assert window.shape.y == 150


def test_cli_requires_both_lon_and_lat(tmp_path):
    full = _full_geobox()
    input_pkl = tmp_path / "full.pkl"
    with open(input_pkl, "wb") as f:
        pickle.dump(full, f)

    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "crop_geobox_pickle.py"), str(input_pkl), str(tmp_path / "out.pkl"), "--lon", "10.0"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--lon and --lat must be given together" in result.stderr

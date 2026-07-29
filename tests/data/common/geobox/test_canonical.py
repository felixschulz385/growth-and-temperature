"""Tests for canonical EPSG:6933 GeoBox construction.

Cross-checks against the specific numbers worked out in docs/design/01-grid.md
so a silent regression in the extent/tiling arithmetic would be caught here,
not just discovered downstream in the convolution engine or storage sizing.
"""

import pytest
from odc.geo.geobox import GeoBox, GeoboxTiles

from src.data.assemble.constants import DEFAULT_TILE_SIZE
from src.data.common.geobox.canonical import (
    canonical_ease_geobox,
    compute_ease_bbox,
    get_or_create_canonical_geobox,
)


def test_compute_ease_bbox_matches_design_doc_x_extent():
    # docs/design/01-grid.md §2: x in [-17,367,530.45, +17,367,530.45] m
    left, bottom, right, top = compute_ease_bbox(lat_clip_deg=60.0)
    assert left == pytest.approx(-17_367_530.45, abs=1.0)
    assert right == pytest.approx(17_367_530.45, abs=1.0)


def test_compute_ease_bbox_matches_design_doc_y_extent_at_60deg():
    # docs/design/01-grid.md §2: y in [-6,351,420.00, +6,351,420.00] m at phi=60
    left, bottom, right, top = compute_ease_bbox(lat_clip_deg=60.0)
    assert bottom == pytest.approx(-6_351_420.00, abs=1.0)
    assert top == pytest.approx(6_351_420.00, abs=1.0)


def test_compute_ease_bbox_symmetric_about_origin():
    left, bottom, right, top = compute_ease_bbox(lat_clip_deg=60.0)
    assert left == pytest.approx(-right)
    assert bottom == pytest.approx(-top)


def test_canonical_ease_geobox_crs_and_resolution():
    gbox = canonical_ease_geobox(resolution_m=1000.0, lat_clip_deg=60.0)
    assert str(gbox.crs).upper() == "EPSG:6933"
    assert gbox.resolution.x == pytest.approx(1000.0)
    assert abs(gbox.resolution.y) == pytest.approx(1000.0)


def test_canonical_ease_geobox_shape_matches_design_doc():
    # docs/design/01-grid.md §2: ~34,735 x 12,703 px (~441.2M px total);
    # GeoBox.from_bbox snaps outward to the pixel grid so the exact figure can
    # be a pixel or two larger than the doc's ceil() estimate.
    gbox = canonical_ease_geobox(resolution_m=1000.0, lat_clip_deg=60.0)
    assert gbox.shape.x == pytest.approx(34_735, abs=5)
    assert gbox.shape.y == pytest.approx(12_703, abs=5)
    total_px = gbox.shape.x * gbox.shape.y
    assert total_px == pytest.approx(441.2e6, rel=0.01)


def test_canonical_ease_geobox_tiles_into_119_tiles_at_default_tile_size():
    # docs/design/01-grid.md §5: 17 x 7 = 119 tiles at DEFAULT_TILE_SIZE=2048,
    # both far under pixel_id's 16-bit-per-axis budget.
    gbox = canonical_ease_geobox(resolution_m=1000.0, lat_clip_deg=60.0)
    tiles = GeoboxTiles(gbox, (DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE))
    assert tiles.shape.x == 17
    assert tiles.shape.y == 7
    assert tiles.shape.x * tiles.shape.y == 119


def test_canonical_ease_geobox_finer_resolution_shrinks_pixel_size():
    coarse = canonical_ease_geobox(resolution_m=2000.0, lat_clip_deg=60.0)
    fine = canonical_ease_geobox(resolution_m=1000.0, lat_clip_deg=60.0)
    assert fine.shape.x > coarse.shape.x
    assert fine.shape.y > coarse.shape.y


def test_canonical_ease_geobox_narrower_clip_shrinks_height_not_width():
    wide = canonical_ease_geobox(resolution_m=1000.0, lat_clip_deg=60.0)
    narrow = canonical_ease_geobox(resolution_m=1000.0, lat_clip_deg=30.0)
    assert narrow.shape.y < wide.shape.y
    assert narrow.shape.x == wide.shape.x  # x is latitude-independent under this projection


def test_get_or_create_canonical_geobox_caches_to_disk(tmp_path):
    cache_path = tmp_path / "ease6933_geobox.pkl"
    assert not cache_path.exists()

    built = get_or_create_canonical_geobox(cache_path, resolution_m=5000.0, lat_clip_deg=60.0)
    assert cache_path.exists()

    loaded = get_or_create_canonical_geobox(cache_path, resolution_m=1000.0, lat_clip_deg=30.0)
    # second call loads the cached geobox rather than rebuilding with the new
    # (intentionally different) args, proving the cache short-circuits
    assert loaded.shape == built.shape
    assert loaded.resolution.x == built.resolution.x


def test_get_or_create_canonical_geobox_force_regenerate_rebuilds(tmp_path):
    cache_path = tmp_path / "ease6933_geobox.pkl"
    first = get_or_create_canonical_geobox(cache_path, resolution_m=5000.0, lat_clip_deg=60.0)
    second = get_or_create_canonical_geobox(
        cache_path, resolution_m=1000.0, lat_clip_deg=60.0, force_regenerate=True
    )
    assert second.shape.x > first.shape.x


def test_get_or_create_canonical_geobox_rejects_non_geobox_pickle(tmp_path):
    cache_path = tmp_path / "not_a_geobox.pkl"
    import pickle

    with open(cache_path, "wb") as f:
        pickle.dump({"not": "a geobox"}, f)

    with pytest.raises(TypeError):
        get_or_create_canonical_geobox(cache_path)

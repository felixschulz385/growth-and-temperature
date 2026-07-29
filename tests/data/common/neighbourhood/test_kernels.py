"""Tests for the latitude-band elliptical kernel registry.

Includes the numeric verification docs/design/01-grid.md §6 and
docs/design/06-open-questions.md item 6 call for: rasterizing a known-radius
geographic circle at several latitudes and confirming the kernel's pixel-space
ellipse actually matches ground-truth geodesic distance, independent of the
closed-form derivation in kernels.anisotropy_scales.
"""

import numpy as np
import pytest
from pyproj import Geod, Transformer

from src.data.common.neighbourhood.kernels import (
    EllipticalKernelRegistry,
    anisotropy_scales,
    compute_band_edges,
    disc_kernel_pixels,
)

STANDARD_PARALLEL_DEG = 30.0


def test_anisotropy_scales_identity_at_standard_parallel():
    scale_ew, scale_ns = anisotropy_scales(STANDARD_PARALLEL_DEG, STANDARD_PARALLEL_DEG)
    assert scale_ew == pytest.approx(1.0)
    assert scale_ns == pytest.approx(1.0)


def test_anisotropy_scales_product_is_one_equal_area():
    for lat in [-59, -30, -10, 0, 10, 30, 59]:
        scale_ew, scale_ns = anisotropy_scales(lat, STANDARD_PARALLEL_DEG)
        assert scale_ew * scale_ns == pytest.approx(1.0, rel=1e-9)


def test_anisotropy_scales_documented_ratio_bounds():
    # docs/design/01-grid.md §1: ratio cos^2(phi_s)/cos^2(phi) is 0.75 at the
    # equator, 1.0 at 30 deg, 3.0 at 60 deg. ratio == (scale_ew/scale_ns).
    for lat, expected_ratio in [(0.0, 0.75), (30.0, 1.0), (60.0, 3.0)]:
        scale_ew, scale_ns = anisotropy_scales(lat, STANDARD_PARALLEL_DEG)
        ratio = scale_ew / scale_ns
        assert ratio == pytest.approx(expected_ratio, rel=1e-6)


def test_compute_band_edges_covers_full_range_monotonically():
    edges = compute_band_edges(lat_max_deg=60.0, tolerance=0.02, standard_parallel_deg=30.0)
    assert edges[0] == 0.0
    assert edges[-1] == pytest.approx(60.0)
    assert np.all(np.diff(edges) > 0)


def test_compute_band_edges_narrower_near_clip_than_near_equator():
    edges = compute_band_edges(lat_max_deg=60.0, tolerance=0.02, standard_parallel_deg=30.0)
    widths = np.diff(edges)
    # first band (touches the equator) should be wider than the last band
    # (touches the 60 deg clip edge) -- illustrative widths in the design doc
    # are ~1 deg near 30 deg and ~0.33 deg near 60 deg.
    assert widths[0] > widths[-1]


def test_disc_kernel_pixels_isotropic_matches_circle_area():
    resolution_m = 1000.0
    radius_m = 10_000.0
    kernel = disc_kernel_pixels(radius_m, resolution_m, scale_ew=1.0, scale_ns=1.0)
    assert kernel.shape[0] == kernel.shape[1]
    assert kernel.shape[0] % 2 == 1
    true_area_px = np.pi * (radius_m / resolution_m) ** 2
    assert kernel.sum() == pytest.approx(true_area_px, rel=0.05)


def test_disc_kernel_pixels_center_always_included():
    kernel = disc_kernel_pixels(5000.0, 1000.0, scale_ew=2.5, scale_ns=0.4)
    center = tuple(s // 2 for s in kernel.shape)
    assert kernel[center]


def test_registry_band_index_for_lat_matches_edges():
    edges = [0.0, 10.0, 20.0, 60.0]
    registry = EllipticalKernelRegistry(edges, radii_km=[1], resolution_m=1000.0)
    idx = registry.band_index_for_lat([0.0, 5.0, 9.99, 10.0, 15.0, 59.9, -5.0, -59.9])
    assert list(idx) == [0, 0, 0, 1, 1, 2, 0, 2]


def test_registry_build_and_kernel_for():
    edges = compute_band_edges(lat_max_deg=10.0, tolerance=0.02)
    registry = EllipticalKernelRegistry(edges, radii_km=[1, 5], resolution_m=1000.0).build()
    kernel = registry.kernel_for(0, 5)
    assert kernel.dtype == bool
    assert kernel.any()


def test_registry_save_load_roundtrip(tmp_path):
    edges = compute_band_edges(lat_max_deg=10.0, tolerance=0.02)
    registry = EllipticalKernelRegistry(edges, radii_km=[1, 5], resolution_m=1000.0).build()
    path = tmp_path / "registry.pkl"
    registry.save(path)
    loaded = EllipticalKernelRegistry.load(path)
    assert loaded.n_bands == registry.n_bands
    np.testing.assert_array_equal(loaded.kernel_for(0, 5), registry.kernel_for(0, 5))


@pytest.mark.parametrize("center_lat_deg", [0.5, 15.0, 30.0, 45.0, 59.0])
def test_elliptical_kernel_matches_geodesic_ground_truth(center_lat_deg):
    """Resolves docs/design/06-open-questions.md item 6.

    Independently rasterizes which pixels are within `radius_m` of a center
    point using WGS84 geodesic distance (pyproj.Geod, not this module's own
    closed-form scale factors), then checks the analytic elliptical kernel
    from `disc_kernel_pixels` agrees with that ground truth to a tight
    tolerance (Jaccard/IoU) -- confirming both the magnitude AND the
    orientation (which axis expands vs. compresses) of the correction.
    """
    resolution_m = 1000.0
    radius_m = 20_000.0
    half = 40  # px, comfortably larger than the kernel's extent at these radii

    fwd = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    inv = Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True)
    cx, cy = fwd.transform(0.0, center_lat_deg)
    center_lon, center_lat = inv.transform(cx, cy)

    geod = Geod(ellps="WGS84")
    ground_truth = np.zeros((2 * half + 1, 2 * half + 1), dtype=bool)
    for iy in range(-half, half + 1):
        for ix in range(-half, half + 1):
            px, py = cx + ix * resolution_m, cy + iy * resolution_m
            lon, lat = inv.transform(px, py)
            _, _, dist_m = geod.inv(center_lon, center_lat, lon, lat)
            ground_truth[iy + half, ix + half] = dist_m <= radius_m

    scale_ew, scale_ns = anisotropy_scales(center_lat_deg, STANDARD_PARALLEL_DEG)
    kernel = disc_kernel_pixels(radius_m, resolution_m, scale_ew, scale_ns)
    kh = kernel.shape[0] // 2
    kw = kernel.shape[1] // 2
    padded_kernel = np.zeros_like(ground_truth)
    padded_kernel[half - kh : half + kh + 1, half - kw : half + kw + 1] = kernel

    intersection = np.logical_and(ground_truth, padded_kernel).sum()
    union = np.logical_or(ground_truth, padded_kernel).sum()
    iou = intersection / union
    assert iou > 0.97, f"lat={center_lat_deg}: kernel/geodesic IoU too low ({iou:.4f})"

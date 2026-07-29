"""Tests for FFT convolution and halo-read primitives."""

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_zeros

from src.data.common.neighbourhood.convolve import (
    convolve_band_aware,
    mask_aware_fft_convolve,
    padded_tile_geobox,
    read_padded_tile,
    trim_halo,
)
from src.data.common.neighbourhood.kernels import EllipticalKernelRegistry, disc_kernel_pixels


def _make_canonical_geobox(width_m=50_000, height_m=50_000, resolution=1000.0):
    half_w, half_h = width_m / 2, height_m / 2
    return GeoBox.from_bbox((-half_w, -half_h, half_w, half_h), crs="EPSG:6933", resolution=resolution)


def test_mask_aware_fft_convolve_uniform_field_matches_kernel_count():
    field = np.full((41, 41), 3.0)
    mask = np.ones((41, 41), dtype=bool)
    kernel = disc_kernel_pixels(5000.0, 1000.0, 1.0, 1.0)
    S, N = mask_aware_fft_convolve(field, mask, kernel)
    center = (20, 20)
    n_px = kernel.sum()
    assert N[center] == pytest.approx(n_px, abs=1e-6)
    assert S[center] == pytest.approx(3.0 * n_px, abs=1e-6)


def test_mask_aware_fft_convolve_tracks_missing_data_at_edge():
    field = np.full((41, 41), 3.0)
    mask = np.ones((41, 41), dtype=bool)
    mask[:, :20] = False  # left half missing
    kernel = disc_kernel_pixels(5000.0, 1000.0, 1.0, 1.0)
    S, N = mask_aware_fft_convolve(field, mask, kernel)
    center = (20, 20)
    # exactly at the missing-data boundary, only ~half the disc contributes
    assert N[center] < kernel.sum() * 0.75
    # mean (S/N) still recovers the true field value despite fewer valid cells
    assert S[center] / N[center] == pytest.approx(3.0, rel=1e-6)


def test_mask_aware_fft_convolve_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mask_aware_fft_convolve(np.zeros((5, 5)), np.zeros((4, 4), dtype=bool), np.zeros((3, 3), dtype=bool))


def test_convolve_band_aware_matches_single_band_result():
    field = np.full((41, 41), 2.0)
    mask = np.ones((41, 41), dtype=bool)
    row_lat_deg = np.linspace(29.0, 31.0, 41)  # entirely inside one band around 30 deg
    edges = [0.0, 60.0]
    registry = EllipticalKernelRegistry(edges, radii_km=[5], resolution_m=1000.0).build()

    S_band, N_band = convolve_band_aware(field, mask, row_lat_deg, 5, registry)
    S_direct, N_direct = mask_aware_fft_convolve(field, mask, registry.kernel_for(0, 5))
    np.testing.assert_allclose(S_band, S_direct)
    np.testing.assert_allclose(N_band, N_direct)


def test_convolve_band_aware_uses_different_kernel_per_band():
    field = np.ones((41, 41))
    mask = np.ones((41, 41), dtype=bool)
    # first 20 rows in band 0, rest in band 1
    row_lat_deg = np.concatenate([np.full(20, 5.0), np.full(21, 55.0)])
    edges = [0.0, 30.0, 60.0]
    registry = EllipticalKernelRegistry(edges, radii_km=[10], resolution_m=1000.0).build()

    S, N = convolve_band_aware(field, mask, row_lat_deg, 10, registry)
    # the two bands have different anisotropy -> different disc pixel counts
    # -> N should differ between a row deep in band 0 vs deep in band 1
    assert N[5, 20] != pytest.approx(N[35, 20])


def test_padded_tile_geobox_grows_shape_by_halo():
    gbox = _make_canonical_geobox()
    padded = padded_tile_geobox(gbox, r_max_m=5000.0)
    assert padded.shape.x == gbox.shape.x + 10
    assert padded.shape.y == gbox.shape.y + 10


def test_read_padded_tile_interior_tile_has_no_fill():
    canonical = _make_canonical_geobox(width_m=200_000, height_m=200_000)
    source = xr_zeros(canonical, dtype="float32")
    source.values[...] = 7.0

    # a small tile well inside the canonical grid
    tile_gbox = GeoBox.from_bbox((-5000, -5000, 5000, 5000), crs="EPSG:6933", resolution=1000.0)
    padded = read_padded_tile(source, tile_gbox, r_max_m=3000.0, fill_value=np.nan)
    assert not np.isnan(padded.values).any()
    assert (padded.values == 7.0).all()


def test_read_padded_tile_edge_tile_fills_outside_source_extent():
    canonical = _make_canonical_geobox(width_m=20_000, height_m=20_000)
    source = xr_zeros(canonical, dtype="float32")
    source.values[...] = 7.0

    # a tile whose halo extends past the canonical grid's edge
    edge_bb = canonical.boundingbox
    tile_gbox = GeoBox.from_bbox(
        (edge_bb.right - 4000, edge_bb.bottom, edge_bb.right, edge_bb.bottom + 4000),
        crs="EPSG:6933",
        resolution=1000.0,
    )
    padded = read_padded_tile(source, tile_gbox, r_max_m=5000.0, fill_value=np.nan)
    assert np.isnan(padded.values).any()
    assert (padded.values[~np.isnan(padded.values)] == 7.0).all()


def test_trim_halo_recovers_core_shape():
    da = xr.DataArray(np.arange(100).reshape(10, 10), dims=("y", "x"))
    trimmed = trim_halo(da, halo_y=2, halo_x=3)
    assert trimmed.shape == (6, 4)

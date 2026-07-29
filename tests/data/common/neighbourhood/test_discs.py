"""Tests for the convolve_discs public interface and tile orchestration."""

import numpy as np
import pytest
from odc.geo.geobox import GeoBox, GeoboxTiles
from odc.geo.xr import xr_zeros

from src.data.common.neighbourhood.discs import (
    apply_own_country_mask,
    convolve_discs,
    convolve_tile,
    row_latitudes_deg,
)
from src.data.common.neighbourhood.kernels import EllipticalKernelRegistry, compute_band_edges


def _make_geobox(width_m=60_000, height_m=60_000, resolution=1000.0):
    half_w, half_h = width_m / 2, height_m / 2
    return GeoBox.from_bbox((-half_w, -half_h, half_w, half_h), crs="EPSG:6933", resolution=resolution)


def _build_registry(lat_max_deg=60.0, radii_km=(1, 5, 10)):
    edges = compute_band_edges(lat_max_deg=lat_max_deg, tolerance=0.02)
    return EllipticalKernelRegistry(edges, radii_km=list(radii_km), resolution_m=1000.0).build()


def test_row_latitudes_deg_monotonically_decreasing_north_to_south():
    gbox = _make_geobox()
    lat = row_latitudes_deg(gbox)
    assert lat[0] > lat[-1]  # row 0 is the top (north) row
    assert np.all(np.diff(lat) < 0)


def test_convolve_discs_mask_on_mask_is_degenerate_case():
    """docs/design/03-neighbourhood-engine.md §4: convolve_discs(mask, mask, ...)
    is the natural unit test -- S_d and N_d must be identical."""
    gbox = _make_geobox()
    mask = xr_zeros(gbox, dtype="bool")
    mask.values[...] = True
    mask.values[10:15, 10:15] = False  # a hole, to make this non-trivial

    registry = _build_registry(radii_km=[5])
    S_d, N_d = convolve_discs(mask, mask, [5], registry)
    np.testing.assert_allclose(S_d.values, N_d.values)


def test_convolve_discs_shape_and_coords():
    gbox = _make_geobox()
    variable = xr_zeros(gbox, dtype="float32")
    variable.values[...] = 1.0
    mask = xr_zeros(gbox, dtype="bool")
    mask.values[...] = True

    ladder = [1, 5, 10]
    registry = _build_registry(radii_km=ladder)
    S_d, N_d = convolve_discs(variable, mask, ladder, registry)

    assert S_d.dims[0] == "radius_km"
    assert list(S_d.coords["radius_km"].values) == ladder
    assert S_d.shape == (len(ladder),) + variable.shape
    assert N_d.shape == S_d.shape


def test_convolve_discs_shape_mismatch_raises():
    gbox = _make_geobox()
    variable = xr_zeros(gbox, dtype="float32")
    mask = xr_zeros(_make_geobox(width_m=30_000, height_m=30_000), dtype="bool")
    registry = _build_registry(radii_km=[1])
    with pytest.raises(ValueError):
        convolve_discs(variable, mask, [1], registry)


def test_convolve_tile_output_matches_tile_core_shape():
    canonical = _make_geobox(width_m=200_000, height_m=200_000)
    variable = xr_zeros(canonical, dtype="float32")
    variable.values[...] = 4.0
    mask = xr_zeros(canonical, dtype="bool")
    mask.values[...] = True

    tile_size = 40
    tiles = GeoboxTiles(canonical, (tile_size, tile_size))
    # pick an interior tile so the halo read has no edge fill
    ix, iy = tiles.shape[0] // 2, tiles.shape[1] // 2
    tile_gbox = tiles[ix, iy]

    r_max_m = 10_000.0
    ladder = [1, 5, 10]
    registry = _build_registry(radii_km=ladder)

    S_d, N_d = convolve_tile(variable, mask, tile_gbox, r_max_m, ladder, registry)

    assert S_d.shape == (len(ladder),) + tile_gbox.shape.yx
    # fully interior, fully valid, uniform field: disc means should recover
    # the true field value exactly, and disc counts should match kernel size
    ring_means = S_d.values / N_d.values
    np.testing.assert_allclose(ring_means, 4.0, rtol=1e-6)


def test_apply_own_country_mask_restricts_to_matching_country():
    gbox = _make_geobox()
    mask = xr_zeros(gbox, dtype="bool")
    mask.values[...] = True
    country = xr_zeros(gbox, dtype="int32")
    country.values[...] = 1
    country.values[:30, :] = 2

    restricted = apply_own_country_mask(mask, country, own_country_id=1)
    assert restricted.values[:30, :].sum() == 0
    assert restricted.values[30:, :].all()

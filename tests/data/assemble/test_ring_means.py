"""Tests for the ring-mean tabularization handoff interface."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox, GeoboxTiles
from odc.geo.xr import xr_zeros

from src.data.assemble.ring_means import (
    make_canonical_pixel_ids,
    ring_means_from_discs,
    tabularize_tile,
)
from src.data.assemble.utils import decode_pixel_id
from src.data.common.neighbourhood.discs import convolve_tile
from src.data.common.neighbourhood.kernels import EllipticalKernelRegistry, compute_band_edges


def _make_geobox(width_m=20_000, height_m=20_000, resolution=1000.0):
    return GeoBox.from_bbox((0, 0, width_m, height_m), crs="EPSG:6933", resolution=resolution)


def _synthetic_discs(ladder_km, shape, value=4.0):
    """S_d/N_d as if convolving a uniform, fully-valid field.

    N_d must grow with radius (like a real disc's pixel count, ~pi*r^2) so
    consecutive ladder radii have a strictly positive annulus pixel count --
    otherwise every ring after the first is a 0/0 (empty-annulus) case by
    construction, not a meaningful test of the uniform-field-recovery
    property.
    """
    radii = np.asarray(sorted(ladder_km), dtype=float)
    counts_per_radius = np.pi * radii**2  # monotonically increasing disc area
    n = counts_per_radius[:, None, None] * np.ones((1,) + shape)
    s = n * value
    coords = {"radius_km": list(sorted(ladder_km))}
    dims = ("radius_km", "y", "x")
    S_d = xr.DataArray(s, dims=dims, coords=coords, name="S_d")
    N_d = xr.DataArray(n, dims=dims, coords=coords, name="N_d")
    return S_d, N_d


def test_make_canonical_pixel_ids_round_trips_through_decode_pixel_id():
    gbox = _make_geobox(width_m=5000, height_m=5000)
    pixel_ids = make_canonical_pixel_ids(ix=3, iy=7, tile_geobox=gbox)
    h, w = gbox.shape
    assert pixel_ids.shape == (h, w)

    flat = pixel_ids.values.reshape(-1)
    assert len(np.unique(flat)) == flat.size  # every pixel_id unique within the tile

    for local_pixel in [0, w - 1, flat.size - 1]:
        decoded_ix, decoded_iy, decoded_local = decode_pixel_id(np.uint64(flat[local_pixel]))
        assert decoded_ix == 3
        assert decoded_iy == 7
        assert decoded_local == local_pixel


def test_make_canonical_pixel_ids_rejects_out_of_range_tile_index():
    gbox = _make_geobox(width_m=2000, height_m=2000)
    with pytest.raises(ValueError):
        make_canonical_pixel_ids(ix=2**16, iy=0, tile_geobox=gbox)


def test_ring_means_from_discs_recovers_uniform_field_value():
    ladder = [1, 2, 3, 5]
    S_d, N_d = _synthetic_discs(ladder, shape=(4, 4), value=6.0)
    ring_means = ring_means_from_discs(S_d, N_d)
    np.testing.assert_allclose(ring_means.values, 6.0)


def test_ring_means_from_discs_first_ring_uses_disc_not_difference():
    ladder = [1, 2]
    # radius 1: sum=10, count=2 -> mean 5; radius 2: sum=30, count=6 -> mean 5
    S = np.array([[[10.0]], [[30.0]]])
    N = np.array([[[2.0]], [[6.0]]])
    S_d = xr.DataArray(S, dims=("radius_km", "y", "x"), coords={"radius_km": ladder})
    N_d = xr.DataArray(N, dims=("radius_km", "y", "x"), coords={"radius_km": ladder})
    ring_means = ring_means_from_discs(S_d, N_d)
    # ring 0 (innermost disc): 10/2 = 5
    assert ring_means.values[0, 0, 0] == pytest.approx(5.0)
    # ring 1 (annulus): (30-10)/(6-2) = 5
    assert ring_means.values[1, 0, 0] == pytest.approx(5.0)


def test_ring_means_from_discs_undefined_ring_is_nan_not_inf():
    ladder = [1, 2]
    S = np.array([[[0.0]], [[0.0]]])
    N = np.array([[[0.0]], [[0.0]]])
    S_d = xr.DataArray(S, dims=("radius_km", "y", "x"), coords={"radius_km": ladder})
    N_d = xr.DataArray(N, dims=("radius_km", "y", "x"), coords={"radius_km": ladder})
    ring_means = ring_means_from_discs(S_d, N_d)
    assert np.isnan(ring_means.values).all()


def test_ring_means_from_discs_mismatched_ladder_raises():
    S_d = xr.DataArray(np.zeros((2, 2, 2)), dims=("radius_km", "y", "x"), coords={"radius_km": [1, 2]})
    N_d = xr.DataArray(np.zeros((2, 2, 2)), dims=("radius_km", "y", "x"), coords={"radius_km": [1, 3]})
    with pytest.raises(ValueError):
        ring_means_from_discs(S_d, N_d)


def test_tabularize_tile_row_count_matches_mask_and_pixel_ids_decode():
    gbox = _make_geobox(width_m=3000, height_m=3000)  # 3x3 tile
    ladder = [1, 2]
    S_d, N_d = _synthetic_discs(ladder, shape=gbox.shape.yx, value=4.0)

    mask = xr_zeros(gbox, dtype="bool")
    mask.values[...] = True
    mask.values[0, 0] = False  # one invalid cell

    df = tabularize_tile(S_d, N_d, mask, gbox, ix=2, iy=5, year=2020)

    assert len(df) == gbox.shape.x * gbox.shape.y - 1
    assert set(df.columns) == {"pixel_id", "year", "L_1km", "S_1km", "N_1km", "L_2km", "S_2km", "N_2km"}
    assert (df["year"] == 2020).all()
    np.testing.assert_allclose(df["L_1km"].values, 4.0)
    np.testing.assert_allclose(df["L_2km"].values, 4.0)

    decoded = [decode_pixel_id(np.uint64(pid)) for pid in df["pixel_id"].values[:5]]
    assert all(d[0] == 2 and d[1] == 5 for d in decoded)


def test_tabularize_tile_includes_country_id_when_provided():
    gbox = _make_geobox(width_m=2000, height_m=2000)
    ladder = [1]
    S_d, N_d = _synthetic_discs(ladder, shape=gbox.shape.yx, value=1.0)
    mask = xr_zeros(gbox, dtype="bool")
    mask.values[...] = True
    country = xr_zeros(gbox, dtype="int32")
    country.values[...] = 42

    df = tabularize_tile(S_d, N_d, mask, gbox, ix=0, iy=0, year=2019, country_raster=country)
    assert "country_id" in df.columns
    assert (df["country_id"] == 42).all()


def test_tabularize_tile_empty_mask_returns_empty_dataframe_with_columns():
    gbox = _make_geobox(width_m=2000, height_m=2000)
    ladder = [1, 5]
    S_d, N_d = _synthetic_discs(ladder, shape=gbox.shape.yx, value=1.0)
    mask = xr_zeros(gbox, dtype="bool")  # all False

    df = tabularize_tile(S_d, N_d, mask, gbox, ix=0, iy=0, year=2020)
    assert len(df) == 0
    assert "L_1km" in df.columns and "L_5km" in df.columns


def test_tabularize_tile_shape_mismatch_raises():
    gbox = _make_geobox(width_m=2000, height_m=2000)
    other = _make_geobox(width_m=4000, height_m=4000)
    S_d, N_d = _synthetic_discs([1], shape=gbox.shape.yx, value=1.0)
    mask = xr_zeros(other, dtype="bool")
    with pytest.raises(ValueError):
        tabularize_tile(S_d, N_d, mask, gbox, ix=0, iy=0, year=2020)


def test_end_to_end_convolve_tile_then_tabularize():
    """Full step-7 pipeline: canonical grid -> convolve_tile -> tabularize_tile."""
    canonical = _make_geobox(width_m=60_000, height_m=60_000)
    variable = xr_zeros(canonical, dtype="float32")
    variable.values[...] = 2.5
    mask = xr_zeros(canonical, dtype="bool")
    mask.values[...] = True

    ladder = [1, 2, 3]
    edges = compute_band_edges(lat_max_deg=60.0, tolerance=0.02)
    registry = EllipticalKernelRegistry(edges, radii_km=ladder, resolution_m=1000.0).build()

    tile_size = 20
    tiles = GeoboxTiles(canonical, (tile_size, tile_size))
    row, col = tiles.shape.y // 2, tiles.shape.x // 2
    tile_gbox = tiles[row, col]

    S_d, N_d = convolve_tile(variable, mask, tile_gbox, r_max_m=3000.0, ladder_km=ladder, kernel_registry=registry)
    tile_mask = mask.isel(
        y=slice(row * tile_size, (row + 1) * tile_size), x=slice(col * tile_size, (col + 1) * tile_size)
    )

    df = tabularize_tile(S_d, N_d, tile_mask, tile_gbox, ix=col, iy=row, year=2021)

    assert len(df) == tile_size * tile_size
    # interior tile, fully valid, uniform field -> every ring mean recovers 2.5 exactly
    for r in ladder:
        np.testing.assert_allclose(df[f"L_{r}km"].values, 2.5, rtol=1e-6)
    assert df["pixel_id"].is_unique

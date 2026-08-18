"""Tests for the Zarr disc-ladder store (docs/design/02-storage.md)."""

import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox, GeoboxTiles
from odc.geo.xr import xr_zeros

from src.data.common.geobox.cell_id import encode_cell_ids
from src.data.common.neighbourhood.discs import convolve_discs, convolve_tile
from src.data.common.neighbourhood.kernels import EllipticalKernelRegistry, compute_band_edges
from src.data.common.neighbourhood.store import (
    create_empty_disc_count_store,
    create_empty_disc_sum_store,
    write_disc_tile,
    write_disc_tile_parquet,
)
from src.data.common.tiling import iter_tiles


def _make_geobox(width_m=20_000, height_m=10_000, resolution=1000.0):
    return GeoBox.from_bbox((0, 0, width_m, height_m), crs="EPSG:6933", resolution=resolution)


def _build_registry(radii_km=(1, 2, 3)):
    edges = compute_band_edges(lat_max_deg=60.0, tolerance=0.02)
    return EllipticalKernelRegistry(edges, radii_km=list(radii_km), resolution_m=1000.0).build()


def test_create_empty_disc_sum_store_creates_zarr_skeleton(tmp_path):
    gbox = _make_geobox()
    path = tmp_path / "s_d.zarr"
    ok = create_empty_disc_sum_store(path, gbox, years=[2020, 2021], ladder_km=[1, 2, 3], tile_size=10)
    assert ok
    assert path.exists()

    ds = xr.open_zarr(str(path), consolidated=False, mask_and_scale=False)
    assert "S_d" in ds
    assert ds["S_d"].dims == ("time", "radius_km", "y", "x")
    assert ds["S_d"].shape == (2, 3, gbox.shape.y, gbox.shape.x)
    assert np.isnan(ds["S_d"].values).all()


def test_create_empty_disc_count_store_uses_zero_fill(tmp_path):
    gbox = _make_geobox()
    path = tmp_path / "n_d.zarr"
    ok = create_empty_disc_count_store(path, gbox, years=[2020], ladder_km=[1, 2, 3], tile_size=10)
    assert ok

    ds = xr.open_zarr(str(path), consolidated=False, mask_and_scale=False)
    assert "N_d" in ds
    assert ds["N_d"].dtype == np.uint16
    assert (ds["N_d"].values == 0).all()


def test_write_disc_tile_lands_in_correct_region_others_untouched(tmp_path):
    gbox = _make_geobox()
    tile_size = 10
    ladder = [1, 2, 3]
    path = tmp_path / "s_d.zarr"
    create_empty_disc_sum_store(path, gbox, years=[2020, 2021], ladder_km=ladder, tile_size=tile_size)

    tiles = GeoboxTiles(gbox, (tile_size, tile_size))
    assert tiles.shape.y == 1 and tiles.shape.x == 2  # sanity: 20km/10km grid at 1km res, 10px tiles

    tile00 = tiles[0, 0]
    field = xr_zeros(tile00, dtype="float32") + 7.0
    stacked = xr.concat([field] * len(ladder), dim="radius_km").assign_coords(radius_km=ladder)

    assert write_disc_tile(path, stacked, year=2020, variable="S_d")

    ds = xr.open_zarr(str(path), consolidated=False, mask_and_scale=False)
    time0 = pd.Timestamp("2020-12-31")
    written = ds["S_d"].sel(time=time0, radius_km=1).values
    # tile (0,0) occupies the first `tile_size` columns
    assert (written[:, :tile_size] == 7.0).all()
    # the other tile is still untouched (NaN fill)
    assert np.isnan(written[:, tile_size:]).all()
    # the other year is entirely untouched
    time1 = pd.Timestamp("2021-12-31")
    assert np.isnan(ds["S_d"].sel(time=time1, radius_km=1).values).all()


def test_write_disc_tile_dtype_param_skips_stored_dtype_lookup(tmp_path, monkeypatch):
    """A caller writing many tiles to the same store in a loop should be
    able to resolve the dtype once and pass it in, instead of paying for a
    redundant `xr.open_zarr()` metadata parse (`_stored_dtype()`) on every
    single tile write."""
    import src.data.common.neighbourhood.store as store_module

    gbox = _make_geobox()
    tile_size = 10
    ladder = [1, 2, 3]
    path = tmp_path / "s_d.zarr"
    create_empty_disc_sum_store(path, gbox, years=[2020], ladder_km=ladder, tile_size=tile_size)

    tiles = GeoboxTiles(gbox, (tile_size, tile_size))
    tile00 = tiles[0, 0]
    field = xr_zeros(tile00, dtype="float32") + 7.0
    stacked = xr.concat([field] * len(ladder), dim="radius_km").assign_coords(radius_km=ladder)

    calls = []
    original = store_module._stored_dtype

    def _counting_stored_dtype(*a, **k):
        calls.append(1)
        return original(*a, **k)

    monkeypatch.setattr(store_module, "_stored_dtype", _counting_stored_dtype)

    assert write_disc_tile(path, stacked, year=2020, variable="S_d", dtype=np.dtype("float32"))
    assert calls == []  # _stored_dtype() never called -- dtype was passed in

    ds = xr.open_zarr(str(path), consolidated=False, mask_and_scale=False)
    time0 = pd.Timestamp("2020-12-31")
    written = ds["S_d"].sel(time=time0, radius_km=1).values
    assert (written[:, :tile_size] == 7.0).all()


def test_write_disc_tile_two_tiles_both_land_correctly(tmp_path):
    gbox = _make_geobox()
    tile_size = 10
    ladder = [1, 2, 3]
    path = tmp_path / "s_d.zarr"
    create_empty_disc_sum_store(path, gbox, years=[2020], ladder_km=ladder, tile_size=tile_size)

    tiles = GeoboxTiles(gbox, (tile_size, tile_size))
    for col, value in [(0, 5.0), (1, 9.0)]:
        tile_gbox = tiles[0, col]
        field = xr_zeros(tile_gbox, dtype="float32") + value
        stacked = xr.concat([field] * len(ladder), dim="radius_km").assign_coords(radius_km=ladder)
        assert write_disc_tile(path, stacked, year=2020, variable="S_d")

    ds = xr.open_zarr(str(path), consolidated=False, mask_and_scale=False)
    written = ds["S_d"].sel(time=pd.Timestamp("2020-12-31"), radius_km=1).values
    assert (written[:, :tile_size] == 5.0).all()
    assert (written[:, tile_size:] == 9.0).all()


def test_end_to_end_convolve_tile_then_write_disc_tile(tmp_path):
    """The full step-6 pipeline: canonical grid -> convolve_tile -> write_disc_tile -> read back."""
    canonical = _make_geobox(width_m=100_000, height_m=100_000)
    variable = xr_zeros(canonical, dtype="float32")
    variable.values[...] = 3.0
    mask = xr_zeros(canonical, dtype="bool")
    mask.values[...] = True

    ladder = [1, 2, 3]
    registry = _build_registry(radii_km=ladder)
    tile_size = 20
    tiles = GeoboxTiles(canonical, (tile_size, tile_size))
    row, col = tiles.shape.y // 2, tiles.shape.x // 2
    tile_gbox = tiles[row, col]

    S_d, N_d = convolve_tile(variable, mask, tile_gbox, r_max_m=3000.0, ladder_km=ladder, kernel_registry=registry)

    s_path = tmp_path / "s_d.zarr"
    create_empty_disc_sum_store(s_path, canonical, years=[2020], ladder_km=ladder, tile_size=tile_size)
    assert write_disc_tile(s_path, S_d, year=2020, variable="S_d")

    ds = xr.open_zarr(str(s_path), consolidated=False, mask_and_scale=False)
    written = ds["S_d"].sel(time=pd.Timestamp("2020-12-31"), radius_km=1)
    # interior, fully-valid, uniform field -> disc mean == 3.0 wherever N_d > 0 in this tile
    tile_written = written.isel(
        y=slice(row * tile_size, (row + 1) * tile_size), x=slice(col * tile_size, (col + 1) * tile_size)
    )
    assert not np.isnan(tile_written.values).any()


def test_write_disc_tile_casts_float_convolution_output_to_stored_integer_dtype(tmp_path):
    """Regression test: convolve_tile's N_d is always float64 (scipy.signal.fftconvolve's
    native output dtype), but the count store is uint16 (docs/design/02-storage.md §6).

    Caught via scripts/validate_backbone_subset.py: writing that float64 array
    straight into a uint16 zarr region previously corrupted values (xarray
    silently truncating float bit patterns into the integer dtype) instead of
    rounding/casting cleanly.
    """
    canonical = _make_geobox(width_m=60_000, height_m=60_000)
    variable = xr_zeros(canonical, dtype="float32")
    variable.values[...] = 1.0
    mask = xr_zeros(canonical, dtype="bool")
    mask.values[...] = True

    ladder = [1, 2, 3]
    registry = _build_registry(radii_km=ladder)
    tile_size = 20
    tiles = GeoboxTiles(canonical, (tile_size, tile_size))
    row, col = tiles.shape.y // 2, tiles.shape.x // 2
    tile_gbox = tiles[row, col]

    S_d, N_d = convolve_tile(variable, mask, tile_gbox, r_max_m=3000.0, ladder_km=ladder, kernel_registry=registry)
    assert N_d.dtype == np.float64  # sanity: confirms the scenario this test guards against

    n_path = tmp_path / "n_d.zarr"
    create_empty_disc_count_store(n_path, canonical, years=[2020], ladder_km=ladder, tile_size=tile_size)
    assert write_disc_tile(n_path, N_d, year=2020, variable="N_d")

    ds = xr.open_zarr(str(n_path), consolidated=False, mask_and_scale=False)
    assert ds["N_d"].dtype == np.uint16
    written = ds["N_d"].sel(time=pd.Timestamp("2020-12-31"), radius_km=1).isel(
        y=slice(row * tile_size, (row + 1) * tile_size), x=slice(col * tile_size, (col + 1) * tile_size)
    )
    # interior, fully-valid, uniform-mask tile -> N_d should equal the disc
    # pixel count exactly (an integer already, before rounding), not garbage
    # from a truncated float64->uint16 bit reinterpretation
    expected = np.round(N_d.sel(radius_km=1).values).astype(np.uint16)
    np.testing.assert_array_equal(written.values, expected)
    assert written.values.max() < 1000  # sanity ceiling: not a bit-pattern-corrupted huge value


def _middle_tile(canonical, tile_size):
    tiles = list(iter_tiles(canonical, tile_size=tile_size))
    return tiles[len(tiles) // 2]


def test_write_disc_tile_parquet_round_trips_and_sorts_by_cell_id(tmp_path):
    canonical = _make_geobox(width_m=100_000, height_m=100_000)
    variable = xr_zeros(canonical, dtype="float32")
    variable.values[...] = 3.0
    mask = xr_zeros(canonical, dtype="bool")
    mask.values[...] = True

    ladder = [1, 2, 3]
    registry = _build_registry(radii_km=ladder)
    tile_size = 20
    full_width = canonical.shape.x
    tile = _middle_tile(canonical, tile_size)

    S_d, N_d = convolve_tile(
        variable, mask, tile.geobox, r_max_m=3000.0, ladder_km=ladder, kernel_registry=registry
    )

    out_dir = tmp_path / "s_d_parquet"
    out_path = write_disc_tile_parquet(out_dir, S_d, N_d, tile, year=2020, full_width=full_width)

    expected_path = out_dir / f"ix={tile.row}" / f"iy={tile.col}" / "part-2020.parquet"
    assert str(out_path) == str(expected_path)
    assert expected_path.exists()

    df = pd.read_parquet(out_path)
    h, w = tile.geobox.shape
    assert len(df) == h * w * len(ladder)
    assert set(df.columns) == {"cell_id", "year", "radius_km", "S_d", "N_d"}
    assert (df["year"] == 2020).all()

    # sorted by (cell_id, radius_km)
    assert list(df["cell_id"]) == sorted(df["cell_id"])
    for cid, group in df.groupby("cell_id"):
        assert list(group["radius_km"]) == sorted(group["radius_km"])

    # cell_id values match an independently computed encoding
    row0, col0 = tile.y_slice.start, tile.x_slice.start
    expected_ids = set(encode_cell_ids(row0, col0, tile.geobox, full_width).reshape(-1).tolist())
    assert set(df["cell_id"].unique().tolist()) == expected_ids

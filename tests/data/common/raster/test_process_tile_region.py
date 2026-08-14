"""process_tile_region(): per-output-tile reprojection + region write, the
compute path PREPARE uses to write per-(tile, year) regions rather than
`write_year_to_zarr`'s whole-extent-per-year write. Regions must land exactly
on the pre-created zarr's chunk boundaries (chunk_size == tile_size) and the
tile-by-tile result must match a single whole-extent write of the same
source data byte-for-byte.
"""

import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox

from src.data.common import tiling
from src.data.common.raster.spatial import SpatialProcessor


def _sample_source_ds(year: int, size: int = 8):
    lon = np.linspace(-1.0, 1.0, size)
    lat = np.linspace(1.0, -1.0, size)
    data = np.arange(size * size, dtype="float32").reshape(1, 1, size, size)
    ds = xr.Dataset(
        {"value": (("time", "band", "latitude", "longitude"), data)},
        coords={
            "time": [pd.Timestamp(f"{year}-12-31")],
            "band": [1],
            "latitude": lat,
            "longitude": lon,
        },
    )
    return ds.rio.write_crs(4326)


def test_tile_by_tile_write_matches_whole_extent_write(tmp_path):
    target_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.25)  # 8x8
    tile_size = 4  # 2x2 tile grid
    year = 2020

    processor = SpatialProcessor(hpc_root=str(tmp_path))
    source_ds = _sample_source_ds(year)

    # Reference: one whole-extent write.
    ref_path = tmp_path / "reference.zarr"
    assert processor.create_empty_target_zarr(
        str(ref_path), target_geobox, [year], ["value"], dtype="float32", packaging_attrs={},
    )
    assert processor.write_year_to_zarr(source_ds.copy(deep=True), str(ref_path), year, target_geobox)

    # Tiled: one empty zarr, chunk-aligned to tile_size, written tile by tile.
    tiled_path = tmp_path / "tiled.zarr"
    assert processor.create_empty_target_zarr(
        str(tiled_path), target_geobox, [year], ["value"], dtype="float32", packaging_attrs={},
        chunk_size=(tile_size, tile_size),
    )
    dim_y, dim_x = target_geobox.dimensions
    for tile in tiling.iter_tiles(target_geobox, tile_size=tile_size):
        ok = processor.process_tile_region(
            source_ds.copy(deep=True), str(tiled_path), tile, (dim_y, dim_x),
        )
        assert ok, f"tile {tile.id} failed"

    ref_ds = xr.open_zarr(str(ref_path), consolidated=False, decode_coords="all")
    tiled_ds = xr.open_zarr(str(tiled_path), consolidated=False, decode_coords="all")
    try:
        np.testing.assert_allclose(
            ref_ds["value"].values, tiled_ds["value"].values, equal_nan=True,
        )
    finally:
        ref_ds.close()
        tiled_ds.close()


def test_process_tile_region_only_touches_its_own_slice(tmp_path):
    """Writing one tile must not disturb neighboring tiles' still-zero cells."""
    target_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.25)
    tile_size = 4
    year = 2020

    processor = SpatialProcessor(hpc_root=str(tmp_path))
    output_path = tmp_path / "output.zarr"
    assert processor.create_empty_target_zarr(
        str(output_path), target_geobox, [year], ["value"], dtype="float32", packaging_attrs={},
        chunk_size=(tile_size, tile_size),
    )

    dim_y, dim_x = target_geobox.dimensions
    tiles = list(tiling.iter_tiles(target_geobox, tile_size=tile_size))
    first_tile = tiles[0]
    source_ds = _sample_source_ds(year)

    assert processor.process_tile_region(source_ds, str(output_path), first_tile, (dim_y, dim_x))

    ds = xr.open_zarr(str(output_path), consolidated=False, decode_coords="all")
    try:
        arr = ds["value"].isel(time=0, band=0).values
        untouched = arr[first_tile.y_slice.stop :, first_tile.x_slice.stop :]
        assert np.all(np.isnan(untouched))  # default float32 fill value
    finally:
        ds.close()

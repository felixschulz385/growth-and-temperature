"""process_tile_region(): per-output-tile reprojection + parquet write, the
compute path PREPARE uses to write per-(tile, year) `cell_id`-keyed parquet
parts. Values must match a manual reproject-onto-tile computation, and one
tile's write must not disturb any other tile's output (trivially true here
since each tile is its own file, unlike the old shared-Zarr-store region
write this replaced)."""

import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_reproject

from src.data.common import tiling
from src.data.common.geobox.cell_id import encode_cell_ids
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


def test_process_tile_region_values_match_direct_reproject(tmp_path):
    target_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.25)  # 8x8
    tile_size = 4  # 2x2 tile grid
    year = 2020
    full_width = target_geobox.shape.x

    processor = SpatialProcessor(hpc_root=str(tmp_path))
    output_path = tmp_path / "output"

    for tile in tiling.iter_tiles(target_geobox, tile_size=tile_size):
        source_ds = _sample_source_ds(year)
        assert processor.process_tile_region(source_ds.copy(deep=True), str(output_path), tile, year, full_width)

        part = output_path / f"ix={tile.row}" / f"iy={tile.col}" / f"part-{year}.parquet"
        assert part.exists()
        df = pd.read_parquet(part)

        expected_ds = xr_reproject(_sample_source_ds(year), tile.geobox, resampling="nearest")
        expected_values = np.asarray(expected_ds["value"].values).reshape(-1)
        expected_ids = encode_cell_ids(tile.y_slice.start, tile.x_slice.start, tile.geobox, full_width)

        assert set(df["cell_id"]) == set(expected_ids.reshape(-1).tolist())
        assert (df["year"] == year).all()
        assert list(df["cell_id"]) == sorted(df["cell_id"])  # sorted by cell_id
        merged = df.set_index("cell_id").loc[expected_ids.reshape(-1)]
        np.testing.assert_allclose(merged["value"].values, expected_values, equal_nan=True)


def test_process_tile_region_only_writes_its_own_tile_file(tmp_path):
    """Writing one tile creates only that tile's own part file -- no other
    tile's output exists yet, and no shared store is touched."""
    target_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.25)
    tile_size = 4
    year = 2020
    full_width = target_geobox.shape.x

    processor = SpatialProcessor(hpc_root=str(tmp_path))
    output_path = tmp_path / "output"

    tiles = list(tiling.iter_tiles(target_geobox, tile_size=tile_size))
    first_tile = tiles[0]
    source_ds = _sample_source_ds(year)

    assert processor.process_tile_region(source_ds, str(output_path), first_tile, year, full_width)

    parts = sorted(output_path.glob("ix=*/iy=*/part-*.parquet"))
    assert len(parts) == 1
    assert parts[0] == output_path / f"ix={first_tile.row}" / f"iy={first_tile.col}" / f"part-{year}.parquet"


def test_process_tile_region_two_tiles_produce_two_independent_files(tmp_path):
    target_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.25)
    tile_size = 4
    year = 2020
    full_width = target_geobox.shape.x

    processor = SpatialProcessor(hpc_root=str(tmp_path))
    output_path = tmp_path / "output"

    tiles = list(tiling.iter_tiles(target_geobox, tile_size=tile_size))
    for tile in tiles[:2]:
        source_ds = _sample_source_ds(year)
        assert processor.process_tile_region(source_ds, str(output_path), tile, year, full_width)

    parts = sorted(output_path.glob("ix=*/iy=*/part-*.parquet"))
    assert len(parts) == 2
    dfs = [pd.read_parquet(p) for p in parts]
    assert set(dfs[0]["cell_id"]).isdisjoint(set(dfs[1]["cell_id"]))


def test_process_tile_region_reproject_false_tabulates_georegistered_nan_canvas(tmp_path):
    """`reproject=False` (MODIS/GLASS-MODIS) path: a georegistered all-NaN
    `xr_zeros(tile.geobox)` canvas is written straight to parquet, one row
    per pixel, NaN preserved -- no `xr_reproject` call."""
    from odc.geo.xr import xr_zeros

    target_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.25)
    tile_size, year = 4, 2020
    full_width = target_geobox.shape.x
    processor = SpatialProcessor(hpc_root=str(tmp_path))
    output_path = tmp_path / "output"

    tile = next(iter(tiling.iter_tiles(target_geobox, tile_size=tile_size)))
    canvas = xr.Dataset({"lst": xr.full_like(xr_zeros(tile.geobox, "float32"), np.nan)})
    assert processor.process_tile_region(canvas, str(output_path), tile, year, full_width, reproject=False)

    df = pd.read_parquet(output_path / f"ix={tile.row}" / f"iy={tile.col}" / f"part-{year}.parquet")
    h, w = tile.geobox.shape
    assert len(df) == h * w
    assert df["lst"].isna().all()
    assert set(df["cell_id"]) == set(
        encode_cell_ids(tile.y_slice.start, tile.x_slice.start, tile.geobox, full_width).reshape(-1).tolist()
    )

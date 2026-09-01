"""build_sinusoidal_mosaic_raw_getter: the PREPARE `raw_getter` that reprojects
each fetched sinusoidal source tile onto the output tile's own geobox and
overlays first-wins onto a georegistered NaN canvas
(docs/design/15-modis-prepare-2002-tile-failures.md)."""

import os

import numpy as np
import rasterio
import xarray as xr
from odc.geo.geobox import GeoBox
from rasterio.transform import from_bounds

from src.data.common import tiling
from src.data.common.prepare import sinusoidal_mosaic
from src.data.common.prepare.sinusoidal_mosaic import build_sinusoidal_mosaic_raw_getter
from src.data.sources.modis import tiles as modis_util


def _read_annual_geotiff(path, year):
    """Stand-in for a source's own reader: one var per band, no time/band dims."""
    da = rasterio_open_as_da(path)
    return da


def rasterio_open_as_da(path):
    import rioxarray as rxr

    da = rxr.open_rasterio(path, masked=True, chunks=True)
    with rasterio.open(path) as src:
        names = src.descriptions
    ds = xr.Dataset({name or f"b{i}": da.isel(band=i, drop=True) for i, name in enumerate(names)})
    return ds.rio.write_crs(da.rio.crs)


def _write_tif(path, value, *, bounds, crs, band_names=("v",), size=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    transform = from_bounds(*bounds, size, size)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=len(band_names),
        dtype="float32", crs=crs, transform=transform, nodata=np.nan,
    ) as dst:
        for i, name in enumerate(band_names, start=1):
            dst.write(np.full((size, size), value, dtype="float32"), i)
            dst.set_band_description(i, name)


def _tiles(target_geobox, tile_size):
    return {(t.row, t.col): t for t in tiling.iter_tiles(target_geobox, tile_size=tile_size)}


def test_empty_year_index_returns_none(tmp_path):
    getter = build_sinusoidal_mosaic_raw_getter(
        stage1_root=str(tmp_path), target_geobox=GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.5),
        read_annual_geotiff=_read_annual_geotiff,
    )
    tile = next(iter(tiling.iter_tiles(GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.5), tile_size=2)))
    assert getter(tile, 2019) is None  # no FETCH output for the year -> retryable


def test_no_overlap_returns_nan_canvas_on_tile_geobox(tmp_path):
    tg = GeoBox.from_bbox((-4, -4, 4, 4), crs="EPSG:4326", resolution=1.0)  # 8x8, 4x4 tiles @ size 2
    _write_tif(os.path.join(tmp_path, "2019", "h00v00.tif"), 5.0, bounds=(2, 2, 4, 4), crs="EPSG:4326")

    getter = build_sinusoidal_mosaic_raw_getter(
        stage1_root=str(tmp_path), target_geobox=tg, read_annual_geotiff=_read_annual_geotiff, pad_pixels=0
    )
    tiles = _tiles(tg, 2)
    sw = getter(tiles[(3, 0)], 2019)  # south-west tile, source is north-east
    assert sw.odc.geobox == tiles[(3, 0)].geobox
    assert list(sw.data_vars) == ["v"]
    assert np.isnan(sw["v"].values).all()
    assert sw["v"].shape == tiles[(3, 0)].geobox.shape


def test_single_source_tile_reprojects_onto_output_grid(tmp_path):
    """Genuine sinusoidal -> EASE6933 reprojection (the `_write_tile_tif`
    fixtures elsewhere are same-CRS)."""
    x0, y0, x1, y1 = modis_util.tile_bounds_m(18, 8)  # equatorial Africa
    _write_tif(
        os.path.join(tmp_path, "2019", "h18v08.tif"), 300.0,
        bounds=(x0, y0, x1, y1), crs=modis_util.SINUSOIDAL_PROJ4, size=20,
    )
    # A small EASE6933 grid overlapping that sinusoidal tile's footprint.
    tg = GeoBox.from_bbox((0, -1_000_000, 1_500_000, 900_000), crs="EPSG:6933", resolution=100_000)

    getter = build_sinusoidal_mosaic_raw_getter(
        stage1_root=str(tmp_path), target_geobox=tg, read_annual_geotiff=_read_annual_geotiff
    )
    tile = next(iter(tiling.iter_tiles(tg, tile_size=max(tg.shape))))
    out = getter(tile, 2019)

    assert out.odc.geobox == tile.geobox
    vals = out["v"].values
    assert np.isnan(vals).any()  # partial coverage
    covered = vals[~np.isnan(vals)]
    assert covered.size > 0 and np.allclose(covered, 300.0)


def test_multiple_overlapping_tiles_first_wins(tmp_path):
    tg = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.5)
    _write_tif(os.path.join(tmp_path, "2019", "h00v00.tif"), 1.0, bounds=(-1, -1, 1, 1), crs="EPSG:4326")
    _write_tif(os.path.join(tmp_path, "2019", "h01v00.tif"), 2.0, bounds=(-1, -1, 1, 1), crs="EPSG:4326")

    getter = build_sinusoidal_mosaic_raw_getter(
        stage1_root=str(tmp_path), target_geobox=tg, read_annual_geotiff=_read_annual_geotiff
    )
    tile = next(iter(tiling.iter_tiles(tg, tile_size=2)))
    out = getter(tile, 2019)
    assert set(np.unique(out["v"].values)) == {1.0}  # h00v00 sorts before h01v00


def test_source_tiles_are_materialised_not_dask(tmp_path):
    tg = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.5)
    _write_tif(os.path.join(tmp_path, "2019", "h00v00.tif"), 7.0, bounds=(-1, -1, 1, 1), crs="EPSG:4326")

    getter = build_sinusoidal_mosaic_raw_getter(
        stage1_root=str(tmp_path), target_geobox=tg, read_annual_geotiff=_read_annual_geotiff
    )
    tile = next(iter(tiling.iter_tiles(tg, tile_size=2)))
    out = getter(tile, 2019)
    assert out.chunks is None or all(len(c) == 0 for c in out.chunks.values())


def test_lru_never_exceeds_configured_size(tmp_path):
    tg = GeoBox.from_bbox((-4, -4, 4, 4), crs="EPSG:4326", resolution=1.0)
    for h in range(6):
        _write_tif(
            os.path.join(tmp_path, "2019", f"h0{h}v00.tif"), float(h),
            bounds=(-4 + h, -4, -3 + h, 4), crs="EPSG:4326",
        )
    getter = build_sinusoidal_mosaic_raw_getter(
        stage1_root=str(tmp_path), target_geobox=tg, read_annual_geotiff=_read_annual_geotiff,
        source_tile_cache_size=3,
    )
    for tile in tiling.iter_tiles(tg, tile_size=2):
        getter(tile, 2019)
    assert len(getter.source_tile_cache) <= 3


def test_reproject_failure_on_one_source_tile_is_skipped_not_raised(tmp_path, monkeypatch):
    tg = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.5)
    _write_tif(os.path.join(tmp_path, "2019", "h00v00.tif"), 1.0, bounds=(-1, -1, 1, 1), crs="EPSG:4326")
    _write_tif(os.path.join(tmp_path, "2019", "h01v00.tif"), 2.0, bounds=(-1, -1, 1, 1), crs="EPSG:4326")

    real = sinusoidal_mosaic.xr_reproject

    def flaky(src, how, **kw):
        if float(np.asarray(next(iter(src.data_vars.values())).values).flat[0]) == 1.0:
            raise RuntimeError("simulated degenerate footprint")
        return real(src, how, **kw)

    monkeypatch.setattr(sinusoidal_mosaic, "xr_reproject", flaky)

    getter = build_sinusoidal_mosaic_raw_getter(
        stage1_root=str(tmp_path), target_geobox=tg, read_annual_geotiff=_read_annual_geotiff
    )
    tile = next(iter(tiling.iter_tiles(tg, tile_size=2)))
    out = getter(tile, 2019)  # must not raise
    assert set(np.unique(out["v"].values)) == {2.0}  # h00v00 skipped, h01v00 filled


def test_ragged_tile_shape_preserved(tmp_path):
    # 5x5 grid, tile_size 3 -> a ragged 2-wide / 2-tall last row & col.
    tg = GeoBox.from_bbox((0, 0, 5, 5), crs="EPSG:4326", resolution=1.0)
    _write_tif(os.path.join(tmp_path, "2019", "h00v00.tif"), 9.0, bounds=(0, 0, 5, 5), crs="EPSG:4326")

    getter = build_sinusoidal_mosaic_raw_getter(
        stage1_root=str(tmp_path), target_geobox=tg, read_annual_geotiff=_read_annual_geotiff
    )
    tiles = _tiles(tg, 3)
    corner = tiles[(1, 1)]  # ragged last row + last col
    out = getter(corner, 2019)
    assert out["v"].shape == corner.geobox.shape == (2, 2)
    assert out.odc.geobox == corner.geobox

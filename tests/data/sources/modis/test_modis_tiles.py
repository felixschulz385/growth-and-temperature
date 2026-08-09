"""compute_land_tiles(): docs/design/07a-modis-band-reference.md's

"confirm against the actual sinusoidal tile grid + a land mask before
finalizing tile lists" -- exercises the real reprojection + per-tile overlap
logic (mirrors gadm.py's pre-filter pattern) against small synthetic land
layers, not the real multi-hundred-MB OSM download.
"""

import geopandas as gpd
import shapely.geometry
from pyproj import Transformer

from src.data.sources.modis import tiles as modis_util


def _write_gpkg(tmp_path, name, geoms, crs):
    path = str(tmp_path / name)
    gpd.GeoDataFrame(geometry=list(geoms), crs=crs).to_file(path, driver="GPKG")
    return path


def _tile_center_box_m(h, v, half_extent_m=1000.0):
    x0, y0, x1, y1 = modis_util.tile_bounds_m(h, v)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return shapely.geometry.box(cx - half_extent_m, cy - half_extent_m, cx + half_extent_m, cy + half_extent_m)


def test_compute_land_tiles_finds_overlapping_tile(tmp_path):
    land_path = _write_gpkg(
        tmp_path, "land.gpkg", [_tile_center_box_m(18, 4)], crs=modis_util.SINUSOIDAL_PROJ4
    )
    result = modis_util.compute_land_tiles(land_path, lat_clip_deg=60.0)
    assert result == {"h18v04"}


def test_compute_land_tiles_excludes_tiles_without_overlap(tmp_path):
    land_path = _write_gpkg(
        tmp_path, "land.gpkg",
        [_tile_center_box_m(18, 4), _tile_center_box_m(20, 8)],
        crs=modis_util.SINUSOIDAL_PROJ4,
    )
    result = modis_util.compute_land_tiles(land_path, lat_clip_deg=60.0)
    assert result == {"h18v04", "h20v08"}
    assert "h09v05" not in result


def test_compute_land_tiles_respects_lat_clip_deg(tmp_path):
    # v=0 is the top tile row, entirely beyond the 60-degree clip even
    # though it is (in this synthetic layer) fully land-covered.
    land_path = _write_gpkg(
        tmp_path, "land.gpkg", [_tile_center_box_m(18, 0)], crs=modis_util.SINUSOIDAL_PROJ4
    )
    result = modis_util.compute_land_tiles(land_path, lat_clip_deg=60.0)
    assert result == set()


def test_compute_land_tiles_reprojects_from_wgs84(tmp_path):
    h, v = 12, 7
    x0, y0, x1, y1 = modis_util.tile_bounds_m(h, v)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    transformer = Transformer.from_crs(modis_util.SINUSOIDAL_PROJ4, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(cx, cy)

    land_path = _write_gpkg(
        tmp_path, "land_wgs84.gpkg", [shapely.geometry.Point(lon, lat).buffer(0.01)], crs="EPSG:4326"
    )
    result = modis_util.compute_land_tiles(land_path, lat_clip_deg=60.0)
    assert result == {f"h{h:02d}v{v:02d}"}


def test_compute_land_tiles_feeds_get_modis_sinusoidal_tiles(tmp_path):
    land_path = _write_gpkg(
        tmp_path, "land.gpkg",
        [_tile_center_box_m(18, 4), _tile_center_box_m(20, 8)],
        crs=modis_util.SINUSOIDAL_PROJ4,
    )
    land_tiles = modis_util.compute_land_tiles(land_path, lat_clip_deg=60.0)
    tiles = modis_util.get_modis_sinusoidal_tiles(lat_clip_deg=60.0, land_tiles=land_tiles)
    assert set(tiles) == {"h18v04", "h20v08"}

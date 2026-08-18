"""Regression tests for the ease6933 grid-switch correctness fix in
GadmSource/OsmSource: before this fix, both `_execute_grid` methods called
`get_or_create_geobox()` directly (ignoring `ctx.grid_id`) and hardcoded
`latitude`/`longitude` dim names in their zarr-writing helpers -- a
projected canonical geobox (`y`/`x` dims) would have raised a `KeyError`.
"""

from pathlib import Path

from src.data.common.geobox.canonical import canonical_ease_geobox
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry


def _make(tmp_path, source_id, grid_id="legacy_4326", **raw):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), grid_id=grid_id
    )
    cfg = SourceConfig.from_dict(source_id, dict(raw))
    cls = registry.load(source_id)
    return cls(ctx, cfg), ctx


def _coarse_ease_geobox():
    # 50km resolution -> tiny grid, cheap for a unit test.
    return canonical_ease_geobox(resolution_m=50_000.0, lat_clip_deg=60.0)


def test_rasterize_tile_needs_gdf_reprojected_to_target_crs_first(tmp_path):
    """Regression test for a confirmed real-data bug: `_rasterize_tile`'s
    per-tile overlap pre-filter compares a `tile_polygon` built in the
    *target* geobox's CRS (e.g. EASE6933 projected meters) against each
    GeoDataFrame's geometries via plain shapely `.intersects()`, which never
    reprojects. Left in GADM's native WGS84 lon/lat degrees, that comparison
    is numerically incompatible (~1e7-magnitude meters vs +/-180/+/-90
    degrees) and silently finds ~no overlap for ~every tile -- confirmed on
    real HPC output via src.data.sources.verify: every GID_N level came back
    ~99.98% nodata despite valid input geometries and a clean (no-exception)
    run. `_rasterize_levels` reprojects each level's GeoDataFrame to the
    target CRS once, up front, before calling `_rasterize_tile`.

    Uses a single tile (taken from a real geobox, so it has genuine
    lon/lat<->projected-meters georeferencing). `_rasterize_tile` returns a
    plain in-memory `xr.Dataset` (no Zarr round-trip/CF fill-value decoding
    involved any more), so an untouched pixel is literally `0`, no NaN
    subtlety to account for.
    """
    import geopandas as gpd
    import shapely.geometry
    from odc.geo import GeoboxTiles

    from src.data.common.tiling import Tile
    from src.data.sources.misc.gadm import GadmSource

    geobox = _coarse_ease_geobox()
    tiles = GeoboxTiles(geobox, (16, 16))
    # A tile comfortably away from the meter-CRS origin in both dims (real
    # bounds ~[-11.0M, -2.4M] to [-10.2M, -1.6M] meters) -- no plausible
    # degree-valued geometry can numerically fall inside it by accident, so
    # the bug can't be masked by a spurious near-origin-tile "hit".
    target_tile_geobox = tiles[10, 8]
    tile = Tile(row=10, col=8, geobox=target_tile_geobox, y_slice=slice(0, 0), x_slice=slice(0, 0))

    # A small WGS84 polygon centered on this tile's own real geographic
    # footprint -- should legitimately rasterize once reprojected to the
    # target CRS, and *only* then.
    wgs84_bounds = target_tile_geobox.extent.to_crs("EPSG:4326").boundingbox
    cx = (wgs84_bounds.left + wgs84_bounds.right) / 2
    cy = (wgs84_bounds.bottom + wgs84_bounds.top) / 2
    polygon = shapely.geometry.box(cx - 0.5, cy - 0.5, cx + 0.5, cy + 0.5)
    gdf_wgs84 = gpd.GeoDataFrame({"GID_0": ["AAA"]}, geometry=[polygon], crs="EPSG:4326")
    level_code_to_id = {"GID_0": {"AAA": 1}}

    # Bug reproduction: pass the GeoDataFrame in its native (unreprojected) CRS.
    buggy_ds = GadmSource._rasterize_tile({"GID_0": gdf_wgs84}, level_code_to_id, tile)
    assert int((buggy_ds["GID_0"].values != 0).sum()) == 0

    # Fix: reproject to the target geobox's own CRS first, like
    # _rasterize_levels does now.
    reprojected = {"GID_0": gdf_wgs84.to_crs(geobox.crs)}
    fixed_ds = GadmSource._rasterize_tile(reprojected, level_code_to_id, tile)
    assert int((fixed_ds["GID_0"].values != 0).sum()) > 0


def test_gadm_rasterize_levels_threads_ctx_grid_id_into_target_geobox(tmp_path, monkeypatch):
    # `_rasterize_levels()` is called directly by `_execute_prepare`
    # (src/data/sources/misc/gadm.py module docstring) -- exercised
    # directly here rather than through a StepTarget, since there's no
    # separate GRID target.
    gadm, ctx = _make(tmp_path, "gadm", grid_id="ease6933")

    import src.data.common.geobox as geobox_module
    import src.data.common.prepare.driver as driver_module

    captured = {}
    fake_geobox = _coarse_ease_geobox()

    def fake_get_target_geobox(passed_ctx):
        captured["ctx"] = passed_ctx
        return fake_geobox

    def fake_run_tiled_prepare(*, target_geobox, **kwargs):
        captured["target_geobox"] = target_geobox
        return True

    monkeypatch.setattr(geobox_module, "get_target_geobox", fake_get_target_geobox)
    monkeypatch.setattr(driver_module, "run_tiled_prepare", fake_run_tiled_prepare)
    monkeypatch.setattr(type(gadm), "_dask_client", lambda self: _NullContext())

    import geopandas as gpd
    from shapely.geometry import Point

    gdf_adm0 = gpd.GeoDataFrame({"GID_0": ["AAA"]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    adm0_path = str(tmp_path / "gadm_levelADM_0_simplified.gpkg")
    gdf_adm0.to_file(adm0_path, driver="GPKG")

    monkeypatch.setattr(gpd, "read_file", lambda path, engine=None: gdf_adm0)

    output_path = str(tmp_path / "out" / "countries_grid")
    assert gadm._rasterize_levels([adm0_path], output_path) is True
    assert captured["ctx"] is ctx
    assert captured["target_geobox"] is fake_geobox


class _NullContext:
    def __enter__(self):
        return _FakeClient()

    def __exit__(self, *exc):
        return False


class _FakeClient:
    dashboard_link = None


def test_osm_rasterize_writes_cell_id_keyed_parquet_for_ease_geobox(tmp_path, monkeypatch):
    """`_rasterize()` now runs on `run_tiled_prepare(years=None,
    reproject=False, ...)` -- called directly by `_execute_prepare`
    (src/data/sources/misc/osm.py module docstring) -- exercised directly
    here rather than through a StepTarget, since there's no separate GRID
    target. Output is a directory of `cell_id`-keyed parquet parts, one per
    tile, no year column (static, no temporal dimension)."""
    osm, ctx = _make(tmp_path, "osm", grid_id="ease6933")

    import src.data.common.geobox as geobox_module

    fake_geobox = _coarse_ease_geobox()
    monkeypatch.setattr(geobox_module, "get_target_geobox", lambda passed_ctx: fake_geobox)

    import geopandas as gpd
    from shapely.geometry import Polygon

    gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:4326")
    input_path = str(tmp_path / "land_polygons.gpkg")
    gdf.to_file(input_path, driver="GPKG")

    output_path = str(tmp_path / "out" / "land_mask")
    assert osm._rasterize(input_path, output_path) is True

    import pandas as pd

    parts = sorted(Path(output_path).glob("ix=*/iy=*/part.parquet"))
    assert parts
    df = pd.concat(pd.read_parquet(p) for p in parts)
    assert "year" not in df.columns
    assert set(df.columns) == {"cell_id", "land_mask"}

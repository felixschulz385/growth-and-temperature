"""Grid-geobox regressions mirroring GADM's own
(tests/data/sources/misc/test_gadm_osm_grid_geobox.py): the CRS-
reprojection-before-tiling fix GADM hit for real (commit f653033) --
carried over here since `_rasterize_tile` uses the identical CRS-naive
`.intersects()`-adjacent prefilter pattern (`gdf.sindex.query()`)."""

from src.data.common.geobox.canonical import canonical_ease_geobox
from src.data.common.tiling import Tile
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.ecoregions.source import EcoregionsSource


def _make(tmp_path, grid_id="legacy_4326"):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), grid_id=grid_id)
    cfg = SourceConfig.from_dict("ecoregions", {"url": "https://example.test/eco.zip", "name": "eco.zip"})
    cls = registry.load("ecoregions")
    return cls(ctx, cfg), ctx


def _coarse_ease_geobox():
    return canonical_ease_geobox(resolution_m=50_000.0, lat_clip_deg=60.0)


def test_rasterize_tile_needs_gdf_reprojected_to_target_crs_first(tmp_path):
    """Same failure mode GADM hit on real data: the per-tile overlap
    prefilter compares raw shapely geometries against a tile polygon
    already in the target (projected, e.g. EASE6933) CRS -- left in WGS84
    degrees, that silently finds ~no overlap everywhere. `_rasterize_tile`
    returns a plain in-memory `xr.Dataset` (no Zarr round-trip/CF fill-value
    decoding), so an untouched pixel is literally `0`."""
    import geopandas as gpd
    import shapely.geometry
    from odc.geo import GeoboxTiles

    geobox = _coarse_ease_geobox()
    tiles = GeoboxTiles(geobox, (16, 16))
    target_tile_geobox = tiles[10, 8]
    tile = Tile(row=10, col=8, geobox=target_tile_geobox, y_slice=slice(0, 0), x_slice=slice(0, 0))

    wgs84_bounds = target_tile_geobox.extent.to_crs("EPSG:4326").boundingbox
    cx = (wgs84_bounds.left + wgs84_bounds.right) / 2
    cy = (wgs84_bounds.bottom + wgs84_bounds.top) / 2
    polygon = shapely.geometry.box(cx - 0.5, cy - 0.5, cx + 0.5, cy + 0.5)
    gdf_wgs84 = gpd.GeoDataFrame(
        {"REALM": ["Nearctic"], "BIOME_NUM": [1], "ECO_ID": [101]}, geometry=[polygon], crs="EPSG:4326"
    )
    code_to_id = {"REALM": {"Nearctic": 1}, "BIOME_NUM": {1: 1}, "ECO_ID": {101: 1}}

    # Bug reproduction: pass the GeoDataFrame in its native (unreprojected) CRS.
    buggy_ds = EcoregionsSource._rasterize_tile(gdf_wgs84, code_to_id, tile)
    assert int((buggy_ds["realm_id"].values != 0).sum()) == 0

    # Fix: reproject to the target geobox's own CRS first, like
    # _execute_ecoregions_grid does now.
    reprojected = gdf_wgs84.to_crs(geobox.crs)
    fixed_ds = EcoregionsSource._rasterize_tile(reprojected, code_to_id, tile)
    assert int((fixed_ds["realm_id"].values != 0).sum()) > 0
    assert int((fixed_ds["biome_id"].values != 0).sum()) > 0
    assert int((fixed_ds["eco_id"].values != 0).sum()) > 0


def test_rasterize_tile_paints_same_mask_into_all_three_variables(tmp_path):
    """One rasterize() call per polygon is reused for all three id-grids
    (unlike GADM's genuinely-distinct-geometries-per-level case) -- so a
    painted pixel in realm_id must also be painted in biome_id/eco_id."""
    import geopandas as gpd
    import shapely.geometry
    import numpy as np
    from odc.geo import GeoboxTiles

    geobox = _coarse_ease_geobox()
    tiles = GeoboxTiles(geobox, (16, 16))
    target_tile_geobox = tiles[10, 8]
    tile = Tile(row=10, col=8, geobox=target_tile_geobox, y_slice=slice(0, 0), x_slice=slice(0, 0))

    wgs84_bounds = target_tile_geobox.extent.to_crs("EPSG:4326").boundingbox
    cx = (wgs84_bounds.left + wgs84_bounds.right) / 2
    cy = (wgs84_bounds.bottom + wgs84_bounds.top) / 2
    polygon = shapely.geometry.box(cx - 0.5, cy - 0.5, cx + 0.5, cy + 0.5)
    gdf = gpd.GeoDataFrame(
        {"REALM": ["Nearctic"], "BIOME_NUM": [1], "ECO_ID": [101]}, geometry=[polygon], crs="EPSG:4326"
    ).to_crs(geobox.crs)
    code_to_id = {"REALM": {"Nearctic": 1}, "BIOME_NUM": {1: 1}, "ECO_ID": {101: 1}}

    ds = EcoregionsSource._rasterize_tile(gdf, code_to_id, tile)

    realm_mask = ds["realm_id"].values != 0
    biome_mask = ds["biome_id"].values != 0
    eco_mask = ds["eco_id"].values != 0
    assert np.array_equal(realm_mask, biome_mask)
    assert np.array_equal(realm_mask, eco_mask)
    assert int(realm_mask.sum()) > 0


def test_execute_ecoregions_grid_threads_ctx_grid_id_into_target_geobox(tmp_path, monkeypatch):
    # `_execute_ecoregions_grid()` is exercised directly here rather than
    # through a StepTarget with real data, mirroring gadm's analogous test.
    source, ctx = _make(tmp_path, grid_id="ease6933")

    import contextlib

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
    monkeypatch.setattr(type(source), "_dask_client", lambda self: contextlib.nullcontext(_FakeClient()))

    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {"REALM": ["Nearctic"], "BIOME_NUM": [1], "BIOME_NAME": ["x"], "ECO_ID": [101], "ECO_NAME": ["y"]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    vector_path = source._vector_path()
    import os

    os.makedirs(os.path.dirname(vector_path), exist_ok=True)
    gdf.to_file(vector_path, driver="GPKG")
    monkeypatch.setattr(source, "_ensure_vector_file", lambda raw_file: vector_path)

    from src.data.sources.steps import Completion, PipelineStep, StepTarget

    target = StepTarget(
        source_id=source.ID, step=PipelineStep.PREPARE, key="ecoregions_grid",
        output_path=str(tmp_path / "out" / "ecoregions"),
        inputs=("dummy_raw_file",), completion=Completion.MARKER,
    )
    assert source._execute_ecoregions_grid(target) is True
    assert captured["ctx"] is ctx
    assert captured["target_geobox"] is fake_geobox


class _FakeClient:
    dashboard_link = None

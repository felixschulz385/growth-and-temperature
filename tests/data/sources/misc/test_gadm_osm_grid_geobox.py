"""Regression tests for the ease6933 grid-switch correctness fix in
GadmSource/OsmSource: before this fix, both `_execute_grid` methods called
`get_or_create_geobox()` directly (ignoring `ctx.grid_id`) and hardcoded
`latitude`/`longitude` dim names in their zarr-writing helpers -- a
projected canonical geobox (`y`/`x` dims) would have raised a `KeyError`.
"""

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


def test_gadm_create_empty_zarr_uses_y_x_dims_for_ease_geobox(tmp_path):
    from src.data.sources.misc.gadm import GadmSource

    geobox = _coarse_ease_geobox()
    output_path = str(tmp_path / "countries_grid.zarr")

    assert GadmSource._create_empty_gadm_zarr(output_path, geobox, ["GID_0", "GID_1"])

    import xarray as xr

    ds = xr.open_zarr(output_path, consolidated=False)
    assert set(ds["GID_0"].dims) == {"y", "x"}
    assert set(ds["GID_1"].dims) == {"y", "x"}


def test_process_gadm_tiles_needs_gdf_reprojected_to_target_crs_first(tmp_path):
    """Regression test for a confirmed real-data bug: `_process_gadm_tiles`'s
    per-tile overlap pre-filter compares a `tile_polygon` built in the
    *target* geobox's CRS (e.g. EASE6933 projected meters) against each
    GeoDataFrame's geometries via plain shapely `.intersects()`, which never
    reprojects. Left in GADM's native WGS84 lon/lat degrees, that comparison
    is numerically incompatible (~1e7-magnitude meters vs +/-180/+/-90
    degrees) and silently finds ~no overlap for ~every tile -- confirmed on
    real HPC output via src.data.sources.verify: every GID_N level came back
    ~99.98% nodata despite valid input geometries and a clean (no-exception)
    run. `_execute_grid` now reprojects each level's GeoDataFrame to the
    target CRS once, up front, before calling this method.

    Uses a single tile (taken from a real geobox, so it has genuine
    lon/lat<->projected-meters georeferencing) as the *entire* test grid,
    rather than a subregion of a larger one -- selecting one tile's region
    back out of a bigger store via `.sel()` on unrounded tile coordinates
    against the store's own `.round(5)`-written ones is its own source of
    flakiness, orthogonal to what this test is actually about. Checks
    `.notnull()`, not `!= 0`: unwritten/no-unit pixels decode to NaN via
    this variable's `_FillValue=0`, and `NaN != 0` is `True` in IEEE float
    semantics -- `!= 0` alone can't distinguish "genuinely painted" from
    "never written."
    """
    import geopandas as gpd
    import shapely.geometry
    import xarray as xr
    from odc.geo import GeoboxTiles

    from src.data.sources.misc.gadm import GadmSource

    geobox = _coarse_ease_geobox()
    tiles = GeoboxTiles(geobox, (16, 16))
    # A tile comfortably away from the meter-CRS origin in both dims (real
    # bounds ~[-11.0M, -2.4M] to [-10.2M, -1.6M] meters) -- no plausible
    # degree-valued geometry can numerically fall inside it by accident, so
    # the bug can't be masked by a spurious near-origin-tile "hit".
    target_tile = tiles[10, 8]
    single_tile_grid = GeoboxTiles(target_tile, (16, 16))  # tiles the tile itself -> exactly one tile

    # A small WGS84 polygon centered on this tile's own real geographic
    # footprint -- should legitimately rasterize once reprojected to the
    # target CRS, and *only* then.
    wgs84_bounds = target_tile.extent.to_crs("EPSG:4326").boundingbox
    cx = (wgs84_bounds.left + wgs84_bounds.right) / 2
    cy = (wgs84_bounds.bottom + wgs84_bounds.top) / 2
    polygon = shapely.geometry.box(cx - 0.5, cy - 0.5, cx + 0.5, cy + 0.5)
    gdf_wgs84 = gpd.GeoDataFrame({"GID_0": ["AAA"]}, geometry=[polygon], crs="EPSG:4326")
    level_code_to_id = {"GID_0": {"AAA": 1}}

    buggy_path = str(tmp_path / "buggy.zarr")
    assert GadmSource._create_empty_gadm_zarr(buggy_path, target_tile, ["GID_0"])
    # Bug reproduction: pass the GeoDataFrame in its native (unreprojected) CRS.
    assert GadmSource._process_gadm_tiles(single_tile_grid, buggy_path, {"GID_0": gdf_wgs84}, level_code_to_id)
    buggy_ds = xr.open_zarr(buggy_path, consolidated=False)
    assert int(buggy_ds["GID_0"].notnull().sum()) == 0

    fixed_path = str(tmp_path / "fixed.zarr")
    assert GadmSource._create_empty_gadm_zarr(fixed_path, target_tile, ["GID_0"])
    # Fix: reproject to the target geobox's own CRS first, like _execute_grid does now.
    reprojected = {"GID_0": gdf_wgs84.to_crs(geobox.crs)}
    assert GadmSource._process_gadm_tiles(single_tile_grid, fixed_path, reprojected, level_code_to_id)
    fixed_ds = xr.open_zarr(fixed_path, consolidated=False)
    assert int(fixed_ds["GID_0"].notnull().sum()) > 0


def test_gadm_create_empty_zarr_crs_is_readable_after_round_trip(tmp_path):
    """Regression test: `.rio.write_crs()` records the CRS as each data
    variable's own `encoding["grid_mapping"]`, not an attr -- the explicit
    `encoding=` dict `_create_empty_gadm_zarr` passes to `to_zarr()` used to
    silently drop that link (no "grid_mapping" key), leaving the store with
    a valid but undiscoverable CRS. Found via src.data.sources.verify
    catching a real "no CRS found" failure on HPC-produced gadm output."""
    import rioxarray  # noqa: F401 -- registers the .rio accessor
    import xarray as xr

    from src.data.sources.misc.gadm import GadmSource

    geobox = _coarse_ease_geobox()
    output_path = str(tmp_path / "countries_grid.zarr")
    assert GadmSource._create_empty_gadm_zarr(output_path, geobox, ["GID_0"])

    ds = xr.open_zarr(output_path, consolidated=False, decode_coords="all")
    assert ds["GID_0"].encoding.get("grid_mapping") == "spatial_ref"
    assert ds.rio.crs is not None
    assert ds.attrs.get("crs")  # redundant plain-string fallback


def test_gadm_create_empty_zarr_uses_lat_lon_dims_for_legacy_geobox(tmp_path, monkeypatch):
    from src.data.sources.misc.gadm import GadmSource

    class _FakeLegacyGeobox:
        shape = (4, 6)
        crs = "EPSG:4326"
        dimensions = ("latitude", "longitude")

        def __init__(self):
            import numpy as np

            self.coords = {
                "latitude": _FakeCoord(np.linspace(10, 0, 4)),
                "longitude": _FakeCoord(np.linspace(0, 12, 6)),
            }

    class _FakeCoord:
        def __init__(self, values):
            self.values = values

    output_path = str(tmp_path / "countries_grid_legacy.zarr")
    assert GadmSource._create_empty_gadm_zarr(output_path, _FakeLegacyGeobox(), ["GID_0"])

    import xarray as xr

    ds = xr.open_zarr(output_path, consolidated=False)
    assert set(ds["GID_0"].dims) == {"latitude", "longitude"}


def test_gadm_rasterize_levels_threads_ctx_grid_id_into_target_geobox(tmp_path, monkeypatch):
    # `_rasterize_levels()` is called directly by `_execute_prepare`
    # (src/data/sources/misc/gadm.py module docstring) -- exercised
    # directly here rather than through a StepTarget, since there's no
    # separate GRID target.
    gadm, ctx = _make(tmp_path, "gadm", grid_id="ease6933")

    import src.data.common.geobox as geobox_module
    import src.data.sources.misc.gadm as gadm_module

    captured = {}
    fake_geobox = _coarse_ease_geobox()

    def fake_get_target_geobox(passed_ctx):
        captured["ctx"] = passed_ctx
        return fake_geobox

    monkeypatch.setattr(geobox_module, "get_target_geobox", fake_get_target_geobox)
    monkeypatch.setattr(
        gadm_module.GadmSource, "_create_empty_gadm_zarr", staticmethod(lambda *a, **k: True)
    )
    monkeypatch.setattr(
        gadm_module.GadmSource, "_process_gadm_tiles", staticmethod(lambda *a, **k: True)
    )
    monkeypatch.setattr(type(gadm), "_dask_client", lambda self: _NullContext())

    import geopandas as gpd
    from shapely.geometry import Point

    gdf_adm0 = gpd.GeoDataFrame({"GID_0": ["AAA"]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    adm0_path = str(tmp_path / "gadm_levelADM_0_simplified.gpkg")
    gdf_adm0.to_file(adm0_path, driver="GPKG")

    monkeypatch.setattr(gpd, "read_file", lambda path, engine=None: gdf_adm0)

    output_path = str(tmp_path / "out" / "countries_grid.zarr")
    assert gadm._rasterize_levels([adm0_path], output_path) is True
    assert captured["ctx"] is ctx


class _NullContext:
    def __enter__(self):
        return _FakeClient()

    def __exit__(self, *exc):
        return False


class _FakeClient:
    dashboard_link = None


def test_osm_rasterize_uses_y_x_dims_for_ease_geobox(tmp_path, monkeypatch):
    # `_rasterize()` is called directly by `_execute_prepare`
    # (src/data/sources/misc/osm.py module docstring) -- exercised directly
    # here rather than through a StepTarget, since there's no separate GRID
    # target.
    osm, ctx = _make(tmp_path, "osm", grid_id="ease6933")

    import src.data.common.geobox as geobox_module

    fake_geobox = _coarse_ease_geobox()
    monkeypatch.setattr(geobox_module, "get_target_geobox", lambda passed_ctx: fake_geobox)

    import geopandas as gpd
    from shapely.geometry import Polygon

    gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:4326")
    input_path = str(tmp_path / "land_polygons.gpkg")
    gdf.to_file(input_path, driver="GPKG")

    output_path = str(tmp_path / "out" / "land_mask.zarr")
    assert osm._rasterize(input_path, output_path) is True

    import xarray as xr

    ds = xr.open_zarr(output_path, consolidated=False)
    assert set(ds["land_mask"].dims) == {"y", "x"}


def test_osm_rasterize_crs_is_readable_after_round_trip(tmp_path, monkeypatch):
    """Regression test: unlike every other GRID step, OSM's writer relied
    solely on rasterize()'s own georeferencing instead of an explicit
    .rio.write_crs() + "grid_mapping" encoding entry -- see
    write_crs_and_grid_mapping_encoding()'s docstring for why that silently
    leaves `.rio.crs` unreadable on a later open even with valid CRS
    metadata in the store."""
    import rioxarray  # noqa: F401 -- registers the .rio accessor
    import xarray as xr

    osm, ctx = _make(tmp_path, "osm", grid_id="ease6933")

    import src.data.common.geobox as geobox_module

    fake_geobox = _coarse_ease_geobox()
    monkeypatch.setattr(geobox_module, "get_target_geobox", lambda passed_ctx: fake_geobox)

    import geopandas as gpd
    from shapely.geometry import Polygon

    gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:4326")
    input_path = str(tmp_path / "land_polygons.gpkg")
    gdf.to_file(input_path, driver="GPKG")

    output_path = str(tmp_path / "out" / "land_mask.zarr")
    assert osm._rasterize(input_path, output_path) is True

    ds = xr.open_zarr(output_path, consolidated=False, decode_coords="all")
    assert ds["land_mask"].encoding.get("grid_mapping") == "spatial_ref"
    assert ds.rio.crs is not None
    assert ds.attrs.get("crs")  # redundant plain-string fallback

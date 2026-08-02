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

    assert GadmSource._create_empty_gadm_zarr(output_path, geobox, include_subdivisions=True)

    import xarray as xr

    ds = xr.open_zarr(output_path, consolidated=False)
    assert set(ds["country"].dims) == {"y", "x"}
    assert set(ds["subdivision"].dims) == {"y", "x"}


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
    assert GadmSource._create_empty_gadm_zarr(output_path, _FakeLegacyGeobox(), include_subdivisions=False)

    import xarray as xr

    ds = xr.open_zarr(output_path, consolidated=False)
    assert set(ds["country"].dims) == {"latitude", "longitude"}


def test_gadm_execute_grid_threads_ctx_grid_id_into_target_geobox(tmp_path, monkeypatch):
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

    gdf_adm0 = gpd.GeoDataFrame({"GID_0": ["AAA"]}, geometry=[Point(0, 0)])
    adm0_path = str(tmp_path / "adm0.gpkg")
    gdf_adm0.to_file(adm0_path, driver="GPKG")

    monkeypatch.setattr(gpd, "read_file", lambda path, engine=None: gdf_adm0)

    from src.data.sources.steps import Completion, PipelineStep, StepTarget

    target = StepTarget(
        source_id=gadm.ID,
        step=PipelineStep.GRID,
        key="gadm",
        output_path=str(tmp_path / "out" / "countries_grid.zarr"),
        inputs=(adm0_path,),
        completion=Completion.MARKER,
    )
    assert gadm._execute_grid(target) is True
    assert captured["ctx"] is ctx


class _NullContext:
    def __enter__(self):
        return _FakeClient()

    def __exit__(self, *exc):
        return False


class _FakeClient:
    dashboard_link = None


def test_osm_execute_grid_uses_y_x_dims_for_ease_geobox(tmp_path, monkeypatch):
    osm, ctx = _make(tmp_path, "osm", grid_id="ease6933")

    import src.data.common.geobox as geobox_module

    fake_geobox = _coarse_ease_geobox()
    monkeypatch.setattr(geobox_module, "get_target_geobox", lambda passed_ctx: fake_geobox)

    import geopandas as gpd
    from shapely.geometry import Polygon

    gdf = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:4326")
    input_path = str(tmp_path / "land_polygons.gpkg")
    gdf.to_file(input_path, driver="GPKG")

    from src.data.sources.steps import Completion, PipelineStep, StepTarget

    target = StepTarget(
        source_id=osm.ID,
        step=PipelineStep.GRID,
        key="osm",
        output_path=str(tmp_path / "out" / "land_mask.zarr"),
        inputs=(input_path,),
        completion=Completion.MARKER,
    )
    assert osm._execute_grid(target) is True

    import xarray as xr

    ds = xr.open_zarr(target.output_path, consolidated=False)
    assert set(ds["land_mask"].dims) == {"y", "x"}

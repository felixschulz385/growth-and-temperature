"""Grid-geobox regressions mirroring GADM's own
(tests/data/sources/misc/test_gadm_osm_grid_geobox.py): y/x vs
latitude/longitude dim handling, and the CRS-reprojection-before-tiling fix
GADM hit for real (commit f653033) -- carried over here since
`_process_ecoregions_tiles` uses the identical CRS-naive `.intersects()`
prefilter pattern."""

from src.data.common.geobox.canonical import canonical_ease_geobox
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.ecoregions.source import CLASS_COLUMNS, EcoregionsSource


def _make(tmp_path, grid_id="legacy_4326"):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), grid_id=grid_id)
    cfg = SourceConfig.from_dict("ecoregions", {"url": "https://example.test/eco.zip", "name": "eco.zip"})
    cls = registry.load("ecoregions")
    return cls(ctx, cfg), ctx


def _coarse_ease_geobox():
    return canonical_ease_geobox(resolution_m=50_000.0, lat_clip_deg=60.0)


def test_create_empty_zarr_uses_y_x_dims_for_ease_geobox(tmp_path):
    geobox = _coarse_ease_geobox()
    output_path = str(tmp_path / "ecoregions_grid.zarr")

    assert EcoregionsSource._create_empty_ecoregions_zarr(output_path, geobox)

    import xarray as xr

    ds = xr.open_zarr(output_path, consolidated=False)
    for var in CLASS_COLUMNS.values():
        assert set(ds[var].dims) == {"y", "x"}


def test_create_empty_zarr_crs_is_readable_after_round_trip(tmp_path):
    import rioxarray  # noqa: F401 -- registers the .rio accessor
    import xarray as xr

    geobox = _coarse_ease_geobox()
    output_path = str(tmp_path / "ecoregions_grid.zarr")
    assert EcoregionsSource._create_empty_ecoregions_zarr(output_path, geobox)

    ds = xr.open_zarr(output_path, consolidated=False, decode_coords="all")
    assert ds["realm_id"].encoding.get("grid_mapping") == "spatial_ref"
    assert ds.rio.crs is not None
    assert ds.attrs.get("crs")


def test_process_tiles_needs_gdf_reprojected_to_target_crs_first(tmp_path):
    """Same failure mode GADM hit on real data: the per-tile `.intersects()`
    prefilter compares raw shapely geometries against a tile polygon already
    in the target (projected, e.g. EASE6933) CRS -- left in WGS84 degrees,
    that silently finds ~no overlap everywhere."""
    import geopandas as gpd
    import shapely.geometry
    import xarray as xr
    from odc.geo import GeoboxTiles

    geobox = _coarse_ease_geobox()
    tiles = GeoboxTiles(geobox, (16, 16))
    target_tile = tiles[10, 8]
    single_tile_grid = GeoboxTiles(target_tile, (16, 16))

    wgs84_bounds = target_tile.extent.to_crs("EPSG:4326").boundingbox
    cx = (wgs84_bounds.left + wgs84_bounds.right) / 2
    cy = (wgs84_bounds.bottom + wgs84_bounds.top) / 2
    polygon = shapely.geometry.box(cx - 0.5, cy - 0.5, cx + 0.5, cy + 0.5)
    gdf_wgs84 = gpd.GeoDataFrame(
        {"REALM": ["Nearctic"], "BIOME_NUM": [1], "ECO_ID": [101]}, geometry=[polygon], crs="EPSG:4326"
    )
    code_to_id = {"REALM": {"Nearctic": 1}, "BIOME_NUM": {1: 1}, "ECO_ID": {101: 1}}

    buggy_path = str(tmp_path / "buggy.zarr")
    assert EcoregionsSource._create_empty_ecoregions_zarr(buggy_path, target_tile)
    assert EcoregionsSource._process_ecoregions_tiles(single_tile_grid, buggy_path, gdf_wgs84, code_to_id)
    buggy_ds = xr.open_zarr(buggy_path, consolidated=False)
    assert int(buggy_ds["realm_id"].notnull().sum()) == 0

    fixed_path = str(tmp_path / "fixed.zarr")
    assert EcoregionsSource._create_empty_ecoregions_zarr(fixed_path, target_tile)
    reprojected = gdf_wgs84.to_crs(geobox.crs)
    assert EcoregionsSource._process_ecoregions_tiles(single_tile_grid, fixed_path, reprojected, code_to_id)
    fixed_ds = xr.open_zarr(fixed_path, consolidated=False)
    assert int(fixed_ds["realm_id"].notnull().sum()) > 0
    assert int(fixed_ds["biome_id"].notnull().sum()) > 0
    assert int(fixed_ds["eco_id"].notnull().sum()) > 0


def test_process_tiles_paints_same_mask_into_all_three_variables(tmp_path):
    """One rasterize() call per polygon is reused for all three id-grids
    (unlike GADM's genuinely-distinct-geometries-per-level case) -- so a
    painted pixel in realm_id must also be painted in biome_id/eco_id."""
    import geopandas as gpd
    import shapely.geometry
    import xarray as xr
    from odc.geo import GeoboxTiles

    geobox = _coarse_ease_geobox()
    tiles = GeoboxTiles(geobox, (16, 16))
    target_tile = tiles[10, 8]
    single_tile_grid = GeoboxTiles(target_tile, (16, 16))

    wgs84_bounds = target_tile.extent.to_crs("EPSG:4326").boundingbox
    cx = (wgs84_bounds.left + wgs84_bounds.right) / 2
    cy = (wgs84_bounds.bottom + wgs84_bounds.top) / 2
    polygon = shapely.geometry.box(cx - 0.5, cy - 0.5, cx + 0.5, cy + 0.5)
    gdf = gpd.GeoDataFrame(
        {"REALM": ["Nearctic"], "BIOME_NUM": [1], "ECO_ID": [101]}, geometry=[polygon], crs="EPSG:4326"
    ).to_crs(geobox.crs)
    code_to_id = {"REALM": {"Nearctic": 1}, "BIOME_NUM": {1: 1}, "ECO_ID": {101: 1}}

    output_path = str(tmp_path / "out.zarr")
    assert EcoregionsSource._create_empty_ecoregions_zarr(output_path, target_tile)
    assert EcoregionsSource._process_ecoregions_tiles(single_tile_grid, output_path, gdf, code_to_id)

    ds = xr.open_zarr(output_path, consolidated=False)
    realm_mask = ds["realm_id"].notnull() & (ds["realm_id"] != 0)
    biome_mask = ds["biome_id"].notnull() & (ds["biome_id"] != 0)
    eco_mask = ds["eco_id"].notnull() & (ds["eco_id"] != 0)
    assert bool((realm_mask == biome_mask).all())
    assert bool((realm_mask == eco_mask).all())
    assert int(realm_mask.sum()) > 0

"""SpatialProcessor must not let a leaked "spatial_ref" data-variable name
corrupt the real CRS coordinate it writes via `.rio.write_crs()`.

Root cause (found via src.data.sources.verify catching real "no CRS found"
failures on HPC-produced acag/esacci/ntl_harm/eog GRID outputs): every
source's `get_variables_func` callback -- and `process_spatial_standard`'s
own fallback when none is given -- opens a sample zarr without
`decode_coords="all"`, so rioxarray's CRS grid-mapping variable isn't
recognized as a coordinate and leaks into `sample.data_vars.keys()` as if it
were real data. `create_empty_target_zarr` then builds a full
(time, band, y, x) zeros array for it and zarr-encodes it with the
chunked/packed encoding meant for actual data variables -- which then
corrupts the real scalar CRS variable `.rio.write_crs()` writes under the
same name right after, into a metadata-less scalar `.rio.crs` can't read
back.
"""

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401 -- registers the .rio accessor
from odc.geo.geobox import GeoBox

from src.data.common.raster.spatial import (
    SpatialProcessor,
    _NON_DATA_VAR_NAMES,
    reproject_for_tile_overlap,
    write_crs_and_grid_mapping_encoding,
)


def _write_sample_source_zarr(path, year: int, size: int = 8):
    """Mirrors every real source's own annual PREPARE-stage zarr: a data
    variable plus a CRS written via `.rio.write_crs()` before `to_zarr()` --
    which is exactly what makes "spatial_ref" show up in `data_vars` when
    read back without `decode_coords="all"`."""
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
    ds = ds.rio.write_crs(4326)
    ds.to_zarr(str(path), mode="w", zarr_format=3, consolidated=False)


def test_sample_zarr_actually_reproduces_the_leak(tmp_path):
    # Sanity check on the premise itself: reading the fixture back without
    # decode_coords="all" (exactly what every get_variables_func does) must
    # show "spatial_ref" polluting data_vars, or this test doesn't test
    # anything real.
    path = tmp_path / "source_2020.zarr"
    _write_sample_source_zarr(path, 2020)
    sample = xr.open_zarr(str(path), mask_and_scale=False, chunks="auto")
    assert "spatial_ref" in sample.data_vars


def test_process_spatial_standard_output_has_readable_crs(tmp_path, monkeypatch):
    source_path = tmp_path / "source_2020.zarr"
    _write_sample_source_zarr(source_path, 2020)

    target_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.25)
    processor = SpatialProcessor(hpc_root=str(tmp_path))
    monkeypatch.setattr(processor, "get_target_geobox", lambda: target_geobox)

    output_path = tmp_path / "output.zarr"
    success = processor.process_spatial_standard(
        source_files=[str(source_path)],
        output_path=str(output_path),
        years_to_process=[2020],
        year_pattern_func=lambda p: 2020,
    )
    assert success

    result = xr.open_zarr(str(output_path), consolidated=False, decode_coords="all")
    assert result.rio.crs is not None
    assert result.rio.crs.to_epsg() == 4326


def test_process_spatial_standard_output_does_not_have_bogus_spatial_ref_data_var(tmp_path, monkeypatch):
    # The concrete corruption signature observed on real data: a
    # "spatial_ref" data variable shaped like the real data (instead of a
    # scalar CRS coordinate) and encoded with the wrong dtype.
    source_path = tmp_path / "source_2020.zarr"
    _write_sample_source_zarr(source_path, 2020)

    target_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.25)
    processor = SpatialProcessor(hpc_root=str(tmp_path))
    monkeypatch.setattr(processor, "get_target_geobox", lambda: target_geobox)

    output_path = tmp_path / "output.zarr"
    assert processor.process_spatial_standard(
        source_files=[str(source_path)],
        output_path=str(output_path),
        years_to_process=[2020],
        year_pattern_func=lambda p: 2020,
    )

    # Read without decode_coords -- if "spatial_ref" leaked through as a
    # real data variable, it would show up here with the "value" variable's
    # own (time, band, y, x) shape instead of being a 0-d coordinate.
    raw = xr.open_zarr(str(output_path), consolidated=False, mask_and_scale=False)
    assert "value" in raw.data_vars
    if "spatial_ref" in raw.variables:
        assert raw["spatial_ref"].ndim == 0, "spatial_ref must stay a scalar CRS coordinate, not a data array"


def test_create_empty_target_zarr_filters_leaked_spatial_ref_from_variables(tmp_path):
    # Direct unit test of the defensive filter, simulating a get_variables_func
    # that (like every real one) leaked "spatial_ref" into its variable list.
    target_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.25)
    processor = SpatialProcessor(hpc_root=str(tmp_path))

    output_path = tmp_path / "output.zarr"
    success = processor.create_empty_target_zarr(
        output_path=str(output_path),
        target_geobox=target_geobox,
        years=[2020],
        variables=["value", "spatial_ref"],  # "spatial_ref" leaked in, like a real get_variables_func
        dtype="float32",
    )
    assert success

    ds = xr.open_zarr(str(output_path), consolidated=False, decode_coords="all")
    assert list(ds.data_vars) == ["value"]
    assert ds.rio.crs is not None
    assert ds.rio.crs.to_epsg() == 4326


def test_create_empty_target_zarr_preserves_grid_mapping_through_explicit_encoding(tmp_path):
    """The second, more fundamental bug: even with a clean variables list
    (no leaked "spatial_ref"), `.rio.write_crs()` records the CRS link as
    each data variable's own `encoding["grid_mapping"] = "spatial_ref"` --
    not as an attr. The explicit `encoding=` dict passed to `to_zarr()`
    becomes each variable's *entire* encoding (not merged with what
    write_crs() just set), so omitting `grid_mapping` there silently drops
    the link: "spatial_ref" itself stays perfectly valid, but no data
    variable points to it, so `.rio.crs` returns None on every read even
    though the CRS metadata genuinely exists in the store."""
    target_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.25)
    processor = SpatialProcessor(hpc_root=str(tmp_path))

    output_path = tmp_path / "output.zarr"
    assert processor.create_empty_target_zarr(
        output_path=str(output_path),
        target_geobox=target_geobox,
        years=[2020],
        variables=["value"],  # clean list -- no leaked "spatial_ref" this time
        dtype="float32",
    )

    ds = xr.open_zarr(str(output_path), consolidated=False, decode_coords="all")
    assert ds["value"].encoding.get("grid_mapping") == "spatial_ref"
    assert ds.rio.crs is not None
    assert ds.rio.crs.to_epsg() == 4326

    # Redundant fallback (matches gadm's/osm's own defensive pattern):
    # a plain string CRS attr survives independently of grid_mapping.
    assert ds.attrs.get("crs") == "EPSG:4326"


def test_non_data_var_names_includes_spatial_ref():
    assert "spatial_ref" in _NON_DATA_VAR_NAMES


def test_write_crs_and_grid_mapping_encoding_adds_grid_mapping_to_every_entry(tmp_path):
    """The shared helper hand-rolled zarr writers (gadm/snl_mining/glass/
    berman_mining/ecoregions) should use instead of copy-pasting the same
    write_crs()+grid_mapping fix each time."""
    target_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.25)
    ny, nx = target_geobox.shape
    dim_y, dim_x = target_geobox.dimensions
    ds = xr.Dataset(
        {
            "a": ((dim_y, dim_x), np.zeros((ny, nx), dtype="float32")),
            "b": ((dim_y, dim_x), np.zeros((ny, nx), dtype="uint16")),
        },
        coords={dim_y: target_geobox.coords[dim_y].values, dim_x: target_geobox.coords[dim_x].values},
    )
    base_encoding = {"a": {"dtype": "float32"}, "b": {"dtype": "uint16"}}

    ds, encoding = write_crs_and_grid_mapping_encoding(ds, target_geobox, base_encoding)

    assert encoding["a"] == {"dtype": "float32", "grid_mapping": "spatial_ref"}
    assert encoding["b"] == {"dtype": "uint16", "grid_mapping": "spatial_ref"}
    assert ds.attrs.get("crs") == "EPSG:4326"

    output_path = tmp_path / "output.zarr"
    ds.to_zarr(str(output_path), mode="w", encoding=encoding, zarr_format=3, consolidated=False)
    result = xr.open_zarr(str(output_path), consolidated=False, decode_coords="all")
    assert result.rio.crs is not None
    assert result.rio.crs.to_epsg() == 4326
    assert result["a"].encoding.get("grid_mapping") == "spatial_ref"


def test_reproject_for_tile_overlap_reprojects_to_target_crs():
    import geopandas as gpd
    import shapely.geometry

    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[shapely.geometry.box(-1, -1, 1, 1)], crs="EPSG:4326")
    reprojected = reproject_for_tile_overlap(gdf, "EPSG:6933")
    assert reprojected.crs.to_epsg() == 6933
    # Sanity: reprojecting WGS84 degrees to EASE6933 meters changes magnitude.
    assert reprojected.geometry.iloc[0].bounds != gdf.geometry.iloc[0].bounds

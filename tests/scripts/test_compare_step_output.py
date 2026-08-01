"""scripts/compare_step_output.py -- the migration hard gate's diff tool.

docs/design/09-integrated-pipeline.md step 9: this is the tool the hard gate
uses to confirm old-code and new-code output are equivalent before cutover.
The 0-d-coord case here is a real bug found while running that gate for real
against ACAG's PREPARE output -- `xr.Dataset.rio.write_crs()` attaches a
`spatial_ref` coordinate with shape `()` (a CRS grid-mapping scalar, not an
array), and `list(0-d array)` raises "iteration over a 0-d array" rather than
comparing -- so every real raster target with a written CRS would have
crashed the comparison tool itself, not the pipeline being validated.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import compare_step_output as cso  # noqa: E402


def _write_ds(path, *, pm25_value=1.0, spatial_ref_value=4326):
    ds = xr.Dataset(
        {"pm25": (("latitude", "longitude"), np.full((2, 2), pm25_value, dtype="float32"))},
        coords={
            "latitude": [10.0, 20.0],
            "longitude": [30.0, 40.0],
            "spatial_ref": np.array(spatial_ref_value),  # 0-d, mirrors rio.write_crs()
        },
    )
    ds.to_zarr(path, mode="w", zarr_format=3, consolidated=False)


def test_compare_zarr_handles_scalar_coord_when_equivalent(tmp_path):
    old_path, new_path = tmp_path / "old.zarr", tmp_path / "new.zarr"
    _write_ds(old_path)
    _write_ds(new_path)

    assert cso.compare_zarr(old_path, new_path, rtol=1e-6, atol=1e-6) == []


def test_compare_zarr_flags_differing_scalar_coord(tmp_path):
    old_path, new_path = tmp_path / "old.zarr", tmp_path / "new.zarr"
    _write_ds(old_path, spatial_ref_value=4326)
    _write_ds(new_path, spatial_ref_value=3857)

    problems = cso.compare_zarr(old_path, new_path, rtol=1e-6, atol=1e-6)
    assert any("spatial_ref" in p for p in problems)


def test_compare_zarr_flags_differing_values(tmp_path):
    old_path, new_path = tmp_path / "old.zarr", tmp_path / "new.zarr"
    _write_ds(old_path, pm25_value=1.0)
    _write_ds(new_path, pm25_value=2.0)

    problems = cso.compare_zarr(old_path, new_path, rtol=1e-6, atol=1e-6)
    assert any("pm25" in p for p in problems)


def _write_gpkg(path, *, gid0_value="CHE", area_value=1.0, shift=0.0, layer="ADM_0", row_order=None):
    import geopandas as gpd
    from shapely.geometry import box

    rows = [
        {"GID_0": gid0_value, "area": area_value, "geometry": box(0 + shift, 0, 1 + shift, 1)},
        {"GID_0": "DEU", "area": 2.0, "geometry": box(2, 0, 3, 1)},
    ]
    if row_order == "reversed":
        rows = list(reversed(rows))
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf.to_file(path, layer=layer, driver="GPKG")


def test_compare_geopackage_equivalent(tmp_path):
    old_path, new_path = tmp_path / "old.gpkg", tmp_path / "new.gpkg"
    _write_gpkg(old_path)
    _write_gpkg(new_path)

    assert cso.compare_geopackage(old_path, new_path, rtol=1e-6, atol=1e-6) == []


def test_compare_geopackage_ignores_row_order_when_natural_key_present(tmp_path):
    old_path, new_path = tmp_path / "old.gpkg", tmp_path / "new.gpkg"
    _write_gpkg(old_path)
    _write_gpkg(new_path, row_order="reversed")

    assert cso.compare_geopackage(old_path, new_path, rtol=1e-6, atol=1e-6) == []


def test_compare_geopackage_flags_differing_attribute(tmp_path):
    old_path, new_path = tmp_path / "old.gpkg", tmp_path / "new.gpkg"
    _write_gpkg(old_path, area_value=1.0)
    _write_gpkg(new_path, area_value=1.5)

    problems = cso.compare_geopackage(old_path, new_path, rtol=1e-6, atol=1e-6)
    assert any("area" in p for p in problems)


def test_compare_geopackage_flags_differing_geometry(tmp_path):
    old_path, new_path = tmp_path / "old.gpkg", tmp_path / "new.gpkg"
    _write_gpkg(old_path, shift=0.0)
    _write_gpkg(new_path, shift=0.5)

    problems = cso.compare_geopackage(old_path, new_path, rtol=1e-6, atol=1e-6)
    assert any("geometries differ" in p for p in problems)


def test_compare_geopackage_flags_differing_layers(tmp_path):
    old_path, new_path = tmp_path / "old.gpkg", tmp_path / "new.gpkg"
    _write_gpkg(old_path, layer="ADM_0")
    _write_gpkg(new_path, layer="ADM_1")

    problems = cso.compare_geopackage(old_path, new_path, rtol=1e-6, atol=1e-6)
    assert any("layers differ" in p for p in problems)


def test_compare_dispatches_gpkg_by_extension(tmp_path):
    old_path, new_path = tmp_path / "old.gpkg", tmp_path / "new.gpkg"
    _write_gpkg(old_path)
    _write_gpkg(new_path)

    assert cso.compare(old_path, new_path, rtol=1e-6, atol=1e-6) == []


def _write_duckdb(path, *, count_value=5, row_order=None):
    import duckdb

    con = duckdb.connect(str(path))
    try:
        rows = [("m1", 2018, count_value), ("m2", 2019, count_value + 1)]
        if row_order == "reversed":
            rows = list(reversed(rows))
        con.execute("CREATE TABLE adm1_year_counts (property_id VARCHAR, year INTEGER, count INTEGER)")
        con.executemany("INSERT INTO adm1_year_counts VALUES (?, ?, ?)", rows)
    finally:
        con.close()


def test_compare_duckdb_equivalent(tmp_path):
    old_path, new_path = tmp_path / "old.duckdb", tmp_path / "new.duckdb"
    _write_duckdb(old_path)
    _write_duckdb(new_path)

    assert cso.compare_duckdb(old_path, new_path, rtol=1e-6, atol=1e-6) == []


def test_compare_duckdb_ignores_row_order_when_natural_key_present(tmp_path):
    old_path, new_path = tmp_path / "old.duckdb", tmp_path / "new.duckdb"
    _write_duckdb(old_path)
    _write_duckdb(new_path, row_order="reversed")

    assert cso.compare_duckdb(old_path, new_path, rtol=1e-6, atol=1e-6) == []


def test_compare_duckdb_flags_differing_values(tmp_path):
    old_path, new_path = tmp_path / "old.duckdb", tmp_path / "new.duckdb"
    _write_duckdb(old_path, count_value=5)
    _write_duckdb(new_path, count_value=99)

    problems = cso.compare_duckdb(old_path, new_path, rtol=1e-6, atol=1e-6)
    assert any("count" in p for p in problems)


def test_compare_duckdb_flags_differing_tables(tmp_path):
    import duckdb

    old_path, new_path = tmp_path / "old.duckdb", tmp_path / "new.duckdb"
    _write_duckdb(old_path)
    con = duckdb.connect(str(new_path))
    con.execute("CREATE TABLE something_else (x INTEGER)")
    con.close()

    problems = cso.compare_duckdb(old_path, new_path, rtol=1e-6, atol=1e-6)
    assert any("tables differ" in p for p in problems)


def test_compare_dispatches_duckdb_by_extension(tmp_path):
    old_path, new_path = tmp_path / "old.duckdb", tmp_path / "new.duckdb"
    _write_duckdb(old_path)
    _write_duckdb(new_path)

    assert cso.compare(old_path, new_path, rtol=1e-6, atol=1e-6) == []

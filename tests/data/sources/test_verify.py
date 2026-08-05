"""src.data.sources.verify -- the generic GRID-output sanity checker and its
fingerprint-checked cache.

Covers: format dispatch (zarr/geotiff/parquet/unrecognized), the four checks
("has expected variables", "has a CRS", "sample is finite", "sample is
within value_range"), range_vars excluding a variable from the range check,
striding robustness against globally-sparse data (a fixed center-crop would
false-fail these), and the manifest cache (hit/miss/invalidation/force).
"""

import json
import os
import time

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401 -- registers the .rio accessor
import xarray as xr

from src.data.sources.steps import mark_complete
from src.data.sources.verify import VerificationResult, manifest_path, verify_grid_output


def _write_zarr(path, data_vars, *, write_crs=True, crs_attr=False):
    shape = next(iter(data_vars.values())).shape
    ds = xr.Dataset(
        {name: (("y", "x"), arr) for name, arr in data_vars.items()},
        coords={"y": np.arange(shape[0]), "x": np.arange(shape[1])},
    )
    if write_crs:
        ds = ds.rio.write_crs("EPSG:4326")
    elif crs_attr:
        ds.attrs["crs"] = "EPSG:4326"
    ds.to_zarr(path, mode="w", consolidated=False)
    return path


def _write_geotiff(path, arr, *, nodata=None, crs="EPSG:4326"):
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype=arr.dtype, crs=crs, transform=from_origin(0, 0, 1, 1), nodata=nodata,
    ) as dst:
        dst.write(arr, 1)
    return path


# --- format dispatch / missing path -------------------------------------


def test_missing_path_fails(tmp_path):
    result = verify_grid_output(str(tmp_path / "nope.zarr"))
    assert result.ok is False
    assert "does not exist" in result.detail


def test_unrecognized_format_is_existence_only_ok(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"whatever")
    result = verify_grid_output(str(path))
    assert result.ok is True
    assert "existence-only" in result.detail


# --- zarr ------------------------------------------------------------------


def test_zarr_good_output_passes(tmp_path):
    path = str(tmp_path / "good.zarr")
    _write_zarr(path, {"pm25": np.random.uniform(5, 50, size=(20, 20)).astype("float32")})
    result = verify_grid_output(path, expected_vars=("pm25",), value_range=(0, 500))
    assert result.ok is True


def test_zarr_missing_expected_variable_fails(tmp_path):
    path = str(tmp_path / "missingvar.zarr")
    _write_zarr(path, {"pm25": np.random.uniform(5, 50, size=(20, 20)).astype("float32")})
    result = verify_grid_output(path, expected_vars=("nope",))
    assert result.ok is False
    assert "missing expected variable" in result.detail


def test_zarr_no_data_vars_at_all_fails_when_no_expected_vars_given(tmp_path):
    path = str(tmp_path / "empty.zarr")
    ds = xr.Dataset(coords={"y": np.arange(5), "x": np.arange(5)}).rio.write_crs("EPSG:4326")
    ds.to_zarr(path, mode="w", consolidated=False)
    result = verify_grid_output(path)
    assert result.ok is False
    assert "no data variables" in result.detail


def test_zarr_no_crs_fails(tmp_path):
    path = str(tmp_path / "nocrs.zarr")
    _write_zarr(path, {"pm25": np.random.uniform(5, 50, size=(20, 20)).astype("float32")}, write_crs=False)
    result = verify_grid_output(path, expected_vars=("pm25",))
    assert result.ok is False
    assert "no CRS" in result.detail


def test_zarr_crs_as_attrs_string_is_accepted(tmp_path):
    # osm's land_mask.zarr only sets attrs["crs"], never calls .rio.write_crs
    # on the Dataset itself -- the fallback must accept that too.
    path = str(tmp_path / "attrcrs.zarr")
    _write_zarr(
        path, {"land_mask": np.random.randint(0, 2, size=(20, 20)).astype("uint8")},
        write_crs=False, crs_attr=True,
    )
    result = verify_grid_output(path, expected_vars=("land_mask",), value_range=(0, 1))
    assert result.ok is True


def test_zarr_all_nan_sample_fails(tmp_path):
    path = str(tmp_path / "allnan.zarr")
    _write_zarr(path, {"pm25": np.full((20, 20), np.nan, dtype="float32")})
    result = verify_grid_output(path, expected_vars=("pm25",))
    assert result.ok is False
    assert "entirely nodata/NaN" in result.detail


def test_zarr_out_of_range_sample_fails(tmp_path):
    path = str(tmp_path / "badrange.zarr")
    _write_zarr(path, {"pm25": np.full((20, 20), 99999.0, dtype="float32")})
    result = verify_grid_output(path, expected_vars=("pm25",), value_range=(0, 500))
    assert result.ok is False
    assert "outside expected range" in result.detail


def test_zarr_sparse_scattered_data_passes_via_striding(tmp_path):
    # A fixed center-crop would find nothing but zeros here -- striding
    # across the whole extent must still find the scattered real values.
    arr = np.zeros((2000, 2000), dtype="float32")
    arr[::37, ::41] = 5.0
    path = str(tmp_path / "sparse.zarr")
    _write_zarr(path, {"nb_mines_a": arr})
    result = verify_grid_output(path, expected_vars=("nb_mines_a",), value_range=(0, 50))
    assert result.ok is True


def test_zarr_sparse_corner_only_data_passes_via_striding(tmp_path):
    arr = np.zeros((2000, 2000), dtype="float32")
    arr[0:5, 0:5] = 5.0  # only in one corner, would be missed by a naive center-crop
    path = str(tmp_path / "cornersparse.zarr")
    _write_zarr(path, {"nb_mines_a": arr})
    result = verify_grid_output(path, expected_vars=("nb_mines_a",), value_range=(0, 50))
    assert result.ok is True


def test_zarr_range_vars_excludes_variable_from_range_check(tmp_path):
    path = str(tmp_path / "glasslike.zarr")
    _write_zarr(
        path,
        {
            "mean": np.random.uniform(250, 300, size=(20, 20)).astype("float32"),
            "std": np.random.uniform(1, 10, size=(20, 20)).astype("float32"),  # not on the Kelvin scale
        },
    )
    # Without range_vars, "std"'s small values fail the Kelvin range check.
    result_no_exclusion = verify_grid_output(path, expected_vars=("mean", "std"), value_range=(150, 350))
    assert result_no_exclusion.ok is False

    # With range_vars limited to "mean", "std" is still checked for presence
    # and finiteness but not against the Kelvin range.
    result_excluded = verify_grid_output(
        path, expected_vars=("mean", "std"), value_range=(150, 350), range_vars=("mean",)
    )
    assert result_excluded.ok is True


# --- geotiff -----------------------------------------------------------------


def test_geotiff_good_output_passes(tmp_path):
    path = str(tmp_path / "good.tif")
    arr = np.random.uniform(200, 300, size=(20, 20)).astype("float32")
    _write_geotiff(path, arr, nodata=np.nan)
    result = verify_grid_output(path, value_range=(150, 350))
    assert result.ok is True


def test_geotiff_no_crs_fails(tmp_path):
    path = str(tmp_path / "nocrs.tif")
    arr = np.random.uniform(200, 300, size=(20, 20)).astype("float32")
    _write_geotiff(path, arr, nodata=np.nan, crs=None)
    result = verify_grid_output(path)
    assert result.ok is False
    assert "no CRS" in result.detail


def test_geotiff_all_nodata_fails(tmp_path):
    path = str(tmp_path / "allnodata.tif")
    arr = np.full((20, 20), np.nan, dtype="float32")
    _write_geotiff(path, arr, nodata=np.nan)
    result = verify_grid_output(path)
    assert result.ok is False
    assert "entirely nodata/NaN" in result.detail


# --- parquet -----------------------------------------------------------------


def test_table_good_output_passes(tmp_path):
    path = str(tmp_path / "good.parquet")
    pd.DataFrame({"GID_0": [1, 2, 3], "year": [2000, 2001, 2002], "reg_fav": [True, False, True]}).to_parquet(path)
    result = verify_grid_output(path, expected_vars=("GID_0", "year", "reg_fav"))
    assert result.ok is True


def test_table_missing_column_fails(tmp_path):
    path = str(tmp_path / "missingcol.parquet")
    pd.DataFrame({"GID_0": [1, 2, 3]}).to_parquet(path)
    result = verify_grid_output(path, expected_vars=("GID_0", "year"))
    assert result.ok is False
    assert "missing expected column" in result.detail


def test_table_empty_fails(tmp_path):
    path = str(tmp_path / "empty.parquet")
    pd.DataFrame({"GID_0": pd.Series(dtype="int64")}).to_parquet(path)
    result = verify_grid_output(path)
    assert result.ok is False
    assert "zero rows" in result.detail


# --- caching -------------------------------------------------------------


def test_manifest_path_is_sibling_verification_folder_not_inside_store():
    mpath = manifest_path("/data/misc/grid/gadm/countries_grid.zarr")
    assert mpath == "/data/misc/grid/gadm/_verification/countries_grid.zarr.json"


def test_second_call_hits_cache_without_reverifying(tmp_path, monkeypatch):
    import src.data.sources.verify as verify_mod

    path = str(tmp_path / "good.zarr")
    _write_zarr(path, {"pm25": np.random.uniform(5, 50, size=(20, 20)).astype("float32")})

    calls = []
    orig = verify_mod._run_verification

    def spy(*args, **kwargs):
        calls.append(1)
        return orig(*args, **kwargs)

    monkeypatch.setattr(verify_mod, "_run_verification", spy)

    r1 = verify_grid_output(path, expected_vars=("pm25",), value_range=(0, 500))
    assert r1.ok is True
    assert len(calls) == 1
    assert os.path.exists(manifest_path(path))

    r2 = verify_grid_output(path, expected_vars=("pm25",), value_range=(0, 500))
    assert r2 == r1
    assert len(calls) == 1  # not re-run


def test_manifest_content_matches_result(tmp_path):
    path = str(tmp_path / "good.zarr")
    _write_zarr(path, {"pm25": np.random.uniform(5, 50, size=(20, 20)).astype("float32")})
    verify_grid_output(path, expected_vars=("pm25",), value_range=(0, 500))

    with open(manifest_path(path)) as fh:
        data = json.load(fh)
    assert data["ok"] is True
    assert "fingerprint" in data
    assert "checked_at" in data


def test_marker_touch_invalidates_cache(tmp_path, monkeypatch):
    import src.data.sources.verify as verify_mod

    path = str(tmp_path / "good.zarr")
    _write_zarr(path, {"pm25": np.random.uniform(5, 50, size=(20, 20)).astype("float32")})
    mark_complete(path)

    calls = []
    orig = verify_mod._run_verification
    monkeypatch.setattr(verify_mod, "_run_verification", lambda *a, **kw: (calls.append(1), orig(*a, **kw))[1])

    verify_grid_output(path, expected_vars=("pm25",))
    assert len(calls) == 1
    verify_grid_output(path, expected_vars=("pm25",))
    assert len(calls) == 1  # cache hit, marker unchanged

    time.sleep(0.01)
    mark_complete(path)  # simulates a fresh GRID re-run
    verify_grid_output(path, expected_vars=("pm25",))
    assert len(calls) == 2  # fingerprint changed -> cache invalidated


def test_force_bypasses_cache(tmp_path, monkeypatch):
    import src.data.sources.verify as verify_mod

    path = str(tmp_path / "good.zarr")
    _write_zarr(path, {"pm25": np.random.uniform(5, 50, size=(20, 20)).astype("float32")})

    calls = []
    orig = verify_mod._run_verification
    monkeypatch.setattr(verify_mod, "_run_verification", lambda *a, **kw: (calls.append(1), orig(*a, **kw))[1])

    verify_grid_output(path, expected_vars=("pm25",))
    assert len(calls) == 1
    verify_grid_output(path, expected_vars=("pm25",), force=True)
    assert len(calls) == 2


def test_cache_is_scoped_per_verification_parameters_not_just_path(tmp_path):
    # Two callers checking the *same* store with different parameters (e.g.
    # a source's own expected_vars vs. an assembly config's narrower
    # columns) must not reuse each other's cached verdict.
    path = str(tmp_path / "glasslike.zarr")
    _write_zarr(
        path,
        {
            "mean": np.random.uniform(250, 300, size=(20, 20)).astype("float32"),
            "std": np.random.uniform(1, 10, size=(20, 20)).astype("float32"),
        },
    )
    strict = verify_grid_output(path, expected_vars=("mean", "std"), value_range=(150, 350))
    assert strict.ok is False  # "std" fails the Kelvin range

    lenient = verify_grid_output(
        path, expected_vars=("mean", "std"), value_range=(150, 350), range_vars=("mean",)
    )
    assert lenient.ok is True  # different range_vars -> must not reuse the strict cache entry

    # Re-checking with the original strict parameters must still see the
    # original (failing) result, not the lenient one.
    strict_again = verify_grid_output(path, expected_vars=("mean", "std"), value_range=(150, 350))
    assert strict_again.ok is False


def test_manifest_write_failure_does_not_break_verification(tmp_path, monkeypatch):
    # Caching is a speed optimization -- if the manifest can't be written
    # (e.g. read-only filesystem), verification must still return a result.
    import src.data.sources.verify as verify_mod

    path = str(tmp_path / "good.zarr")
    _write_zarr(path, {"pm25": np.random.uniform(5, 50, size=(20, 20)).astype("float32")})

    def raise_makedirs(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(verify_mod.os, "makedirs", raise_makedirs)
    result = verify_grid_output(path, expected_vars=("pm25",), value_range=(0, 500))
    assert result.ok is True


def test_verification_result_is_frozen_and_comparable():
    a = VerificationResult(True, "ok")
    b = VerificationResult(True, "ok")
    assert a == b
    with pytest.raises(Exception):
        a.ok = False

"""Assembly's hard verification gate: `validate_assembly_config` runs
`verify_grid_output` against every dataset path (not just an existence
check) and turns failures into hard errors; `run_assembly` raises rather
than silently `return`ing when those errors are present -- see
src/data/assemble/config.py and src/data/assemble/workflow.py.

Before this gate existed, a missing dataset path only logged a warning and
a broken/empty dataset was never checked at all -- `run_assembly` would
still exit 0 (confirmed via src/cli/main.py, which only turns exceptions
into a non-zero exit code)."""

import numpy as np
import pytest
import rioxarray  # noqa: F401 -- registers the .rio accessor
import xarray as xr

from src.data.assemble.config import validate_assembly_config
from src.data.assemble.workflow import run_assembly


def _write_good_zarr(path):
    ds = xr.Dataset(
        {"pm25": (("y", "x"), np.random.uniform(5, 50, size=(20, 20)).astype("float32"))},
        coords={"y": np.arange(20), "x": np.arange(20)},
    ).rio.write_crs("EPSG:4326")
    ds.to_zarr(path, mode="w", consolidated=False)


def _write_all_nan_zarr(path):
    ds = xr.Dataset(
        {"pm25": (("y", "x"), np.full((20, 20), np.nan, dtype="float32"))},
        coords={"y": np.arange(20), "x": np.arange(20)},
    ).rio.write_crs("EPSG:4326")
    ds.to_zarr(path, mode="w", consolidated=False)


def _config(tmp_path, dataset_path, **dataset_overrides):
    return {
        "output_path": str(tmp_path / "out"),
        "datasets": {"pm25_ds": {"path": dataset_path, **dataset_overrides}},
    }


# --- validate_assembly_config -------------------------------------------


def test_missing_dataset_path_is_a_hard_error(tmp_path):
    errors = validate_assembly_config(_config(tmp_path, str(tmp_path / "nope.zarr")))
    assert any("does not exist" in e for e in errors)


def test_valid_dataset_output_passes_with_no_errors(tmp_path):
    path = str(tmp_path / "good.zarr")
    _write_good_zarr(path)
    errors = validate_assembly_config(_config(tmp_path, path))
    assert errors == []


def test_all_nan_dataset_output_is_a_hard_error(tmp_path):
    path = str(tmp_path / "bad.zarr")
    _write_all_nan_zarr(path)
    errors = validate_assembly_config(_config(tmp_path, path))
    assert any("failed output verification" in e for e in errors)
    assert any("nodata/NaN" in e for e in errors)


def test_dataset_columns_config_is_used_as_expected_vars(tmp_path):
    path = str(tmp_path / "good.zarr")
    _write_good_zarr(path)
    errors = validate_assembly_config(_config(tmp_path, path, columns=["not_a_real_column"]))
    assert any("missing expected variable" in e for e in errors)


def test_dataset_missing_path_field_is_still_reported(tmp_path):
    config = {"output_path": str(tmp_path / "out"), "datasets": {"pm25_ds": {}}}
    errors = validate_assembly_config(config)
    assert any("missing required 'path' field" in e for e in errors)


# --- run_assembly raises instead of silently returning -------------------


def test_run_assembly_raises_on_missing_dataset_path(tmp_path):
    config = _config(tmp_path, str(tmp_path / "nope.zarr"))
    with pytest.raises(ValueError, match="Assembly configuration invalid"):
        run_assembly(config)


def test_run_assembly_raises_on_failed_output_verification(tmp_path):
    path = str(tmp_path / "bad.zarr")
    _write_all_nan_zarr(path)
    config = _config(tmp_path, path)
    with pytest.raises(ValueError, match="Assembly configuration invalid"):
        run_assembly(config)

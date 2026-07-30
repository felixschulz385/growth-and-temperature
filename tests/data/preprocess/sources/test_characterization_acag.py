"""Characterization tests for the *current* ACAGPreprocessor (pre-migration).

docs/design/09-integrated-pipeline.md §10 step 0: before any source moves into
the new src/data/sources/ + src/data/pipeline/ shape, its existing behaviour
must be pinned down with a test, so the migration can be checked against these
exact values rather than trusted from reading a diff. This file intentionally
imports the OLD src.data.preprocess.sources.acag module and must keep passing
unmodified until that module is deleted in migration step 10 -- if a "cleanup"
change to the old code breaks this file, that is a sign the migration oracle
itself moved, not that the test is stale.
"""

import os

import pandas as pd
import pytest

from src.data.preprocess.sources.acag import ACAGPreprocessor


def _write_index(index_dir, data_path, rows):
    safe = data_path.replace("/", "_").replace("\\", "_")
    os.makedirs(index_dir, exist_ok=True)
    path = os.path.join(index_dir, f"parquet_{safe}.parquet")
    pd.DataFrame(rows).to_parquet(path)
    return path


def _make_preprocessor(tmp_path, stage, year_range=(2019, 2021), rows=None):
    hpc_root = str(tmp_path)
    data_path = "acag/pm25"
    if rows is None:
        rows = [
            {"relative_path": "GL/Annual/V6GL02.04.CNNPM25.GL.201901-201912.nc", "status_category": "completed"},
            {"relative_path": "GL/Annual/V6GL02.04.CNNPM25.GL.202001-202012.nc4", "status_category": "completed"},
            {"relative_path": "GL/Annual/V6GL02.04.CNNPM25.GL.202101-202112.nc", "status_category": "completed"},
            # not "completed" -- must be excluded
            {"relative_path": "GL/Annual/V6GL02.04.CNNPM25.GL.202201-202212.nc", "status_category": "pending"},
        ]
    _write_index(os.path.join(hpc_root, "hpc_data_index"), data_path, rows)
    return ACAGPreprocessor(
        stage=stage,
        year_range=list(year_range),
        hpc_target=hpc_root,
        data_path=data_path,
    )


def test_hpc_output_path_annual(tmp_path):
    pp = _make_preprocessor(tmp_path, "annual")
    assert pp.get_hpc_output_path("annual") == os.path.join(str(tmp_path), "acag/pm25", "processed", "stage_1")


def test_hpc_output_path_spatial(tmp_path):
    pp = _make_preprocessor(tmp_path, "spatial")
    assert pp.get_hpc_output_path("spatial") == os.path.join(str(tmp_path), "acag/pm25", "processed", "stage_2")


def test_hpc_output_path_unknown_stage_raises(tmp_path):
    pp = _make_preprocessor(tmp_path, "annual")
    with pytest.raises(ValueError):
        pp.get_hpc_output_path("bogus")


def test_annual_targets_one_per_year_prefers_nc4(tmp_path):
    pp = _make_preprocessor(tmp_path, "annual")
    targets = pp.get_preprocessing_targets("annual", year_range=(2019, 2021))

    assert [t["year"] for t in targets] == [2019, 2020, 2021]
    for t in targets:
        assert t["stage"] == "annual"
        assert t["dependencies"] == []
        assert t["metadata"]["source_type"] == "acag"
        assert t["output_path"] == os.path.join(
            str(tmp_path), "acag/pm25", "processed", "stage_1", f"{t['year']}.zarr"
        )
    # 2020 has both a .nc and a .nc4 candidate in the raw inventory shape --
    # .nc4 must win (this repo's stated preference, acag.py:259-264).
    assert targets[1]["source_files"] == ["GL/Annual/V6GL02.04.CNNPM25.GL.202001-202012.nc4"]


def test_annual_targets_excludes_incomplete_and_out_of_range(tmp_path):
    pp = _make_preprocessor(tmp_path, "annual", year_range=(2019, 2020))
    targets = pp.get_preprocessing_targets("annual", year_range=(2019, 2020))
    years = [t["year"] for t in targets]
    assert years == [2019, 2020]  # 2021 excluded by year_range, 2022 excluded by status


def test_annual_targets_empty_when_index_missing(tmp_path):
    pp = ACAGPreprocessor(
        stage="annual",
        year_range=[2019, 2021],
        hpc_target=str(tmp_path),
        data_path="acag/pm25_never_indexed",
    )
    assert pp.get_preprocessing_targets("annual") == []


def test_spatial_target_lists_available_annual_zarrs_and_missing_years(tmp_path):
    pp = _make_preprocessor(tmp_path, "spatial", year_range=(2019, 2022))
    annual_dir = pp.get_hpc_output_path("annual")
    os.makedirs(annual_dir, exist_ok=True)
    for year in (2019, 2020):
        os.makedirs(os.path.join(annual_dir, f"{year}.zarr"))

    targets = pp.get_preprocessing_targets("spatial", year_range=(2019, 2022))

    assert len(targets) == 1
    target = targets[0]
    assert target["stage"] == "spatial"
    assert sorted(target["metadata"]["years_available"]) == [2019, 2020]
    assert target["metadata"]["missing_years"] == [2021, 2022]
    assert target["output_path"] == os.path.join(
        pp.get_hpc_output_path("spatial"), "acag_pm25_timeseries_reprojected.zarr"
    )


def test_get_preprocessing_targets_unknown_stage_raises_and_is_caught(tmp_path):
    # get_preprocessing_targets wraps the ValueError from an unknown stage in a
    # bare except/return [] -- pinning that (arguably-unfortunate) behaviour so
    # the migration doesn't silently start raising where it used to swallow.
    pp = _make_preprocessor(tmp_path, "annual")
    assert pp.get_preprocessing_targets("bogus") == []

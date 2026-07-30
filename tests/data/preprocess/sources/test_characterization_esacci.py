"""Characterization tests for the *current* ESACCIPreprocessor (pre-migration).

Mirrors tests/data/preprocess/sources/test_characterization_acag.py's approach
(docs/design/09-integrated-pipeline.md §10 step 0) -- must keep passing
unmodified against the OLD src.data.preprocess.sources.esacci module until
that module is deleted in migration step 10.
"""

import os

import pandas as pd
import pytest

from src.data.preprocess.sources.esacci import ESACCIPreprocessor


def _write_index(index_dir, data_path, rows):
    safe = data_path.replace("/", "_").replace("\\", "_")
    os.makedirs(index_dir, exist_ok=True)
    path = os.path.join(index_dir, f"parquet_{safe}.parquet")
    pd.DataFrame(rows).to_parquet(path)


def _make_preprocessor(tmp_path, stage, year_range=(2019, 2021), rows=None):
    hpc_root = str(tmp_path)
    data_path = "esacci/landcover"
    if rows is None:
        rows = [
            {"relative_path": "2019/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2019-v2.0.7.nc", "status_category": "completed"},
            {"relative_path": "2020/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2020-v2.0.7.nc4", "status_category": "completed"},
            {"relative_path": "2021/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2021-v2.0.7.nc", "status_category": "completed"},
            {"relative_path": "2022/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2022-v2.0.7.nc", "status_category": "pending"},
        ]
    _write_index(os.path.join(hpc_root, "hpc_data_index"), data_path, rows)
    return ESACCIPreprocessor(
        stage=stage,
        year_range=list(year_range),
        hpc_target=hpc_root,
        data_path=data_path,
    )


def test_default_variables_to_keep(tmp_path):
    pp = _make_preprocessor(tmp_path, "annual")
    assert pp.variables_to_keep == ["lccs_class"]


def test_hpc_output_path_annual_and_spatial(tmp_path):
    pp = _make_preprocessor(tmp_path, "annual")
    assert pp.get_hpc_output_path("annual") == os.path.join(str(tmp_path), "esacci/landcover", "processed", "stage_1")
    assert pp.get_hpc_output_path("spatial") == os.path.join(str(tmp_path), "esacci/landcover", "processed", "stage_2")


def test_annual_targets_one_per_year_prefers_nc4(tmp_path):
    pp = _make_preprocessor(tmp_path, "annual")
    targets = pp.get_preprocessing_targets("annual", year_range=(2019, 2021))

    assert [t["year"] for t in targets] == [2019, 2020, 2021]
    for t in targets:
        assert t["metadata"]["source_type"] == "esacci"
        assert t["output_path"] == os.path.join(
            str(tmp_path), "esacci/landcover", "processed", "stage_1", f"{t['year']}.zarr"
        )
    assert targets[1]["source_files"] == ["2020/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2020-v2.0.7.nc4"]


def test_spatial_target_lists_available_annual_zarrs(tmp_path):
    pp = _make_preprocessor(tmp_path, "spatial", year_range=(2019, 2022))
    annual_dir = pp.get_hpc_output_path("annual")
    os.makedirs(annual_dir, exist_ok=True)
    for year in (2019, 2020):
        os.makedirs(os.path.join(annual_dir, f"{year}.zarr"))

    targets = pp.get_preprocessing_targets("spatial", year_range=(2019, 2022))

    assert len(targets) == 1
    target = targets[0]
    assert sorted(target["metadata"]["years_available"]) == [2019, 2020]
    assert target["metadata"]["missing_years"] == [2021, 2022]
    assert target["output_path"] == os.path.join(
        pp.get_hpc_output_path("spatial"), "esacci_lc_timeseries_reprojected.zarr"
    )

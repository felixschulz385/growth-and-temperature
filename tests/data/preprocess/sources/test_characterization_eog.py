"""Characterization tests for the *current* EOGPreprocessor (pre-migration).

docs/design/09-integrated-pipeline.md §10 step 0.

**Real bug pinned deliberately, not a quirk**: `_generate_annual_targets`
calls `self._extract_year_from_path(...)` and `self._select_best_file_for_year(...)`
(eog.py:295, eog.py:303) -- neither method is defined anywhere in
`EOGPreprocessor`, nor inherited from `AbstractPreprocessor`. Every call to
`get_preprocessing_targets("annual", ...)` therefore raises `AttributeError`
inside the try/except in `get_preprocessing_targets`, which logs and returns
`[]`. **EOG's annual/PREPARE target generation has never worked** -- verified
by direct execution, not inferred from reading the code. This is fixed in the
migration (`src/data/sources/eog.py`), with the fix isolated to that file and
called out explicitly rather than silently ported -- see that module's
docstring. This test pins the CURRENT (broken) behaviour so the bug's
existence stays verifiable even after the old module is deleted in step 10.
"""

import os

import pandas as pd

from src.data.preprocess.sources.eog import EOGPreprocessor


def _write_index(index_dir, data_path, rows):
    safe = data_path.replace("/", "_").replace("\\", "_")
    os.makedirs(index_dir, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(index_dir, f"parquet_{safe}.parquet"))


def _make_preprocessor(tmp_path, stage, source_type="viirs", year_range=(2019, 2021), rows=None):
    hpc_root = str(tmp_path)
    data_path = {"dmsp": "eog/dmsp", "viirs": "eog/viirs", "dvnl": "eog/dvnl"}[source_type]
    base_url = {
        "dmsp": "https://eogdata.mines.edu/wwwdata/dmsp/v4composites_rearrange/",
        "viirs": "https://eogdata.mines.edu/nighttime_light/annual/v21/",
        "dvnl": "https://eogdata.mines.edu/wwwdata/viirs_products/dvnl/",
    }[source_type]
    if rows is None:
        rows = [{"relative_path": "F182019.v4d_web.stable_lights.avg_vis.tif", "status_category": "completed"}]
    _write_index(os.path.join(hpc_root, "hpc_data_index"), data_path, rows)
    return EOGPreprocessor(stage=stage, year_range=list(year_range), hpc_target=hpc_root, data_path=data_path, base_url=base_url)


def test_source_type_derivation_from_data_path(tmp_path):
    assert _make_preprocessor(tmp_path, "annual", "dmsp").source_type == "dmsp"
    assert _make_preprocessor(tmp_path, "annual", "viirs").source_type == "viirs_annual"
    assert _make_preprocessor(tmp_path, "annual", "dvnl").source_type == "viirs_dvnl"


def test_default_resampling_is_sum(tmp_path):
    pp = _make_preprocessor(tmp_path, "annual")
    assert pp.resampling == "sum"


def test_hpc_output_path(tmp_path):
    pp = _make_preprocessor(tmp_path, "annual", "viirs")
    assert pp.get_hpc_output_path("annual") == os.path.join(str(tmp_path), "eog/viirs", "processed", "stage_1")
    assert pp.get_hpc_output_path("spatial") == os.path.join(str(tmp_path), "eog/viirs", "processed", "stage_2")


def test_annual_targets_are_always_empty_due_to_the_missing_method_bug(tmp_path):
    pp = _make_preprocessor(tmp_path, "annual", "viirs")
    # BUG (pinned, see module docstring): should be [{"year": 2019, ...}] but
    # AttributeError inside _generate_annual_targets is swallowed -> [].
    assert pp.get_preprocessing_targets("annual", year_range=(2019, 2021)) == []


def test_spatial_target_lists_available_annual_zarrs(tmp_path):
    pp = _make_preprocessor(tmp_path, "spatial", "viirs", year_range=(2019, 2022))
    annual_dir = pp.get_hpc_output_path("annual")
    os.makedirs(annual_dir, exist_ok=True)
    for year in (2019, 2020):
        os.makedirs(os.path.join(annual_dir, f"{year}.zarr"))

    targets = pp.get_preprocessing_targets("spatial", year_range=(2019, 2022))
    assert len(targets) == 1
    target = targets[0]
    assert sorted(target["metadata"]["years_available"]) == [2019, 2020]
    assert target["metadata"]["missing_years"] == [2021, 2022]
    assert target["output_path"] == f"{pp.get_hpc_output_path('spatial')}/viirs_annual_timeseries_reprojected.zarr"

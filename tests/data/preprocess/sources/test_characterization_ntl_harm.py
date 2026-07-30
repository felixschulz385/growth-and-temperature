"""Characterization tests for the *current* NTLHarmPreprocessor (pre-migration).

docs/design/09-integrated-pipeline.md §10 step 0. Two quirks specific to this
source, pinned deliberately rather than "fixed" during migration (a
migration must reproduce behaviour, not improve it silently):

1. `_generate_annual_targets` iterates a plain dict of `{year: [files]}` in
   *insertion* order (first file's year first), NOT sorted by year -- unlike
   acag/esacci's `_gen_annual_targets`, which explicitly does
   `for year in sorted(files_by_year)`.
2. `_select_best_file_for_year` prefers `.tif > .zip > .tar.gz > .gz`
   (opposite direction from acag/esacci's nc4-over-nc preference).
"""

import os

import pandas as pd

from src.data.preprocess.sources.ntl_harm import NTLHarmPreprocessor


def _write_index(index_dir, data_path, rows):
    safe = data_path.replace("/", "_").replace("\\", "_")
    os.makedirs(index_dir, exist_ok=True)
    path = os.path.join(index_dir, f"parquet_{safe}.parquet")
    pd.DataFrame(rows).to_parquet(path)


def _make_preprocessor(tmp_path, stage, year_range=(2019, 2021), rows=None):
    hpc_root = str(tmp_path)
    data_path = "ntl_harm/harmonized"
    if rows is None:
        rows = [
            {"relative_path": "harmonized_2021.tif", "status_category": "completed"},
            {"relative_path": "harmonized_2019.tif", "status_category": "completed"},
            {"relative_path": "harmonized_2020.zip", "status_category": "completed"},
            {"relative_path": "harmonized_2020.tif", "status_category": "completed"},
            {"relative_path": "harmonized_2022.tif", "status_category": "pending"},
        ]
    _write_index(os.path.join(hpc_root, "hpc_data_index"), data_path, rows)
    return NTLHarmPreprocessor(stage=stage, year_range=list(year_range), hpc_target=hpc_root, data_path=data_path)


def test_default_resampling_is_sum(tmp_path):
    pp = _make_preprocessor(tmp_path, "annual")
    assert pp.resampling == "sum"


def test_resampling_overridable(tmp_path):
    hpc_root = str(tmp_path)
    _write_index(os.path.join(hpc_root, "hpc_data_index"), "ntl_harm/harmonized", [])
    pp = NTLHarmPreprocessor(
        stage="annual", year_range=[2019, 2019], hpc_target=hpc_root, data_path="ntl_harm/harmonized",
        resampling="nearest",
    )
    assert pp.resampling == "nearest"


def test_hpc_output_path(tmp_path):
    pp = _make_preprocessor(tmp_path, "annual")
    assert pp.get_hpc_output_path("annual") == os.path.join(str(tmp_path), "ntl_harm/harmonized", "processed", "stage_1")
    assert pp.get_hpc_output_path("spatial") == os.path.join(str(tmp_path), "ntl_harm/harmonized", "processed", "stage_2")


def test_annual_targets_are_in_file_insertion_order_not_sorted_by_year(tmp_path):
    # Quirk (1): 2021 appears first in the index rows above, so it must be
    # the first target -- NOT sorted ascending like acag/esacci.
    pp = _make_preprocessor(tmp_path, "annual")
    targets = pp.get_preprocessing_targets("annual", year_range=(2019, 2021))
    assert [t["year"] for t in targets] == [2021, 2019, 2020]


def test_annual_target_prefers_tif_over_zip(tmp_path):
    # Quirk (2): .tif beats .zip (opposite preference direction from acag's nc4-over-nc).
    pp = _make_preprocessor(tmp_path, "annual")
    targets = pp.get_preprocessing_targets("annual", year_range=(2019, 2021))
    by_year = {t["year"]: t for t in targets}
    assert by_year[2020]["source_files"] == ["harmonized_2020.tif"]
    assert by_year[2020]["metadata"]["total_candidates"] == 2


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
    assert target["output_path"] == f"{pp.get_hpc_output_path('spatial')}/ntl_harm_timeseries_reprojected.zarr"

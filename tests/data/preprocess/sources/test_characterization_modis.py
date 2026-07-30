"""Characterization tests for the *current* MODISPreprocessor (pre-migration).

docs/design/09-integrated-pipeline.md §10 step 0. MODIS has no FETCH/download
side (it streams via STAC inside "annual"), so this covers only
get_hpc_output_path/target-generation/get_transfer_units -- already the
closest-to-target-shape source in the repo.
"""

import os

from src.data.preprocess.sources.modis import MODISPreprocessor


def _make_preprocessor(tmp_path, stage, tiles=("h18v04", "h20v08"), year_range=(2019, 2020), **extra):
    return MODISPreprocessor(
        stage=stage, year_range=list(year_range), hpc_target=str(tmp_path),
        tiles=list(tiles), **extra,
    )


def test_hpc_output_path_uses_ease6933_suffix_for_spatial(tmp_path):
    pp = _make_preprocessor(tmp_path, "annual")
    assert pp.get_hpc_output_path("annual") == os.path.join(str(tmp_path), "modis/21A2", "processed", "stage_1")
    assert pp.get_hpc_output_path("spatial") == os.path.join(str(tmp_path), "modis/21A2", "processed", "stage_2_ease6933")


def test_data_path_defaults_to_product_specific(tmp_path):
    pp = _make_preprocessor(tmp_path, "annual", product="11A1")
    assert pp.data_path == "modis/11A1"


def test_annual_targets_one_per_tile_year(tmp_path):
    pp = _make_preprocessor(tmp_path, "annual", tiles=("h18v04", "h20v08"), year_range=(2019, 2020))
    targets = pp.get_preprocessing_targets("annual")
    keys = {(t["year"], t["tile"]) for t in targets}
    assert keys == {(2019, "h18v04"), (2019, "h20v08"), (2020, "h18v04"), (2020, "h20v08")}
    sample = next(t for t in targets if t["year"] == 2019 and t["tile"] == "h18v04")
    assert sample["output_path"] == os.path.join(pp.get_hpc_output_path("annual"), "2019", "h18v04.tif")


def test_spatial_targets_only_for_years_with_stage1_output(tmp_path):
    pp = _make_preprocessor(tmp_path, "spatial", year_range=(2019, 2020))
    stage1 = pp.get_hpc_output_path("annual")
    year_dir = os.path.join(stage1, "2019")
    os.makedirs(year_dir, exist_ok=True)
    open(os.path.join(year_dir, "h18v04.tif"), "w").close()

    targets = pp.get_preprocessing_targets("spatial")
    assert len(targets) == 1
    assert targets[0]["year"] == 2019
    assert targets[0]["source_files"] == [os.path.join(year_dir, "h18v04.tif")]
    assert targets[0]["output_path"] == os.path.join(
        pp.get_hpc_output_path("spatial"), "modis_21A2_timeseries_reprojected.zarr"
    )


def test_transfer_units_one_per_tile_year_file(tmp_path):
    pp = _make_preprocessor(tmp_path, "annual", year_range=(2019, 2020))
    stage1 = pp.get_hpc_output_path("annual")
    for year, tile in [("2019", "h18v04"), ("2019", "h20v08")]:
        d = os.path.join(stage1, year)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, f"{tile}.tif"), "w").close()

    units = pp.get_transfer_units("annual")
    assert {u["unit_id"] for u in units} == {"2019/h18v04.tif", "2019/h20v08.tif"}
    assert all(u["remote_path"] == os.path.relpath(u["local_path"], str(tmp_path)) for u in units)


def test_transfer_units_spatial_falls_back_to_single_unit_default(tmp_path):
    pp = _make_preprocessor(tmp_path, "spatial", year_range=(2019, 2020))
    units = pp.get_transfer_units("spatial")
    assert len(units) == 1
    assert units[0]["unit_id"] == "spatial"

"""Characterization tests for the *current* GlassPreprocessor (pre-migration).

docs/design/09-integrated-pipeline.md §10 step 0. Covers the testable "plan"
surface only (target generation, filename parsing, output paths) -- the
heavy raster-processing methods (`_process_file_group_hpc`,
`_calculate_statistics`, `_process_years_chunked`, ...) have no fast unit-test
surface without real GLASS HDF fixtures and are ported mechanically in
src/data/sources/glass.py without new tests here, consistent with how this
migration has treated every source's heavy compute step.

Note: `self.data_source` ("MODIS"/"AVHRR") is derived from the config's
`type` string containing "avhrr"/"modis" (glass.py:126-132) -- NOT from
`data_path` or `base_url` -- and if neither substring is present, the
attribute is silently never set at all (a latent AttributeError waiting to
happen on the very next line, `self.path_prefix = ... if self.data_source ==
'MODIS' else ...` -- this only doesn't blow up today because both real
call sites' `type:` values are literally "glass_modis"/"glass_avhrr").
"""

import os

import pandas as pd

from src.data.preprocess.sources.glass import GlassPreprocessor


def _write_index(index_dir, data_path, rows):
    safe = data_path.replace("/", "_").replace("\\", "_")
    os.makedirs(index_dir, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(index_dir, f"parquet_{safe}.parquet"))


def _make_preprocessor(tmp_path, kind, stage="annual", year_range=(2019, 2021), rows=None, **extra):
    hpc_root = str(tmp_path)
    data_path = {"MODIS": "glass/LST/MODIS/Daily/1KM", "AVHRR": "glass/LST/AVHRR/0.05D"}[kind]
    type_str = {"MODIS": "glass_modis", "AVHRR": "glass_avhrr"}[kind]
    base_url = {
        "MODIS": "https://glass.hku.hk/archive/LST/MODIS/Daily/1KM/",
        "AVHRR": "https://glass.hku.hk/archive/LST/AVHRR/0.05D/",
    }[kind]
    if rows is None:
        if kind == "MODIS":
            rows = [
                {"relative_path": "GLASS06A01.V01.A2019055.h25v06.2022021.hdf", "status_category": "completed"},
                {"relative_path": "GLASS06A01.V01.A2019056.h25v06.2022021.hdf", "status_category": "completed"},
                {"relative_path": "GLASS06A01.V01.A2020001.h26v06.2022021.hdf", "status_category": "completed"},
            ]
        else:
            rows = [
                {"relative_path": "GLASS08B31.V40.A2019001.2021259.hdf", "status_category": "completed"},
                {"relative_path": "GLASS08B31.V40.A2020001.2021259.hdf", "status_category": "completed"},
            ]
    _write_index(os.path.join(hpc_root, "hpc_data_index"), data_path, rows)
    return GlassPreprocessor(
        stage=stage, year_range=list(year_range), hpc_target=hpc_root, data_path=data_path,
        type=type_str, base_url=base_url, **extra,
    )


def test_data_source_kind_derived_from_type_string(tmp_path):
    assert _make_preprocessor(tmp_path, "MODIS").data_source == "MODIS"
    assert _make_preprocessor(tmp_path, "AVHRR").data_source == "AVHRR"


def test_hpc_output_path_uses_path_prefix_not_data_path(tmp_path):
    # get_hpc_output_path uses self.path_prefix (a MODIS/AVHRR constant),
    # NOT self.data_path -- these can legitimately differ, e.g. if data_path
    # is overridden for indexing purposes while output still lands under the
    # fixed path_prefix.
    hpc_root = str(tmp_path)
    _write_index(os.path.join(hpc_root, "hpc_data_index"), "glass/some/other/path", [])
    pp = GlassPreprocessor(
        stage="annual", year_range=[2019, 2021], hpc_target=hpc_root,
        data_path="glass/some/other/path", type="glass_modis",
        base_url="https://glass.hku.hk/archive/LST/MODIS/Daily/1KM/",
    )
    assert pp.get_hpc_output_path("annual") == os.path.join(
        hpc_root, "glass/LST/MODIS/Daily/1KM/", "processed", "stage_1"
    )


def test_modis_annual_targets_grouped_by_year_and_grid_cell(tmp_path):
    pp = _make_preprocessor(tmp_path, "MODIS")
    targets = pp.get_preprocessing_targets("annual", year_range=(2019, 2021))

    keyed = {(t["year"], t["grid_cell"]): t for t in targets}
    assert set(keyed) == {(2019, "h25v06"), (2020, "h26v06")}
    assert sorted(keyed[(2019, "h25v06")]["source_files"]) == [
        "GLASS06A01.V01.A2019055.h25v06.2022021.hdf",
        "GLASS06A01.V01.A2019056.h25v06.2022021.hdf",
    ]
    assert keyed[(2019, "h25v06")]["output_path"] == f"{pp.get_hpc_output_path('annual')}/2019/h25v06.zarr"


def test_avhrr_annual_targets_grouped_by_year_only(tmp_path):
    pp = _make_preprocessor(tmp_path, "AVHRR")
    targets = pp.get_preprocessing_targets("annual", year_range=(2019, 2021))

    keyed = {t["year"]: t for t in targets}
    assert set(keyed) == {2019, 2020}
    assert keyed[2019]["grid_cell"] == "global"
    assert keyed[2019]["output_path"] == f"{pp.get_hpc_output_path('annual')}/2019.zarr"


def test_modis_grid_cell_filter(tmp_path):
    pp = _make_preprocessor(tmp_path, "MODIS", grid_cells=["h25v06"])
    targets = pp.get_preprocessing_targets("annual", year_range=(2019, 2021))
    assert [(t["year"], t["grid_cell"]) for t in targets] == [(2019, "h25v06")]


def test_modis_spatial_target_is_single_combined_target(tmp_path):
    pp = _make_preprocessor(tmp_path, "MODIS", year_range=(2019, 2020))
    annual_dir = pp.get_hpc_output_path("annual")
    for year, cell in [(2019, "h25v06"), (2020, "h26v06")]:
        d = os.path.join(annual_dir, str(year))
        os.makedirs(d, exist_ok=True)
        os.makedirs(os.path.join(d, f"{cell}.zarr"))

    targets = pp.get_preprocessing_targets("spatial", year_range=(2019, 2020))
    assert len(targets) == 1
    assert targets[0]["grid_cell"] == "all_cells"
    assert targets[0]["output_path"] == f"{pp.get_hpc_output_path('spatial')}/modis_timeseries_reprojected.zarr"
    assert sorted(targets[0]["metadata"]["grid_cells"]) == ["h25v06", "h26v06"]


def test_avhrr_spatial_target_is_single_global_target(tmp_path):
    pp = _make_preprocessor(tmp_path, "AVHRR", year_range=(2019, 2020))
    annual_dir = pp.get_hpc_output_path("annual")
    os.makedirs(annual_dir, exist_ok=True)
    os.makedirs(os.path.join(annual_dir, "2019.zarr"))

    targets = pp.get_preprocessing_targets("spatial", year_range=(2019, 2020))
    assert len(targets) == 1
    assert targets[0]["grid_cell"] == "global"
    assert targets[0]["metadata"]["missing_years"] == [2020]
    assert targets[0]["output_path"] == f"{pp.get_hpc_output_path('spatial')}/avhrr_timeseries_reprojected.zarr"

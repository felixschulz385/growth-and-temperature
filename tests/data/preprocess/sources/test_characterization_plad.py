"""Characterization tests for the *current* PLADPreprocessor (pre-migration).

docs/design/09-integrated-pipeline.md §10 step 0. PLAD implements only
`stage="spatial"` -- there is no separate "annual"/"vector" pre-step; panel
construction and rasterization happen together in one stage (mapped to
GRID in the new vocabulary, with no PREPARE step -- correcting an earlier,
inaccurate assumption in the planning notes that PLAD had three stages like
acag/esacci).
"""

import os

from src.data.preprocess.sources.plad import PLADPreprocessor


def _make_preprocessor(tmp_path, admin_level=1, year_range=(1980, 2022)):
    return PLADPreprocessor(hpc_target=str(tmp_path), admin_level=admin_level, year_range=list(year_range))


def test_only_spatial_stage_is_supported(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        PLADPreprocessor(hpc_target=str(tmp_path), stage="annual")


def test_hpc_output_path_ignores_data_path_uses_fixed_plad_prefix(tmp_path):
    pp = _make_preprocessor(tmp_path, admin_level=1)
    # get_hpc_output_path hardcodes "plad" (not self.data_path) as the base.
    assert pp.get_hpc_output_path("spatial") == os.path.join(str(tmp_path), "plad", "processed", "stage_2")


def test_spatial_target_output_path_includes_admin_level(tmp_path):
    pp1 = _make_preprocessor(tmp_path, admin_level=1)
    pp2 = _make_preprocessor(tmp_path, admin_level=2)
    t1 = pp1.get_preprocessing_targets("spatial")[0]
    t2 = pp2.get_preprocessing_targets("spatial")[0]
    assert t1["output_path"] == f"{pp1.get_hpc_output_path('spatial')}/plad_adm1_timeseries_reprojected.zarr"
    assert t2["output_path"] == f"{pp2.get_hpc_output_path('spatial')}/plad_adm2_timeseries_reprojected.zarr"
    assert t1["metadata"]["admin_level"] == 1


def test_gadm_files_resolved_from_preprocessed_stage1_not_grid(tmp_path):
    pp = _make_preprocessor(tmp_path)
    gadm_dir = os.path.join(str(tmp_path), "misc", "processed", "stage_1", "gadm")
    os.makedirs(gadm_dir, exist_ok=True)
    open(os.path.join(gadm_dir, "gadm_levelADM_1_simplified.gpkg"), "w").close()

    files = pp._resolve_gadm_files_from_preprocessed()
    assert files == {"gadm_adm1": os.path.join(gadm_dir, "gadm_levelADM_1_simplified.gpkg")}


def test_admin_level_must_be_1_or_2(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        PLADPreprocessor(hpc_target=str(tmp_path), admin_level=3)

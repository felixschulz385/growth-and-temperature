"""Characterization tests for the *current* SnlMiningPreprocessor
(pre-migration). docs/design/09-integrated-pipeline.md §10 step 0 / §6.

Covers path resolution and target generation (the testable "plan" surface)
-- the DuckDB feature-building and tile rasterization are heavy compute,
ported mechanically without new fast-unit-test surface, consistent with how
this migration has treated every source's heavy compute step.
"""

import os

from src.data.preprocess.sources.snl_mining import SnlMiningPreprocessor


def _make_preprocessor(tmp_path, **kwargs):
    kwargs.setdefault("hpc_target", str(tmp_path))
    return SnlMiningPreprocessor(**kwargs)


def test_default_duckdb_and_prepared_db_paths(tmp_path):
    pp = _make_preprocessor(tmp_path)
    assert pp.duckdb_path == os.path.join(
        str(tmp_path), "snl_mining", "processed", "stage_0", "manual_xls", "snl_mining_manual_export.duckdb"
    )
    assert pp.prepared_db_path == os.path.join(str(tmp_path), "snl_mining", "processed", "stage_1", "snl_mining_prepared.duckdb")


def test_hpc_output_path(tmp_path):
    pp = _make_preprocessor(tmp_path)
    assert pp.get_hpc_output_path("spatial") == os.path.join(str(tmp_path), "snl_mining", "processed", "stage_2")


def test_only_spatial_stage_is_supported(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        SnlMiningPreprocessor(hpc_target=str(tmp_path), stage="annual")


def test_default_radius_and_admin_variables(tmp_path):
    pp = _make_preprocessor(tmp_path)
    assert pp.buffer_tables == {
        "mine_count_10km": ("mine_buffers_10km", 10000),
        "mine_count_20km": ("mine_buffers_20km", 20000),
        "mine_count_50km": ("mine_buffers_50km", 50000),
    }
    assert pp.admin_tables["mine_count_adm1"]["geometry_path"] == os.path.join(
        str(tmp_path), "misc", "processed", "stage_1", "gadm", "gadm_levelADM_1_simplified.gpkg"
    )
    assert pp.output_variables == [
        "mine_count_10km", "mine_count_20km", "mine_count_50km", "mine_count_adm1", "mine_count_adm2",
    ]


def test_spatial_target_single_combined_target(tmp_path):
    pp = _make_preprocessor(tmp_path, year_range=[2000, 2020])
    targets = pp.get_preprocessing_targets("spatial")
    assert len(targets) == 1
    t = targets[0]
    assert t["output_path"] == os.path.join(pp.get_hpc_output_path("spatial"), "snl_mining_timeseries_reprojected.zarr")
    assert t["dependencies"] == [pp.duckdb_path]
    assert t["metadata"]["year_range"] == [2000, 2020]

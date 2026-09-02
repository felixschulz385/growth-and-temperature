"""`run_workflow_with_config` reads the single `assembly:` block, always merges
every source, and fans one `run_assembly` pass out per grid-shake variant
(src/data/assemble/workflow.py).
"""

import pytest

from src.data.assemble import workflow
from src.data.assemble.workflow import _resolve_grid_resolution, run_workflow_with_config


def _config(**overrides):
    cfg = {
        "paths": {"data_root": "/tmp/data"},
        "pipeline": {"grid": "ease6933"},
        "assembly": {
            "output_root": "/tmp/assembled",
            "compression": "zstd",
            "land_mask": True,
            "sources": {
                "glass_modis": {"data_path": "glass/x", "family": "f", "index_cols": ["pixel_id", "year"]},
                "eog_viirs": {"data_path": "eog/viirs", "family": "g", "index_cols": ["pixel_id", "year"]},
                "land_mask": {"data_path": "misc", "family": "land_mask", "index_cols": ["pixel_id"]},
            },
        },
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def captured(monkeypatch):
    calls = []
    monkeypatch.setattr(workflow, "run_assembly", lambda ac, full: calls.append(ac))
    return calls


# --- grid label -> resolution ------------------------------------------------


def test_resolve_grid_resolution_native_is_none():
    assert _resolve_grid_resolution("1km") is None


def test_resolve_grid_resolution_coarse_is_metres():
    assert _resolve_grid_resolution("10km") == 10000.0


def test_resolve_grid_resolution_unknown_raises():
    with pytest.raises(ValueError, match="Unknown --grid label"):
        _resolve_grid_resolution("7km")


# --- fan-out per shake variant --------------------------------------------------


def test_default_run_is_one_base_variant_with_all_sources(captured):
    run_workflow_with_config({**_config(), "cli_overrides": {"grid_label": "1km", "shake": "none"}})

    assert len(captured) == 1
    ac = captured[0]
    assert ac["output_path"] == "/tmp/assembled/grid=1km/shake=base"
    assert set(ac["datasets"]) == {"glass_modis", "eog_viirs", "land_mask"}
    assert ac["processing"]["resolution"] is None
    assert ac["processing"]["grid_label"] == "1km"
    assert ac["processing"]["apply_land_mask"] is True
    assert ac["processing"]["compression"] == "zstd"


def test_quad_shake_fans_out_to_four_sibling_tables(captured):
    run_workflow_with_config({**_config(), "cli_overrides": {"grid_label": "10km", "shake": "quad"}})

    paths = [ac["output_path"] for ac in captured]
    assert paths == [
        "/tmp/assembled/grid=10km/shake=base",
        "/tmp/assembled/grid=10km/shake=s0",
        "/tmp/assembled/grid=10km/shake=s1",
        "/tmp/assembled/grid=10km/shake=s2",
    ]
    # every variant carries every source and the same coarse resolution
    for ac in captured:
        assert set(ac["datasets"]) == {"glass_modis", "eog_viirs", "land_mask"}
        assert ac["processing"]["resolution"] == 10000.0
    assert [ac["processing"]["shake_offset"] for ac in captured] == [
        [0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
    ]


def test_single_offset_label_writes_only_that_partition(captured):
    run_workflow_with_config({**_config(), "cli_overrides": {"grid_label": "10km", "shake": "s1"}})

    assert [ac["output_path"] for ac in captured] == ["/tmp/assembled/grid=10km/shake=s1"]
    assert captured[0]["processing"]["shake_offset"] == [0.0, 0.5]


def test_update_mode_and_datasource_flow_through(captured):
    run_workflow_with_config({
        **_config(),
        "cli_overrides": {"grid_label": "1km", "shake": "none", "assembly_mode": "update", "datasource": "eog_viirs"},
    })

    assert captured[0]["processing"]["assembly_mode"] == "update"
    assert captured[0]["processing"]["datasource"] == "eog_viirs"


def test_missing_assembly_block_raises():
    with pytest.raises(ValueError, match="no 'assembly:' block"):
        run_workflow_with_config({"paths": {}, "cli_overrides": {}})


# --- output_root is data_root-relative (no ${DATA_NOBACKUP}) -------------------


def test_relative_output_root_is_joined_to_data_root(captured):
    cfg = _config()
    cfg["assembly"]["output_root"] = "assembled"
    run_workflow_with_config({**cfg, "cli_overrides": {"grid_label": "10km", "shake": "none"}})
    assert captured[0]["output_path"] == "/tmp/data/assembled/grid=10km/shake=base"


def test_omitted_output_root_defaults_under_data_root(captured):
    cfg = _config()
    cfg["assembly"].pop("output_root")
    run_workflow_with_config({**cfg, "cli_overrides": {"grid_label": "1km", "shake": "none"}})
    assert captured[0]["output_path"] == "/tmp/data/assembled/grid=1km/shake=base"


def test_absolute_output_root_is_left_alone(captured):
    run_workflow_with_config({**_config(), "cli_overrides": {"grid_label": "1km", "shake": "none"}})
    assert captured[0]["output_path"] == "/tmp/assembled/grid=1km/shake=base"


def test_relative_dataset_path_resolves_against_data_root(tmp_path):
    from src.data.assemble.config import resolve_dataset_paths

    ac = {"datasets": {"cc": {"path": "prepared/misc/adm/x.parquet", "join_on": "GID_0"}}}
    resolve_dataset_paths(ac, str(tmp_path), grid_id="ease6933")
    assert ac["datasets"]["cc"]["path"] == str(tmp_path / "prepared/misc/adm/x.parquet")


def test_absolute_dataset_path_is_left_alone(tmp_path):
    from src.data.assemble.config import resolve_dataset_paths

    ac = {"datasets": {"cc": {"path": "/abs/x.parquet", "join_on": "GID_0"}}}
    resolve_dataset_paths(ac, str(tmp_path), grid_id="ease6933")
    assert ac["datasets"]["cc"]["path"] == "/abs/x.parquet"

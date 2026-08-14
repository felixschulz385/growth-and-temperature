"""Tests for scripts/migrate_layout_v2.py -- the layout:v2 physical
migration tool. Exercises the move-planning logic against a fake tmp_path
tree; no real HPC/SLURM involved.
"""

import os
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from migrate_layout_v2 import (  # noqa: E402
    GRID_FILENAME_AND_FAMILY,
    MigrationTally,
    build_source_pair,
    do_move,
    migrate_grid,
    migrate_source,
    plan_move,
)

from src.data.pipeline.config import SourceConfig  # noqa: E402
from src.data.pipeline.context import PipelineContext  # noqa: E402
from src.data.sources.acag import AcagSource  # noqa: E402
from src.data.sources.misc.gadm import GadmSource  # noqa: E402
from src.data.sources.snl_mining.source import SnlMiningSource  # noqa: E402
from src.data.sources.steps import PipelineStep  # noqa: E402


def _acag_pair(tmp_path, grid_id="legacy_4326"):
    data_root = str(tmp_path / "data_root")
    cfg = SourceConfig.from_dict("acag", {"data_path": "acag/pm25"})
    legacy = AcagSource(PipelineContext(data_root=data_root, grid_id=grid_id, layout="legacy"), cfg)
    v2 = AcagSource(PipelineContext(data_root=data_root, grid_id=grid_id, layout="v2"), cfg)
    return legacy, v2


def _gadm_pair(tmp_path):
    data_root = str(tmp_path / "data_root")
    cfg = SourceConfig.from_dict("gadm", {})
    legacy = GadmSource(PipelineContext(data_root=data_root, layout="legacy"), cfg)
    v2 = GadmSource(PipelineContext(data_root=data_root, layout="v2"), cfg)
    return legacy, v2


# -- plan_move() ---------------------------------------------------------


def test_plan_move_nothing_to_migrate_when_neither_path_exists(tmp_path):
    tally = MigrationTally()
    result = plan_move("x", str(tmp_path / "old"), str(tmp_path / "new"), tally)
    assert result is None
    assert tally.skipped_nothing_to_migrate == ["x"]


def test_plan_move_already_migrated_when_only_new_exists(tmp_path):
    new_path = tmp_path / "new"
    new_path.mkdir()
    tally = MigrationTally()
    result = plan_move("x", str(tmp_path / "old"), str(new_path), tally)
    assert result is None
    assert tally.skipped_already_migrated == ["x"]


def test_plan_move_returns_pair_when_only_old_exists(tmp_path):
    old_path = tmp_path / "old"
    old_path.mkdir()
    tally = MigrationTally()
    result = plan_move("x", str(old_path), str(tmp_path / "new"), tally)
    assert result == (str(old_path), str(tmp_path / "new"))
    assert tally.moved == [] and tally.skipped_nothing_to_migrate == []


def test_plan_move_conflict_skipped_when_both_exist(tmp_path):
    old_path = tmp_path / "old"
    new_path = tmp_path / "new"
    old_path.mkdir()
    new_path.mkdir()
    tally = MigrationTally()
    result = plan_move("x", str(old_path), str(new_path), tally)
    assert result is None
    assert tally.skipped_conflict == ["x"]
    # Neither side is touched.
    assert old_path.exists() and new_path.exists()


# -- do_move() -------------------------------------------------------------


def test_do_move_dry_run_touches_nothing(tmp_path):
    old_path = tmp_path / "old"
    old_path.mkdir()
    new_path = tmp_path / "sub" / "new"
    tally = MigrationTally()

    do_move("x", str(old_path), str(new_path), execute=False, tally=tally)

    assert old_path.exists()
    assert not new_path.exists()
    assert tally.moved == ["x"]


def test_do_move_execute_moves_the_directory(tmp_path):
    old_path = tmp_path / "old"
    old_path.mkdir()
    (old_path / "file.txt").write_text("hello")
    new_path = tmp_path / "sub" / "new"
    tally = MigrationTally()

    do_move("x", str(old_path), str(new_path), execute=True, tally=tally)

    assert not old_path.exists()
    assert (new_path / "file.txt").read_text() == "hello"
    assert tally.moved == ["x"]


def test_do_move_execute_records_failure_without_raising(tmp_path, monkeypatch):
    import migrate_layout_v2

    old_path = tmp_path / "old"
    old_path.mkdir()
    new_path = tmp_path / "new"
    tally = MigrationTally()

    def boom(*args, **kwargs):
        raise OSError("simulated failure")

    monkeypatch.setattr(migrate_layout_v2.shutil, "move", boom)

    do_move("x", str(old_path), str(new_path), execute=True, tally=tally)

    assert tally.failed == ["x"]
    assert tally.moved == []
    assert old_path.exists()  # untouched on failure


# -- migrate_source() end-to-end against fake FETCH/PREPARE/GRID trees -----


def test_migrate_source_moves_fetch_prepare_grid(tmp_path):
    # gadm's current code no longer declares a separate GRID step (Plan 2's
    # PREPARE+GRID merge) -- but pre-merge runs may still have real
    # legacy-layout GRID output on disk, and migrate_grid() migrates it
    # regardless of what the *current* class's STEPS says (it's keyed off
    # the static GRID_FILENAME_AND_FAMILY mapping instead -- see
    # migrate_grid()'s docstring). PipelineStep.GRID is used explicitly here
    # rather than legacy.STEPS[2], which no longer includes it.
    legacy, v2 = _gadm_pair(tmp_path)

    raw_dir = legacy.output_root(legacy.STEPS[0])  # FETCH
    prepare_dir = legacy.output_root(legacy.STEPS[1])  # PREPARE
    grid_dir = legacy.output_root(PipelineStep.GRID)  # legacy GRID output, pre-merge
    os.makedirs(raw_dir)
    os.makedirs(prepare_dir)
    legacy_filename, v2_family = GRID_FILENAME_AND_FAMILY["gadm"]
    os.makedirs(os.path.join(grid_dir, legacy_filename))

    tally = MigrationTally()
    migrate_source("gadm", legacy, v2, execute=True, tally=tally)

    assert v2.output_root(v2.STEPS[0]) and os.path.isdir(v2.output_root(v2.STEPS[0]))
    assert os.path.isdir(v2.output_root(v2.STEPS[1]))
    assert os.path.isdir(os.path.join(v2.output_root(PipelineStep.GRID), f"{v2_family}.zarr"))
    assert not os.path.exists(raw_dir)
    assert not os.path.exists(prepare_dir)
    assert not os.path.exists(os.path.join(grid_dir, legacy_filename))
    assert tally.failed == []


def test_migrate_source_dry_run_leaves_everything_in_place(tmp_path):
    legacy, v2 = _acag_pair(tmp_path)
    raw_dir = legacy.output_root(legacy.STEPS[0])
    os.makedirs(raw_dir)

    tally = MigrationTally()
    migrate_source("acag", legacy, v2, execute=False, tally=tally)

    assert os.path.isdir(raw_dir)
    assert not os.path.exists(v2.output_root(v2.STEPS[0]))


def test_migrate_source_skips_source_with_nothing_to_migrate(tmp_path):
    # gadm is used here (not acag) because GRID_FILENAME_AND_FAMILY still
    # knows about its historical GRID output and gadm-specific sidecars,
    # even though its current class no longer declares GRID (Plan 2's
    # PREPARE+GRID merge) -- migrate_grid() migrates by that static mapping,
    # not the current class's STEPS (see its docstring).
    legacy, v2 = _gadm_pair(tmp_path)
    tally = MigrationTally()
    migrate_source("gadm", legacy, v2, execute=True, tally=tally)
    assert tally.moved == []
    # FETCH, PREPARE, GRID, plus gadm's two GRID sidecars (country/subdivision
    # code mapping JSON).
    assert len(tally.skipped_nothing_to_migrate) == 5


# -- gadm sidecar handling ---------------------------------------------------


def test_migrate_grid_moves_gadm_sidecars_alongside_the_zarr(tmp_path):
    # migrate_grid() migrates a source's legacy GRID output regardless of
    # whether its current class still declares GRID (gadm's PREPARE+GRID
    # merge dropped it, but pre-merge on-disk data still needs migrating) --
    # see migrate_grid()'s docstring. PipelineStep.GRID is used explicitly
    # here rather than legacy.STEPS[-1], which is now PREPARE for gadm.
    legacy, v2 = _gadm_pair(tmp_path)
    grid_dir = legacy.output_root(PipelineStep.GRID)
    legacy_filename, v2_family = GRID_FILENAME_AND_FAMILY["gadm"]
    os.makedirs(os.path.join(grid_dir, legacy_filename))
    Path(grid_dir, "country_code_mapping.json").write_text("{}")
    Path(grid_dir, "subdivision_code_mapping.json").write_text("{}")

    tally = MigrationTally()
    migrate_grid("gadm", legacy, v2, execute=True, tally=tally)

    new_grid_dir = v2.output_root(PipelineStep.GRID)
    assert os.path.isdir(os.path.join(new_grid_dir, f"{v2_family}.zarr"))
    assert os.path.exists(os.path.join(new_grid_dir, "country_code_mapping.json"))
    assert os.path.exists(os.path.join(new_grid_dir, "subdivision_code_mapping.json"))
    assert tally.failed == []


def test_migrate_grid_skips_sidecar_that_does_not_exist(tmp_path):
    # Only country_code_mapping.json present (no ADM1 -> no subdivision file)
    # -- must not fail trying to move a sidecar that was never written.
    legacy, v2 = _gadm_pair(tmp_path)
    grid_dir = legacy.output_root(PipelineStep.GRID)
    legacy_filename, _ = GRID_FILENAME_AND_FAMILY["gadm"]
    os.makedirs(os.path.join(grid_dir, legacy_filename))
    Path(grid_dir, "country_code_mapping.json").write_text("{}")

    tally = MigrationTally()
    migrate_grid("gadm", legacy, v2, execute=True, tally=tally)

    assert tally.failed == []
    assert "gadm/GRID/subdivision_code_mapping.json" in tally.skipped_nothing_to_migrate


# -- snl_mining PREPARE exception -------------------------------------------


def test_snl_mining_prepare_is_never_migrated(tmp_path):
    data_root = str(tmp_path / "data_root")
    cfg = SourceConfig.from_dict("snl_mining", {})
    legacy = SnlMiningSource(PipelineContext(data_root=data_root, layout="legacy"), cfg)
    v2 = SnlMiningSource(PipelineContext(data_root=data_root, layout="v2"), cfg)

    prepare_dir = os.path.dirname(legacy.prepared_db_path)
    os.makedirs(prepare_dir, exist_ok=True)
    Path(legacy.prepared_db_path).write_text("fake duckdb")

    from migrate_layout_v2 import migrate_fetch_or_prepare
    from src.data.sources.steps import PipelineStep

    tally = MigrationTally()
    migrate_fetch_or_prepare("snl_mining", PipelineStep.PREPARE, legacy, v2, execute=True, tally=tally)

    # Untouched -- snl_mining's PREPARE artefact is a documented exception.
    assert os.path.exists(legacy.prepared_db_path)
    assert tally.moved == []
    assert tally.skipped_nothing_to_migrate == []
    assert tally.skipped_already_migrated == []


# -- build_source_pair() / registry integration -----------------------------


def test_build_source_pair_forces_ease_grid_id_for_modis(tmp_path):
    config = {"sources": {"modis": {"tiles": ["h18v04"], "year_range": [2019, 2019]}}}
    legacy, v2 = build_source_pair("modis", config, str(tmp_path / "data_root"), "legacy_4326")
    assert legacy.ctx.grid_id == "ease6933"
    assert v2.ctx.grid_id == "ease6933"


def test_build_source_pair_respects_requested_grid_id_for_non_modis(tmp_path):
    config = {"sources": {"acag": {"data_path": "acag/pm25"}}}
    legacy, v2 = build_source_pair("acag", config, str(tmp_path / "data_root"), "ease6933")
    assert legacy.ctx.grid_id == "ease6933"

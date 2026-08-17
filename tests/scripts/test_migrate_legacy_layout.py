"""Tests for scripts/migrate_legacy_layout.py -- the legacy-layout physical
migration tool. Exercises the move-planning logic against a fake tmp_path
tree; no real HPC/SLURM involved.
"""

import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from migrate_legacy_layout import (  # noqa: E402
    GRID_FILENAME_AND_FAMILY,
    MigrationTally,
    build_source,
    do_move,
    legacy_output_root,
    legacy_raw_root,
    migrate_grid,
    migrate_source,
    plan_move,
)

from src.data.pipeline.config import SourceConfig  # noqa: E402
from src.data.pipeline.context import PipelineContext  # noqa: E402
from src.data.sources import layout  # noqa: E402
from src.data.sources.acag import AcagSource  # noqa: E402
from src.data.sources.misc.gadm import GadmSource  # noqa: E402
from src.data.sources.snl_mining.source import SnlMiningSource  # noqa: E402
from src.data.sources.steps import PipelineStep  # noqa: E402


def _acag_source(tmp_path, grid_id="legacy_4326"):
    data_root = str(tmp_path / "data_root")
    cfg = SourceConfig.from_dict("acag", {"data_path": "acag/pm25"})
    return AcagSource(PipelineContext(data_root=data_root, grid_id=grid_id), cfg)


def _gadm_source(tmp_path, grid_id="legacy_4326"):
    data_root = str(tmp_path / "data_root")
    cfg = SourceConfig.from_dict("gadm", {})
    return GadmSource(PipelineContext(data_root=data_root, grid_id=grid_id), cfg)


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
    import migrate_legacy_layout

    old_path = tmp_path / "old"
    old_path.mkdir()
    new_path = tmp_path / "new"
    tally = MigrationTally()

    def boom(*args, **kwargs):
        raise OSError("simulated failure")

    monkeypatch.setattr(migrate_legacy_layout.shutil, "move", boom)

    do_move("x", str(old_path), str(new_path), execute=True, tally=tally)

    assert tally.failed == ["x"]
    assert tally.moved == []
    assert old_path.exists()  # untouched on failure


# -- legacy path formulas ---------------------------------------------------


def test_legacy_raw_root_shape():
    assert legacy_raw_root("/root", "acag/pm25") == "/root/acag/pm25/raw"
    assert legacy_raw_root("/root", "misc", namespace="gadm") == "/root/misc/raw/gadm"


def test_legacy_output_root_prepare_and_grid_shape():
    assert legacy_output_root("/root", "acag/pm25", PipelineStep.PREPARE) == "/root/acag/pm25/processed/stage_1"
    assert legacy_output_root("/root", "modis", PipelineStep.GRID, grid_id="ease6933") == "/root/modis/processed/stage_2_ease6933"
    assert legacy_output_root("/root", "acag", PipelineStep.GRID) == "/root/acag/processed/stage_2"


# -- migrate_source() end-to-end against fake FETCH/PREPARE/GRID trees -----


def test_migrate_source_moves_fetch_prepare_grid(tmp_path):
    # gadm's current code doesn't declare a separate GRID step, but earlier
    # runs may still have real legacy-layout GRID output on disk, and
    # migrate_grid() migrates it regardless of what the *current* class's
    # STEPS says (it's keyed off the static GRID_FILENAME_AND_FAMILY mapping
    # instead).
    source = _gadm_source(tmp_path)

    raw_dir = legacy_raw_root(source.ctx.data_root, source.cfg.data_path, namespace=source.cfg.namespace)
    prepare_dir = legacy_output_root(source.ctx.data_root, source.cfg.data_path, PipelineStep.PREPARE, namespace=source.cfg.namespace)
    grid_dir = legacy_output_root(source.ctx.data_root, source.cfg.data_path, PipelineStep.GRID)
    os.makedirs(raw_dir)
    os.makedirs(prepare_dir)
    legacy_filename, family = GRID_FILENAME_AND_FAMILY["gadm"]
    os.makedirs(os.path.join(grid_dir, legacy_filename))

    tally = MigrationTally()
    migrate_source("gadm", source, grid_id="legacy_4326", execute=True, tally=tally)

    assert os.path.isdir(source.output_root(PipelineStep.FETCH))
    assert os.path.isdir(source.output_root(PipelineStep.PREPARE, agg=layout.ADM_AGG))
    assert os.path.isdir(os.path.join(source.output_root(PipelineStep.GRID), f"{family}.zarr"))
    assert not os.path.exists(raw_dir)
    assert not os.path.exists(prepare_dir)
    assert not os.path.exists(os.path.join(grid_dir, legacy_filename))
    assert tally.failed == []


def test_migrate_source_dry_run_leaves_everything_in_place(tmp_path):
    source = _acag_source(tmp_path)
    raw_dir = legacy_raw_root(source.ctx.data_root, source.cfg.data_path, namespace=source.cfg.namespace)
    os.makedirs(raw_dir)

    tally = MigrationTally()
    migrate_source("acag", source, grid_id="legacy_4326", execute=False, tally=tally)

    assert os.path.isdir(raw_dir)
    assert not os.path.exists(source.output_root(PipelineStep.FETCH))


def test_migrate_source_skips_source_with_nothing_to_migrate(tmp_path):
    source = _gadm_source(tmp_path)
    tally = MigrationTally()
    migrate_source("gadm", source, grid_id="legacy_4326", execute=True, tally=tally)
    assert tally.moved == []
    # FETCH, PREPARE, GRID, plus gadm's two GRID sidecars (country/subdivision
    # code mapping JSON).
    assert len(tally.skipped_nothing_to_migrate) == 5


# -- gadm sidecar handling ---------------------------------------------------


def test_migrate_grid_moves_gadm_sidecars_alongside_the_zarr(tmp_path):
    source = _gadm_source(tmp_path)
    grid_dir = legacy_output_root(source.ctx.data_root, source.cfg.data_path, PipelineStep.GRID)
    legacy_filename, family = GRID_FILENAME_AND_FAMILY["gadm"]
    os.makedirs(os.path.join(grid_dir, legacy_filename))
    Path(grid_dir, "country_code_mapping.json").write_text("{}")
    Path(grid_dir, "subdivision_code_mapping.json").write_text("{}")

    tally = MigrationTally()
    migrate_grid("gadm", source, grid_id="legacy_4326", execute=True, tally=tally)

    new_grid_dir = source.output_root(PipelineStep.GRID)
    assert os.path.isdir(os.path.join(new_grid_dir, f"{family}.zarr"))
    assert os.path.exists(os.path.join(new_grid_dir, "country_code_mapping.json"))
    assert os.path.exists(os.path.join(new_grid_dir, "subdivision_code_mapping.json"))
    assert tally.failed == []


def test_migrate_grid_skips_sidecar_that_does_not_exist(tmp_path):
    # Only country_code_mapping.json present (no ADM1 -> no subdivision file)
    # -- must not fail trying to move a sidecar that was never written.
    source = _gadm_source(tmp_path)
    grid_dir = legacy_output_root(source.ctx.data_root, source.cfg.data_path, PipelineStep.GRID)
    legacy_filename, _family = GRID_FILENAME_AND_FAMILY["gadm"]
    os.makedirs(os.path.join(grid_dir, legacy_filename))
    Path(grid_dir, "country_code_mapping.json").write_text("{}")

    tally = MigrationTally()
    migrate_grid("gadm", source, grid_id="legacy_4326", execute=True, tally=tally)

    assert tally.failed == []
    assert "gadm/GRID/subdivision_code_mapping.json" in tally.skipped_nothing_to_migrate


# -- snl_mining PREPARE exception -------------------------------------------


def test_snl_mining_prepare_is_never_migrated(tmp_path):
    data_root = str(tmp_path / "data_root")
    cfg = SourceConfig.from_dict("snl_mining", {})
    source = SnlMiningSource(PipelineContext(data_root=data_root), cfg)

    prepare_dir = os.path.dirname(source.prepared_db_path)
    os.makedirs(prepare_dir, exist_ok=True)
    Path(source.prepared_db_path).write_text("fake duckdb")

    from migrate_legacy_layout import migrate_prepare

    tally = MigrationTally()
    migrate_prepare("snl_mining", source, execute=True, tally=tally)

    # Untouched -- snl_mining's PREPARE artefact is a documented exception.
    assert os.path.exists(source.prepared_db_path)
    assert tally.moved == []
    assert tally.skipped_nothing_to_migrate == []
    assert tally.skipped_already_migrated == []


# -- build_source() / registry integration -----------------------------


def test_build_source_forces_ease_grid_id_for_modis(tmp_path):
    config = {"sources": {"modis": {"tiles": ["h18v04"], "year_range": [2019, 2019]}}}
    source = build_source("modis", config, str(tmp_path / "data_root"), "legacy_4326")
    # MODIS's own output_root() override forces ease6933 for PREPARE/GRID
    # regardless of the requested grid_id.
    assert "ease6933" in source.output_root(PipelineStep.GRID)


def test_build_source_respects_requested_grid_id_for_non_modis(tmp_path):
    config = {"sources": {"acag": {"data_path": "acag/pm25"}}}
    source = build_source("acag", config, str(tmp_path / "data_root"), "ease6933")
    assert source.ctx.grid_id == "ease6933"

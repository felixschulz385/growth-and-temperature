"""SnlMiningSource.plan() must reproduce the relevant slice of the old
SnlMiningPreprocessor's behaviour, plus the new PREPARE/GRID split.
Oracle: tests/data/preprocess/sources/test_characterization_snl_mining.py.
"""

import os

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.snl_mining.source import SnlMiningSource
from src.data.sources.steps import PipelineStep, TargetSelection


def _make_source(tmp_path, *, grid_id="legacy_4326", **raw):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), grid_id=grid_id
    )
    cfg = SourceConfig.from_dict("snl_mining", dict(raw))
    return SnlMiningSource(ctx, cfg), ctx


def test_no_fetch_step():
    assert SnlMiningSource.STEPS == (PipelineStep.PREPARE, PipelineStep.GRID)


def test_requires_gadm_prepare():
    from src.data.sources import registry

    assert registry.resolve("snl_mining").requires == (("gadm", PipelineStep.PREPARE),)


def test_default_duckdb_and_prepared_db_paths(tmp_path):
    source, ctx = _make_source(tmp_path)
    assert source.duckdb_path == os.path.join(
        ctx.data_root, "snl_mining", "processed", "stage_0", "manual_xls", "snl_mining_manual_export.duckdb"
    )
    assert source.prepared_db_path == os.path.join(ctx.data_root, "snl_mining", "processed", "stage_1", "snl_mining_prepared.duckdb")


def test_default_radius_and_admin_variables(tmp_path):
    source, ctx = _make_source(tmp_path)
    assert source.buffer_tables == {
        "mine_count_10km": ("mine_buffers_10km", 10000),
        "mine_count_20km": ("mine_buffers_20km", 20000),
        "mine_count_50km": ("mine_buffers_50km", 50000),
    }
    assert source.admin_tables["mine_count_adm1"]["geometry_path"] == os.path.join(
        ctx.data_root, "misc", "processed", "stage_1", "gadm", "gadm_levelADM_1_simplified.gpkg"
    )


def test_prepare_plan_empty_when_stage0_duckdb_missing(tmp_path):
    source, _ = _make_source(tmp_path)
    assert source.plan(PipelineStep.PREPARE, TargetSelection()) == []


def test_prepare_plan_target_when_stage0_duckdb_present(tmp_path):
    source, _ = _make_source(tmp_path)
    os.makedirs(os.path.dirname(source.duckdb_path), exist_ok=True)
    open(source.duckdb_path, "w").close()

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    assert targets[0].output_path == source.prepared_db_path
    assert targets[0].inputs == (source.duckdb_path,)


def test_grid_plan_empty_when_prepared_db_missing(tmp_path):
    source, _ = _make_source(tmp_path)
    assert source.plan(PipelineStep.GRID, TargetSelection()) == []


def test_output_root_grid_matches_old_get_hpc_output_path(tmp_path):
    source, ctx = _make_source(tmp_path)
    assert source.output_root(PipelineStep.GRID) == os.path.join(ctx.data_root, "snl_mining", "processed", "stage_2")


def test_output_root_grid_honors_ease6933(tmp_path):
    # Regression test: _output_root() used to hardcode "stage_2" and ignore
    # ctx.grid_id entirely, unlike every other source's output_root().
    source, ctx = _make_source(tmp_path, grid_id="ease6933")
    assert source.output_root(PipelineStep.GRID) == os.path.join(
        ctx.data_root, "snl_mining", "processed", "stage_2_ease6933"
    )


def test_get_or_create_geobox_delegates_to_shared_target_helper(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path, grid_id="ease6933")

    calls = []

    def fake_get_target_geobox(passed_ctx):
        calls.append(passed_ctx)
        return "fake-canonical-geobox"

    import src.data.sources.snl_mining.source as snl_source_module

    monkeypatch.setattr(snl_source_module, "get_target_geobox", fake_get_target_geobox)

    assert source._get_or_create_geobox() == "fake-canonical-geobox"
    assert calls == [ctx]

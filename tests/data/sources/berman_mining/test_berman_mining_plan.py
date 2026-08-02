"""BermanMiningSource.plan() must reproduce the old
BermanMiningPreprocessor's targets. Oracle:
tests/data/preprocess/sources/test_characterization_berman_mining.py.
"""

import os

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.berman_mining import BermanMiningSource
from src.data.sources.steps import PipelineStep, TargetSelection


def _make_source(tmp_path, grid_id="legacy_4326", **raw):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), grid_id=grid_id
    )
    cfg = SourceConfig.from_dict("berman_mining", dict(raw))
    return BermanMiningSource(ctx, cfg), ctx


def test_no_prepare_step():
    assert BermanMiningSource.STEPS == (PipelineStep.FETCH, PipelineStep.GRID)


def test_no_requires_on_gadm():
    from src.data.sources import registry

    assert registry.resolve("berman_mining").requires == ()


def test_default_mining_data_path(tmp_path):
    source, ctx = _make_source(tmp_path)
    assert source.mining_data_path == os.path.join(ctx.data_root, "berman_mining", "raw", "baseline", "BCRT_baseline.dta")


def test_hpc_output_path(tmp_path):
    source, ctx = _make_source(tmp_path)
    assert source.output_root(PipelineStep.GRID) == os.path.join(ctx.data_root, "berman_mining", "processed", "stage_2")


def test_grid_target(tmp_path):
    source, _ = _make_source(tmp_path, year_range=[2000, 2010])
    targets = source.plan(PipelineStep.GRID, TargetSelection())
    assert len(targets) == 1
    assert targets[0].output_path == os.path.join(
        source.output_root(PipelineStep.GRID), "berman_mining_timeseries_reprojected.zarr"
    )
    assert targets[0].meta["year_range"] == (2000, 2010)


def test_get_or_create_geobox_delegates_to_shared_target_helper(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path, grid_id="ease6933")

    import src.data.common.geobox as geobox_module

    calls = []

    def fake_get_target_geobox(passed_ctx):
        calls.append(passed_ctx)
        return "fake-canonical-geobox"

    monkeypatch.setattr(geobox_module, "get_target_geobox", fake_get_target_geobox)

    assert source._get_or_create_geobox() == "fake-canonical-geobox"
    assert calls == [ctx]

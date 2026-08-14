"""OsmSource: ledger-free FETCH/PREPARE (docs/design successor to the
ledger, Plan 2 PREPARE+GRID merge). plan() is a bare live `os.path.exists()`
check against the raw fetched file -- see
tests/data/sources/acag/test_acag_plan.py's non-tiled counterpart
(commodity_prices) for the same shape.
"""

import os

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.steps import Completion, PipelineStep, TargetSelection


def _make(tmp_path):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), layout="legacy"
    )
    cfg = SourceConfig.from_dict("osm", {})
    cls = registry.load("osm")
    return cls(ctx, cfg), ctx


def _write_fake_raw_zip(source):
    raw_file = source._raw_file_path()
    os.makedirs(os.path.dirname(raw_file), exist_ok=True)
    open(raw_file, "w").close()
    return raw_file


def test_steps_is_fetch_and_prepare_only():
    from src.data.sources.misc.osm import OsmSource

    assert OsmSource.STEPS == (PipelineStep.FETCH, PipelineStep.PREPARE)


def test_prepare_plan_empty_when_raw_file_missing(tmp_path):
    source, _ = _make(tmp_path)
    assert source.plan(PipelineStep.PREPARE, TargetSelection()) == []


def test_prepare_plan_one_target_when_raw_file_present(tmp_path):
    source, _ = _make(tmp_path)
    raw_file = _write_fake_raw_zip(source)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    target = targets[0]
    assert target.key == "osm"
    assert target.inputs == (raw_file,)
    assert target.completion == Completion.MARKER
    assert target.output_path == source._output_path()
    assert target.output_path.endswith("land_mask.zarr")

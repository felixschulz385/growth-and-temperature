"""OsmSource.plan() reads a reconciled ledger instead of falling back to
live discovery -- the static/singleton-target counterpart to
tests/data/sources/misc/test_gadm_ledger_plan.py.
"""

import os

from src.data.common.ledger.store import SourceLedger
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.reconcile import reconcile_step
from src.data.sources.steps import PipelineStep, TargetSelection, mark_complete


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


def _write_fake_vector_file(source):
    vector_path = os.path.join(source.output_root(PipelineStep.PREPARE), "land_polygons_simplified.gpkg")
    os.makedirs(os.path.dirname(vector_path), exist_ok=True)
    open(vector_path, "w").close()
    return vector_path


def test_plan_prepare_reads_from_reconciled_ledger(tmp_path):
    source, ctx = _make(tmp_path)
    raw_file = _write_fake_raw_zip(source)

    local_ledger_path = os.path.join(ctx.local_index_dir, "misc_osm.duckdb")
    with SourceLedger.open(local_ledger_path, data_path="misc/osm") as ledger:
        reconcile_step(source, PipelineStep.PREPARE, ledger)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    assert targets[0].key == "osm"
    assert targets[0].inputs == (raw_file,)
    assert targets[0].output_path == os.path.join(
        source.output_root(PipelineStep.PREPARE), "land_polygons_simplified.gpkg"
    )


def test_plan_grid_reads_vector_path_from_ledger_meta(tmp_path):
    source, ctx = _make(tmp_path)
    vector_path = _write_fake_vector_file(source)

    local_ledger_path = os.path.join(ctx.local_index_dir, "misc_osm.duckdb")
    with SourceLedger.open(local_ledger_path, data_path="misc/osm") as ledger:
        reconcile_step(source, PipelineStep.GRID, ledger)

    targets = source.plan(PipelineStep.GRID, TargetSelection())
    assert len(targets) == 1
    assert targets[0].inputs == (vector_path,)


def test_plan_falls_back_to_discovery_when_ledger_unpopulated(tmp_path):
    source, _ = _make(tmp_path)
    _write_fake_raw_zip(source)
    # local_index_dir is configured but nothing has been reconciled yet.
    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    assert targets[0].key == "osm"

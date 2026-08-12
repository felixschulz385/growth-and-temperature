"""PlaDSource.plan() reads a reconciled ledger instead of falling back to
live discovery -- the static/singleton-target counterpart to
tests/data/sources/misc/test_gadm_ledger_plan.py.
"""

import json
import os

from src.data.common.ledger.paths import ledger_path
from src.data.common.ledger.store import SourceLedger
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.plad import PlaDSource
from src.data.sources.reconcile import reconcile_step
from src.data.sources.steps import PipelineStep, TargetSelection


def _make(tmp_path, admin_level=1, year_range=(1980, 2022)):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), layout="legacy"
    )
    cfg = SourceConfig.from_dict("plad", {"admin_level": admin_level, "year_range": list(year_range)})
    return PlaDSource(ctx, cfg), ctx


def _write_gadm_mapping(source, ctx):
    from src.data.sources.misc.gadm import gid_mapping_path

    mapping_path = gid_mapping_path(ctx.data_root, ctx.grid_id, ctx.layout, source._gid_column)
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    with open(mapping_path, "w") as f:
        json.dump({"USA.1_1": 5}, f)
    return mapping_path


def test_plan_grid_reads_mapping_file_from_ledger_meta(tmp_path):
    source, ctx = _make(tmp_path)
    mapping_file = _write_gadm_mapping(source, ctx)

    local_ledger_path = ledger_path(ctx.local_index_dir, source.data_path)
    with SourceLedger.open(local_ledger_path, data_path=source.data_path) as ledger:
        reconcile_step(source, PipelineStep.GRID, ledger)

    targets = source.plan(PipelineStep.GRID, TargetSelection())
    assert len(targets) == 1
    assert targets[0].inputs == (mapping_file,)
    assert targets[0].key == "adm1"


def test_plan_falls_back_to_discovery_when_ledger_unpopulated(tmp_path):
    source, ctx = _make(tmp_path)
    _write_gadm_mapping(source, ctx)
    # local_index_dir is configured but nothing has been reconciled yet.
    targets = source.plan(PipelineStep.GRID, TargetSelection())
    assert len(targets) == 1
    assert targets[0].key == "adm1"

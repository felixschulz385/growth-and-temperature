"""reconcile_step() against real local-disk completion state + a fake HPC
client -- no live SSH target needed.
"""

import os

import pytest

from src.data.common.ledger.store import SourceLedger
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.base import DataSource
from src.data.sources.reconcile import reconcile_step
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection, mark_complete


class _FakeSource(DataSource):
    ID = "fake"
    STEPS = (PipelineStep.PREPARE, PipelineStep.GRID)

    def __init__(self, ctx, cfg, targets):
        super().__init__(ctx, cfg)
        self._targets = targets

    def _plan(self, step, selection: TargetSelection):
        return [t for t in self._targets if t.step is step]

    def _execute(self, target):
        return True


@pytest.fixture
def source_and_ctx(tmp_path):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("fake", {"data_path": "fake"})
    return ctx, cfg


@pytest.fixture
def ledger(tmp_path):
    path = str(tmp_path / "fake.duckdb")
    with SourceLedger.open(path, data_path="fake") as led:
        yield led


class _FakeHPCClient:
    def __init__(self, existing_remote_paths):
        self.existing = set(existing_remote_paths)
        self.checked = None

    def check_files_exist(self, remote_paths):
        self.checked = list(remote_paths)
        return {p: p in self.existing for p in remote_paths}


def test_reconcile_step_rejects_fetch(source_and_ctx, ledger):
    ctx, cfg = source_and_ctx
    source = _FakeSource(ctx, cfg, targets=[])
    with pytest.raises(ValueError):
        reconcile_step(source, PipelineStep.FETCH, ledger)


def test_reconcile_step_marks_local_complete_path_exists(source_and_ctx, ledger, tmp_path):
    ctx, cfg = source_and_ctx
    output_path = str(tmp_path / "2020.parquet")
    open(output_path, "w").close()
    target = StepTarget(
        source_id="fake", step=PipelineStep.PREPARE, key="2020",
        output_path=output_path, completion=Completion.PATH_EXISTS,
    )
    source = _FakeSource(ctx, cfg, targets=[target])

    result = reconcile_step(source, PipelineStep.PREPARE, ledger)
    assert result == {"total": 1, "local_complete": 1, "remote_verified": 0}
    assert ledger.local_state("prepare", "2020") == "complete"


def test_reconcile_step_marks_local_complete_marker(source_and_ctx, ledger, tmp_path):
    ctx, cfg = source_and_ctx
    output_dir = str(tmp_path / "grid.zarr")
    os.makedirs(output_dir, exist_ok=True)
    target = StepTarget(
        source_id="fake", step=PipelineStep.GRID, key="all",
        output_path=output_dir, completion=Completion.MARKER,
    )
    mark_complete(output_dir)
    source = _FakeSource(ctx, cfg, targets=[target])

    result = reconcile_step(source, PipelineStep.GRID, ledger)
    assert result["local_complete"] == 1
    assert ledger.local_state("grid", "all") == "complete"


def test_reconcile_step_not_complete_when_missing(source_and_ctx, ledger, tmp_path):
    ctx, cfg = source_and_ctx
    target = StepTarget(
        source_id="fake", step=PipelineStep.PREPARE, key="2020",
        output_path=str(tmp_path / "does_not_exist.parquet"), completion=Completion.PATH_EXISTS,
    )
    source = _FakeSource(ctx, cfg, targets=[target])

    result = reconcile_step(source, PipelineStep.PREPARE, ledger)
    assert result == {"total": 1, "local_complete": 0, "remote_verified": 0}
    assert ledger.local_state("prepare", "2020") == "missing"


def test_reconcile_step_checks_remote_marker_path_not_directory(source_and_ctx, ledger, tmp_path):
    ctx, cfg = source_and_ctx
    remote_data_root = str(tmp_path / "remote")
    output_dir = os.path.join(remote_data_root, "grid", "acag.zarr")
    target = StepTarget(
        source_id="fake", step=PipelineStep.GRID, key="all",
        output_path=output_dir, completion=Completion.MARKER,
    )
    source = _FakeSource(ctx, cfg, targets=[target])

    expected_marker = os.path.relpath(output_dir, remote_data_root).rstrip("/") + ".complete"
    client = _FakeHPCClient(existing_remote_paths=[expected_marker])

    result = reconcile_step(source, PipelineStep.GRID, ledger, client=client, remote_data_root=remote_data_root)
    assert result["remote_verified"] == 1
    assert ledger.remote_state("grid", "all") == "verified"
    # The probed path is the marker file, never the bare directory path.
    assert client.checked == [expected_marker]


def test_reconcile_step_remote_not_verified_when_absent(source_and_ctx, ledger, tmp_path):
    ctx, cfg = source_and_ctx
    remote_data_root = str(tmp_path / "remote")
    output_path = os.path.join(remote_data_root, "prepared", "2020.parquet")
    target = StepTarget(
        source_id="fake", step=PipelineStep.PREPARE, key="2020",
        output_path=output_path, completion=Completion.PATH_EXISTS,
    )
    source = _FakeSource(ctx, cfg, targets=[target])
    client = _FakeHPCClient(existing_remote_paths=[])

    result = reconcile_step(source, PipelineStep.PREPARE, ledger, client=client, remote_data_root=remote_data_root)
    assert result["remote_verified"] == 0
    assert ledger.remote_state("prepare", "2020") == "missing"

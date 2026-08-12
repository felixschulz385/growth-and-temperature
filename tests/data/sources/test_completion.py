"""is_complete()'s local/remote state matrix -- docs/design/10-fetch-ledger.md §6.

`require_remote=False` (every source except MODIS's FETCH) must behave
exactly as before: local-disk-only, ledger argument ignored entirely.
`require_remote=True` additionally requires ledger-confirmed HPC verification.
"""

import os

import pytest

from src.data.common.ledger.schema import LocalState
from src.data.common.ledger.store import PushResult, SourceLedger
from src.data.sources.steps import Completion, PipelineStep, StepTarget, is_complete, local_drift, mark_complete


@pytest.fixture
def ledger(tmp_path):
    path = str(tmp_path / "modis.duckdb")
    with SourceLedger.open(path, data_path="modis") as led:
        yield led


def _path_exists_target(tmp_path, *, require_remote=False, exists=True):
    output_path = str(tmp_path / "2020" / "h09v05.tif")
    if exists:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        open(output_path, "w").close()
    return StepTarget(
        source_id="modis", step=PipelineStep.FETCH, key="2020/h09v05",
        output_path=output_path, completion=Completion.PATH_EXISTS, require_remote=require_remote,
    )


def test_local_only_target_ignores_ledger_when_absent(tmp_path):
    target = _path_exists_target(tmp_path, require_remote=False, exists=True)
    assert is_complete(target) is True
    assert is_complete(target, ledger=None) is True


def test_local_only_target_false_when_missing(tmp_path):
    target = _path_exists_target(tmp_path, require_remote=False, exists=False)
    assert is_complete(target) is False


def test_require_remote_without_ledger_falls_back_to_local(tmp_path):
    # No ledger passed -- every existing caller (handle_plan/handle_run today)
    # keeps its current local-only behavior unchanged.
    target = _path_exists_target(tmp_path, require_remote=True, exists=True)
    assert is_complete(target, ledger=None) is True


def test_require_remote_false_locally_even_with_ledger_verified(tmp_path, ledger):
    # Local file missing -- remote verification can't make it "complete".
    target = _path_exists_target(tmp_path, require_remote=True, exists=False)
    ledger.ensure_artifact("fetch", "2020/h09v05")
    ledger.record_push_batch("fetch", [PushResult(unit_id="2020/h09v05", ok=True)])
    assert is_complete(target, ledger=ledger) is False


def test_require_remote_false_when_not_yet_verified(tmp_path, ledger):
    target = _path_exists_target(tmp_path, require_remote=True, exists=True)
    ledger.ensure_artifact("fetch", "2020/h09v05")
    assert is_complete(target, ledger=ledger) is False


def test_require_remote_true_when_local_and_remote_verified(tmp_path, ledger):
    target = _path_exists_target(tmp_path, require_remote=True, exists=True)
    ledger.ensure_artifact("fetch", "2020/h09v05")
    ledger.record_push_batch("fetch", [PushResult(unit_id="2020/h09v05", ok=True)])
    assert is_complete(target, ledger=ledger) is True


def test_marker_completion_with_require_remote(tmp_path, ledger):
    output_dir = str(tmp_path / "grid" / "modis.zarr")
    os.makedirs(output_dir, exist_ok=True)
    mark_complete(output_dir)
    target = StepTarget(
        source_id="modis", step=PipelineStep.GRID, key="all",
        output_path=output_dir, completion=Completion.MARKER, require_remote=True,
    )
    assert is_complete(target, ledger=ledger) is False  # not yet pushed

    ledger.ensure_artifact("grid", "all")
    ledger.record_push_batch("grid", [PushResult(unit_id="all", ok=True)])
    assert is_complete(target, ledger=ledger) is True


def test_never_completion_always_false_regardless_of_require_remote(tmp_path, ledger):
    target = StepTarget(
        source_id="modis", step=PipelineStep.FETCH, key="all",
        output_path=str(tmp_path), completion=Completion.NEVER, require_remote=True,
    )
    assert is_complete(target, ledger=ledger) is False


# ---------------------------------------------------------------------------
# local_drift() -- the automatic self-heal trigger (ledger-as-source-of-truth)
# ---------------------------------------------------------------------------


def test_local_drift_false_when_ledger_agrees_with_disk_complete(tmp_path, ledger):
    target = _path_exists_target(tmp_path, exists=True)
    ledger.ensure_artifact("fetch", "2020/h09v05")
    ledger.set_local_state("fetch", "2020/h09v05", LocalState.COMPLETE)
    assert local_drift(target, ledger) is False


def test_local_drift_false_when_ledger_agrees_with_disk_missing(tmp_path, ledger):
    target = _path_exists_target(tmp_path, exists=False)
    ledger.ensure_artifact("fetch", "2020/h09v05")
    assert local_drift(target, ledger) is False


def test_local_drift_true_when_ledger_says_complete_but_disk_missing(tmp_path, ledger):
    target = _path_exists_target(tmp_path, exists=False)
    ledger.ensure_artifact("fetch", "2020/h09v05")
    ledger.set_local_state("fetch", "2020/h09v05", LocalState.COMPLETE)
    assert local_drift(target, ledger) is True


def test_local_drift_true_when_disk_complete_but_ledger_has_no_row(tmp_path, ledger):
    target = _path_exists_target(tmp_path, exists=True)
    assert local_drift(target, ledger) is True


def test_local_drift_false_for_never_completion(tmp_path, ledger):
    target = StepTarget(
        source_id="modis", step=PipelineStep.FETCH, key="all",
        output_path=str(tmp_path), completion=Completion.NEVER,
    )
    assert local_drift(target, ledger) is False


def test_local_drift_true_for_marker_completion_mismatch(tmp_path, ledger):
    output_dir = str(tmp_path / "grid" / "modis.zarr")
    os.makedirs(output_dir, exist_ok=True)
    mark_complete(output_dir)
    target = StepTarget(
        source_id="modis", step=PipelineStep.GRID, key="all",
        output_path=output_dir, completion=Completion.MARKER,
    )
    ledger.ensure_artifact("grid", "all")  # left at default local_state='missing'
    assert local_drift(target, ledger) is True

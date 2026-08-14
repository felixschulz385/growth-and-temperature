"""is_complete()'s local-disk completion policy."""

import os

from src.data.sources.steps import Completion, PipelineStep, StepTarget, is_complete, mark_complete


def _path_exists_target(tmp_path, *, exists=True):
    output_path = str(tmp_path / "2020" / "h09v05.tif")
    if exists:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        open(output_path, "w").close()
    return StepTarget(
        source_id="modis", step=PipelineStep.FETCH, key="2020/h09v05",
        output_path=output_path, completion=Completion.PATH_EXISTS,
    )


def test_path_exists_target_true_when_present(tmp_path):
    target = _path_exists_target(tmp_path, exists=True)
    assert is_complete(target) is True


def test_path_exists_target_false_when_missing(tmp_path):
    target = _path_exists_target(tmp_path, exists=False)
    assert is_complete(target) is False


def test_marker_completion_true_only_after_mark_complete(tmp_path):
    output_dir = str(tmp_path / "grid" / "modis.zarr")
    os.makedirs(output_dir, exist_ok=True)
    target = StepTarget(
        source_id="modis", step=PipelineStep.GRID, key="all",
        output_path=output_dir, completion=Completion.MARKER,
    )
    assert is_complete(target) is False
    mark_complete(output_dir)
    assert is_complete(target) is True


def test_never_completion_always_false(tmp_path):
    target = StepTarget(
        source_id="modis", step=PipelineStep.FETCH, key="all",
        output_path=str(tmp_path), completion=Completion.NEVER,
    )
    assert is_complete(target) is False

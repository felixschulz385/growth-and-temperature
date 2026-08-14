"""is_complete()'s completion policies -- including the local-disk ones
(PATH_EXISTS/MARKER/NEVER) and PRECOMPUTED (decided once at plan time,
e.g. against a remote HPC listing -- src.data.common.fetch.manifest
.resolve_fetch_listing)."""

import os

from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection, is_complete, mark_complete


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


def test_precomputed_completion_reads_meta_not_disk(tmp_path):
    # output_path deliberately doesn't exist on disk -- PRECOMPUTED targets
    # (e.g. a FETCH tile-year judged complete against a remote HPC listing)
    # have no single local path for a bare os.path.exists() to re-check.
    missing_path = str(tmp_path / "nope" / "h09v05.tif")
    complete = StepTarget(
        source_id="modis", step=PipelineStep.FETCH, key="2020/h09v05",
        output_path=missing_path, completion=Completion.PRECOMPUTED, meta={"complete": True},
    )
    outstanding = StepTarget(
        source_id="modis", step=PipelineStep.FETCH, key="2020/h10v05",
        output_path=missing_path, completion=Completion.PRECOMPUTED, meta={"complete": False},
    )
    assert is_complete(complete) is True
    assert is_complete(outstanding) is False


def test_precomputed_completion_defaults_false_without_meta():
    target = StepTarget(
        source_id="modis", step=PipelineStep.FETCH, key="2020/h09v05",
        output_path="/nope", completion=Completion.PRECOMPUTED,
    )
    assert is_complete(target) is False


def test_target_selection_local_only_defaults_true():
    # The safe default -- data run/data plan explicitly opt out
    # (src/cli/data/handlers.py's _selection_from_args), everything else
    # (data summary, _check_requires) keeps this default unchanged.
    assert TargetSelection().local_only is True

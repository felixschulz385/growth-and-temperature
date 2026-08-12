"""``data summary``'s "verified" column -- GRID-output verification
status (src.data.sources.verify), replacing the old completion-only
"complete" column. "-" when the source has no GRID step or GRID has no
targets; "pending" when GRID has targets but none are complete yet; "yes"/
"FAILED (n/m)" once at least one GRID target is complete and gets verified."""

import argparse
import os

import pytest

from src.cli.data import handlers
from src.data.sources.steps import Completion, PipelineStep, StepTarget


def _path_exists_target(tmp_path, name, step, exists):
    output_path = str(tmp_path / name)
    if exists:
        open(output_path, "w").close()
    return StepTarget(source_id="x", step=step, key=name, output_path=output_path, completion=Completion.PATH_EXISTS)


def _never_target(tmp_path):
    return StepTarget(
        source_id="x", step=PipelineStep.FETCH, key="all",
        output_path=str(tmp_path), completion=Completion.NEVER,
    )


# --- _summarize_targets ------------------------------------------------


def test_summarize_targets_empty_is_vacuously_complete():
    summary, complete = handlers._summarize_targets([])
    assert summary == "no targets"
    assert complete is True


def test_summarize_targets_all_complete(tmp_path):
    targets = [_path_exists_target(tmp_path, "a", PipelineStep.GRID, exists=True)]
    summary, complete = handlers._summarize_targets(targets)
    assert summary == "1/1 (100%)"
    assert complete is True


def test_summarize_targets_partial_is_incomplete(tmp_path):
    targets = [
        _path_exists_target(tmp_path, "a", PipelineStep.GRID, exists=True),
        _path_exists_target(tmp_path, "b", PipelineStep.GRID, exists=False),
    ]
    summary, complete = handlers._summarize_targets(targets)
    assert summary == "1/2 (50%)"
    assert complete is False


def test_summarize_targets_fetch_pseudo_target_has_no_complete_concept(tmp_path):
    summary, complete = handlers._summarize_targets([_never_target(tmp_path)])
    assert summary == "no local data"
    assert complete is None

    os.makedirs(tmp_path, exist_ok=True)
    open(tmp_path / "file.tif", "w").close()
    summary, complete = handlers._summarize_targets([_never_target(tmp_path)])
    assert summary == "1 file(s) fetched"
    assert complete is None


# --- _print_source_summary --------------------------------------------


def test_print_source_summary_includes_verified_column(capsys):
    rows = {
        "acag": {"fetch": "3 file(s) fetched", "prepare": "1/1 (100%)", "grid": "1/1 (100%)", "verified": "yes"},
        "modis": {"fetch": "no local data", "prepare": "-", "grid": "0/2 (0%)", "verified": "pending"},
    }
    handlers._print_source_summary(rows)
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].split() == ["source", "fetch", "prepare", "grid", "verified"]
    assert "yes" in lines[1] or "yes" in lines[2]
    assert "pending" in out


def test_print_source_summary_empty(capsys):
    handlers._print_source_summary({})
    assert "No sources found." in capsys.readouterr().out


# --- handle_summary end-to-end ------------------------------------------


def _fake_config(tmp_path):
    return {
        "paths": {"data_root": str(tmp_path / "data_root"), "local_index_dir": str(tmp_path / "index")},
        "sources": {"gadm": {}},
    }


def test_handle_summary_reports_verified_dash_when_grid_has_no_targets(tmp_path, monkeypatch, capsys):
    # No raw fetch file -> PREPARE plans no targets; no prepared levels ->
    # GRID plans no targets either. "verified" is only meaningful once GRID
    # has at least one target.
    monkeypatch.setattr(handlers, "load_config_with_env_vars", lambda path: _fake_config(tmp_path))
    args = argparse.Namespace(log_level="ERROR", debug=False, config="unused.yaml", source="gadm")

    handlers.handle_summary(args)
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("gadm"))
    assert row.split()[-1] == "-"


def test_handle_summary_reports_verified_dash_when_prepare_is_pending(tmp_path, monkeypatch, capsys):
    # A raw fetch file exists -> PREPARE plans a real target, but its output
    # doesn't exist yet -> GRID (which needs PREPARE's ADM_0 file) still
    # plans no targets either, so "verified" stays "-".
    monkeypatch.setattr(handlers, "load_config_with_env_vars", lambda path: _fake_config(tmp_path))
    args = argparse.Namespace(log_level="ERROR", debug=False, config="unused.yaml", source="gadm")

    raw_dir = tmp_path / "data_root" / "misc" / "raw" / "gadm"
    os.makedirs(raw_dir, exist_ok=True)
    open(raw_dir / "gadm_410-levels.zip", "w").close()

    handlers.handle_summary(args)
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("gadm"))
    assert row.split()[-1] == "-"


def test_handle_summary_fetch_column_reports_mismatched_filename_not_generic_count(tmp_path, monkeypatch, capsys):
    # gadm is a ConfiguredFilesFetchMixin source: a file present under the
    # wrong name must be surfaced as a specific mismatch (via verify_fetch()),
    # not folded into a generic "N file(s) fetched" count that looks
    # identical whether the right file is there or not.
    monkeypatch.setattr(handlers, "load_config_with_env_vars", lambda path: _fake_config(tmp_path))
    args = argparse.Namespace(log_level="ERROR", debug=False, config="unused.yaml", source="gadm")

    raw_dir = tmp_path / "data_root" / "misc" / "raw" / "gadm"
    os.makedirs(raw_dir, exist_ok=True)
    open(raw_dir / "some_other_export.zip", "w").close()

    handlers.handle_summary(args)
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("gadm"))
    assert "gadm_410-levels.zip" in row  # what was expected
    assert "some_other_export.zip" in row  # what's actually there
    assert "file(s) fetched" not in row


def _complete_gadm_grid_target(tmp_path, monkeypatch):
    """Plans a real GRID target for gadm (needs PREPARE's ADM_0 vector file
    to exist) and marks its output complete, without writing an actual zarr
    store -- `source.verify_grid()` is monkeypatched separately per-test, so
    nothing ever tries to open the (nonexistent) store's contents."""
    from src.data.sources.steps import mark_complete

    monkeypatch.setattr(handlers, "load_config_with_env_vars", lambda path: _fake_config(tmp_path))
    raw_dir = tmp_path / "data_root" / "misc" / "raw" / "gadm"
    os.makedirs(raw_dir, exist_ok=True)
    open(raw_dir / "gadm_410-levels.zip", "w").close()

    from src.data.pipeline.config import SourceConfig
    from src.data.pipeline.context import PipelineContext
    from src.data.sources.misc.gadm import GadmSource
    from src.data.sources.steps import PipelineStep, TargetSelection

    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    gadm = GadmSource(ctx, SourceConfig.from_dict("gadm", {}))
    vector_dir = gadm.output_root(PipelineStep.PREPARE)
    os.makedirs(vector_dir, exist_ok=True)
    open(os.path.join(vector_dir, "gadm_levelADM_0_simplified.gpkg"), "w").close()

    target = gadm.plan(PipelineStep.GRID, TargetSelection())[0]
    os.makedirs(os.path.dirname(target.output_path), exist_ok=True)
    mark_complete(target.output_path)


def test_handle_summary_reports_verified_yes_when_grid_verification_passes(tmp_path, monkeypatch, capsys):
    from src.data.sources.base import DataSource
    from src.data.sources.verify import VerificationResult

    _complete_gadm_grid_target(tmp_path, monkeypatch)
    monkeypatch.setattr(DataSource, "verify_grid", lambda self, target: VerificationResult(True, "ok"))

    args = argparse.Namespace(log_level="ERROR", debug=False, config="unused.yaml", source="gadm")
    handlers.handle_summary(args)
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("gadm"))
    assert row.split()[-1] == "yes"


def test_handle_summary_reports_verified_failed_when_grid_verification_fails(tmp_path, monkeypatch, capsys):
    from src.data.sources.base import DataSource
    from src.data.sources.verify import VerificationResult

    _complete_gadm_grid_target(tmp_path, monkeypatch)
    monkeypatch.setattr(DataSource, "verify_grid", lambda self, target: VerificationResult(False, "boom"))

    args = argparse.Namespace(log_level="ERROR", debug=False, config="unused.yaml", source="gadm")
    handlers.handle_summary(args)
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("gadm"))
    assert "FAILED (0/1)" in row


def test_handle_summary_unknown_source_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(handlers, "load_config_with_env_vars", lambda path: _fake_config(tmp_path))
    args = argparse.Namespace(log_level="ERROR", debug=False, config="unused.yaml", source="not-a-real-source")
    with pytest.raises(KeyError):
        handlers.handle_summary(args)

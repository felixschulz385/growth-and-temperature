"""``pipeline summary``'s "complete" column -- whether every implemented
non-FETCH step is fully done (FETCH's sync-missing pseudo-target has no
complete/incomplete concept, so it's excluded from the overall verdict)."""

import argparse
import os

import pytest

from src.cli.pipeline import handlers
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


def test_print_source_summary_includes_complete_column(capsys):
    rows = {
        "acag": {"fetch": "3 file(s) fetched", "prepare": "1/1 (100%)", "grid": "1/1 (100%)", "complete": "yes"},
        "modis": {"fetch": "no local data", "prepare": "-", "grid": "0/2 (0%)", "complete": "no"},
    }
    handlers._print_source_summary(rows)
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].split() == ["source", "fetch", "prepare", "grid", "complete"]
    assert "yes" in lines[1] or "yes" in lines[2]
    assert "no" in out


def test_print_source_summary_empty(capsys):
    handlers._print_source_summary({})
    assert "No sources found." in capsys.readouterr().out


# --- handle_summary end-to-end ------------------------------------------


def _fake_config(tmp_path):
    return {
        "paths": {"data_root": str(tmp_path / "data_root"), "local_index_dir": str(tmp_path / "index")},
        "sources": {"gadm": {}},
    }


def test_handle_summary_reports_complete_yes_when_nothing_is_pending(tmp_path, monkeypatch, capsys):
    # No raw fetch file -> PREPARE plans no targets; no prepared levels ->
    # GRID plans no targets either. Nothing pending is vacuously "complete".
    monkeypatch.setattr(handlers, "load_config_with_env_vars", lambda path: _fake_config(tmp_path))
    args = argparse.Namespace(log_level="ERROR", debug=False, config="unused.yaml", source="gadm")

    handlers.handle_summary(args)
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("gadm"))
    assert row.split()[-1] == "yes"


def test_handle_summary_reports_complete_no_when_prepare_is_pending(tmp_path, monkeypatch, capsys):
    # A raw fetch file exists -> PREPARE plans a real target, but its
    # output directory doesn't exist yet -> that step isn't complete.
    monkeypatch.setattr(handlers, "load_config_with_env_vars", lambda path: _fake_config(tmp_path))
    args = argparse.Namespace(log_level="ERROR", debug=False, config="unused.yaml", source="gadm")

    raw_dir = tmp_path / "data_root" / "misc" / "raw" / "gadm"
    os.makedirs(raw_dir, exist_ok=True)
    open(raw_dir / "gadm_410-levels.zip", "w").close()

    handlers.handle_summary(args)
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("gadm"))
    assert row.split()[-1] == "no"


def test_handle_summary_unknown_source_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(handlers, "load_config_with_env_vars", lambda path: _fake_config(tmp_path))
    args = argparse.Namespace(log_level="ERROR", debug=False, config="unused.yaml", source="not-a-real-source")
    with pytest.raises(KeyError):
        handlers.handle_summary(args)

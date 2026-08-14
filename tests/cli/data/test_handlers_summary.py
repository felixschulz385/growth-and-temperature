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
        "acag": {"fetch": "3 file(s) fetched", "prepare": "1/1 (100%)", "grid": "-", "verified": "yes"},
        "modis": {"fetch": "no local data", "prepare": "-", "grid": "-", "verified": "pending"},
    }
    handlers._print_source_summary(rows)
    out = capsys.readouterr().out
    lines = out.splitlines()
    # GRID is display-only omitted -- no registered source declares it any
    # more (module-level _SUMMARY_STEPS docstring).
    assert lines[0].split() == ["source", "fetch", "prepare", "verified"]
    assert "yes" in lines[1] or "yes" in lines[2]
    assert "pending" in out


def test_print_source_summary_empty(capsys):
    handlers._print_source_summary({})
    assert "No sources found." in capsys.readouterr().out


def test_print_source_summary_wraps_long_text_to_fit_narrow_terminal(capsys, monkeypatch):
    import os
    import shutil

    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((60, 24)))
    rows = {
        "plad": {
            "fetch": "0 complete, 26 outstanding, 0 unavailable -- run `data run --step fetch` to discover files",
            "prepare": "no targets",
            "grid": "-",
            "verified": "-",
        },
    }
    handlers._print_source_summary(rows)
    lines = capsys.readouterr().out.splitlines()
    assert all(len(line) <= 60 for line in lines)
    # Every word from the long fetch message survives somewhere in the
    # wrapped output (nothing silently truncated/dropped), even though it
    # no longer fits on one line.
    joined = " ".join(lines)
    for word in rows["plad"]["fetch"].split():
        assert word in joined


def test_print_source_summary_single_line_per_row_when_it_fits(capsys, monkeypatch):
    import os
    import shutil

    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((300, 24)))
    rows = {"acag": {"fetch": "3 file(s) fetched", "prepare": "1/1 (100%)", "grid": "-", "verified": "yes"}}
    handlers._print_source_summary(rows)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3  # header, separator, one row -- no wrap needed


# --- handle_summary end-to-end ------------------------------------------


@pytest.fixture(autouse=True)
def _wide_terminal(monkeypatch):
    """Every test in this module except the dedicated wrapping tests below
    asserts on a single unbroken row line -- force a wide terminal so
    _print_source_summary()'s wrapping never kicks in here. The wrapping
    tests locally override this with a narrow width."""
    import os
    import shutil

    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): os.terminal_size((300, 24)))


def _fake_config(tmp_path):
    return {
        "paths": {"data_root": str(tmp_path / "data_root"), "local_index_dir": str(tmp_path / "index")},
        "sources": {"gadm": {}},
    }


def test_handle_summary_reports_verified_dash_when_grid_has_no_targets(tmp_path, monkeypatch, capsys):
    # No raw fetch file -> PREPARE (gadm's own final, verifiable step)
    # plans no targets. "verified" is only meaningful once it has at least
    # one target.
    monkeypatch.setattr(handlers, "load_config_with_env_vars", lambda path: _fake_config(tmp_path))
    args = argparse.Namespace(log_level="ERROR", debug=False, config="unused.yaml", source="gadm")

    handlers.handle_summary(args)
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("gadm"))
    assert row.split()[-1] == "-"


def test_handle_summary_reports_verified_pending_when_prepare_target_exists_but_incomplete(tmp_path, monkeypatch, capsys):
    # A raw fetch file exists -> PREPARE (gadm's own single remaining step)
    # plans a real target immediately, but its output doesn't exist yet ->
    # "verified" is "pending", not "-".
    monkeypatch.setattr(handlers, "load_config_with_env_vars", lambda path: _fake_config(tmp_path))
    args = argparse.Namespace(log_level="ERROR", debug=False, config="unused.yaml", source="gadm")

    raw_dir = tmp_path / "data_root" / "misc" / "raw" / "gadm"
    os.makedirs(raw_dir, exist_ok=True)
    open(raw_dir / "gadm_410-levels.zip", "w").close()

    handlers.handle_summary(args)
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("gadm"))
    assert row.split()[-1] == "pending"


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
    """Plans a real PREPARE target for gadm -- its own final, verifiable
    output -- and marks its output complete, without writing an actual
    zarr store --
    `source.verify_grid()` is monkeypatched separately per-test, so nothing
    ever tries to open the (nonexistent) store's contents."""
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

    target = gadm.plan(PipelineStep.PREPARE, TargetSelection())[0]
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


# --- _summarize_by_tile / --by-tile -------------------------------------


class _FakeTiledSource:
    """Just enough of a `DataSource` for `_summarize_by_tile()`: a `tile_size`
    attribute and a `ctx` `get_target_geobox()` reads (patched out below, so
    its actual value never matters)."""

    tile_size = 2048

    def __init__(self):
        self.ctx = object()


def test_summarize_by_tile_none_for_non_marker_target(tmp_path):
    target = _path_exists_target(tmp_path, "a", PipelineStep.PREPARE, exists=True)
    assert handlers._summarize_by_tile(_FakeTiledSource(), target) is None


def test_summarize_by_tile_none_when_years_meta_missing(tmp_path):
    target = StepTarget(
        source_id="x", step=PipelineStep.PREPARE, key="a",
        output_path=str(tmp_path / "a.zarr"), completion=Completion.MARKER, meta={},
    )
    assert handlers._summarize_by_tile(_FakeTiledSource(), target) is None


def test_summarize_by_tile_none_when_source_has_no_tile_size(tmp_path):
    target = StepTarget(
        source_id="x", step=PipelineStep.PREPARE, key="a",
        output_path=str(tmp_path / "a.zarr"), completion=Completion.MARKER, meta={"years": [2020]},
    )

    class _NoTileSize:
        ctx = object()

    assert handlers._summarize_by_tile(_NoTileSize(), target) is None


def test_summarize_by_tile_reports_per_unit_counts(tmp_path, monkeypatch):
    target = StepTarget(
        source_id="x", step=PipelineStep.PREPARE, key="a",
        output_path=str(tmp_path / "a.zarr"), completion=Completion.MARKER, meta={"years": [2020, 2021]},
    )
    source = _FakeTiledSource()

    monkeypatch.setattr("src.data.common.geobox.get_target_geobox", lambda ctx: "fake-geobox")
    monkeypatch.setattr(
        "src.data.common.prepare.driver.prepare_status",
        lambda output_path, years, geobox, tile_size: {"complete": 3, "outstanding": 1, "unavailable": 0},
    )

    detail = handlers._summarize_by_tile(source, target)
    assert detail == "3/4 complete, 1 outstanding, 0 unavailable"


def test_handle_summary_by_tile_shows_per_target_breakdown(tmp_path, monkeypatch, capsys):
    # acag is a real tiled PREPARE source (run_tiled_prepare()-shaped) --
    # with --by-tile, its PREPARE column reports the per-unit breakdown
    # instead of the collapsed complete/total summary.
    config = {
        "paths": {"data_root": str(tmp_path / "data_root"), "local_index_dir": str(tmp_path / "index")},
        "sources": {"acag": {}},
    }
    monkeypatch.setattr(handlers, "load_config_with_env_vars", lambda path: config)
    monkeypatch.setattr(
        handlers, "_summarize_by_tile", lambda source, target: "2/4 complete, 2 outstanding, 0 unavailable"
    )

    from src.data.sources.steps import Completion as _Completion, StepTarget as _StepTarget

    fake_target = _StepTarget(
        source_id="acag", step=PipelineStep.PREPARE, key="pm25",
        output_path=str(tmp_path / "acag.zarr"), completion=_Completion.MARKER, meta={"years": [2020]},
    )
    from src.data.sources.acag import AcagSource

    monkeypatch.setattr(AcagSource, "plan", lambda self, step, selection: [fake_target] if step.value == "prepare" else [])

    args = argparse.Namespace(log_level="ERROR", debug=False, config="unused.yaml", source="acag", by_tile=True)
    handlers.handle_summary(args)
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("acag"))
    assert "pm25: 2/4 complete, 2 outstanding, 0 unavailable" in row


# --- _summarize_fetch / cached_entrypoint_counts: network-free by design -


class _FakeYearlySource:
    """Just enough of a `RemoteFileCatalog` source for `_summarize_fetch()`:
    an entrypoint source with nothing cached yet, but a declared, static
    (network-free) `get_all_entrypoints()` and a real `filename_to_entrypoint()`
    -- the shape EOG VIIRS/ESACCI/GLASS/ACAG are in on a fresh checkout."""

    has_entrypoints = True
    STATIC_ENTRYPOINTS = True
    RAW_LISTING_DEPTH = 1

    def __init__(self, ctx, cfg):
        self.ctx = ctx
        self.cfg = cfg

    def get_all_entrypoints(self):
        if not self.cfg.year_range:
            return []
        return [{"year": y} for y in range(self.cfg.year_range[0], self.cfg.year_range[1] + 1)]

    def filename_to_entrypoint(self, relative_path):
        import re

        match = re.search(r"(\d{4})", relative_path)
        return {"year": int(match.group(1))} if match else None

    @staticmethod
    def get_file_hash(url):
        import hashlib

        return hashlib.md5(url.encode("utf-8")).hexdigest()


def _make_yearly_source(tmp_path, year_range=None):
    from src.data.pipeline.config import SourceConfig
    from src.data.pipeline.context import PipelineContext

    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    raw = {"data_path": "yearly"}
    if year_range is not None:
        raw["year_range"] = list(year_range)
    cfg = SourceConfig.from_dict("yearly", raw)
    return _FakeYearlySource(ctx, cfg)


def test_summarize_fetch_falls_back_to_entrypoint_counts_when_nothing_cached(tmp_path):
    from src.data.sources import layout

    source = _make_yearly_source(tmp_path, year_range=(2020, 2022))
    raw_root = layout.raw_root(source.ctx.data_root, source.cfg.data_path, layout=source.ctx.layout)
    os.makedirs(raw_root, exist_ok=True)
    open(os.path.join(raw_root, "file_2020.tif"), "w").close()

    summary = handlers._summarize_fetch(source, detailed=False)
    assert summary == "1 complete, 2 outstanding, 0 unavailable"


def test_summarize_fetch_reports_not_yet_crawled_without_year_range(tmp_path):
    source = _make_yearly_source(tmp_path, year_range=None)
    summary = handlers._summarize_fetch(source, detailed=False)
    assert summary == "not yet crawled -- run `data run --step fetch` to discover files"


# --- _summarize_fetch_targets: MODIS-shaped FETCH (no crawl, real targets) -


def test_summarize_fetch_targets_buckets_complete_outstanding_unavailable(tmp_path):
    from src.data.common.fetch import manifest

    raw_root = tmp_path / "raw"
    os.makedirs(raw_root, exist_ok=True)
    status_dir = str(raw_root)
    targets = [
        _path_exists_target(raw_root, "a", PipelineStep.FETCH, exists=True),
        _path_exists_target(raw_root, "b", PipelineStep.FETCH, exists=False),
        _path_exists_target(raw_root, "c", PipelineStep.FETCH, exists=False),
    ]
    manifest.record_failure(status_dir, "c", "boom", max_attempts=1, permanent=True)  # -> unavailable

    class _FakeSource:
        def output_root(self, step):
            return status_dir

    summary = handlers._summarize_fetch_targets(_FakeSource(), targets, detailed=False)
    assert summary == "1 complete, 1 outstanding, 1 unavailable"


def test_summarize_fetch_targets_detailed_splits_never_attempted_and_retrying(tmp_path):
    from src.data.common.fetch import manifest

    raw_root = tmp_path / "raw"
    os.makedirs(raw_root, exist_ok=True)
    status_dir = str(raw_root)
    targets = [
        _path_exists_target(raw_root, "never", PipelineStep.FETCH, exists=False),
        _path_exists_target(raw_root, "retrying", PipelineStep.FETCH, exists=False),
    ]
    manifest.record_failure(status_dir, "retrying", "transient", max_attempts=5)

    class _FakeSource:
        def output_root(self, step):
            return status_dir

    summary = handlers._summarize_fetch_targets(_FakeSource(), targets, detailed=True)
    assert summary == "0 complete, 2 outstanding, 0 unavailable (outstanding: 1 never attempted, 1 retrying)"


def test_handle_summary_shows_verify_fetch_for_source_without_fetch_step(tmp_path, monkeypatch, capsys):
    # snl_mining declares STEPS=(PREPARE,) -- no FETCH -- but still depends
    # on a manual export (verify_fetch()). The FETCH column must report that
    # instead of the "-" every other stepless column gets.
    config = {
        "paths": {"data_root": str(tmp_path / "data_root"), "local_index_dir": str(tmp_path / "index")},
        "sources": {"snl_mining": {}},
    }
    monkeypatch.setattr(handlers, "load_config_with_env_vars", lambda path: config)
    args = argparse.Namespace(log_level="ERROR", debug=False, config="unused.yaml", source="snl_mining")

    handlers.handle_summary(args)
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("snl_mining"))
    assert "manual export" in row
    assert "MISSING" in row


def test_handle_summary_unknown_source_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(handlers, "load_config_with_env_vars", lambda path: _fake_config(tmp_path))
    args = argparse.Namespace(log_level="ERROR", debug=False, config="unused.yaml", source="not-a-real-source")
    with pytest.raises(KeyError):
        handlers.handle_summary(args)

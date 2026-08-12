"""`_heal_local_drift()` -- the "...or when conflict is detected" half of
ledger-as-source-of-truth (the other half is the explicit `data
reconcile` command). `handle_plan`/`handle_run` call this after detecting
`local_drift()` on a read-only ledger pass; this file tests the healing
function directly against a real (temp-file) ledger.
"""

from types import SimpleNamespace

from src.cli.data import handlers
from src.data.common.ledger.schema import LocalState
from src.data.common.ledger.store import SourceLedger
from src.data.sources.steps import PipelineStep


def _fake_source(tmp_path, data_path="acag/pm25"):
    ctx = SimpleNamespace(local_index_dir=str(tmp_path / "index"))
    return SimpleNamespace(ID="acag", ctx=ctx, data_path=data_path)


def test_heal_local_drift_noop_on_empty_list(tmp_path):
    source = _fake_source(tmp_path)
    # Must not raise, and must not require local_index_dir to even exist.
    handlers._heal_local_drift(source, PipelineStep.PREPARE, [])


def test_heal_local_drift_noop_when_no_local_index_dir(tmp_path):
    source = SimpleNamespace(ID="acag", ctx=SimpleNamespace(local_index_dir=None), data_path="acag/pm25")
    # No ledger to open -- must not raise even with a non-empty drift list.
    handlers._heal_local_drift(source, PipelineStep.PREPARE, [("2020", LocalState.COMPLETE)])


def test_heal_local_drift_corrects_flagged_rows(tmp_path):
    source = _fake_source(tmp_path)
    local_ledger_path = str(tmp_path / "index" / "acag_pm25.duckdb")
    with SourceLedger.open(local_ledger_path, data_path="acag/pm25") as ledger:
        ledger.ensure_artifact("prepare", "2019")
        ledger.ensure_artifact("prepare", "2020")
        ledger.set_local_state("prepare", "2019", LocalState.COMPLETE)  # wrong: disk says missing
        ledger.set_local_state("prepare", "2020", LocalState.MISSING)  # wrong: disk says complete

    handlers._heal_local_drift(
        source, PipelineStep.PREPARE, [("2019", LocalState.MISSING), ("2020", LocalState.COMPLETE)]
    )

    with SourceLedger.open(local_ledger_path, data_path="acag/pm25", read_only=True) as ledger:
        assert ledger.local_state("prepare", "2019") == LocalState.MISSING
        assert ledger.local_state("prepare", "2020") == LocalState.COMPLETE


def test_handle_plan_and_run_heal_drift_end_to_end(tmp_path):
    import argparse

    from src.data.common.ledger.paths import ledger_path
    from src.data.pipeline.config import SourceConfig
    from src.data.pipeline.context import PipelineContext
    from src.data.sources.base import DataSource
    from src.data.sources.steps import Completion, StepTarget

    class _FakeSource(DataSource):
        ID = "fake"
        STEPS = (PipelineStep.PREPARE,)

        def _plan(self, step, selection):
            output_path = str(tmp_path / "2020.parquet")
            open(output_path, "w").close()  # disk says complete
            return [
                StepTarget(
                    source_id="fake", step=PipelineStep.PREPARE, key="2020",
                    output_path=output_path, completion=Completion.PATH_EXISTS,
                )
            ]

        def _execute(self, target):
            return True

    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("fake", {"data_path": "fake"})

    local_ledger_path = ledger_path(ctx.local_index_dir, "fake")
    with SourceLedger.open(local_ledger_path, data_path="fake") as ledger:
        ledger.ensure_artifact("prepare", "2020")
        ledger.set_local_state("prepare", "2020", LocalState.MISSING)  # ledger stale: disk already has the file

    args = argparse.Namespace(
        source="fake", config=None, step="prepare", log_level="WARNING", debug=False,
        years=None, year_range=None, keys=None, override=False,
    )

    def _fake_build(args_):
        return _FakeSource(ctx, cfg), ctx

    orig_build = handlers._build
    handlers._build = _fake_build
    try:
        handlers.handle_plan(args)
    finally:
        handlers._build = orig_build

    with SourceLedger.open(local_ledger_path, data_path="fake", read_only=True) as ledger:
        assert ledger.local_state("prepare", "2020") == LocalState.COMPLETE

"""`data run --step fetch`'s per-file auto-transfer: for a source with a
real per-target FETCH list (MODIS/GLASS), each target should be pushed to
HPC right after it downloads successfully, not batched until the whole run
finishes -- and one target failing must not block an already-successful
target's push. (Driver-based sources -- acag/esacci/ntl_harm/eog -- push
per-file inside `run_fetch()` itself instead; see
tests/data/common/fetch/test_driver.py.)
"""

import argparse
import os
from types import SimpleNamespace

import pytest

from src.cli.data.handlers import _push_one_target, handle_run
from src.data.common.hpc.push import PushResult
from src.data.sources.steps import Completion, PipelineStep, StepTarget


class _FakePusher:
    def __init__(self, ok=True, error=None):
        self.calls = []
        self._ok = ok
        self._error = error

    def push_unit(self, unit):
        self.calls.append(unit)
        return PushResult(unit_id=unit.unit_id, ok=self._ok, error=self._error)


def _source(tmp_path, *, last_fetch_output_path=None):
    ctx = SimpleNamespace(data_root=str(tmp_path / "data_root"))
    source = SimpleNamespace(ctx=ctx)
    if last_fetch_output_path is not None:
        source._last_fetch_output_path = last_fetch_output_path
    return source


def _target(output_path, key="2019/h18v04"):
    return StepTarget(
        source_id="modis", step=PipelineStep.FETCH, key=key, output_path=output_path, completion=Completion.PATH_EXISTS
    )


def test_push_one_target_uses_target_output_path_by_default(tmp_path):
    output_path = os.path.join(str(tmp_path / "data_root"), "prepared", "modis", "2019", "h18v04.tif")
    pusher = _FakePusher()
    source = _source(tmp_path)

    _push_one_target(pusher, source, _target(output_path))

    assert len(pusher.calls) == 1
    unit = pusher.calls[0]
    assert unit.unit_id == "2019/h18v04"
    assert unit.local_path == output_path
    assert unit.remote_path == os.path.relpath(output_path, source.ctx.data_root).replace(os.sep, "/")


def test_push_one_target_prefers_last_fetch_output_path_when_source_sets_it(tmp_path):
    # GLASS's real filename (unpredictable trailing processing-date) is only
    # known at execute time -- target.output_path can be a synthetic
    # "pending" placeholder that was never actually written to.
    data_root = str(tmp_path / "data_root")
    real_path = os.path.join(data_root, "raw", "glass_avhrr", "2019", "GLASS08B31.V40.A2019364.2021259.hdf")
    placeholder_path = os.path.join(data_root, "raw", "glass_avhrr", "2019", "pending.2019364.hdf")
    pusher = _FakePusher()
    source = _source(tmp_path, last_fetch_output_path=real_path)

    _push_one_target(pusher, source, _target(placeholder_path, key="2019/364"))

    assert pusher.calls[0].local_path == real_path
    assert pusher.calls[0].local_path != placeholder_path


def test_push_one_target_logs_but_does_not_raise_on_push_failure(tmp_path, caplog):
    output_path = os.path.join(str(tmp_path / "data_root"), "a.tif")
    pusher = _FakePusher(ok=False, error="simulated failure")
    source = _source(tmp_path)

    with caplog.at_level("WARNING"):
        _push_one_target(pusher, source, _target(output_path))  # must not raise

    assert "simulated failure" in caplog.text


# --- handle_run wiring: pushes as it goes, a later failure doesn't undo an
# earlier success's push -----------------------------------------------------


class _FakeCfg:
    source_id = "modis"
    override = False
    raw: dict = {}


class _FakeFetchSource:
    """Minimal stand-in exposing exactly what `handle_run` touches."""

    def __init__(self, data_root, *, fail_keys=()):
        self.ctx = SimpleNamespace(data_root=data_root, ssh_target="user@host:base", key_file=None)
        self.cfg = _FakeCfg()
        self._fail_keys = set(fail_keys)
        self.executed = []
        self.closed = False

    def plan(self, step, selection):
        return [
            _target(os.path.join(self.ctx.data_root, "a.tif"), key="a"),
            _target(os.path.join(self.ctx.data_root, "b.tif"), key="b"),
        ]

    def execute(self, target):
        self.executed.append(target.key)
        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)
        open(target.output_path, "w").close()
        return target.key not in self._fail_keys

    def close(self):
        self.closed = True


def _run_args(**overrides):
    base = dict(
        log_level="ERROR", debug=False, config="unused", source="modis", step="fetch",
        override=False, years=None, keys=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_handle_run_pushes_each_target_as_it_succeeds_not_only_at_the_end(tmp_path, monkeypatch):
    import src.cli.data.handlers as handlers_module

    fake_source = _FakeFetchSource(str(tmp_path / "data_root"), fail_keys={"b"})
    monkeypatch.setattr(handlers_module, "_build", lambda args, step: (fake_source, {}))
    monkeypatch.setattr(handlers_module, "resolve_transfer_mode", lambda source: "auto")
    monkeypatch.setattr(handlers_module, "_maybe_auto_transfer", lambda source, step: None)

    pusher = _FakePusher()
    monkeypatch.setattr("src.data.common.hpc.client.HPCClient", lambda target, key_file=None: object())
    monkeypatch.setattr("src.data.common.hpc.push.HPCPusher", lambda client: pusher)

    with pytest.raises(RuntimeError):
        # target "b" fails -- handle_run still raises for it after the loop.
        handlers_module.handle_run(_run_args())

    # "a" succeeded and was pushed immediately; "b" failed and was never
    # pushed -- one target failing doesn't block an already-successful
    # target's push (unlike the old end-of-run-only auto-transfer, which
    # skipped the whole batch if anything in the run failed).
    assert [u.unit_id for u in pusher.calls] == ["a"]
    assert fake_source.closed is True


def test_handle_run_does_not_push_when_transfer_mode_is_manual(tmp_path, monkeypatch):
    import src.cli.data.handlers as handlers_module

    fake_source = _FakeFetchSource(str(tmp_path / "data_root"))
    monkeypatch.setattr(handlers_module, "_build", lambda args, step: (fake_source, {}))
    monkeypatch.setattr(handlers_module, "resolve_transfer_mode", lambda source: "manual")
    monkeypatch.setattr(handlers_module, "_maybe_auto_transfer", lambda source, step: None)

    def _boom(*a, **k):
        raise AssertionError("must not construct an HPC client when transfer_mode is manual")

    monkeypatch.setattr("src.data.common.hpc.client.HPCClient", _boom)

    handlers_module.handle_run(_run_args())  # both targets succeed -- must not raise

    assert sorted(fake_source.executed) == ["a", "b"]

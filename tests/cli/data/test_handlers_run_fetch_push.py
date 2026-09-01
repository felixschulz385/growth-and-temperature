"""`data run --step fetch`'s auto-transfer: for a source with a real
per-target FETCH list (MODIS/GLASS), successfully-fetched targets are queued
and pushed in batches of up to `tar_max_files` (`push_batched()`'s
tar+extract amortization) instead of one `push_unit()` (3 SSH connections:
mkdir/rsync/verify) per file -- and one target failing must not lose an
already-successful target's push. (Driver-based sources -- acag/esacci/
ntl_harm/eog -- push per-file inside `run_fetch()` itself instead; see
tests/data/common/fetch/test_driver.py.)
"""

import argparse
import os
from types import SimpleNamespace

import pytest

from src.cli.data.handlers import _push_pending_batch, _resolve_push_unit, handle_run
from src.data.common.hpc.push import PushResult
from src.data.sources.steps import Completion, PipelineStep, StepTarget


class _FakePusher:
    def __init__(self, ok=True, error=None):
        self.calls = []
        self.batch_calls = []
        self._ok = ok
        self._error = error

    def push_unit(self, unit):
        self.calls.append(unit)
        return PushResult(unit_id=unit.unit_id, ok=self._ok, error=self._error)

    def push_batched(self, units, remote_base_dir, *, max_files, max_bytes):
        self.batch_calls.append((list(units), remote_base_dir, max_files, max_bytes))
        return [PushResult(unit_id=u.unit_id, ok=self._ok, error=self._error) for u in units]


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


def test_resolve_push_unit_uses_target_output_path_by_default(tmp_path):
    output_path = os.path.join(str(tmp_path / "data_root"), "prepared", "modis", "2019", "h18v04.tif")
    source = _source(tmp_path)

    unit = _resolve_push_unit(source, _target(output_path))

    assert unit.unit_id == "2019/h18v04"
    assert unit.local_path == output_path
    assert unit.remote_path == os.path.relpath(output_path, source.ctx.data_root).replace(os.sep, "/")


def test_resolve_push_unit_prefers_last_fetch_output_path_when_source_sets_it(tmp_path):
    # GLASS's real filename (unpredictable trailing processing-date) is only
    # known at execute time -- target.output_path can be a synthetic
    # "pending" placeholder that was never actually written to.
    data_root = str(tmp_path / "data_root")
    real_path = os.path.join(data_root, "raw", "glass_avhrr", "2019", "GLASS08B31.V40.A2019364.2021259.hdf")
    placeholder_path = os.path.join(data_root, "raw", "glass_avhrr", "2019", "pending.2019364.hdf")
    source = _source(tmp_path, last_fetch_output_path=real_path)

    unit = _resolve_push_unit(source, _target(placeholder_path, key="2019/364"))

    assert unit.local_path == real_path
    assert unit.local_path != placeholder_path


def test_push_pending_batch_single_unit_falls_back_to_push_unit(tmp_path):
    # Tarring one file has no amortization to offer -- a lone unit (a
    # trailing remainder, or a source that rarely pushes) goes through
    # push_unit() directly rather than push_batched().
    output_path = os.path.join(str(tmp_path / "data_root"), "a.tif")
    source = _source(tmp_path)
    pusher = _FakePusher()

    _push_pending_batch(pusher, [_resolve_push_unit(source, _target(output_path))], tar_max_files=100, tar_max_size_mb=500)

    assert [u.unit_id for u in pusher.calls] == ["2019/h18v04"]
    assert pusher.batch_calls == []


def test_push_pending_batch_single_unit_logs_but_does_not_raise_on_push_failure(tmp_path, caplog):
    output_path = os.path.join(str(tmp_path / "data_root"), "a.tif")
    source = _source(tmp_path)
    pusher = _FakePusher(ok=False, error="simulated failure")

    with caplog.at_level("WARNING"):
        _push_pending_batch(  # must not raise
            pusher, [_resolve_push_unit(source, _target(output_path))], tar_max_files=100, tar_max_size_mb=500,
        )

    assert "simulated failure" in caplog.text


def test_push_pending_batch_multiple_units_uses_push_batched_with_rebased_paths(tmp_path):
    data_root = str(tmp_path / "data_root")
    source = _source(tmp_path)
    output_a = os.path.join(data_root, "raw", "modis", "2019", "h18v04.tif")
    output_b = os.path.join(data_root, "raw", "modis", "2019", "h19v05.tif")
    pending = [
        _resolve_push_unit(source, _target(output_a, key="2019/h18v04")),
        _resolve_push_unit(source, _target(output_b, key="2019/h19v05")),
    ]
    pusher = _FakePusher()

    _push_pending_batch(pusher, pending, tar_max_files=100, tar_max_size_mb=500)

    assert pusher.calls == []
    assert len(pusher.batch_calls) == 1
    units, remote_base_dir, max_files, max_bytes = pusher.batch_calls[0]
    assert remote_base_dir == "raw/modis/2019"
    assert sorted((u.unit_id, u.remote_path) for u in units) == [
        ("2019/h18v04", "h18v04.tif"),
        ("2019/h19v05", "h19v05.tif"),
    ]
    assert (max_files, max_bytes) == (100, 500 * 1024 * 1024)


def test_push_pending_batch_multiple_units_logs_partial_failures(tmp_path, caplog):
    data_root = str(tmp_path / "data_root")
    source = _source(tmp_path)
    pending = [
        _resolve_push_unit(source, _target(os.path.join(data_root, "2019", "a.tif"), key="a")),
        _resolve_push_unit(source, _target(os.path.join(data_root, "2019", "b.tif"), key="b")),
    ]
    pusher = _FakePusher(ok=False, error="simulated failure")

    with caplog.at_level("WARNING"):
        _push_pending_batch(pusher, pending, tar_max_files=100, tar_max_size_mb=500)

    assert "simulated failure" in caplog.text


# --- handle_run wiring: queues successes and flushes in batches, a later
# failure doesn't undo an earlier success's push --------------------------


class _FakeCfg:
    source_id = "modis"
    override = False
    raw: dict = {}


class _FakeFetchSource:
    """Minimal stand-in exposing exactly what `handle_run` touches."""

    def __init__(self, data_root, *, fail_keys=(), keys=("a", "b"), raw=None):
        self.ctx = SimpleNamespace(data_root=data_root, ssh_target="user@host:base", key_file=None)
        self.cfg = _FakeCfg()
        if raw is not None:
            self.cfg.raw = raw
        self._fail_keys = set(fail_keys)
        self._keys = keys
        self.executed = []
        self.closed = False

    def plan(self, step, selection):
        return [_target(os.path.join(self.ctx.data_root, f"{key}.tif"), key=key) for key in self._keys]

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


def test_handle_run_flushes_successful_pushes_even_when_a_later_target_fails(tmp_path, monkeypatch):
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

    # "a" succeeded and was queued+flushed (as a lone unit, via push_unit);
    # "b" failed and was never queued -- one target failing doesn't lose an
    # already-successful target's push.
    assert [u.unit_id for u in pusher.calls] == ["a"]
    assert fake_source.closed is True


def test_handle_run_batches_pushes_once_tar_max_files_is_reached(tmp_path, monkeypatch):
    import src.cli.data.handlers as handlers_module

    fake_source = _FakeFetchSource(
        str(tmp_path / "data_root"), keys=("a", "b", "c"), raw={"download": {"tar_max_files": 2}},
    )
    monkeypatch.setattr(handlers_module, "_build", lambda args, step: (fake_source, {}))
    monkeypatch.setattr(handlers_module, "resolve_transfer_mode", lambda source: "auto")
    monkeypatch.setattr(handlers_module, "_maybe_auto_transfer", lambda source, step: None)

    pusher = _FakePusher()
    monkeypatch.setattr("src.data.common.hpc.client.HPCClient", lambda target, key_file=None: object())
    monkeypatch.setattr("src.data.common.hpc.push.HPCPusher", lambda client: pusher)

    handlers_module.handle_run(_run_args())  # all three succeed -- must not raise

    # First 2 successes hit the tar_max_files=2 threshold mid-loop and flush
    # as one push_batched() call; the trailing 1 flushes as a lone push_unit()
    # after the loop.
    assert len(pusher.batch_calls) == 1
    assert sorted(u.unit_id for u in pusher.batch_calls[0][0]) == ["a", "b"]
    assert [u.unit_id for u in pusher.calls] == ["c"]


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

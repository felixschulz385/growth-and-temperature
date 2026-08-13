"""``data transfer``'s push-strategy routing (_push_transfer_units) and
its ledger-tracked skip-already-verified behavior (handle_transfer) --
docs/design/10-fetch-ledger.md. No live SSH target needed.
"""

import argparse
import os
import tarfile

import pytest

from src.cli.data.handlers import _push_transfer_units, _run_transfer_pass
from src.data.common.hpc.push import HPCPusher
from src.data.common.ledger.store import SourceLedger
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.steps import PipelineStep, TransferUnit


class _FakeHPCClient:
    def __init__(self, base_path="/remote/base"):
        self.base_path = base_path
        self.remote_files: set[str] = set()
        self.rsync_calls = []

    def _resolve(self, remote_path):
        if remote_path.startswith("/") or not self.base_path:
            return remote_path
        return f"{self.base_path}/{remote_path}"

    def ensure_directory(self, remote_path):
        return True

    def rsync_transfer(self, source_path, target_path, source_is_local, options, show_progress):
        self.rsync_calls.append((source_path, target_path, source_is_local))
        if not target_path.endswith(".tar.gz"):
            # Direct single-file push (no tar/extract step) -- simulate the
            # file landing so check_file_exists() finds it.
            self.remote_files.add(self._resolve(target_path))
        return True, "ok"

    def extract_tar(self, tar_path, extraction_dir):
        local_tar = next((src for src, dst, is_local in self.rsync_calls if dst == tar_path and is_local), None)
        if local_tar and os.path.exists(local_tar):
            full_dir = self._resolve(extraction_dir)
            with tarfile.open(local_tar) as tar:
                for name in tar.getnames():
                    self.remote_files.add(f"{full_dir.rstrip('/')}/{name}")
        return True

    def check_file_exists(self, remote_path):
        return self._resolve(remote_path) in self.remote_files

    def check_files_exist(self, remote_paths):
        return {p: self._resolve(p) in self.remote_files for p in remote_paths}

    def execute_command(self, command):
        if command.startswith("find"):
            prefix = command.split("'")[1].rstrip("/")
            matches = [p for p in self.remote_files if p.startswith(prefix + "/")]
            return True, "\n".join(matches), ""
        return True, "", ""


def _write_file(path, content=b"data"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
    return path


def test_single_unit_uses_push_unit_not_batched(tmp_path):
    client = _FakeHPCClient()
    pusher = HPCPusher(client)
    local = _write_file(str(tmp_path / "acag.zarr" / "0.0"))
    units = [TransferUnit(unit_id="all", local_path=str(tmp_path / "acag.zarr"), remote_path="grid/acag.zarr")]

    results = _push_transfer_units(pusher, units, tar_max_files=100, tar_max_size_mb=500)

    assert len(results) == 1
    assert results[0].ok is True
    # A directory unit tars with its own basename as arcname, landing at
    # exactly the given remote_path -- confirms push_unit's (not
    # push_batched's) tar/extract path ran.
    assert "/remote/base/grid/acag.zarr/0.0" in client.remote_files


def test_many_single_file_units_sharing_a_tree_use_push_batched(tmp_path):
    client = _FakeHPCClient()
    pusher = HPCPusher(client)
    units = [
        TransferUnit(
            unit_id=f"{year}/h09v05",
            local_path=_write_file(str(tmp_path / str(year) / "h09v05.tif")),
            remote_path=f"prepared/modis/{year}/h09v05.tif",
        )
        for year in (2019, 2020, 2021)
    ]

    results = _push_transfer_units(pusher, units, tar_max_files=100, tar_max_size_mb=500)

    assert len(results) == 3
    assert all(r.ok for r in results)
    # Batched into the shared "prepared/modis" ancestor, one rsync per tar
    # batch (not one rsync per file) -- confirms push_batched ran.
    tar_pushes = [c for c in client.rsync_calls if c[1].endswith(".tar.gz")]
    assert len(tar_pushes) == 1
    for year in (2019, 2020, 2021):
        assert f"/remote/base/prepared/modis/{year}/h09v05.tif" in client.remote_files


def test_many_single_file_units_split_into_multiple_batches(tmp_path):
    client = _FakeHPCClient()
    pusher = HPCPusher(client)
    units = [
        TransferUnit(
            unit_id=f"{year}/h09v05",
            local_path=_write_file(str(tmp_path / str(year) / "h09v05.tif")),
            remote_path=f"prepared/modis/{year}/h09v05.tif",
        )
        for year in range(2015, 2021)
    ]

    results = _push_transfer_units(pusher, units, tar_max_files=2, tar_max_size_mb=500)

    assert len(results) == 6
    assert all(r.ok for r in results)
    tar_pushes = [c for c in client.rsync_calls if c[1].endswith(".tar.gz")]
    assert len(tar_pushes) == 3  # 6 units / max_files=2 per batch


def test_mixed_files_and_directories_falls_back_to_concurrent_per_unit(tmp_path):
    client = _FakeHPCClient()
    pusher = HPCPusher(client)
    file_unit = TransferUnit(
        unit_id="a", local_path=_write_file(str(tmp_path / "a.tif")), remote_path="grid/a.tif",
    )
    dir_unit = TransferUnit(
        # `os.path.dirname`, not `.rsplit("/", 1)` -- the latter is a no-op
        # on Windows (`_write_file` returns a backslash-separated path
        # there), silently leaving `local_path` pointed at the file "0.0"
        # instead of its parent directory "b.zarr".
        unit_id="b", local_path=os.path.dirname(_write_file(str(tmp_path / "b.zarr" / "0.0"))),
        remote_path="grid/b.zarr",
    )

    results = _push_transfer_units(pusher, [file_unit, dir_unit], tar_max_files=100, tar_max_size_mb=500)

    assert {r.unit_id for r in results} == {"a", "b"}
    assert all(r.ok for r in results)
    assert "/remote/base/grid/a.tif" in client.remote_files
    assert "/remote/base/grid/b.zarr/0.0" in client.remote_files


# --- _run_transfer_pass: reconciles both ledger copies, not just push ------


class _FakeHPCClientWithLedgerPull(_FakeHPCClient):
    """`_FakeHPCClient`, but any `.duckdb` pull is served from a real,
    pre-built remote ledger file -- lets `_run_transfer_pass`'s new
    `merge_from_remote()` call (src/cli/data/handlers.py) exercise real
    merge behavior instead of a no-op "no remote copy yet"."""

    def __init__(self, remote_ledger_file: str, base_path="/remote/base"):
        super().__init__(base_path=base_path)
        self._remote_ledger_file = remote_ledger_file

    def check_file_exists(self, remote_path):
        if remote_path.endswith(".duckdb"):
            return True
        return super().check_file_exists(remote_path)

    def rsync_transfer(self, source_path, target_path, source_is_local, options, show_progress):
        if not source_is_local and source_path.endswith(".duckdb"):
            import shutil

            shutil.copy(self._remote_ledger_file, target_path)
            self.rsync_calls.append((source_path, target_path, source_is_local))
            return True, "ok"
        return super().rsync_transfer(source_path, target_path, source_is_local, options, show_progress)


class _FakeSource:
    """Minimal stand-in exposing exactly what `_run_transfer_pass` touches:
    `transfer_units()`, `data_path`, `cfg.raw`, `ctx.staging_dir`/
    `ctx.local_index_dir`."""

    def __init__(self, ctx, cfg, units):
        self.ctx = ctx
        self.cfg = cfg
        self._units = units

    @property
    def data_path(self):
        return self.cfg.data_path

    def transfer_units(self, step):
        return self._units


def test_run_transfer_pass_merges_remote_ledger_before_pushing_local_back(tmp_path):
    # A row `record_push_batch` below never touches, representing state some
    # OTHER machine already pushed and recorded remotely -- if
    # `_run_transfer_pass` only overwrites remote from local (the old
    # behavior), this row is invisible to it; if it merges first, this row
    # ends up in the LOCAL ledger too by the time the call returns.
    remote_ledger_file = str(tmp_path / "remote.duckdb")
    with SourceLedger.open(remote_ledger_file, data_path="fake") as seed:
        seed.ensure_artifact("prepare", "other-machine-unit", local_path="/elsewhere/x")
        seed.set_local_state("prepare", "other-machine-unit", "complete")
        seed.set_remote_state("prepare", "other-machine-unit", "verified")

    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"),
        staging_dir=str(tmp_path / "staging"), ssh_target="user@host:/remote/base",
    )
    cfg = SourceConfig.from_dict("fake", {"data_path": "fake"})
    local_path = _write_file(str(tmp_path / "grid" / "a.tif"))
    units = [TransferUnit(unit_id="a", local_path=local_path, remote_path="grid/a.tif")]
    source = _FakeSource(ctx, cfg, units)

    client = _FakeHPCClientWithLedgerPull(remote_ledger_file)
    args = argparse.Namespace(override=False, source="fake")
    local_ledger_path = str(tmp_path / "index" / "fake.duckdb")

    results = _run_transfer_pass(args, source, PipelineStep.GRID, local_ledger_path, client)
    assert len(results) == 1
    assert results[0].ok is True

    with SourceLedger.open(local_ledger_path, data_path="fake", read_only=True) as local_ledger:
        # This pass's own push, recorded locally as always.
        assert local_ledger.remote_state("grid", "a") == "verified"
        # The OTHER machine's row, only present here because merge_from_remote()
        # ran before push_to_remote() re-uploaded the local copy.
        assert local_ledger.local_state("prepare", "other-machine-unit") == "complete"
        assert local_ledger.remote_state("prepare", "other-machine-unit") == "verified"


def test_run_transfer_pass_resets_local_state_after_cleanup_on_success(tmp_path):
    # HPCPusher's cleanup_local=True (the default, always used by
    # _push_transfer_units) deletes the local file once a push succeeds --
    # local_state must reflect that instead of still saying 'complete' for
    # bytes that no longer exist on disk.
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"),
        staging_dir=str(tmp_path / "staging"), ssh_target="user@host:/remote/base",
    )
    cfg = SourceConfig.from_dict("fake", {"data_path": "fake"})
    local_path = _write_file(str(tmp_path / "grid" / "a.tif"))
    units = [TransferUnit(unit_id="a", local_path=local_path, remote_path="grid/a.tif")]
    source = _FakeSource(ctx, cfg, units)

    client = _FakeHPCClient()
    args = argparse.Namespace(override=False, source="fake")
    local_ledger_path = str(tmp_path / "index" / "fake.duckdb")

    results = _run_transfer_pass(args, source, PipelineStep.GRID, local_ledger_path, client)
    assert results[0].ok is True
    assert not os.path.exists(local_path)  # HPCPusher really did clean it up

    with SourceLedger.open(local_ledger_path, data_path="fake", read_only=True) as local_ledger:
        assert local_ledger.local_state("grid", "a") == "missing"
        assert local_ledger.remote_state("grid", "a") == "verified"


def test_run_transfer_pass_leaves_local_state_alone_on_push_failure(tmp_path):
    class _FailingHPCClient(_FakeHPCClient):
        def rsync_transfer(self, source_path, target_path, source_is_local, options, show_progress):
            self.rsync_calls.append((source_path, target_path, source_is_local))
            return False, "simulated rsync failure"

    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"),
        staging_dir=str(tmp_path / "staging"), ssh_target="user@host:/remote/base",
    )
    cfg = SourceConfig.from_dict("fake", {"data_path": "fake"})
    local_path = _write_file(str(tmp_path / "grid" / "a.tif"))
    units = [TransferUnit(unit_id="a", local_path=local_path, remote_path="grid/a.tif")]
    source = _FakeSource(ctx, cfg, units)

    local_ledger_path = str(tmp_path / "index" / "fake.duckdb")
    # Seed local_state='complete' up front -- the realistic pre-push state
    # (the file genuinely exists) -- so a failing push can be told apart
    # from "just left at the schema default", which a fresh row would also
    # read as 'missing' for reasons unrelated to the fix under test.
    with SourceLedger.open(local_ledger_path, data_path="fake") as seed:
        seed.ensure_artifact("grid", "a", local_path=local_path)
        seed.set_local_state("grid", "a", "complete")

    client = _FailingHPCClient()
    args = argparse.Namespace(override=False, source="fake")

    results = _run_transfer_pass(args, source, PipelineStep.GRID, local_ledger_path, client)
    assert results[0].ok is False
    assert os.path.exists(local_path)  # never cleaned up -- push failed

    with SourceLedger.open(local_ledger_path, data_path="fake", read_only=True) as local_ledger:
        # Must NOT have been reset -- the (ok=False) result is filtered out
        # of the local_state reset, since the file is genuinely still there.
        assert local_ledger.local_state("grid", "a") == "complete"
        assert local_ledger.remote_state("grid", "a") == "failed"

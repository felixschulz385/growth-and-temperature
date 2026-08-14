"""``data transfer``'s push-strategy routing (_push_transfer_units) and its
skip-already-on-HPC behavior (_run_transfer_pass) -- a direct remote
existence check (`check_paths_exist`), no local bookkeeping. No live SSH
target needed.
"""

import argparse
import os
import tarfile

from src.cli.data.handlers import _push_transfer_units, _run_transfer_pass
from src.data.common.hpc.push import HPCPusher
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

    def check_paths_exist(self, remote_paths):
        # `[-e]` semantics: a directory counts as existing if any file
        # under it does, not just an exact match -- mirrors the real
        # HPCClient.check_paths_exist's directory-aware behavior.
        result = {}
        for path in remote_paths:
            full = self._resolve(path)
            result[path] = full in self.remote_files or any(
                f.startswith(full.rstrip("/") + "/") for f in self.remote_files
            )
        return result

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


# --- _run_transfer_pass: skip-check is a direct remote existence check ----


class _FakeSource:
    """Minimal stand-in exposing exactly what `_run_transfer_pass` touches:
    `transfer_units()`, `cfg.raw`."""

    def __init__(self, ctx, cfg, units):
        self.ctx = ctx
        self.cfg = cfg
        self._units = units

    def transfer_units(self, step):
        return self._units


def _make_source(tmp_path, units):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"),
        staging_dir=str(tmp_path / "staging"), ssh_target="user@host:/remote/base",
    )
    cfg = SourceConfig.from_dict("fake", {"data_path": "fake"})
    return _FakeSource(ctx, cfg, units)


def test_run_transfer_pass_pushes_pending_units(tmp_path):
    local_path = _write_file(str(tmp_path / "grid" / "a.tif"))
    units = [TransferUnit(unit_id="a", local_path=local_path, remote_path="grid/a.tif")]
    source = _make_source(tmp_path, units)

    client = _FakeHPCClient()
    args = argparse.Namespace(override=False, source="fake")

    results = _run_transfer_pass(args, source, PipelineStep.GRID, client)
    assert len(results) == 1
    assert results[0].ok is True
    assert "/remote/base/grid/a.tif" in client.remote_files


def test_run_transfer_pass_skips_units_that_already_exist_remotely(tmp_path):
    local_path = _write_file(str(tmp_path / "grid" / "a.tif"))
    units = [TransferUnit(unit_id="a", local_path=local_path, remote_path="grid/a.tif")]
    source = _make_source(tmp_path, units)

    client = _FakeHPCClient()
    client.remote_files.add("/remote/base/grid/a.tif")  # already pushed
    args = argparse.Namespace(override=False, source="fake")

    results = _run_transfer_pass(args, source, PipelineStep.GRID, client)
    assert results == []
    assert client.rsync_calls == []  # never re-pushed
    assert os.path.exists(local_path)  # never touched, so never cleaned up


def test_run_transfer_pass_skip_check_is_directory_aware(tmp_path):
    # A directory-shaped unit (e.g. a Zarr store) already fully present
    # remotely must be recognized as existing even though no single remote
    # path exactly matches `remote_path` itself -- `[-f]`-only semantics
    # would report it missing and re-push it on every pass.
    local_dir = str(tmp_path / "grid" / "acag.zarr")
    _write_file(os.path.join(local_dir, "0.0"))
    units = [TransferUnit(unit_id="all", local_path=local_dir, remote_path="grid/acag.zarr")]
    source = _make_source(tmp_path, units)

    client = _FakeHPCClient()
    client.remote_files.add("/remote/base/grid/acag.zarr/0.0")  # already pushed
    args = argparse.Namespace(override=False, source="fake")

    results = _run_transfer_pass(args, source, PipelineStep.GRID, client)
    assert results == []
    assert client.rsync_calls == []


def test_run_transfer_pass_override_repushes_existing_units(tmp_path):
    local_path = _write_file(str(tmp_path / "grid" / "a.tif"))
    units = [TransferUnit(unit_id="a", local_path=local_path, remote_path="grid/a.tif")]
    source = _make_source(tmp_path, units)

    client = _FakeHPCClient()
    client.remote_files.add("/remote/base/grid/a.tif")
    args = argparse.Namespace(override=True, source="fake")

    results = _run_transfer_pass(args, source, PipelineStep.GRID, client)
    assert len(results) == 1
    assert results[0].ok is True
    assert len(client.rsync_calls) == 1


def test_run_transfer_pass_cleans_up_local_on_success(tmp_path):
    local_path = _write_file(str(tmp_path / "grid" / "a.tif"))
    units = [TransferUnit(unit_id="a", local_path=local_path, remote_path="grid/a.tif")]
    source = _make_source(tmp_path, units)

    client = _FakeHPCClient()
    args = argparse.Namespace(override=False, source="fake")

    results = _run_transfer_pass(args, source, PipelineStep.GRID, client)
    assert results[0].ok is True
    assert not os.path.exists(local_path)  # HPCPusher's cleanup_local=True default


def test_run_transfer_pass_leaves_local_file_alone_on_push_failure(tmp_path):
    class _FailingHPCClient(_FakeHPCClient):
        def rsync_transfer(self, source_path, target_path, source_is_local, options, show_progress):
            self.rsync_calls.append((source_path, target_path, source_is_local))
            return False, "simulated rsync failure"

    local_path = _write_file(str(tmp_path / "grid" / "a.tif"))
    units = [TransferUnit(unit_id="a", local_path=local_path, remote_path="grid/a.tif")]
    source = _make_source(tmp_path, units)

    client = _FailingHPCClient()
    args = argparse.Namespace(override=False, source="fake")

    results = _run_transfer_pass(args, source, PipelineStep.GRID, client)
    assert results[0].ok is False
    assert os.path.exists(local_path)  # never cleaned up -- push failed

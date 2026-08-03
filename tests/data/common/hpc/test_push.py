"""HPCPusher's batching/tar/extract/verify/cleanup logic against a fake
HPCClient that simulates a remote filesystem in memory -- no live SSH target
needed. The fake's `extract_tar` actually reads the *local* tar file that was
rsynced (available on disk in-process) to know which remote paths should
exist afterwards, so verification exercises the real tar member names.
"""

import os
import tarfile

import pytest

from src.data.common.hpc.push import HPCPusher, PushUnit


class _FakeHPCClient:
    def __init__(self, base_path="/remote/base"):
        self.base_path = base_path
        self.dirs_ensured = []
        self.rsync_calls = []
        self.extracted = []
        self.removed_commands = []
        self.remote_files: set[str] = set()
        self.fail_rsync = False
        self.fail_extract = False
        self.verify_overrides: dict[str, bool] = {}

    def ensure_directory(self, remote_path):
        self.dirs_ensured.append(remote_path)
        return True

    def rsync_transfer(self, source_path, target_path, source_is_local, options, show_progress):
        self.rsync_calls.append((source_path, target_path, source_is_local))
        if self.fail_rsync:
            return False, "rsync failed"
        return True, "ok"

    def _resolve(self, remote_path):
        # Mirrors HPCClient's own relative-path resolution against base_path
        # (client.py's ensure_directory/check_file_exists/check_files_exist
        # all do exactly this before touching the remote filesystem).
        if remote_path.startswith("/") or not self.base_path:
            return remote_path
        return f"{self.base_path}/{remote_path}"

    def extract_tar(self, tar_path, extraction_dir):
        self.extracted.append((tar_path, extraction_dir))
        if self.fail_extract:
            return False
        local_tar = next((src for src, dst, is_local in self.rsync_calls if dst == tar_path and is_local), None)
        if local_tar and os.path.exists(local_tar):
            full_dir = self._resolve(extraction_dir)
            with tarfile.open(local_tar) as tar:
                for name in tar.getnames():
                    self.remote_files.add(f"{full_dir.rstrip('/')}/{name}")
        return True

    def check_file_exists(self, remote_path):
        full = self._resolve(remote_path)
        return self.verify_overrides.get(full, full in self.remote_files)

    def check_files_exist(self, remote_paths):
        return {p: self.verify_overrides.get(self._resolve(p), self._resolve(p) in self.remote_files) for p in remote_paths}

    def execute_command(self, command):
        if command.startswith("rm -f"):
            self.removed_commands.append(command)
        if command.startswith("find"):
            # Emulate `find <dir> -type f`: return whatever's under that dir.
            prefix = command.split("'")[1].rstrip("/")
            matches = [p for p in self.remote_files if p.startswith(prefix + "/")]
            return True, "\n".join(matches), ""
        return True, "", ""


@pytest.fixture
def client():
    return _FakeHPCClient()


def _write_file(path: str, content: bytes = b"data") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
    return path


def test_push_batched_single_batch_succeeds(tmp_path, client):
    a = _write_file(str(tmp_path / "a.nc"), b"aaaa")
    b = _write_file(str(tmp_path / "b.nc"), b"bb")
    units = [PushUnit(unit_id="a", local_path=a, remote_path="a.nc"), PushUnit(unit_id="b", local_path=b, remote_path="b.nc")]

    pusher = HPCPusher(client)
    results = pusher.push_batched(units, remote_base_dir="acag/pm25/raw")

    assert {r.unit_id: r.ok for r in results} == {"a": True, "b": True}
    assert {r.unit_id: r.bytes for r in results} == {"a": 4, "b": 2}
    assert not os.path.exists(a)  # cleaned up after verified push
    assert not os.path.exists(b)
    assert client.dirs_ensured == ["acag/pm25/raw/_tar", "acag/pm25/raw"]
    assert len(client.removed_commands) == 1  # remote tar cleaned up


def test_push_batched_preserves_nested_arcname(tmp_path, client):
    src = _write_file(str(tmp_path / "2020" / "055" / "x.hdf"), b"x")
    units = [PushUnit(unit_id="x", local_path=src, remote_path="2020/055/x.hdf")]

    pusher = HPCPusher(client)
    pusher.push_batched(units, remote_base_dir="glass/raw")
    assert "/remote/base/glass/raw/2020/055/x.hdf" in client.remote_files


def test_push_batched_splits_on_max_files(tmp_path, client):
    units = [
        PushUnit(unit_id=str(i), local_path=_write_file(str(tmp_path / f"{i}.nc")), remote_path=f"{i}.nc")
        for i in range(5)
    ]
    pusher = HPCPusher(client)
    results = pusher.push_batched(units, remote_base_dir="root", max_files=2)

    assert len(results) == 5
    assert all(r.ok for r in results)
    # 5 units, max 2 per batch -> 3 batches -> 3 tar transfers.
    tar_pushes = [c for c in client.rsync_calls if c[1].endswith(".tar.gz")]
    assert len(tar_pushes) == 3


def test_push_batched_keeps_oversized_single_unit_in_its_own_batch(tmp_path, client):
    big = _write_file(str(tmp_path / "big.nc"), b"x" * 1000)
    small = _write_file(str(tmp_path / "small.nc"), b"y")
    units = [PushUnit(unit_id="big", local_path=big, remote_path="big.nc"), PushUnit(unit_id="small", local_path=small, remote_path="small.nc")]

    pusher = HPCPusher(client)
    results = pusher.push_batched(units, remote_base_dir="root", max_bytes=10)
    assert len(results) == 2
    assert all(r.ok for r in results)


def test_push_batched_rsync_failure_marks_whole_batch_failed(tmp_path, client):
    a = _write_file(str(tmp_path / "a.nc"))
    client.fail_rsync = True
    pusher = HPCPusher(client)
    results = pusher.push_batched([PushUnit(unit_id="a", local_path=a, remote_path="a.nc")], remote_base_dir="root")
    assert results == [results[0]]
    assert results[0].ok is False
    assert results[0].error == "rsync failed"
    assert os.path.exists(a)  # not cleaned up on failure


def test_push_batched_verification_failure(tmp_path, client):
    a = _write_file(str(tmp_path / "a.nc"))
    pusher = HPCPusher(client)
    # Force the extracted file to appear missing on verify.
    client.verify_overrides["/remote/base/root/a.nc"] = False
    results = pusher.push_batched([PushUnit(unit_id="a", local_path=a, remote_path="a.nc")], remote_base_dir="root")
    assert results[0].ok is False
    assert results[0].error == "verification failed"
    assert os.path.exists(a)


def test_push_unit_single_file(tmp_path, client):
    local = _write_file(str(tmp_path / "modis_2020.tif"), b"tiffdata")
    unit = PushUnit(unit_id="2020", local_path=local, remote_path="grid/2020/2020.tif")

    pusher = HPCPusher(client)
    client.remote_files.add("/remote/base/grid/2020/2020.tif")  # simulate rsync having landed it
    result = pusher.push_unit(unit)

    assert result.ok is True
    assert result.bytes == len(b"tiffdata")
    assert not os.path.exists(local)
    assert client.rsync_calls == [(local, "grid/2020/2020.tif", True)]
    # No tar/extract for a single file.
    assert client.extracted == []


def test_push_unit_directory_tars_and_extracts(tmp_path, client):
    store_dir = tmp_path / "acag.zarr"
    _write_file(str(store_dir / "0.0"), b"chunk")
    _write_file(str(store_dir / ".zattrs"), b"{}")
    unit = PushUnit(unit_id="all", local_path=str(store_dir), remote_path="grid/legacy_4326/acag.zarr")

    pusher = HPCPusher(client)
    result = pusher.push_unit(unit)

    assert result.ok is True
    assert not os.path.exists(store_dir)
    assert "/remote/base/grid/legacy_4326/acag.zarr/0.0" in client.remote_files
    assert "/remote/base/grid/legacy_4326/acag.zarr/.zattrs" in client.remote_files


def test_push_unit_missing_local_path_fails_without_touching_client(tmp_path, client):
    unit = PushUnit(unit_id="missing", local_path=str(tmp_path / "nope"), remote_path="x")
    result = HPCPusher(client).push_unit(unit)
    assert result.ok is False
    assert "does not exist" in result.error
    assert client.rsync_calls == []


def test_push_units_concurrent_runs_all(tmp_path, client):
    units = [
        PushUnit(unit_id=str(i), local_path=_write_file(str(tmp_path / f"f{i}.tif")), remote_path=f"f{i}.tif")
        for i in range(6)
    ]
    for i in range(6):
        client.remote_files.add(f"/remote/base/f{i}.tif")

    results = HPCPusher(client).push_units_concurrent(units, max_workers=3)
    assert {r.unit_id for r in results} == {str(i) for i in range(6)}
    assert all(r.ok for r in results)


def test_push_batched_no_local_cleanup_when_disabled(tmp_path, client):
    a = _write_file(str(tmp_path / "a.nc"))
    pusher = HPCPusher(client)
    pusher.push_batched([PushUnit(unit_id="a", local_path=a, remote_path="a.nc")], remote_base_dir="root", cleanup_local=False)
    assert os.path.exists(a)

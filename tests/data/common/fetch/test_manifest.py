import os
from types import SimpleNamespace

from src.data.common import statusfile
from src.data.common.fetch import manifest


def _write_file(path: str, size: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"x" * size)


def test_snapshot_local_listing_empty_for_missing_root(tmp_path):
    assert manifest.snapshot_local_listing(str(tmp_path / "nope")) == {}


def test_snapshot_local_listing_walks_nested_files(tmp_path):
    root = str(tmp_path / "raw")
    _write_file(os.path.join(root, "2020", "a.nc"), 2000)
    listing = manifest.snapshot_local_listing(root)
    assert listing["2020/a.nc"].size == 2000


def test_snapshot_local_listing_always_excludes_status_subdir(tmp_path):
    root = str(tmp_path / "raw")
    _write_file(os.path.join(root, "a.nc"), 2000)
    statusfile.write(statusfile.status_path(root, "a"), {"status": "retrying", "attempts": 1})
    listing = manifest.snapshot_local_listing(root)
    assert set(listing) == {"a.nc"}


def test_snapshot_local_listing_max_depth_prunes_deeper_files(tmp_path):
    root = str(tmp_path / "raw")
    _write_file(os.path.join(root, "2020", "a.nc"), 2000)
    _write_file(os.path.join(root, "2020", "extra", "b.nc"), 2000)  # deeper than depth=2 expects
    listing = manifest.snapshot_local_listing(root, max_depth=2)
    assert set(listing) == {"2020/a.nc"}


def test_snapshot_local_listing_max_depth_one_stays_flat(tmp_path):
    root = str(tmp_path / "raw")
    _write_file(os.path.join(root, "flat.nc"), 2000)
    _write_file(os.path.join(root, "2020", "a.nc"), 2000)  # would need depth>=2 to be found
    listing = manifest.snapshot_local_listing(root, max_depth=1)
    assert set(listing) == {"flat.nc"}


def test_snapshot_local_listing_max_depth_three_matches_year_day_file(tmp_path):
    root = str(tmp_path / "raw")
    _write_file(os.path.join(root, "2020", "001", "a.nc"), 2000)
    listing = manifest.snapshot_local_listing(root, max_depth=3)
    assert set(listing) == {"2020/001/a.nc"}


class _FakeHPCClient:
    def __init__(self, stdout: str, ok: bool = True):
        self._stdout = stdout
        self._ok = ok

    def execute_command(self, command):
        return self._ok, self._stdout, ""


def test_snapshot_remote_listing_parses_find_output():
    client = _FakeHPCClient("2000 1700000000.5 2020/a.nc\n5000 1700000001.0 2021/b.nc\n")
    listing = manifest.snapshot_remote_listing(client, "/remote/raw")
    assert listing["2020/a.nc"].size == 2000
    assert listing["2021/b.nc"].size == 5000


def test_snapshot_remote_listing_empty_on_command_failure():
    client = _FakeHPCClient("", ok=False)
    assert manifest.snapshot_remote_listing(client, "/remote/raw") == {}


def test_plan_fetch_buckets_complete_outstanding_and_unavailable(tmp_path):
    root = str(tmp_path / "raw")
    _write_file(os.path.join(root, "present.nc"), 2000)
    listing = manifest.snapshot_local_listing(root)

    status_dir = str(tmp_path / "raw")
    statusfile.write(
        statusfile.status_path(status_dir, "dead"), {"status": manifest.STATUS_UNAVAILABLE, "attempts": 5}
    )

    required = [
        manifest.RequiredFile(unit_id="present", relative_path="present.nc", url="https://x/present.nc"),
        manifest.RequiredFile(unit_id="missing", relative_path="missing.nc", url="https://x/missing.nc"),
        manifest.RequiredFile(unit_id="dead", relative_path="dead.nc", url="https://x/dead.nc"),
    ]

    plan = manifest.plan_fetch(required, listing, status_dir)
    assert [r.unit_id for r in plan.complete] == ["present"]
    assert [r.unit_id for r in plan.outstanding] == ["missing"]
    assert [r.unit_id for r in plan.unavailable] == ["dead"]
    assert plan.counts() == {"complete": 1, "outstanding": 1, "unavailable": 1}


def test_plan_fetch_treats_undersized_file_as_outstanding(tmp_path):
    root = str(tmp_path / "raw")
    _write_file(os.path.join(root, "truncated.nc"), 10)  # below _MIN_PLAUSIBLE_BYTES
    listing = manifest.snapshot_local_listing(root)

    required = [manifest.RequiredFile(unit_id="t", relative_path="truncated.nc", url="https://x/t.nc")]
    plan = manifest.plan_fetch(required, listing, str(tmp_path / "raw"))
    assert [r.unit_id for r in plan.outstanding] == ["t"]


def test_plan_fetch_checks_expected_bytes_when_declared(tmp_path):
    root = str(tmp_path / "raw")
    _write_file(os.path.join(root, "a.nc"), 5000)
    listing = manifest.snapshot_local_listing(root)

    required = [
        manifest.RequiredFile(unit_id="a", relative_path="a.nc", url="https://x/a.nc", expected_bytes=5000),
        manifest.RequiredFile(unit_id="b", relative_path="a.nc", url="https://x/a.nc", expected_bytes=9999),
    ]
    plan = manifest.plan_fetch(required, listing, str(tmp_path / "raw"))
    assert [r.unit_id for r in plan.complete] == ["a"]
    assert [r.unit_id for r in plan.outstanding] == ["b"]


def test_record_failure_retries_then_flips_to_unavailable_at_max_attempts(tmp_path):
    status_dir = str(tmp_path / "raw")
    status = None
    for _ in range(3):
        status = manifest.record_failure(status_dir, "u", "boom", max_attempts=3)
    assert status == manifest.STATUS_UNAVAILABLE

    data = statusfile.read(statusfile.status_path(status_dir, "u"))
    assert data["attempts"] == 3
    assert data["last_error"] == "boom"


def test_record_failure_permanent_flips_immediately(tmp_path):
    status_dir = str(tmp_path / "raw")
    status = manifest.record_failure(status_dir, "u", "404 gone", max_attempts=5, permanent=True)
    assert status == manifest.STATUS_UNAVAILABLE


def test_clear_failure_removes_status_file(tmp_path):
    status_dir = str(tmp_path / "raw")
    manifest.record_failure(status_dir, "u", "boom", max_attempts=5)
    path = statusfile.status_path(status_dir, "u")
    assert os.path.exists(path)
    manifest.clear_failure(status_dir, "u")
    assert not os.path.exists(path)


def test_remote_listing_for_local_root_returns_none_without_ssh_target(tmp_path):
    ctx = SimpleNamespace(ssh_target=None, data_root=str(tmp_path), key_file=None)
    assert manifest.remote_listing_for_local_root(ctx, str(tmp_path / "raw")) is None


def test_remote_listing_for_local_root_queries_the_mirrored_remote_path(tmp_path, monkeypatch):
    captured = {}

    class _FakeClient:
        def __init__(self, target, key_file=None):
            captured["target"] = target
            captured["key_file"] = key_file
            self.base_path = "remote_base"

        def execute_command(self, command):
            captured["command"] = command
            return True, "2000 1700000000.0 2020/a.nc\n", ""

    monkeypatch.setattr("src.data.common.hpc.client.HPCClient", _FakeClient)

    data_root = str(tmp_path / "data_root")
    local_root = os.path.join(data_root, "modis", "raw")
    ctx = SimpleNamespace(ssh_target="user@host:remote_base", data_root=data_root, key_file=None)

    listing = manifest.remote_listing_for_local_root(ctx, local_root)
    assert listing["2020/a.nc"].size == 2000
    assert captured["target"] == "user@host:remote_base"
    # The remote `find` root mirrors local_root's offset from ctx.data_root.
    assert "remote_base/modis/raw" in captured["command"]


def test_remote_listing_for_local_root_excludes_status_subdir(tmp_path, monkeypatch):
    class _FakeClient:
        def __init__(self, target, key_file=None):
            self.base_path = "remote_base"

        def execute_command(self, command):
            stdout = f"2000 1700000000.0 a.nc\n10 1700000000.0 {statusfile.STATUS_SUBDIR}/a.json\n"
            return True, stdout, ""

    monkeypatch.setattr("src.data.common.hpc.client.HPCClient", _FakeClient)
    ctx = SimpleNamespace(ssh_target="user@host:remote_base", data_root=str(tmp_path), key_file=None)
    listing = manifest.remote_listing_for_local_root(ctx, str(tmp_path / "raw"))
    assert set(listing) == {"a.nc"}


class _FakeSource:
    def __init__(self, ctx, transfer_mode):
        self.ctx = ctx
        self.cfg = SimpleNamespace(source_id="fake", raw={"transfer_mode": transfer_mode})
        self.ID = "fake"


def test_resolve_fetch_listing_falls_back_to_local_when_manual(tmp_path):
    root = str(tmp_path / "raw")
    _write_file(os.path.join(root, "a.nc"), 2000)
    ctx = SimpleNamespace(ssh_target="user@host:base", data_root=str(tmp_path), key_file=None)
    source = _FakeSource(ctx, "manual")

    listing, from_remote = manifest.resolve_fetch_listing(source, root)
    assert from_remote is False
    assert set(listing) == {"a.nc"}


def test_resolve_fetch_listing_falls_back_to_local_when_no_ssh_target(tmp_path):
    root = str(tmp_path / "raw")
    _write_file(os.path.join(root, "a.nc"), 2000)
    ctx = SimpleNamespace(ssh_target=None, data_root=str(tmp_path), key_file=None)
    source = _FakeSource(ctx, "auto")

    listing, from_remote = manifest.resolve_fetch_listing(source, root)
    assert from_remote is False
    assert set(listing) == {"a.nc"}


def test_resolve_fetch_listing_uses_remote_when_auto_and_ssh_target_configured(tmp_path, monkeypatch):
    class _FakeClient:
        def __init__(self, target, key_file=None):
            self.base_path = "base"

        def execute_command(self, command):
            return True, "2000 1700000000.0 remote_only.nc\n", ""

    monkeypatch.setattr("src.data.common.hpc.client.HPCClient", _FakeClient)
    root = str(tmp_path / "raw")  # deliberately empty locally -- only remote has the file
    ctx = SimpleNamespace(ssh_target="user@host:base", data_root=str(tmp_path), key_file=None)
    source = _FakeSource(ctx, "auto")

    listing, from_remote = manifest.resolve_fetch_listing(source, root)
    assert from_remote is True
    assert set(listing) == {"remote_only.nc"}


def test_resolve_fetch_listing_allow_remote_false_forces_local_even_when_auto(tmp_path, monkeypatch):
    class _FakeClient:
        def __init__(self, target, key_file=None):
            self.base_path = "base"

        def execute_command(self, command):
            raise AssertionError("must not consult the remote target when allow_remote=False")

    monkeypatch.setattr("src.data.common.hpc.client.HPCClient", _FakeClient)
    root = str(tmp_path / "raw")
    _write_file(os.path.join(root, "a.nc"), 2000)
    ctx = SimpleNamespace(ssh_target="user@host:base", data_root=str(tmp_path), key_file=None)
    source = _FakeSource(ctx, "auto")

    listing, from_remote = manifest.resolve_fetch_listing(source, root, allow_remote=False)
    assert from_remote is False
    assert set(listing) == {"a.nc"}

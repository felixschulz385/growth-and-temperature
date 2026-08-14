import os

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

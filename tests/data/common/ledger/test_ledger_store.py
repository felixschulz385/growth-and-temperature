"""SourceLedger's CRUD/query surface against a temp DuckDB file.

docs/design/10-fetch-ledger.md. No fixture data is carried over from the old
UnifiedDataIndex/TransferManifest Parquet formats -- this is a ground-up
replacement, tested against its own new schema only.
"""

import os

import pytest

from src.data.common.ledger.schema import LocalState, RemoteState
from src.data.common.ledger.store import DownloadResult, PushResult, SourceLedger


@pytest.fixture
def ledger(tmp_path):
    path = str(tmp_path / "acag_pm25.duckdb")
    with SourceLedger.open(path, data_path="acag/pm25") as led:
        yield led


def test_open_creates_schema_and_file(tmp_path):
    path = str(tmp_path / "sub" / "esacci_landcover.duckdb")
    with SourceLedger.open(path, data_path="esacci/landcover") as led:
        assert led.stats("fetch") == {
            "total": 0, "local_complete": 0, "remote_verified": 0, "failed": 0, "total_bytes": 0,
        }
    assert os.path.exists(path)


def test_add_remote_files_seeds_artifacts_and_dedupes(ledger):
    files = [("2020/foo.nc", "https://example.com/foo.nc"), ("2021/bar.nc", "https://example.com/bar.nc")]
    added = ledger.add_remote_files(files, get_file_hash=lambda url: url.split("/")[-1])
    assert added == 2
    assert ledger.stats("fetch")["total"] == 2

    # Re-adding the same files is a no-op for the count, not an error.
    added_again = ledger.add_remote_files(files, get_file_hash=lambda url: url.split("/")[-1])
    assert added_again == 0
    assert ledger.stats("fetch")["total"] == 2


def test_pending_fetch_orders_by_size_smallest_first(ledger):
    ledger.add_remote_files(
        [("a.nc", "https://x/a.nc"), ("b.nc", "https://x/b.nc"), ("c.nc", "https://x/c.nc")],
        get_file_hash=lambda url: url.split("/")[-1].replace(".nc", ""),
    )
    ledger.record_download_batch([
        DownloadResult(file_hash="a", ok=True, local_path="/tmp/a", bytes=300),
        DownloadResult(file_hash="b", ok=True, local_path="/tmp/b", bytes=100),
    ])
    # "c" has no bytes recorded yet (NULL) -- NULLS LAST puts it after a/b.
    units = ledger.pending_fetch(limit=10)
    assert [u.file_hash for u in units] == ["b", "a", "c"]


def test_pending_fetch_excludes_remote_verified(ledger):
    ledger.add_remote_files([("a.nc", "https://x/a.nc")], get_file_hash=lambda url: "a")
    assert len(ledger.pending_fetch(10)) == 1

    ledger.record_push_batch("fetch", [PushResult(unit_id="a", ok=True, bytes=100)])
    assert ledger.pending_fetch(10) == []


def test_pending_fetch_retries_failed(ledger):
    ledger.add_remote_files([("a.nc", "https://x/a.nc")], get_file_hash=lambda url: "a")
    ledger.record_push_batch("fetch", [PushResult(unit_id="a", ok=False, error="rsync failed")])
    units = ledger.pending_fetch(10)
    assert [u.file_hash for u in units] == ["a"]


def test_attempts_only_increments_on_failure_not_on_every_download_or_push(ledger):
    """A successful download/push is not a "spent attempt" -- only a failed
    one is. Otherwise a file that downloads fine every cycle but keeps
    failing to push burns the shared `attempts` budget twice as fast as one
    that only fails to download (each cycle: 1 successful download + 1
    failed push used to cost 2 attempts, not 1)."""
    ledger.add_remote_files([("a.nc", "https://x/a.nc")], get_file_hash=lambda url: "a")

    ledger.record_download_batch([DownloadResult(file_hash="a", ok=True, local_path="/tmp/a", bytes=10)])
    ledger.record_push_batch("fetch", [PushResult(unit_id="a", ok=False, error="rsync failed")])
    # One real failure (the push) -- attempts must read 1, not 2.
    units = ledger.pending_fetch(10, max_attempts=2)
    assert [u.file_hash for u in units] == ["a"]

    ledger.record_download_batch([DownloadResult(file_hash="a", ok=True, local_path="/tmp/a", bytes=10)])
    ledger.record_push_batch("fetch", [PushResult(unit_id="a", ok=False, error="rsync failed")])
    # Two real (push) failures now -- excluded once attempts reaches max_attempts=2.
    assert ledger.pending_fetch(10, max_attempts=2) == []


def test_completed_fetch_files_requires_remote_verified(ledger):
    ledger.add_remote_files([("2020/a.nc", "https://x/a.nc")], get_file_hash=lambda url: "a")
    assert ledger.completed_fetch_files() == []

    ledger.record_download_batch([DownloadResult(file_hash="a", ok=True, local_path="/tmp/a", bytes=10)])
    # local-only complete (not yet pushed) still doesn't count -- matches the
    # old system's "completed" meaning HPC-verified, never local-only.
    assert ledger.completed_fetch_files() == []

    ledger.record_push_batch("fetch", [PushResult(unit_id="a", ok=True, bytes=10)])
    assert ledger.completed_fetch_files() == ["2020/a.nc"]


def test_completed_fetch_files_filters_by_year(ledger):
    # Year is extracted from the filename itself (matching the old
    # UnifiedDataIndex's Path(relative_path).name behavior) -- a leading
    # directory component alone (e.g. "2020/a.nc") does not carry a year.
    ledger.add_remote_files(
        [("2020/2020.nc", "https://x/a.nc"), ("2021/2021.nc", "https://x/b.nc")],
        get_file_hash=lambda url: url.split("/")[-1].replace(".nc", ""),
    )
    ledger.record_push_batch("fetch", [
        PushResult(unit_id="a", ok=True),
        PushResult(unit_id="b", ok=True),
    ])
    assert ledger.completed_fetch_files(year=2020) == ["2020/2020.nc"]
    assert ledger.completed_fetch_files(year=2021) == ["2021/2021.nc"]


def test_ensure_artifact_and_local_remote_state(ledger):
    ledger.ensure_artifact("prepare", "2020", local_path="/data/2020.zarr")
    assert ledger.local_state("prepare", "2020") == LocalState.MISSING
    assert ledger.remote_state("prepare", "2020") == RemoteState.MISSING

    ledger.set_local_state("prepare", "2020", LocalState.COMPLETE, size_bytes=1234)
    assert ledger.local_state("prepare", "2020") == LocalState.COMPLETE

    ledger.set_remote_state("prepare", "2020", RemoteState.VERIFIED)
    assert ledger.remote_state("prepare", "2020") == RemoteState.VERIFIED
    assert ledger.step_complete("prepare") is True


def test_remote_states_batched_lookup_matches_per_unit_remote_state(ledger):
    ledger.ensure_artifact("prepare", "a")
    ledger.ensure_artifact("prepare", "b")
    ledger.set_remote_state("prepare", "a", RemoteState.VERIFIED)
    # "c" has no tracked row at all.

    states = ledger.remote_states("prepare", ["a", "b", "c"])
    assert states == {"a": RemoteState.VERIFIED, "b": RemoteState.MISSING}
    assert "c" not in states


def test_remote_states_empty_list_returns_empty_dict(ledger):
    assert ledger.remote_states("prepare", []) == {}


def test_mark_local_and_remote_batch_sets_both_states_for_every_unit(ledger):
    ledger.ensure_artifact("fetch", "a")
    ledger.ensure_artifact("fetch", "b")
    ledger.ensure_artifact("fetch", "c")

    ledger.mark_local_and_remote_batch("fetch", ["a", "b"], LocalState.COMPLETE, RemoteState.VERIFIED)

    assert ledger.local_state("fetch", "a") == LocalState.COMPLETE
    assert ledger.remote_state("fetch", "a") == RemoteState.VERIFIED
    assert ledger.local_state("fetch", "b") == LocalState.COMPLETE
    assert ledger.remote_state("fetch", "b") == RemoteState.VERIFIED
    # Untouched -- not passed in unit_ids.
    assert ledger.local_state("fetch", "c") == LocalState.MISSING


def test_mark_local_and_remote_batch_noop_on_empty_list(ledger):
    ledger.mark_local_and_remote_batch("fetch", [], LocalState.COMPLETE, RemoteState.VERIFIED)  # must not raise


def test_ensure_artifact_preserves_existing_paths_when_not_given(ledger):
    ledger.ensure_artifact("grid", "all", local_path="/data/grid.zarr", remote_path="grid/acag.zarr")
    ledger.ensure_artifact("grid", "all")  # re-touch with no path info
    row = ledger._con.execute(
        "SELECT local_path, remote_path FROM artifacts WHERE step='grid' AND unit_id='all'"
    ).fetchone()
    assert row == ("/data/grid.zarr", "grid/acag.zarr")


def test_step_complete_false_when_nothing_tracked(ledger):
    assert ledger.step_complete("grid") is False


def test_entrypoints_roundtrip(ledger):
    entrypoints = [{"year": 2020}, {"year": 2021, "day": 55}]
    added = ledger.upsert_entrypoints(entrypoints)
    assert added == 2
    assert ledger.upsert_entrypoints(entrypoints) == 0  # already known

    missing = ledger.missing_entrypoints()
    assert len(missing) == 2

    ledger.mark_entrypoint_crawled({"year": 2020})
    missing_after = ledger.missing_entrypoints()
    assert missing_after == [{"year": 2021, "day": 55}]


def test_stats_aggregates_local_and_remote(ledger):
    ledger.add_remote_files(
        [("a.nc", "https://x/a.nc"), ("b.nc", "https://x/b.nc")],
        get_file_hash=lambda url: url.split("/")[-1].replace(".nc", ""),
    )
    ledger.record_download_batch([DownloadResult(file_hash="a", ok=True, bytes=10)])
    ledger.record_push_batch("fetch", [
        PushResult(unit_id="a", ok=True, bytes=10),
        PushResult(unit_id="b", ok=False, error="boom"),
    ])
    stats = ledger.stats("fetch")
    assert stats["total"] == 2
    assert stats["local_complete"] == 1
    assert stats["remote_verified"] == 1
    assert stats["failed"] == 1


class _FakeHPCClient:
    """Minimal stand-in for HPCClient's push/pull surface -- exercises
    push_to_remote/merge_from_remote's call shape without a real SSH target."""

    def __init__(self):
        self.base_path = "/remote/base"
        self.ensured_dirs = []
        self.rsync_calls = []
        self.remote_exists = False

    def ensure_directory(self, remote_path):
        self.ensured_dirs.append(remote_path)
        return True

    def check_file_exists(self, remote_path):
        return self.remote_exists

    def rsync_transfer(self, source_path, target_path, source_is_local, options, show_progress):
        self.rsync_calls.append((source_path, target_path, source_is_local))
        return True, "ok"


def test_push_to_remote_calls_ensure_directory_and_rsync(ledger):
    client = _FakeHPCClient()
    assert ledger.push_to_remote(client) is True
    assert client.ensured_dirs == ["_ledger"]
    assert client.rsync_calls == [(ledger.local_path, "_ledger/acag_pm25.duckdb", True)]


def test_merge_from_remote_noop_when_no_remote_copy(ledger, tmp_path):
    client = _FakeHPCClient()
    client.remote_exists = False
    assert ledger.merge_from_remote(client, str(tmp_path / "tmp")) is True
    assert client.rsync_calls == []


# --- read-only open against a schema-less ledger file ----------------------


def _schemaless_duckdb_file(path: str) -> None:
    """A `.duckdb` file that exists on disk but was never opened read-write
    (so `_ensure_schema()` never ran) -- e.g. an interrupted first open, or
    a bare file created by something else. Confirmed real: this is exactly
    what crashed `plad`'s GRID step and `esacci`'s PREPARE planning inside
    `pipeline summary` with a raw `duckdb.CatalogException`."""
    import duckdb

    duckdb.connect(path).close()


def test_completed_fetch_files_degrades_gracefully_on_schemaless_ledger(tmp_path):
    path = str(tmp_path / "plad.duckdb")
    _schemaless_duckdb_file(path)
    with SourceLedger.open(path, data_path="plad", read_only=True) as ledger:
        assert ledger.completed_fetch_files() == []


def test_remote_state_degrades_gracefully_on_schemaless_ledger(tmp_path):
    path = str(tmp_path / "plad.duckdb")
    _schemaless_duckdb_file(path)
    with SourceLedger.open(path, data_path="plad", read_only=True) as ledger:
        assert ledger.remote_state("fetch", "some-unit") is None


def test_local_state_degrades_gracefully_on_schemaless_ledger(tmp_path):
    """local_state() must route through _execute_readonly_safe() exactly
    like remote_state() just above -- it didn't, until this was fixed, and
    raised a raw duckdb.CatalogException instead of returning None."""
    path = str(tmp_path / "plad.duckdb")
    _schemaless_duckdb_file(path)
    with SourceLedger.open(path, data_path="plad", read_only=True) as ledger:
        assert ledger.local_state("fetch", "some-unit") is None


def test_step_complete_degrades_gracefully_on_schemaless_ledger(tmp_path):
    path = str(tmp_path / "plad.duckdb")
    _schemaless_duckdb_file(path)
    with SourceLedger.open(path, data_path="plad", read_only=True) as ledger:
        assert ledger.step_complete("fetch") is False


def test_readonly_safe_reraises_non_catalog_errors(tmp_path):
    # Only a missing-schema CatalogException is swallowed -- a genuine query
    # bug (e.g. a typo'd column name) must still surface, not be silently
    # treated as "empty."
    path = str(tmp_path / "acag.duckdb")
    with SourceLedger.open(path, data_path="acag/pm25") as ledger:
        with pytest.raises(Exception):
            ledger._execute_readonly_safe("SELECT nonexistent_column FROM artifacts", [])

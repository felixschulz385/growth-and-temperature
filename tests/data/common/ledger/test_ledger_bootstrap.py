"""reconcile_fetch() against a fake RemoteFileCatalog + fake HPC client --
no live SSH target needed.
"""

import pytest

from src.data.common.ledger.bootstrap import reconcile_fetch
from src.data.common.ledger.store import SourceLedger


@pytest.fixture
def ledger(tmp_path):
    path = str(tmp_path / "acag_pm25.duckdb")
    with SourceLedger.open(path, data_path="acag/pm25") as led:
        yield led


class _FakeSource:
    has_entrypoints = False

    def __init__(self, files):
        self._files = files

    def list_remote_files(self, entrypoint=None):
        return self._files

    def get_file_hash(self, file_url):
        return file_url.split("/")[-1]


class _FakeHPCClient:
    def __init__(self, found_relative_paths, base_path="/remote/base"):
        self.base_path = base_path
        self._found = found_relative_paths
        self.commands = []

    def execute_command(self, command):
        self.commands.append(command)
        stdout = "\n".join(self._found)
        return True, stdout, ""


def test_reconcile_fetch_without_client_only_crawls(ledger):
    source = _FakeSource([("a.nc", "https://x/a.nc")])
    result = reconcile_fetch(ledger, source, raw_root="acag/pm25/raw")
    assert result == {"discovered": 1, "verified_present": 0}


def test_reconcile_fetch_marks_present_files_verified(ledger):
    source = _FakeSource([("a.nc", "https://x/a.nc"), ("b.nc", "https://x/b.nc")])
    client = _FakeHPCClient(found_relative_paths=["a.nc"])

    result = reconcile_fetch(ledger, source, raw_root="acag/pm25/raw", client=client)
    assert result == {"discovered": 2, "verified_present": 1}
    assert ledger.remote_state("fetch", "a.nc") == "verified"
    assert ledger.local_state("fetch", "a.nc") == "complete"
    assert ledger.remote_state("fetch", "b.nc") == "missing"

    # The find command targets the fully-resolved remote path.
    assert any("acag/pm25/raw" in cmd and "/remote/base" in cmd for cmd in client.commands)


def test_reconcile_fetch_handles_empty_remote_listing(ledger):
    source = _FakeSource([("a.nc", "https://x/a.nc")])
    client = _FakeHPCClient(found_relative_paths=[])

    result = reconcile_fetch(ledger, source, raw_root="acag/pm25/raw", client=client)
    assert result == {"discovered": 1, "verified_present": 0}
    assert ledger.remote_state("fetch", "a.nc") == "missing"


def test_reconcile_fetch_handles_find_command_failure(ledger, monkeypatch):
    source = _FakeSource([("a.nc", "https://x/a.nc")])

    class _FailingClient(_FakeHPCClient):
        def execute_command(self, command):
            return False, "", "no such file or directory"

    client = _FailingClient(found_relative_paths=[])
    result = reconcile_fetch(ledger, source, raw_root="acag/pm25/raw", client=client)
    assert result == {"discovered": 1, "verified_present": 0}

"""run_fetch_with_client() end-to-end against a fake source + fake HPC
client -- no live SSH target needed.

Two `ledger_mode`s, tested separately (see driver.py's own module
docstring):
- `"local"` (default): crawl -> download -> record local_state. Never
  pushes -- that's `data transfer`'s job now, for every source.
- `"remote"`: skip the crawl, read a pre-seeded remote ledger's own
  worklist, download -> push -> record both states, same fused loop the
  (now-removed) always-push default used to run.
"""

import os
import tarfile

import pytest

from src.data.common.fetch.driver import run_fetch_with_client
from src.data.common.ledger.store import SourceLedger
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext


class _FakeSource:
    ID = "fake"
    has_entrypoints = False

    def __init__(self, ctx, cfg, files, *, fail_urls=()):
        self.ctx = ctx
        self.cfg = cfg
        self._files = files
        self._fail_urls = set(fail_urls)

    @property
    def data_path(self):
        return self.cfg.data_path

    def list_remote_files(self, entrypoint=None):
        return self._files

    def get_file_hash(self, file_url):
        return file_url.split("/")[-1]

    async def download_async(self, source_url, output_path, session):
        if source_url in self._fail_urls:
            raise RuntimeError(f"simulated failure for {source_url}")
        with open(output_path, "wb") as f:
            f.write(f"content-of-{source_url}".encode())


class _FakeHPCClient:
    def __init__(self, base_path="/remote/base"):
        self.base_path = base_path
        self.remote_files: set[str] = set()
        self.rsync_calls = []
        self.ensured = []

    def _resolve(self, remote_path):
        if remote_path.startswith("/") or not self.base_path:
            return remote_path
        return f"{self.base_path}/{remote_path}"

    def ensure_directory(self, remote_path):
        self.ensured.append(remote_path)
        return True

    def rsync_transfer(self, source_path, target_path, source_is_local, options, show_progress):
        self.rsync_calls.append((source_path, target_path, source_is_local))
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
        if command.startswith("rm -f"):
            return True, "", ""
        return True, "", ""


@pytest.fixture
def ctx(tmp_path):
    return PipelineContext(
        data_root=str(tmp_path / "data_root"),
        local_index_dir=str(tmp_path / "index"),
        ssh_target="user@host:/remote/base",
    )


@pytest.fixture
def cfg():
    return SourceConfig.from_dict("fake", {"data_path": "fake"})


def tmp_path_ledger(ctx, data_path):
    from src.data.common.ledger.paths import ledger_path

    return ledger_path(ctx.local_index_dir, data_path)


# --- ledger_mode="local" (the default) --------------------------------------


def test_run_fetch_local_mode_downloads_without_pushing(ctx, cfg):
    files = [("a.nc", "https://x/a.nc"), ("b.nc", "https://x/b.nc")]
    source = _FakeSource(ctx, cfg, files)
    client = _FakeHPCClient()

    ok = run_fetch_with_client(source, client, batch_size=50, max_concurrent_downloads=5)
    assert ok is True
    # The whole point of the split: no push happens at all in local mode.
    assert client.remote_files == set()
    assert client.rsync_calls == []

    with SourceLedger.open(tmp_path_ledger(ctx, "fake"), data_path="fake", read_only=True) as ledger:
        assert ledger.local_state("fetch", "a.nc") == "complete"
        assert ledger.local_state("fetch", "b.nc") == "complete"
        # Not pushed -- completed_fetch_files() (remote_state='verified')
        # correctly still reports neither.
        assert ledger.completed_fetch_files() == []


def test_run_fetch_no_ssh_target_returns_false(tmp_path, cfg):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    source = _FakeSource(ctx, cfg, [("a.nc", "https://x/a.nc")])
    from src.data.common.fetch.driver import run_fetch

    assert run_fetch(source) is False


def test_run_fetch_no_local_index_dir_returns_false(cfg, tmp_path):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=None, ssh_target="user@host:/remote/base",
    )
    source = _FakeSource(ctx, cfg, [("a.nc", "https://x/a.nc")])
    client = _FakeHPCClient()
    assert run_fetch_with_client(source, client) is False


def test_run_fetch_invalid_ledger_mode_raises(ctx, cfg):
    source = _FakeSource(ctx, cfg, [("a.nc", "https://x/a.nc")])
    client = _FakeHPCClient()
    with pytest.raises(ValueError, match="ledger_mode"):
        run_fetch_with_client(source, client, ledger_mode="bogus")


def test_run_fetch_download_failure_marked_failed(ctx, cfg):
    files = [("a.nc", "https://x/a.nc"), ("b.nc", "https://x/b.nc")]
    source = _FakeSource(ctx, cfg, files, fail_urls={"https://x/b.nc"})
    client = _FakeHPCClient()

    ok = run_fetch_with_client(source, client)
    assert ok is False

    with SourceLedger.open(tmp_path_ledger(ctx, "fake"), data_path="fake", read_only=True) as ledger:
        assert ledger.local_state("fetch", "a.nc") == "complete"
        assert ledger.local_state("fetch", "b.nc") == "failed"


def test_run_fetch_second_run_is_a_noop_when_everything_downloaded(ctx, cfg):
    files = [("a.nc", "https://x/a.nc")]
    source = _FakeSource(ctx, cfg, files)
    client = _FakeHPCClient()

    assert run_fetch_with_client(source, client) is True
    assert run_fetch_with_client(source, client) is True
    # Local mode never touches the HPC client at all once nothing is pending.
    assert client.rsync_calls == []


def test_run_fetch_local_mode_releases_ledger_lock_during_download(ctx, cfg, monkeypatch):
    # The whole point of the per-batch (not per-run) connection scheme: no
    # ledger connection may be held while a download is in flight, or a
    # concurrent `data transfer --watch`/`data plan`/`data summary` process
    # would be locked out for the entire fetch run (confirmed empirically --
    # a held read-write DuckDB connection makes a second process's open,
    # read-write OR read-only, fail immediately, not block-and-wait).
    import duckdb

    from src.data.common.fetch import driver as driver_module

    files = [("a.nc", "https://x/a.nc")]
    source = _FakeSource(ctx, cfg, files)
    client = _FakeHPCClient()
    local_ledger_path = tmp_path_ledger(ctx, "fake")

    real_download_batch = driver_module._download_batch
    probed = {"ok": False}

    def _probing_download_batch(source_, pending, staging_dir, max_concurrent):
        # A fresh, independent connection -- simulating a wholly separate
        # process/CLI invocation -- must be able to open the SAME ledger
        # file read-write right now, mid-download.
        con = duckdb.connect(local_ledger_path, read_only=False)
        con.close()
        probed["ok"] = True
        return real_download_batch(source_, pending, staging_dir, max_concurrent)

    monkeypatch.setattr(driver_module, "_download_batch", _probing_download_batch)

    assert run_fetch_with_client(source, client) is True
    assert probed["ok"] is True


# --- ledger_mode="remote" -----------------------------------------------


def _seed_remote_ledger(path: str, data_path: str, files: list[tuple[str, str]]) -> None:
    """Build a real, standalone `.duckdb` ledger file representing "what the
    remote side already knows about" -- crawled and pushed from some other
    machine at some earlier point. `add_remote_files` alone (no download/push
    recorded) leaves every row `remote_state='missing'`, i.e. still
    outstanding -- exactly `--ledger remote` mode's worklist."""
    with SourceLedger.open(path, data_path=data_path) as seed:
        seed.add_remote_files(files, get_file_hash=lambda url: url.split("/")[-1].split(".")[0])


class _FakeHPCClientWithRemoteLedger(_FakeHPCClient):
    """`_FakeHPCClient`, but any `.duckdb` pull (`check_file_exists`/
    `rsync_transfer` against the ledger path) is served from a real,
    pre-built remote ledger file -- exercises `SourceLedger.
    pull_remote_readonly()`/`merge_from_remote()` against real bytes, the
    same way `_FakeHPCClientWithRealPull` does in
    tests/data/common/ledger/test_ledger_store.py."""

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


def test_run_fetch_remote_mode_clears_backlog_without_crawling(ctx, cfg, tmp_path):
    # The crawl source ("fake" `_FakeSource._files`) deliberately does NOT
    # include "a.nc" -- remote mode's worklist must come entirely from the
    # remote ledger, never from `source.list_remote_files()`.
    source = _FakeSource(ctx, cfg, [])
    remote_ledger_file = str(tmp_path / "remote.duckdb")
    _seed_remote_ledger(remote_ledger_file, "fake", [("a.nc", "https://x/a.nc")])
    client = _FakeHPCClientWithRemoteLedger(remote_ledger_file)

    ok = run_fetch_with_client(source, client, ledger_mode="remote")
    assert ok is True
    # Legacy layout (PipelineContext's default): <data_path>/raw/<file> --
    # downloaded from the origin (source.download_async), pushed immediately.
    assert "/remote/base/fake/raw/a.nc" in client.remote_files

    with SourceLedger.open(tmp_path_ledger(ctx, "fake"), data_path="fake", read_only=True) as ledger:
        assert ledger.completed_fetch_files() == ["a.nc"]
        # HPCPusher's cleanup_local=True deleted the staged download once
        # pushed -- local_state must reflect that, not still say 'complete'.
        assert ledger.local_state("fetch", "a") == "missing"
        assert ledger.remote_state("fetch", "a") == "verified"


def test_run_fetch_remote_mode_no_remote_ledger_returns_false(ctx, cfg):
    source = _FakeSource(ctx, cfg, [])
    client = _FakeHPCClient()  # no remote ledger seeded -- check_file_exists() is always False

    assert run_fetch_with_client(source, client, ledger_mode="remote") is False


def test_run_fetch_remote_mode_namespaced_source_raw_root(tmp_path):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"),
        ssh_target="user@host:/remote/base",
    )
    cfg = SourceConfig.from_dict("gadm", {"data_path": "misc", "namespace": "gadm"})
    source = _FakeSource(ctx, cfg, [])
    remote_ledger_file = str(tmp_path / "remote.duckdb")
    _seed_remote_ledger(remote_ledger_file, "misc/gadm", [("gadm.zip", "https://x/gadm.zip")])
    client = _FakeHPCClientWithRemoteLedger(remote_ledger_file)

    assert run_fetch_with_client(source, client, ledger_mode="remote") is True
    assert "/remote/base/misc/raw/gadm/gadm.zip" in client.remote_files


def test_run_fetch_remote_mode_honors_layout_v2(tmp_path):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"),
        ssh_target="user@host:/remote/base", layout="v2",
    )
    cfg = SourceConfig.from_dict("fake", {"data_path": "fake"})
    source = _FakeSource(ctx, cfg, [])
    remote_ledger_file = str(tmp_path / "remote.duckdb")
    _seed_remote_ledger(remote_ledger_file, "fake", [("a.nc", "https://x/a.nc")])
    client = _FakeHPCClientWithRemoteLedger(remote_ledger_file)

    assert run_fetch_with_client(source, client, ledger_mode="remote") is True
    assert "/remote/base/raw/fake/a.nc" in client.remote_files

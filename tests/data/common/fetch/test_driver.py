"""run_fetch_with_client() end-to-end against a fake source + fake HPC
client -- no live SSH target needed. Exercises crawl -> download -> push ->
ledger tracking as one flow.
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


def test_run_fetch_downloads_pushes_and_tracks(ctx, cfg):
    files = [("a.nc", "https://x/a.nc"), ("b.nc", "https://x/b.nc")]
    source = _FakeSource(ctx, cfg, files)
    client = _FakeHPCClient()

    ok = run_fetch_with_client(source, client, batch_size=50, max_concurrent_downloads=5)
    assert ok is True
    # Legacy layout (PipelineContext's default): <data_path>/raw/<file>.
    assert client.remote_files == {"/remote/base/fake/raw/a.nc", "/remote/base/fake/raw/b.nc"}

    with SourceLedger.open(str(tmp_path_ledger(ctx, "fake")), data_path="fake", read_only=True) as ledger:
        assert ledger.completed_fetch_files() == ["a.nc", "b.nc"]


def tmp_path_ledger(ctx, data_path):
    from src.data.common.ledger.paths import ledger_path

    return ledger_path(ctx.local_index_dir, data_path)


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


def test_run_fetch_download_failure_marked_failed_not_pushed(ctx, cfg):
    files = [("a.nc", "https://x/a.nc"), ("b.nc", "https://x/b.nc")]
    source = _FakeSource(ctx, cfg, files, fail_urls={"https://x/b.nc"})
    client = _FakeHPCClient()

    ok = run_fetch_with_client(source, client)
    assert ok is False
    assert "/remote/base/fake/raw/a.nc" in client.remote_files
    assert "/remote/base/fake/raw/b.nc" not in client.remote_files

    with SourceLedger.open(tmp_path_ledger(ctx, "fake"), data_path="fake", read_only=True) as ledger:
        assert ledger.completed_fetch_files() == ["a.nc"]
        assert ledger.local_state("fetch", "b.nc") == "failed"


def test_run_fetch_second_run_is_a_noop_when_everything_pushed(ctx, cfg):
    files = [("a.nc", "https://x/a.nc")]
    source = _FakeSource(ctx, cfg, files)
    client = _FakeHPCClient()

    assert run_fetch_with_client(source, client) is True
    rsync_calls_after_first = len(client.rsync_calls)

    assert run_fetch_with_client(source, client) is True
    # No new downloads/pushes -- the only extra rsync calls are the (up to
    # two) ledger push_to_remote() syncs at start/end of the second run.
    assert len(client.rsync_calls) <= rsync_calls_after_first + 2


def test_run_fetch_namespaced_source_raw_root(tmp_path):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"),
        ssh_target="user@host:/remote/base",
    )
    cfg = SourceConfig.from_dict("gadm", {"data_path": "misc", "namespace": "gadm"})
    source = _FakeSource(ctx, cfg, [("gadm.zip", "https://x/gadm.zip")])
    client = _FakeHPCClient()

    assert run_fetch_with_client(source, client) is True
    assert "/remote/base/misc/raw/gadm/gadm.zip" in client.remote_files


def test_run_fetch_honors_layout_v2(tmp_path):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"),
        ssh_target="user@host:/remote/base", layout="v2",
    )
    cfg = SourceConfig.from_dict("fake", {"data_path": "fake"})
    source = _FakeSource(ctx, cfg, [("a.nc", "https://x/a.nc")])
    client = _FakeHPCClient()

    assert run_fetch_with_client(source, client) is True
    assert "/remote/base/raw/fake/a.nc" in client.remote_files

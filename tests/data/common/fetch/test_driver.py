"""run_fetch() end-to-end against a fake source -- no ledger, no HPC client.
FETCH is purely local now: `catalog.required_files()` -> one local listing
snapshot -> download whatever's outstanding. `data transfer` (separately)
is the only thing that talks to HPC.
"""

import os

import pytest

from src.data.common import lockfile, statusfile
from src.data.common.fetch.driver import run_fetch
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
            f.write(b"x" * 2000)  # above manifest._MIN_PLAUSIBLE_BYTES


@pytest.fixture
def ctx(tmp_path):
    return PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))


@pytest.fixture
def cfg():
    return SourceConfig.from_dict("fake", {"data_path": "fake"})


def _raw_root(ctx, cfg):
    from src.data.sources import layout

    return layout.raw_root(ctx.data_root, cfg.data_path, namespace=cfg.namespace, layout=ctx.layout)


def test_run_fetch_downloads_required_files(ctx, cfg):
    files = [("a.nc", "https://x/a.nc"), ("b.nc", "https://x/b.nc")]
    source = _FakeSource(ctx, cfg, files)

    assert run_fetch(source) is True
    root = _raw_root(ctx, cfg)
    assert os.path.exists(os.path.join(root, "a.nc"))
    assert os.path.exists(os.path.join(root, "b.nc"))
    assert not os.path.exists(os.path.join(root, "a.nc.part"))


def test_run_fetch_second_run_is_a_noop_when_everything_downloaded(ctx, cfg):
    files = [("a.nc", "https://x/a.nc")]
    source = _FakeSource(ctx, cfg, files)

    assert run_fetch(source) is True
    assert run_fetch(source) is True  # nothing outstanding, no re-download attempted


def test_run_fetch_download_failure_returns_false_and_records_status(ctx, cfg):
    files = [("a.nc", "https://x/a.nc"), ("b.nc", "https://x/b.nc")]
    source = _FakeSource(ctx, cfg, files, fail_urls={"https://x/b.nc"})

    assert run_fetch(source) is False
    root = _raw_root(ctx, cfg)
    assert os.path.exists(os.path.join(root, "a.nc"))
    assert not os.path.exists(os.path.join(root, "b.nc"))

    status = statusfile.read(statusfile.status_path(root, "b.nc"))
    assert status["attempts"] == 1
    assert "simulated failure" in status["last_error"]


def test_run_fetch_unavailable_after_max_attempts_no_longer_retried(ctx, cfg):
    files = [("b.nc", "https://x/b.nc")]
    source = _FakeSource(ctx, cfg, files, fail_urls={"https://x/b.nc"})

    for _ in range(5):
        run_fetch(source, max_attempts=3)

    root = _raw_root(ctx, cfg)
    status = statusfile.read(statusfile.status_path(root, "b.nc"))
    assert status["status"] == "unavailable"
    assert status["attempts"] == 3  # stopped bumping once flipped unavailable-driven skip kicks in

    # A run with only an unavailable unit outstanding -- nothing left to
    # attempt, so it succeeds rather than failing forever.
    assert run_fetch(source, max_attempts=3) is True


def test_run_fetch_clears_failure_status_once_it_succeeds(ctx, cfg):
    files = [("a.nc", "https://x/a.nc")]
    source = _FakeSource(ctx, cfg, files, fail_urls={"https://x/a.nc"})
    run_fetch(source)
    root = _raw_root(ctx, cfg)
    assert statusfile.read(statusfile.status_path(root, "a.nc")) is not None

    source._fail_urls = set()  # origin recovers
    assert run_fetch(source) is True
    assert statusfile.read(statusfile.status_path(root, "a.nc")) is None


def test_run_fetch_refuses_concurrent_invocation(ctx, cfg):
    source = _FakeSource(ctx, cfg, [("a.nc", "https://x/a.nc")])
    root = _raw_root(ctx, cfg)
    lock_path = os.path.join(root, statusfile.STATUS_SUBDIR, "fetch.lock")
    lockfile.acquire(lock_path)
    try:
        assert run_fetch(source) is False
    finally:
        lockfile.release(lock_path)


def test_run_fetch_skips_a_file_already_on_hpc_when_transfer_mode_auto(tmp_path, monkeypatch):
    # transfer_mode=auto (src.data.common.fetch.transfer_mode): a file
    # already pushed to HPC counts as fetched even though nothing lands on
    # local disk for it -- only the genuinely-outstanding file gets
    # downloaded.
    class _FakeClient:
        def __init__(self, target, key_file=None):
            self.base_path = "base"

        def execute_command(self, command):
            return True, "2000 1700000000.0 a.nc\n", ""

    monkeypatch.setattr("src.data.common.hpc.client.HPCClient", _FakeClient)

    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), ssh_target="user@host:base"
    )
    cfg = SourceConfig.from_dict("fake", {"data_path": "fake", "transfer_mode": "auto"})
    files = [("a.nc", "https://x/a.nc"), ("b.nc", "https://x/b.nc")]
    source = _FakeSource(ctx, cfg, files)

    assert run_fetch(source) is True
    root = _raw_root(ctx, cfg)
    assert not os.path.exists(os.path.join(root, "a.nc"))  # already on HPC -- never downloaded
    assert os.path.exists(os.path.join(root, "b.nc"))  # genuinely outstanding -- downloaded


def test_run_fetch_stays_local_only_when_transfer_mode_manual_even_with_ssh_target(tmp_path, monkeypatch):
    class _FakeClient:
        def __init__(self, target, key_file=None):
            self.base_path = "base"

        def execute_command(self, command):
            raise AssertionError("transfer_mode=manual must never consult the remote target")

    monkeypatch.setattr("src.data.common.hpc.client.HPCClient", _FakeClient)

    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), ssh_target="user@host:base"
    )
    cfg = SourceConfig.from_dict("fake", {"data_path": "fake"})  # "fake" isn't an auto-transfer default
    source = _FakeSource(ctx, cfg, [("a.nc", "https://x/a.nc")])

    assert run_fetch(source) is True
    root = _raw_root(ctx, cfg)
    assert os.path.exists(os.path.join(root, "a.nc"))

"""``data run --ledger {local,remote}`` validation in `handle_run`
(src/cli/data/handlers.py) -- the `--ledger remote` guard rails: only valid
for `--step fetch`, and only for a `RemoteFileCatalog`-backed source (the
ones `--ledger local`'s crawl-catalog ledger actually applies to).
"""

import argparse

import pytest

from src.cli.data import handlers
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.base import DataSource, RemoteFileCatalog
from src.data.sources.steps import Completion, PipelineStep, StepTarget


class _FakeNonCatalogSource(DataSource):
    """A bare `DataSource` -- no `list_remote_files`/`get_file_hash`/etc, so
    `isinstance(self, RemoteFileCatalog)` is False, same as e.g. MODIS
    (which never implements this protocol at all)."""

    ID = "fake"
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE)

    def _plan(self, step, selection):
        return [StepTarget(source_id="fake", step=step, key="x", output_path="/tmp/x", completion=Completion.NEVER)]

    def _execute(self, target):
        return True


class _FakeCatalogSource(_FakeNonCatalogSource):
    """Adds the `RemoteFileCatalog` protocol methods -- confirms the
    isinstance check actually passes for a source that implements it."""

    DATA_SOURCE_NAME = "fake"
    has_entrypoints = False

    def list_remote_files(self, entrypoint=None):
        return []

    def get_file_hash(self, file_url):
        return file_url

    def get_all_entrypoints(self):
        return []

    def filename_to_entrypoint(self, relative_path):
        return None

    async def download_async(self, source_url, output_path, session=None):
        pass


def _args(*, step, ledger, source="fake"):
    return argparse.Namespace(
        source=source, config=None, step=step, log_level="WARNING", debug=False,
        years=None, year_range=None, keys=None, override=False, ledger=ledger,
    )


def _patch_build(monkeypatch, source):
    ctx, cfg = source.ctx, source.cfg

    def _fake_build(args_, step_):
        return source, ctx

    monkeypatch.setattr(handlers, "_build", _fake_build)
    return ctx, cfg


def test_ledger_remote_rejects_non_fetch_step(tmp_path, monkeypatch):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("fake", {"data_path": "fake"})
    source = _FakeNonCatalogSource(ctx, cfg)
    _patch_build(monkeypatch, source)

    args = _args(step="prepare", ledger="remote")
    with pytest.raises(ValueError, match="only applies to --step fetch"):
        handlers.handle_run(args)


def test_ledger_remote_rejects_non_catalog_source(tmp_path, monkeypatch):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("fake", {"data_path": "fake"})
    source = _FakeNonCatalogSource(ctx, cfg)
    assert not isinstance(source, RemoteFileCatalog)
    _patch_build(monkeypatch, source)

    args = _args(step="fetch", ledger="remote")
    with pytest.raises(ValueError, match="does not use the crawl-catalog ledger"):
        handlers.handle_run(args)


def test_ledger_remote_injects_ledger_mode_into_download_config_for_catalog_source(tmp_path, monkeypatch):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("fake", {"data_path": "fake", "download": {"batch_size": 5}})
    source = _FakeCatalogSource(ctx, cfg)
    assert isinstance(source, RemoteFileCatalog)
    _patch_build(monkeypatch, source)

    args = _args(step="fetch", ledger="remote")
    handlers.handle_run(args)  # must not raise -- plan()/execute() run against _FakeCatalogSource's stub target

    assert source.cfg.raw["download"]["ledger_mode"] == "remote"
    # Pre-existing download config keys survive the injection.
    assert source.cfg.raw["download"]["batch_size"] == 5


def test_ledger_local_default_does_not_touch_download_config(tmp_path, monkeypatch):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("fake", {"data_path": "fake"})
    source = _FakeCatalogSource(ctx, cfg)
    _patch_build(monkeypatch, source)

    args = _args(step="fetch", ledger="local")
    handlers.handle_run(args)

    assert "ledger_mode" not in source.cfg.raw.get("download", {})

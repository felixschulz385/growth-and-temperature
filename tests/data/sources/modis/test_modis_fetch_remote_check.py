"""ModisSource._plan_fetch()'s remote-aware completeness check
(src.data.common.fetch.manifest.resolve_fetch_listing): a
transfer_mode=auto MODIS config (the default) with an HPC target configured
should judge a tile-year "already fetched" against the HPC listing, not
local disk -- since its local .tif gets pushed to HPC right after FETCH and
isn't kept around indefinitely. `data summary`'s local_only selection must
still see the old, purely-local behavior.
"""

import os

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.modis.source import ModisSource
from src.data.sources.steps import Completion, PipelineStep, TargetSelection, is_complete


def _make_source(tmp_path, *, ssh_target=None, tiles=("h18v04", "h20v08"), year_range=(2019, 2019), **extra_raw):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), ssh_target=ssh_target
    )
    cfg = SourceConfig.from_dict("modis", {"year_range": list(year_range), "tiles": list(tiles), **extra_raw})
    return ModisSource(ctx, cfg), ctx


def test_stays_path_exists_when_transfer_mode_manual(tmp_path):
    source, _ = _make_source(tmp_path, ssh_target="user@host:base", transfer_mode="manual")
    targets = source.plan(PipelineStep.FETCH, TargetSelection(local_only=False))
    assert all(t.completion is Completion.PATH_EXISTS for t in targets)


def test_stays_path_exists_without_ssh_target_even_though_auto_by_default(tmp_path):
    source, _ = _make_source(tmp_path, ssh_target=None)
    targets = source.plan(PipelineStep.FETCH, TargetSelection(local_only=False))
    assert all(t.completion is Completion.PATH_EXISTS for t in targets)


def test_uses_precomputed_completion_against_remote_listing_when_auto(tmp_path, monkeypatch):
    class _FakeClient:
        def __init__(self, target, key_file=None):
            self.base_path = "base"

        def execute_command(self, command):
            # Only the h18v04/2019 tile-year is present on the HPC target.
            return True, "2000 1700000000.0 2019/h18v04.tif\n", ""

    monkeypatch.setattr("src.data.common.hpc.client.HPCClient", _FakeClient)

    source, _ = _make_source(tmp_path, ssh_target="user@host:base")  # transfer_mode defaults to "auto" for modis
    targets = {t.key: t for t in source.plan(PipelineStep.FETCH, TargetSelection(local_only=False))}

    assert targets["2019/h18v04"].completion is Completion.PRECOMPUTED
    assert is_complete(targets["2019/h18v04"]) is True
    assert is_complete(targets["2019/h20v08"]) is False
    # Local disk genuinely has nothing -- confirms the remote listing, not
    # local presence, drove the completeness decision.
    assert not os.path.exists(targets["2019/h18v04"].output_path)


def test_local_only_selection_never_touches_the_remote_target(tmp_path, monkeypatch):
    class _FakeClient:
        def __init__(self, target, key_file=None):
            self.base_path = "base"

        def execute_command(self, command):
            raise AssertionError("a local_only selection (e.g. data summary) must never make a live remote call")

    monkeypatch.setattr("src.data.common.hpc.client.HPCClient", _FakeClient)

    source, _ = _make_source(tmp_path, ssh_target="user@host:base")
    # TargetSelection() defaults to local_only=True -- matches `data summary`'s call.
    targets = source.plan(PipelineStep.FETCH, TargetSelection())
    assert all(t.completion is Completion.PATH_EXISTS for t in targets)

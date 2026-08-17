"""GlassAvhrrSource._plan_fetch()'s remote-aware completeness check
(src.data.common.fetch.manifest.resolve_fetch_listing) -- same reasoning as
tests/data/sources/modis/test_modis_fetch_remote_check.py: a
transfer_mode=auto GLASS config (the default for glass_avhrr/glass_modis/
glass_ta_modis) with an HPC target configured should judge a (year, day)
"already fetched" against the HPC listing instead of local disk.

docs/design/12-glass-modis-rebuild.md §6: split off of the former
test_glass_fetch_remote_check.py -- unchanged in behavior, just renamed/
re-imported alongside the AVHRR/MODIS module split.
"""

import os

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.glass.avhrr import GlassAvhrrSource
from src.data.sources.steps import Completion, PipelineStep, TargetSelection, is_complete

_BASE_URL = "https://glass.hku.hk/archive/LST/AVHRR/0.05D/"
_DAY_RANGE = {"start": [2019, 363], "end": [2019, 365]}


def _make_source(tmp_path, *, ssh_target=None, **extra_raw):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), ssh_target=ssh_target
    )
    cfg = SourceConfig.from_dict("glass_avhrr", {"base_url": _BASE_URL, "day_range": _DAY_RANGE, **extra_raw})
    return GlassAvhrrSource(ctx, cfg), ctx


def test_stays_path_exists_when_transfer_mode_manual(tmp_path):
    source, _ = _make_source(tmp_path, ssh_target="user@host:base", transfer_mode="manual")
    targets = source.plan(PipelineStep.FETCH, TargetSelection(local_only=False))
    assert all(t.completion is Completion.PATH_EXISTS for t in targets)


def test_uses_precomputed_completion_against_remote_listing_when_auto(tmp_path, monkeypatch):
    class _FakeClient:
        def __init__(self, target, key_file=None):
            self.base_path = "base"

        def execute_command(self, command):
            # Only day 364 is present on the HPC target -- nothing locally.
            return True, "2000 1700000000.0 2019/GLASS08B31.V40.A2019364.2021259.hdf\n", ""

    monkeypatch.setattr("src.data.common.hpc.client.HPCClient", _FakeClient)

    source, _ = _make_source(tmp_path, ssh_target="user@host:base")  # transfer_mode defaults to "auto" for glass_avhrr
    targets = {t.key: t for t in source.plan(PipelineStep.FETCH, TargetSelection(local_only=False))}

    assert targets["2019/364"].completion is Completion.PRECOMPUTED
    assert is_complete(targets["2019/364"]) is True
    assert is_complete(targets["2019/363"]) is False
    # Local disk genuinely has nothing -- confirms the remote listing, not
    # local presence, drove the completeness decision.
    assert not os.path.exists(targets["2019/364"].output_path)


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

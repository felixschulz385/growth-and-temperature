"""CommodityPricesSource.plan() -- mirrors
tests/data/sources/snl_mining/test_snl_mining_plan.py's `_make_source` /
plan()-target-assertion style."""

import os

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.commodity_prices.source import CommodityPricesSource
from src.data.sources.steps import Completion, PipelineStep, TargetSelection


def _make_source(tmp_path, **raw):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"),
        local_index_dir=str(tmp_path / "index"),
        grid_id="legacy_4326",
    )
    cfg = SourceConfig.from_dict("commodity_prices", dict(raw))
    return CommodityPricesSource(ctx, cfg), ctx


def test_steps_is_fetch_and_prepare_only():
    assert CommodityPricesSource.STEPS == (PipelineStep.FETCH, PipelineStep.PREPARE)


def test_requires_empty():
    from src.data.sources import registry

    assert registry.resolve("commodity_prices").requires == ()


def test_prepare_plan_empty_when_raw_file_missing(tmp_path):
    source, _ = _make_source(tmp_path)
    assert source.plan(PipelineStep.PREPARE, TargetSelection()) == []


def test_prepare_plan_target_when_fetch_output_present(tmp_path):
    source, _ = _make_source(tmp_path)
    fetch_file = source._raw_prices_file()
    os.makedirs(os.path.dirname(fetch_file), exist_ok=True)
    open(fetch_file, "w").close()

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    target = targets[0]
    assert target.output_path == os.path.join(source.output_root(PipelineStep.PREPARE), "commodity_prices.parquet")
    assert target.inputs == (fetch_file,)
    assert target.completion == Completion.PATH_EXISTS


def test_prices_path_override_bypasses_fetch_output(tmp_path):
    source, ctx = _make_source(tmp_path, prices_path="manual/prices.xlsx")
    assert source._raw_prices_file() == os.path.join(ctx.data_root, "manual", "prices.xlsx")

    override_file = os.path.join(ctx.data_root, "manual", "prices.xlsx")
    os.makedirs(os.path.dirname(override_file), exist_ok=True)
    open(override_file, "w").close()

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    assert targets[0].inputs == (override_file,)


def test_default_prices_url_and_name(tmp_path):
    source, _ = _make_source(tmp_path)
    assert source.CONFIGURED_FILES[0].name == "CMO-Historical-Data-Annual.xlsx"
    assert source.CONFIGURED_FILES[0].url.startswith("https://thedocs.worldbank.org/")


def test_prices_url_config_override(tmp_path):
    source, _ = _make_source(tmp_path, prices_url="https://example.org/prices.xlsx", prices_name="prices.xlsx")
    assert source.CONFIGURED_FILES[0].url == "https://example.org/prices.xlsx"
    assert source.CONFIGURED_FILES[0].name == "prices.xlsx"



def test_prepare_plan_always_reflects_this_hosts_own_raw_file(tmp_path):
    # The cross-host ledger merge bug this test used to reproduce
    # (`PermissionError: '/Users'` on scicore from a Mac-written ledger row)
    # cannot recur now that PREPARE has no ledger fast path at all -- plan()
    # is a bare live `os.path.exists()` check against this host's own raw
    # file every call (see module docstring), so there is no foreign-host
    # row to trust. This test is the direct replacement: with no ledger
    # involved, plan() reflects only what's actually on this host's disk.
    source, _ = _make_source(tmp_path)
    fetch_file = source._raw_prices_file()
    os.makedirs(os.path.dirname(fetch_file), exist_ok=True)
    open(fetch_file, "w").close()

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    target = targets[0]
    assert target.output_path == os.path.join(source.output_root(PipelineStep.PREPARE), "commodity_prices.parquet")
    assert target.inputs == (fetch_file,)

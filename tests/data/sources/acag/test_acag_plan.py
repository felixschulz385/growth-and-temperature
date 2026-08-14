"""AcagSource: ledger-free FETCH/PREPARE (docs/design successor to the
ledger, Plan 2 PREPARE+GRID merge). PREPARE is planned by a live crawl of
FETCH's raw output directory; there is no separate GRID step. See
tests/data/sources/ntl_harm/test_ntl_harm_plan.py for the pilot this mirrors.
"""

import os

import pytest

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.acag import AcagSource
from src.data.sources.steps import Completion, PipelineStep, TargetSelection


def _make_source(tmp_path, layout="legacy", **raw):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), layout=layout
    )
    cfg = SourceConfig.from_dict("acag", {"data_path": "acag/pm25", **raw})
    return AcagSource(ctx, cfg), ctx


def _write_raw_file(source, relative_path):
    raw_root = source.output_root(PipelineStep.FETCH)
    full = os.path.join(raw_root, relative_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    open(full, "w").close()


def test_steps_is_fetch_and_prepare_only():
    assert AcagSource.STEPS == (PipelineStep.FETCH, PipelineStep.PREPARE)


def test_output_root_fetch_and_prepare_use_top_level_trees_under_layout_v2(tmp_path):
    source, ctx = _make_source(tmp_path, layout="v2")
    assert source.output_root(PipelineStep.FETCH) == os.path.join(ctx.data_root, "raw", "acag/pm25")


def test_prepare_plan_empty_when_no_raw_files(tmp_path):
    source, _ = _make_source(tmp_path)
    assert source.plan(PipelineStep.PREPARE, TargetSelection()) == []


def test_prepare_plan_one_target_covering_every_available_year_prefers_nc4(tmp_path):
    source, _ = _make_source(tmp_path)
    for rel in (
        "GL/Annual/V6GL02.04.CNNPM25.GL.201901-201912.nc",
        "GL/Annual/V6GL02.04.CNNPM25.GL.202001-202012.nc4",
        "GL/Annual/V6GL02.04.CNNPM25.GL.202101-202112.nc",
    ):
        _write_raw_file(source, rel)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    target = targets[0]
    assert target.key == "all"
    assert target.completion == Completion.MARKER
    assert target.meta["years"] == [2019, 2020, 2021]
    assert target.meta["raw_files"][2020] == "GL/Annual/V6GL02.04.CNNPM25.GL.202001-202012.nc4"
    assert target.output_path.endswith("acag_pm25_timeseries_reprojected.zarr")


def test_prepare_plan_respects_year_selection(tmp_path):
    source, _ = _make_source(tmp_path)
    for rel in (
        "GL/Annual/V6GL02.04.CNNPM25.GL.201901-201912.nc",
        "GL/Annual/V6GL02.04.CNNPM25.GL.202001-202012.nc",
        "GL/Annual/V6GL02.04.CNNPM25.GL.202101-202112.nc",
    ):
        _write_raw_file(source, rel)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection(year_range=(2020, 2021)))
    assert targets[0].meta["years"] == [2020, 2021]


def test_output_path_uses_v2_family_under_layout_v2(tmp_path):
    source, ctx = _make_source(tmp_path, layout="v2")
    assert source._output_path() == os.path.join(ctx.data_root, "grid", "legacy_4326", "pm25.zarr")


def test_unsupported_step_raises_for_a_source_with_a_narrower_contract(tmp_path):
    from src.data.sources.steps import UnsupportedStepError

    class FetchOnlySource(AcagSource):
        STEPS = (PipelineStep.FETCH,)

    source, _ = _make_source(tmp_path)
    fetch_only = FetchOnlySource(source.ctx, source.cfg)
    with pytest.raises(UnsupportedStepError):
        fetch_only.plan(PipelineStep.PREPARE, TargetSelection())

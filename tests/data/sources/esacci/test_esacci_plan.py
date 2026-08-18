"""EsacciSource: FETCH/PREPARE. See
tests/data/sources/acag/test_acag_plan.py for the mirrored shape.
"""

import os

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.esacci import EsacciSource
from src.data.sources.steps import Completion, PipelineStep, TargetSelection


def _make_source(tmp_path, **raw):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index")
    )
    cfg = SourceConfig.from_dict("esacci", {"data_path": "esacci/landcover", **raw})
    return EsacciSource(ctx, cfg), ctx


def _write_raw_file(source, relative_path):
    raw_root = source.output_root(PipelineStep.FETCH)
    full = os.path.join(raw_root, relative_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    open(full, "w").close()


def test_steps_is_fetch_and_prepare_only():
    assert EsacciSource.STEPS == (PipelineStep.FETCH, PipelineStep.PREPARE)


def test_default_variables_to_keep(tmp_path):
    source, _ = _make_source(tmp_path)
    assert source.variables_to_keep == ["lccs_class"]


def test_output_root_fetch_and_prepare_use_top_level_trees(tmp_path):
    source, ctx = _make_source(tmp_path)
    assert source.output_root(PipelineStep.FETCH) == os.path.join(ctx.data_root, "raw", "esacci/landcover")


def test_prepare_plan_empty_when_no_raw_files(tmp_path):
    source, _ = _make_source(tmp_path)
    assert source.plan(PipelineStep.PREPARE, TargetSelection()) == []


def test_prepare_plan_one_target_covering_every_available_year_prefers_nc4(tmp_path):
    source, _ = _make_source(tmp_path)
    for rel in (
        "2019/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2019-v2.0.7.nc",
        "2020/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2020-v2.0.7.nc4",
        "2021/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2021-v2.0.7.nc",
    ):
        _write_raw_file(source, rel)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    target = targets[0]
    assert target.key == "all"
    assert target.completion == Completion.MARKER
    assert target.meta["years"] == [2019, 2020, 2021]
    assert target.meta["raw_files"][2020] == "2020/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2020-v2.0.7.nc4"
    assert target.output_path.endswith("land_cover")


def test_prepare_plan_respects_year_selection(tmp_path):
    source, _ = _make_source(tmp_path)
    for rel in (
        "2019/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2019-v2.0.7.nc",
        "2020/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2020-v2.0.7.nc",
        "2021/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2021-v2.0.7.nc",
    ):
        _write_raw_file(source, rel)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection(year_range=(2020, 2021)))
    assert targets[0].meta["years"] == [2020, 2021]


def test_output_path_uses_family(tmp_path):
    source, ctx = _make_source(tmp_path)
    assert source._output_path() == os.path.join(source.output_root(PipelineStep.GRID), "land_cover")

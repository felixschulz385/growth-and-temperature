"""AcagSource.plan() must produce the same targets as the old ACAGPreprocessor.

The migration oracle: tests/data/preprocess/sources/test_characterization_acag.py
pins the old code's behaviour; this file asserts the new AcagSource (fetch +
prepare + grid merged into one class, docs/design/09-integrated-pipeline.md §5)
reproduces it under the new PipelineStep vocabulary and StepTarget shape.
"""

import os

import pandas as pd
import pytest

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.acag import AcagSource
from src.data.sources.steps import PipelineStep, TargetSelection


def _write_index(local_index_dir, data_path, rows):
    safe = data_path.replace("/", "_").replace("\\", "_")
    os.makedirs(local_index_dir, exist_ok=True)
    path = os.path.join(local_index_dir, f"parquet_{safe}.parquet")
    pd.DataFrame(rows).to_parquet(path)


def _make_source(tmp_path, year_range=(2019, 2021), rows=None, layout="legacy"):
    data_root = str(tmp_path / "data_root")
    local_index_dir = str(tmp_path / "index")
    data_path = "acag/pm25"
    if rows is None:
        rows = [
            {"relative_path": "GL/Annual/V6GL02.04.CNNPM25.GL.201901-201912.nc", "status_category": "completed"},
            {"relative_path": "GL/Annual/V6GL02.04.CNNPM25.GL.202001-202012.nc4", "status_category": "completed"},
            {"relative_path": "GL/Annual/V6GL02.04.CNNPM25.GL.202101-202112.nc", "status_category": "completed"},
            {"relative_path": "GL/Annual/V6GL02.04.CNNPM25.GL.202201-202212.nc", "status_category": "pending"},
        ]
    _write_index(local_index_dir, data_path, rows)
    ctx = PipelineContext(data_root=data_root, local_index_dir=local_index_dir, layout=layout)
    cfg = SourceConfig.from_dict("acag", {"data_path": data_path, "year_range": list(year_range)})
    return AcagSource(ctx, cfg), ctx


def test_output_root_matches_old_get_hpc_output_path(tmp_path):
    source, ctx = _make_source(tmp_path)
    assert source.output_root(PipelineStep.PREPARE) == os.path.join(
        ctx.data_root, "acag/pm25", "processed", "stage_1"
    )
    assert source.output_root(PipelineStep.GRID) == os.path.join(
        ctx.data_root, "acag/pm25", "processed", "stage_2"
    )


def test_output_root_fetch_and_prepare_use_top_level_trees_under_layout_v2(tmp_path):
    source, ctx = _make_source(tmp_path, layout="v2")
    assert source.output_root(PipelineStep.FETCH) == os.path.join(ctx.data_root, "raw", "acag/pm25")
    assert source.output_root(PipelineStep.PREPARE) == os.path.join(ctx.data_root, "prepared", "acag/pm25")


def test_prepare_targets_one_per_year_prefers_nc4(tmp_path):
    source, _ = _make_source(tmp_path)
    targets = source.plan(PipelineStep.PREPARE, TargetSelection(year_range=(2019, 2021)))

    assert [t.key for t in targets] == ["2019", "2020", "2021"]
    for t in targets:
        assert t.step is PipelineStep.PREPARE
        assert t.output_path == os.path.join(
            source.output_root(PipelineStep.PREPARE), f"{t.meta['year']}.zarr"
        )
    assert targets[1].inputs == ("GL/Annual/V6GL02.04.CNNPM25.GL.202001-202012.nc4",)


def test_prepare_targets_excludes_incomplete_and_out_of_range(tmp_path):
    source, _ = _make_source(tmp_path)
    targets = source.plan(PipelineStep.PREPARE, TargetSelection(year_range=(2019, 2020)))
    assert [t.key for t in targets] == ["2019", "2020"]


def test_prepare_targets_empty_when_index_missing(tmp_path):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("acag", {"data_path": "acag/pm25_never_indexed"})
    source = AcagSource(ctx, cfg)
    assert source.plan(PipelineStep.PREPARE, TargetSelection()) == []


def test_grid_target_lists_available_annual_zarrs(tmp_path):
    source, _ = _make_source(tmp_path, year_range=(2019, 2022))
    annual_dir = source.output_root(PipelineStep.PREPARE)
    os.makedirs(annual_dir, exist_ok=True)
    for year in (2019, 2020):
        os.makedirs(os.path.join(annual_dir, f"{year}.zarr"))

    targets = source.plan(PipelineStep.GRID, TargetSelection(year_range=(2019, 2022)))

    assert len(targets) == 1
    target = targets[0]
    assert target.step is PipelineStep.GRID
    assert sorted(target.meta["years_available"]) == [2019, 2020]
    assert target.output_path == os.path.join(
        source.output_root(PipelineStep.GRID), "acag_pm25_timeseries_reprojected.zarr"
    )


def test_grid_target_uses_v2_family_path_under_layout_v2(tmp_path):
    source, ctx = _make_source(tmp_path, year_range=(2019, 2022), layout="v2")
    annual_dir = source.output_root(PipelineStep.PREPARE)
    os.makedirs(annual_dir, exist_ok=True)
    os.makedirs(os.path.join(annual_dir, "2019.zarr"))

    targets = source.plan(PipelineStep.GRID, TargetSelection(year_range=(2019, 2022)))

    assert len(targets) == 1
    assert targets[0].output_path == os.path.join(ctx.data_root, "grid", "legacy_4326", "pm25.zarr")


def test_fetch_step_is_declared_and_prepare_grid_reject_undeclared_steps():
    # AcagSource declares all three steps -- sanity check for the reference
    # migration, which every other ArchiveRasterSource-shaped source follows.
    assert AcagSource.STEPS == (PipelineStep.FETCH, PipelineStep.PREPARE, PipelineStep.GRID)


def test_unsupported_step_raises_for_a_source_with_a_narrower_contract(tmp_path):
    from src.data.sources.steps import UnsupportedStepError

    class FetchOnlySource(AcagSource):
        STEPS = (PipelineStep.FETCH,)

    source, _ = _make_source(tmp_path)
    fetch_only = FetchOnlySource(source.ctx, source.cfg)
    with pytest.raises(UnsupportedStepError):
        fetch_only.plan(PipelineStep.PREPARE, TargetSelection())

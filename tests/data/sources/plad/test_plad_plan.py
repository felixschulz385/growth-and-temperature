"""PlaDSource.plan() must reproduce the old PLADPreprocessor's targets.
Oracle: tests/data/preprocess/sources/test_characterization_plad.py.
"""

import os

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.plad import PlaDSource
from src.data.sources.steps import PipelineStep, TargetSelection


def _make_source(tmp_path, admin_level=1, year_range=(1980, 2022)):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("plad", {"admin_level": admin_level, "year_range": list(year_range)})
    return PlaDSource(ctx, cfg), ctx


def test_no_prepare_step():
    assert PipelineStep.PREPARE not in PlaDSource.STEPS
    assert PlaDSource.STEPS == (PipelineStep.FETCH, PipelineStep.GRID)


def test_requires_gadm_prepare_not_grid():
    assert PlaDSource.REQUIRES == (("gadm", PipelineStep.PREPARE),)


def test_output_root_hardcodes_plad_prefix_ignoring_data_path(tmp_path):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("plad", {"data_path": "something/else"})
    source = PlaDSource(ctx, cfg)
    assert source.output_root(PipelineStep.GRID) == os.path.join(ctx.data_root, "plad", "processed", "stage_2")


def test_admin_level_must_be_1_or_2(tmp_path):
    import pytest

    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("plad", {"admin_level": 3})
    with pytest.raises(ValueError):
        PlaDSource(ctx, cfg)


def test_grid_target_output_path_includes_admin_level(tmp_path):
    s1, _ = _make_source(tmp_path, admin_level=1)
    s2, _ = _make_source(tmp_path, admin_level=2)
    t1 = s1.plan(PipelineStep.GRID, TargetSelection())[0]
    t2 = s2.plan(PipelineStep.GRID, TargetSelection())[0]
    assert t1.output_path == os.path.join(s1.output_root(PipelineStep.GRID), "plad_adm1_timeseries_reprojected.zarr")
    assert t2.output_path == os.path.join(s2.output_root(PipelineStep.GRID), "plad_adm2_timeseries_reprojected.zarr")
    assert t1.meta["admin_level"] == 1


def test_resolve_gadm_files_reads_from_gadm_prepare_output(tmp_path):
    source, ctx = _make_source(tmp_path)
    gadm_dir = os.path.join(ctx.data_root, "misc", "processed", "stage_1", "gadm")
    os.makedirs(gadm_dir, exist_ok=True)
    open(os.path.join(gadm_dir, "gadm_levelADM_1_simplified.gpkg"), "w").close()

    files = source._resolve_gadm_files_from_preprocessed()
    assert files == {"gadm_adm1": os.path.join(gadm_dir, "gadm_levelADM_1_simplified.gpkg")}

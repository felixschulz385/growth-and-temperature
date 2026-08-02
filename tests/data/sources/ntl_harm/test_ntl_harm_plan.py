"""NtlHarmSource.plan() must reproduce the old NTLHarmPreprocessor's targets,
including its two deliberately-preserved quirks (see src/data/sources/ntl_harm.py
module docstring). Oracle: tests/data/preprocess/sources/test_characterization_ntl_harm.py.
"""

import os

import pandas as pd

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.ntl_harm import NtlHarmSource
from src.data.sources.steps import PipelineStep, TargetSelection


def _write_index(local_index_dir, data_path, rows):
    safe = data_path.replace("/", "_").replace("\\", "_")
    os.makedirs(local_index_dir, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(local_index_dir, f"parquet_{safe}.parquet"))


def _make_source(tmp_path, year_range=(2019, 2021), rows=None, layout="legacy"):
    data_root = str(tmp_path / "data_root")
    local_index_dir = str(tmp_path / "index")
    data_path = "ntl_harm/harmonized"
    if rows is None:
        rows = [
            {"relative_path": "harmonized_2021.tif", "status_category": "completed"},
            {"relative_path": "harmonized_2019.tif", "status_category": "completed"},
            {"relative_path": "harmonized_2020.zip", "status_category": "completed"},
            {"relative_path": "harmonized_2020.tif", "status_category": "completed"},
            {"relative_path": "harmonized_2022.tif", "status_category": "pending"},
        ]
    _write_index(local_index_dir, data_path, rows)
    ctx = PipelineContext(data_root=data_root, local_index_dir=local_index_dir, layout=layout)
    cfg = SourceConfig.from_dict("ntl_harm", {"data_path": data_path, "year_range": list(year_range)})
    return NtlHarmSource(ctx, cfg), ctx


def test_output_root_fetch_and_prepare_use_top_level_trees_under_layout_v2(tmp_path):
    source, ctx = _make_source(tmp_path, layout="v2")
    assert source.output_root(PipelineStep.FETCH) == os.path.join(ctx.data_root, "raw", "ntl_harm/harmonized")
    assert source.output_root(PipelineStep.PREPARE) == os.path.join(ctx.data_root, "prepared", "ntl_harm/harmonized")


def test_default_resampling_is_sum(tmp_path):
    source, _ = _make_source(tmp_path)
    assert source.resampling == "sum"


def test_resampling_overridable(tmp_path):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("ntl_harm", {"data_path": "ntl_harm/harmonized", "resampling": "nearest"})
    source = NtlHarmSource(ctx, cfg)
    assert source.resampling == "nearest"


def test_prepare_targets_preserve_file_insertion_order_not_sorted(tmp_path):
    source, _ = _make_source(tmp_path)
    targets = source.plan(PipelineStep.PREPARE, TargetSelection(year_range=(2019, 2021)))
    assert [t.key for t in targets] == ["2021", "2019", "2020"]


def test_prepare_target_prefers_tif_over_zip(tmp_path):
    source, _ = _make_source(tmp_path)
    targets = source.plan(PipelineStep.PREPARE, TargetSelection(year_range=(2019, 2021)))
    by_key = {t.key: t for t in targets}
    assert by_key["2020"].inputs == ("harmonized_2020.tif",)
    assert by_key["2020"].meta["total_candidates"] == 2


def test_grid_target_lists_available_annual_zarrs(tmp_path):
    source, _ = _make_source(tmp_path, year_range=(2019, 2022))
    annual_dir = source.output_root(PipelineStep.PREPARE)
    os.makedirs(annual_dir, exist_ok=True)
    for year in (2019, 2020):
        os.makedirs(os.path.join(annual_dir, f"{year}.zarr"))

    targets = source.plan(PipelineStep.GRID, TargetSelection(year_range=(2019, 2022)))
    assert len(targets) == 1
    assert sorted(targets[0].meta["years_available"]) == [2019, 2020]
    assert targets[0].output_path == os.path.join(
        source.output_root(PipelineStep.GRID), "ntl_harm_timeseries_reprojected.zarr"
    )


def test_grid_target_uses_v2_family_path_under_layout_v2(tmp_path):
    source, ctx = _make_source(tmp_path, year_range=(2019, 2022), layout="v2")
    annual_dir = source.output_root(PipelineStep.PREPARE)
    os.makedirs(annual_dir, exist_ok=True)
    os.makedirs(os.path.join(annual_dir, "2019.zarr"))

    targets = source.plan(PipelineStep.GRID, TargetSelection(year_range=(2019, 2022)))
    assert len(targets) == 1
    assert targets[0].output_path == os.path.join(ctx.data_root, "grid", "legacy_4326", "ntl_harm.zarr")

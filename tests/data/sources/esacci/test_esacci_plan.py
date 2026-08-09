"""EsacciSource.plan() must reproduce the old ESACCIPreprocessor's targets.

Oracle: tests/data/preprocess/sources/test_characterization_esacci.py.
"""

import os

from src.data.common.ledger.store import PushResult, SourceLedger
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.esacci import EsacciSource
from src.data.sources.steps import PipelineStep, TargetSelection


def _write_index(local_index_dir, data_path, rows):
    """Build a ledger with the given (relative_path, status_category) rows --
    "completed" means HPC-verified (docs/design/10-fetch-ledger.md), matching
    what `_plan_prepare`'s `completed_fetch_files()` actually reads."""
    safe = data_path.replace("/", "_").replace("\\", "_")
    os.makedirs(local_index_dir, exist_ok=True)
    path = os.path.join(local_index_dir, f"{safe}.duckdb")
    with SourceLedger.open(path, data_path=data_path) as ledger:
        files = [(row["relative_path"], row["relative_path"]) for row in rows]
        ledger.add_remote_files(files, get_file_hash=lambda url: url)
        completed = [row["relative_path"] for row in rows if row["status_category"] == "completed"]
        if completed:
            ledger.record_push_batch("fetch", (PushResult(unit_id=p, ok=True) for p in completed))


def _make_source(tmp_path, year_range=(2019, 2021), rows=None, layout="legacy"):
    data_root = str(tmp_path / "data_root")
    local_index_dir = str(tmp_path / "index")
    data_path = "esacci/landcover"
    if rows is None:
        rows = [
            {"relative_path": "2019/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2019-v2.0.7.nc", "status_category": "completed"},
            {"relative_path": "2020/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2020-v2.0.7.nc4", "status_category": "completed"},
            {"relative_path": "2021/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2021-v2.0.7.nc", "status_category": "completed"},
            {"relative_path": "2022/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2022-v2.0.7.nc", "status_category": "pending"},
        ]
    _write_index(local_index_dir, data_path, rows)
    ctx = PipelineContext(data_root=data_root, local_index_dir=local_index_dir, layout=layout)
    cfg = SourceConfig.from_dict("esacci", {"data_path": data_path, "year_range": list(year_range)})
    return EsacciSource(ctx, cfg), ctx


def test_default_variables_to_keep(tmp_path):
    source, _ = _make_source(tmp_path)
    assert source.variables_to_keep == ["lccs_class"]


def test_output_root_matches_old_get_hpc_output_path(tmp_path):
    source, ctx = _make_source(tmp_path)
    assert source.output_root(PipelineStep.PREPARE) == os.path.join(ctx.data_root, "esacci/landcover", "processed", "stage_1")
    assert source.output_root(PipelineStep.GRID) == os.path.join(ctx.data_root, "esacci/landcover", "processed", "stage_2")


def test_output_root_fetch_and_prepare_use_top_level_trees_under_layout_v2(tmp_path):
    source, ctx = _make_source(tmp_path, layout="v2")
    assert source.output_root(PipelineStep.FETCH) == os.path.join(ctx.data_root, "raw", "esacci/landcover")
    assert source.output_root(PipelineStep.PREPARE) == os.path.join(ctx.data_root, "prepared", "esacci/landcover")


def test_prepare_targets_one_per_year_prefers_nc4(tmp_path):
    source, _ = _make_source(tmp_path)
    targets = source.plan(PipelineStep.PREPARE, TargetSelection(year_range=(2019, 2021)))

    assert [t.key for t in targets] == ["2019", "2020", "2021"]
    assert targets[1].inputs == ("2020/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2020-v2.0.7.nc4",)


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
        source.output_root(PipelineStep.GRID), "esacci_lc_timeseries_reprojected.zarr"
    )


def test_grid_target_uses_v2_family_path_under_layout_v2(tmp_path):
    source, ctx = _make_source(tmp_path, year_range=(2019, 2022), layout="v2")
    annual_dir = source.output_root(PipelineStep.PREPARE)
    os.makedirs(annual_dir, exist_ok=True)
    os.makedirs(os.path.join(annual_dir, "2019.zarr"))

    targets = source.plan(PipelineStep.GRID, TargetSelection(year_range=(2019, 2022)))

    assert len(targets) == 1
    assert targets[0].output_path == os.path.join(ctx.data_root, "grid", "legacy_4326", "land_cover.zarr")

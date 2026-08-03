"""GlassSource.plan() must reproduce the old GlassPreprocessor's targets for
both the "glass_modis" and "glass_avhrr" registered ids.

Oracle: tests/data/preprocess/sources/test_characterization_glass.py.
"""

import os

from src.data.common.ledger.store import PushResult, SourceLedger
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.glass.source import GlassSource
from src.data.sources.steps import PipelineStep, TargetSelection

_BASE_URLS = {
    "glass_modis": "https://glass.hku.hk/archive/LST/MODIS/Daily/1KM/",
    "glass_avhrr": "https://glass.hku.hk/archive/LST/AVHRR/0.05D/",
}


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
            ledger.record_push_batch(PushResult(step="fetch", unit_id=p, ok=True) for p in completed)


def _make_source(tmp_path, source_id, year_range=(2019, 2021), rows=None, layout="legacy", **extra_raw):
    data_root = str(tmp_path / "data_root")
    local_index_dir = str(tmp_path / "index")
    if rows is None:
        if source_id == "glass_modis":
            rows = [
                {"relative_path": "GLASS06A01.V01.A2019055.h25v06.2022021.hdf", "status_category": "completed"},
                {"relative_path": "GLASS06A01.V01.A2019056.h25v06.2022021.hdf", "status_category": "completed"},
                {"relative_path": "GLASS06A01.V01.A2020001.h26v06.2022021.hdf", "status_category": "completed"},
            ]
        else:
            rows = [
                {"relative_path": "GLASS08B31.V40.A2019001.2021259.hdf", "status_category": "completed"},
                {"relative_path": "GLASS08B31.V40.A2020001.2021259.hdf", "status_category": "completed"},
            ]
    path_prefix = GlassSource.MODIS_PATH_PREFIX if source_id == "glass_modis" else GlassSource.AVHRR_PATH_PREFIX
    _write_index(local_index_dir, path_prefix.rstrip("/"), rows)
    ctx = PipelineContext(data_root=data_root, local_index_dir=local_index_dir, layout=layout)
    cfg = SourceConfig.from_dict(
        source_id, {"year_range": list(year_range), "base_url": _BASE_URLS[source_id], **extra_raw}
    )
    return GlassSource(ctx, cfg), ctx


def test_data_source_kind_derived_from_registered_id(tmp_path):
    assert _make_source(tmp_path, "glass_modis")[0].data_source_kind == "MODIS"
    assert _make_source(tmp_path, "glass_avhrr")[0].data_source_kind == "AVHRR"


def test_output_root_uses_path_prefix(tmp_path):
    source, ctx = _make_source(tmp_path, "glass_modis")
    assert source.output_root(PipelineStep.PREPARE) == os.path.join(
        ctx.data_root, "glass/LST/MODIS/Daily/1KM/", "processed", "stage_1"
    )


def test_output_root_fetch_and_prepare_use_top_level_trees_under_layout_v2(tmp_path):
    source, ctx = _make_source(tmp_path, "glass_modis", layout="v2")
    assert source.output_root(PipelineStep.FETCH) == os.path.join(ctx.data_root, "raw", "glass/LST/MODIS/Daily/1KM/")
    assert source.output_root(PipelineStep.PREPARE) == os.path.join(
        ctx.data_root, "prepared", "glass/LST/MODIS/Daily/1KM/"
    )


def test_resolve_source_file_path_matches_fetch_output_root(tmp_path):
    # PREPARE reads raw files back in by relative_path -- must resolve them
    # under wherever FETCH actually wrote them, not a hardcoded
    # "<path_prefix>/raw/" shape, so this stays correct under either layout.
    legacy_source, legacy_ctx = _make_source(tmp_path, "glass_modis", layout="legacy")
    assert legacy_source._resolve_source_file_path("foo/bar.hdf") == os.path.join(
        legacy_ctx.data_root, "glass/LST/MODIS/Daily/1KM/", "raw", "foo/bar.hdf"
    )

    v2_source, v2_ctx = _make_source(tmp_path, "glass_modis", layout="v2")
    assert v2_source._resolve_source_file_path("foo/bar.hdf") == os.path.join(
        v2_ctx.data_root, "raw", "glass/LST/MODIS/Daily/1KM/", "foo/bar.hdf"
    )


def test_modis_prepare_targets_grouped_by_year_and_grid_cell(tmp_path):
    source, _ = _make_source(tmp_path, "glass_modis")
    targets = source.plan(PipelineStep.PREPARE, TargetSelection(year_range=(2019, 2021)))
    assert {t.key for t in targets} == {"2019/h25v06", "2020/h26v06"}
    by_key = {t.key: t for t in targets}
    assert sorted(by_key["2019/h25v06"].inputs) == [
        "GLASS06A01.V01.A2019055.h25v06.2022021.hdf",
        "GLASS06A01.V01.A2019056.h25v06.2022021.hdf",
    ]
    assert by_key["2019/h25v06"].output_path == os.path.join(
        source.output_root(PipelineStep.PREPARE), "2019", "h25v06.zarr"
    )


def test_avhrr_prepare_targets_grouped_by_year_only(tmp_path):
    source, _ = _make_source(tmp_path, "glass_avhrr")
    targets = source.plan(PipelineStep.PREPARE, TargetSelection(year_range=(2019, 2021)))
    assert {t.key for t in targets} == {"2019", "2020"}


def test_modis_grid_cells_filter(tmp_path):
    source, _ = _make_source(tmp_path, "glass_modis", grid_cells=["h25v06"])
    targets = source.plan(PipelineStep.PREPARE, TargetSelection(year_range=(2019, 2021)))
    assert {t.key for t in targets} == {"2019/h25v06"}


def test_modis_grid_target_is_single_combined_target(tmp_path):
    source, _ = _make_source(tmp_path, "glass_modis", year_range=(2019, 2020))
    annual_dir = source.output_root(PipelineStep.PREPARE)
    for year, cell in [(2019, "h25v06"), (2020, "h26v06")]:
        d = os.path.join(annual_dir, str(year))
        os.makedirs(d, exist_ok=True)
        os.makedirs(os.path.join(d, f"{cell}.zarr"))

    targets = source.plan(PipelineStep.GRID, TargetSelection(year_range=(2019, 2020)))
    assert len(targets) == 1
    assert targets[0].key == "all_cells"
    assert targets[0].output_path == os.path.join(source.output_root(PipelineStep.GRID), "modis_timeseries_reprojected.zarr")
    assert sorted(targets[0].meta["grid_cells"]) == ["h25v06", "h26v06"]


def test_avhrr_grid_target_is_single_global_target(tmp_path):
    source, _ = _make_source(tmp_path, "glass_avhrr", year_range=(2019, 2020))
    annual_dir = source.output_root(PipelineStep.PREPARE)
    os.makedirs(annual_dir, exist_ok=True)
    os.makedirs(os.path.join(annual_dir, "2019.zarr"))

    targets = source.plan(PipelineStep.GRID, TargetSelection(year_range=(2019, 2020)))
    assert len(targets) == 1
    assert targets[0].key == "global"
    assert targets[0].meta["missing_years"] == [2020]
    assert targets[0].output_path == os.path.join(source.output_root(PipelineStep.GRID), "avhrr_timeseries_reprojected.zarr")


def test_grid_target_uses_v2_family_path_under_layout_v2(tmp_path):
    modis_source, modis_ctx = _make_source(tmp_path, "glass_modis", year_range=(2019, 2020), layout="v2")
    annual_dir = modis_source.output_root(PipelineStep.PREPARE)
    for year, cell in [(2019, "h25v06"), (2020, "h26v06")]:
        d = os.path.join(annual_dir, str(year))
        os.makedirs(d, exist_ok=True)
        os.makedirs(os.path.join(d, f"{cell}.zarr"))
    targets = modis_source.plan(PipelineStep.GRID, TargetSelection(year_range=(2019, 2020)))
    assert targets[0].output_path == os.path.join(modis_ctx.data_root, "grid", "legacy_4326", "glass_modis_lst.zarr")

    avhrr_source, avhrr_ctx = _make_source(tmp_path, "glass_avhrr", year_range=(2019, 2020), layout="v2")
    annual_dir = avhrr_source.output_root(PipelineStep.PREPARE)
    os.makedirs(annual_dir, exist_ok=True)
    os.makedirs(os.path.join(annual_dir, "2019.zarr"))
    targets = avhrr_source.plan(PipelineStep.GRID, TargetSelection(year_range=(2019, 2020)))
    assert targets[0].output_path == os.path.join(avhrr_ctx.data_root, "grid", "legacy_4326", "glass_avhrr_lst.zarr")

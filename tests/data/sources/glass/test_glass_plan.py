"""GlassSource: FETCH/PREPARE for both the "glass_modis" and "glass_avhrr"
registered ids. PREPARE is planned by a live crawl of FETCH's raw output
directory; there is no separate GRID step. See
tests/data/sources/acag/test_acag_plan.py for the mirrored shape.
"""

import os

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.glass.source import GlassSource
from src.data.sources.steps import Completion, PipelineStep, TargetSelection

_BASE_URLS = {
    "glass_modis": "https://glass.hku.hk/archive/LST/MODIS/Daily/1KM/",
    "glass_avhrr": "https://glass.hku.hk/archive/LST/AVHRR/0.05D/",
}
_DAY_RANGES = {
    "glass_modis": {"start": [2000, 55], "end": [2020, 365]},
    "glass_avhrr": {"start": [1992, 1], "end": [2020, 365]},
}


def _make_source(tmp_path, source_id, layout="legacy", **extra_raw):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), layout=layout
    )
    raw = {"base_url": _BASE_URLS[source_id], "day_range": _DAY_RANGES[source_id], **extra_raw}
    cfg = SourceConfig.from_dict(source_id, raw)
    return GlassSource(ctx, cfg), ctx


def _write_raw_files(source, filenames):
    raw_root = source.output_root(PipelineStep.FETCH)
    os.makedirs(raw_root, exist_ok=True)
    for fname in filenames:
        open(os.path.join(raw_root, fname), "w").close()


_MODIS_FILES = [
    "GLASS06A01.V01.A2019055.h25v06.2022021.hdf",
    "GLASS06A01.V01.A2019056.h25v06.2022021.hdf",
    "GLASS06A01.V01.A2020001.h26v06.2022021.hdf",
]
_AVHRR_FILES = [
    "GLASS08B31.V40.A2019001.2021259.hdf",
    "GLASS08B31.V40.A2020001.2021259.hdf",
]


def test_steps_is_fetch_and_prepare_only():
    assert GlassSource.STEPS == (PipelineStep.FETCH, PipelineStep.PREPARE)


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


def test_prepare_plan_empty_when_no_raw_files(tmp_path):
    source, _ = _make_source(tmp_path, "glass_modis")
    assert source.plan(PipelineStep.PREPARE, TargetSelection()) == []


def test_modis_daily_files_grouped_by_year_and_grid_cell(tmp_path):
    source, _ = _make_source(tmp_path, "glass_modis")
    _write_raw_files(source, _MODIS_FILES)

    groups = source._group_daily_files(TargetSelection(year_range=(2019, 2021)))
    assert {g["key"] for g in groups} == {"2019/h25v06", "2020/h26v06"}
    by_key = {g["key"]: g for g in groups}
    assert sorted(by_key["2019/h25v06"]["files"]) == [
        "GLASS06A01.V01.A2019055.h25v06.2022021.hdf",
        "GLASS06A01.V01.A2019056.h25v06.2022021.hdf",
    ]
    assert source._annual_zarr_path(by_key["2019/h25v06"]) == os.path.join(
        source.output_root(PipelineStep.PREPARE), "2019", "h25v06.zarr"
    )


def test_avhrr_daily_files_grouped_by_year_only(tmp_path):
    source, _ = _make_source(tmp_path, "glass_avhrr")
    _write_raw_files(source, _AVHRR_FILES)

    groups = source._group_daily_files(TargetSelection(year_range=(2019, 2021)))
    assert {g["key"] for g in groups} == {"2019", "2020"}


def test_modis_grid_cells_filter(tmp_path):
    source, _ = _make_source(tmp_path, "glass_modis", grid_cells=["h25v06"])
    _write_raw_files(source, _MODIS_FILES)

    groups = source._group_daily_files(TargetSelection(year_range=(2019, 2021)))
    assert {g["key"] for g in groups} == {"2019/h25v06"}


def test_modis_prepare_plan_is_a_single_merged_target(tmp_path):
    source, _ = _make_source(tmp_path, "glass_modis")
    _write_raw_files(source, _MODIS_FILES)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection(year_range=(2019, 2020)))
    assert len(targets) == 1
    target = targets[0]
    assert target.key == "all"
    assert target.completion == Completion.MARKER
    assert sorted(target.meta["years_available"]) == [2019, 2020]
    assert target.output_path == os.path.join(source.output_root(PipelineStep.GRID), "modis_timeseries_reprojected.zarr")


def test_avhrr_prepare_plan_is_a_single_merged_target(tmp_path):
    source, _ = _make_source(tmp_path, "glass_avhrr")
    _write_raw_files(source, _AVHRR_FILES)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection(year_range=(2019, 2020)))
    assert len(targets) == 1
    assert targets[0].output_path == os.path.join(
        source.output_root(PipelineStep.GRID), "avhrr_timeseries_reprojected.zarr"
    )


def test_output_path_uses_v2_family_under_layout_v2(tmp_path):
    modis_source, modis_ctx = _make_source(tmp_path, "glass_modis", layout="v2")
    assert modis_source._grid_output_path() == os.path.join(modis_ctx.data_root, "grid", "legacy_4326", "glass_modis_lst.zarr")

    avhrr_source, avhrr_ctx = _make_source(tmp_path, "glass_avhrr", layout="v2")
    assert avhrr_source._grid_output_path() == os.path.join(
        avhrr_ctx.data_root, "grid", "legacy_4326", "glass_avhrr_lst.zarr"
    )

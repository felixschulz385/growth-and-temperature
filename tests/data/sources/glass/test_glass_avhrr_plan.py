"""GlassAvhrrSource: FETCH/PREPARE for the "glass_avhrr" registered id.
PREPARE is planned by a live crawl of FETCH's raw output directory; there is
no separate GRID step. See tests/data/sources/acag/test_acag_plan.py for the
mirrored shape.

docs/design/12-glass-modis-rebuild.md §6: split off of the former
test_glass_plan.py along the AVHRR/MODIS line -- these cases are AVHRR-
specific and unchanged in behavior; MODIS's own PREPARE plan shape is
covered by test_glass_modis_plan.py.
"""

import os

from src.data.sources import layout
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.glass.avhrr import GlassAvhrrSource
from src.data.sources.steps import Completion, PipelineStep, TargetSelection

_BASE_URL = "https://glass.hku.hk/archive/LST/AVHRR/0.05D/"
_DAY_RANGE = {"start": [1992, 1], "end": [2020, 365]}


def _make_source(tmp_path, **extra_raw):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index")
    )
    raw = {"base_url": _BASE_URL, "day_range": _DAY_RANGE, **extra_raw}
    cfg = SourceConfig.from_dict("glass_avhrr", raw)
    return GlassAvhrrSource(ctx, cfg), ctx


def _write_raw_files(source, filenames):
    raw_root = source.output_root(PipelineStep.FETCH)
    os.makedirs(raw_root, exist_ok=True)
    for fname in filenames:
        open(os.path.join(raw_root, fname), "w").close()


_AVHRR_FILES = [
    "GLASS08B31.V40.A2019001.2021259.hdf",
    "GLASS08B31.V40.A2020001.2021259.hdf",
]


def test_steps_is_fetch_and_prepare_only():
    assert GlassAvhrrSource.STEPS == (PipelineStep.FETCH, PipelineStep.PREPARE)


def test_output_root_fetch_and_prepare_use_top_level_trees(tmp_path):
    source, ctx = _make_source(tmp_path)
    assert source.output_root(PipelineStep.FETCH) == os.path.join(ctx.data_root, "raw", "glass/LST/AVHRR/0.05D/")
    # PREPARE requires an `agg` bucket (src/data/sources/layout.py);
    # GLASS-AVHRR's per-year annual-stats zarr is a pixel-grid store, so it
    # goes under the "crs" bucket -- see GlassAvhrrSource.output_root's
    # docstring.
    assert source.output_root(PipelineStep.PREPARE) == os.path.join(
        ctx.data_root, "prepared", "glass/LST/AVHRR/0.05D/", layout.CRS_AGG
    )


def test_resolve_source_file_path_matches_fetch_output_root(tmp_path):
    # PREPARE reads raw files back in by relative_path -- must resolve them
    # under wherever FETCH actually wrote them, not a hardcoded
    # "<path_prefix>/raw/" shape.
    source, ctx = _make_source(tmp_path)
    assert source._resolve_source_file_path("foo/bar.hdf") == os.path.join(
        ctx.data_root, "raw", "glass/LST/AVHRR/0.05D/", "foo/bar.hdf"
    )


def test_prepare_plan_empty_when_no_raw_files(tmp_path):
    source, _ = _make_source(tmp_path)
    assert source.plan(PipelineStep.PREPARE, TargetSelection()) == []


def test_avhrr_daily_files_grouped_by_year_only(tmp_path):
    source, _ = _make_source(tmp_path)
    _write_raw_files(source, _AVHRR_FILES)

    groups = source._group_daily_files(TargetSelection(year_range=(2019, 2021)))
    assert {g["key"] for g in groups} == {"2019", "2020"}


def test_avhrr_prepare_plan_is_a_single_merged_target(tmp_path):
    source, _ = _make_source(tmp_path)
    _write_raw_files(source, _AVHRR_FILES)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection(year_range=(2019, 2020)))
    assert len(targets) == 1
    target = targets[0]
    assert target.key == "all"
    assert target.completion == Completion.MARKER
    assert sorted(target.meta["years_available"]) == [2019, 2020]
    assert target.output_path == source._grid_output_path()


def test_output_path_uses_family(tmp_path):
    source, ctx = _make_source(tmp_path)
    _write_raw_files(source, _AVHRR_FILES)
    assert source._grid_output_path() == os.path.join(
        ctx.data_root, "prepared", "glass/LST/AVHRR/0.05D/", "crs", "legacy_4326", "glass_avhrr_lst.zarr"
    )

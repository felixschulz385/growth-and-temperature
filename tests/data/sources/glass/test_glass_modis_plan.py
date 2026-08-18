"""GlassModisSource: PREPARE plan shape -- a single `key="all"` target
covering every available year, mirroring `ModisSource._discover_prepare`
(`modis/source.py`), with `_execute_prepare` looping over years internally
via the shared `run_tiled_prepare` driver."""

import os

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.glass.modis import GlassModisSource
from src.data.sources.steps import Completion, PipelineStep, TargetSelection

_DAY_RANGE = {"start": [2019, 1], "end": [2020, 3]}


def _make_source(tmp_path, source_id="glass_modis", **extra_raw):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index")
    )
    base_url = {
        "glass_modis": "https://glass.hku.hk/archive/LST/MODIS/Daily/1KM/",
        "glass_ta_modis": "https://glass.hku.hk/archive/Ta/MODIS/",
    }[source_id]
    raw = {"base_url": base_url, "day_range": _DAY_RANGE, "land_tiles": ["h08v05", "h01v07"], **extra_raw}
    cfg = SourceConfig.from_dict(source_id, raw)
    return GlassModisSource(ctx, cfg), ctx


def _write_tile_tifs(source, year, tiles):
    year_dir = os.path.join(source.output_root(PipelineStep.FETCH), str(year))
    os.makedirs(year_dir, exist_ok=True)
    for tile in tiles:
        open(os.path.join(year_dir, f"{tile}.tif"), "wb").close()


def test_steps_is_fetch_and_prepare_only():
    assert GlassModisSource.STEPS == (PipelineStep.FETCH, PipelineStep.PREPARE)


def test_discover_prepare_one_target_covering_available_years(tmp_path):
    source, _ = _make_source(tmp_path)
    _write_tile_tifs(source, 2019, ["h08v05", "h01v07"])
    _write_tile_tifs(source, 2020, ["h08v05"])

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    assert targets[0].key == "all"
    assert targets[0].meta["years"] == [2019, 2020]
    assert targets[0].completion == Completion.MARKER


def test_discover_prepare_skips_years_with_no_tile_output(tmp_path):
    source, _ = _make_source(tmp_path)
    _write_tile_tifs(source, 2019, ["h08v05"])
    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    assert targets[0].meta["years"] == [2019]


def test_prepare_output_path_uses_variant_specific_family(tmp_path):
    lst_source, lst_ctx = _make_source(tmp_path, "glass_modis")
    ta_source, ta_ctx = _make_source(tmp_path, "glass_ta_modis")

    assert lst_source._prepare_output_path() == os.path.join(
        lst_ctx.data_root, "prepared", "glass/LST/MODIS/Daily/1KM/", "crs", "legacy_4326", "glass_modis_lst"
    )
    assert ta_source._prepare_output_path() == os.path.join(
        ta_ctx.data_root, "prepared", "glass/Ta/MODIS/", "crs", "legacy_4326", "glass_modis_ta"
    )
    # Different variants never collide on the same store.
    assert lst_source._prepare_output_path() != ta_source._prepare_output_path()


def test_output_root_fetch_uses_base_default_prepare_uses_grid_tier(tmp_path):
    source, ctx = _make_source(tmp_path)
    assert source.output_root(PipelineStep.FETCH) == os.path.join(ctx.data_root, "raw", "glass/LST/MODIS/Daily/1KM/")
    assert source.output_root(PipelineStep.PREPARE) == os.path.join(
        ctx.data_root, "prepared", "glass/LST/MODIS/Daily/1KM/", "crs", "legacy_4326"
    )
    assert source.output_root(PipelineStep.PREPARE) == source.output_root(PipelineStep.GRID)


def test_transfer_units_fetch_lists_per_tile_year_tifs(tmp_path):
    source, _ = _make_source(tmp_path)
    _write_tile_tifs(source, 2019, ["h08v05", "h01v07"])

    units = source.transfer_units(PipelineStep.FETCH)
    assert {u.unit_id for u in units} == {"2019/h08v05.tif", "2019/h01v07.tif"}

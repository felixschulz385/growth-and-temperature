"""GlassModisSource: PREPARE plan shape (docs/design/12-glass-modis-rebuild.md
§4) -- one target per year, `inputs` populated straight from FETCH's raw
output directory's per-tile GeoTIFFs, mirroring `ModisSource._discover_
prepare` (`modis/source.py:556-607`)."""

import os

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.glass.modis import GlassModisSource
from src.data.sources.steps import Completion, PipelineStep, TargetSelection

_DAY_RANGE = {"start": [2019, 1], "end": [2020, 3]}


def _make_source(tmp_path, source_id="glass_modis", layout="legacy", **extra_raw):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), layout=layout
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


def test_discover_prepare_one_target_per_year_with_tile_inputs(tmp_path):
    source, _ = _make_source(tmp_path)
    _write_tile_tifs(source, 2019, ["h08v05", "h01v07"])
    _write_tile_tifs(source, 2020, ["h08v05"])

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert {t.key for t in targets} == {"2019", "2020"}
    by_key = {t.key: t for t in targets}
    assert set(os.path.basename(f) for f in by_key["2019"].inputs) == {"h08v05.tif", "h01v07.tif"}
    assert by_key["2019"].completion == Completion.NEVER
    assert by_key["2019"].output_path == by_key["2020"].output_path  # one shared multi-year store


def test_discover_prepare_skips_years_with_no_tile_output(tmp_path):
    source, _ = _make_source(tmp_path)
    _write_tile_tifs(source, 2019, ["h08v05"])
    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert {t.key for t in targets} == {"2019"}


def test_prepare_output_path_uses_variant_specific_v2_family(tmp_path):
    lst_source, lst_ctx = _make_source(tmp_path, "glass_modis", layout="v2")
    ta_source, ta_ctx = _make_source(tmp_path, "glass_ta_modis", layout="v2")

    assert lst_source._prepare_output_path() == os.path.join(
        lst_ctx.data_root, "prepared", "glass/LST/MODIS/Daily/1KM/", "crs", "legacy_4326", "glass_modis_lst.zarr"
    )
    assert ta_source._prepare_output_path() == os.path.join(
        ta_ctx.data_root, "prepared", "glass/Ta/MODIS/", "crs", "legacy_4326", "glass_modis_ta.zarr"
    )
    # Different variants never collide on the same store.
    assert lst_source._prepare_output_path() != ta_source._prepare_output_path()


def test_output_root_fetch_uses_base_default_prepare_uses_grid_tier(tmp_path):
    source, ctx = _make_source(tmp_path, layout="v2")
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

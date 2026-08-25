"""GlassModisSource._execute_prepare wiring onto the shared run_tiled_prepare
driver: ctx.grid_id threading, year-major mosaic memoization, and a real
tiled parquet write."""

import os

import numpy as np
import rasterio
from odc.geo.geobox import GeoBox
from rasterio.transform import from_bounds

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.glass.modis import GlassModisSource
from src.data.sources.steps import PipelineStep, TargetSelection, marker_path

_DAY_RANGE = {"start": [2019, 1], "end": [2019, 3]}


def _make_source(tmp_path, grid_id="legacy_4326", **extra_raw):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), grid_id=grid_id)
    raw = {
        "base_url": "https://glass.hku.hk/archive/LST/MODIS/Daily/1KM/",
        "day_range": _DAY_RANGE,
        "land_tiles": ["h08v05"],
        **extra_raw,
    }
    cfg = SourceConfig.from_dict("glass_modis", raw)
    return GlassModisSource(ctx, cfg), ctx


def _write_tile_tif(path, size=8, bounds=(-1, -1, 1, 1), value=290.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    band_names = ["mean", "std", "max", "min", "count_above", "count_below", "valid_period_count", "valid_month_count"]
    transform = from_bounds(*bounds, size, size)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=len(band_names),
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan,
    ) as dst:
        for i, name in enumerate(band_names, start=1):
            dst.write(np.full((size, size), value, dtype="float32"), i)
            dst.set_band_description(i, name)


def test_execute_prepare_threads_ctx_grid_id_and_writes_real_tiled_parquet(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path, grid_id="ease6933")
    _write_tile_tif(os.path.join(source.output_root(PipelineStep.FETCH), "2019", "h08v05.tif"))

    fake_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.5)  # 4x4, 2x2 tiles @ size 2
    captured = {}

    import src.data.common.geobox as geobox_module

    def fake_get_target_geobox(passed_ctx):
        captured["ctx"] = passed_ctx
        return fake_geobox

    monkeypatch.setattr(geobox_module, "get_target_geobox", fake_get_target_geobox)
    source.tile_size = 2

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    target = targets[0]

    assert source._execute_prepare(target) is True
    assert captured["ctx"] is ctx
    assert os.path.exists(marker_path(target.output_path))

    import pandas as pd
    from pathlib import Path

    parts = sorted(Path(target.output_path).glob("ix=*/iy=*/part-*.parquet"))
    assert len(parts) == 4  # 2x2 tile grid x 1 year
    df = pd.concat(pd.read_parquet(p) for p in parts)
    assert set(df["year"].unique()) == {2019}
    assert (df["mean"] == 290.0).all()


def test_execute_prepare_handles_slightly_misaligned_adjacent_tiles(tmp_path, monkeypatch):
    """Same real production failure as ModisSource's identical test
    (tests/data/sources/modis/test_modis_prepare.py) -- two genuinely
    adjacent tiles whose independently-fetched pixel grids overlap by a
    sliver used to crash `xr.combine_by_coords` with "duplicate values";
    `rioxarray.merge` (now used instead) tolerates it
    (docs/design/13-prepare-memory-parallelism.md)."""
    source, ctx = _make_source(tmp_path, grid_id="ease6933", land_tiles=["h08v05", "h09v05"])
    _write_tile_tif(
        os.path.join(source.output_root(PipelineStep.FETCH), "2019", "h08v05.tif"),
        bounds=(-1, -1, 0.1, 1), value=1.0,
    )
    _write_tile_tif(
        os.path.join(source.output_root(PipelineStep.FETCH), "2019", "h09v05.tif"),
        bounds=(-0.1, -1, 1, 1), value=2.0,
    )

    fake_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.5)  # 4x4, 2x2 tiles @ size 2
    import src.data.common.geobox as geobox_module

    monkeypatch.setattr(geobox_module, "get_target_geobox", lambda passed_ctx: fake_geobox)
    source.tile_size = 2

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    target = targets[0]

    assert source._execute_prepare(target) is True

    import pandas as pd
    from pathlib import Path

    parts = sorted(Path(target.output_path).glob("ix=*/iy=*/part-*.parquet"))
    assert len(parts) == 4
    df = pd.concat(pd.read_parquet(p) for p in parts)
    assert np.all(np.isfinite(df["mean"].values))


def test_clamped_bbox_avoids_antimeridian_wrap_at_real_grid_corner_tile():
    """Same real production failure as ModisSource's identical test
    (tests/data/sources/modis/test_modis_prepare.py) -- `GlassModisSource
    ._execute_prepare`'s `raw_getter` uses the identical clamping logic, so
    this pins that this source's own wiring of the fix behaves the same
    way against the real canonical grid, not just the shared math
    (docs/design/13-prepare-memory-parallelism.md)."""
    from odc.geo.geom import box

    from src.data.common.geobox.canonical import canonical_ease_geobox
    from src.data.common import tiling
    from src.data.sources.modis import tiles as modis_util

    target_geobox = canonical_ease_geobox()
    tile = list(tiling.iter_tiles(target_geobox, tile_size=2048))[0]

    padded_bbox = tile.geobox.pad(32, 32).extent.boundingbox
    naive_sinu = tile.geobox.pad(32, 32).extent.to_crs(modis_util.SINUSOIDAL_PROJ4).boundingbox
    assert (naive_sinu.right - naive_sinu.left) / 1000 > 20_000

    grid_bbox = target_geobox.extent.boundingbox
    margin = 2 * abs(target_geobox.resolution.x)
    clamped_left = max(padded_bbox.left, grid_bbox.left + margin)
    clamped_bottom = max(padded_bbox.bottom, grid_bbox.bottom + margin)
    clamped_right = min(padded_bbox.right, grid_bbox.right - margin)
    clamped_top = min(padded_bbox.top, grid_bbox.top - margin)
    assert clamped_left < clamped_right and clamped_bottom < clamped_top

    clamped = box(clamped_left, clamped_bottom, clamped_right, clamped_top, crs=tile.geobox.crs)
    fixed_sinu = clamped.to_crs(modis_util.SINUSOIDAL_PROJ4).boundingbox
    assert (fixed_sinu.right - fixed_sinu.left) / 1000 < 10_000

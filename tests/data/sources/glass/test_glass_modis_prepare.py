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


def _write_tile_tif(path, size=8, bounds=(-1, -1, 1, 1)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    band_names = ["mean", "std", "max", "min", "count_above", "count_below", "valid_period_count", "valid_month_count"]
    transform = from_bounds(*bounds, size, size)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=len(band_names),
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan,
    ) as dst:
        for i, name in enumerate(band_names, start=1):
            dst.write(np.full((size, size), 290.0, dtype="float32"), i)
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

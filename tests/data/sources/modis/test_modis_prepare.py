"""ModisSource._execute_prepare wiring onto the shared run_tiled_prepare
driver: year-major mosaic memoization, target_geobox threading, and a real
tiled parquet write."""

import os

import numpy as np
import pytest
import rasterio
from odc.geo.geobox import GeoBox
from rasterio.transform import from_bounds

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.modis.source import ModisSource
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection, marker_path


def _make_source(tmp_path, **raw):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("modis", raw)
    return ModisSource(ctx, cfg), ctx


def _write_tile_tif(path, value, band_names, bounds=(-1, -1, 1, 1), size=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    transform = from_bounds(*bounds, size, size)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=len(band_names),
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan,
    ) as dst:
        for i, name in enumerate(band_names, start=1):
            dst.write(np.full((size, size), value, dtype="float32"), i)
            dst.set_band_description(i, name)


def test_execute_prepare_writes_real_tiled_parquet_output(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path, year_range=[2019, 2019])
    _write_tile_tif(
        os.path.join(source.output_root(PipelineStep.FETCH), "2019", "h18v04.tif"),
        290.0, ["lst_night_mean"],
    )

    fake_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.5)  # 4x4, 2x2 tiles @ size 2
    import src.data.common.geobox as geobox_module

    monkeypatch.setattr(geobox_module, "get_or_create_canonical_geobox", lambda cache_path: fake_geobox)
    source.tile_size = 2

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    target = targets[0]

    assert source._execute_prepare(target) is True
    assert os.path.exists(marker_path(target.output_path))

    import pandas as pd
    from pathlib import Path

    parts = sorted(Path(target.output_path).glob("ix=*/iy=*/part-*.parquet"))
    assert len(parts) == 4  # 2x2 tile grid x 1 year
    df = pd.concat(pd.read_parquet(p) for p in parts)
    assert set(df["year"].unique()) == {2019}
    assert set(df.columns) == {"cell_id", "year", "lst_night_mean"}
    assert (df["lst_night_mean"] == 290.0).all()


def test_execute_prepare_reuses_one_years_mosaic_at_a_time(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path, year_range=[2019, 2020])
    _write_tile_tif(
        os.path.join(source.output_root(PipelineStep.FETCH), "2019", "h18v04.tif"), 1.0, ["lst_night_mean"]
    )
    _write_tile_tif(
        os.path.join(source.output_root(PipelineStep.FETCH), "2020", "h18v04.tif"), 2.0, ["lst_night_mean"]
    )

    fake_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.5)
    import src.data.common.geobox as geobox_module

    monkeypatch.setattr(geobox_module, "get_or_create_canonical_geobox", lambda cache_path: fake_geobox)
    source.tile_size = 2

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    target = targets[0]
    assert target.meta["years"] == [2019, 2020]

    assert source._execute_prepare(target) is True

    import pandas as pd
    from pathlib import Path

    parts = sorted(Path(target.output_path).glob("ix=*/iy=*/part-*.parquet"))
    df = pd.concat(pd.read_parquet(p) for p in parts)
    assert set(df.loc[df["year"] == 2019, "lst_night_mean"].unique()) == {1.0}
    assert set(df.loc[df["year"] == 2020, "lst_night_mean"].unique()) == {2.0}

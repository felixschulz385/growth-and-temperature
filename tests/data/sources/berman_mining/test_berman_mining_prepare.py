"""_execute_prepare wiring onto run_tiled_prepare: grid_id threading, year
derivation from the pre-reprojection mines_ds, and a real tiled write."""

import os

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401 -- registers the .rio accessor
import xarray as xr
from odc.geo.geobox import GeoBox

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.berman_mining import BermanMiningSource
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection, marker_path


def _make_source(tmp_path, grid_id="legacy_4326", **raw):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), grid_id=grid_id
    )
    cfg = SourceConfig.from_dict("berman_mining", dict(raw))
    return BermanMiningSource(ctx, cfg), ctx


def _fake_mines_ds():
    # odc-geo needs >=2 coordinate values per spatial axis to infer a valid
    # affine/GeoBox (a single-point lat/lon can't imply a pixel resolution)
    # -- a 2x2 spatial grid is the minimum that reprojects cleanly.
    lat, lon = [0.75, 0.25], [0.25, 0.75]
    data = np.zeros((2, 2, 2))
    data[0, 0, :] = [1.0, 2.0]
    return xr.Dataset(
        {
            "nb_mines_a": (("latitude", "longitude", "year"), data),
            "nb_diamond": (("latitude", "longitude", "year"), data),
        },
        coords={"latitude": lat, "longitude": lon, "year": [2019, 2020]},
    ).rio.write_crs(4326)


def test_execute_prepare_threads_ctx_grid_id_and_derives_years_pre_reproject(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path, grid_id="ease6933")
    monkeypatch.setattr(source, "_create_mining_dataset", lambda year_range: _fake_mines_ds())

    import src.data.common.geobox as geobox_module
    import src.data.common.prepare.driver as driver_module

    captured = {}
    fake_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.25)

    def fake_get_target_geobox(passed_ctx):
        captured["ctx"] = passed_ctx
        return fake_geobox

    def fake_run_tiled_prepare(*, target_geobox, years, **kwargs):
        captured["target_geobox"] = target_geobox
        captured["years"] = years
        return True

    monkeypatch.setattr(geobox_module, "get_target_geobox", fake_get_target_geobox)
    monkeypatch.setattr(driver_module, "run_tiled_prepare", fake_run_tiled_prepare)

    target = StepTarget(
        source_id=source.ID, step=PipelineStep.PREPARE, key="all",
        output_path=str(tmp_path / "out" / "berman_mining"),
        completion=Completion.MARKER, meta={"year_range": None},
    )
    assert source._execute_prepare(target) is True
    assert captured["ctx"] is ctx
    assert captured["target_geobox"] is fake_geobox
    assert captured["years"] == [2019, 2020]


def test_execute_prepare_writes_real_tiled_parquet_output(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path)
    monkeypatch.setattr(source, "_create_mining_dataset", lambda year_range: _fake_mines_ds())

    fake_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.25)  # 8x8, 2x2 tiles @ size 4
    monkeypatch.setattr(source, "_get_or_create_geobox", lambda: fake_geobox)
    source.tile_size = 4

    target = StepTarget(
        source_id=source.ID, step=PipelineStep.PREPARE, key="all",
        output_path=str(tmp_path / "out" / "berman_mining"),
        completion=Completion.MARKER, meta={"year_range": None},
    )
    assert source._execute_prepare(target) is True
    assert os.path.exists(marker_path(target.output_path))

    from pathlib import Path

    parts = sorted(Path(target.output_path).glob("ix=*/iy=*/part-*.parquet"))
    assert len(parts) == 8  # 2x2 tile grid x 2 years
    df = pd.concat(pd.read_parquet(p) for p in parts)
    assert set(df["year"].unique()) == {2019, 2020}
    assert set(df.columns) == {"cell_id", "year", "nb_mines_a", "nb_diamond"}
    # cast to uint8 with fillna(255) -- no NaN should survive into output
    assert df["nb_mines_a"].dtype == np.uint8
    assert df["nb_diamond"].dtype == np.uint8

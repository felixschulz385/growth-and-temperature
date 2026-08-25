"""NtlHarmSource: FETCH/PREPARE.

PREPARE is planned by a live crawl of FETCH's raw output directory (see
src/data/sources/ntl_harm.py's module docstring) and executed as one tiled
`run_tiled_prepare()` call per source (no separate GRID step, no
intermediate annual zarr).
`test_execute_prepare_writes_a_real_reprojected_tiled_output` exercises the
full raw-file -> tiled zarr path against a real (synthetic) raster instead
of mocks.
"""

import os

import numpy as np
import rasterio
from odc.geo.geobox import GeoBox
from rasterio.transform import from_origin

import src.data.common.geobox as geobox_module
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.ntl_harm import NtlHarmSource
from src.data.sources.steps import Completion, PipelineStep, TargetSelection, marker_path


def _make_source(tmp_path, **raw):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("ntl_harm", {"data_path": "ntl_harm/harmonized", **raw})
    return NtlHarmSource(ctx, cfg), ctx


def _write_raw_file(source, filename):
    raw_root = source.output_root(PipelineStep.FETCH)
    os.makedirs(raw_root, exist_ok=True)
    open(os.path.join(raw_root, filename), "w").close()


def test_steps_is_fetch_and_prepare_only():
    assert NtlHarmSource.STEPS == (PipelineStep.FETCH, PipelineStep.PREPARE)


def test_get_all_entrypoints_is_static_from_year_range(tmp_path, monkeypatch):
    """`get_all_entrypoints()` must never hit the network -- `data summary`
    (`catalog.cached_entrypoint_counts()`) relies on `STATIC_ENTRYPOINTS`
    sources being genuinely safe to call outside a real FETCH run."""
    import requests

    def _boom(*args, **kwargs):
        raise AssertionError("get_all_entrypoints() must not touch the network")

    monkeypatch.setattr(requests, "get", _boom)

    source, _ = _make_source(tmp_path, year_range=[2020, 2022])
    assert source.get_all_entrypoints() == [
        {"year": 2020, "day": 1}, {"year": 2021, "day": 1}, {"year": 2022, "day": 1},
    ]


def test_default_resampling_is_sum(tmp_path):
    source, _ = _make_source(tmp_path)
    assert source.resampling == "sum"


def test_resampling_overridable(tmp_path):
    source, _ = _make_source(tmp_path, resampling="nearest")
    assert source.resampling == "nearest"


def test_prepare_plan_empty_when_no_raw_files(tmp_path):
    source, _ = _make_source(tmp_path)
    assert source.plan(PipelineStep.PREPARE, TargetSelection()) == []


def test_prepare_plan_one_target_covering_every_available_year(tmp_path):
    source, _ = _make_source(tmp_path)
    for fname in ("harmonized_2019.tif", "harmonized_2020.tif", "harmonized_2021.tif"):
        _write_raw_file(source, fname)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    target = targets[0]
    assert target.key == "all"
    assert target.meta["years"] == [2019, 2020, 2021]
    assert target.completion == Completion.MARKER
    assert target.output_path.endswith("ntl_harm")


def test_prepare_plan_prefers_tif_over_zip_per_year(tmp_path):
    source, _ = _make_source(tmp_path)
    _write_raw_file(source, "harmonized_2020.zip")
    _write_raw_file(source, "harmonized_2020.tif")

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert targets[0].meta["raw_files"][2020] == "harmonized_2020.tif"


def test_prepare_plan_respects_year_selection(tmp_path):
    source, _ = _make_source(tmp_path)
    for fname in ("harmonized_2019.tif", "harmonized_2020.tif", "harmonized_2021.tif"):
        _write_raw_file(source, fname)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection(year_range=(2020, 2021)))
    assert targets[0].meta["years"] == [2020, 2021]


def _write_sample_geotiff(path, size=8, value=1234.0):
    transform = from_origin(-1.0, 1.0, 2.0 / size, 2.0 / size)
    data = np.full((1, size, size), value, dtype="float32")
    with rasterio.open(
        str(path), "w", driver="GTiff", height=size, width=size, count=1, dtype="float32",
        crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data)


def test_execute_prepare_writes_a_real_reprojected_tiled_output(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path, tile_size=4)
    os.makedirs(source.output_root(PipelineStep.FETCH), exist_ok=True)
    _write_sample_geotiff(
        os.path.join(source.output_root(PipelineStep.FETCH), "harmonized_2020.tif"), size=8, value=1234.0
    )

    target_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.25)  # 8x8, 2x2 tiles @ size 4
    monkeypatch.setattr(geobox_module, "get_target_geobox", lambda passed_ctx: target_geobox)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    target = targets[0]

    ok = source._execute_prepare(target)
    assert ok is True
    assert os.path.exists(marker_path(target.output_path))

    import glob

    import pandas as pd

    parts = sorted(glob.glob(os.path.join(target.output_path, "ix=*", "iy=*", "part-2020.parquet")))
    assert len(parts) == 4  # 2x2 tile grid at tile_size=4 on an 8x8 geobox
    df = pd.concat(pd.read_parquet(p) for p in parts)
    assert len(df) == 64  # 8x8 pixels total
    assert np.all(np.isfinite(df[source.VARIABLE_NAME].values))


def test_execute_prepare_handles_zip_wrapped_source(tmp_path, monkeypatch):
    """`_load_year`'s zip-extract temp dir is deliberately never deleted (no
    safe point to do so under concurrent worker access -- see
    src/data/common/prepare/raster_year_parallel.py's module docstring), so
    this only checks the pipeline still produces correct output through a
    real Dask client, not that the dir gets cleaned up."""
    import zipfile

    source, ctx = _make_source(tmp_path, tile_size=4)
    os.makedirs(source.output_root(PipelineStep.FETCH), exist_ok=True)
    tif_path = os.path.join(tmp_path, "harmonized_2020.tif")
    _write_sample_geotiff(tif_path, size=8, value=1234.0)
    zip_path = os.path.join(source.output_root(PipelineStep.FETCH), "harmonized_2020.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(tif_path, arcname="harmonized_2020.tif")

    target_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.25)  # 8x8, 2x2 tiles @ size 4
    monkeypatch.setattr(geobox_module, "get_target_geobox", lambda passed_ctx: target_geobox)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    target = targets[0]

    assert source._execute_prepare(target) is True

    import glob

    import pandas as pd

    parts = sorted(glob.glob(os.path.join(target.output_path, "ix=*", "iy=*", "part-2020.parquet")))
    df = pd.concat(pd.read_parquet(p) for p in parts)
    assert np.all(np.isfinite(df[source.VARIABLE_NAME].values))

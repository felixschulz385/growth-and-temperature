"""EsacciSource._execute_prepare: lazy per-tile clip (sel_bbox), routed
through Dask's worker processes (run_tiled_prepare_dask_year_major) instead
of the old single-process serial loop -- see
docs/design/13-prepare-memory-parallelism.md. `_load_nc_as_dataset` itself
needs `h5netcdf` (not installed in this environment), so it's monkeypatched
(as a plain 3-arg function, matching the now-`@staticmethod` signature
`(file_path, year, temp_dir)`) to return a synthetic dask-backed Dataset
shaped like a real read (chunked, EPSG:4326, `latitude`/`longitude` dims) --
this test exercises the clip/compute/reproject wiring around it, not the
netCDF read itself. Uses the source's real (small, local) Dask client -- not
a stub -- since `client.submit` now needs an actual `distributed.Client`.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.esacci import EsacciSource
from src.data.sources.steps import PipelineStep, TargetSelection, marker_path


def _make_source(tmp_path, **raw):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("esacci", {"data_path": "esacci/landcover", **raw})
    return EsacciSource(ctx, cfg), ctx


def _write_raw_file(source, relative_path):
    raw_root = source.output_root(PipelineStep.FETCH)
    full = os.path.join(raw_root, relative_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    open(full, "w").close()


def _fake_year_dataset(year: int, value: int) -> xr.Dataset:
    lat = np.linspace(9, -9, 8)  # descending, matches real ESA CCI convention
    lon = np.linspace(-9, 9, 8)  # ascending
    data = np.full((1, 1, 8, 8), value, dtype="uint8")
    ds = xr.Dataset(
        {"lccs_class": (("time", "band", "latitude", "longitude"), data)},
        coords={"time": [pd.Timestamp(f"{year}-12-31")], "band": [1], "latitude": lat, "longitude": lon},
    ).chunk({"latitude": 4, "longitude": 4})
    return ds.rio.write_crs("EPSG:4326")


def test_execute_prepare_clips_per_tile_and_writes_real_output(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path, year_range=[2020, 2020])
    _write_raw_file(source, "2020/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2020-v2.0.7.nc")

    monkeypatch.setattr(
        EsacciSource, "_load_nc_as_dataset", lambda file_path, year, temp_dir, extra=None: _fake_year_dataset(year, 42)
    )

    fake_geobox = GeoBox.from_bbox((-8, -8, 8, 8), crs="EPSG:4326", resolution=4)  # 4x4, 2x2 tiles @ size 2
    import src.data.common.geobox as geobox_module

    monkeypatch.setattr(geobox_module, "get_target_geobox", lambda ctx: fake_geobox)
    source.tile_size = 2

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    target = targets[0]

    assert source._execute_prepare(target) is True
    assert os.path.exists(marker_path(target.output_path))

    parts = sorted(Path(target.output_path).glob("ix=*/iy=*/part-*.parquet"))
    assert len(parts) == 4  # 2x2 tile grid x 1 year
    df = pd.concat(pd.read_parquet(p) for p in parts)
    assert set(df["year"].unique()) == {2020}
    assert (df["lccs_class"] == 42).all()


def test_execute_prepare_handles_multiple_years(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path, year_range=[2019, 2020])
    _write_raw_file(source, "2019/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2019-v2.0.7.nc")
    _write_raw_file(source, "2020/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2020-v2.0.7.nc")

    monkeypatch.setattr(
        EsacciSource,
        "_load_nc_as_dataset",
        lambda file_path, year, temp_dir, extra=None: _fake_year_dataset(year, 1 if year == 2019 else 2),
    )

    fake_geobox = GeoBox.from_bbox((-8, -8, 8, 8), crs="EPSG:4326", resolution=4)
    import src.data.common.geobox as geobox_module

    monkeypatch.setattr(geobox_module, "get_target_geobox", lambda ctx: fake_geobox)
    source.tile_size = 2

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    target = targets[0]
    assert target.meta["years"] == [2019, 2020]

    assert source._execute_prepare(target) is True

    parts = sorted(Path(target.output_path).glob("ix=*/iy=*/part-*.parquet"))
    df = pd.concat(pd.read_parquet(p) for p in parts)
    assert set(df.loc[df["year"] == 2019, "lccs_class"].unique()) == {1}
    assert set(df.loc[df["year"] == 2020, "lccs_class"].unique()) == {2}

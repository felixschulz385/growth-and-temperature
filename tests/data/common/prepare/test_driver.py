"""run_tiled_prepare() end-to-end against a fake raw_getter -- no ledger, no
per-source knowledge. Mirrors tests/data/common/fetch/test_driver.py's shape:
downloads/writes what's outstanding, resumes cleanly, records failures,
refuses concurrent invocation.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from src.data.common import lockfile, statusfile, tiling
from src.data.common.prepare.driver import prepare_status, run_tiled_prepare, status_dir_for
from src.data.common.raster.spatial import SpatialProcessor
from src.data.sources.steps import marker_path


@pytest.fixture
def target_geobox():
    return GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.25)  # 8x8, 2x2 @ tile_size=4


@pytest.fixture
def processor(tmp_path):
    return SpatialProcessor(hpc_root=str(tmp_path))


def _fake_source_ds(year: int, size: int = 8):
    lon = np.linspace(-1.0, 1.0, size)
    lat = np.linspace(1.0, -1.0, size)
    data = np.full((1, 1, size, size), float(year), dtype="float32")
    ds = xr.Dataset(
        {"value": (("time", "band", "latitude", "longitude"), data)},
        coords={"time": [pd.Timestamp(f"{year}-12-31")], "band": [1], "latitude": lat, "longitude": lon},
    )
    return ds.rio.write_crs(4326)


def _always_returns_full_extent(tile, year):
    # Every raw-getter call gets the full source extent -- no halo needed
    # for this synthetic fixture, since it isn't a real edge-effect test.
    return _fake_source_ds(year)


def _make_getter(fail_units=frozenset()):
    calls = []

    def getter(tile, year):
        calls.append((year, tile.id))
        if (year, tile.id) in fail_units:
            raise RuntimeError(f"simulated failure for {year}/{tile.id}")
        return _fake_source_ds(year)

    getter.calls = calls
    return getter


def test_run_tiled_prepare_fills_every_tile_and_marks_complete(tmp_path, target_geobox, processor):
    output_path = str(tmp_path / "output")
    ok = run_tiled_prepare(
        output_path=output_path,
        years=[2020],
        variables=["value"],
        target_geobox=target_geobox,
        processor=processor,
        raw_getter=_always_returns_full_extent,
        target_dims=target_geobox.dimensions,
        tile_size=4,
        packaging_attrs={},
    )
    assert ok is True
    assert os.path.exists(marker_path(output_path))

    parts = sorted(Path(output_path).glob("ix=*/iy=*/part-2020.parquet"))
    assert len(parts) == 4  # 2x2 tile grid at tile_size=4 on an 8x8 geobox
    df = pd.concat(pd.read_parquet(p) for p in parts)
    assert (df["value"] == 2020.0).all()
    assert (df["year"] == 2020).all()


def test_second_run_is_a_noop_skips_completed_units(tmp_path, target_geobox, processor):
    output_path = str(tmp_path / "output")
    getter = _make_getter()
    kwargs = dict(
        output_path=output_path, years=[2020], variables=["value"], target_geobox=target_geobox,
        processor=processor, raw_getter=getter, target_dims=target_geobox.dimensions, tile_size=4,
        packaging_attrs={},
    )
    assert run_tiled_prepare(**kwargs) is True
    n_calls_after_first = len(getter.calls)

    assert run_tiled_prepare(**kwargs) is True
    assert len(getter.calls) == n_calls_after_first  # nothing re-fetched


def test_failure_in_one_tile_records_status_and_returns_false(tmp_path, target_geobox, processor):
    output_path = str(tmp_path / "output")
    getter = _make_getter(fail_units={(2020, "0000_0000")})

    ok = run_tiled_prepare(
        output_path=output_path, years=[2020], variables=["value"], target_geobox=target_geobox,
        processor=processor, raw_getter=getter, target_dims=target_geobox.dimensions, tile_size=4,
        packaging_attrs={},
    )
    assert ok is False
    assert not os.path.exists(marker_path(output_path))  # not complete -- one tile still missing

    status_dir = status_dir_for(output_path)
    status = statusfile.read(statusfile.status_path(status_dir, "2020/0000_0000"))
    assert status["attempts"] == 1
    assert "simulated failure" in status["last_error"]


def test_retry_after_failure_succeeds_and_completes(tmp_path, target_geobox, processor):
    output_path = str(tmp_path / "output")
    fail_unit = (2020, "0000_0000")
    getter = _make_getter(fail_units={fail_unit})
    kwargs = dict(
        output_path=output_path, years=[2020], variables=["value"], target_geobox=target_geobox,
        processor=processor, target_dims=target_geobox.dimensions, tile_size=4, packaging_attrs={},
    )
    assert run_tiled_prepare(raw_getter=getter, **kwargs) is False

    getter2 = _make_getter()  # origin "recovers"
    assert run_tiled_prepare(raw_getter=getter2, **kwargs) is True
    assert os.path.exists(marker_path(output_path))
    # Only the previously-outstanding tile was retried, not every tile again.
    assert len(getter2.calls) == 1


def test_processing_version_bump_forces_reprocessing(tmp_path, target_geobox, processor):
    output_path = str(tmp_path / "output")
    getter_v1 = _make_getter()
    common = dict(
        output_path=output_path, years=[2020], variables=["value"], target_geobox=target_geobox,
        processor=processor, target_dims=target_geobox.dimensions, tile_size=4, packaging_attrs={},
    )
    assert run_tiled_prepare(raw_getter=getter_v1, processing_version="1", **common) is True

    getter_v2 = _make_getter()
    assert run_tiled_prepare(raw_getter=getter_v2, processing_version="2", **common) is True
    assert len(getter_v2.calls) == 4  # all 4 tiles reprocessed under the new version


def test_run_tiled_prepare_refuses_concurrent_invocation(tmp_path, target_geobox, processor):
    output_path = str(tmp_path / "output")
    status_dir = status_dir_for(output_path)
    lock_path = os.path.join(status_dir, "prepare.lock")
    lockfile.acquire(lock_path)
    try:
        ok = run_tiled_prepare(
            output_path=output_path, years=[2020], variables=["value"], target_geobox=target_geobox,
            processor=processor, raw_getter=_always_returns_full_extent, target_dims=target_geobox.dimensions,
            tile_size=4, packaging_attrs={},
        )
        assert ok is False
    finally:
        lockfile.release(lock_path)


def test_prepare_status_counts_before_any_run(tmp_path, target_geobox):
    output_path = str(tmp_path / "output")
    counts = prepare_status(output_path, [2020], target_geobox, tile_size=4)
    assert counts == {"complete": 0, "outstanding": 4, "unavailable": 0}


def test_run_tiled_prepare_static_years_none_writes_one_part_per_tile_no_year(tmp_path, target_geobox, processor):
    output_path = str(tmp_path / "output")

    def static_getter(tile, year):
        assert year is None
        return _fake_source_ds(2020)  # value irrelevant to year; getter ignores it

    ok = run_tiled_prepare(
        output_path=output_path,
        years=None,
        target_geobox=target_geobox,
        processor=processor,
        raw_getter=static_getter,
        tile_size=4,
    )
    assert ok is True
    assert os.path.exists(marker_path(output_path))

    parts = sorted(Path(output_path).glob("ix=*/iy=*/part.parquet"))
    assert len(parts) == 4
    df = pd.concat(pd.read_parquet(p) for p in parts)
    assert "year" not in df.columns
    assert (df["value"] == 2020.0).all()

    status_dir = status_dir_for(output_path)
    # static units are keyed by tile id alone, not "<year>/<tile id>"
    assert statusfile.read(statusfile.status_path(status_dir, "0000_0000")) is not None


def test_run_tiled_prepare_reproject_false_uses_raw_getter_output_as_is(tmp_path, target_geobox, processor):
    output_path = str(tmp_path / "output")
    tile_size = 4

    def prerasterized_getter(tile, year):
        h, w = tile.geobox.shape
        return xr.Dataset({"land_mask": (("y", "x"), np.ones((h, w), dtype="uint8"))})

    ok = run_tiled_prepare(
        output_path=output_path,
        years=None,
        target_geobox=target_geobox,
        processor=processor,
        raw_getter=prerasterized_getter,
        tile_size=tile_size,
        reproject=False,
    )
    assert ok is True

    parts = sorted(Path(output_path).glob("ix=*/iy=*/part.parquet"))
    assert len(parts) == 4
    df = pd.read_parquet(parts[0])
    assert (df["land_mask"] == 1).all()
    assert len(df) == tile_size * tile_size


def test_prepare_status_counts_after_partial_failure(tmp_path, target_geobox, processor):
    output_path = str(tmp_path / "output")
    getter = _make_getter(fail_units={(2020, "0000_0000")})
    run_tiled_prepare(
        output_path=output_path, years=[2020], variables=["value"], target_geobox=target_geobox,
        processor=processor, raw_getter=getter, target_dims=target_geobox.dimensions, tile_size=4,
        packaging_attrs={}, max_attempts=5,
    )
    counts = prepare_status(output_path, [2020], target_geobox, tile_size=4)
    assert counts == {"complete": 3, "outstanding": 1, "unavailable": 0}

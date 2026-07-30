"""Tests for the per-variable resampling override in SpatialProcessor.

docs/design/04-ingest.md §1: `process_spatial_standard` hardcoded
`resampling="nearest"`, which is flux-destroying for radiance fields like
nighttime lights. This must become a per-call override -- never a new shared
default -- since categorical sources (ESACCI land cover) still need
"nearest"/"mode".
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

import src.data.common.raster.spatial as spatial_module
from src.data.common.raster.spatial import SpatialProcessor


def _write_sample_source_zarr(path, year: int, size: int = 8):
    lon = np.linspace(-1.0, 1.0, size)
    lat = np.linspace(1.0, -1.0, size)
    data = np.arange(size * size, dtype="float32").reshape(1, 1, size, size)
    ds = xr.Dataset(
        {"value": (("time", "band", "latitude", "longitude"), data)},
        coords={
            "time": [pd.Timestamp(f"{year}-12-31")],
            "band": [1],
            "latitude": lat,
            "longitude": lon,
        },
    )
    ds = ds.rio.write_crs(4326)
    ds.to_zarr(str(path), mode="w", zarr_format=3, consolidated=False)


def _run_with_resampling_spy(tmp_path, monkeypatch, resampling_kwargs):
    source_path = tmp_path / "source_2020.zarr"
    _write_sample_source_zarr(source_path, 2020)

    target_geobox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.25)

    processor = SpatialProcessor(hpc_root=str(tmp_path))
    monkeypatch.setattr(processor, "get_target_geobox", lambda: target_geobox)

    captured = {}
    real_xr_reproject = spatial_module.xr_reproject

    def spy_xr_reproject(*args, **kwargs):
        captured["resampling"] = kwargs.get("resampling")
        return real_xr_reproject(*args, **kwargs)

    monkeypatch.setattr(spatial_module, "xr_reproject", spy_xr_reproject)

    output_path = tmp_path / "output.zarr"
    success = processor.process_spatial_standard(
        source_files=[str(source_path)],
        output_path=str(output_path),
        years_to_process=[2020],
        year_pattern_func=lambda p: 2020,
        **resampling_kwargs,
    )
    assert success
    return captured


def test_process_spatial_standard_defaults_to_nearest(tmp_path, monkeypatch):
    """Unchanged behaviour: no resampling override given -> still "nearest"."""
    captured = _run_with_resampling_spy(tmp_path, monkeypatch, {})
    assert captured["resampling"] == "nearest"


def test_process_spatial_standard_threads_sum_resampling(tmp_path, monkeypatch):
    """docs/design/04-ingest.md §1: lights must be able to request "sum"."""
    captured = _run_with_resampling_spy(tmp_path, monkeypatch, {"resampling": "sum"})
    assert captured["resampling"] == "sum"


def test_write_year_to_zarr_default_signature_still_nearest():
    import inspect

    sig = inspect.signature(SpatialProcessor.write_year_to_zarr)
    assert sig.parameters["resampling"].default == "nearest"


def test_process_spatial_standard_default_signature_still_nearest():
    import inspect

    sig = inspect.signature(SpatialProcessor.process_spatial_standard)
    assert sig.parameters["resampling"].default == "nearest"

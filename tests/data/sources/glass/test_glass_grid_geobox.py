"""Regression tests for the ease6933 grid-switch correctness fix in
GlassSource: before this fix, `_execute_grid` called `get_or_create_geobox()`
directly (ignoring `ctx.grid_id`), and `_create_empty_target_zarr`/
`_process_year_tiles` both hardcoded `latitude`/`longitude` dim names --
a projected canonical geobox (`y`/`x` dims) would have raised a `KeyError`
in either method.
"""

import contextlib

import numpy as np
import pandas as pd
import xarray as xr

from src.data.common.geobox.canonical import canonical_ease_geobox
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.glass.source import GlassSource
from src.data.sources.steps import Completion, PipelineStep, StepTarget


def _coarse_ease_geobox():
    return canonical_ease_geobox(resolution_m=50_000.0, lat_clip_deg=60.0)


def _make_source(tmp_path, grid_id="legacy_4326", layout="legacy"):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"),
        local_index_dir=str(tmp_path / "index"),
        grid_id=grid_id,
        layout=layout,
    )
    cfg = SourceConfig.from_dict(
        "glass_modis",
        {
            "base_url": "https://glass.hku.hk/archive/LST/MODIS/Daily/1KM/",
            "day_range": {"start": [2000, 55], "end": [2020, 365]},
        },
    )
    return GlassSource(ctx, cfg), ctx


def _write_sample_zarr(path):
    ds = xr.Dataset(
        {"lst": (("time", "band", "y", "x"), np.zeros((1, 1, 2, 2), dtype=np.uint16))},
        coords={"time": pd.to_datetime(["2019-12-31"]), "band": [1], "y": [1, 0], "x": [0, 1]},
    )
    ds.to_zarr(path, mode="w", consolidated=False)


def test_create_empty_target_zarr_uses_y_x_dims_for_ease_geobox(tmp_path):
    source, _ = _make_source(tmp_path)
    sample_path = str(tmp_path / "2019.zarr")
    _write_sample_zarr(sample_path)

    output_path = str(tmp_path / "out" / "modis_timeseries_reprojected.zarr")
    geobox = _coarse_ease_geobox()

    assert source._create_empty_target_zarr(output_path, geobox, (sample_path,))

    ds = xr.open_zarr(output_path, consolidated=False)
    assert set(ds["lst"].dims) == {"time", "band", "y", "x"}


def test_execute_prepare_threads_ctx_grid_id_into_target_geobox(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path, grid_id="ease6933")

    import src.data.common.geobox as geobox_module

    captured = {}
    fake_geobox = _coarse_ease_geobox()

    def fake_get_target_geobox(passed_ctx):
        captured["ctx"] = passed_ctx
        return fake_geobox

    monkeypatch.setattr(geobox_module, "get_target_geobox", fake_get_target_geobox)
    monkeypatch.setattr(
        source, "_group_daily_files", lambda selection: [{"year": 2019, "grid_cell": "h25v06", "key": "2019/h25v06", "files": []}]
    )
    monkeypatch.setattr(source, "_ensure_annual_zarr", lambda group: "dummy.zarr")
    monkeypatch.setattr(source, "_create_empty_target_zarr", lambda *a, **k: True)
    monkeypatch.setattr(source, "_process_years_chunked", lambda *a, **k: True)
    monkeypatch.setattr(type(source), "_dask_client", lambda self: contextlib.nullcontext(_FakeClient()))

    target = StepTarget(
        source_id=source.ID,
        step=PipelineStep.PREPARE,
        key="all",
        output_path=str(tmp_path / "out" / "modis_timeseries_reprojected.zarr"),
        inputs=(),
        completion=Completion.MARKER,
        meta={"years_available": [2019], "group_keys": ["2019/h25v06"]},
    )
    assert source._execute_prepare(target) is True
    assert captured["ctx"] is ctx


class _FakeClient:
    dashboard_link = None


def test_multi_file_year_temp_path_uses_layout_output_root_not_string_split(tmp_path, monkeypatch):
    # Regression test: _process_years_chunked used to derive the PREPARE-
    # stage temp path by string-splitting the GRID output_path on the
    # literal substring "stage_2" -- a hack that silently breaks once GRID
    # output no longer contains that substring at all (layout=v2's
    # grid/<grid_id>/ paths never do).
    import os

    import src.data.sources.layout as layout_module

    for layout in ("legacy", "v2"):
        source, ctx = _make_source(tmp_path, layout=layout)
        captured = {}

        def fake_aggregate(self_unused, year_files, annual_temp_path, year):
            captured["annual_temp_path"] = annual_temp_path
            return True

        monkeypatch.setattr(GlassSource, "_aggregate_year_files", fake_aggregate)
        monkeypatch.setattr(source, "_process_year_tiles", lambda *a, **k: True)

        geobox = _coarse_ease_geobox()
        # data_source_kind="MODIS" (default for glass_modis) extracts the
        # year from a "/YYYY/" path segment, not the filename.
        year_files = ["/2019/h18v04.zarr", "/2019/h20v08.zarr"]
        source._process_years_chunked(year_files, "unused_output_path", geobox, [2019])

        prepare_root = layout_module.output_root(
            ctx.data_root, source.path_prefix, PipelineStep.PREPARE, layout=layout
        )
        assert captured["annual_temp_path"] == os.path.join(prepare_root, "2019", "temp_combined.tzarr")

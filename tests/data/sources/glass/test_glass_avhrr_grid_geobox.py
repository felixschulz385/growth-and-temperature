"""Regression test for the ease6933 grid-switch correctness fix in GLASS-
AVHRR's PREPARE path: `_execute_prepare` must thread `ctx.grid_id` into
`get_target_geobox()` rather than hardcoding the legacy EPSG:4326 grid.

docs/design/12-glass-modis-rebuild.md §6: this source's PREPARE step now
runs on the shared `run_tiled_prepare` driver (y/x-vs-latitude/longitude dim
handling and per-tile reprojection are exercised generically by
tests/data/common/raster/test_process_tile_region.py and
tests/data/common/prepare/test_driver.py) -- this file only covers what's
specific to GlassAvhrrSource's own `_execute_prepare` wiring.
"""

import contextlib

from src.data.common.geobox.canonical import canonical_ease_geobox
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.steps import Completion, PipelineStep, StepTarget


def _coarse_ease_geobox():
    return canonical_ease_geobox(resolution_m=50_000.0, lat_clip_deg=60.0)


def _make_source(tmp_path, grid_id="legacy_4326"):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"),
        local_index_dir=str(tmp_path / "index"),
        grid_id=grid_id,
    )
    cfg = SourceConfig.from_dict(
        "glass_avhrr",
        {
            "base_url": "https://glass.hku.hk/archive/LST/AVHRR/0.05D/",
            "day_range": {"start": [1992, 1], "end": [2020, 365]},
        },
    )
    from src.data.sources.glass.avhrr import GlassAvhrrSource

    return GlassAvhrrSource(ctx, cfg), ctx


class _FakeClient:
    dashboard_link = None


def test_execute_prepare_threads_ctx_grid_id_into_target_geobox(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path, grid_id="ease6933")

    import src.data.common.geobox as geobox_module
    import src.data.common.prepare.driver as driver_module

    captured = {}
    fake_geobox = _coarse_ease_geobox()

    def fake_get_target_geobox(passed_ctx):
        captured["ctx"] = passed_ctx
        return fake_geobox

    def fake_run_tiled_prepare(*, target_geobox, **kwargs):
        captured["target_geobox"] = target_geobox
        return True

    monkeypatch.setattr(geobox_module, "get_target_geobox", fake_get_target_geobox)
    monkeypatch.setattr(driver_module, "run_tiled_prepare", fake_run_tiled_prepare)
    monkeypatch.setattr(
        source, "_group_daily_files", lambda selection: [{"year": 2019, "grid_cell": "global", "key": "2019", "files": []}]
    )
    monkeypatch.setattr(source, "_ensure_annual_zarr", lambda group: "dummy.zarr")
    monkeypatch.setattr(type(source), "_dask_client", lambda self: contextlib.nullcontext(_FakeClient()))

    target = StepTarget(
        source_id=source.ID,
        step=PipelineStep.PREPARE,
        key="all",
        output_path=str(tmp_path / "out" / "avhrr_timeseries_reprojected"),
        inputs=(),
        completion=Completion.MARKER,
        meta={"years_available": [2019], "group_keys": ["2019/h25v06"]},
    )
    assert source._execute_prepare(target) is True
    assert captured["ctx"] is ctx
    assert captured["target_geobox"] is fake_geobox

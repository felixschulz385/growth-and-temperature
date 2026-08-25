"""Regression test for the ease6933 grid-switch correctness fix.

Before this fix, AcagSource._execute_grid constructed SpatialProcessor with
no `target_geobox`, so it always reprojected onto the legacy EPSG:4326 grid
regardless of `ctx.grid_id`, silently mislabeling output written under a
`stage_2_ease6933` path. This is representative of every other tiled raster
source (esacci, eog, glass) -- all follow the identical
`get_target_geobox(self.ctx)` -> `SpatialProcessor(target_geobox=...)`
pattern in their own `_execute_prepare` (post-Plan-2, PREPARE does what used
to be `_execute_grid`'s job -- see src/data/sources/acag.py's module
docstring).

`get_target_geobox` itself is monkeypatched rather than exercised for real:
its own branching (legacy vs canonical) is covered by
tests/data/common/geobox/test_target_selection.py.
"""

import contextlib

import src.data.common.geobox as geobox_module
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.acag import AcagSource
from src.data.sources.steps import Completion, PipelineStep, StepTarget


class _FakeGeobox:
    def __init__(self, crs):
        self.crs = crs
        self.dimensions = ("latitude", "longitude")


class _SpyRunTiledPrepare:
    """Records the kwargs `run_tiled_prepare` was called with; never
    touches dask/xarray/zarr."""

    captured_kwargs = None

    def __call__(self, **kwargs):
        type(self).captured_kwargs = kwargs
        return True


def _make_source(tmp_path, grid_id):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), grid_id=grid_id
    )
    cfg = SourceConfig.from_dict("acag", {"data_path": "acag/pm25"})
    return AcagSource(ctx, cfg), ctx


def _run_execute_prepare(tmp_path, grid_id, fake_geobox, monkeypatch):
    import src.data.common.prepare.driver as driver_module

    source, ctx = _make_source(tmp_path, grid_id)
    monkeypatch.setattr(driver_module, "run_tiled_prepare", _SpyRunTiledPrepare())
    monkeypatch.setattr(geobox_module, "get_target_geobox", lambda passed_ctx: fake_geobox)
    monkeypatch.setattr(type(source), "_dask_client", lambda self: contextlib.nullcontext("fake-client"))

    target = StepTarget(
        source_id=source.ID,
        step=PipelineStep.PREPARE,
        key="all",
        output_path=str(tmp_path / "out" / "acag_pm25_timeseries_reprojected.zarr"),
        inputs=("dummy_input.nc",),
        completion=Completion.MARKER,
        meta={"years": [2019], "raw_files": {2019: "dummy_input.nc"}},
    )
    result = source._execute_prepare(target)
    assert result is True


def test_execute_prepare_threads_legacy_geobox_through_to_run_tiled_prepare(tmp_path, monkeypatch):
    fake_geobox = _FakeGeobox(crs="EPSG:4326")
    _run_execute_prepare(tmp_path, "legacy_4326", fake_geobox, monkeypatch)
    assert _SpyRunTiledPrepare.captured_kwargs["target_geobox"] is fake_geobox


def test_execute_prepare_threads_canonical_ease_geobox_through_to_run_tiled_prepare(tmp_path, monkeypatch):
    fake_geobox = _FakeGeobox(crs="EPSG:6933")
    _run_execute_prepare(tmp_path, "ease6933", fake_geobox, monkeypatch)
    assert _SpyRunTiledPrepare.captured_kwargs["target_geobox"] is fake_geobox

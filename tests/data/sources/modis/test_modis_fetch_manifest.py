"""ModisSource's FETCH step: completion is plain local-disk presence
(`Completion.PATH_EXISTS`, matching every other source,
`src.data.common.fetch.manifest`), and a failed (year, tile) unit's
retry/error history lives in a small JSON sidecar under FETCH's own output
root, written via `manifest.record_failure`/`clear_failure` directly (MODIS
has no crawl catalog, see module docstring).
"""

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401 -- registers the .rio accessor _write_annual_geotiff needs
import xarray as xr

from src.data.common import statusfile
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.modis.source import ModisSource
from src.data.sources.steps import PipelineStep, TargetSelection

pytest.importorskip("rasterio")


def _make_source(tmp_path, tiles=("h18v04",), year_range=(2019, 2019)):
    data_root = str(tmp_path / "data_root")
    local_index_dir = str(tmp_path / "index")
    ctx = PipelineContext(data_root=data_root, local_index_dir=local_index_dir, layout="legacy")
    cfg = SourceConfig.from_dict("modis", {"year_range": list(year_range), "tiles": list(tiles)})
    return ModisSource(ctx, cfg), ctx


def _fake_dataset() -> xr.Dataset:
    lst = xr.DataArray(
        np.full((1, 2, 2), 280.0, dtype="float32"),
        dims=("time", "y", "x"),
        coords={"time": [pd.Timestamp("2019-06-01")]},
    )
    qc = xr.DataArray(np.zeros((1, 2, 2), dtype="uint8"), dims=("time", "y", "x"), coords={"time": lst.time})
    return xr.Dataset({"lst": lst, "qc": qc})


def test_execute_fetch_writes_output_and_clears_any_prior_failure_status(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path)
    monkeypatch.setattr(source, "_search_items", lambda tile, year: ["fake-item"])
    monkeypatch.setattr(source, "_load_tile_year", lambda items: _fake_dataset())

    targets = source.plan(PipelineStep.FETCH, TargetSelection())
    assert len(targets) == 1
    target = targets[0]

    ok = source.execute(target)
    assert ok is True
    source.close()

    import os

    assert os.path.exists(target.output_path)
    status_dir = source.output_root(PipelineStep.FETCH)
    assert statusfile.read(statusfile.status_path(status_dir, target.key)) is None


def test_execute_fetch_records_failure_status_on_no_stac_items(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path)
    monkeypatch.setattr(source, "_search_items", lambda tile, year: [])

    targets = source.plan(PipelineStep.FETCH, TargetSelection())
    target = targets[0]

    ok = source.execute(target)
    assert ok is False
    source.close()

    status_dir = source.output_root(PipelineStep.FETCH)
    status = statusfile.read(statusfile.status_path(status_dir, target.key))
    assert status is not None
    assert status["status"] == "retrying"
    assert status["attempts"] == 1


def test_execute_fetch_retried_after_failure_can_succeed(tmp_path, monkeypatch):
    """Partial-download resumability: a tile-year that failed once (e.g. a
    transient STAC search error) is retried on the next `data run` call
    and, on success, ends up complete -- not stuck failed forever."""
    source, ctx = _make_source(tmp_path)
    monkeypatch.setattr(source, "_search_items", lambda tile, year: [])

    targets = source.plan(PipelineStep.FETCH, TargetSelection())
    target = targets[0]
    assert source.execute(target) is False
    source.close()

    source2, _ = _make_source(tmp_path)
    monkeypatch.setattr(source2, "_search_items", lambda tile, year: ["fake-item"])
    monkeypatch.setattr(source2, "_load_tile_year", lambda items: _fake_dataset())
    targets2 = source2.plan(PipelineStep.FETCH, TargetSelection())
    assert source2.execute(targets2[0]) is True
    source2.close()

    import os

    assert os.path.exists(target.output_path)
    status_dir = source.output_root(PipelineStep.FETCH)
    # Cleared by the successful retry, not left stuck 'retrying'.
    assert statusfile.read(statusfile.status_path(status_dir, target.key)) is None

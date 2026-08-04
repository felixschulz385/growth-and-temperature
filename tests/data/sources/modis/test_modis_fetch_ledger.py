"""ModisSource's FETCH step tracks each (year, tile) unit's local state in
its own DuckDB ledger (docs/design/10-fetch-ledger.md), the same generic
`artifacts` table PREPARE/GRID units use -- MODIS has no crawl catalog to
seed it from (see module docstring), so `_execute_fetch()` calls
`ensure_artifact`/`set_local_state` directly instead.
"""

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401 -- registers the .rio accessor _write_annual_geotiff needs
import xarray as xr

from src.data.common.ledger.paths import ledger_path
from src.data.common.ledger.store import SourceLedger
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


def test_execute_fetch_records_complete_in_ledger(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path)
    monkeypatch.setattr(source, "_search_items", lambda tile, year: ["fake-item"])
    monkeypatch.setattr(source, "_load_tile_year", lambda items: _fake_dataset())

    targets = source.plan(PipelineStep.FETCH, TargetSelection())
    assert len(targets) == 1
    target = targets[0]

    ok = source.execute(target)
    assert ok is True
    source.close()

    with SourceLedger.open(
        ledger_path(ctx.local_index_dir, source.data_path), data_path=source.data_path, read_only=True
    ) as ledger:
        assert ledger.local_state("fetch", target.key) == "complete"


def test_execute_fetch_records_failed_in_ledger_on_no_stac_items(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path)
    monkeypatch.setattr(source, "_search_items", lambda tile, year: [])

    targets = source.plan(PipelineStep.FETCH, TargetSelection())
    target = targets[0]

    ok = source.execute(target)
    assert ok is False
    source.close()

    with SourceLedger.open(
        ledger_path(ctx.local_index_dir, source.data_path), data_path=source.data_path, read_only=True
    ) as ledger:
        assert ledger.local_state("fetch", target.key) == "failed"


def test_execute_fetch_retried_after_failure_can_succeed(tmp_path, monkeypatch):
    """Partial-download resumability: a tile-year that failed once (e.g. a
    transient STAC search error) is retried on the next `pipeline run` call
    and, on success, ends up `complete` -- not stuck `failed` forever."""
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

    with SourceLedger.open(
        ledger_path(ctx.local_index_dir, source.data_path), data_path=source.data_path, read_only=True
    ) as ledger:
        assert ledger.local_state("fetch", target.key) == "complete"

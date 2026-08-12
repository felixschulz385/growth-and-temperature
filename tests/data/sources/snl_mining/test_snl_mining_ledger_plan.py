"""SnlMiningSource.plan() reads a reconciled ledger instead of falling back
to live discovery -- the static/singleton-target counterpart to
tests/data/sources/misc/test_gadm_ledger_plan.py.

GRID is the interesting case here: live discovery (_discover_grid) opens
prepared_db_path directly and runs `SELECT DISTINCT year FROM active_mines`
to compute meta["years"]; the ledger-backed fast path must read that back
from the ledger's persisted meta instead of touching DuckDB at all.
"""

import os

import duckdb

from src.data.common.ledger.paths import ledger_path
from src.data.common.ledger.store import SourceLedger
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.reconcile import reconcile_step
from src.data.sources.snl_mining.source import SnlMiningSource
from src.data.sources.steps import PipelineStep, TargetSelection


def _make(tmp_path, grid_id="legacy_4326", layout="legacy"):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"),
        local_index_dir=str(tmp_path / "index"),
        grid_id=grid_id,
        layout=layout,
    )
    cfg = SourceConfig.from_dict("snl_mining", {})
    return SnlMiningSource(ctx, cfg), ctx


def _write_fake_stage0_and_prices(source):
    os.makedirs(os.path.dirname(source.duckdb_path), exist_ok=True)
    open(source.duckdb_path, "w").close()
    os.makedirs(os.path.dirname(source.commodity_prices_path), exist_ok=True)
    open(source.commodity_prices_path, "w").close()


def _write_fake_prepared_db_with_active_mines(source, years):
    os.makedirs(os.path.dirname(source.prepared_db_path), exist_ok=True)
    con = duckdb.connect(source.prepared_db_path)
    con.execute("CREATE TABLE active_mines (property_id VARCHAR, year INTEGER)")
    for year in years:
        con.execute("INSERT INTO active_mines VALUES (?, ?)", ["m1", year])
    con.close()


def test_plan_prepare_reads_from_reconciled_ledger(tmp_path):
    source, ctx = _make(tmp_path)
    _write_fake_stage0_and_prices(source)

    local_ledger_path = ledger_path(ctx.local_index_dir, source.data_path)
    with SourceLedger.open(local_ledger_path, data_path=source.data_path) as ledger:
        reconcile_step(source, PipelineStep.PREPARE, ledger)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    assert targets[0].output_path == source.prepared_db_path
    assert targets[0].inputs == (source.duckdb_path, source.commodity_prices_path)


def test_plan_grid_reads_years_from_ledger_meta_without_duckdb_query(tmp_path, monkeypatch):
    source, ctx = _make(tmp_path)
    _write_fake_prepared_db_with_active_mines(source, [2019, 2020])

    local_ledger_path = ledger_path(ctx.local_index_dir, source.data_path)
    with SourceLedger.open(local_ledger_path, data_path=source.data_path) as ledger:
        reconcile_step(source, PipelineStep.GRID, ledger)

    # Ledger is now populated -- assert the fast path doesn't touch DuckDB at
    # all by making any further _connect_duckdb call blow up.
    def _boom(*args, **kwargs):
        raise AssertionError("ledger-backed _plan_grid must not open DuckDB")

    monkeypatch.setattr(source, "_connect_duckdb", _boom)

    targets = source.plan(PipelineStep.GRID, TargetSelection())
    assert len(targets) == 1
    assert targets[0].meta["years"] == [2019, 2020]
    assert targets[0].inputs == (source.prepared_db_path,)
    assert os.path.basename(targets[0].output_path) == source.output_filename


def test_plan_falls_back_to_discovery_when_ledger_unpopulated(tmp_path):
    source, _ = _make(tmp_path)
    _write_fake_prepared_db_with_active_mines(source, [2021])
    # local_index_dir is configured but nothing has been reconciled yet.
    targets = source.plan(PipelineStep.GRID, TargetSelection())
    assert len(targets) == 1
    assert targets[0].meta["years"] == [2021]

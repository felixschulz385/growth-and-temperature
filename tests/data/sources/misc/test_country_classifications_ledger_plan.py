"""CountryClassificationsSource.plan() reads a reconciled ledger instead of
falling back to live discovery -- the static/singleton-target counterpart to
tests/data/sources/misc/test_gadm_ledger_plan.py.
"""

import os

from src.data.common.ledger.store import SourceLedger
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.reconcile import reconcile_step
from src.data.sources.steps import PipelineStep, TargetSelection


def _make(tmp_path):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), layout="legacy"
    )
    cfg = SourceConfig.from_dict("country_classifications", {})
    cls = registry.load("country_classifications")
    return cls(ctx, cfg), ctx


def _write_fake_hdi_and_wb(source):
    hdi_file, wb_file = source._raw_file("hdi"), source._raw_file("worldbank")
    os.makedirs(os.path.dirname(hdi_file), exist_ok=True)
    open(hdi_file, "w").close()
    open(wb_file, "w").close()
    return hdi_file, wb_file


def _write_fake_prepare_and_gadm(source, ctx):
    classifications_parquet = os.path.join(
        source.output_root(PipelineStep.PREPARE), "classifications.parquet"
    )
    os.makedirs(os.path.dirname(classifications_parquet), exist_ok=True)
    open(classifications_parquet, "w").close()

    gadm_zarr = os.path.join(ctx.data_root, "misc", "processed", "stage_2", "gadm", "countries_grid.zarr")
    os.makedirs(gadm_zarr, exist_ok=True)
    return classifications_parquet, gadm_zarr


def test_plan_prepare_reads_from_reconciled_ledger(tmp_path):
    source, ctx = _make(tmp_path)
    hdi_file, wb_file = _write_fake_hdi_and_wb(source)

    local_ledger_path = os.path.join(ctx.local_index_dir, "misc_country_classifications.duckdb")
    with SourceLedger.open(local_ledger_path, data_path="misc/country_classifications") as ledger:
        reconcile_step(source, PipelineStep.PREPARE, ledger)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    assert targets[0].key == "country_classifications"
    assert set(targets[0].inputs) == {hdi_file, wb_file}


def test_plan_grid_reads_paths_from_ledger_meta(tmp_path):
    source, ctx = _make(tmp_path)
    classifications_parquet, gadm_zarr = _write_fake_prepare_and_gadm(source, ctx)

    local_ledger_path = os.path.join(ctx.local_index_dir, "misc_country_classifications.duckdb")
    with SourceLedger.open(local_ledger_path, data_path="misc/country_classifications") as ledger:
        reconcile_step(source, PipelineStep.GRID, ledger)

    targets = source.plan(PipelineStep.GRID, TargetSelection())
    assert len(targets) == 1
    assert targets[0].inputs == (classifications_parquet, gadm_zarr)


def test_plan_falls_back_to_discovery_when_ledger_unpopulated(tmp_path):
    source, _ = _make(tmp_path)
    _write_fake_hdi_and_wb(source)
    # local_index_dir is configured but nothing has been reconciled yet.
    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    assert targets[0].key == "country_classifications"

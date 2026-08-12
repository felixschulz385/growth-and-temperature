"""GadmSource.plan() reads a reconciled ledger instead of falling back to
live discovery -- the static/singleton-target counterpart to
tests/data/sources/acag/test_acag_plan.py's per-year ledger-backed tests.
"""

import os
import zipfile

import geopandas as gpd
from shapely.geometry import Point

from src.data.common.ledger.store import SourceLedger
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.reconcile import reconcile_step
from src.data.sources.steps import PipelineStep, TargetSelection, mark_complete


def _make(tmp_path):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), layout="legacy"
    )
    cfg = SourceConfig.from_dict("gadm", {})
    cls = registry.load("gadm")
    return cls(ctx, cfg), ctx


def _write_fake_raw_zip(source, tmp_path):
    raw_file = source._raw_file_path()
    os.makedirs(os.path.dirname(raw_file), exist_ok=True)
    gpkg_path = tmp_path / "gadm_410.gpkg"
    gpd.GeoDataFrame({"GID_0": ["AAA"]}, geometry=[Point(0, 0)], crs="EPSG:4326").to_file(
        gpkg_path, driver="GPKG", layer="ADM_0"
    )
    with zipfile.ZipFile(raw_file, "w") as zf:
        zf.write(gpkg_path, arcname="gadm_410.gpkg")
    return raw_file


def _write_fake_level_files(source):
    vector_dir = source.output_root(PipelineStep.PREPARE)
    os.makedirs(vector_dir, exist_ok=True)
    level0 = os.path.join(vector_dir, "gadm_levelADM_0_simplified.gpkg")
    level1 = os.path.join(vector_dir, "gadm_levelADM_1_simplified.gpkg")
    open(level0, "w").close()
    open(level1, "w").close()
    mark_complete(vector_dir)
    return [level0, level1]


def test_plan_prepare_reads_from_reconciled_ledger(tmp_path):
    source, ctx = _make(tmp_path)
    raw_file = _write_fake_raw_zip(source, tmp_path)

    local_ledger_path = os.path.join(ctx.local_index_dir, "misc_gadm.duckdb")
    with SourceLedger.open(local_ledger_path, data_path="misc/gadm") as ledger:
        reconcile_step(source, PipelineStep.PREPARE, ledger)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    assert targets[0].key == "gadm"
    assert targets[0].inputs == (raw_file,)
    assert targets[0].output_path == source.output_root(PipelineStep.PREPARE)


def test_plan_grid_reads_level_files_from_ledger_meta(tmp_path):
    source, ctx = _make(tmp_path)
    level_files = _write_fake_level_files(source)

    local_ledger_path = os.path.join(ctx.local_index_dir, "misc_gadm.duckdb")
    with SourceLedger.open(local_ledger_path, data_path="misc/gadm") as ledger:
        reconcile_step(source, PipelineStep.GRID, ledger)

    targets = source.plan(PipelineStep.GRID, TargetSelection())
    assert len(targets) == 1
    assert list(targets[0].inputs) == level_files
    assert targets[0].output_path == source._grid_output_path()


def test_plan_falls_back_to_discovery_when_ledger_unpopulated(tmp_path):
    source, _ = _make(tmp_path)
    _write_fake_raw_zip(source, tmp_path)
    # local_index_dir is configured but nothing has been reconciled yet.
    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    assert targets[0].key == "gadm"

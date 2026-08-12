"""EcoregionsSource.plan() reads a reconciled ledger instead of falling back
to live discovery -- the static/singleton-target counterpart to
tests/data/sources/misc/test_gadm_ledger_plan.py. GRID's two possible
targets (`ecoregions_grid`, `gadm_gid3_dominant`) both come back from one
`_plan_from_ledger` call.
"""

import json
import os

import geopandas as gpd
from shapely.geometry import Point

from src.data.common.ledger.paths import ledger_path
from src.data.common.ledger.store import SourceLedger
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.misc.gadm import gid_mapping_path
from src.data.sources.reconcile import reconcile_step
from src.data.sources.steps import PipelineStep, TargetSelection, mark_complete


def _make(tmp_path, layout_mode="legacy"):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), layout=layout_mode)
    cfg = SourceConfig.from_dict("ecoregions", {"url": "https://example.test/eco.zip", "name": "eco.zip"})
    cls = registry.load("ecoregions")
    return cls(ctx, cfg), ctx


def _write_fake_raw_file(source):
    raw_file = source._raw_file_path()
    os.makedirs(os.path.dirname(raw_file), exist_ok=True)
    open(raw_file, "w").close()
    return raw_file


def _write_vector_file(source):
    vector_file = os.path.join(source.output_root(PipelineStep.PREPARE), "ecoregions_simplified.gpkg")
    os.makedirs(os.path.dirname(vector_file), exist_ok=True)
    gpd.GeoDataFrame(
        {"REALM": ["Nearctic"], "BIOME_NUM": [1], "BIOME_NAME": ["B"], "ECO_ID": [101], "ECO_NAME": ["E"]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    ).to_file(vector_file, driver="GPKG")
    mark_complete(source.output_root(PipelineStep.PREPARE))
    return vector_file


def _write_gadm_gid3_artifacts(ctx):
    gadm_prepare_dir = layout.output_root(ctx.data_root, "misc", PipelineStep.PREPARE, namespace="gadm", layout=ctx.layout)
    os.makedirs(gadm_prepare_dir, exist_ok=True)
    gadm_gid3_file = os.path.join(gadm_prepare_dir, "gadm_levelADM_3_simplified.gpkg")
    gpd.GeoDataFrame({"GID_3": ["AAA.1.1_1"]}, geometry=[Point(0, 0)], crs="EPSG:4326").to_file(
        gadm_gid3_file, driver="GPKG"
    )

    mapping_path = gid_mapping_path(ctx.data_root, ctx.grid_id, ctx.layout, "GID_3")
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    with open(mapping_path, "w") as f:
        json.dump({"AAA.1.1_1": 1}, f)
    return gadm_gid3_file, mapping_path


def test_plan_prepare_reads_from_reconciled_ledger(tmp_path):
    source, ctx = _make(tmp_path)
    raw_file = _write_fake_raw_file(source)

    local_ledger_path = ledger_path(ctx.local_index_dir, source.data_path)
    with SourceLedger.open(local_ledger_path, data_path=source.data_path) as ledger:
        reconcile_step(source, PipelineStep.PREPARE, ledger)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    assert targets[0].key == "ecoregions"
    assert targets[0].inputs == (raw_file,)


def test_plan_grid_reads_both_targets_from_ledger(tmp_path):
    source, ctx = _make(tmp_path)
    vector_file = _write_vector_file(source)
    gadm_gid3_file, gadm_gid3_mapping = _write_gadm_gid3_artifacts(ctx)

    local_ledger_path = ledger_path(ctx.local_index_dir, source.data_path)
    with SourceLedger.open(local_ledger_path, data_path=source.data_path) as ledger:
        reconcile_step(source, PipelineStep.GRID, ledger)

    targets = source.plan(PipelineStep.GRID, TargetSelection())
    assert {t.key for t in targets} == {"ecoregions_grid", "gadm_gid3_dominant"}

    eco_target = next(t for t in targets if t.key == "ecoregions_grid")
    assert eco_target.inputs == (vector_file,)

    dominant_target = next(t for t in targets if t.key == "gadm_gid3_dominant")
    assert dominant_target.inputs == (vector_file, gadm_gid3_file, gadm_gid3_mapping)


def test_plan_falls_back_to_discovery_when_ledger_unpopulated(tmp_path):
    source, _ = _make(tmp_path)
    _write_fake_raw_file(source)
    # local_index_dir is configured but nothing has been reconciled yet.
    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    assert targets[0].key == "ecoregions"

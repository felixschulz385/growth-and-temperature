"""EcoregionsSource.plan() shape: FETCH follows the
ConfiguredFilesFetchMixin/NEVER pattern shared with gadm/osm/
country_classifications; PREPARE always emits an `ecoregions_grid` target
once the raw file exists (Plan 2's PREPARE+GRID merge, docs/design successor
to the ledger) and conditionally emits a second `gadm_gid3_dominant` target
only once GADM's own PREPARE artifacts (REQUIRES) are actually present on
disk."""

import json
import os

import geopandas as gpd
from shapely.geometry import Point

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.misc.gadm import gid_mapping_path
from src.data.sources.steps import Completion, PipelineStep, TargetSelection


def _make(tmp_path, layout_mode="legacy"):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), layout=layout_mode)
    cfg = SourceConfig.from_dict("ecoregions", {"url": "https://example.test/eco.zip", "name": "eco.zip"})
    cls = registry.load("ecoregions")
    return cls(ctx, cfg), ctx


def _write_raw_file(source):
    raw_file = source._raw_file_path()
    os.makedirs(os.path.dirname(raw_file), exist_ok=True)
    open(raw_file, "w").close()
    return raw_file


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


def test_plan_fetch_target_shape(tmp_path):
    source, _ = _make(tmp_path)
    targets = source.plan(PipelineStep.FETCH, TargetSelection())
    assert len(targets) == 1
    assert targets[0].completion == Completion.NEVER
    assert targets[0].key == "all"


def test_plan_prepare_empty_without_raw_file(tmp_path):
    source, _ = _make(tmp_path)
    assert source.plan(PipelineStep.PREPARE, TargetSelection()) == []


def test_plan_prepare_only_ecoregions_grid_target_without_gadm_artifacts(tmp_path):
    source, _ = _make(tmp_path)
    _write_raw_file(source)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert [t.key for t in targets] == ["ecoregions_grid"]
    assert targets[0].completion == Completion.MARKER


def test_plan_prepare_adds_gid3_dominant_target_once_gadm_artifacts_exist(tmp_path):
    source, ctx = _make(tmp_path)
    _write_raw_file(source)
    _write_gadm_gid3_artifacts(ctx)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert [t.key for t in targets] == ["ecoregions_grid", "gadm_gid3_dominant"]

    dominant_target = targets[1]
    assert dominant_target.completion == Completion.PATH_EXISTS
    assert dominant_target.output_path.endswith("dominant_biome_by_gid3.parquet")
    assert len(dominant_target.inputs) == 3


def test_registry_requires_gadm_prepare():
    # gadm's PREPARE now does what used to be its separate GRID step
    # directly (Plan 2's PREPARE+GRID merge) -- PipelineStep.GRID no longer
    # exists anywhere. Scoped to ecoregions' own (merged) PREPARE step --
    # only _plan_prepare()'s gadm_gid3_dominant target touches gadm.
    spec = registry.resolve("ecoregions")
    assert spec.requires == ((PipelineStep.PREPARE, "gadm", PipelineStep.PREPARE),)
    assert spec.requires_for(PipelineStep.FETCH) == ()
    assert spec.requires_for(PipelineStep.PREPARE) == (("gadm", PipelineStep.PREPARE),)

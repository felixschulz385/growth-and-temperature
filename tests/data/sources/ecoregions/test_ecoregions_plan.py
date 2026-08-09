"""EcoregionsSource.plan() shape: FETCH/PREPARE targets follow the
ConfiguredFilesFetchMixin/PATH_EXISTS pattern shared with gadm/osm/
country_classifications; GRID conditionally emits a second
`gadm_gid3_dominant` target only once GADM's own PREPARE+GRID artifacts
(REQUIRES) are actually present on disk."""

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


def _write_vector_file(source):
    vector_file = os.path.join(source.output_root(PipelineStep.PREPARE), "ecoregions_simplified.gpkg")
    os.makedirs(os.path.dirname(vector_file), exist_ok=True)
    gpd.GeoDataFrame(
        {"REALM": ["Nearctic"], "BIOME_NUM": [1], "BIOME_NAME": ["B"], "ECO_ID": [101], "ECO_NAME": ["E"]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    ).to_file(vector_file, driver="GPKG")
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


def test_plan_fetch_target_shape(tmp_path):
    source, _ = _make(tmp_path)
    targets = source.plan(PipelineStep.FETCH, TargetSelection())
    assert len(targets) == 1
    assert targets[0].completion == Completion.NEVER
    assert targets[0].key == "all"


def test_plan_prepare_empty_without_raw_file(tmp_path):
    source, _ = _make(tmp_path)
    assert source.plan(PipelineStep.PREPARE, TargetSelection()) == []


def test_plan_grid_empty_without_prepare_output(tmp_path):
    source, _ = _make(tmp_path)
    assert source.plan(PipelineStep.GRID, TargetSelection()) == []


def test_plan_grid_only_ecoregions_target_without_gadm_artifacts(tmp_path):
    source, _ = _make(tmp_path)
    _write_vector_file(source)

    targets = source.plan(PipelineStep.GRID, TargetSelection())
    assert [t.key for t in targets] == ["ecoregions_grid"]


def test_plan_grid_adds_gid3_dominant_target_once_gadm_artifacts_exist(tmp_path):
    source, ctx = _make(tmp_path)
    _write_vector_file(source)
    _write_gadm_gid3_artifacts(ctx)

    targets = source.plan(PipelineStep.GRID, TargetSelection())
    assert [t.key for t in targets] == ["ecoregions_grid", "gadm_gid3_dominant"]

    dominant_target = targets[1]
    assert dominant_target.completion == Completion.PATH_EXISTS
    assert dominant_target.output_path.endswith("dominant_biome_by_gid3.parquet")
    assert len(dominant_target.inputs) == 3


def test_registry_requires_gadm_prepare_and_grid():
    spec = registry.resolve("ecoregions")
    assert ("gadm", PipelineStep.PREPARE) in spec.requires
    assert ("gadm", PipelineStep.GRID) in spec.requires

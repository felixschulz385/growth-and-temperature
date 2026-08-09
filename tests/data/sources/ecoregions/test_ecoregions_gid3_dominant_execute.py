"""EcoregionsSource._execute_gid3_dominant(): end-to-end wiring from two
prepared .gpkg files + GADM's GID_3 mapping json to the small
GID_3-keyed dominant-biome parquet -- mirrors
tests/data/sources/misc/test_country_classifications_execute.py's own
GID_0-keyed-table wiring test, the closest existing precedent."""

import json
import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import src.data.common.geobox as geobox_module
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.steps import Completion, PipelineStep, StepTarget


class _FakeGeobox:
    crs = "EPSG:4326"


def _make(tmp_path, monkeypatch):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("ecoregions", {"url": "https://example.test/eco.zip", "name": "eco.zip"})
    cls = registry.load("ecoregions")
    source = cls(ctx, cfg)
    monkeypatch.setattr(geobox_module, "get_target_geobox", lambda passed_ctx: _FakeGeobox())
    return source, ctx


def test_execute_gid3_dominant_writes_gid3_keyed_parquet(tmp_path, monkeypatch):
    source, ctx = _make(tmp_path, monkeypatch)

    ecoregions_file = str(tmp_path / "ecoregions_simplified.gpkg")
    gpd.GeoDataFrame(
        {
            "REALM": ["Nearctic", "Nearctic"],
            "BIOME_NUM": [1, 2],
            "BIOME_NAME": ["Biome One", "Biome Two"],
            "ECO_ID": [101, 102],
            "ECO_NAME": ["Eco One", "Eco Two"],
        },
        geometry=[box(0, 0, 10, 6), box(0, 6, 10, 10)],
        crs="EPSG:4326",
    ).to_file(ecoregions_file, driver="GPKG")

    gadm_gid3_file = str(tmp_path / "gadm_levelADM_3_simplified.gpkg")
    gpd.GeoDataFrame({"GID_3": ["AAA.1.1_1"]}, geometry=[box(0, 0, 10, 10)], crs="EPSG:4326").to_file(
        gadm_gid3_file, driver="GPKG"
    )

    mapping_file = str(tmp_path / "GID_3_code_mapping.json")
    with open(mapping_file, "w") as f:
        json.dump({"AAA.1.1_1": 5}, f)

    target = StepTarget(
        source_id=source.ID,
        step=PipelineStep.GRID,
        key="gadm_gid3_dominant",
        output_path=str(tmp_path / "out" / "dominant_biome_by_gid3.parquet"),
        inputs=(ecoregions_file, gadm_gid3_file, mapping_file),
        completion=Completion.PATH_EXISTS,
    )

    assert source.execute(target) is True
    result = pd.read_parquet(target.output_path)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["GID_3"] == 5
    assert row["GID_3_code"] == "AAA.1.1_1"
    assert row["dominant_biome_num"] == 1
    assert row["dominant_biome_name"] == "Biome One"
    assert row["biome_area_frac"] == 0.6


def test_execute_gid3_dominant_is_idempotent(tmp_path, monkeypatch):
    source, ctx = _make(tmp_path, monkeypatch)

    ecoregions_file = str(tmp_path / "ecoregions_simplified.gpkg")
    gpd.GeoDataFrame(
        {"REALM": ["Nearctic"], "BIOME_NUM": [1], "BIOME_NAME": ["B"], "ECO_ID": [101], "ECO_NAME": ["E"]},
        geometry=[box(0, 0, 10, 10)],
        crs="EPSG:4326",
    ).to_file(ecoregions_file, driver="GPKG")

    gadm_gid3_file = str(tmp_path / "gadm_levelADM_3_simplified.gpkg")
    gpd.GeoDataFrame({"GID_3": ["AAA.1.1_1"]}, geometry=[box(0, 0, 10, 10)], crs="EPSG:4326").to_file(
        gadm_gid3_file, driver="GPKG"
    )

    mapping_file = str(tmp_path / "GID_3_code_mapping.json")
    with open(mapping_file, "w") as f:
        json.dump({"AAA.1.1_1": 5}, f)

    target = StepTarget(
        source_id=source.ID,
        step=PipelineStep.GRID,
        key="gadm_gid3_dominant",
        output_path=str(tmp_path / "out" / "dominant_biome_by_gid3.parquet"),
        inputs=(ecoregions_file, gadm_gid3_file, mapping_file),
        completion=Completion.PATH_EXISTS,
    )
    assert source.execute(target) is True
    assert source.execute(target) is True

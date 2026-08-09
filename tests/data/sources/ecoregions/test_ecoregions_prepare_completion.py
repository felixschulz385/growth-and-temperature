"""PREPARE produces one fixed-name file (`ecoregions_simplified.gpkg`) --
unlike GADM's variable-count-of-per-level-files directory, this uses
`Completion.PATH_EXISTS` (matching `country_classifications`'s single
`classifications.parquet` PREPARE output), not `Completion.MARKER`."""

import os
import zipfile

import geopandas as gpd
from shapely.geometry import Point

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.ecoregions.source import DEFAULT_NAME, DEFAULT_URL
from src.data.sources.steps import Completion, PipelineStep, TargetSelection


def _make(tmp_path, **raw):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("ecoregions", {"url": "https://example.test/eco.zip", "name": "eco.zip", **raw})
    cls = registry.load("ecoregions")
    return cls(ctx, cfg), ctx


def _write_fake_raw_zip(source, tmp_path, columns=None):
    raw_file = source._raw_file_path()
    os.makedirs(os.path.dirname(raw_file), exist_ok=True)
    gpkg_path = tmp_path / "ecoregions_raw.gpkg"
    attrs = columns or {"REALM": ["Nearctic"], "BIOME_NUM": [1], "BIOME_NAME": ["Tropical"], "ECO_ID": [101], "ECO_NAME": ["Test eco"]}
    gpd.GeoDataFrame(attrs, geometry=[Point(0, 0)], crs="EPSG:4326").to_file(gpkg_path, driver="GPKG")
    with zipfile.ZipFile(raw_file, "w") as zf:
        zf.write(gpkg_path, arcname="ecoregions_raw.gpkg")
    return raw_file


def test_constructor_defaults_to_the_resolve_arcgis_query_url(tmp_path):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("ecoregions", {})
    cls = registry.load("ecoregions")
    source = cls(ctx, cfg)
    assert source.CONFIGURED_FILES[0].url == DEFAULT_URL
    assert source.CONFIGURED_FILES[0].name == DEFAULT_NAME


def test_constructor_config_overrides_default_url_and_name(tmp_path):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("ecoregions", {"url": "https://example.test/override.zip", "name": "override.zip"})
    source = registry.load("ecoregions")(ctx, cfg)
    assert source.CONFIGURED_FILES[0].url == "https://example.test/override.zip"
    assert source.CONFIGURED_FILES[0].name == "override.zip"


def test_plan_prepare_target_uses_path_exists_completion(tmp_path):
    source, _ = _make(tmp_path)
    _write_fake_raw_zip(source, tmp_path)

    target = source.plan(PipelineStep.PREPARE, TargetSelection())[0]
    assert target.completion == Completion.PATH_EXISTS


def test_execute_prepare_writes_simplified_gpkg_with_expected_columns(tmp_path):
    source, _ = _make(tmp_path)
    _write_fake_raw_zip(source, tmp_path)

    target = source.plan(PipelineStep.PREPARE, TargetSelection())[0]
    assert source.execute(target) is True
    assert os.path.exists(target.output_path)

    gdf = gpd.read_file(target.output_path, engine="pyogrio")
    for col in ("REALM", "BIOME_NUM", "BIOME_NAME", "ECO_ID", "ECO_NAME"):
        assert col in gdf.columns


def test_execute_prepare_is_idempotent(tmp_path):
    source, _ = _make(tmp_path)
    _write_fake_raw_zip(source, tmp_path)

    target = source.plan(PipelineStep.PREPARE, TargetSelection())[0]
    assert source.execute(target) is True
    assert source.execute(target) is True


def test_execute_prepare_fails_loudly_on_missing_columns(tmp_path):
    source, _ = _make(tmp_path)
    _write_fake_raw_zip(source, tmp_path, columns={"NOT_REALM": ["x"]})

    target = source.plan(PipelineStep.PREPARE, TargetSelection())[0]
    assert source.execute(target) is False
    assert not os.path.exists(target.output_path)

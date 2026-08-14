"""PREPARE's phase 1 (`_ensure_vector_file`) produces one fixed-name
simplified vector file (`ecoregions_simplified.gpkg`), shared by both PREPARE
targets. The `ecoregions_grid` PREPARE target itself uses `Completion.MARKER` (a
tiled zarr store) -- unlike `country_classifications`'s single-file
`Completion.PATH_EXISTS` PREPARE output, this source's PREPARE produces a
tiled raster, same rationale as gadm's own rasterized `Completion.MARKER`
GID_N grids. Full rasterization needs a real geobox/dask -- exercised
separately in test_ecoregions_grid_geobox.py -- so this file stubs at the
`_ensure_vector_file` phase-1 boundary, mirroring
tests/data/sources/misc/test_gadm_prepare_completion.py's identical
approach for gadm."""

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


def test_plan_prepare_target_uses_marker_completion(tmp_path):
    source, _ = _make(tmp_path)
    _write_fake_raw_zip(source, tmp_path)

    target = source.plan(PipelineStep.PREPARE, TargetSelection())[0]
    assert target.key == "ecoregions_grid"
    assert target.completion == Completion.MARKER


def test_ensure_vector_file_writes_simplified_gpkg_with_expected_columns(tmp_path):
    source, _ = _make(tmp_path)
    raw_file = _write_fake_raw_zip(source, tmp_path)

    vector_path = source._ensure_vector_file(raw_file)
    assert vector_path is not None
    assert os.path.exists(vector_path)

    gdf = gpd.read_file(vector_path, engine="pyogrio")
    for col in ("REALM", "BIOME_NUM", "BIOME_NAME", "ECO_ID", "ECO_NAME"):
        assert col in gdf.columns


def test_ensure_vector_file_is_idempotent(tmp_path):
    source, _ = _make(tmp_path)
    raw_file = _write_fake_raw_zip(source, tmp_path)

    assert source._ensure_vector_file(raw_file) is not None
    assert source._ensure_vector_file(raw_file) is not None


def test_ensure_vector_file_fails_loudly_on_missing_columns(tmp_path):
    source, _ = _make(tmp_path)
    raw_file = _write_fake_raw_zip(source, tmp_path, columns={"NOT_REALM": ["x"]})

    assert source._ensure_vector_file(raw_file) is None
    assert not os.path.exists(source._vector_path())

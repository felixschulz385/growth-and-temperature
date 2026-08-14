"""GadmSource: ledger-free FETCH/PREPARE (docs/design successor to the
ledger, Plan 2 PREPARE+GRID merge). plan() is a bare live `os.path.exists()`
check against the raw fetched file -- see
tests/data/sources/misc/test_osm_ledger_plan.py's identical shape.
"""

import os
import zipfile

import geopandas as gpd
from shapely.geometry import Point

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.steps import Completion, PipelineStep, TargetSelection


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


def test_steps_is_fetch_and_prepare_only():
    from src.data.sources.misc.gadm import GadmSource

    assert GadmSource.STEPS == (PipelineStep.FETCH, PipelineStep.PREPARE)


def test_prepare_plan_empty_when_raw_file_missing(tmp_path):
    source, _ = _make(tmp_path)
    assert source.plan(PipelineStep.PREPARE, TargetSelection()) == []


def test_prepare_plan_one_target_when_raw_file_present(tmp_path):
    source, _ = _make(tmp_path)
    raw_file = _write_fake_raw_zip(source, tmp_path)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    target = targets[0]
    assert target.key == "gadm"
    assert target.inputs == (raw_file,)
    assert target.completion == Completion.MARKER
    assert target.output_path == source._grid_output_path()

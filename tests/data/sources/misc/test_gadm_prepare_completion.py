"""Regression test: `pipeline plan --source gadm --step prepare` used to
always report `[pending]` regardless of on-disk state, because PREPARE's
StepTarget used `completion=Completion.NEVER` (which `is_complete()` always
reports False for) while the real skip-check lived only inside
`_execute_prepare`'s custom `os.listdir` logic, invisible to `is_complete()`/
`pipeline plan`. PREPARE now uses `Completion.MARKER` (a `.complete` sibling
file next to the output directory), matching how GRID already reports its
own completion in this same file."""

import os
import zipfile

import geopandas as gpd
from shapely.geometry import Point

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.steps import Completion, PipelineStep, TargetSelection, is_complete, marker_path


def _make(tmp_path, layout="legacy"):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), layout=layout
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


def test_plan_prepare_target_uses_marker_completion(tmp_path):
    source, _ = _make(tmp_path)
    _write_fake_raw_zip(source, tmp_path)

    target = source.plan(PipelineStep.PREPARE, TargetSelection())[0]
    assert target.completion == Completion.MARKER


def test_plan_prepare_is_pending_before_execute_and_complete_after(tmp_path):
    source, _ = _make(tmp_path)
    _write_fake_raw_zip(source, tmp_path)

    target = source.plan(PipelineStep.PREPARE, TargetSelection())[0]
    assert is_complete(target) is False

    assert source.execute(target) is True
    assert os.path.exists(marker_path(target.output_path))
    assert is_complete(target) is True

    # Re-running plan() after completion must keep reporting complete.
    target_again = source.plan(PipelineStep.PREPARE, TargetSelection())[0]
    assert is_complete(target_again) is True


def test_execute_prepare_is_idempotent_via_marker(tmp_path):
    source, _ = _make(tmp_path)
    _write_fake_raw_zip(source, tmp_path)

    target = source.plan(PipelineStep.PREPARE, TargetSelection())[0]
    assert source.execute(target) is True
    # Second execute hits the MARKER-based skip path, not a re-extraction.
    assert source.execute(target) is True


def test_execute_prepare_legacy_fallback_when_marker_missing_but_files_exist(tmp_path):
    # Pre-existing runs from before the MARKER policy was added: level files
    # exist on disk but no .complete marker yet.
    source, _ = _make(tmp_path)
    raw_file = _write_fake_raw_zip(source, tmp_path)

    output_base = source.output_root(PipelineStep.PREPARE)
    os.makedirs(output_base, exist_ok=True)
    open(os.path.join(output_base, "gadm_levelADM_0_simplified.gpkg"), "w").close()

    target = source.plan(PipelineStep.PREPARE, TargetSelection())[0]
    assert is_complete(target) is False  # no marker yet

    # Corrupt the raw zip so a real re-extraction would fail loudly --
    # proves the legacy fallback path is what actually skipped the work.
    os.remove(raw_file)
    open(raw_file, "w").close()

    assert source.execute(target) is True
    assert is_complete(target) is True

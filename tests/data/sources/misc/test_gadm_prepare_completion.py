"""Regression test: `data plan --source gadm --step prepare` used to
always report `[pending]` regardless of on-disk state, because PREPARE's
StepTarget used `completion=Completion.NEVER` (which `is_complete()` always
reports False for) while the real skip-check lived only inside
`_execute_prepare`'s custom `os.listdir` logic, invisible to `is_complete()`/
`data plan`. PREPARE now uses `Completion.MARKER` (a `.complete` sibling
file next to the final output zarr -- what used to be GRID's own output,
Plan 2's PREPARE+GRID merge, docs/design successor to the ledger).

`_rasterize_levels` (phase 2: tiled rasterization, needs a real target
geobox/dask client) is stubbed out in every test here -- these tests are
about resumability/marker semantics around the merged PREPARE step, not
rasterization correctness (covered by
tests/data/sources/misc/test_gadm_osm_grid_geobox.py instead).
"""

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


def _fake_rasterize_levels(level_files, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return True


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


def test_plan_prepare_is_pending_before_execute_and_complete_after(tmp_path, monkeypatch):
    source, _ = _make(tmp_path)
    _write_fake_raw_zip(source, tmp_path)
    monkeypatch.setattr(source, "_rasterize_levels", _fake_rasterize_levels)

    target = source.plan(PipelineStep.PREPARE, TargetSelection())[0]
    assert is_complete(target) is False

    assert source.execute(target) is True
    assert os.path.exists(marker_path(target.output_path))
    assert is_complete(target) is True

    # Re-running plan() after completion must keep reporting complete.
    target_again = source.plan(PipelineStep.PREPARE, TargetSelection())[0]
    assert is_complete(target_again) is True


def test_execute_prepare_is_idempotent_via_marker(tmp_path, monkeypatch):
    source, _ = _make(tmp_path)
    _write_fake_raw_zip(source, tmp_path)

    calls = []

    def fake_rasterize(level_files, output_path):
        calls.append(1)
        return _fake_rasterize_levels(level_files, output_path)

    monkeypatch.setattr(source, "_rasterize_levels", fake_rasterize)

    target = source.plan(PipelineStep.PREPARE, TargetSelection())[0]
    assert source.execute(target) is True
    # Second execute hits the MARKER-based skip path, not a re-run.
    assert source.execute(target) is True
    assert len(calls) == 1


def test_execute_prepare_reuses_existing_level_files_without_reextracting(tmp_path, monkeypatch):
    # Level files already on disk (from a prior partial run) -- phase 1
    # (vector extraction) must be skipped, not redone, even though the
    # overall target isn't marked complete yet (phase 2 never ran).
    source, _ = _make(tmp_path)
    raw_file = _write_fake_raw_zip(source, tmp_path)

    vector_dir = source._vector_dir()
    os.makedirs(vector_dir, exist_ok=True)
    open(os.path.join(vector_dir, "gadm_levelADM_0_simplified.gpkg"), "w").close()

    # Corrupt the raw zip so a real re-extraction would fail loudly --
    # proves phase 1 was actually skipped, not silently re-run.
    os.remove(raw_file)
    open(raw_file, "w").close()

    monkeypatch.setattr(source, "_rasterize_levels", _fake_rasterize_levels)

    target = source.plan(PipelineStep.PREPARE, TargetSelection())[0]
    assert source.execute(target) is True
    assert is_complete(target) is True

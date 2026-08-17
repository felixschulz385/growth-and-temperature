"""Tests for scripts/clean_prepare_scratch.py -- the per-year "annual zarr"
scratch cleanup tool. Exercises the find/delete logic against a fake
tmp_path tree; no real HPC/SLURM involved.
"""

import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from clean_prepare_scratch import (  # noqa: E402
    YEAR_ZARR_RE,
    candidate_prepare_dirs,
    dir_size,
    find_scratch_entries,
    human,
)
from migrate_legacy_layout import build_source, interim_output_root  # noqa: E402

from src.data.pipeline.config import SourceConfig  # noqa: E402
from src.data.pipeline.context import PipelineContext  # noqa: E402
from src.data.sources import layout  # noqa: E402
from src.data.sources.acag import AcagSource  # noqa: E402
from src.data.sources.steps import PipelineStep  # noqa: E402


def _acag_source(tmp_path):
    data_root = str(tmp_path / "data_root")
    cfg = SourceConfig.from_dict("acag", {"data_path": "acag/pm25"})
    return AcagSource(PipelineContext(data_root=data_root, grid_id="legacy_4326"), cfg)


# -- YEAR_ZARR_RE / find_scratch_entries() -----------------------------------


def test_year_zarr_regex_matches_bare_year_and_monthly_variant():
    assert YEAR_ZARR_RE.match("2000.zarr")
    assert YEAR_ZARR_RE.match("1994_monthly.zarr")
    assert not YEAR_ZARR_RE.match("modis_lst_21a2.zarr")
    assert not YEAR_ZARR_RE.match("pm25.zarr")
    assert not YEAR_ZARR_RE.match("20000.zarr")  # not a 4-digit year


def test_find_scratch_entries_ignores_non_matching_siblings(tmp_path):
    directory = tmp_path / "prepare"
    directory.mkdir()
    (directory / "2000.zarr").mkdir()
    (directory / "2001_monthly.zarr").mkdir()
    (directory / "pm25.zarr").mkdir()
    (directory / "notes.txt").write_text("keep me")

    found = find_scratch_entries(str(directory))
    names = sorted(os.path.basename(p) for p in found)
    assert names == ["2000.zarr", "2001_monthly.zarr"]


def test_find_scratch_entries_missing_directory_returns_empty(tmp_path):
    assert find_scratch_entries(str(tmp_path / "does-not-exist")) == []


# -- dir_size() / human() -----------------------------------------------------


def test_dir_size_sums_nested_files(tmp_path):
    d = tmp_path / "store.zarr"
    d.mkdir()
    (d / "a.bin").write_bytes(b"x" * 10)
    (d / "sub").mkdir()
    (d / "sub" / "b.bin").write_bytes(b"y" * 5)
    assert dir_size(str(d)) == 15


def test_human_formats_reasonable_units():
    assert human(500) == "500.0B"
    assert human(2048) == "2.0KB"


# -- candidate_prepare_dirs() -------------------------------------------------


def test_candidate_prepare_dirs_includes_all_three_layouts_for_acag(tmp_path):
    source = _acag_source(tmp_path)
    candidates = candidate_prepare_dirs("acag", source, source.ctx.data_root)
    assert set(candidates) == {"legacy", "interim", "current"}
    assert candidates["current"] == source.output_root(PipelineStep.PREPARE, agg=layout.CRS_AGG)


def test_candidate_prepare_dirs_skips_current_for_prepare_exception_sources(tmp_path):
    from src.data.sources.snl_mining.source import SnlMiningSource

    data_root = str(tmp_path / "data_root")
    cfg = SourceConfig.from_dict("snl_mining", {})
    source = SnlMiningSource(PipelineContext(data_root=data_root), cfg)

    candidates = candidate_prepare_dirs("snl_mining", source, data_root)
    assert "current" not in candidates
    assert set(candidates) == {"legacy", "interim"}


# -- end-to-end: dead leftover (acag) vs live scratch (glass_avhrr) ----------


def test_dead_leftover_scratch_is_found_but_real_family_store_is_not(tmp_path):
    """acag never writes a bare <year>.zarr under the current pipeline
    (pre-2d3bf1a dead leftover only) -- confirms the scanner finds it under
    the interim location without touching a real family.zarr store sharing
    the "current" directory tree."""
    source = _acag_source(tmp_path)
    interim_dir = interim_output_root(source.ctx.data_root, source.cfg.data_path, PipelineStep.PREPARE)
    os.makedirs(os.path.join(interim_dir, "2000.zarr"))

    real_store = source.output_root(PipelineStep.PREPARE, agg=layout.CRS_AGG)
    os.makedirs(os.path.join(real_store, "legacy_4326"))
    Path(real_store, "legacy_4326", "pm25.zarr").mkdir()

    candidates = candidate_prepare_dirs("acag", source, source.ctx.data_root)
    found_interim = find_scratch_entries(candidates["interim"])
    found_current = find_scratch_entries(candidates["current"])

    assert [os.path.basename(p) for p in found_interim] == ["2000.zarr"]
    assert found_current == []  # pm25.zarr must never match YEAR_ZARR_RE

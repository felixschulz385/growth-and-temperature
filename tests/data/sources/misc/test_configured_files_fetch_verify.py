"""ConfiguredFilesFetchMixin.verify_fetch() -- osm/gadm/country_classifications
fetch a small, fixed list of named files. This checks each is actually
present at its exact expected path, distinguishing "files fetched under the
wrong name" from "nothing fetched" -- both of which the generic disk-walk
FETCH summary ("N file(s) fetched") reports identically, silently causing
PREPARE to plan zero targets with no indication why (the bug this exists to
surface directly instead)."""

import os

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.steps import PipelineStep


def _make(tmp_path, source_id, **raw):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict(source_id, dict(raw))
    cls = registry.load(source_id)
    return cls(ctx, cfg), ctx


def test_verify_fetch_ok_when_all_expected_files_present(tmp_path):
    source, _ = _make(tmp_path, "osm")
    fetch_dir = source.output_root(PipelineStep.FETCH)
    os.makedirs(fetch_dir, exist_ok=True)
    open(os.path.join(fetch_dir, source.CONFIGURED_FILES[0].name), "w").close()

    result = source.verify_fetch()
    assert result.ok is True
    assert "1 expected file(s) present" in result.detail


def test_verify_fetch_fails_when_directory_empty(tmp_path):
    source, _ = _make(tmp_path, "osm")
    os.makedirs(source.output_root(PipelineStep.FETCH), exist_ok=True)

    result = source.verify_fetch()
    assert result.ok is False
    assert "missing 1/1" in result.detail
    assert "empty or doesn't exist" in result.detail


def test_verify_fetch_fails_when_directory_missing_entirely(tmp_path):
    source, _ = _make(tmp_path, "osm")
    result = source.verify_fetch()
    assert result.ok is False
    assert "empty or doesn't exist" in result.detail


def test_verify_fetch_reports_mismatched_filenames_not_just_missing(tmp_path):
    # The exact bug this exists to catch: files genuinely present, but under
    # names that don't match any CONFIGURED_FILES entry.
    source, _ = _make(tmp_path, "gadm")
    fetch_dir = source.output_root(PipelineStep.FETCH)
    os.makedirs(fetch_dir, exist_ok=True)
    open(os.path.join(fetch_dir, "some_other_export.zip"), "w").close()

    result = source.verify_fetch()
    assert result.ok is False
    assert "gadm_410-levels.zip" in result.detail  # what was expected
    assert "some_other_export.zip" in result.detail  # what's actually there


def test_verify_fetch_country_classifications_checks_both_hdi_and_worldbank(tmp_path):
    source, _ = _make(tmp_path, "country_classifications")
    fetch_dir = source.output_root(PipelineStep.FETCH)
    os.makedirs(fetch_dir, exist_ok=True)
    # Only the HDI file present -- World Bank file still missing.
    open(os.path.join(fetch_dir, "HDR25.csv"), "w").close()

    result = source.verify_fetch()
    assert result.ok is False
    assert "missing 1/2" in result.detail
    assert "DR0095334.xlsx" in result.detail

    open(os.path.join(fetch_dir, "DR0095334.xlsx"), "w").close()
    result = source.verify_fetch()
    assert result.ok is True


def test_verify_fetch_respects_configured_filename_overrides(tmp_path):
    source, _ = _make(tmp_path, "country_classifications", hdi_name="custom_hdi.csv")
    fetch_dir = source.output_root(PipelineStep.FETCH)
    os.makedirs(fetch_dir, exist_ok=True)
    open(os.path.join(fetch_dir, "custom_hdi.csv"), "w").close()
    open(os.path.join(fetch_dir, "DR0095334.xlsx"), "w").close()

    result = source.verify_fetch()
    assert result.ok is True

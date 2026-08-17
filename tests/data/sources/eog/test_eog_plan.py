"""EogSource: FETCH/PREPARE. See
tests/data/sources/acag/test_acag_plan.py for the mirrored shape.

Also covers the missing-method bug fix (see module docstring in
src/data/sources/eog/source.py): PREPARE planning used to always be empty,
now produces real targets.
"""

import os

import pytest

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.eog.source import EogSource
from src.data.sources.steps import Completion, PipelineStep, TargetSelection

_BASE_URLS = {
    "dmsp": "https://eogdata.mines.edu/wwwdata/dmsp/v4composites_rearrange/",
    "viirs": "https://eogdata.mines.edu/nighttime_light/annual/v21/",
    "dvnl": "https://eogdata.mines.edu/wwwdata/viirs_products/dvnl/",
}
_DATA_PATHS = {"dmsp": "eog/dmsp", "viirs": "eog/viirs", "dvnl": "eog/dvnl"}


@pytest.fixture(autouse=True)
def _no_real_eog_credentials_file(monkeypatch, tmp_path):
    """Guard every test in this module against a real
    orchestration/secrets/eog.credentials.json existing on the machine
    running these tests -- credential resolution here must be driven by
    EOG_USERNAME/EOG_PASSWORD only, never ambient host state (see
    src/data/sources/eog/credentials.py)."""
    from src.data.sources.eog import credentials as eog_credentials

    monkeypatch.setattr(eog_credentials, "DEFAULT_CREDENTIALS_PATH", tmp_path / "unused-eog-credentials.json")


def _make_source(tmp_path, source_type="viirs", **raw):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index")
    )
    cfg = SourceConfig.from_dict(
        f"eog_{source_type}",
        {"data_path": _DATA_PATHS[source_type], "base_url": _BASE_URLS[source_type], **raw},
    )
    return EogSource(ctx, cfg), ctx


def _write_raw_file(source, filename):
    raw_root = source.output_root(PipelineStep.FETCH)
    os.makedirs(raw_root, exist_ok=True)
    open(os.path.join(raw_root, filename), "w").close()


def test_steps_is_fetch_and_prepare_only():
    assert EogSource.STEPS == (PipelineStep.FETCH, PipelineStep.PREPARE)


def test_source_type_derivation_from_source_id(tmp_path, monkeypatch):
    monkeypatch.setenv("EOG_USERNAME", "x")
    monkeypatch.setenv("EOG_PASSWORD", "x")
    assert _make_source(tmp_path, "dmsp")[0].source_type == "dmsp"
    assert _make_source(tmp_path, "viirs")[0].source_type == "viirs_annual"
    assert _make_source(tmp_path, "dvnl")[0].source_type == "viirs_dvnl"


def test_source_type_derivation_ignores_data_path_and_base_url(tmp_path, monkeypatch):
    # The bug being fixed: source_type used to be guessed from data_path/
    # base_url content, disconnected from the actual source_id/alias -- so a
    # "dmsp"-labeled data_path/base_url under a genuinely different
    # source_id must NOT flip the derived variant anymore.
    monkeypatch.setenv("EOG_USERNAME", "x")
    monkeypatch.setenv("EOG_PASSWORD", "x")
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict(
        "eog_viirs", {"data_path": "eog/dmsp", "base_url": _BASE_URLS["dmsp"], "year_range": [2019, 2020]}
    )
    assert EogSource(ctx, cfg).source_type == "viirs_annual"


def test_source_type_derivation_raises_on_unrecognized_source_id(tmp_path, monkeypatch):
    monkeypatch.setenv("EOG_USERNAME", "x")
    monkeypatch.setenv("EOG_PASSWORD", "x")
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("eog", {"data_path": "eog/misc", "base_url": _BASE_URLS["viirs"]})
    with pytest.raises(ValueError, match="Cannot derive EOG source_type"):
        EogSource(ctx, cfg)


def test_output_root_fetch_and_prepare_use_top_level_trees(tmp_path):
    source, ctx = _make_source(tmp_path, "viirs")
    assert source.output_root(PipelineStep.FETCH) == os.path.join(ctx.data_root, "raw", "eog/viirs")


def test_default_resampling_is_sum(tmp_path):
    source, _ = _make_source(tmp_path)
    assert source.resampling == "sum"


def test_base_url_is_required(tmp_path):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("eog_viirs", {"data_path": "eog/viirs"})
    with pytest.raises(ValueError, match="base_url"):
        EogSource(ctx, cfg)


def test_prepare_plan_empty_when_no_raw_files(tmp_path):
    source, _ = _make_source(tmp_path)
    assert source.plan(PipelineStep.PREPARE, TargetSelection()) == []


def test_prepare_plan_one_target_covering_every_available_year(tmp_path):
    # The bug fix: this used to always be empty (module docstring).
    source, _ = _make_source(tmp_path)
    for fname in (
        "F182019.v4d_web.stable_lights.avg_vis.tif",
        "F182020.v4d_web.stable_lights.avg_vis.tgz",
        "F182020.v4d_web.stable_lights.avg_vis.tif",
    ):
        _write_raw_file(source, fname)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    target = targets[0]
    assert target.key == "all"
    assert target.completion == Completion.MARKER
    assert target.meta["years"] == [2019, 2020]


def test_prepare_target_prefers_tif_over_tgz_per_file_extensions_order(tmp_path):
    source, _ = _make_source(tmp_path)
    for fname in (
        "F182020.v4d_web.stable_lights.avg_vis.tgz",
        "F182020.v4d_web.stable_lights.avg_vis.tif",
    ):
        _write_raw_file(source, fname)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert targets[0].meta["raw_files"][2020] == "F182020.v4d_web.stable_lights.avg_vis.tif"


def test_output_path_uses_source_type_family(tmp_path):
    for source_type, family in (
        ("dmsp", "eog_dmsp"),
        ("viirs", "eog_viirs_annual"),
        ("dvnl", "eog_viirs_dvnl"),
    ):
        source, ctx = _make_source(tmp_path, source_type)
        assert source._output_path() == os.path.join(ctx.data_root, "grid", "legacy_4326", f"{family}.zarr")


def test_filename_to_entrypoint_extracts_year_for_viirs_annual(tmp_path):
    source, _ = _make_source(tmp_path, "viirs")
    filename = "VNL_v21_npp_202001-202012_global_vcmslcfg_c202103122300.average_masked.dat.tif.gz"
    assert source.filename_to_entrypoint(filename) == {"year": 2020}


def test_filename_to_entrypoint_none_for_unrecognized_filename(tmp_path):
    source, _ = _make_source(tmp_path, "viirs")
    assert source.filename_to_entrypoint("not_a_viirs_file.tif") is None


def test_filename_to_entrypoint_dmsp_always_none(tmp_path):
    # DMSP/DVNL: entrypoints not used at all (matches old EOGDataSource).
    source, _ = _make_source(tmp_path, "dmsp")
    assert source.filename_to_entrypoint("F182020.v4d_web.stable_lights.avg_vis.tif") is None

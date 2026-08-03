"""EogSource.plan() must reproduce the old EOGPreprocessor's behaviour --
EXCEPT for the missing-method bug (docs/design/09-integrated-pipeline.md §5,
src/data/sources/eog/source.py module docstring), which is fixed here and
must now produce real, non-empty PREPARE targets.

Oracle for everything else: tests/data/preprocess/sources/test_characterization_eog.py.
"""

import os

from src.data.common.ledger.store import PushResult, SourceLedger
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.eog.source import EogSource
from src.data.sources.steps import PipelineStep, TargetSelection

_BASE_URLS = {
    "dmsp": "https://eogdata.mines.edu/wwwdata/dmsp/v4composites_rearrange/",
    "viirs": "https://eogdata.mines.edu/nighttime_light/annual/v21/",
    "dvnl": "https://eogdata.mines.edu/wwwdata/viirs_products/dvnl/",
}
_DATA_PATHS = {"dmsp": "eog/dmsp", "viirs": "eog/viirs", "dvnl": "eog/dvnl"}


def _write_index(local_index_dir, data_path, rows):
    """Build a ledger with the given (relative_path, status_category) rows --
    "completed" means HPC-verified (docs/design/10-fetch-ledger.md), matching
    what `_plan_prepare`'s `completed_fetch_files()` actually reads."""
    safe = data_path.replace("/", "_").replace("\\", "_")
    os.makedirs(local_index_dir, exist_ok=True)
    path = os.path.join(local_index_dir, f"{safe}.duckdb")
    with SourceLedger.open(path, data_path=data_path) as ledger:
        files = [(row["relative_path"], row["relative_path"]) for row in rows]
        ledger.add_remote_files(files, get_file_hash=lambda url: url)
        completed = [row["relative_path"] for row in rows if row["status_category"] == "completed"]
        if completed:
            ledger.record_push_batch(PushResult(step="fetch", unit_id=p, ok=True) for p in completed)


def _make_source(tmp_path, source_type="viirs", year_range=(2019, 2021), rows=None, layout="legacy"):
    data_root = str(tmp_path / "data_root")
    local_index_dir = str(tmp_path / "index")
    data_path = _DATA_PATHS[source_type]
    if rows is None:
        rows = [
            {"relative_path": "F182019.v4d_web.stable_lights.avg_vis.tif", "status_category": "completed"},
            {"relative_path": "F182020.v4d_web.stable_lights.avg_vis.tgz", "status_category": "completed"},
            {"relative_path": "F182020.v4d_web.stable_lights.avg_vis.tif", "status_category": "completed"},
        ]
    _write_index(local_index_dir, data_path, rows)
    ctx = PipelineContext(data_root=data_root, local_index_dir=local_index_dir, layout=layout)
    cfg = SourceConfig.from_dict(
        "eog_viirs", {"data_path": data_path, "year_range": list(year_range), "base_url": _BASE_URLS[source_type]}
    )
    return EogSource(ctx, cfg), ctx


def test_source_type_derivation_from_data_path(tmp_path, monkeypatch):
    monkeypatch.setenv("EOG_USERNAME", "x")
    monkeypatch.setenv("EOG_PASSWORD", "x")
    assert _make_source(tmp_path, "dmsp")[0].source_type == "dmsp"
    assert _make_source(tmp_path, "viirs")[0].source_type == "viirs_annual"
    assert _make_source(tmp_path, "dvnl")[0].source_type == "viirs_dvnl"


def test_output_root_fetch_and_prepare_use_top_level_trees_under_layout_v2(tmp_path):
    source, ctx = _make_source(tmp_path, "viirs", layout="v2")
    assert source.output_root(PipelineStep.FETCH) == os.path.join(ctx.data_root, "raw", "eog/viirs")
    assert source.output_root(PipelineStep.PREPARE) == os.path.join(ctx.data_root, "prepared", "eog/viirs")


def test_default_resampling_is_sum(tmp_path):
    source, _ = _make_source(tmp_path)
    assert source.resampling == "sum"


def test_base_url_is_required(tmp_path):
    import pytest

    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("eog_viirs", {"data_path": "eog/viirs"})
    with pytest.raises(ValueError, match="base_url"):
        EogSource(ctx, cfg)


def test_prepare_targets_are_no_longer_empty_the_bug_is_fixed(tmp_path):
    source, _ = _make_source(tmp_path)
    targets = source.plan(PipelineStep.PREPARE, TargetSelection(year_range=(2019, 2021)))
    assert [t.key for t in targets] == ["2019", "2020"]


def test_prepare_target_prefers_tif_over_tgz_per_file_extensions_order(tmp_path):
    source, _ = _make_source(tmp_path)
    targets = source.plan(PipelineStep.PREPARE, TargetSelection(year_range=(2019, 2021)))
    by_key = {t.key: t for t in targets}
    assert by_key["2020"].inputs == ("F182020.v4d_web.stable_lights.avg_vis.tif",)
    assert by_key["2020"].meta["total_candidates"] == 2


def test_grid_output_filename_uses_source_type(tmp_path):
    source, _ = _make_source(tmp_path, "dmsp", year_range=(2019, 2020))
    annual_dir = source.output_root(PipelineStep.PREPARE)
    os.makedirs(annual_dir, exist_ok=True)
    os.makedirs(os.path.join(annual_dir, "2019.zarr"))

    targets = source.plan(PipelineStep.GRID, TargetSelection(year_range=(2019, 2020)))
    assert len(targets) == 1
    assert targets[0].output_path == os.path.join(source.output_root(PipelineStep.GRID), "dmsp_timeseries_reprojected.zarr")


def test_grid_target_uses_v2_family_path_under_layout_v2(tmp_path):
    for source_type, family in (
        ("dmsp", "eog_dmsp"),
        ("viirs", "eog_viirs_annual"),
        ("dvnl", "eog_viirs_dvnl"),
    ):
        source, ctx = _make_source(tmp_path, source_type, year_range=(2019, 2020), layout="v2")
        annual_dir = source.output_root(PipelineStep.PREPARE)
        os.makedirs(annual_dir, exist_ok=True)
        os.makedirs(os.path.join(annual_dir, "2019.zarr"))

        targets = source.plan(PipelineStep.GRID, TargetSelection(year_range=(2019, 2020)))
        assert len(targets) == 1
        assert targets[0].output_path == os.path.join(ctx.data_root, "grid", "legacy_4326", f"{family}.zarr")

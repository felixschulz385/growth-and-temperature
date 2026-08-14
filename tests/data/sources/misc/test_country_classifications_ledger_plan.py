"""CountryClassificationsSource: ledger-free FETCH/PREPARE (docs/design
successor to the ledger, Plan 2 PREPARE+GRID merge). plan() is a bare live
`os.path.exists()` check against the raw fetched files -- see
tests/data/sources/misc/test_gadm_ledger_plan.py's identical shape.
"""

import os

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.steps import Completion, PipelineStep, TargetSelection


def _make(tmp_path):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"), layout="legacy"
    )
    cfg = SourceConfig.from_dict("country_classifications", {})
    cls = registry.load("country_classifications")
    return cls(ctx, cfg), ctx


def _write_fake_hdi_and_wb(source):
    hdi_file, wb_file = source._raw_file("hdi"), source._raw_file("worldbank")
    os.makedirs(os.path.dirname(hdi_file), exist_ok=True)
    open(hdi_file, "w").close()
    open(wb_file, "w").close()
    return hdi_file, wb_file


def test_steps_is_fetch_and_prepare_only():
    from src.data.sources.misc.country_classifications import CountryClassificationsSource

    assert CountryClassificationsSource.STEPS == (PipelineStep.FETCH, PipelineStep.PREPARE)


def test_requires_gadm_prepare():
    spec = registry.resolve("country_classifications")
    assert spec.requires == ((PipelineStep.PREPARE, "gadm", PipelineStep.PREPARE),)


def test_prepare_plan_empty_when_no_raw_files(tmp_path):
    source, _ = _make(tmp_path)
    assert source.plan(PipelineStep.PREPARE, TargetSelection()) == []


def test_prepare_plan_one_target_when_raw_files_present(tmp_path):
    source, _ = _make(tmp_path)
    hdi_file, wb_file = _write_fake_hdi_and_wb(source)

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    target = targets[0]
    assert target.key == "country_classifications"
    assert set(target.inputs) == {hdi_file, wb_file}
    assert target.completion == Completion.PATH_EXISTS
    assert target.output_path == source._output_path()
    assert target.output_path.endswith("classifications_by_gid0.parquet")


def test_prepare_plan_tracks_which_of_hdi_worldbank_present(tmp_path):
    source, _ = _make(tmp_path)
    hdi_file = source._raw_file("hdi")
    os.makedirs(os.path.dirname(hdi_file), exist_ok=True)
    open(hdi_file, "w").close()

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    assert targets[0].meta["has_hdi"] is True
    assert targets[0].meta["has_wb"] is False

"""`_check_requires()`'s ledger lookup for the misc-split sources
(gadm/osm/country_classifications/ecoregions, all `cfg.data_path="misc"`,
disambiguated via `cfg.namespace`) -- regression coverage for the bug where
it read `requires_cfg.data_path` directly instead of mirroring the
overridden `DataSource.data_path` property (src/data/sources/base.py),
computing the wrong ledger filename ("misc.duckdb" instead of
"misc_gadm.duckdb") and silently falling through to a local-disk-only check.
"""

from src.cli.data.handlers import _check_requires
from src.data.common.ledger.paths import ledger_path
from src.data.common.ledger.store import SourceLedger
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.steps import MissingPrerequisiteError, PipelineStep


def _config():
    return {"sources": {"gadm": {"data_path": "misc", "namespace": "gadm"}}}


def _mark_step_complete(ledger_path_, data_path, step):
    with SourceLedger.open(ledger_path_, data_path=data_path) as ledger:
        ledger.ensure_artifact(step.value, "all")
        ledger.set_local_state(step.value, "all", "complete")


def test_check_requires_finds_misc_split_prerequisite_via_its_namespaced_ledger(tmp_path):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    config = _config()
    spec = registry.resolve("ecoregions")  # REQUIRES = gadm PREPARE + gadm GRID

    gadm_data_path = "misc/gadm"  # what GadmSource.data_path actually resolves to
    gadm_ledger_path = ledger_path(ctx.local_index_dir, gadm_data_path)
    import os

    os.makedirs(os.path.dirname(gadm_ledger_path), exist_ok=True)
    _mark_step_complete(gadm_ledger_path, gadm_data_path, PipelineStep.PREPARE)
    _mark_step_complete(gadm_ledger_path, gadm_data_path, PipelineStep.GRID)

    # gadm's local output directories were never created -- only its ledger
    # (the cross-machine case: gadm ran elsewhere and was pushed via
    # `data transfer`). Before the fix, _check_requires computed
    # ledger_path(..., "misc") -> a file that doesn't exist, silently skipped
    # the ledger check, and raised on the local os.path.exists() fallback.
    _check_requires(spec, ctx, config)  # must not raise


def test_check_requires_still_raises_when_prerequisite_truly_incomplete(tmp_path):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    config = _config()
    spec = registry.resolve("ecoregions")

    try:
        _check_requires(spec, ctx, config)
    except MissingPrerequisiteError as e:
        assert e.requires_id == "gadm"
    else:
        raise AssertionError("expected MissingPrerequisiteError")

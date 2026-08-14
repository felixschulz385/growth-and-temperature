"""`_check_requires()` builds the actual required source and checks
`is_complete()` against its own `plan()`-ed targets, rather than guessing a
path from `layout.output_root(data_path, step)`. Regression coverage for the
bug that guess reintroduced: gadm's PREPARE writes to GRID's path, so the
generic per-step path guess silently pointed at the wrong (always-empty)
directory.
"""

import os

from src.cli.data.handlers import _check_requires
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.misc.gadm import GadmSource
from src.data.sources.steps import MissingPrerequisiteError, PipelineStep, mark_complete


def _config():
    return {"sources": {"gadm": {"data_path": "misc", "namespace": "gadm"}}}


def test_check_requires_finds_prerequisite_via_its_real_planned_output(tmp_path):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    config = _config()
    spec = registry.resolve("ecoregions")  # REQUIRES = gadm PREPARE, scoped to ecoregions' own PREPARE

    from src.data.pipeline.config import get_source_config

    source = GadmSource(ctx, get_source_config(config, "gadm"))
    raw_file = source._raw_file_path()
    os.makedirs(os.path.dirname(raw_file), exist_ok=True)
    open(raw_file, "w").close()

    # gadm's PREPARE target's real output is its final grid zarr (what used
    # to be GRID's own output path) -- mark it complete directly, without
    # running the actual vector-extraction/rasterization pipeline.
    output_path = source._grid_output_path()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mark_complete(output_path)

    _check_requires(spec, ctx, config, PipelineStep.PREPARE)  # must not raise


def test_check_requires_still_raises_when_prerequisite_truly_incomplete(tmp_path):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    config = _config()
    spec = registry.resolve("ecoregions")

    try:
        _check_requires(spec, ctx, config, PipelineStep.PREPARE)
    except MissingPrerequisiteError as e:
        assert e.requires_id == "gadm"
    else:
        raise AssertionError("expected MissingPrerequisiteError")


def test_check_requires_raises_when_raw_file_exists_but_output_not_yet_complete(tmp_path):
    # A planned-but-not-yet-complete target must still gate -- plan()
    # returning something isn't enough on its own, is_complete() must agree.
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    config = _config()
    spec = registry.resolve("ecoregions")

    from src.data.pipeline.config import get_source_config

    source = GadmSource(ctx, get_source_config(config, "gadm"))
    raw_file = source._raw_file_path()
    os.makedirs(os.path.dirname(raw_file), exist_ok=True)
    open(raw_file, "w").close()
    # No mark_complete() -- the target exists but isn't done yet.

    try:
        _check_requires(spec, ctx, config, PipelineStep.PREPARE)
    except MissingPrerequisiteError as e:
        assert e.requires_id == "gadm"
    else:
        raise AssertionError("expected MissingPrerequisiteError")


def test_check_requires_does_not_gate_fetch(tmp_path):
    # ecoregions' REQUIRES on gadm is scoped to its own (merged) PREPARE
    # step only -- FETCH must run unblocked even with no gadm output
    # anywhere.
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    config = _config()
    spec = registry.resolve("ecoregions")

    _check_requires(spec, ctx, config, PipelineStep.FETCH)  # must not raise

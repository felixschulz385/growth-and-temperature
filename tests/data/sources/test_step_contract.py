"""Cross-source contract checks -- docs/design/09-integrated-pipeline.md §2/§4.

Parameterized over every currently-registered source rather than one fixed
list, so a newly-migrated source is covered automatically.
"""

import pytest

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.steps import PipelineStep, StepTarget, TargetSelection, UnsupportedStepError

_ALL_STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE, PipelineStep.GRID)

#: Config a source requires beyond {"data_path": ...} just to construct --
#: e.g. eog's base_url has no sensible default (docs/design/09-integrated-pipeline.md §5).
_EXTRA_CONFIG: dict[str, dict] = {
    "eog": {"base_url": "https://example.invalid/eog"},
    "glass_modis": {
        "base_url": "https://example.invalid/glass/modis/",
        "day_range": {"start": [2000, 55], "end": [2020, 365]},
    },
    "glass_avhrr": {
        "base_url": "https://example.invalid/glass/avhrr/",
        "day_range": {"start": [1992, 1], "end": [2020, 365]},
    },
}

#: `source_id` to actually construct with, for sources whose bare `spec.id`
#: isn't itself a valid config-block key (EOG's `source_type` -- DMSP/VIIRS-
#: annual/DVNL -- is derived from `cfg.source_id`, src/data/sources/eog/
#: source.py, so it must be built with one of its real aliases, not the
#: generic "eog"). Defaults to `spec.id` when a source has no entry here.
_CONSTRUCT_AS: dict[str, str] = {
    "eog": "eog_viirs",
}


def _instantiate(spec, tmp_path):
    cls = registry.load(spec.id)
    ctx = PipelineContext(data_root=str(tmp_path / "data"), local_index_dir=str(tmp_path / "index"))
    source_id = _CONSTRUCT_AS.get(spec.id, spec.id)
    cfg = SourceConfig.from_dict(source_id, {"data_path": spec.id, **_EXTRA_CONFIG.get(spec.id, {})})
    return cls(ctx, cfg)


@pytest.mark.parametrize("spec", registry.all_specs(), ids=lambda s: s.id)
def test_plan_rejects_a_step_outside_steps(spec, tmp_path):
    source = _instantiate(spec, tmp_path)
    for step in _ALL_STEPS:
        if step in source.STEPS:
            continue
        with pytest.raises(UnsupportedStepError):
            source.plan(step, TargetSelection())


@pytest.mark.parametrize("spec", registry.all_specs(), ids=lambda s: s.id)
def test_execute_rejects_a_step_outside_steps(spec, tmp_path):
    source = _instantiate(spec, tmp_path)
    for step in _ALL_STEPS:
        if step in source.STEPS:
            continue
        fake_target = StepTarget(source_id=spec.id, step=step, key="x", output_path=str(tmp_path / "x"))
        with pytest.raises(UnsupportedStepError):
            source.execute(fake_target)


@pytest.mark.parametrize("spec", registry.all_specs(), ids=lambda s: s.id)
def test_requires_edges_name_a_registered_source(spec):
    for _my_step, requires_id, _requires_step in spec.requires:
        registry.resolve(requires_id)  # raises ValueError if unknown


@pytest.mark.parametrize("spec", registry.all_specs(), ids=lambda s: s.id)
def test_requires_never_points_at_itself(spec):
    for _my_step, requires_id, _requires_step in spec.requires:
        assert requires_id != spec.id, f"{spec.id} REQUIRES itself"


@pytest.mark.parametrize("spec", registry.all_specs(), ids=lambda s: s.id)
def test_requires_edges_scoped_to_a_step_this_source_implements(spec):
    for my_step, _requires_id, _requires_step in spec.requires:
        assert my_step in spec.steps, f"{spec.id} REQUIRES entry scoped to step '{my_step.value}' it doesn't implement"

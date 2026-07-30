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
    "glass_modis": {"base_url": "https://example.invalid/glass/modis/"},
    "glass_avhrr": {"base_url": "https://example.invalid/glass/avhrr/"},
}


def _instantiate(spec, tmp_path):
    cls = registry.load(spec.id)
    ctx = PipelineContext(data_root=str(tmp_path / "data"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict(spec.id, {"data_path": spec.id, **_EXTRA_CONFIG.get(spec.id, {})})
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
    for requires_id, _requires_step in spec.requires:
        registry.resolve(requires_id)  # raises ValueError if unknown


@pytest.mark.parametrize("spec", registry.all_specs(), ids=lambda s: s.id)
def test_requires_never_points_at_itself(spec):
    for requires_id, _requires_step in spec.requires:
        assert requires_id != spec.id, f"{spec.id} REQUIRES itself"

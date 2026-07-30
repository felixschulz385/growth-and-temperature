"""Registry invariants -- docs/design/09-integrated-pipeline.md §4.

Guards against the exact bug class that motivated this redesign: a source
name that resolves in one registry but not another (`--source eog_viirs`
raising ImportError from the old preprocess factory, docs/design/09-integrated-pipeline.md §1).
"""

from src.data.sources import registry
from src.data.sources.base import DataSource
from src.data.sources.steps import PipelineStep


def test_every_registered_module_actually_imports():
    for spec in registry.all_specs():
        cls = registry.load(spec.id)
        assert issubclass(cls, DataSource)
        assert cls.__name__ == spec.class_name


def test_spec_steps_match_the_loaded_class_attribute():
    for spec in registry.all_specs():
        cls = registry.load(spec.id)
        assert tuple(cls.STEPS) == tuple(spec.steps), f"{spec.id}: spec.steps drifted from {cls.__name__}.STEPS"


def test_spec_requires_match_the_loaded_class_attribute():
    for spec in registry.all_specs():
        cls = registry.load(spec.id)
        assert tuple(cls.REQUIRES) == tuple(spec.requires), f"{spec.id}: spec.requires drifted from {cls.__name__}.REQUIRES"


def test_no_duplicate_aliases_across_sources():
    seen: dict[str, str] = {}
    for spec in registry.all_specs():
        for name in spec.all_names:
            key = name.lower()
            assert key not in seen, f"alias '{name}' registered to both '{seen[key]}' and '{spec.id}'"
            seen[key] = spec.id


def test_resolve_is_case_insensitive_and_covers_every_alias():
    for spec in registry.all_specs():
        for name in spec.all_names:
            assert registry.resolve(name.upper()).id == spec.id
            assert registry.resolve(name.lower()).id == spec.id


def test_unknown_source_raises_with_a_helpful_message():
    import pytest

    with pytest.raises(ValueError, match="Unknown source"):
        registry.resolve("definitely-not-a-registered-source")


def test_every_source_declares_at_least_one_step():
    for spec in registry.all_specs():
        assert spec.steps, f"{spec.id} declares no PipelineStep at all"
        assert all(isinstance(s, PipelineStep) for s in spec.steps)

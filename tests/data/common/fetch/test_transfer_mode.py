from types import SimpleNamespace

from src.data.common.fetch.transfer_mode import resolve_transfer_mode


def _source(default_transfer_mode, *, raw=None):
    cfg = SimpleNamespace(raw=raw or {})
    return SimpleNamespace(cfg=cfg, DEFAULT_TRANSFER_MODE=default_transfer_mode)


def test_defaults_to_the_sources_class_level_default():
    assert resolve_transfer_mode(_source("auto")) == "auto"
    assert resolve_transfer_mode(_source("manual")) == "manual"


def test_explicit_config_overrides_the_default_either_direction():
    assert resolve_transfer_mode(_source("auto", raw={"transfer_mode": "manual"})) == "manual"
    assert resolve_transfer_mode(_source("manual", raw={"transfer_mode": "auto"})) == "auto"

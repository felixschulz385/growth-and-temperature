from types import SimpleNamespace

from src.data.common.fetch.transfer_mode import resolve_transfer_mode


def _source(source_id, *, source_id_attr=None, raw=None):
    cfg = SimpleNamespace(source_id=source_id, raw=raw or {})
    return SimpleNamespace(cfg=cfg, ID=source_id_attr or source_id)


def test_defaults_to_auto_for_a_known_high_volume_source():
    assert resolve_transfer_mode(_source("modis")) == "auto"


def test_defaults_to_manual_for_an_unlisted_source():
    assert resolve_transfer_mode(_source("gadm")) == "manual"


def test_matches_by_cfg_source_id_not_class_id():
    # GlassModisSource registers two ids ("glass_modis"/"glass_ta_modis",
    # docs/design/12-glass-modis-rebuild.md §4) sharing one class whose .ID
    # is always "glass_modis" -- matching on cfg.source_id (the config key
    # the instance was actually built under) is required for either alias to
    # resolve to "auto". glass_avhrr is its own separate class/id.
    assert resolve_transfer_mode(_source("glass_modis", source_id_attr="glass_modis")) == "auto"
    assert resolve_transfer_mode(_source("glass_ta_modis", source_id_attr="glass_modis")) == "auto"
    assert resolve_transfer_mode(_source("glass_avhrr", source_id_attr="glass_avhrr")) == "auto"


def test_explicit_config_overrides_the_default_either_direction():
    assert resolve_transfer_mode(_source("modis", raw={"transfer_mode": "manual"})) == "manual"
    assert resolve_transfer_mode(_source("gadm", raw={"transfer_mode": "auto"})) == "auto"


def test_defaults_to_auto_for_eog_viirs_by_its_actual_config_id():
    # EogSource.ID is the shared literal "eog" for all three aliases
    # (src/data/sources/eog/source.py), but the config block key actually
    # used in orchestration/configs/data.yaml -- and thus cfg.source_id -- is
    # "eog_viirs" ("eog_dmsp"/"eog_dvnl" for the other two, disabled,
    # aliases). Matching only "eog" here would never fire for any real
    # config.
    assert resolve_transfer_mode(_source("eog_viirs", source_id_attr="eog")) == "auto"
    assert resolve_transfer_mode(_source("eog_dmsp", source_id_attr="eog")) == "auto"
    assert resolve_transfer_mode(_source("eog_dvnl", source_id_attr="eog")) == "auto"

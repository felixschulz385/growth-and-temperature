"""
Registry mapping data-source names to the module that can build them.

Each source module exposes a ``NAMES`` tuple of alias strings and a
``from_config(dataset_name, config, *, base_url, file_extensions,
output_path, source_config, **kwargs)`` function. Modules are imported
lazily — only the one matching module is imported per call — so heavy
per-source dependencies (e.g. Selenium for eog/glass) aren't loaded unless
that source is actually requested.
"""

import importlib

_REGISTRY: dict[str, str] = {}


def _register(module_path: str, names: tuple[str, ...]) -> None:
    for name in names:
        _REGISTRY[name.lower()] = module_path


_register("src.data.download.sources.misc", ("misc",))
_register("src.data.download.sources.glass.source", ("glass_modis", "glass_avhrr"))
_register("src.data.download.sources.eog.source", ("eog_dmsp", "eog_viirs", "eog_dvnl"))
_register("src.data.download.sources.ntl_harm", ("ntl_harm", "ntlharm", "harmonized_ntl"))
_register("src.data.download.sources.harvard", ("harvard_plad", "harvard"))
_register("src.data.download.sources.manual", ("berman_mining", "berman", "mining_conflict"))
_register("src.data.download.sources.acag", ("acag", "acag_pm25", "pm25"))
_register("src.data.download.sources.esacci", ("esacci", "esa_cci", "esacci_lc", "landcover"))


def get_factory(dataset_name: str):
    """Return the ``from_config`` function registered for *dataset_name*."""
    module_path = _REGISTRY.get(dataset_name.lower())
    if module_path is None:
        raise ValueError(f"Unknown data source: {dataset_name}")
    module = importlib.import_module(module_path)
    return module.from_config

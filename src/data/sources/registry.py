"""Registry mapping source ids/aliases to the module+class that implements them.

docs/design/09-integrated-pipeline.md §4: extends the download side's proven
lazy-import pattern (src/data/download/sources/registry.py) to cover the whole
merged pipeline, and adds metadata (`steps`/`requires`) available *without*
importing the module -- so `data list` and the SLURM generator can
validate `(source, step)` pairs without pulling per-source dependencies
(Selenium, pystac, duckdb, ...) into the process.

Modules register themselves by calling `register()` at import time from this
module's `_register_all()` -- one call per source package, added as each
source migrates (docs/design/09-integrated-pipeline.md §10). A module is only
actually imported (via `load`/`create`) when the caller asks for that specific
source.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.data.sources.steps import PipelineStep

if TYPE_CHECKING:
    from src.data.pipeline.config import SourceConfig
    from src.data.pipeline.context import PipelineContext
    from src.data.sources.base import DataSource


@dataclass(frozen=True)
class SourceSpec:
    id: str
    aliases: tuple[str, ...]
    module: str
    class_name: str
    steps: tuple[PipelineStep, ...]
    #: `(my_step, prereq_source_id, prereq_step)` triples -- see
    #: `DataSource.REQUIRES`'s docstring (src/data/sources/base.py).
    requires: tuple[tuple[PipelineStep, str, PipelineStep], ...] = ()

    @property
    def all_names(self) -> tuple[str, ...]:
        return (self.id, *self.aliases)

    def requires_for(self, step: PipelineStep) -> tuple[tuple[str, PipelineStep], ...]:
        """This source's `(prereq_source_id, prereq_step)` requirements that
        gate *step* specifically, dropping the ones scoped to its other
        steps."""
        return tuple((rid, rstep) for my_step, rid, rstep in self.requires if my_step is step)


_REGISTRY: dict[str, SourceSpec] = {}
_BY_ID: dict[str, SourceSpec] = {}


def register(
    id: str,
    module: str,
    class_name: str,
    steps: tuple[PipelineStep, ...],
    *,
    aliases: tuple[str, ...] = (),
    requires: tuple[tuple[PipelineStep, str, PipelineStep], ...] = (),
) -> None:
    if id in _BY_ID:
        raise ValueError(f"Source id '{id}' is already registered (module={_BY_ID[id].module})")
    spec = SourceSpec(id=id, aliases=aliases, module=module, class_name=class_name, steps=steps, requires=requires)
    _BY_ID[id] = spec
    for name in spec.all_names:
        key = name.lower()
        if key in _REGISTRY:
            raise ValueError(f"Alias '{name}' is already registered to source '{_REGISTRY[key].id}'")
        _REGISTRY[key] = spec


def resolve(name: str) -> SourceSpec:
    """Alias/id -> SourceSpec. Does not import the *resolved* source's own
    module (only every source module's top-level `register()` call, via
    `_register_all()`, which is cheap and import-free by construction)."""
    _register_all()
    spec = _REGISTRY.get(name.lower())
    if spec is None:
        raise ValueError(f"Unknown source: '{name}'. Registered: {sorted(_BY_ID)}")
    return spec


def describe(name: str) -> SourceSpec:
    """Alias for resolve(), for CLI readability (`data list` calls this)."""
    return resolve(name)


def load(name: str) -> type["DataSource"]:
    """Import the module for *name* and return its DataSource subclass."""
    spec = resolve(name)
    module = importlib.import_module(spec.module)
    if not hasattr(module, spec.class_name):
        raise AttributeError(f"Module {spec.module} does not define class {spec.class_name}")
    return getattr(module, spec.class_name)


def create(name: str, ctx: "PipelineContext", cfg: "SourceConfig") -> "DataSource":
    cls = load(name)
    return cls.from_config(ctx, cfg)


def all_specs() -> tuple[SourceSpec, ...]:
    _register_all()
    return tuple(_BY_ID.values())


_registered_all = False


def _reset_for_tests() -> None:
    """Test-only: clear the registry so tests can register fakes in isolation."""
    global _registered_all
    _REGISTRY.clear()
    _BY_ID.clear()
    _registered_all = False


def _register_all() -> None:
    """Import every source package so its module-level `register(...)` call
    runs. Idempotent and cheap after the first call -- `resolve()`/
    `all_specs()` call this automatically, so callers never need to remember
    to.

    docs/design/09-integrated-pipeline.md §10: sources are added to
    `_SOURCE_MODULES` one at a time as each migrates; nothing here imports a
    source's heavy dependencies (Selenium, pystac, duckdb, ...) until that
    source's own module does so at import time.
    """
    global _registered_all
    if _registered_all:
        return
    for module_name in _SOURCE_MODULES:
        importlib.import_module(module_name)
    _registered_all = True


#: Populated incrementally as each source migrates
#: (docs/design/09-integrated-pipeline.md §5/§10). Each module registers
#: itself via `registry.register(...)` at import time.
_SOURCE_MODULES: tuple[str, ...] = (
    "src.data.sources.acag",
    "src.data.sources.esacci",
    "src.data.sources.ntl_harm",
    "src.data.sources.eog.source",
    "src.data.sources.glass.source",
    "src.data.sources.modis.source",
    "src.data.sources.misc.osm",
    "src.data.sources.misc.gadm",
    "src.data.sources.misc.country_classifications",
    "src.data.sources.ecoregions.source",
    "src.data.sources.commodity_prices.source",
    "src.data.sources.plad",
    "src.data.sources.berman_mining",
    "src.data.sources.snl_mining.source",
)

"""Lazy package init (PEP 562): `.source` pulls in the full pipeline stack
(geobox -> pandas -> pyarrow), which every submodule import of this package
would otherwise force -- including `snl_mining.scraper`, an unrelated,
standalone tool (source.py's own docstring) that shouldn't need it. Deferring
the import until `SnlMiningSource` is actually accessed keeps
`import src.data.sources.snl_mining.scraper...` cheap, which matters when
`scraper/stages/regularize_detail_exports.py` spawns many worker processes
that each re-run this package's `__init__.py` on startup.
"""

from __future__ import annotations

__all__ = ["SnlMiningSource"]


def __getattr__(name: str):
    if name == "SnlMiningSource":
        from .source import SnlMiningSource

        return SnlMiningSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

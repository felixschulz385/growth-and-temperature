"""Resolves a source's `transfer_mode` (auto|manual): whether its completed
FETCH output should be pushed to HPC without an explicit `data transfer`
call, and -- the same underlying question -- whether "already fetched"
should be judged against the HPC target instead of local disk.

An "auto" source pushes fetched files to HPC right after FETCH
(`_maybe_auto_transfer()`, `src/cli/data/handlers.py`) and isn't expected to
keep every copy on local disk indefinitely -- for those, checking local
presence to decide what's still outstanding would make an already-pushed,
locally-pruned file look outstanding forever. `manual` sources keep the
local-disk-is-truth default (`src/data/common/fetch/manifest.py`'s
`resolve_fetch_listing()`/`src/data/common/fetch/driver.py`'s `run_fetch()`).
"""

from __future__ import annotations

from typing import Any

#: Sources that default to "auto" unless a config explicitly overrides
#: `transfer_mode` -- the high-disk-usage raster sources, where keeping
#: every fetched file on local disk indefinitely isn't practical. Matched
#: against `source.cfg.source_id` (the config key the source was created
#: under, e.g. "glass_modis"/"modis_robustness_11a1"), not `source.ID` --
#: several ids here (glass_modis/glass_avhrr) share one class/`.ID` value
#: ("glass"), so `.ID` alone can't distinguish them.
AUTO_TRANSFER_DEFAULT_SOURCES = frozenset(
    {
        "modis",
        "modis_lst",
        "modis_robustness_11a1",
        "modis_extended",
        "glass_modis",
        "glass_ta_modis",
        "glass_avhrr",
        "acag",
        "esacci",
        "ntl_harm",
        "eog_dmsp",
        "eog_viirs",
        "eog_dvnl",
    }
)


def resolve_transfer_mode(source: Any) -> str:
    """`sources.<id>.transfer_mode` if configured, else the default implied
    by whether this source's config id is in `AUTO_TRANSFER_DEFAULT_SOURCES`."""
    source_id = getattr(source.cfg, "source_id", None) or getattr(source, "ID", None)
    default_mode = "auto" if (source_id and source_id.lower() in AUTO_TRANSFER_DEFAULT_SOURCES) else "manual"
    return source.cfg.raw.get("transfer_mode", default_mode)

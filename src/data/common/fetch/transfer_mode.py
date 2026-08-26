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

The default itself is declared per-source-class
(`DataSource.DEFAULT_TRANSFER_MODE`, `src/data/sources/base.py`) rather than
here, so a new high-disk-usage source's author sets it where they're already
required to look (next to `has_entrypoints`) instead of separately
discovering and editing a central registry of source ids.
"""

from __future__ import annotations

from typing import Any


def resolve_transfer_mode(source: Any) -> str:
    """`sources.<id>.transfer_mode` if configured, else *source*'s class-level
    `DEFAULT_TRANSFER_MODE`."""
    configured = source.cfg.raw.get("transfer_mode")
    return configured if configured is not None else source.DEFAULT_TRANSFER_MODE

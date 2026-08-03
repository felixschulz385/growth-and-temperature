"""Ledger file path conventions -- local and remote.

Mirrors `src.data.sources.layout.index_path()`'s sanitization
(`data_path.replace("/", "_").replace("\\\\", "_")`) so this is a drop-in
rename of today's per-source Parquet index file, not a new directory scheme.
"""

from __future__ import annotations

import os

#: Directory (relative to the HPC target's base path) the remote copy of
#: every source's ledger lives under -- renamed from today's hardcoded
#: `hpc_data_index/` since "nothing has to remain" (docs/design/10-fetch-ledger.md).
REMOTE_LEDGER_DIR = "_ledger"


def safe_data_path(data_path: str) -> str:
    return data_path.replace("/", "_").replace("\\", "_")


def ledger_path(local_index_dir: str | None, data_path: str) -> str | None:
    """Local `.duckdb` path for *data_path*'s ledger, or `None` if
    `local_index_dir` isn't configured -- mirrors `layout.index_path()`'s
    same None-on-unconfigured contract (src/data/sources/layout.py) so
    callers can share the same `if not path: treat as "nothing to open"`
    guard already used throughout src/data/sources/*.py's `_plan_prepare()`.
    """
    if not local_index_dir:
        return None
    return os.path.join(local_index_dir, f"{safe_data_path(data_path)}.duckdb")


def remote_ledger_path(data_path: str) -> str:
    """Remote `.duckdb` path, relative to the HPC target's base path
    (`HPCClient.base_path` resolves this the same way it resolves every
    other relative remote path)."""
    return f"{REMOTE_LEDGER_DIR}/{safe_data_path(data_path)}.duckdb"

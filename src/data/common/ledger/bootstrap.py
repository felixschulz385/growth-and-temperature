"""FETCH catalog bootstrap/reconciliation: reconcile a `SourceLedger`'s
FETCH-side state against real remote-origin and (optionally) real HPC
filesystem state, treated as ground truth -- rather than trusting whatever
the ledger currently believes.

docs/design/10-fetch-ledger.md §5 ("no migration of old data" -- this is the
one-time/occasional operator tool, run via `pipeline reconcile`, that
replaces converting the old `UnifiedDataIndex`/`TransferManifest` Parquet
files). PREPARE/GRID reconciliation is a separate module
(`src/data/sources/reconcile.py`) since it needs `DataSource.plan()` --
this module stays dependency-free of `src.data.sources`, matching every
other module under `common/`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.data.common.ledger import catalog
from src.data.common.ledger.catalog import RemoteFileCatalogLike
from src.data.common.ledger.schema import LocalState, RemoteState
from src.data.common.ledger.store import SourceLedger

logger = logging.getLogger(__name__)


def _full_remote_path(client: Any, remote_path: str) -> str:
    base_path = getattr(client, "base_path", None)
    if remote_path.startswith("/") or not base_path:
        return remote_path
    return f"{base_path}/{remote_path}"


def reconcile_fetch(
    ledger: SourceLedger,
    source: RemoteFileCatalogLike,
    *,
    raw_root: str,
    client: Optional[Any] = None,
) -> dict[str, int]:
    """Reconcile *ledger*'s FETCH catalog against ground truth.

    1. Crawl the remote origin fresh (`catalog.refresh`) -- "what should
       exist," exactly like a normal fetch's first step.
    2. If *client* (an `HPCClient`) is given, list every file actually
       present under *raw_root* (a path relative to `client.base_path` --
       callers compute this the same way the FETCH driver will, e.g.
       `layout.raw_root("", data_path, layout=ctx.layout)`) on the HPC
       filesystem in one SSH round trip, and mark any `remote_files` row
       whose `relative_path` matches as `local_state=complete,
       remote_state=verified`. Physical presence with the correct relative
       path *is* ground truth here -- every row is checked, not sampled,
       since this is a one-time reconciliation, not the hot-path
       sample-verify `HPCPusher` uses.
    3. Anything not found in that listing is left `missing`/`missing`
       ("still needs fetching") -- correct, not a failure.

    Returns `{"discovered": <new remote files found>, "verified_present":
    <files confirmed present on HPC>}`.
    """
    added = catalog.refresh(ledger, source)
    result = {"discovered": added, "verified_present": 0}

    if client is None:
        return result

    full_raw_root = _full_remote_path(client, raw_root)
    success, stdout, stderr = client.execute_command(f"find '{full_raw_root}' -type f -printf '%P\\n'")
    if not success:
        logger.warning("Could not list remote raw files under %s: %s", full_raw_root, stderr)
        return result

    remote_relative_paths = {line.strip() for line in stdout.splitlines() if line.strip()}
    if not remote_relative_paths:
        logger.info("No files found under %s on HPC yet", full_raw_root)
        return result

    matched = [
        file_hash for file_hash, relative_path in ledger.iter_remote_files() if relative_path in remote_relative_paths
    ]
    ledger.mark_local_and_remote_batch("fetch", matched, LocalState.COMPLETE, RemoteState.VERIFIED)
    result["verified_present"] = len(matched)
    logger.info("Verified %d/%d catalog file(s) present on HPC", len(matched), len(remote_relative_paths))
    return result

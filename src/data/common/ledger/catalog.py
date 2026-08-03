"""Crawl-catalog refresh: entrypoint discovery + remote-file listing merged
into a `SourceLedger`. Replaces `UnifiedDataIndex.build_index_from_source`/
`_load_entrypoints`/`_find_missing_entrypoints`/`_add_files_to_index`
(src/data/common/index/unified_index.py).

docs/design/10-fetch-ledger.md. `RemoteFileCatalogLike` below is a
structural (duck-typed) subset of `src.data.sources.base.RemoteFileCatalog`
-- deliberately not imported, so `common/` stays independent of `sources/`
(the rest of the repo only ever imports the other direction).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from src.data.common.ledger.store import SourceLedger
from src.data.common.ledger.store import entrypoint_key as _entrypoint_key

logger = logging.getLogger(__name__)


class RemoteFileCatalogLike(Protocol):
    has_entrypoints: bool

    def list_remote_files(self, entrypoint: dict[str, Any] | None = None) -> list[tuple[str, str]]: ...
    def get_file_hash(self, file_url: str) -> str: ...
    def get_all_entrypoints(self) -> list[dict[str, Any]]: ...


def refresh(ledger: SourceLedger, source: RemoteFileCatalogLike) -> int:
    """Crawl whatever's new from *source* into *ledger*. Returns the number
    of newly discovered remote files.

    Simple (non-entrypoint) sources: one `list_remote_files()` call, same as
    today.

    Entrypoint-based sources (EOG/GLASS/...): `get_all_entrypoints()` is
    called on every refresh (a top-level directory listing, e.g. one crawl
    call per available year) -- deliberately *not* cached forever the way
    the old JSON side-file cache did, since caching it forever means a
    genuinely new entrypoint (e.g. this year's data becoming available)
    would never be discovered without an operator manually deleting the
    cache file. `entrypoints.crawled` tracks the actually-expensive part:
    only entrypoints not yet crawled get `list_remote_files(entrypoint)`
    called (a full recursive listing), and each is marked crawled
    immediately after -- a re-run only pays for genuinely new entrypoints,
    unlike the old code's per-call full-column `filename_to_entrypoint()`
    re-scan to figure out what's missing.
    """
    if not getattr(source, "has_entrypoints", False):
        remote_files = list(source.list_remote_files())
        logger.info("Found %d remote file(s)", len(remote_files))
        return ledger.add_remote_files(remote_files, get_file_hash=source.get_file_hash)

    all_entrypoints = source.get_all_entrypoints()
    logger.info("Found %d entrypoint(s)", len(all_entrypoints))
    if not all_entrypoints:
        raise ValueError("No entrypoints found -- cannot refresh catalog")

    newly_known = ledger.upsert_entrypoints(all_entrypoints)
    if newly_known:
        logger.info("Registered %d new entrypoint(s)", newly_known)

    to_crawl = ledger.missing_entrypoints()
    if not to_crawl:
        logger.info("No missing entrypoints -- catalog is up to date")
        return 0

    total_added = 0
    for i, entrypoint in enumerate(to_crawl, start=1):
        logger.info("Crawling entrypoint %d/%d: %s", i, len(to_crawl), entrypoint)
        try:
            remote_files = list(source.list_remote_files(entrypoint))
        except Exception:
            logger.exception("Error crawling entrypoint %s -- will retry next refresh", entrypoint)
            continue
        if remote_files:
            total_added += ledger.add_remote_files(
                remote_files, get_file_hash=source.get_file_hash, entrypoint_key=_entrypoint_key(entrypoint)
            )
        else:
            logger.warning("No files found for entrypoint: %s", entrypoint)
        ledger.mark_entrypoint_crawled(entrypoint)

    logger.info("Added %d new remote file(s)", total_added)
    return total_added

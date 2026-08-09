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

    Entrypoint-based sources (`has_entrypoints=True`: esacci/acag/ntl_harm/
    glass -- not EOG, which crawls one flat tree): `get_all_entrypoints()` is
    called on every refresh (a top-level directory listing, e.g. one crawl
    call per available year) -- deliberately *not* cached forever the way
    the old JSON side-file cache did, since caching it forever means a
    genuinely new entrypoint (e.g. this year's data becoming available)
    would never be discovered without an operator manually deleting the
    cache file. `entrypoints.crawled` tracks the actually-expensive part:
    only entrypoints not yet crawled get `list_remote_files(entrypoint)`
    called (a full recursive listing), and each is marked crawled once that
    call actually yields files -- a re-run only pays for genuinely new or
    still-empty entrypoints, unlike the old code's per-call full-column
    `filename_to_entrypoint()` re-scan to figure out what's missing. A
    zero-file result is deliberately left unmarked rather than assumed
    complete -- see the inline comment below.
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
            ledger.mark_entrypoint_crawled(entrypoint)
        else:
            # Deliberately NOT marked crawled -- a zero-file result is
            # indistinguishable here from a transient failure a source
            # swallowed internally instead of raising (e.g. ntl_harm's
            # `_get_figshare_files()` catches `requests.RequestException`
            # and returns `[]`). Marking crawled anyway would permanently
            # drop that entrypoint from every future refresh with no
            # visible error -- the same self-healing behavior the old
            # `UnifiedDataIndex._find_missing_entrypoints` had (it derived
            # "missing" from actually-indexed files, so a zero-result crawl
            # was retried automatically). Leaving it unmarked means a
            # genuinely-empty entrypoint (no data published yet) is
            # re-crawled every refresh too, but that's a bounded, cheap
            # re-listing cost -- silently and permanently losing a whole
            # entrypoint's files is not an acceptable trade against it.
            logger.warning("No files found for entrypoint: %s -- will retry next refresh", entrypoint)

    logger.info("Added %d new remote file(s)", total_added)
    return total_added

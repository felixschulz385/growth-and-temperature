"""Enumerate a FETCH-capable source's required remote files. Reuses each
source's existing `RemoteFileCatalog` surface
(`has_entrypoints`/`list_remote_files`/`get_all_entrypoints`/`get_file_hash`,
`src.data.sources.base`) completely unchanged.

Entrypoint-based sources (`has_entrypoints=True`: esacci/acag/ntl_harm/glass)
crawl one directory listing per entrypoint (e.g. one per year) -- expensive
enough that re-doing it on every `data fetch` call would be wasteful. Each
entrypoint's file listing is cached to a small JSON sidecar
(`_status/entrypoints/<key>.json`): an entrypoint already cached is never
re-crawled unless the caller explicitly asks for a refresh. A zero-file
crawl result is
deliberately NOT cached, so a transient failure a source swallows internally
(e.g. a 5xx a crawler treats as "nothing here") gets retried next call
instead of silently and permanently losing that entrypoint's files.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.data.common import statusfile
from src.data.common.fetch.manifest import RequiredFile

logger = logging.getLogger(__name__)

ENTRYPOINT_CACHE_SUBDIR = "_status/entrypoints"


def entrypoint_key(entrypoint: dict[str, Any]) -> str:
    """Stable dedup key for an entrypoint dict: `"<year>_<day>"` or `"<year>"`."""
    if "day" in entrypoint:
        return f"{entrypoint['year']}_{entrypoint['day']}"
    return str(entrypoint["year"])


def required_files(source: Any, status_dir: str, *, refresh_entrypoints: bool = False) -> list[RequiredFile]:
    """Every remote file *source* declares, as `RequiredFile`s. Simple
    (non-entrypoint) sources: one `list_remote_files()` call, fresh every
    time -- it's already just one listing, not a per-entrypoint crawl.
    Entrypoint sources: see module docstring for the caching behavior."""
    get_hash = source.get_file_hash

    if not getattr(source, "has_entrypoints", False):
        pairs = list(source.list_remote_files())
        return [RequiredFile(unit_id=get_hash(url), relative_path=rel, url=url) for rel, url in pairs]

    required: list[RequiredFile] = []
    for entrypoint in source.get_all_entrypoints():
        key = entrypoint_key(entrypoint)
        cache_path = statusfile.status_path(status_dir, key, subdir=ENTRYPOINT_CACHE_SUBDIR)
        cached = None if refresh_entrypoints else statusfile.read(cache_path)
        if cached is not None:
            pairs = cached["files"]
        else:
            try:
                pairs = list(source.list_remote_files(entrypoint))
            except Exception:
                logger.exception("Error crawling entrypoint %s -- will retry next run", entrypoint)
                continue
            if pairs:
                statusfile.write(cache_path, {"files": pairs})
            else:
                logger.warning("No files found for entrypoint %s -- will retry next run", entrypoint)

        for rel, url in pairs:
            required.append(RequiredFile(unit_id=get_hash(url), relative_path=rel, url=url))

    return required


def cached_required_files(source: Any, status_dir: str) -> "list[RequiredFile] | None":
    """Network-free variant for display-only callers (`data summary`):
    never calls `list_remote_files()`/`get_all_entrypoints()` -- both can
    themselves hit the network (GLASS's `get_all_entrypoints()` crawls its
    directory tree just to enumerate years), not just the per-entrypoint
    listing `required_files()` already caches. Only reads whatever's
    already on disk under `_status/entrypoints/`.

    Non-entrypoint sources (module docstring: `has_entrypoints=False` means
    "one `list_remote_files()` call, fresh every time", no cache at all) have
    nothing to read without a live call -- returns `None` for those, not
    `[]`, so a caller can tell "unknown without a live crawl" apart from
    "genuinely zero files required". Entrypoint sources: entrypoints never
    crawled by a real `data run --step fetch` yet are silently omitted, not
    reported as missing.
    """
    if not getattr(source, "has_entrypoints", False):
        return None

    get_hash = source.get_file_hash
    cache_dir = os.path.join(status_dir, ENTRYPOINT_CACHE_SUBDIR)
    try:
        filenames = os.listdir(cache_dir)
    except OSError:
        filenames = []

    required: list[RequiredFile] = []
    for filename in filenames:
        if not filename.endswith(".json"):
            continue
        cached = statusfile.read(os.path.join(cache_dir, filename))
        if not cached:
            continue
        for rel, url in cached.get("files", []):
            required.append(RequiredFile(unit_id=get_hash(url), relative_path=rel, url=url))
    return required

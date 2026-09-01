"""Enumerate a FETCH-capable source's required remote files. Reuses each
source's existing `RemoteFileCatalog` surface
(`has_entrypoints`/`list_remote_files`/`get_all_entrypoints`/`get_file_hash`,
`src.data.sources.base`) completely unchanged.

Entrypoint-based sources (`has_entrypoints=True`: esacci/acag/ntl_harm/eog)
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
from src.data.common.fetch import manifest
from src.data.common.fetch.manifest import RequiredFile

logger = logging.getLogger(__name__)

ENTRYPOINT_CACHE_SUBDIR = "_status/entrypoints"

#: Cache key for non-entrypoint sources' one-shot `list_remote_files()`
#: result -- same sidecar mechanism as a real entrypoint (module docstring),
#: just keyed by this constant instead of a year/day, so `cached_required_files()`
#: has something to read for these sources too instead of always returning
#: `None` (see that function's docstring).
_FLAT_CACHE_KEY = "_flat"


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

    if not source.has_entrypoints:
        pairs = list(source.list_remote_files())
        if pairs:
            cache_path = statusfile.status_path(status_dir, _FLAT_CACHE_KEY, subdir=ENTRYPOINT_CACHE_SUBDIR)
            statusfile.write(cache_path, {"files": pairs})
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
                fallback = statusfile.read(cache_path)
                if fallback is None:
                    continue
                pairs = fallback["files"]
            if pairs:
                statusfile.write(cache_path, {"files": pairs})
                manifest.clear_failure(status_dir, key)
            else:
                manifest.record_failure(status_dir, key, "no files listed for entrypoint")
                logger.warning("No files found for entrypoint %s -- will retry next run", entrypoint)

        for rel, url in pairs:
            required.append(RequiredFile(unit_id=get_hash(url), relative_path=rel, url=url))

    # A file legitimately shared by several entrypoints (e.g. one combined
    # multi-year archive covering a run of yearly entrypoints, as EOG's
    # 2012-2016 gas-flare survey does) must be fetched once, not once per
    # entrypoint -- duplicate RequiredFiles for the same local path
    # otherwise download concurrently and race on the same "<path>.part".
    seen: set[str] = set()
    deduped: list[RequiredFile] = []
    for req in required:
        if req.relative_path in seen:
            continue
        seen.add(req.relative_path)
        deduped.append(req)
    return deduped


def cached_required_files(source: Any, status_dir: str) -> "list[RequiredFile] | None":
    """Network-free variant for display-only callers (`data summary`):
    never calls `list_remote_files()`/`get_all_entrypoints()` -- both can
    themselves hit the network (GLASS's `get_all_entrypoints()` crawls its
    directory tree just to enumerate years), not just the per-entrypoint
    listing `required_files()` already caches. Only reads whatever's
    already on disk under `_status/entrypoints/`.

    Non-entrypoint sources (module docstring: `has_entrypoints=False` means
    "one `list_remote_files()` call, fresh every time") still get a cache --
    `required_files()` writes its listing to the same `_flat` sidecar every
    real run, same as an entrypoint's per-year cache. A source never run
    through a real `data run --step fetch` yet has no sidecar at all --
    returns `None` for those, not `[]`, so a caller can tell "unknown
    without a live crawl" apart from "genuinely zero files required".
    Entrypoint sources: entrypoints never crawled by a real
    `data run --step fetch` yet are silently omitted, not reported as
    missing.
    """
    get_hash = source.get_file_hash
    if not source.has_entrypoints:
        cache_path = statusfile.status_path(status_dir, _FLAT_CACHE_KEY, subdir=ENTRYPOINT_CACHE_SUBDIR)
        cached = statusfile.read(cache_path)
        if cached is None:
            return None
        return [RequiredFile(unit_id=get_hash(url), relative_path=rel, url=url) for rel, url in cached["files"]]

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


def cached_entrypoint_counts(
    source: Any, status_dir: str, listing: dict
) -> "tuple[int, int, int] | None":
    """Network-free complete/outstanding/unavailable counts at *entrypoint*
    granularity -- the fallback `_summarize_fetch()` (`src.cli.data.handlers`)
    uses to report the same three-bucket vocabulary as a real file-level
    crawl even before `required_files()` has ever populated
    `cached_required_files()`.

    `complete` comes from *listing* (a `manifest.snapshot_local_listing()`
    result) mapped back to entrypoints via `source.filename_to_entrypoint()`
    -- an entrypoint's actual files being present on disk, not merely that
    its remote listing was cached (that would call a not-yet-downloaded
    entrypoint "complete"). `unavailable` comes from each entrypoint's own
    status sidecar (`manifest.record_failure()`, keyed by
    `entrypoint_key()`). The full target list itself
    (`source.get_all_entrypoints()`) only gets called for sources that
    declare `STATIC_ENTRYPOINTS = True` -- i.e. verified network-free
    (esacci/acag/eog/ntl_harm are all `year_range`-derived); a future
    entrypoint source whose `get_all_entrypoints()` genuinely needs a live
    call (not declaring the flag) gets `None` here instead, letting the
    caller fall back further."""
    if not source.has_entrypoints or not getattr(source, "STATIC_ENTRYPOINTS", False):
        return None
    entrypoints = source.get_all_entrypoints()
    if not entrypoints:
        return None

    found_keys = set()
    for rel in listing:
        entrypoint = source.filename_to_entrypoint(rel)
        if entrypoint is not None:
            found_keys.add(entrypoint_key(entrypoint))

    status_filenames = statusfile.list_status_filenames(status_dir)

    complete = outstanding = unavailable = 0
    for entrypoint in entrypoints:
        key = entrypoint_key(entrypoint)
        if key in found_keys:
            complete += 1
            continue
        filename = f"{statusfile.sanitize_unit_id(key)}.json"
        if filename in status_filenames and (statusfile.read(statusfile.status_path(status_dir, key)) or {}).get(
            "status"
        ) == manifest.STATUS_UNAVAILABLE:
            unavailable += 1
        else:
            outstanding += 1
    return complete, outstanding, unavailable

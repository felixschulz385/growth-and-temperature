"""The FETCH driver. For a `transfer_mode=auto` source (`src.data.common
.fetch.transfer_mode`) -- one whose local copy is disposable because it gets
pushed to HPC right after FETCH -- this both *reads* the HPC target (to
decide what's already fetched, so a locally-pruned file doesn't look
outstanding forever) and *pushes* to it, immediately after each individual
file downloads successfully, not just once at the end of the run
(`_maybe_auto_transfer()`, `src/cli/data/handlers.py`, still runs too, as a
batched safety net for anything an individual push failed to deliver). A
`transfer_mode=manual` source (the default) stays exactly as before: purely
local-disk, never touching HPC at all; `data transfer`
(`src/cli/data/handlers.py`) is the only thing that pushes for those. A
source's `download:` config doesn't need `ledger_mode`/`ledger_push_every`/
`tar_max_files`/`tar_max_size_mb` (silently ignored via `**_ignored_config`
below, so a not-yet-cleaned-up config block doesn't error).

One pass: `catalog.required_files()` enumerates what a source needs (reusing
its existing `RemoteFileCatalog` crawl surface unchanged -- see that
module's docstring), diffed against one cached listing
(`manifest.resolve_fetch_listing()`/`manifest.plan_fetch()`, local or
remote per `transfer_mode`) into complete/outstanding/unavailable. Only
`outstanding` gets downloaded -- always to local disk, regardless of
`transfer_mode`; only the completeness check's source of truth changes.
`lockfile` guards against an accidental second invocation for the same
source while one is already running -- this codebase only ever runs one
FETCH worker per source at a time by design.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from src.data.common import lockfile, statusfile
from src.data.common.fetch import catalog, manifest
from src.data.common.fetch.transfer_mode import resolve_transfer_mode

logger = logging.getLogger(__name__)


async def _download_one(
    source: Any, req: manifest.RequiredFile, local_path: str, session: Any, semaphore: asyncio.Semaphore
) -> tuple[manifest.RequiredFile, bool, Optional[str]]:
    async with semaphore:
        # Atomic write: download to a ".part" file, os.replace() only on
        # success, so a killed-mid-download file never looks complete to
        # the next run's listing snapshot.
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        part_path = f"{local_path}.part"
        try:
            await source.download_async(req.url, part_path, session)
            os.replace(part_path, local_path)
            return req, True, None
        except Exception as exc:
            logger.error("Failed to download %s: %s", req.relative_path, exc)
            if os.path.exists(part_path):
                try:
                    os.remove(part_path)
                except OSError:
                    pass
            return req, False, str(exc)


async def _download_batch_async(
    source: Any, batch: list[manifest.RequiredFile], raw_root: str, max_concurrent: int
) -> list[tuple[manifest.RequiredFile, bool, Optional[str]]]:
    import aiohttp

    semaphore = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(limit=20, limit_per_host=10)
    timeout = aiohttp.ClientTimeout(total=300, connect=30)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            _download_one(source, req, os.path.join(raw_root, req.relative_path), session, semaphore)
            for req in batch
        ]
        return list(await asyncio.gather(*tasks))


def _download_batch(
    source: Any, batch: list[manifest.RequiredFile], raw_root: str, max_concurrent: int
) -> list[tuple[manifest.RequiredFile, bool, Optional[str]]]:
    """One event loop per batch, same tradeoff as before the ledger removal:
    a driver that's synchronous end-to-end rather than needing its own
    long-lived event loop, at the cost of reconnecting each batch (default
    batch_size=50 keeps that cheap relative to the actual downloads)."""
    return asyncio.run(_download_batch_async(source, batch, raw_root, max_concurrent))


def _push_one(pusher: Any, ctx: Any, raw_root: str, req: manifest.RequiredFile) -> None:
    """Push *req*'s just-downloaded file to HPC immediately -- the golden
    invariant every FETCH transfer path relies on (`DataSource
    .transfer_units()`'s docstring): a local FETCH path's remote counterpart
    is always that same path relative to `ctx.data_root`, mirrored under the
    HPC target's own base path. A push failure here is logged, not raised --
    the file did genuinely download; `_maybe_auto_transfer()`'s end-of-run
    batched pass (`src/cli/data/handlers.py`) or a manual `data transfer`
    will pick up anything missed."""
    from src.data.common.hpc.push import PushUnit

    local_path = os.path.join(raw_root, req.relative_path)
    remote_path = os.path.relpath(local_path, ctx.data_root).replace(os.sep, "/")
    result = pusher.push_unit(PushUnit(unit_id=req.unit_id, local_path=local_path, remote_path=remote_path))
    if not result.ok:
        logger.warning("Auto-transfer failed for %s: %s", req.relative_path, result.error)


def run_fetch(
    source: Any,
    *,
    batch_size: int = 50,
    max_concurrent_downloads: int = 5,
    max_attempts: int = manifest.DEFAULT_MAX_ATTEMPTS,
    refresh_entrypoints: bool = False,
    **_ignored_config: Any,
) -> bool:
    """Drive one full FETCH pass for *source*.

    `source` only needs to satisfy `RemoteFileCatalog`
    (`src.data.sources.base`) plus expose `.ctx`/`.cfg`/`.data_path`/
    `.output_root()` -- every FETCH-capable `DataSource` already does.
    Callers pass this straight
    through from `sources.<id>.download` config
    (`run_fetch(self, **self.cfg.raw.get("download", {}))`).

    Returns `True` iff every outstanding file downloaded successfully (no
    failures) -- units already `unavailable` (permanently given up on, see
    `manifest.record_failure`) don't count against this; an operator needs
    to fix the source config for those, not re-run fetch.
    """
    from src.data.sources.steps import PipelineStep

    ctx = source.ctx
    raw_root = source.output_root(PipelineStep.FETCH)
    os.makedirs(raw_root, exist_ok=True)
    source_id = getattr(source, "ID", "?")

    lock_path = os.path.join(raw_root, statusfile.STATUS_SUBDIR, "fetch.lock")
    try:
        lockfile.acquire(lock_path)
    except lockfile.LockHeldError as exc:
        logger.warning("Fetch already running for %s -- skipping this invocation: %s", source_id, exc)
        return False

    pusher = None
    if resolve_transfer_mode(source) == "auto" and ctx.ssh_target:
        from src.data.common.hpc.client import HPCClient
        from src.data.common.hpc.push import HPCPusher

        pusher = HPCPusher(HPCClient(target=ctx.ssh_target, key_file=ctx.key_file))

    try:
        required = catalog.required_files(source, raw_root, refresh_entrypoints=refresh_entrypoints)
        listing, from_remote = manifest.resolve_fetch_listing(source, raw_root)
        if from_remote:
            logger.info(
                "transfer_mode=auto for %s -- checking outstanding downloads against the HPC target instead of local disk",
                source_id,
            )
        plan = manifest.plan_fetch(required, listing, raw_root)

        if plan.unavailable:
            logger.warning(
                "%d unit(s) for %s marked unavailable -- fix the source URL/config to retry them",
                len(plan.unavailable), source_id,
            )
        if not plan.outstanding:
            logger.info("Fetch complete for %s: nothing outstanding", source_id)
            return True

        total_downloaded = total_failed = 0
        for i in range(0, len(plan.outstanding), batch_size):
            batch = plan.outstanding[i : i + batch_size]
            for req, ok, error in _download_batch(source, batch, raw_root, max_concurrent_downloads):
                if ok:
                    manifest.clear_failure(raw_root, req.unit_id)
                    total_downloaded += 1
                    if pusher is not None:
                        _push_one(pusher, ctx, raw_root, req)
                else:
                    manifest.record_failure(raw_root, req.unit_id, error or "unknown error", max_attempts=max_attempts)
                    total_failed += 1

        logger.info(
            "Fetch complete for %s: %d downloaded, %d failed", source_id, total_downloaded, total_failed,
        )
        return total_failed == 0
    finally:
        lockfile.release(lock_path)

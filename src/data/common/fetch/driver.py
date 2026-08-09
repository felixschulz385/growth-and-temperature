"""The FETCH driver: replaces `common/fetch/async_downloader.py`
(`run_async_download_workflow`/`AsyncHPCDownloader`).

docs/design/10-fetch-ledger.md §2/§6. Drives, per invocation: pull+merge the
source's DuckDB ledger, crawl for new remote files (`common.ledger.catalog`),
then loop downloading pending files and pushing each batch to HPC
(`common.hpc.push.HPCPusher`) until nothing's left pending.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from src.data.common.hpc.push import HPCPusher, PushUnit
from src.data.common.ledger import catalog
from src.data.common.ledger.paths import ledger_path
from src.data.common.ledger.store import DownloadResult
from src.data.common.ledger.store import SourceLedger

logger = logging.getLogger(__name__)


async def _download_one(source: Any, unit, local_path: str, session, semaphore: asyncio.Semaphore) -> DownloadResult:
    async with semaphore:
        # Atomic write (fixes the old downloader's direct-to-final-path
        # writes): download to a ".part" file, os.replace() only on success,
        # so a killed-mid-download file never looks complete.
        part_path = f"{local_path}.part"
        try:
            await source.download_async(unit.source_url, part_path, session)
            os.replace(part_path, local_path)
            return DownloadResult(file_hash=unit.file_hash, ok=True, local_path=local_path, bytes=os.path.getsize(local_path))
        except Exception as exc:
            logger.error("Failed to download %s: %s", unit.relative_path, exc)
            if os.path.exists(part_path):
                try:
                    os.remove(part_path)
                except OSError:
                    pass
            return DownloadResult(file_hash=unit.file_hash, ok=False, error=str(exc))


async def _download_batch_async(
    source: Any, pending: list, staging_dir: str, max_concurrent: int
) -> list[DownloadResult]:
    import aiohttp

    semaphore = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(limit=20, limit_per_host=10)
    timeout = aiohttp.ClientTimeout(total=300, connect=30)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for unit in pending:
            local_path = os.path.join(staging_dir, f"{unit.file_hash}_{os.path.basename(unit.relative_path)}")
            tasks.append(_download_one(source, unit, local_path, session, semaphore))
        return list(await asyncio.gather(*tasks))


def _download_batch(source: Any, pending: list, staging_dir: str, max_concurrent: int) -> list[DownloadResult]:
    """One event loop per batch -- a small tradeoff against the old
    `AsyncHPCDownloader`'s single aiohttp session reused across the whole
    run, in exchange for a driver that's synchronous end-to-end (matching
    every other step's `_execute_*`) rather than needing its own long-lived
    event loop. Connection pooling still applies *within* a batch, which is
    where concurrency actually matters (default batch_size=50,
    max_concurrent_downloads=5)."""
    return asyncio.run(_download_batch_async(source, pending, staging_dir, max_concurrent))


def run_fetch(
    source: Any,
    *,
    batch_size: int = 50,
    max_concurrent_downloads: int = 5,
    tar_max_files: int = 100,
    tar_max_size_mb: int = 500,
    ledger_push_every: int = 10,
    **_ignored_config: Any,
) -> bool:
    """Drive one full FETCH pass for *source*.

    `source` only needs to satisfy `RemoteFileCatalog`
    (`src.data.sources.base`) plus expose `.ctx`/`.cfg`/`.data_path` -- every
    FETCH-capable `DataSource` already does. Callers pass this straight
    through from `sources.<id>.download` config
    (`run_fetch(self, **self.cfg.raw.get("download", {}))`); `**_ignored_config`
    absorbs any keys this driver doesn't recognize rather than erroring.

    Returns `True` iff every pending file was downloaded and verified on
    HPC (no failures) -- `False` on any failure or missing configuration,
    matching the old `_execute_fetch` boolean contract.
    """
    ctx = source.ctx
    if not ctx.ssh_target:
        logger.warning("Fetch requires an HPC/remote target to be configured.")
        return False

    from src.data.common.hpc.client import HPCClient

    client = HPCClient(target=ctx.ssh_target, key_file=ctx.key_file)
    return run_fetch_with_client(
        source, client,
        batch_size=batch_size, max_concurrent_downloads=max_concurrent_downloads,
        tar_max_files=tar_max_files, tar_max_size_mb=tar_max_size_mb, ledger_push_every=ledger_push_every,
    )


def run_fetch_with_client(
    source: Any,
    client: Any,
    *,
    batch_size: int = 50,
    max_concurrent_downloads: int = 5,
    tar_max_files: int = 100,
    tar_max_size_mb: int = 500,
    ledger_push_every: int = 10,
) -> bool:
    """`run_fetch()`'s actual implementation, taking an already-constructed
    HPC client -- the seam tests use to inject a fake client. `run_fetch()`
    itself is just this plus real `HPCClient` construction from `ctx`."""
    ctx = source.ctx
    from src.data.sources import layout

    local_ledger_path = ledger_path(ctx.local_index_dir, source.data_path)
    if local_ledger_path is None:
        logger.warning("paths.local_index_dir is not configured -- cannot fetch without a ledger location.")
        return False

    # NOTE: source.cfg.data_path/source.cfg.namespace (the resolved,
    # post-__init__-default config), not source.data_path -- that property
    # is overridden by the misc-split sources (gadm/osm/country_classifications)
    # to a combined "<data_path>/<namespace>" string purely for ledger-file
    # naming (see base.py's DataSource.data_path docstring), which would
    # double up the namespace segment if fed into raw_root() alongside
    # namespace= too.
    raw_root = layout.raw_root("", source.cfg.data_path, namespace=source.cfg.namespace, layout=ctx.layout)
    staging_dir = os.path.join(ctx.staging_dir or os.path.dirname(local_ledger_path), "fetch_staging")
    os.makedirs(staging_dir, exist_ok=True)

    pusher = HPCPusher(client)
    total_downloaded = total_failed = 0

    with SourceLedger.open(local_ledger_path, data_path=source.data_path) as ledger:
        ledger.merge_from_remote(client, os.path.join(staging_dir, "ledger_merge_tmp"))

        added = catalog.refresh(ledger, source)
        if added:
            logger.info("Discovered %d new remote file(s)", added)

        batches_since_push = 0
        while True:
            pending = ledger.pending_fetch(batch_size)
            if not pending:
                break
            pending_by_hash = {u.file_hash: u for u in pending}

            results = _download_batch(source, pending, staging_dir, max_concurrent_downloads)
            ledger.record_download_batch(results)

            ok_results = [r for r in results if r.ok]
            total_downloaded += len(ok_results)
            total_failed += len(results) - len(ok_results)

            if ok_results:
                push_units = [
                    PushUnit(
                        unit_id=r.file_hash,
                        local_path=r.local_path,
                        remote_path=pending_by_hash[r.file_hash].relative_path,
                    )
                    for r in ok_results
                ]
                push_results = pusher.push_batched(
                    push_units, raw_root, max_files=tar_max_files, max_bytes=tar_max_size_mb * 1024 * 1024,
                )
                ledger.record_push_batch("fetch", push_results)

            batches_since_push += 1
            if batches_since_push >= ledger_push_every:
                ledger.push_to_remote(client)
                batches_since_push = 0

        ledger.push_to_remote(client)

    logger.info(
        "Fetch complete for %s: %d downloaded, %d failed", getattr(source, "ID", "?"), total_downloaded, total_failed
    )
    return total_failed == 0

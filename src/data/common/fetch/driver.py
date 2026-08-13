"""The FETCH driver: replaces `common/fetch/async_downloader.py`
(`run_async_download_workflow`/`AsyncHPCDownloader`).

docs/design/10-fetch-ledger.md §2/§6. Two `ledger_mode`s, chosen via
`--ledger` on `data run --step fetch` (src/cli/data/commands.py) and threaded
down through each source's `sources.<id>.download` config block:

- `"local"` (default): download-only. Pull+merge the source's ledger, crawl
  for new remote files (`common.ledger.catalog`), download whatever isn't
  locally complete yet. Never pushes -- that's `data transfer`'s job now,
  for every source (this used to be MODIS's own special case; every other
  source fused fetch+push into one loop, which this mode un-fuses).
- `"remote"`: push-immediately backlog-clearing. Skips the origin crawl
  entirely and instead reads a read-only, remote-pulled copy of the ledger
  for its worklist -- "whatever the remote side still doesn't have" --
  downloading each file from the source's real origin (not from HPC) and
  pushing it back out right away. For a machine that already has someone
  else's crawl results on HPC and wants to clear exactly what's still
  outstanding, without needing to reach the origin's own listing/search API.

Neither mode holds one ledger connection open for the whole run. DuckDB
allows exactly one read-write connection to a `.duckdb` file at a time,
across processes (`SourceLedger`'s own docstring) -- the download loop's
actual work (network I/O) happens with NO connection open at all, and each
ledger touch (read the next pending batch, record results) is its own brief
`open_with_retry()`/close. This is `ModisSource._ledger_ensure_artifact()`'s
existing pattern, generalized to every other FETCH-capable source: without
it, a multi-minute-to-hour fetch run would hold the lock continuously,
starving a concurrently-running `data transfer --watch` (or `data plan`/
`data summary`) from ever getting a turn until the whole run finishes --
confirmed empirically (a held read-write connection makes both a second
read-write open *and* a read-only open from another process fail
immediately, not block-and-wait). `catalog.refresh()` (the one-time crawl at
the start of `--ledger local` mode) is the one exception -- it still holds
its own connection through its network calls, a smaller, separate
limitation not addressed here (it's a shared module also used by `data
index`/`data reconcile`).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from src.data.common.hpc.push import HPCPusher, PushUnit
from src.data.common.ledger import catalog
from src.data.common.ledger.paths import ledger_path
from src.data.common.ledger.schema import LocalState
from src.data.common.ledger.store import DownloadResult
from src.data.common.ledger.store import SourceLedger

logger = logging.getLogger(__name__)

#: Upper bound on one `--ledger remote` run's single `pending_fetch()` read
#: of the whole backlog (`_run_fetch_remote_backlog`, below). A real
#: per-source backlog is thousands of files at most, not millions -- this
#: exists as a sanity ceiling, not a real limit any source is expected to hit.
_MAX_REMOTE_BACKLOG = 1_000_000


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
    max_concurrent_downloads=5). No ledger connection is open during this
    call -- the real reason batches use brief, separate ledger opens rather
    than one held for the whole run: this is where all the *time* goes."""
    return asyncio.run(_download_batch_async(source, pending, staging_dir, max_concurrent))


def _push_ledger_to_remote(local_ledger_path: str, data_path: str, client: Any) -> bool:
    """Brief acquire-close-then-push: `SourceLedger.push_to_remote()` only
    ever reads `local_path`/`data_path` (plain attributes, never the DB
    connection itself), so the lock only needs to be held long enough to
    construct the instance -- not for the rsync, which can take real time
    for a large ledger file."""
    ledger = SourceLedger.open_with_retry(local_ledger_path, data_path=data_path)
    ledger.close()
    return ledger.push_to_remote(client)


def run_fetch(
    source: Any,
    *,
    batch_size: int = 50,
    max_concurrent_downloads: int = 5,
    tar_max_files: int = 100,
    tar_max_size_mb: int = 500,
    ledger_push_every: int = 10,
    ledger_mode: str = "local",
    **_ignored_config: Any,
) -> bool:
    """Drive one full FETCH pass for *source*.

    `source` only needs to satisfy `RemoteFileCatalog`
    (`src.data.sources.base`) plus expose `.ctx`/`.cfg`/`.data_path` -- every
    FETCH-capable `DataSource` already does. Callers pass this straight
    through from `sources.<id>.download` config
    (`run_fetch(self, **self.cfg.raw.get("download", {}))`); `**_ignored_config`
    absorbs any keys this driver doesn't recognize rather than erroring.
    `ledger_mode` is injected into that same config dict by `handle_run`
    (src/cli/data/handlers.py) when `--ledger remote` is passed -- see this
    module's own docstring for what each mode does.

    Returns `True` iff every pending file was downloaded (and, in
    `ledger_mode="remote"`, verified on HPC) with no failures -- `False` on
    any failure or missing configuration, matching the old `_execute_fetch`
    boolean contract.
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
        ledger_mode=ledger_mode,
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
    ledger_mode: str = "local",
) -> bool:
    """`run_fetch()`'s actual implementation, taking an already-constructed
    HPC client -- the seam tests use to inject a fake client. `run_fetch()`
    itself is just this plus real `HPCClient` construction from `ctx`."""
    if ledger_mode not in ("local", "remote"):
        raise ValueError(f"Unknown ledger_mode: {ledger_mode!r} -- must be 'local' or 'remote'")

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

    if ledger_mode == "local":
        ok = _run_fetch_local(source, client, local_ledger_path, staging_dir, batch_size, max_concurrent_downloads)
    else:
        ok = _run_fetch_remote_backlog(
            source, client, local_ledger_path, staging_dir, raw_root, batch_size, max_concurrent_downloads,
            tar_max_files, tar_max_size_mb, ledger_push_every,
        )
        # `ledger_mode="local"` never pushes, so there's nothing to sync out
        # in that mode -- only "remote" needs this final push.
        _push_ledger_to_remote(local_ledger_path, source.data_path, client)

    return ok


def _run_fetch_local(
    source: Any, client: Any, local_ledger_path: str, staging_dir: str, batch_size: int, max_concurrent_downloads: int,
) -> bool:
    """Download-only: crawl the source's own origin, download whatever isn't
    locally complete yet, record `local_state`. Never pushes -- push
    separately via `data transfer --step fetch`.

    Each ledger access is its own brief `open_with_retry()`/close, not one
    connection held for the whole run -- see this module's own docstring."""
    with SourceLedger.open_with_retry(local_ledger_path, data_path=source.data_path) as ledger:
        ledger.merge_from_remote(client, os.path.join(staging_dir, "ledger_merge_tmp"))
        added = catalog.refresh(ledger, source)
    if added:
        logger.info("Discovered %d new remote file(s)", added)

    total_downloaded = total_failed = 0
    while True:
        with SourceLedger.open_with_retry(local_ledger_path, data_path=source.data_path) as ledger:
            pending = ledger.pending_download(batch_size)
        if not pending:
            break

        results = _download_batch(source, pending, staging_dir, max_concurrent_downloads)

        with SourceLedger.open_with_retry(local_ledger_path, data_path=source.data_path) as ledger:
            ledger.record_download_batch(results)

        ok_results = [r for r in results if r.ok]
        total_downloaded += len(ok_results)
        total_failed += len(results) - len(ok_results)

    logger.info(
        "Fetch (download-only) complete for %s: %d downloaded, %d failed",
        getattr(source, "ID", "?"), total_downloaded, total_failed,
    )
    return total_failed == 0


def _run_fetch_remote_backlog(
    source: Any, client: Any, local_ledger_path: str, staging_dir: str, raw_root: str, batch_size: int,
    max_concurrent_downloads: int, tar_max_files: int, tar_max_size_mb: int, ledger_push_every: int,
) -> bool:
    """`--ledger remote`: skip the origin crawl, read a remote-pulled,
    read-only ledger snapshot for the worklist (`pending_fetch()` --
    unchanged, "not yet HPC-verified"), download each file from the source's
    real origin, and push it back immediately -- today's original
    (pre-split) fused fetch+push loop, just fed from a different, read-only
    worklist connection instead of this run's own local/crawled one.

    Each *local* ledger access is its own brief `open_with_retry()`/close,
    same reasoning as `_run_fetch_local`'s. `remote_ledger` (the pulled
    snapshot) is a private temp file this run alone holds -- no contention
    concern there, so it's opened once and kept for the whole call."""
    # Seeds this run's own local `artifacts`/`remote_files` rows from the
    # remote ledger first -- required, not just a freshness nicety, so the
    # `record_download_batch`/`record_push_batch` UPDATEs below (keyed on an
    # already-existing row) have something to update even on a machine that
    # has never crawled this source locally before.
    with SourceLedger.open_with_retry(local_ledger_path, data_path=source.data_path) as ledger:
        ledger.merge_from_remote(client, os.path.join(staging_dir, "ledger_merge_tmp"))

    remote_tmp_path = os.path.join(staging_dir, "remote_ledger_readonly.duckdb")
    remote_ledger: Optional[SourceLedger] = SourceLedger.pull_remote_readonly(client, source.data_path, remote_tmp_path)
    if remote_ledger is None:
        logger.warning(
            "--ledger remote: no remote ledger found yet for %s -- nothing to clear", source.data_path,
        )
        return False

    pusher = HPCPusher(client)
    total_downloaded = total_failed = 0
    batches_since_push = 0
    try:
        # A single up-front read of the whole backlog, chunked into batches
        # locally -- NOT a `while True: pending = remote_ledger.pending_fetch(...)`
        # loop like `_run_fetch_local`'s. `remote_ledger` is a static,
        # read-only snapshot (this run's own progress is recorded into the
        # separate local ledger, never back into it), so re-querying it
        # would keep returning the exact same rows forever -- confirmed by
        # an actual infinite loop while writing this. `_MAX_REMOTE_BACKLOG`
        # bounds the single query; a real per-source backlog is thousands of
        # files at most, not millions.
        backlog = remote_ledger.pending_fetch(_MAX_REMOTE_BACKLOG)
        for i in range(0, len(backlog), batch_size):
            pending = backlog[i : i + batch_size]
            pending_by_hash = {u.file_hash: u for u in pending}

            results = _download_batch(source, pending, staging_dir, max_concurrent_downloads)

            with SourceLedger.open_with_retry(local_ledger_path, data_path=source.data_path) as ledger:
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
                with SourceLedger.open_with_retry(local_ledger_path, data_path=source.data_path) as ledger:
                    ledger.record_push_batch("fetch", push_results)
                    # HPCPusher's cleanup_local=True (the default push_batched()
                    # always uses here) already deleted each pushed file's local
                    # copy -- reflect that in local_state too, so a later
                    # `pending_download()`/`fetch_transfer_units()` read doesn't
                    # keep believing it's still on disk. remote_state='verified'
                    # + local_state='missing' stays fully distinguishable from
                    # "never fetched" (remote_state='missing' too).
                    ledger.set_local_states_batch(
                        "fetch", [(r.unit_id, LocalState.MISSING) for r in push_results if r.ok]
                    )

            batches_since_push += 1
            if batches_since_push >= ledger_push_every:
                _push_ledger_to_remote(local_ledger_path, source.data_path, client)
                batches_since_push = 0
    finally:
        remote_ledger.close()
        try:
            os.remove(remote_tmp_path)
        except OSError:
            pass

    # Final reconcile before the caller's own push -- picks up anything
    # another machine pushed concurrently while this backlog-clearing run
    # was in progress.
    with SourceLedger.open_with_retry(local_ledger_path, data_path=source.data_path) as ledger:
        ledger.merge_from_remote(client, os.path.join(staging_dir, "ledger_merge_tmp"))

    logger.info(
        "Fetch (remote-ledger backlog) complete for %s: %d downloaded, %d failed",
        getattr(source, "ID", "?"), total_downloaded, total_failed,
    )
    return total_failed == 0

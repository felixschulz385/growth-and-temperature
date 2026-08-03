"""One unified push-to-HPC primitive, replacing both FETCH's inline
tar/rsync/extract/verify (`common/fetch/async_downloader.py`) and the
separate `pipeline transfer` path (`common/hpc/transfer.py`).

docs/design/10-fetch-ledger.md §2. Built entirely on the existing, unchanged
`HPCClient` (`ensure_directory`, `rsync_transfer`, `extract_tar`,
`check_files_exist`, `execute_command`). Two batching strategies over the
same push+verify+cleanup shape:

- `push_batched()` -- FETCH's many-small-files case: pack a batch into one
  tar, one rsync + remote extract + sample-verify round trip per batch.
- `push_unit()` -- PREPARE/GRID transfer's few-large-artifacts case: a
  directory (e.g. a Zarr store) is tarred and extracted the same way; a
  single file is rsynced directly, no tar/extract needed for one file.
- `push_units_concurrent()` -- thread-pooled `push_unit()` calls, replacing
  `transfer_units()`'s old serial loop (rsync/ssh are blocking subprocess
  I/O, so a small `ThreadPoolExecutor` is the right primitive here, not
  asyncio).

Verification stays sample-based (a handful of files via one batched
`check_files_exist` round trip), not full checksums -- rsync already
guarantees byte-level transfer integrity; the actual risk worth guarding is a
failed/partial *remote extraction*, which sample existence-checking catches.
"""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

_RSYNC_OPTIONS = {"compress": True, "archive": True, "partial": True, "verbose": False}


@dataclass(frozen=True)
class PushUnit:
    """One thing to push to HPC.

    `push_unit()`/`push_units_concurrent()` (few-large-artifacts): `remote_path`
    is the *full* remote destination path (relative to the HPC target's base
    path) -- matches `transfer.py`'s old `{unit_id, local_path, remote_path}`
    shape exactly.

    `push_batched()` (many-small-files): `remote_path` is instead the tar
    *arcname*, relative to that call's own `remote_base_dir` -- multiple
    units in one batch share one extraction root, mirroring
    `async_downloader.py`'s old per-file `relative_path` behavior.
    """

    unit_id: str
    local_path: str
    remote_path: str


@dataclass(frozen=True)
class PushResult:
    unit_id: str
    ok: bool
    bytes: Optional[int] = None
    error: Optional[str] = None


def _full_remote_path(client: Any, remote_path: str) -> str:
    base_path = getattr(client, "base_path", None)
    if remote_path.startswith("/") or not base_path:
        return remote_path
    return f"{base_path}/{remote_path}"


def _cleanup_local(path: str) -> None:
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
    except OSError:
        logger.warning("Could not clean up local path %s", path, exc_info=True)


class HPCPusher:
    def __init__(self, client: Any, *, sample_verify_n: int = 5):
        self.client = client
        self.sample_verify_n = sample_verify_n

    # ------------------------------------------------------------------
    # FETCH: many-small-files, tar-batched
    # ------------------------------------------------------------------

    def push_batched(
        self,
        units: list[PushUnit],
        remote_base_dir: str,
        *,
        max_files: int = 100,
        max_bytes: int = 500 * 1024 * 1024,
        cleanup_local: bool = True,
    ) -> list[PushResult]:
        """Push *units* to `remote_base_dir` in one or more tar batches,
        split on *max_files*/*max_bytes*. Every unit in a batch succeeds or
        fails together (matches the old `AsyncHPCDownloader`'s per-batch,
        not per-file, status update)."""
        results: list[PushResult] = []
        for batch in self._split_batches(units, max_files, max_bytes):
            results.extend(self._push_one_batch(batch, remote_base_dir, cleanup_local=cleanup_local))
        return results

    @staticmethod
    def _split_batches(units: list[PushUnit], max_files: int, max_bytes: int) -> list[list[PushUnit]]:
        batches: list[list[PushUnit]] = []
        current: list[PushUnit] = []
        current_bytes = 0
        for unit in units:
            try:
                size = os.path.getsize(unit.local_path)
            except OSError:
                size = 0
            # Always keep at least one unit per batch, even if it alone
            # exceeds max_bytes, so an oversized single file doesn't stall
            # the loop forever.
            if current and (len(current) >= max_files or current_bytes + size > max_bytes):
                batches.append(current)
                current, current_bytes = [], 0
            current.append(unit)
            current_bytes += size
        if current:
            batches.append(current)
        return batches

    def _push_one_batch(
        self, batch: list[PushUnit], remote_base_dir: str, *, cleanup_local: bool
    ) -> list[PushResult]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        tar_dir = tempfile.mkdtemp(prefix="hpc_push_")
        tar_path = os.path.join(tar_dir, f"batch_{timestamp}_{len(batch)}.tar.gz")

        try:
            sizes: dict[str, int] = {}
            with tarfile.open(tar_path, "w:gz") as tar:
                for unit in batch:
                    tar.add(unit.local_path, arcname=unit.remote_path)
                    try:
                        sizes[unit.unit_id] = os.path.getsize(unit.local_path)
                    except OSError:
                        sizes[unit.unit_id] = 0

            remote_tar_dir = f"{remote_base_dir.rstrip('/')}/_tar"
            remote_tar_path = f"{remote_tar_dir}/{os.path.basename(tar_path)}"
            self.client.ensure_directory(remote_tar_dir)
            self.client.ensure_directory(remote_base_dir)

            success, summary = self.client.rsync_transfer(
                tar_path, remote_tar_path, source_is_local=True, options=_RSYNC_OPTIONS, show_progress=False,
            )
            if not success:
                logger.error("Batch transfer failed: %s", summary)
                return [PushResult(unit_id=u.unit_id, ok=False, error=summary) for u in batch]

            if not self.client.extract_tar(remote_tar_path, remote_base_dir):
                logger.error("Batch extraction failed for %s", remote_tar_path)
                return [PushResult(unit_id=u.unit_id, ok=False, error="extraction failed") for u in batch]

            sample = batch[: self.sample_verify_n]
            sample_remote_paths = [f"{remote_base_dir.rstrip('/')}/{u.remote_path}" for u in sample]
            existence = self.client.check_files_exist(sample_remote_paths)
            if not all(existence.values()):
                missing = [p for p, ok in existence.items() if not ok]
                logger.error("Batch verification failed -- missing on HPC: %s", missing)
                return [PushResult(unit_id=u.unit_id, ok=False, error="verification failed") for u in batch]

            full_remote_tar_path = _full_remote_path(self.client, remote_tar_path)
            self.client.execute_command(f"rm -f '{full_remote_tar_path}'")

            if cleanup_local:
                for unit in batch:
                    _cleanup_local(unit.local_path)

            return [PushResult(unit_id=u.unit_id, ok=True, bytes=sizes.get(u.unit_id)) for u in batch]
        except Exception as exc:
            logger.exception("Error pushing batch")
            return [PushResult(unit_id=u.unit_id, ok=False, error=str(exc)) for u in batch]
        finally:
            shutil.rmtree(tar_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # PREPARE/GRID transfer: few large artifacts, one unit at a time
    # ------------------------------------------------------------------

    def push_unit(self, unit: PushUnit, *, cleanup_local: bool = True) -> PushResult:
        """Push one artifact. `unit.local_path` a directory (e.g. a Zarr
        store): tar -> rsync -> remote extract -> verify, amortizing
        per-chunk rsync overhead for a tree of many small files.
        `unit.local_path` a single file: direct rsync -> verify, no
        tar/extract -- there's only one file, so the overhead tar exists to
        amortize doesn't apply."""
        if not os.path.exists(unit.local_path):
            return PushResult(unit_id=unit.unit_id, ok=False, error=f"local path does not exist: {unit.local_path}")

        if os.path.isfile(unit.local_path):
            return self._push_single_file(unit, cleanup_local=cleanup_local)
        return self._push_directory(unit, cleanup_local=cleanup_local)

    def push_units_concurrent(
        self, units: Iterable[PushUnit], *, max_workers: int = 4, cleanup_local: bool = True
    ) -> list[PushResult]:
        """Thread-pooled `push_unit()` calls -- rsync/ssh are blocking
        subprocess I/O, so threads (not asyncio) parallelize them."""
        units = list(units)
        if not units:
            return []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(lambda u: self.push_unit(u, cleanup_local=cleanup_local), units))

    def _push_single_file(self, unit: PushUnit, *, cleanup_local: bool) -> PushResult:
        try:
            size = os.path.getsize(unit.local_path)
        except OSError:
            size = None

        remote_parent = os.path.dirname(unit.remote_path.rstrip("/"))
        if remote_parent:
            self.client.ensure_directory(remote_parent)

        success, summary = self.client.rsync_transfer(
            unit.local_path, unit.remote_path, source_is_local=True, options=_RSYNC_OPTIONS, show_progress=False,
        )
        if not success:
            logger.error("Transfer failed for %s: %s", unit.unit_id, summary)
            return PushResult(unit_id=unit.unit_id, ok=False, error=summary)

        full_remote_path = _full_remote_path(self.client, unit.remote_path)
        if not self.client.check_file_exists(full_remote_path):
            logger.error("Verification failed for %s: %s not found on HPC", unit.unit_id, full_remote_path)
            return PushResult(unit_id=unit.unit_id, ok=False, error="verification failed")

        if cleanup_local:
            _cleanup_local(unit.local_path)
        return PushResult(unit_id=unit.unit_id, ok=True, bytes=size)

    def _push_directory(self, unit: PushUnit, *, cleanup_local: bool) -> PushResult:
        tar_dir = tempfile.mkdtemp(prefix="hpc_push_")
        tar_filename = f"{unit.unit_id.replace('/', '_')}.tar.gz"
        tar_path = os.path.join(tar_dir, tar_filename)

        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(unit.local_path, arcname=os.path.basename(os.path.normpath(unit.local_path)))
            try:
                size = os.path.getsize(tar_path)
            except OSError:
                size = None

            remote_parent = os.path.dirname(unit.remote_path.rstrip("/"))
            remote_tar_path = f"{remote_parent}/{tar_filename}" if remote_parent else tar_filename
            if remote_parent:
                self.client.ensure_directory(remote_parent)

            success, summary = self.client.rsync_transfer(
                tar_path, remote_tar_path, source_is_local=True, options=_RSYNC_OPTIONS, show_progress=False,
            )
            if not success:
                logger.error("Transfer failed for %s: %s", unit.unit_id, summary)
                return PushResult(unit_id=unit.unit_id, ok=False, error=summary)

            extract_dir = remote_parent or "."
            if not self.client.extract_tar(remote_tar_path, extract_dir):
                logger.error("Extraction failed for %s", unit.unit_id)
                return PushResult(unit_id=unit.unit_id, ok=False, error="extraction failed")

            if not self._verify_remote_dir(unit.remote_path):
                logger.error("Verification failed for %s", unit.unit_id)
                return PushResult(unit_id=unit.unit_id, ok=False, error="verification failed")

            full_remote_tar_path = _full_remote_path(self.client, remote_tar_path)
            self.client.execute_command(f"rm -f '{full_remote_tar_path}'")

            if cleanup_local:
                _cleanup_local(unit.local_path)
            return PushResult(unit_id=unit.unit_id, ok=True, bytes=size)
        except Exception as exc:
            logger.exception("Error pushing unit %s", unit.unit_id)
            return PushResult(unit_id=unit.unit_id, ok=False, error=str(exc))
        finally:
            shutil.rmtree(tar_dir, ignore_errors=True)

    def _verify_remote_dir(self, remote_path: str) -> bool:
        """Sample a few files under the transferred+extracted remote
        directory (one `find` + one batched `check_files_exist`)."""
        full_remote_path = _full_remote_path(self.client, remote_path)
        success, stdout, _ = self.client.execute_command(
            f"find '{full_remote_path}' -type f | head -n {self.sample_verify_n}"
        )
        if not success or not stdout.strip():
            return False
        sample_files = [line for line in stdout.strip().splitlines() if line]
        if not sample_files:
            return False
        results = self.client.check_files_exist(sample_files)
        return all(results.values())

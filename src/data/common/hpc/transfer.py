"""
Generic HPC transfer capability: push a preprocess stage's local output to the
configured HPC target over SSH.

docs/design/08-hpc-transfer.md — a thin orchestration layer over the existing
`HPCClient` (tar -> rsync -> remote extract -> verify), mirroring the pattern
already used for raw downloads in `src/data/download/async_downloader.py`
(`_create_tar_archive`/`_transfer_and_extract`/`_verify_extracted_files`).
"""

import logging
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.common.hpc.client import HPCClient

logger = logging.getLogger(__name__)

STATUS_PENDING = "pending"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class TransferManifest:
    """Per-unit transfer status, parquet-backed.

    `UnifiedDataIndex` (src/data/common/index/unified_index.py) is schema-
    locked to the download subsystem's shape (file_hash/source_url/
    relative_path, entrypoint-driven index building) — repurposing it for
    generic transfer units would need more than a namespace convention. This
    is a small, purpose-built manifest using the same underlying storage
    technology (parquet) rather than a second index format, per
    docs/design/08-hpc-transfer.md §3.
    """

    _COLUMNS = [
        "unit_id", "local_path", "remote_path", "status", "attempts",
        "last_error", "updated_at",
    ]

    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self):
        import pandas as pd
        if self.manifest_path.exists():
            return pd.read_parquet(self.manifest_path)
        return pd.DataFrame(columns=self._COLUMNS)

    def _save(self, df) -> None:
        df.to_parquet(self.manifest_path, index=False)

    def status_of(self, unit_id: str) -> Optional[str]:
        df = self._load()
        rows = df[df["unit_id"] == unit_id]
        if rows.empty:
            return None
        return rows.iloc[-1]["status"]

    def record(
        self, unit_id: str, local_path: str, remote_path: str, status: str,
        error: Optional[str] = None,
    ) -> None:
        import pandas as pd
        df = self._load()
        mask = df["unit_id"] == unit_id
        attempts = int(df.loc[mask, "attempts"].iloc[-1]) + 1 if mask.any() else 1
        df = df[~mask]
        new_row = pd.DataFrame([{
            "unit_id": unit_id,
            "local_path": local_path,
            "remote_path": remote_path,
            "status": status,
            "attempts": attempts,
            "last_error": error or "",
            "updated_at": pd.Timestamp.now(),
        }])
        self._save(pd.concat([df, new_row], ignore_index=True))


def _tar_directory(local_path: str, tar_path: str) -> None:
    """Tar a local directory tree (e.g. a Zarr store's chunk tree).

    Deliberate choice, not arbitrary: a Zarr store is a directory of
    thousands of small chunk files, and plain `rsync -a` on that tree pays
    per-file SSH/rsync protocol overhead per chunk. Tar-then-transfer-then-
    untar amortizes that into one file transfer — the same problem
    `async_downloader.py`'s `_create_tar_archive` already solves for raw
    downloaded files.
    """
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(local_path, arcname=os.path.basename(os.path.normpath(local_path)))


def _full_remote_path(client: HPCClient, remote_path: str) -> str:
    if remote_path.startswith("/") or not client.base_path:
        return remote_path
    return f"{client.base_path}/{remote_path}"


def _verify_remote(client: HPCClient, remote_path: str, sample_size: int = 5) -> bool:
    """Sample a few files under the transferred+extracted remote path.

    Mirrors `async_downloader.py`'s `_verify_extracted_files` sampling
    approach — full checksum verification of every chunk is unnecessary given
    rsync's own transfer integrity guarantees; sampling catches a failed or
    partial extraction, the actual failure mode worth guarding against here.
    """
    full_remote_path = _full_remote_path(client, remote_path)
    success, stdout, _ = client.execute_command(
        f"find '{full_remote_path}' -type f | head -n {sample_size}"
    )
    if not success or not stdout.strip():
        return False
    sample_files = [line for line in stdout.strip().splitlines() if line]
    return bool(sample_files) and all(client.check_file_exists(f) for f in sample_files)


def transfer_unit(
    client: HPCClient,
    unit: Dict[str, Any],
    manifest: Optional[TransferManifest] = None,
    override: bool = False,
) -> bool:
    """Push one transfer unit (``{unit_id, local_path, remote_path}``) to the
    HPC target: tar -> rsync -> remote extract -> verify."""
    unit_id = unit["unit_id"]
    local_path = unit["local_path"]
    remote_path = unit["remote_path"]

    if not os.path.exists(local_path):
        logger.error("Transfer unit %s: local path does not exist: %s", unit_id, local_path)
        return False

    if manifest is not None and not override and manifest.status_of(unit_id) == STATUS_COMPLETED:
        logger.info("Skipping transfer unit %s -- already completed", unit_id)
        return True

    tar_dir = tempfile.mkdtemp(prefix="hpc_transfer_")
    tar_filename = f"{unit_id.replace('/', '_')}.tar.gz"
    tar_path = os.path.join(tar_dir, tar_filename)

    try:
        logger.info("Tarring transfer unit %s: %s", unit_id, local_path)
        _tar_directory(local_path, tar_path)

        remote_parent = os.path.dirname(remote_path.rstrip("/"))
        remote_tar_path = f"{remote_parent}/{tar_filename}" if remote_parent else tar_filename

        if remote_parent:
            client.ensure_directory(remote_parent)

        logger.info("Transferring %s to HPC", tar_filename)
        success, summary = client.rsync_transfer(
            tar_path, remote_tar_path, source_is_local=True,
            options={"compress": True, "archive": True, "partial": True, "verbose": False},
            show_progress=False,
        )
        if not success:
            logger.error("Transfer failed for unit %s: %s", unit_id, summary)
            if manifest is not None:
                manifest.record(unit_id, local_path, remote_path, STATUS_FAILED, error=summary)
            return False

        logger.info("Extracting %s on HPC", tar_filename)
        extract_dir = remote_parent or "."
        if not client.extract_tar(remote_tar_path, extract_dir):
            logger.error("Extraction failed for unit %s", unit_id)
            if manifest is not None:
                manifest.record(unit_id, local_path, remote_path, STATUS_FAILED, error="extraction failed")
            return False

        if not _verify_remote(client, remote_path):
            logger.error("Verification failed for unit %s", unit_id)
            if manifest is not None:
                manifest.record(unit_id, local_path, remote_path, STATUS_FAILED, error="verification failed")
            return False

        full_tar_path = _full_remote_path(client, remote_tar_path)
        client.execute_command(f"rm -f '{full_tar_path}'")

        if manifest is not None:
            manifest.record(unit_id, local_path, remote_path, STATUS_COMPLETED)
        logger.info("Transfer unit %s completed", unit_id)
        return True

    except Exception:
        logger.exception("Error transferring unit %s", unit_id)
        if manifest is not None:
            manifest.record(unit_id, local_path, remote_path, STATUS_FAILED, error="exception")
        return False
    finally:
        shutil.rmtree(tar_dir, ignore_errors=True)


def transfer_units(
    ssh_target: str,
    key_file: Optional[str],
    units: List[Dict[str, Any]],
    manifest_path: Optional[str] = None,
    override: bool = False,
) -> bool:
    """Push a list of transfer units to the HPC target.

    Returns True iff every unit transferred (or was already completed).
    """
    client = HPCClient(ssh_target, key_file=key_file)
    manifest = TransferManifest(manifest_path) if manifest_path else None

    all_succeeded = True
    for unit in units:
        if not transfer_unit(client, unit, manifest=manifest, override=override):
            all_succeeded = False
    return all_succeeded

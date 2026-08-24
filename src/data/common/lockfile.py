"""One lockfile per (source, stage) -- guards against an accidental double
invocation (overlapping cron/manual runs), not real parallel workers: this
codebase only ever runs one worker per source-stage at a time by design, so
this is a tripwire, not a scheduler.

Contents: PID + hostname + timestamp. A lock is stale (safe to steal) if its
hostname differs from ours (can't check a remote PID, so fall back to the
timestamp ceiling) or its PID is no longer alive on this host, or its
timestamp is older than `staleness_seconds` -- covers the classic failure
mode of a killed/OOM'd HPC job leaving a lock nothing will ever clear.
"""

from __future__ import annotations

import json
import os
import socket
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

#: A real FETCH/PREPARE run can take hours on HPC -- generous, but still
#: bounded, so a lock left behind by a killed job eventually self-clears
#: without operator intervention.
DEFAULT_STALENESS_SECONDS = 12 * 3600


class LockHeldError(RuntimeError):
    """Raised when *path* is held by another live, non-stale process."""


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just owned by someone else
    except OSError:
        return False
    return True


def _is_stale(info: dict, staleness_seconds: float) -> bool:
    if info.get("hostname") != socket.gethostname():
        # Can't check a remote PID's liveness -- age is the only signal.
        pass
    elif not _is_pid_alive(int(info.get("pid", -1))):
        return True

    try:
        acquired_at = datetime.fromisoformat(info["acquired_at"])
    except (KeyError, ValueError):
        return True  # unreadable timestamp -- treat as stale, not immortal
    age = (datetime.now(timezone.utc) - acquired_at).total_seconds()
    return age > staleness_seconds


def _write_payload(path: str) -> None:
    payload = {"pid": os.getpid(), "hostname": socket.gethostname(), "acquired_at": datetime.now(timezone.utc).isoformat()}
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def acquire(path: str, *, staleness_seconds: float = DEFAULT_STALENESS_SECONDS) -> None:
    """Claim *path* as this process's lock, stealing it first if it's stale.
    Raises `LockHeldError` if another live, non-stale process holds it.

    Uses O_CREAT|O_EXCL for the actual claim so two processes racing to
    acquire a not-yet-existing lock can't both succeed; only the (much
    narrower) steal-a-stale-lock path still has a check-then-write gap.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        _write_payload(path)
        return
    except FileExistsError:
        pass

    try:
        with open(path, encoding="utf-8") as fh:
            info = json.load(fh)
    except (OSError, json.JSONDecodeError):
        info = {}
    if not _is_stale(info, staleness_seconds):
        raise LockHeldError(
            f"{path} is held by pid={info.get('pid')} on host={info.get('hostname')} "
            f"since {info.get('acquired_at')}"
        )

    release(path)
    try:
        _write_payload(path)
    except FileExistsError as e:
        raise LockHeldError(f"{path} was re-acquired by another process while stealing a stale lock") from e


def release(path: str) -> None:
    """Best-effort remove -- always called from a `finally`, so a missing
    file (already cleaned up, or never successfully acquired) isn't an error."""
    try:
        os.remove(path)
    except OSError:
        pass


@contextmanager
def held(path: str, *, staleness_seconds: float = DEFAULT_STALENESS_SECONDS) -> Iterator[None]:
    """`with lockfile.held(path): ...` -- acquire on entry, release on exit
    (including on exception)."""
    acquire(path, staleness_seconds=staleness_seconds)
    try:
        yield
    finally:
        release(path)

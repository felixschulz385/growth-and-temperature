import json
import os
import socket
from datetime import datetime, timedelta, timezone

import pytest

from src.data.common import lockfile


def test_acquire_then_release_allows_reacquire(tmp_path):
    path = str(tmp_path / "source.lock")
    lockfile.acquire(path)
    assert os.path.exists(path)
    lockfile.release(path)
    assert not os.path.exists(path)
    lockfile.acquire(path)  # must not raise -- lock was released
    lockfile.release(path)


def test_acquire_raises_when_held_by_a_live_process_on_this_host(tmp_path):
    path = str(tmp_path / "source.lock")
    payload = {
        "pid": os.getpid(),  # this test process -- definitely alive
        "hostname": socket.gethostname(),
        "acquired_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w") as fh:
        json.dump(payload, fh)

    with pytest.raises(lockfile.LockHeldError):
        lockfile.acquire(path)


def test_acquire_steals_lock_with_dead_pid_on_this_host(tmp_path):
    path = str(tmp_path / "source.lock")
    # A PID essentially guaranteed not to be alive.
    payload = {"pid": 2**30, "hostname": socket.gethostname(), "acquired_at": datetime.now(timezone.utc).isoformat()}
    with open(path, "w") as fh:
        json.dump(payload, fh)

    lockfile.acquire(path)  # must not raise -- stale, stolen
    info = json.load(open(path))
    assert info["pid"] == os.getpid()


def test_acquire_steals_lock_older_than_staleness_ceiling_on_another_host(tmp_path):
    path = str(tmp_path / "source.lock")
    old = datetime.now(timezone.utc) - timedelta(seconds=100)
    payload = {"pid": 1, "hostname": "some-other-host", "acquired_at": old.isoformat()}
    with open(path, "w") as fh:
        json.dump(payload, fh)

    lockfile.acquire(path, staleness_seconds=10)  # must not raise -- older than ceiling


def test_acquire_respects_staleness_ceiling_for_recent_foreign_host_lock(tmp_path):
    path = str(tmp_path / "source.lock")
    payload = {"pid": 1, "hostname": "some-other-host", "acquired_at": datetime.now(timezone.utc).isoformat()}
    with open(path, "w") as fh:
        json.dump(payload, fh)

    with pytest.raises(lockfile.LockHeldError):
        lockfile.acquire(path, staleness_seconds=3600)


def test_held_context_manager_releases_on_exception(tmp_path):
    path = str(tmp_path / "source.lock")
    with pytest.raises(ValueError):
        with lockfile.held(path):
            assert os.path.exists(path)
            raise ValueError("boom")
    assert not os.path.exists(path)

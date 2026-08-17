"""PipelineContext: the runtime environment shared by every source.

docs/design/09-integrated-pipeline.md §8: replaces both the download side's
`WorkflowContext` (src/data/download/workflow/context.py) and the preprocess
side's `PreprocessWorkflowContext` (src/data/preprocess/workflow.py) with one
class. The persistent-session registry (needed by EOG/GLASS's Selenium
sessions) is lifted from `WorkflowContext` verbatim -- the preprocess side
never had one because, pre-merge, sources never ran their own fetch step.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Optional

from src.data.common.dask.client import DEFAULT_DASHBOARD_PORT
from src.data.sources.layout import EASE_GRID_ID, LEGACY_GRID_ID

logger = logging.getLogger(__name__)


class PipelineContext:
    """Shared runtime state: paths, HPC/remote target, dask settings, and
    per-source-keyed persistent sessions (e.g. Selenium logins)."""

    def __init__(
        self,
        data_root: str,
        *,
        local_index_dir: Optional[str] = None,
        ssh_target: Optional[str] = None,
        key_file: Optional[str] = None,
        grid_id: str = LEGACY_GRID_ID,
        dask_threads: Optional[int] = None,
        dask_memory_limit: Optional[str] = None,
        dashboard_port: int = DEFAULT_DASHBOARD_PORT,
        staging_dir: Optional[str] = None,
    ):
        self.data_root = data_root
        self.local_index_dir = str(Path(local_index_dir).expanduser().resolve()) if local_index_dir else None
        if self.local_index_dir:
            Path(self.local_index_dir).mkdir(parents=True, exist_ok=True)

        self.ssh_target = ssh_target
        self.key_file = self._normalize_key_file_path(key_file) if key_file else None
        self.remote_host, self.remote_data_root = self._split_ssh_target(ssh_target)

        if grid_id not in (LEGACY_GRID_ID, EASE_GRID_ID):
            raise ValueError(f"Unknown grid_id '{grid_id}', expected one of ({LEGACY_GRID_ID!r}, {EASE_GRID_ID!r})")
        self.grid_id = grid_id

        self.dask_threads = dask_threads
        self.dask_memory_limit = dask_memory_limit
        self.dashboard_port = dashboard_port

        self.staging_dir = staging_dir or (
            str(Path(self.local_index_dir) / "staging") if self.local_index_dir else None
        )
        if self.staging_dir:
            Path(self.staging_dir).mkdir(parents=True, exist_ok=True)

        self._persistent_sessions: dict[str, Any] = {}
        self._session_locks: dict[str, threading.RLock] = {}

    @staticmethod
    def _split_ssh_target(ssh_target: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """`user@host:/path` -> ("user@host", "/path"); matches
        WorkflowContext's hpc_host/hpc_path split (context.py:44-51)."""
        if not ssh_target:
            return None, None
        if ":" in ssh_target:
            host, path = ssh_target.split(":", 1)
            return host, PurePosixPath(path).as_posix().rstrip("/")
        return ssh_target, None

    @staticmethod
    def _normalize_key_file_path(key_file: str) -> str:
        return str(Path(key_file).expanduser().resolve())

    # -- persistent sessions (thread-safe get-or-create / teardown) --------

    def get_persistent_session(self, key: str, creator_fn: Callable[[], Any]) -> Any:
        if key not in self._session_locks:
            self._session_locks[key] = threading.RLock()
        with self._session_locks[key]:
            if self._persistent_sessions.get(key) is None:
                logger.info("Creating new persistent session for %s", key)
                self._persistent_sessions[key] = creator_fn()
        return self._persistent_sessions[key]

    def close_persistent_session(self, key: str) -> None:
        lock = self._session_locks.get(key)
        if not lock:
            return
        with lock:
            session = self._persistent_sessions.pop(key, None)
            if session is None:
                return
            logger.info("Closing persistent session for %s", key)
            try:
                if hasattr(session, "close"):
                    session.close()
                elif hasattr(session, "quit"):
                    session.quit()
            except Exception:
                logger.warning("Error closing session for %s", key, exc_info=True)

    def close_all_persistent_sessions(self) -> None:
        for key in list(self._persistent_sessions):
            self.close_persistent_session(key)

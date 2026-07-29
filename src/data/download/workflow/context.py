"""
Workflow context: staging directories, SSH key handling, and persistent
session management for basic and HPC download workflows.
"""

import logging
import platform
import threading
from pathlib import Path, PurePosixPath
from typing import Optional
import os

logger = logging.getLogger(__name__)

# Windows compatibility constants
IS_WINDOWS = platform.system() == 'Windows'


class WorkflowContext:
    """Unified context for workflow execution supporting both basic and HPC environments."""

    def __init__(self, bucket_name: Optional[str] = None, hpc_target: str = None,
                 local_index_dir: str = None, key_file: str = None):
        """
        Initialize the workflow context.

        Args:
            bucket_name: Storage bucket name (legacy parameter)
            hpc_target: SSH target for HPC (user@server:/path)
            local_index_dir: Directory for local index storage
            key_file: Path to SSH private key file (optional)
        """
        self.bucket_name = bucket_name
        self.hpc_target = hpc_target
        self.key_file = self._normalize_key_file_path(key_file) if key_file else None

        # Set up directories
        if hpc_target and local_index_dir:
            # HPC workflow context
            self.local_index_dir = str(Path(local_index_dir).expanduser().resolve())
            Path(self.local_index_dir).mkdir(parents=True, exist_ok=True)

            # Extract HPC host and path
            if ":" in hpc_target:
                parts = hpc_target.split(":", 1)
                self.hpc_host = parts[0]
                # Always use forward slashes for remote paths (POSIX)
                self.hpc_path = PurePosixPath(parts[1]).as_posix().rstrip('/')
            else:
                self.hpc_host = hpc_target
                self.hpc_path = ""

            # Create staging directory using pathlib
            self.staging_dir = str(Path(self.local_index_dir) / "staging")
            Path(self.staging_dir).mkdir(parents=True, exist_ok=True)

            logger.debug(f"Initialized HPC context with host: {self.hpc_host}, path: {self.hpc_path}")
            logger.debug(f"Local index directory: {self.local_index_dir}")
            logger.debug(f"Staging directory: {self.staging_dir}")
        else:
            # Default context
            data_nobackup_root = os.environ.get("DATA_NOBACKUP", os.path.join(os.getcwd(), "data_nobackup"))
            self.staging_dir = os.path.join(data_nobackup_root, "staging")
            os.makedirs(self.staging_dir, exist_ok=True)

        # Session management
        self._persistent_sessions = {}
        self._session_locks = {}  # Add locks for thread safety

        # Log SSH key file info for debugging
        if self.key_file:
            logger.debug(f"Using SSH key file: {self.key_file}")
            key_path = Path(self.key_file)
            if key_path.exists():
                logger.debug(f"SSH key file exists: {key_path}")
                # On Windows, check file permissions
                if IS_WINDOWS:
                    self._check_windows_key_permissions(key_path)
            else:
                logger.warning(f"SSH key file does not exist: {key_path}")

    def _normalize_key_file_path(self, key_file: str) -> Optional[str]:
        """Normalize SSH key file path for cross-platform compatibility."""
        if not key_file:
            return None

        # Expand user directory and resolve path
        key_path = Path(key_file).expanduser().resolve()

        # On Windows, convert to string with forward slashes for SSH
        if IS_WINDOWS:
            # SSH on Windows expects forward slashes
            return str(key_path).replace('\\', '/')
        else:
            return str(key_path)

    def _check_windows_key_permissions(self, key_path: Path):
        """Check SSH key file permissions on Windows."""
        try:
            import stat
            file_stat = key_path.stat()
            # On Windows, warn if file is readable by others
            if file_stat.st_mode & (stat.S_IRGRP | stat.S_IROTH):
                logger.warning(f"SSH key file {key_path} may have overly permissive permissions on Windows")
        except Exception as e:
            logger.debug(f"Could not check key file permissions: {e}")

    def get_persistent_session(self, key: str, creator_fn):
        """
        Get or create a persistent session object for a given key.
        This method is thread-safe.

        Args:
            key: Unique key for the session (e.g., data source name)
            creator_fn: Function to create the session if not present
        Returns:
            The persistent session object
        """
        # Create lock for this session key if it doesn't exist
        if key not in self._session_locks:
            self._session_locks[key] = threading.RLock()

        # Use lock to ensure thread safety
        with self._session_locks[key]:
            if key not in self._persistent_sessions or self._persistent_sessions[key] is None:
                logger.info(f"Creating new persistent session for {key}")
                try:
                    self._persistent_sessions[key] = creator_fn()
                except Exception as e:
                    logger.error(f"Failed to create persistent session for {key}: {e}")
                    # Don't store None - keep trying to recreate on failure
                    raise

        return self._persistent_sessions[key]

    def close_persistent_session(self, key: str):
        """
        Close and remove a persistent session for a given key.
        This method is thread-safe.
        """
        # Use lock if it exists
        lock = self._session_locks.get(key)
        if lock:
            with lock:
                sess = self._persistent_sessions.pop(key, None)
                if sess:
                    logger.info(f"Closing persistent session for {key}")
                    try:
                        # Try direct close() method first
                        if hasattr(sess, "close"):
                            sess.close()
                        # Then try quit() method for selenium WebDriver
                        elif hasattr(sess, "quit"):
                            sess.quit()
                    except Exception as e:
                        logger.warning(f"Error closing session for {key}: {e}")

    def close_all_persistent_sessions(self):
        """
        Close all persistent sessions.
        """
        logger.info(f"Closing all {len(self._persistent_sessions)} persistent sessions")
        for key in list(self._persistent_sessions.keys()):
            self.close_persistent_session(key)

"""
Base workflow for managing data downloads.
"""

import os
import re
import time
import yaml
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


def load_config_with_env_vars(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration file with environment variables expanded."""
    env_pattern = re.compile(r'\${([^}^{]+)}')
    
    # Function to replace environment variables in strings
    def replace_env_vars(value: str) -> str:
        def replace(match):
            env_var = match.group(1)
            return os.environ.get(env_var, '')
        return env_pattern.sub(replace, value)
    
    # Process all items in a structure
    def process_item(item):
        if isinstance(item, dict):
            return {k: process_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [process_item(i) for i in item]
        elif isinstance(item, str):
            return replace_env_vars(item)
        else:
            return item
            
    # Load the YAML file
    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Process environment variables
    return process_item(config)


class WorkflowContext:
    """Base context for workflow execution."""
    
    def __init__(self, bucket_name: Optional[str] = None):
        """Initialize the workflow context."""
        self.bucket_name = bucket_name
        self.staging_dir = os.path.join(os.getcwd(), "staging")
        os.makedirs(self.staging_dir, exist_ok=True)
        self._persistent_sessions = {}
        self._session_locks = {}  # Add locks for thread safety

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


class TaskHandlers:
    """Base class for task handlers."""
    
    @staticmethod
    def handle_index(data_source, download_index, context, task_config):
        """Handle index-building task."""
        raise NotImplementedError("Subclasses must implement handle_index")
    
    @staticmethod
    def handle_download(data_source, download_index, context, task_config):
        """Handle download task."""
        raise NotImplementedError("Subclasses must implement handle_download")
    
    @staticmethod
    def handle_validate(data_source, download_index, context, task_config):
        """Handle validation task."""
        raise NotImplementedError("Subclasses must implement handle_validate")


def setup_progress_reporting(download_index, job_queue, stop_event, interval=10):
    """
    Set up a thread for progress reporting - PLACEHOLDER.
    
    Args:
        download_index: Download index
        job_queue: Job queue (can be sync or async)
        stop_event: Stop event
        interval: Reporting interval in seconds
        
    Returns:
        Thread: Progress reporting thread
    """
    logger.warning("Progress reporting functionality has been removed")
    logger.info("This is a placeholder for future implementation")
    
    # TODO: Implement progress reporting
    # - Monitor download progress
    # - Report statistics
    # - Save index periodically
    
    # Return a dummy thread for compatibility
    def dummy_reporter():
        while not stop_event.is_set():
            time.sleep(interval)
    
    thread = threading.Thread(target=dummy_reporter)
    thread.daemon = True
    thread.start()
    return thread


def queue_files_for_download(download_index, job_queue, batch_size=500, max_queue_size=1000):
    """
    Queue files for download - PLACEHOLDER.
    
    Args:
        download_index: Download index
        job_queue: Job queue (sync or async)
        batch_size: Size of each batch to query
        max_queue_size: Maximum queue size
        
    Returns:
        int: Total number of files queued
    """
    logger.warning("File queuing functionality has been removed")
    logger.info("This is a placeholder for future implementation")
    
    # TODO: Implement file queuing
    # - Query pending files from index
    # - Add to download queue
    # - Handle batch processing
    # - Support async/sync queues
    
    return 0  # Not implemented
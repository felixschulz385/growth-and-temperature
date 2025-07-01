"""
Unified workflow for managing data operations across different environments.
Cross-platform compatible (Windows/Linux) with HPC-specific optimizations.
"""

import os
import re
import time
import json
import yaml
import logging
import threading
import platform
import asyncio
from pathlib import Path, PurePosixPath
from typing import Dict, Any, List, Union, Optional
from datetime import datetime

from gnt.data.common.index.unified_index import UnifiedDataIndex
from gnt.data.download.sources.factory import create_data_source
from gnt.data.common.hpc.client import HPCClient

# Configure logging
logger = logging.getLogger(__name__)

# Windows compatibility constants
IS_WINDOWS = platform.system() == 'Windows'


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
            self.staging_dir = os.path.join(os.getcwd(), "staging")
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


class TaskHandlers:
    """Unified task handlers for all workflow operations."""
    
    @staticmethod
    def handle_index(data_source, download_index, context, task_config):
        """Handle index-building task."""
        logger.info(f"Building index for {data_source.DATA_SOURCE_NAME}")
        
        # Extract parameters from task config
        rebuild = task_config.get('rebuild', False)
        only_missing_entrypoints = task_config.get('only_missing_entrypoints', True)
        sync_direction = task_config.get('sync_direction', 'auto')
        
        try:
            # Check for schema conversion setting
            force_schema_conversion = task_config.get('force_schema_conversion', False)
            
            # If schema conversion is forced, rebuild the index
            if force_schema_conversion:
                logger.info("Force schema conversion enabled - rebuilding index")
                rebuild = True
            
            # Sync index first if configured and HPC target is available
            if hasattr(context, 'hpc_target') and context.hpc_target and sync_direction != 'none':
                success = download_index.ensure_synced_index(
                    hpc_target=context.hpc_target,
                    sync_direction=sync_direction,
                    key_file=context.key_file
                )
                if not success:
                    logger.warning("Index sync failed, continuing with local index")
            
            # Get build_index_from_source parameters
            build_params = {
                'data_source': data_source,
                'rebuild': rebuild,
                'only_missing_entrypoints': only_missing_entrypoints
            }
            
            # Add schema parameters only if the method accepts them
            import inspect
            build_index_sig = inspect.signature(download_index.build_index_from_source)
            if 'schema_dtypes' in build_index_sig.parameters:
                build_params['schema_dtypes'] = getattr(data_source, 'schema_dtypes', {})
            if 'force_schema_conversion' in build_index_sig.parameters:
                build_params['force_schema_conversion'] = force_schema_conversion
            
            # Build index from source with appropriate parameters
            try:
                files_indexed = download_index.build_index_from_source(**build_params)
            except ValueError as e:
                if "Schema" in str(e) or "migrate" in str(e):
                    logger.warning(f"Schema migration error: {e}")
                    logger.info("Attempting to rebuild index with schema conversion")
                    # Force rebuild on schema errors
                    build_params['rebuild'] = True
                    files_indexed = download_index.build_index_from_source(**build_params)
                else:
                    raise
            
            logger.info(f"Index building complete: {files_indexed} files indexed")
            
            # Save index
            download_index.save()
            
            # Sync back to HPC if configured
            if (hasattr(context, 'hpc_target') and context.hpc_target and 
                sync_direction in ['auto', 'push']):
                download_index.sync_index_with_hpc(
                    hpc_target=context.hpc_target,
                    direction='push',
                    key_file=context.key_file
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error in index task: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return False
    
    @staticmethod
    def handle_download(data_source, download_index, context, task_config):
        """Handle download task using async downloader."""
        logger.info("Starting async download task")
        
        try:
            # Import the async downloader
            from gnt.data.download.async_downloader import run_async_download_workflow
            
            # Check if we have HPC context
            if hasattr(context, 'hpc_target') and context.hpc_target:
                # Ensure index is synced before downloads
                logger.info("Ensuring index is synced before downloads")
                sync_success = download_index.ensure_synced_index(
                    hpc_target=context.hpc_target,
                    sync_direction='pull',  # Always pull before downloads
                    key_file=context.key_file
                )
                
                if not sync_success:
                    logger.warning("Index sync failed, continuing with local index")
                
                # Create HPC client
                hpc_client = HPCClient(
                    target=context.hpc_target,
                    key_file=context.key_file
                )
                
                # Run async download workflow
                return asyncio.run(run_async_download_workflow(
                    data_source=data_source,
                    index=download_index,
                    hpc_client=hpc_client,
                    context=context,
                    config=task_config
                ))
            else:
                logger.warning("Download requires HPC target configuration")
                return False
            
        except ImportError as e:
            logger.error(f"Error importing async downloader: {e}")
            return False
        except Exception as e:
            logger.error(f"Error in download task: {e}")
            return False
    
    @staticmethod
    def handle_validate(data_source, download_index, context, task_config):
        """Handle validation task - PLACEHOLDER."""
        logger.warning("Validation workflow functionality has been removed")
        logger.info("This is a placeholder for future implementation")
        
        # TODO: Implement validation workflow
        # - Verify downloaded files
        # - Check batch integrity
        # - Validate transfers to HPC
        # - Report status
        
        return False  # Not implemented


def setup_progress_reporting(download_index, stop_event, interval=30):
    """
    Set up lightweight progress reporting thread.
    
    Args:
        download_index: Download index
        stop_event: Stop event
        interval: Reporting interval in seconds
        
    Returns:
        Thread: Progress reporting thread
    """
    def progress_reporter():
        """Report download progress periodically."""
        start_time = time.time()
        
        while not stop_event.is_set():
            try:
                # Get current stats
                stats = download_index.get_download_stats()
                elapsed = time.time() - start_time
                
                logger.info(f"Progress: {stats['completed']} completed, "
                           f"{stats['pending']} pending, "
                           f"{stats['failed']} failed, "
                           f"{stats['downloading']} downloading "
                           f"(elapsed: {elapsed:.1f}s)")
                
                # Save index periodically
                download_index.save()
                
            except Exception as e:
                logger.error(f"Error in progress reporting: {e}")
            
            # Wait for next interval or stop event
            stop_event.wait(interval)
    
    thread = threading.Thread(target=progress_reporter, name="ProgressReporter")
    thread.daemon = True
    thread.start()
    return thread


def queue_files_for_download(download_index, batch_size=500, max_files=None):
    """
    Efficiently queue files for download from Parquet index.
    
    Args:
        download_index: UnifiedDataIndex instance
        batch_size: Size of each batch to query
        max_files: Maximum number of files to queue (None for all)
        
    Returns:
        int: Total number of files available for download
    """
    try:
        # Get count of pending files
        pending_count = download_index.count_pending_files()
        
        if pending_count == 0:
            logger.info("No files pending download")
            return 0
        
        # Limit files if specified
        if max_files:
            pending_count = min(pending_count, max_files)
        
        logger.info(f"Found {pending_count} files pending download")
        return pending_count
        
    except Exception as e:
        logger.error(f"Error queuing files for download: {e}")
        return 0


def run_workflow_with_config(config: Dict[str, Any]):
    """
    Main entry point for running unified workflow with configuration.
    
    Args:
        config: Configuration dictionary containing all workflow settings
    """
    logger.info("Starting unified workflow")
    
    try:
        # Extract configuration sections
        source_config = config.get('source', {})
        index_config = config.get('index', {})
        workflow_config = config.get('workflow', {})
        hpc_config = config.get('hpc', {})
        
        # Handle case where source name might be passed separately
        if 'source_name' in config and not any(k in source_config for k in ['name', 'dataset_name', 'source_name', 'type']):
            source_config['name'] = config['source_name']
        
        # Create data source
        data_source = create_data_source(source_config)
        logger.info(f"Created data source: {data_source.DATA_SOURCE_NAME}")
        
        # Create workflow context - HPC if target specified, otherwise basic
        if hpc_config.get('target'):
            context = WorkflowContext(
                bucket_name=None,
                hpc_target=hpc_config['target'],
                local_index_dir=index_config.get('local_dir'),
                key_file=hpc_config.get('key_file')
            )
            logger.info("Using HPC workflow context")
        else:
            context = WorkflowContext(bucket_name=config.get('bucket_name'))
            logger.info("Using basic workflow context")
        
        # Extract schema options before passing to UnifiedDataIndex
        schema_dtypes = getattr(data_source, 'schema_dtypes', {})
        enforce_schema = index_config.get('enforce_schema', True)
        
        # Create download index - don't pass schema_dtypes directly
        download_index = UnifiedDataIndex(
            bucket_name="",
            data_source=data_source,
            local_index_dir=getattr(context, 'local_index_dir', None),
            key_file=getattr(context, 'key_file', None),
            hpc_mode=bool(hpc_config.get('target'))  # Enable HPC mode if target is specified
        )
        
        # Set schema options as attributes if available
        if hasattr(download_index, 'set_schema_options') and schema_dtypes:
            download_index.set_schema_options(schema_dtypes, enforce_schema)
        else:
            # Fallback - set attributes directly if needed
            if hasattr(download_index, 'schema_dtypes') and schema_dtypes:
                download_index.schema_dtypes = schema_dtypes
            if hasattr(download_index, 'enforce_schema'):
                download_index.enforce_schema = enforce_schema
        
        # Execute tasks in order
        tasks = workflow_config.get('tasks', [])
        task_handlers = TaskHandlers()
        
        for task in tasks:
            task_type = task.get('type')
            task_config = task.get('config', {})
            
            # Check for schema conversion flag in task config or index config
            if task_type == 'index':
                if 'force_schema_conversion' not in task_config:
                    # If not in task config, check index config
                    task_config['force_schema_conversion'] = index_config.get('force_schema_conversion', False)
            
            logger.info(f"Executing task: {task_type}")
            
            if task_type == 'index':
                success = task_handlers.handle_index(data_source, download_index, context, task_config)
            elif task_type == 'download':
                success = task_handlers.handle_download(data_source, download_index, context, task_config)
            elif task_type == 'validate':
                success = task_handlers.handle_validate(data_source, download_index, context, task_config)
            else:
                logger.error(f"Unknown task type: {task_type}")
                success = False
            
            if not success:
                logger.error(f"Task {task_type} failed")
                return False
        
        logger.info("Unified workflow completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in unified workflow: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return False
    finally:
        # Clean up context
        if 'context' in locals():
            context.close_all_persistent_sessions()


# Backward compatibility aliases
run_hpc_workflow_with_config = run_workflow_with_config
HPCWorkflowContext = WorkflowContext
HPCTaskHandlers = TaskHandlers

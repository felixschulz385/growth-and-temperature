"""
HPC-specific workflow for managing downloads and transfers to HPC systems.
Cross-platform compatible (Windows/Linux).
"""

import os
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
from gnt.data.download.workflow import WorkflowContext, TaskHandlers
from gnt.data.download.workflow import setup_progress_reporting, load_config_with_env_vars
from gnt.data.download.sources.factory import create_data_source
from gnt.data.common.hpc.client import HPCClient
from gnt.data.download.async_downloader import run_async_download_workflow

# Configure logging
logger = logging.getLogger(__name__)

# Windows compatibility constants
IS_WINDOWS = platform.system() == 'Windows'

class HPCWorkflowContext(WorkflowContext):
    """Context for HPC workflow execution."""
    
    def __init__(self, hpc_target: str, local_index_dir: str = None, key_file: str = None):
        """
        Initialize the HPC workflow context.
        
        Args:
            hpc_target: SSH target for HPC (user@server:/path)
            local_index_dir: Directory for local index storage
            key_file: Path to SSH private key file (optional)
        """
        super().__init__(bucket_name=None)  # Bucket name not used
        self.hpc_target = hpc_target
        
        # Use pathlib for cross-platform path handling
        if local_index_dir:
            self.local_index_dir = str(Path(local_index_dir).expanduser().resolve())
        else:
            self.local_index_dir = str(Path.home() / "hpc_data_index")
        
        self.key_file = self._normalize_key_file_path(key_file)
        Path(self.local_index_dir).mkdir(parents=True, exist_ok=True)
        
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


class HPCTaskHandlers(TaskHandlers):
    """HPC-specific task handlers."""
    
    @staticmethod
    def handle_index(data_source, download_index, context, task_config):
        """Handle index-building task for HPC workflow."""
        logger.info(f"Building index for {data_source.DATA_SOURCE_NAME}")
        
        # Extract parameters from task config
        rebuild = task_config.get('rebuild', False)
        only_missing_entrypoints = task_config.get('only_missing_entrypoints', True)
        sync_direction = task_config.get('sync_direction', 'auto')
        
        try:
            # Sync index first if configured
            if sync_direction != 'none':
                success = download_index.ensure_synced_index(
                    hpc_target=context.hpc_target,
                    sync_direction=sync_direction,
                    key_file=context.key_file
                )
                if not success:
                    logger.warning("Index sync failed, continuing with local index")
            
            # Build index from source
            files_indexed = download_index.build_index_from_source(
                data_source=data_source,
                rebuild=rebuild,
                only_missing_entrypoints=only_missing_entrypoints
            )
            
            logger.info(f"Index building complete: {files_indexed} files indexed")
            
            # Save index
            download_index.save()
            
            # Sync back to HPC if configured
            if sync_direction in ['auto', 'push']:
                download_index.sync_index_with_hpc(
                    hpc_target=context.hpc_target,
                    direction='push',
                    key_file=context.key_file
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error in index task: {e}")
            return False
    
    @staticmethod
    def handle_download(data_source, download_index, context, task_config):
        """Handle download task for HPC workflow using async downloader."""
        logger.info("Starting async HPC download task")
        
        try:
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
            
        except Exception as e:
            logger.error(f"Error in async download task: {e}")
            return False
    
    @staticmethod
    def handle_validate(data_source, download_index, context, task_config):
        """Handle validation task for HPC workflow - PLACEHOLDER."""
        logger.warning("Validation workflow functionality has been removed")
        logger.info("This is a placeholder for future implementation")
        
        # TODO: Implement validation workflow
        # - Verify downloaded files
        # - Check batch integrity
        # - Validate transfers to HPC
        # - Report status
        
        return False  # Not implemented


def run_hpc_workflow_with_config(config: Dict[str, Any]):
    """
    Main entry point for running HPC workflow with configuration.
    
    Args:
        config: Configuration dictionary containing all workflow settings
    """
    logger.info("Starting HPC workflow")
    
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
        
        # Create HPC context
        context = HPCWorkflowContext(
            hpc_target=hpc_config['target'],
            local_index_dir=index_config.get('local_dir'),
            key_file=hpc_config.get('key_file')
        )
        
        # Create download index with HPC mode enabled
        download_index = UnifiedDataIndex(
            bucket_name=None,  # Not used in HPC workflow
            data_source=data_source,
            local_index_dir=context.local_index_dir,
            key_file=context.key_file,
            hpc_mode=True  # Enable HPC optimizations
        )
        
        # Execute tasks in order
        tasks = workflow_config.get('tasks', [])
        task_handlers = HPCTaskHandlers()
        
        for task in tasks:
            task_type = task.get('type')
            task_config = task.get('config', {})
            
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
        
        logger.info("HPC workflow completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in HPC workflow: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return False
    finally:
        # Clean up context
        if 'context' in locals():
            context.close_all_persistent_sessions()
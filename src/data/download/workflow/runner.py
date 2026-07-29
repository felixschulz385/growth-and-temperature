"""
Main entry point for running the unified download workflow.
"""

import logging
from typing import Any, Dict

from src.data.common.index.unified_index import UnifiedDataIndex
from src.data.download.sources.factory import create_data_source
from src.config.runtime import get_paths_config, get_remote_config

from .context import WorkflowContext
from .handlers import TaskHandlers

logger = logging.getLogger(__name__)


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
        paths_config = get_paths_config(config)
        remote_config = get_remote_config(config)

        # Handle case where source name might be passed separately
        if 'source_name' in config and not any(k in source_config for k in ['name', 'dataset_name', 'source_name', 'type']):
            source_config['name'] = config['source_name']

        # Create data source
        data_source = create_data_source(source_config)
        logger.info(f"Created data source: {data_source.DATA_SOURCE_NAME}")

        # Create workflow context - HPC if target specified, otherwise basic
        if remote_config.get('ssh_target'):
            context = WorkflowContext(
                bucket_name=None,
                hpc_target=remote_config['ssh_target'],
                local_index_dir=index_config.get('local_dir') or paths_config.get('local_index_dir'),
                key_file=remote_config.get('key_file')
            )
            logger.info("Using remote transfer workflow context")
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
            hpc_mode=bool(remote_config.get('ssh_target'))
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

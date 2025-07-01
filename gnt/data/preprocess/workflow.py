"""
Workflow management for preprocessing geodata.

This module provides functions to execute preprocessing workflows
for different types of geodata, handling the instantiation of 
appropriate preprocessors based on configuration settings.

When running in Docker/Kubernetes:
1. Input/output paths in configurations should refer to paths within the container
2. Volumes must be properly mounted to make data accessible to the container
3. For large data processing, ensure sufficient resources are allocated in pod specs
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import json
import yaml
from datetime import datetime
import os
import signal
import sys

from gnt.data.preprocess.sources.factory import create_preprocessor
from gnt.data.common.gcs.client import GCSClient

# Configure logging
logger = logging.getLogger(__name__)

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a JSON or YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    if config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    elif config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def expand_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively expand environment variables in string values within a config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with environment variables expanded
    """
    if isinstance(config, dict):
        return {k: expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(i) for i in config]
    elif isinstance(config, str):
        return os.path.expandvars(config)
    else:
        return config


def process_task(task_config: Dict[str, Any]) -> None:
    """
    Process a single preprocessing task.
    
    Args:
        task_config: Configuration for the task
    """
    # Add memory management parameters if specified
    memory_limit = task_config.pop("memory_limit", None)
    if memory_limit:
        try:
            import resource
            # Convert to bytes (assuming memory_limit is in MB)
            resource.setrlimit(resource.RLIMIT_AS, 
                              (memory_limit * 1024 * 1024, 
                               memory_limit * 1024 * 1024))
            logger.info(f"Memory limit set to {memory_limit}MB")
        except (ImportError, ValueError, resource.error) as e:
            logger.warning(f"Could not set memory limit: {str(e)}")
    
    try:
        # Get mode from task config
        mode = task_config.pop("mode", "preprocess")
        
        if mode == "generate_targets":
            # New mode: generate preprocessing targets
            handle_generate_targets_task(task_config)
        elif mode == "validate":
            # Handle validation task
            handle_validate_task(preprocessor_name, task_config)
        else:
            # Create preprocessor instance using the factory
            preprocessor = create_preprocessor(preprocessor_name, task_config)
            
            # Get targets for this task
            stage = task_config.get('stage', 'annual')
            year_range = task_config.get('year_range')
            
            targets = preprocessor.get_preprocessing_targets(stage, year_range)
            
            # Filter to ready targets only
            from gnt.data.preprocess.cache import PreprocessingTargetCache
            cache = PreprocessingTargetCache("/tmp/preprocessing_cache", preprocessor_name)
            valid_targets = cache.validate_targets(targets)
            ready_targets = [t for t in valid_targets if t.get('status') == 'ready']
            
            logger.info(f"Found {len(ready_targets)} ready targets out of {len(targets)} total")
            
            # Process each ready target
            for target in ready_targets:
                preprocessor.process_target(target)
        
    except Exception as e:
        logger.error(f"Error processing task with {task_config.get('preprocessor', 'unknown')}: {str(e)}")
        raise


def handle_validate_task(preprocessor_name: str, task_config: Dict[str, Any]) -> None:
    """
    Handle validation of preprocessing index and results.
    
    Args:
        preprocessor_name: Name of the preprocessor
        task_config: Configuration for the task
    """
    logger.info(f"Starting validation for {preprocessor_name}")
    
    # Create a copy of the task config to avoid modifying the original
    validation_config = task_config.copy()
    
    # For validation tasks, add placeholder year_range if not provided
    # This is needed for preprocessors like EOG that require year information
    if preprocessor_name.lower() == 'eog' and 'year' not in validation_config and 'year_range' not in validation_config:
        logger.debug("Adding placeholder year_range for validation task")
        validation_config['year_range'] = [2000, 2000]  # Dummy year range that won't be used
    
    # Create preprocessor for accessing index
    preprocessor = create_preprocessor(preprocessor_name, validation_config)
    
    # Force refresh option from task config
    force_refresh = task_config.get("force_refresh_gcs", False)
    
    # Get cleanup options
    fix_orphaned_transfers = task_config.get("fix_orphaned_transfers", True)
    remove_missing_from_index = task_config.get("remove_missing_from_index", False)
    auto_index_files = task_config.get("auto_index_files", True)
    
    # Access the preprocessing index
    if hasattr(preprocessor, 'preprocessing_index') and preprocessor.preprocessing_index:
        index = preprocessor.preprocessing_index
        
        # Temporarily disable auto-saves by increasing the save interval
        # Save original settings to restore later
        original_save_interval = index.save_interval_seconds
        original_operations_count = index._operations_since_save
        index.save_interval_seconds = 3600  # Set to 1 hour to prevent auto-saves during validation
        
        # Check if the index has validate_against_gcs method
        if hasattr(index, 'validate_against_gcs'):
            # Create GCS client if needed
            gcs_client = None
            if hasattr(preprocessor, 'bucket_name'):
                gcs_client = GCSClient(preprocessor.bucket_name)
            
            # Set validation paths to only include annual and spatial
            validation_paths = []
            
            # Add annual path if it exists
            if hasattr(index, 'annual_path'):
                validation_paths.append(index.annual_path)
            
            # Add spatial path if it exists
            if hasattr(index, 'spatial_path'):
                validation_paths.append(index.spatial_path)
                
            try:
                # Run validation
                logger.info(f"Validating preprocessing index against GCS for paths: {validation_paths}")
                validation_results = index.validate_against_gcs(
                    gcs_client=gcs_client,
                    force_file_list_update=force_refresh,
                    paths_to_check=validation_paths
                )
                
                # Log validation results
                logger.info("Validation results:")
                for stage, missing in validation_results.get('missing_from_index', {}).items():
                    logger.info(f"  Stage {stage}: {len(missing)} files in GCS not in index")
                
                for stage, missing in validation_results.get('missing_from_gcs', {}).items():
                    logger.info(f"  Stage {stage}: {len(missing)} files in index not in GCS")
                
                orphaned_transfers = validation_results.get('orphaned_transfers', [])
                logger.info(f"  Orphaned transfers: {len(orphaned_transfers)}")
                
                # Track changes to determine if we need to save at the end
                changes_made = False
                
                # Cleanup if requested
                if fix_orphaned_transfers or remove_missing_from_index:
                    logger.info("Performing cleanup operations...")
                    cleanup_results = index.cleanup_missing_files(
                        fix_orphaned_transfers=fix_orphaned_transfers,
                        remove_missing_from_index=remove_missing_from_index
                    )
                    
                    # Prevent auto-save by resetting operations counter
                    index._operations_since_save = 0
                    
                    logger.info(f"Cleanup results:")
                    logger.info(f"  Removed {cleanup_results['orphaned_transfers_removed']} orphaned transfers")
                    logger.info(f"  Removed {cleanup_results['missing_files_removed']} missing files from index")
                    
                    if cleanup_results['orphaned_transfers_removed'] > 0 or cleanup_results['missing_files_removed'] > 0:
                        changes_made = True
                
                # Add missing files to index if requested
                if auto_index_files:
                    # Count total missing files across all stages
                    total_missing = sum(len(files) for files in validation_results.get('missing_from_index', {}).values())
                    
                    if total_missing > 0:
                        logger.info(f"Indexing {total_missing} missing files found during validation...")
                        files_added = 0
                        
                        # Process each stage separately
                        for stage, missing_files in validation_results.get('missing_from_index', {}).items():
                            if missing_files:
                                logger.info(f"Adding {len(missing_files)} missing {stage} files to index")
                                for blob_path in missing_files:
                                    try:
                                        # First try the regular parser
                                        info = index._parse_blob_path(blob_path)
                                            
                                        # Add the file to index if we have a year
                                        if info.get('year') is not None:
                                            index.add_file(
                                                stage=stage,
                                                year=info['year'],
                                                grid_cell=info.get('grid_cell', 'global'),
                                                status=index.STATUS_COMPLETED,
                                                blob_path=blob_path,
                                                metadata={"indexed_during_validation": True}
                                            )
                                            files_added += 1
                                        else:
                                            logger.warning(f"Could not extract year from path: {blob_path}")
                                    except Exception as e:
                                        logger.warning(f"Error adding {blob_path} to index: {str(e)}")
                        
                        # Prevent auto-save from triggering by resetting counter
                        index._operations_since_save = 0
                        
                        logger.info(f"Successfully added {files_added} files to index")
                        if files_added > 0:
                            changes_made = True
                
                # Show final statistics
                try:
                    stats = index.get_stats()
                    logger.info("Preprocessing index statistics:")
                    for stage, data in stats.get('stages', {}).items():
                        if data and isinstance(data, dict):
                            total = data.get('total', 0)
                            completed = data.get('completed', 0) if isinstance(data.get('completed'), int) else 0
                            processing = data.get('processing', 0) if isinstance(data.get('processing'), int) else 0
                            failed = data.get('failed', 0) if isinstance(data.get('failed'), int) else 0
                            logger.info(f"  Stage {stage}: {total} total, {completed} completed, {processing} processing, {failed} failed")
                except Exception as e:
                    logger.warning(f"Error getting stats: {e}")
                
                # Save index once at the end if changes were made
                # This is the key part we're fixing!
                if changes_made:
                    logger.info("Saving validated index...")
                    index.save()
                else:
                    logger.info("No changes made to index, skipping save operation")
                
            except Exception as e:
                logger.error(f"Unexpected error during validation: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        else:
            logger.error("Preprocessing index does not support validation")
        
        # Restore original settings
        index.save_interval_seconds = original_save_interval
        # Do not restore operations count - this would trigger an immediate save
        # instead, we've already saved if changes were made
        
    else:
        logger.error("Preprocessor does not have a valid preprocessing index")


class PreprocessWorkflowContext:
    """Unified context for preprocessing workflow execution."""
    
    def __init__(self, hpc_config: Dict[str, Any] = None, gcs_config: Dict[str, Any] = None):
        """
        Initialize the preprocessing workflow context.
        
        Args:
            hpc_config: HPC configuration dictionary
            gcs_config: GCS configuration dictionary
        """
        self.hpc_config = hpc_config or {}
        self.gcs_config = gcs_config or {}
        
        # Set up staging directory
        self.staging_dir = os.path.join(os.getcwd(), "preprocessing_staging")
        os.makedirs(self.staging_dir, exist_ok=True)
        
        logger.debug(f"Initialized preprocessing context with staging dir: {self.staging_dir}")


class PreprocessTaskHandlers:
    """Unified task handlers for preprocessing workflow operations."""
    
    @staticmethod
    def handle_preprocess(source_config: Dict[str, Any], context: PreprocessWorkflowContext, task_config: Dict[str, Any]):
        """Handle preprocessing task."""
        logger.info(f"Starting preprocessing task for {source_config.get('name', 'unknown')}")
        
        try:
            # Prepare task configuration in the format expected by existing process_task
            legacy_task_config = task_config.copy()
            
            # Set preprocessor name from source configuration
            legacy_task_config['preprocessor'] = source_config.get('name')
            
            # Add HPC configuration if available
            if context.hpc_config:
                for key, value in context.hpc_config.items():
                    legacy_task_config[f"hpc_{key}"] = value
                    
            # Add GCS configuration if available  
            if context.gcs_config:
                for key, value in context.gcs_config.items():
                    legacy_task_config[f"gcs_{key}"] = value
            
            # Set data source
            legacy_task_config['data_source'] = source_config.get('name')
            
            # Run the legacy process_task function
            process_task(legacy_task_config)
            
            logger.info(f"Preprocessing task completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in preprocessing task: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return False
    
    @staticmethod
    def handle_validate(source_config: Dict[str, Any], context: PreprocessWorkflowContext, task_config: Dict[str, Any]):
        """Handle validation task."""
        logger.info(f"Starting validation task for {source_config.get('name', 'unknown')}")
        
        try:
            # Prepare task configuration for validation
            legacy_task_config = task_config.copy()
            
            # Set preprocessor name and mode
            legacy_task_config['preprocessor'] = source_config.get('name')
            legacy_task_config['mode'] = 'validate'
            
            # Add HPC configuration if available
            if context.hpc_config:
                for key, value in context.hpc_config.items():
                    legacy_task_config[f"hpc_{key}"] = value
                    
            # Add GCS configuration if available
            if context.gcs_config:
                for key, value in context.gcs_config.items():
                    legacy_task_config[f"gcs_{key}"] = value
            
            # Set data source
            legacy_task_config['data_source'] = source_config.get('name')
            
            # Run the legacy process_task function in validation mode
            process_task(legacy_task_config)
            
            logger.info(f"Validation task completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in validation task: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return False


def run_workflow_with_config(config: Dict[str, Any]):
    """
    Main entry point for running unified preprocessing workflow with configuration.
    This function mirrors the structure of workflow_unified.py for consistency.
    
    Args:
        config: Configuration dictionary containing all workflow settings
    """
    logger.info("Starting unified preprocessing workflow")
    
    try:
        # Extract configuration sections
        source_config = config.get('source', {})
        preprocess_config = config.get('preprocess', {})
        workflow_config = config.get('workflow', {})
        hpc_config = config.get('hpc', {})
        gcs_config = config.get('gcs', {})
        
        # Handle case where source name might be passed separately
        if 'source_name' in config and not any(k in source_config for k in ['name', 'dataset_name', 'source_name', 'type']):
            source_config['name'] = config['source_name']
        
        # Create workflow context
        context = PreprocessWorkflowContext(
            hpc_config=hpc_config,
            gcs_config=gcs_config
        )
        
        logger.info(f"Created preprocessing workflow context for source: {source_config.get('name', 'unknown')}")
        
        # Execute tasks in order
        tasks = workflow_config.get('tasks', [])
        task_handlers = PreprocessTaskHandlers()
        
        for task in tasks:
            task_type = task.get('type')
            task_config = task.get('config', {})
            
            logger.info(f"Executing preprocessing task: {task_type}")
            
            if task_type == 'preprocess':
                success = task_handlers.handle_preprocess(source_config, context, task_config)
            elif task_type == 'validate':
                success = task_handlers.handle_validate(source_config, context, task_config)
            else:
                logger.error(f"Unknown preprocessing task type: {task_type}")
                success = False
            
            if not success:
                logger.error(f"Preprocessing task {task_type} failed")
                return False
        
        logger.info("Unified preprocessing workflow completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in unified preprocessing workflow: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return False


def run_workflow(config_path: Union[str, Path]) -> None:
    """
    Run the complete preprocessing workflow defined in the configuration file.
    
    Args:
        config_path: Path to the workflow configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Expand environment variables in the config
    config = expand_env_vars(config)
    
    # Get workflow tasks
    tasks = config.get("tasks", [])
    if not tasks:
        logger.warning("No tasks defined in the workflow configuration")
        return
    
    # Process each task in order
    logger.info(f"Starting workflow with {len(tasks)} tasks")
    
    for i, task in enumerate(tasks):
        task_type = task.get("mode", "preprocess")
        preprocessor = task.get("preprocessor", "unknown")
        logger.info(f"Task {i+1}/{len(tasks)}: {task_type} {preprocessor}")
        process_task(task)
    
    logger.info("Workflow completed successfully")


def handle_sigterm(signum, frame):
    """Handle SIGTERM signal gracefully."""
    logger.info("Received SIGTERM signal, shutting down...")
    sys.exit(0)

# Register the handler
signal.signal(signal.SIGTERM, handle_sigterm)
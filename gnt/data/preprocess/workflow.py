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
        # Get preprocessor name
        preprocessor_name = task_config.pop("preprocessor")
        
        # Create preprocessor instance using the factory
        preprocessor = create_preprocessor(preprocessor_name, task_config)
        
        # Log task start
        logger.info(f"Starting {preprocessor_name} with stage '{preprocessor.stage}'")
        
        # Run preprocessing
        start_time = datetime.now()
        preprocessor.preprocess()
        duration = datetime.now() - start_time
        
        # Log completion
        logger.info(f"Completed {preprocessor_name} in {duration}")
        
    except Exception as e:
        logger.error(f"Error processing task with {task_config.get('preprocessor', 'unknown')}: {str(e)}")
        raise


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
        logger.info(f"Task {i+1}/{len(tasks)}: {task.get('preprocessor', 'unknown')}")
        process_task(task)
    
    logger.info("Workflow completed successfully")


def handle_sigterm(signum, frame):
    """Handle SIGTERM signal gracefully."""
    logger.info("Received SIGTERM signal, shutting down...")
    sys.exit(0)

# Register the handler
signal.signal(signal.SIGTERM, handle_sigterm)
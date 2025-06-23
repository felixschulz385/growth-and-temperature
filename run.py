#!/usr/bin/env python3
"""
Unified entry point for the GNT data system.

This script provides a command-line interface to run both download and preprocessing
workflows with a common configuration structure.

Usage examples:
  # Run download workflow for a specific source
  python run.py download --config config.yaml --source glass
  
  # Build or update the index for a specific source
  python run.py index --config config.yaml --source glass
  
  # Run preprocessing for a specific source
  python run.py preprocess --config config.yaml --source glass
  
  # Run validation for a specific source
  python run.py validate --config config.yaml --source glass
  
  # Extract transferred batches for a specific source
  python run.py extract --config config.yaml --source glass
"""

import os
import sys
import argparse
import logging
import re
from pathlib import Path
import yaml
import json
import tempfile
from typing import Dict, Any, Optional, Union

# Add the project root directory to Python's module search path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to Python path: {project_root}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("main")


def load_config_with_env_vars(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML/JSON configuration file with environment variable expansion."""
    env_pattern = re.compile(r'\${([^}^{]+)}')
    
    # Function to replace environment variables in strings
    def replace_env_vars(value: str) -> str:
        def replace(match):
            env_var = match.group(1)
            return os.environ.get(env_var, '')
        return env_pattern.sub(replace, value)
    
    # Function to process all strings in a nested structure
    def process_item(item):
        if isinstance(item, dict):
            return {k: process_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [process_item(i) for i in item]
        elif isinstance(item, str):
            return replace_env_vars(item)
        else:
            return item
    
    # Load and process the YAML/JSON file
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() == '.json':
            config = json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
    
    # Expand all environment variables
    return process_item(config)


def merge_configs(global_config: Dict[str, Any], source_config: Dict[str, Any], 
                 operation_type: str) -> Dict[str, Any]:
    """Merge global and source-specific configurations."""
    result = {}
    
    # For index operation, use download settings if no specific index settings
    if operation_type == "index" and "index" not in global_config and "index" not in source_config:
        # Start with global download config
        if 'download' in global_config:
            result.update(global_config['download'])
        # Override with source-specific download config
        if 'download' in source_config:
            result.update(source_config['download'])
    else:
        # Start with global operation-specific config
        if operation_type in global_config:
            result.update(global_config[operation_type])
    
    # Add global common settings that apply to all operations
    if 'common' in global_config:
        result.update(global_config['common'])
    
    # Override with source-specific operation config
    if operation_type in source_config:
        result.update(source_config[operation_type])
    
    # Add source-specific parameters that aren't operation-specific
    for key, value in source_config.items():
        if key not in ['index', 'download', 'preprocess', 'validate']:
            result[key] = value
    
    return result


def setup_logging(level: str, log_file: Optional[str] = None, debug: bool = False):
    """Configure logging with the specified level and output file."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # If debug mode is enabled, override to DEBUG level
    if debug:
        numeric_level = logging.DEBUG
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(stdout_handler)
    
    # Add file handler if specified
    if log_file:
        log_dir = Path(os.path.dirname(log_file))
        if log_dir and str(log_dir) != ".":
            log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(file_handler)
    
    logger.debug("Logging configured successfully")


def check_environment(operation_type: str) -> bool:
    """Check if the environment is properly set up for the requested operation."""
    try:
        if operation_type in ["download", "index"]:
            import requests
            import bs4
            
        elif operation_type == "preprocess":
            import xarray
            import rioxarray
            import numpy
            import pandas

        # Common packages for all operations
        import yaml
        
        logger.debug(f"Environment check passed for {operation_type} operation")
        return True
        
    except ImportError as e:
        logger.error(f"Environment check failed for {operation_type} operation: {str(e)}")
        logger.error("Please ensure all required packages are installed.")
        return False


def run_operation(operation_type: str, source: str, config: Dict[str, Any], mode: str = None):
    """
    Run the specified data operation for a source.
    
    Args:
        operation_type: Type of operation ('index', 'download', 'preprocess', 'validate', 'extract')
        source: Data source name
        config: Full configuration dictionary
        mode: Override mode (optional)
    """
    # Ensure the source exists in configuration
    if 'sources' not in config or source not in config['sources']:
        raise ValueError(f"Source '{source}' not found in configuration")

    # Merge global and source-specific configurations
    source_config = config['sources'][source]
    merged_config = merge_configs(config, source_config, operation_type)
    
    # Add source name to the merged config
    merged_config['data_source'] = source
    
    # Set mode based on operation type
    if mode:
        # Mode explicitly provided as argument
        merged_config['mode'] = mode
    else:
        # Set default mode based on operation type
        if operation_type == "index":
            merged_config['mode'] = "index"
        elif operation_type == "download":
            merged_config['mode'] = "download"
        elif operation_type == "validate_download":
            merged_config['mode'] = "validate_download"
        elif operation_type == "extract":
            merged_config['mode'] = "extract"  # Added extract mode
        else:
            merged_config['mode'] = operation_type
    
    # Add HPC/GCS configuration if available
    if 'hpc' in config:
        for key, value in config['hpc'].items():
            merged_config[f"hpc_{key}"] = value
            
    if 'gcs' in config:
        for key, value in config['gcs'].items():
            merged_config[f"gcs_{key}"] = value
    
    # Determine which module to use
    if operation_type in ["download", "index", "extract", "validate_download"]:  # Added "extract" to this condition
        # Import the download workflow - use lazy import to avoid circular dependencies
        try:
            # Use a lazy import approach to avoid circular imports
            import importlib
            hpc_module = importlib.import_module('gnt.data.download.hpc_workflow')
            
            # Check if the newer function exists
            if hasattr(hpc_module, 'run_hpc_workflow_with_config'):
                logger.info("Using new HPC workflow interface")
                hpc_module.run_hpc_workflow_with_config(merged_config)
            else:
                # Fall back to legacy interface using task format
                logger.info("Using legacy HPC workflow interface")
                
                # Convert config back to task format
                task = merged_config.copy()
                task['data_source'] = source
                task['mode'] = merged_config['mode']
                
                # Create a task-based config structure
                task_config = {
                    "hpc_target": merged_config.get("hpc_target", ""),
                    "tasks": [task]
                }
                
                # Use a temporary file for the task config
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as temp_file:
                    yaml.dump(task_config, temp_file)
                    temp_path = temp_file.name
                
                try:
                    hpc_module.run_hpc_workflow(temp_path)
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        
        except ImportError as e:
            logger.error(f"Error importing HPC workflow module: {e}")
            raise ValueError(f"Could not import HPC workflow module: {e}") from e
    
    elif operation_type in ["preprocess", "validate_preprocess"]:
        # For preprocess/validate, need to convert to task format expected by existing workflow
        task = merged_config.copy()
        task['preprocessor'] = source
        
        # Import the preprocess workflow
        try:
            from gnt.data.preprocess.workflow import process_task
            process_task(task)
        except ImportError as e:
            logger.error(f"Error importing preprocess workflow module: {e}")
            raise ValueError(f"Could not import preprocess workflow module: {e}") from e
    
    else:
        raise ValueError(f"Unsupported operation type: {operation_type}")


def main():
    """Main entry point for the unified data system."""
    parser = argparse.ArgumentParser(
        description="GNT Data System - Unified entry point for download and preprocessing operations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main operation type argument
    parser.add_argument(
        "operation",
        choices=["index", "download", "preprocess", "validate_download", "extract"],  # Added "extract"
        help="Operation to perform"
    )
    
    # Common arguments
    parser.add_argument(
        "--config", 
        required=True,
        help="Path to unified configuration file (YAML or JSON)"
    )
    
    parser.add_argument(
        "--source",
        required=True,
        help="Data source name (must be defined in config)"
    )
    
    parser.add_argument(
        "--mode",
        help="Override operation mode (for advanced usage)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )
    
    parser.add_argument(
        "--log-file",
        help="Path to the log file"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Set up logging
        log_file = args.log_file or f"{args.operation}-{args.source}.log"
        setup_logging(args.log_level, log_file, args.debug)
        
        logger.info(f"Starting GNT {args.operation} system for {args.source}")
        
        # Check environment for requested operation
        if not check_environment(args.operation):
            logger.critical(f"Exiting due to environment check failure")
            return 1
        
        # Load configuration
        config = load_config_with_env_vars(args.config)
        
        # Run the operation
        run_operation(args.operation, args.source, config, args.mode)
        
        logger.info(f"{args.operation.title()} operation for {args.source} completed successfully")
        return 0
        
    except Exception as e:
        logger.exception(f"Unexpected error occurred: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
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
            # Handle default values in environment variables
            if ':-' in env_var:
                var_name, default_value = env_var.split(':-', 1)
                return os.environ.get(var_name, default_value)
            return os.environ.get(env_var, '')
        return env_pattern.sub(replace, value)
    
    # Function to process all strings in a nested structure
    def process_item(item):
        if isinstance(item, dict):
            return {k: process_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [process_item(i) for i in item]
        elif isinstance(item, str):
            expanded = replace_env_vars(item)
            # Try to convert numeric strings to numbers
            if expanded != item:  # Only if expansion occurred
                try:
                    # Try integer first
                    if expanded.isdigit() or (expanded.startswith('-') and expanded[1:].isdigit()):
                        return int(expanded)
                    # Try float
                    return float(expanded)
                except ValueError:
                    # Return as string if conversion fails
                    return expanded
            return expanded
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

def run_operation(operation_type: str, source: str, config: Dict[str, Any], mode: str = None, stage: str = None, override_level: int = None, cli_overrides: Dict[str, Any] = None):
    """
    Run the specified data operation for a source.
    
    Args:
        operation_type: Type of operation ('index', 'download', 'preprocess', 'validate', 'extract', 'assemble', 'demean')
        source: Data source name or assembly name for assemble/demean operations
        config: Full configuration dictionary
        mode: Override mode (optional)
        stage: Stage for preprocess operation (optional)
        override_level: Override level for demeaning (0=none, 1=results, 2=intermediate+results)
        cli_overrides: CLI parameter overrides (optional)
    """
    
    if operation_type == "assemble":
        # Handle assembly operations differently
        assembly_config = config.copy()  # Pass full config instead of subset
        assembly_config['assembly_name'] = source
        
        # Apply CLI overrides for assembly operations
        if cli_overrides:
            assembly_config['cli_overrides'] = cli_overrides
        
        # Import and run the assembly workflow
        try:
            import importlib
            assembly_module = importlib.import_module('gnt.data.assemble.workflow')
            
            logger.info(f"Running assembly workflow: {source}")
            assembly_module.run_workflow_with_config(assembly_config)
                        
        except ImportError as e:
            logger.error(f"Error importing assembly module: {e}")
            raise ValueError(f"Could not import assembly module: {e}") from e
        
        return
    
    if operation_type == "demean":
        # Handle demeaning operations
        demean_config = config.copy()  # Pass full config
        
        # Import and run the demeaning workflow
        try:
            import importlib
            demean_module = importlib.import_module('gnt.data.assemble.demean')
            
            logger.info(f"Running demeaning workflow for assembly: {source} (override_level={override_level or 0})")
            demean_module.run_workflow_with_config(demean_config, assembly_name=source, override_level=override_level or 0)
                        
        except ImportError as e:
            logger.error(f"Error importing demeaning module: {e}")
            raise ValueError(f"Could not import demeaning module: {e}") from e
        
        return
    
    # Ensure the source exists in configuration
    if 'sources' not in config or source not in config['sources']:
        raise ValueError(f"Source '{source}' not found in configuration")

    # Get source-specific configuration
    source_config = config['sources'][source].copy()
    
    # Determine which module to use
    if operation_type in ["download", "index", "extract", "validate_download"]:
        # Build HPC workflow configuration structure
        hpc_workflow_config = {
            'source': source_config,
            'index': {
                'local_dir': config.get('hpc', {}).get('local_index_dir'),
                'rebuild': operation_type == 'index',
                'only_missing_entrypoints': True,
                'sync_direction': 'auto'
            },
            'workflow': {
                'tasks': []
            },
            'hpc': config.get('hpc', {}),
            'source_name': source
        }
        
        # Define workflow tasks based on operation type
        if operation_type == "index":
            hpc_workflow_config['workflow']['tasks'] = [
                {
                    'type': 'index',
                    'config': {
                        'rebuild': False,
                        'only_missing_entrypoints': True,
                        'sync_direction': 'auto'
                    }
                }
            ]
        elif operation_type == "download":
            # Check how many files are pending before starting
            pending_count = 0
            try:
                # Create a temporary index to check pending files
                temp_hpc_config = hpc_workflow_config.copy()
                temp_index = None
                
                # Try to get pending count for progress reporting
                from gnt.data.common.index.unified_index import UnifiedDataIndex
                from gnt.data.download.sources.factory import create_data_source
                
                temp_data_source = create_data_source(temp_hpc_config['source'])
                temp_index = UnifiedDataIndex(
                    bucket_name="",
                    data_source=temp_data_source,
                    local_index_dir=temp_hpc_config['index']['local_dir'],
                    key_file=temp_hpc_config['hpc'].get('key_file'),
                    hpc_mode=True
                )
                
                pending_count = temp_index.count_pending_files()
                logger.info(f"Found {pending_count} files pending download for {source}")
                
            except Exception as e:
                logger.debug(f"Could not get pending file count: {e}")
            
            # Extract download configuration
            download_config = {
                'batch_size': source_config.get('download', {}).get('batch_size', 50),
                'max_concurrent_downloads': source_config.get('download', {}).get('max_concurrent_downloads', 5),
                'tar_max_files': source_config.get('download', {}).get('tar_max_files', 100),
                'tar_max_size_mb': source_config.get('download', {}).get('tar_max_size_mb', 500)
            }
            
            hpc_workflow_config['workflow']['tasks'] = [
                {
                    'type': 'download',
                    'config': download_config
                }
            ]
        elif operation_type == "extract":
            logger.error("Extract operations are not currently implemented")
            return
        elif operation_type == "validate_download":
            logger.error("Validation operations are not currently implemented")
            return
        
        # Import and run the unified workflow
        try:
            import importlib
            workflow_module = importlib.import_module('gnt.data.download.workflow_unified')
            
            logger.info("Running unified workflow with structured configuration")
            workflow_module.run_workflow_with_config(hpc_workflow_config)
                        
        except ImportError as e:
            logger.error(f"Error importing unified workflow module: {e}")
            raise ValueError(f"Could not import unified workflow module: {e}") from e
    
    elif operation_type in ["preprocess", "validate_preprocess"]:
        # Build preprocessing workflow configuration structure
        preprocess_workflow_config = {
            'source': source_config,
            'preprocess': config.get('preprocess', {}),
            'workflow': {
                'tasks': []
            },
            'hpc': config.get('hpc', {}),
            'gcs': config.get('gcs', {}),
            'sources': config.get('sources', {}),  # Add sources configuration
            'source_name': source
        }
        
        # Define workflow tasks based on operation type
        if operation_type == "preprocess":
            task_config = preprocess_workflow_config['preprocess'].copy()
            task_config['mode'] = mode or 'preprocess'
            if stage:
                task_config['stage'] = stage
            
            preprocess_workflow_config['workflow']['tasks'] = [
                {
                    'type': 'preprocess',
                    'config': task_config
                }
            ]
        
        # Import and run the unified preprocessing workflow
        try:
            import importlib
            preprocess_workflow_module = importlib.import_module('gnt.data.preprocess.workflow')
            
            logger.info("Running unified preprocessing workflow with structured configuration")
            preprocess_workflow_module.run_workflow_with_config(preprocess_workflow_config)
                        
        except ImportError as e:
            logger.error(f"Error importing preprocessing workflow module: {e}")
            raise ValueError(f"Could not import preprocessing workflow module: {e}") from e
    
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
        choices=["index", "download", "preprocess", "validate_download", "extract", "assemble", "demean", "analysis"],
        help="Operation to perform"
    )
    
    # Common arguments
    parser.add_argument(
        "--config", 
        required=True,
        help="Path to unified configuration file (YAML or JSON)"
    )
    
    # Make source optional for analysis operations
    parser.add_argument(
        "--source",
        help="Data source name (must be defined in config) or assembly name for assemble operation"
    )
    
    # Analysis-specific arguments
    parser.add_argument(
        "--analysis-type",
        choices=['online_rls', 'online_2sls', 'list'],
        help="Type of analysis to run (required for analysis operation)"
    )
    
    parser.add_argument(
        "--specification", "-s",
        help="Analysis specification to use (for analysis operation)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output directory for analysis results"
    )
    
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=True,
        help="Enable verbose progress output (default: True)"
    )
    
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Disable verbose progress output"
    )
    
    parser.add_argument(
        "--subsource",
        help="Subsource name for misc preprocessor (e.g., 'osm', 'gadm')"
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
    
    parser.add_argument(
        "--stage",
        help="Processing stage for preprocess operation (e.g., annual, spatial, vector)"
    )

    # Dask configuration arguments (for preprocess and assemble operations)
    parser.add_argument('--dask-threads', type=int, help='Number of Dask threads (overrides config)')
    parser.add_argument('--dask-memory-limit', help='Dask memory limit (overrides config)')
    parser.add_argument('--temp-dir', help='Temporary directory (overrides config)')
    parser.add_argument('--dashboard-port', type=int, default=8787, help='Dask dashboard port')

    # Additional arguments for preprocessing and assembly
    parser.add_argument('--year', type=int, help='Specific year to process')
    parser.add_argument('--year-range', nargs=2, type=int, metavar=('START', 'END'),
                       help='Year range to process (start end)')
    parser.add_argument('--grid-cells', nargs='+', help='Grid cells to process (MODIS only)')
    parser.add_argument('--override', action='store_true', help='Override existing outputs')

    # Assembly-specific arguments
    parser.add_argument('--tile-size', type=int, help='Tile size for assembly operations (overrides config)')
    parser.add_argument('--compression', help='Compression format for parquet files (overrides config)')

    # Override level argument for demeaning operations
    parser.add_argument(
        '--override-level',
        type=int,
        choices=[0, 1, 2],
        default=0,
        help='Override level for demeaning (0=none, 1=remove results, 2=remove intermediate+results)'
    )

    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments based on operation type
    if args.operation == "analysis":
        if not args.analysis_type:
            parser.error("Analysis operation requires --analysis-type")
        if args.analysis_type == "online_rls" and not args.specification:
            parser.error("Online RLS analysis requires --specification")
        # For analysis, config should point to analysis.yaml by default
        if args.config == "config.yaml":  # Default was changed
            args.config = "orchestration/configs/analysis.yaml"
    else:
        if not args.source:
            parser.error(f"{args.operation} operation requires --source")
    
    try:
        # Set up logging
        log_file = args.log_file or f"{args.operation}-{args.source or 'analysis'}.log"
        setup_logging(args.log_level, log_file, args.debug)
        
        if args.operation == "analysis":
            logger.info(f"Starting GNT analysis system: {args.analysis_type}")
            
            # Load analysis configuration
            config = load_config_with_env_vars(args.config)
            
            # Import and run analysis
            from gnt.analysis.entrypoint import run_online_rls, run_online_2sls, list_analyses, setup_logging as analysis_setup_logging
            
            # Setup analysis-specific logging
            if args.debug:
                config.setdefault('logging', {})['level'] = 'DEBUG'
            analysis_setup_logging(config)
            
            # Set default output directory
            output_dir = args.output
            if not output_dir:
                output_dir = config.get('output', {}).get('base_path', 
                                                          f"{project_root}/data_nobackup/analysis")
            
            # Determine verbosity
            verbose = args.verbose and not args.quiet
            
            if args.analysis_type == 'list':
                list_analyses(config)
            elif args.analysis_type == 'online_rls':
                # Validate specification
                specs = config['analyses']['online_rls']['specifications']
                if args.specification not in specs:
                    logger.error(f"Unknown specification: {args.specification}")
                    logger.info(f"Available specifications: {list(specs.keys())}")
                    return 1
                
                run_online_rls(config, args.specification, output_dir, verbose)
            elif args.analysis_type == 'online_2sls':
                # Validate specification exists
                if 'online_2sls' not in config.get('analyses', {}):
                    logger.error("Online 2SLS analysis not configured in config file")
                    return 1
                
                specs = config['analyses']['online_2sls']['specifications']
                if args.specification not in specs:
                    logger.error(f"Unknown 2SLS specification: {args.specification}")
                    logger.info(f"Available 2SLS specifications: {list(specs.keys())}")
                    return 1
                
                run_online_2sls(config, args.specification, output_dir, verbose)
            else:
                logger.error(f"Analysis type '{args.analysis_type}' not yet implemented")
                return 1
        else:
            logger.info(f"Starting GNT {args.operation} system for {args.source}")
            
            # Load configuration
            config = load_config_with_env_vars(args.config)

            # Prepare CLI overrides dictionary
            cli_overrides = {}
            
            # Apply CLI overrides to configuration based on operation type
            if args.operation in ["preprocess", "assemble"]:
                # Apply CLI overrides to preprocess configuration section (for preprocess)
                if args.operation == "preprocess":
                    preprocess_config = config.setdefault('preprocess', {})
                    source_config = config.setdefault('sources', {}).setdefault(args.source, {})
                    
                    # Add subsource to preprocess config if specified
                    if args.subsource:
                        preprocess_config['subsource'] = args.subsource
                        logger.info(f"Setting subsource from CLI: {args.subsource}")
                    
                    # Override Dask configuration in preprocess config
                    if args.dask_threads is not None:
                        preprocess_config['dask_threads'] = args.dask_threads
                        logger.info(f"Overriding dask_threads from CLI: {args.dask_threads}")
                    if args.dask_memory_limit is not None:
                        preprocess_config['dask_memory_limit'] = args.dask_memory_limit
                        logger.info(f"Overriding dask_memory_limit from CLI: {args.dask_memory_limit}")
                    if args.temp_dir is not None:
                        preprocess_config['temp_dir'] = args.temp_dir
                        logger.info(f"Overriding temp_dir from CLI: {args.temp_dir}")
                    if args.dashboard_port != 8787:
                        preprocess_config['dashboard_port'] = args.dashboard_port
                        logger.info(f"Overriding dashboard_port from CLI: {args.dashboard_port}")
                    
                    # Apply preprocessing-specific overrides to source configuration
                    if args.year is not None:
                        source_config['year'] = args.year
                        logger.info(f"Overriding year from CLI: {args.year}")
                    if args.year_range is not None:
                        source_config['year_range'] = args.year_range
                        logger.info(f"Overriding year_range from CLI: {args.year_range}")
                    if args.grid_cells is not None:
                        source_config['grid_cells'] = args.grid_cells
                        logger.info(f"Overriding grid_cells from CLI: {args.grid_cells}")
                    if args.override:
                        source_config['override'] = True
                        logger.info("Override mode enabled from CLI")
                    
                    # Set stage if provided
                    if args.stage:
                        preprocess_config['stage'] = args.stage
                        logger.info(f"Setting stage from CLI: {args.stage}")
                
                # Prepare CLI overrides for assembly operations
                elif args.operation == "assemble":
                    # Collect Dask-related overrides
                    if args.dask_threads is not None:
                        cli_overrides['dask_threads'] = args.dask_threads
                    if args.dask_memory_limit is not None:
                        cli_overrides['dask_memory_limit'] = args.dask_memory_limit
                    if args.temp_dir is not None:
                        cli_overrides['temp_dir'] = args.temp_dir
                    if args.dashboard_port != 8787:
                        cli_overrides['dashboard_port'] = args.dashboard_port
                    
                    # Collect assembly-specific overrides
                    if args.tile_size is not None:
                        cli_overrides['tile_size'] = args.tile_size
                        logger.info(f"Overriding tile_size from CLI: {args.tile_size}")
                    if args.compression is not None:
                        cli_overrides['compression'] = args.compression
                        logger.info(f"Overriding compression from CLI: {args.compression}")

            # Run the operation
            run_operation(args.operation, args.source, config, args.mode, getattr(args, "stage", None), args.override_level, cli_overrides)
        
        logger.info(f"{args.operation.title()} operation completed successfully")
        return 0
        
    except Exception as e:
        logger.exception(f"Unexpected error occurred: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()

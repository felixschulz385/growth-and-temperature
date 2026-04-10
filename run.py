#!/usr/bin/env python3
"""
Unified entry point for the GNT data system.

This script provides a command-line interface to run both download and preprocessing
workflows with a common configuration structure.

Usage examples:
  # ── Download / preprocess ─────────────────────────────────────────────────
  python run.py download --config config.yaml --source glass
  python run.py index    --config config.yaml --source glass
  python run.py preprocess --config config.yaml --source glass

  # ── Analysis ──────────────────────────────────────────────────────────────
  # List available models
  python run.py analysis --list-models

  # Run a single model on this machine (no SLURM)
  python run.py analysis --model my_model

  # Submit one or more tables/models as a SLURM batch job
  python run.py submit --tables table_main table_robustness
  python run.py submit --tables table_main --mem 64GB --time 4:00:00

  # ── Tables ────────────────────────────────────────────────────────────────
  # Status overview of all tables and their last results
  python run.py summary

  # Render table files (HTML + LaTeX) to output/analysis/tables
  python run.py tables
  python run.py tables --source table_main --formats html latex

  # ── Maintenance ───────────────────────────────────────────────────────────
  python run.py cleanup
  python run.py cleanup --dry-run
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
from gnt.config.runtime import (
    get_legacy_hpc_compat_config,
    get_paths_config,
    get_remote_config,
    resolve_local_index_dir,
)

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


_NOISY_GEO_LOGGERS = (
    "rasterio",
    "rasterio.env",
    "rasterio._env",
    "rasterio._warp",
    "rasterio._base",
)


def load_config_with_env_vars(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML/JSON/Excel configuration file with environment variable expansion."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    # Analysis configs (Excel) are handled by workflow.load_config
    if config_path.suffix.lower() in ['.xlsx', '.xls']:
        from gnt.analysis.workflow import load_config as _load_analysis_config
        return _load_analysis_config(config_path)

    # YAML / JSON loading
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
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() == '.json':
            config = json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {config_path.suffix}")

    if config_path.suffix.lower() in ['.yaml', '.yml']:
        local_config_path = config_path.with_name(
            f"{config_path.stem}.local{config_path.suffix}"
        )
        if local_config_path.exists():
            with open(local_config_path, 'r') as f:
                local_config = yaml.safe_load(f) or {}
            config = _deep_merge_dicts(config, local_config)
    
    # Expand all environment variables
    return process_item(config)


def setup_logging(level: str, log_file: Optional[str] = None, debug: bool = False):
    """Configure logging with the specified level (SLURM handles file output)."""
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
    
    # Add stdout handler only (SLURM handles file output)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(stdout_handler)

    # Keep rasterio/GDAL chatter quiet even in debug mode.
    for name in _NOISY_GEO_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
    
    logger.debug("Logging configured successfully")

def run_operation(operation_type: str, source: str, config: Dict[str, Any], mode: str = None, stage: str = None, override_level: int = None, cli_overrides: Dict[str, Any] = None):
    """
    Run the specified data operation for a source.
    
    Args:
        operation_type: Type of operation ('index', 'download', 'preprocess', 'validate', 'extract', 'assemble', 'demean', 'tables')
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
    
    elif operation_type == "tables":
        try:
            from gnt.analysis import AnalysisConfig, generate_all_tables

            excel_path = (cli_overrides or {}).get('analysis_config') or None
            output_dir  = (cli_overrides or {}).get('output_dir')  or None
            output_formats = (cli_overrides or {}).get('formats') or None
            table_names = [source] if source else None

            cfg = AnalysisConfig(excel_path)
            logger.info(f"Generating tables: {table_names or 'all'}")
            generate_all_tables(cfg, table_names=table_names, output_dir=output_dir,
                                output_formats=output_formats)
            logger.info("Table generation completed successfully")
        except Exception as e:
            logger.error(f"Error during table generation: {e}")
            raise

        return

    # Ensure the source exists in configuration
    if 'sources' not in config or source not in config['sources']:
        raise ValueError(f"Source '{source}' not found in configuration")

    # Get source-specific configuration
    source_config = config['sources'][source].copy()
    
    # Filter misc files if specified via CLI
    if source == 'misc' and cli_overrides and 'misc_files' in cli_overrides:
        misc_files_filter = cli_overrides['misc_files']
        if misc_files_filter:
            logger.info(f"Filtering misc files to: {misc_files_filter}")
            # Filter the sources dictionary to only include specified files
            all_sources = source_config.get('sources', {})
            filtered_sources = {k: v for k, v in all_sources.items() if k in misc_files_filter}
            
            if not filtered_sources:
                logger.warning(f"No matching misc files found for filter: {misc_files_filter}")
                logger.info(f"Available misc files: {list(all_sources.keys())}")
                return
            
            source_config['sources'] = filtered_sources
            logger.info(f"Will download {len(filtered_sources)} misc files: {list(filtered_sources.keys())}")
    
    # Determine which module to use
    if operation_type in ["download", "index", "extract", "validate_download"]:
        # Build HPC workflow configuration structure
        paths_config = get_paths_config(config)
        remote_config = get_remote_config(config)
        hpc_workflow_config = {
            'source': source_config,
            'index': {
                'local_dir': resolve_local_index_dir(config),
                'rebuild': operation_type == 'index',
                'only_missing_entrypoints': True,
                'sync_direction': 'auto'
            },
            'workflow': {
                'tasks': []
            },
            'paths': paths_config,
            'remote': remote_config,
            'hpc': get_legacy_hpc_compat_config(config),
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
                    key_file=temp_hpc_config['remote'].get('key_file'),
                    hpc_mode=bool(temp_hpc_config['remote'].get('ssh_target'))
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
            'paths': get_paths_config(config),
            'remote': get_remote_config(config),
            'hpc': get_legacy_hpc_compat_config(config),
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


# ---------------------------------------------------------------------------
# New-style command dispatch helpers (added during CLI refactor)
# ---------------------------------------------------------------------------
_NEW_DOMAINS = {"download", "preprocess", "assemble", "analysis"}
_DOMAIN_SUBCMDS: dict = {
    "download":   {"index", "run"},
    "preprocess": {"run"},
    "assemble":   {"create", "update", "demean"},
    "analysis":   {"run", "submit", "summary", "tables", "cleanup"},
}
# Legacy top-level commands that map 1:1 to a new domain/subcommand
_LEGACY_SIMPLE: dict = {
    "index":   ("download",  "index"),
    "demean":  ("assemble",  "demean"),
    "submit":  ("analysis",  "submit"),
    "summary": ("analysis",  "summary"),
    "tables":  ("analysis",  "tables"),
    "cleanup": ("analysis",  "cleanup"),
}


def _dispatch_new_cli(new_argv: list) -> int:
    """Delegate to gnt.cli.main with the given argument list."""
    from gnt.cli.main import main as _new_main
    return _new_main(new_argv)


def main():
    """Main entry point for the unified data system.

    Supported command styles:

    *Legacy* (still works, will print deprecation notice for some commands):
      python run.py download   --config c.yaml --source s
      python run.py preprocess --config c.yaml --source s
      python run.py assemble   --config c.yaml --source s
      python run.py analysis   --model m
      python run.py submit     --tables t
      python run.py summary
      python run.py tables
      python run.py cleanup

    *New-style* (preferred):
      python run.py download   run    --config c.yaml --source s
      python run.py download   index  --config c.yaml --source s
      python run.py preprocess run    --config c.yaml --source s
      python run.py assemble   create --config c.yaml --source s
      python run.py assemble   update --config c.yaml --source s --datasource d
      python run.py assemble   demean --config c.yaml --source s
      python run.py analysis   run    --model m
      python run.py analysis   submit --tables t
      python run.py analysis   summary
      python run.py analysis   tables
      python run.py analysis   cleanup
    """
    # ── New-style dispatch ──────────────────────────────────────────────────
    # Detect domain/subcommand pairs and route to the new modular CLI.
    _argv = sys.argv[1:]
    if len(_argv) >= 1:
        _cmd = _argv[0]

        # Simple legacy remaps (single-verb → domain subcommand)
        if _cmd in _LEGACY_SIMPLE:
            import warnings
            _domain, _subcmd = _LEGACY_SIMPLE[_cmd]
            warnings.warn(
                f"'run.py {_cmd}' is deprecated; "
                f"use 'run.py {_domain} {_subcmd}' instead.",
                DeprecationWarning, stacklevel=2,
            )
            return _dispatch_new_cli([_domain, _subcmd] + _argv[1:])

        # Already new-style domain/subcommand
        if _cmd in _NEW_DOMAINS and len(_argv) >= 2:
            _subcmd = _argv[1]
            if _subcmd in _DOMAIN_SUBCMDS.get(_cmd, set()):
                return _dispatch_new_cli(_argv)

        # Legacy 'assemble --update' → 'assemble update'
        if _cmd == "assemble" and "--update" in _argv:
            import warnings
            warnings.warn(
                "'run.py assemble --update' is deprecated; "
                "use 'run.py assemble update' instead.",
                DeprecationWarning, stacklevel=2,
            )
            new_tail = [a for a in _argv[1:] if a != "--update"]
            return _dispatch_new_cli(["assemble", "update"] + new_tail)

        # Legacy 'download --config …' (no subcommand) → 'download run'
        if _cmd == "download" and (len(_argv) < 2 or _argv[1].startswith("-")):
            import warnings
            warnings.warn(
                "'run.py download' is deprecated; "
                "use 'run.py download run' instead.",
                DeprecationWarning, stacklevel=2,
            )
            return _dispatch_new_cli(["download", "run"] + _argv[1:])

        # Legacy 'assemble --config …' (create mode, no subcommand) → 'assemble create'
        if _cmd == "assemble" and (len(_argv) < 2 or _argv[1].startswith("-")):
            import warnings
            warnings.warn(
                "'run.py assemble' is deprecated; "
                "use 'run.py assemble create' instead.",
                DeprecationWarning, stacklevel=2,
            )
            new_tail = [a for a in _argv[1:] if a != "--create"]
            return _dispatch_new_cli(["assemble", "create"] + new_tail)

        # Legacy 'preprocess --config …' (no subcommand) → 'preprocess run'
        if _cmd == "preprocess" and (len(_argv) < 2 or _argv[1].startswith("-")):
            import warnings
            warnings.warn(
                "'run.py preprocess' is deprecated; "
                "use 'run.py preprocess run' instead.",
                DeprecationWarning, stacklevel=2,
            )
            return _dispatch_new_cli(["preprocess", "run"] + _argv[1:])

        # Legacy 'analysis --model …' (no subcommand) → 'analysis run'
        if _cmd == "analysis" and (len(_argv) < 2 or _argv[1].startswith("-")):
            import warnings
            warnings.warn(
                "'run.py analysis' is deprecated; "
                "use 'run.py analysis run' instead.",
                DeprecationWarning, stacklevel=2,
            )
            return _dispatch_new_cli(["analysis", "run"] + _argv[1:])

    # ── Legacy fallback ─────────────────────────────────────────────────────
    # Commands that were not translated above fall through to the original
    # implementation below (index, extract, validate_*).
    parser = argparse.ArgumentParser(
        description="GNT Data System - Unified entry point for download and preprocessing operations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main operation type argument
    parser.add_argument(
        "operation",
        choices=["index", "download", "preprocess", "validate_download", "validate_preprocess", "extract", "assemble", "demean", "analysis", "submit", "summary", "tables", "cleanup"],
        help="Operation to perform"
    )
    
    # Common arguments
    parser.add_argument(
        "--config", 
        help="Path to unified configuration file (YAML or JSON)"
    )
    
    # Make source optional for certain operations
    parser.add_argument(
        "--source",
        help="Data source name (must be defined in config), assembly name for assemble operation, or table name for tables operation"
    )
    
    # Analysis-specific arguments
    parser.add_argument(
        "--model", "-m",
        help="Model name to run for analysis operation"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available analysis models"
    )
    
    parser.add_argument(
        "--dataset",
        help="Override dataset path for analysis (overrides data_source in specification)"
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
        "--misc-files",
        nargs='+',
        help="Specific misc file keys to download (e.g., 'osm', 'gadm', 'hdi')"
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
        "--dry-run",
        action="store_true",
        help="For cleanup operation: show what would be deleted without actually deleting"
    )
    
    parser.add_argument(
        "--stage",
        help="Processing stage for preprocess operation (e.g., annual, spatial, vector)"
    )

    # Administrative level argument for PLAD preprocessing
    parser.add_argument(
        '--admin-level',
        type=int,
        choices=[1, 2],
        help='Administrative level for PLAD preprocessor (1 or 2) - overrides config'
    )

    # Dask configuration arguments (for preprocess and assemble operations)
    parser.add_argument('--dask-threads', type=int, help='Number of Dask threads (overrides config)')
    parser.add_argument('--dask-memory-limit', help='Dask memory limit per worker (overrides config, e.g., "4GB", "32GB")')
    parser.add_argument('--temp-dir', help='Temporary directory (overrides config)')
    parser.add_argument('--dashboard-port', type=int, default=8787, help='Dask dashboard port')
    parser.add_argument('--local-directory', help='Directory for Dask worker spilling (overrides config)')

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

    # Assembly mode arguments
    parser.add_argument(
        '--create',
        action='store_true',
        help='Assembly mode: recreate all tiles (default behavior)'
    )
    
    parser.add_argument(
        '--update',
        action='store_true',
        help='Assembly mode: update existing tiles with new datasource'
    )
    
    parser.add_argument(
        '--datasource',
        help='Datasource name to update (required with --update)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        default=None,
        help='Overwrite existing tiles in create mode (default: True). Use --no-overwrite to skip existing tiles.'
    )
    
    parser.add_argument(
        '--no-overwrite',
        dest='overwrite',
        action='store_false',
        help='Skip existing tiles in create mode instead of overwriting them'
    )

    # Table-specific arguments
    parser.add_argument(
        "--table-config",
        help="Path to table configuration file (YAML) - overrides default"
    )

    parser.add_argument(
        "--analysis-config",
        help="Path to analysis.xlsx (overrides orchestration/configs/analysis.xlsx)"
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory for generated tables (overrides config default)"
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["html", "latex", "tex"],
        metavar="FMT",
        help="Output formats for tables: html, latex, tex (default: from Excel or html)"
    )

    # Submit / SLURM arguments
    parser.add_argument(
        "--tables",
        nargs="+",
        metavar="TABLE_OR_MODEL",
        help="Table or model names to submit as a SLURM job (submit operation)"
    )

    parser.add_argument("--mem",            default="128GB",   help="SLURM memory (default: 128GB)")
    parser.add_argument("--time",           default=None,      help="SLURM time limit override (default: auto)")
    parser.add_argument("--qos",            default="1week",   help="SLURM QOS (default: 1week)")
    parser.add_argument("--partition",      default="scicore", help="SLURM partition (default: scicore)")
    parser.add_argument("--cpus-per-task",  type=int, default=8, help="SLURM CPUs per task (default: 8)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments based on operation type
    if args.operation == "analysis":
        if not args.list_models and not args.model:
            parser.error("Analysis operation requires --model <model_name> or --list-models")
        if not args.config or args.config == "config.yaml":
            args.config = "orchestration/configs/analysis.xlsx"
    elif args.operation == "submit":
        if not args.tables and not args.source:
            parser.error("submit operation requires --tables <table_or_model> [...]  (or --source for a single entry)")
    elif args.operation in ("tables", "summary", "cleanup"):
        pass
    else:
        if not args.config:
            parser.error(f"{args.operation} operation requires --config")
        if not args.source:
            parser.error(f"{args.operation} operation requires --source")
    
    try:
        # Set up logging (SLURM handles file output automatically)
        setup_logging(args.log_level, debug=args.debug)
        
        if args.operation == "cleanup":
            logger.info("Starting analysis results cleanup")

            from gnt.analysis import cleanup_analysis_results

            output_dir = args.output or str(Path(project_root) / "output" / "analysis")
            if not Path(output_dir).exists():
                logger.error(f"Output directory not found: {output_dir}")
                return 1

            logger.info(f"Cleaning up results in: {output_dir}")
            if args.dry_run:
                logger.info("DRY RUN MODE - no files will be deleted")

            cleanup_analysis_results(output_dir, dry_run=args.dry_run)
            
        elif args.operation == "analysis":
            logger.debug("Starting GNT analysis system: DuckReg")

            from gnt.analysis import AnalysisConfig, run_duckreg

            cfg = AnalysisConfig(args.config)
            output_dir = args.output or str(cfg.base_path)

            # Handle --list-models
            if args.list_models:
                model_names = cfg.get_model_names()
                if not model_names:
                    logger.info("No models found in configuration")
                    return 0
                print("\nAvailable models:")
                print("=" * 80)
                for model_name in model_names:
                    spec = cfg.get_model_spec(model_name)
                    print(f"\n{model_name}")
                    print(f"  Description: {spec.get('description', 'N/A')}")
                    print(f"  Data source: {spec.get('data_source', 'N/A')}")
                    print(f"  Formula    : {spec.get('formula', 'N/A')}")
                print("\n" + "=" * 80)
                return 0

            model_names = cfg.get_model_names()
            if args.model not in model_names:
                logger.error(f"Unknown model: {args.model}")
                logger.info(f"Available models: {model_names}")
                return 1

            logger.debug(f"Running model: {args.model}")
            run_duckreg(cfg, args.model, output_dir, dataset_override=args.dataset)

        elif args.operation == "submit":
            from gnt.analysis import AnalysisConfig
            from gnt.analysis.config import seconds_to_slurm_time, PROJECT_ROOT
            from gnt.analysis.slurm import (
                resolve_table_model_pairs, make_job_label,
                write_job_script, submit_job, ONE_WEEK_SECONDS,
            )
            try:
                from duckreg._version import __version__ as _duckreg_ver
            except ImportError:
                _duckreg_ver = "unknown"

            identifiers = args.tables or [args.source]
            cfg = AnalysisConfig(args.analysis_config or None)
            pairs, total_secs = resolve_table_model_pairs(identifiers, cfg)

            print(f"Total runtime across all identifiers: {seconds_to_slurm_time(total_secs)}")

            if total_secs > ONE_WEEK_SECONDS:
                logger.error(
                    f"Combined runtime {seconds_to_slurm_time(total_secs)} exceeds the 1-week "
                    "QOS limit. Split the tables across multiple jobs."
                )
                return 1

            slurm_time   = args.time or seconds_to_slurm_time(total_secs)
            job_label    = make_job_label(identifiers)
            slurm_kwargs = {
                'mem':           args.mem,
                'time':          slurm_time,
                'qos':           args.qos,
                'partition':     args.partition,
                'cpus_per_task': args.cpus_per_task,
            }

            print(f"Creating job script... (duckreg {_duckreg_ver})")
            job_path = write_job_script(pairs, PROJECT_ROOT, job_label, _duckreg_ver)

            print("Submitting job to SLURM...")
            job_id = submit_job(job_path, slurm_kwargs)

            total_models = sum(len(m) for _, m in pairs)
            print(f"\nJob submitted successfully!")
            print(f"  Job ID  : {job_id}")
            print(f"  Labels  : {', '.join(identifiers)}")
            print(f"  Models  : {total_models}")
            print(f"  duckreg : {_duckreg_ver}")
            print(f"  Memory  : {args.mem}")
            print(f"  Time    : {slurm_time}")
            print(f"  QOS     : {args.qos}")

            import os; os.remove(job_path)

        elif args.operation == "summary":
            from gnt.analysis import AnalysisConfig, summarize_tables

            cfg = AnalysisConfig(args.analysis_config or None)
            summarize_tables(cfg)

        elif args.operation == "tables":
            logger.info("Starting GNT table generation system")

            cli_overrides = {}
            if args.analysis_config:
                cli_overrides['analysis_config'] = args.analysis_config
            if args.output_dir:
                cli_overrides['output_dir'] = args.output_dir
            if args.formats:
                cli_overrides['formats'] = args.formats

            run_operation(args.operation, args.source, {}, None, None, None, cli_overrides)
        else:
            logger.info(f"Starting GNT {args.operation} system for {args.source}")
            
            # Load configuration
            config = load_config_with_env_vars(args.config)

            # Prepare CLI overrides dictionary
            cli_overrides = {}
            
            # Add misc files filter to CLI overrides
            if hasattr(args, 'misc_files') and args.misc_files:
                cli_overrides['misc_files'] = args.misc_files
            
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
                    if args.local_directory is not None:
                        preprocess_config['local_directory'] = args.local_directory
                        logger.info(f"Overriding local_directory from CLI: {args.local_directory}")
                    
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
                    
                    # Set admin_level if provided (for PLAD preprocessor)
                    if args.admin_level is not None:
                        source_config['admin_level'] = args.admin_level
                        logger.info(f"Overriding admin_level from CLI: {args.admin_level}")
                
                # Prepare CLI overrides for assembly operations
                elif args.operation == "assemble":
                    # Validate assembly mode arguments
                    if args.update and not args.datasource:
                        logger.error("--update mode requires --datasource")
                        return 1
                    if args.update and args.create:
                        logger.error("Cannot use both --update and --create")
                        return 1
                    
                    # Set assembly mode (default to create)
                    if args.update:
                        cli_overrides['assembly_mode'] = 'update'
                        cli_overrides['datasource'] = args.datasource
                        logger.info(f"Assembly mode: UPDATE datasource '{args.datasource}'")
                    else:
                        cli_overrides['assembly_mode'] = 'create'
                        logger.info("Assembly mode: CREATE (recreate all tiles)")
                    
                    # Collect Dask-related overrides
                    if args.dask_threads is not None:
                        cli_overrides['dask_threads'] = args.dask_threads
                    if args.dask_memory_limit is not None:
                        cli_overrides['dask_memory_limit'] = args.dask_memory_limit
                        logger.info(f"Overriding dask_memory_limit from CLI: {args.dask_memory_limit}")
                    if args.temp_dir is not None:
                        cli_overrides['temp_dir'] = args.temp_dir
                    if args.dashboard_port != 8787:
                        cli_overrides['dashboard_port'] = args.dashboard_port
                    if args.local_directory is not None:
                        cli_overrides['local_directory'] = args.local_directory
                        logger.info(f"Overriding local_directory from CLI: {args.local_directory}")
                    
                    # Assembly-specific overrides
                    if args.tile_size is not None:
                        cli_overrides['tile_size'] = args.tile_size
                        logger.info(f"Overriding tile_size from CLI: {args.tile_size}")
                    if args.compression is not None:
                        cli_overrides['compression'] = args.compression
                        logger.info(f"Overriding compression from CLI: {args.compression}")
                    if args.overwrite is not None:
                        cli_overrides['overwrite'] = args.overwrite
                        logger.info(f"Overriding overwrite from CLI: {args.overwrite}")

            # Run the operation
            run_operation(args.operation, args.source, config, args.mode, getattr(args, "stage", None), args.override_level, cli_overrides)
        
        logger.info(f"{args.operation.title()} operation completed successfully")
        return 0
        
    except Exception as e:
        logger.exception(f"Unexpected error occurred: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


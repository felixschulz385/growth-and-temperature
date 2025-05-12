#!/usr/bin/env python3
"""
Main entry point for the geodata download system.

This script provides a command-line interface to run download workflows
with various options for configuration and execution.
"""

import os
import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Optional

from workflow import run_workflow  # Changed from 'run' to 'run_workflow'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Geodata download system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "config",
        help="Path to the workflow configuration file (YAML or JSON)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )
    
    parser.add_argument(
        "--log-file",
        default="download.log",
        help="Path to the log file"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    
    return parser.parse_args()


def setup_logging(level: str, log_file: Optional[str] = None):
    """Configure logging with the specified level and output file."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
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


def parse_bool_env(key, default=False):
    """Parse boolean environment variable with fallback."""
    val = os.environ.get(key, str(default)).strip().lower()
    return val in ('true', 't', 'yes', 'y', '1')


def check_environment():
    """Check if the environment is properly set up for downloading."""
    try:
        import requests
        import google.cloud.storage
        import bs4
        
        logger.debug("Environment check passed: All required packages are available")
        return True
    except ImportError as e:
        logger.error(f"Environment check failed: {str(e)}")
        logger.error("Please ensure all required packages are installed.")
        return False


def main():
    """Main entry point for the download system."""
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Set up logging
        setup_logging(args.log_level, args.log_file)
        
        # Configure debug logging if requested
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
        
        logger.info("Starting geodata download system")
        
        # Check environment
        if not check_environment():
            logger.critical("Exiting due to environment check failure")
            return 1
        
        # Validate config file path
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return 1
        
        # Run the workflow
        logger.info(f"Running download workflow with configuration from {config_path}")
        run_workflow(config_path)
        
        logger.info("Download completed successfully")
        return 0
        
    except Exception as e:
        logger.exception(f"Unexpected error occurred: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
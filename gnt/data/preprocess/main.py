#!/usr/bin/env python3
"""
Main entry point for the geodata preprocessing system.

This script provides a command-line interface to run preprocessing workflows
with various options for configuration and execution.
"""

import os
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from workflow import run_workflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("preprocess.log")
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Geodata preprocessing system",
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
        default="preprocess.log",
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
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(file_handler)
    
    logger.debug("Logging configured successfully")
    

def setup_kubernetes_logging():
    """Configure logging with Kubernetes metadata if available."""
    k8s_metadata = {}
    
    # Get pod metadata from environment variables
    for var in ['HOSTNAME', 'KUBERNETES_SERVICE_HOST', 'KUBERNETES_POD_NAME', 
                'KUBERNETES_POD_NAMESPACE']:
        if var in os.environ:
            k8s_metadata[var] = os.environ[var]
    
    if k8s_metadata:
        logger.info(f"Running in Kubernetes environment: {k8s_metadata}")


def check_environment():
    """Check if the environment is properly set up for preprocessing."""
    try:
        import yaml
        import xarray
        import rioxarray
        import numpy
        import pandas
        
        logger.debug("Environment check passed: All required packages are available")
        return True
    except ImportError as e:
        logger.error(f"Environment check failed: {str(e)}")
        logger.error("Please ensure all required packages are installed.")
        return False


def main():
    """Main entry point for the preprocessing system."""
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Set up logging
        setup_logging(args.log_level, args.log_file)
        
        # Configure debug logging if requested
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
        
        logger.info("Starting geodata preprocessing system")
        
        # Set up Kubernetes logging if applicable
        setup_kubernetes_logging()
        
        # Check environment
        if not check_environment():
            logger.critical("Exiting due to environment check failure")
            return 1  # Use return instead of sys.exit() inside functions
        
        # Validate config file path
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return 1
        
        # Run the workflow
        logger.info(f"Running workflow with configuration from {config_path}")
        run_workflow(config_path)
        
        logger.info("Processing completed successfully")
        return 0
        
    except Exception as e:
        logger.exception(f"Unexpected error occurred: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
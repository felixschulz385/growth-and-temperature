#!/usr/bin/env python3
"""
Complete Download Pipeline Script

This script automates the entire download process for a specified data source,
executing the following steps in sequence:
1. Index - Build/update the data source index
2. Download - Download files and transfer to HPC
3. Extract - Extract the transferred files on HPC
4. Validate - Validate the downloaded files on HPC

Usage:
    python download_pipeline.py --config <config_file> --source <source_name> [options]

Example:
    python download_pipeline.py --config orchestration/configs/data.yaml --source glass_modis
"""

import os
import sys
import argparse
import logging
import time
import subprocess
from pathlib import Path
from typing import Optional, List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("download_pipeline")


class DownloadPipeline:
    """Orchestrates the complete download pipeline for a data source."""
    
    def __init__(self, config_path: str, source: str, debug: bool = False, 
                 skip_steps: Optional[List[str]] = None):
        """
        Initialize the download pipeline.
        
        Args:
            config_path: Path to the configuration file
            source: Name of the data source
            debug: Enable debug mode
            skip_steps: List of steps to skip ('index', 'download', 'extract', 'validate')
        """
        self.config_path = Path(config_path)
        self.source = source
        self.debug = debug
        self.skip_steps = skip_steps or []
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    def run(self) -> bool:
        """
        Execute the complete download pipeline.
        
        Returns:
            bool: True if all steps completed successfully, False otherwise
        """
        logger.info(f"Starting download pipeline for source: {self.source}")
        logger.info(f"Using configuration: {self.config_path}")
        
        steps = [
            ("index", "Building/updating index"),
            ("download", "Downloading files"),
            ("extract", "Extracting files on HPC"),
            ("validate_download", "Validating downloaded files")
        ]
        
        success = True
        start_time = time.time()
        
        for step_name, step_description in steps:
            if step_name in self.skip_steps:
                logger.info(f"Skipping step: {step_description}")
                continue
                
            logger.info(f"Step: {step_description}")
            step_start = time.time()
            
            step_success = self._run_step(step_name)
            
            step_duration = time.time() - step_start
            if step_success:
                logger.info(f"Step completed in {step_duration:.1f} seconds: {step_description}")
            else:
                logger.error(f"Step failed after {step_duration:.1f} seconds: {step_description}")
                success = False
                break
        
        total_duration = time.time() - start_time
        if success:
            logger.info(f"All steps completed successfully in {total_duration:.1f} seconds")
        else:
            logger.error(f"Pipeline failed after {total_duration:.1f} seconds")
        
        return success
    
    def _run_step(self, operation: str) -> bool:
        """
        Run a single step of the pipeline.
        
        Args:
            operation: Operation type ('index', 'download', 'extract', 'validate_download')
            
        Returns:
            bool: True if the operation completed successfully, False otherwise
        """
        # Build command
        cmd = [
            sys.executable,
            "run.py",
            operation,
            "--config", str(self.config_path),
            "--source", self.source
        ]
        
        # Add debug flag if enabled
        if self.debug:
            cmd.append("--debug")
        
        # Generate a log file for this step
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = log_dir / f"{self.source}_{operation}_{timestamp}.log"
        
        logger.info(f"Running: {' '.join(cmd)}")
        logger.info(f"Log file: {log_file}")
        
        try:
            # Run the command and capture output
            with open(log_file, "w") as log_out:
                process = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False  # Don't raise exception on non-zero exit
                )
            
                # Write output to log file
                log_out.write(process.stdout)
                
                # Stream important output to console
                for line in process.stdout.splitlines():
                    if any(level in line for level in ["ERROR", "CRITICAL", "WARNING"]):
                        logger.warning(line)
            
            # Check if process was successful
            if process.returncode != 0:
                logger.error(f"Process failed with exit code {process.returncode}")
                logger.error(f"See log file for details: {log_file}")
                return False
                
            return True
            
        except Exception as e:
            logger.exception(f"Error running {operation}: {e}")
            return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Execute complete download pipeline for a data source",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        required=True,
        help="Path to configuration file (YAML or JSON)"
    )
    
    parser.add_argument(
        "--source",
        required=True,
        help="Data source name (must be defined in config)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    
    parser.add_argument(
        "--skip",
        choices=["index", "download", "extract", "validate"],
        nargs="+",
        help="Skip specific steps in the pipeline"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        pipeline = DownloadPipeline(
            config_path=args.config,
            source=args.source,
            debug=args.debug,
            skip_steps=args.skip
        )
        
        success = pipeline.run()
        return 0 if success else 1
    
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
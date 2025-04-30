# main.py
import sys
import json
import logging
import os
from pathlib import Path

from gnt.data.download.workflow import run
from gnt.data.download.sources.factory import create_data_source

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_bool_env(key, default=False):
    """Parse boolean environment variable with fallback."""
    val = os.environ.get(key, str(default)).strip().lower()
    return val in ('true', 't', 'yes', 'y', '1')

def main(config_path=None):
    try:
        # Require a config file path
        if not config_path or not os.path.exists(config_path):
            logger.error("Configuration file required but not found. Please provide a valid JSON config file path.")
            return 1
            
        # Config file approach
        logger.info(f"Reading configuration from: {config_path}")
        with open(config_path) as f:
            config = json.load(f)
        
        # Validate config
        if config.get('processing_mode') != 'download':
            logger.error(f"Invalid processing mode: {config.get('processing_mode')}")
            return 1
            
        # Extract parameters
        dataset = config.get('dataset')
        parameters = config.get('parameters', {})
        
        # Configure debug logging if requested
        if parameters.get('debug', False):
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
        
        # Extract config parameters with defaults
        gcs_bucket = parameters.get('gcs_bucket', 'growthandheat')
        base_url = parameters.get('base_url')
        file_extensions = parameters.get('file_extensions', '.hdf').split(',')
        max_concurrent_downloads = int(parameters.get('max_concurrent_downloads', 2))
        max_queue_size = int(parameters.get('max_queue_size', 8))
        output_path = parameters.get('output_path', '')
        download_limit = int(parameters.get('download_limit', 0))
        
        # Get workflow config options
        auto_index = parse_bool_env('AUTO_INDEX', parameters.get('auto_index', True))
        build_index_only = parse_bool_env('BUILD_INDEX_ONLY', parameters.get('build_index_only', False))
        
        # Create data source using factory
        data_source = create_data_source(
            dataset_name=dataset,
            base_url=base_url,
            file_extensions=file_extensions,
        )
        
        # Log the configuration
        logger.info(f"Configuration: auto_index={auto_index}, build_index_only={build_index_only}")
        logger.info(f"Running workflow with max_concurrent_downloads={max_concurrent_downloads}, "
                  f"max_queue_size={max_queue_size}")
        
        # Run the workflow
        run(
            data_source, 
            gcs_bucket, 
            max_concurrent_downloads=max_concurrent_downloads,
            max_queue_size=max_queue_size,
            auto_index=auto_index,
            build_index_only=build_index_only
        )
        
        logger.info("Download process completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    # Get config path from command line argument
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Add file log handler if not in test mode
    if not parse_bool_env('LOCAL_TEST', False):
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / 'download.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
    
    sys.exit(main(config_path))
import logging
from typing import Dict, Any, List

from .registry import get_factory

logger = logging.getLogger(__name__)

def validate_required_params(config: Dict[str, Any], required: List[str]):
    """Validate that required parameters are present in config."""
    missing = [param for param in required if param not in config]
    if missing:
        raise ValueError(f"Missing required parameters: {', '.join(missing)}")

def create_data_source(source_config):
    """
    Create a data source instance based on configuration.

    Args:
        source_config: Configuration dictionary or string

    Returns:
        BaseDataSource: Instance of the appropriate data source
    """
    # Handle both dict and string inputs for backward compatibility
    if isinstance(source_config, dict):
        # Try multiple possible keys for dataset name
        dataset_name = (source_config.get('name') or
                       source_config.get('dataset_name') or
                       source_config.get('source_name') or
                       source_config.get('type'))

        if not dataset_name:
            # If no dataset name in config, check if there's only one key that could be the source name
            possible_keys = [k for k in source_config.keys() if k not in ['base_url', 'url', 'file_extensions', 'output_path']]
            if len(possible_keys) == 1:
                dataset_name = possible_keys[0]
            else:
                raise ValueError(f"Source configuration must contain 'name', 'dataset_name', 'source_name', or 'type' field. Available keys: {list(source_config.keys())}")

        config = source_config
    else:
        # Legacy string input
        dataset_name = source_config
        config = {}

    # Extract parameters from config
    # Check for dataset-specific configuration
    if dataset_name in config and isinstance(config[dataset_name], dict):
        # Dataset name is a key containing the configuration
        dataset_config = config[dataset_name]
        base_url = dataset_config.get('base_url', dataset_config.get('url'))
        file_extensions = dataset_config.get('file_extensions')
        output_path = dataset_config.get('output_path', dataset_config.get('data_path'))
    else:
        # Configuration is at the top level
        base_url = config.get('base_url', config.get('url'))
        file_extensions = config.get('file_extensions')
        output_path = config.get('output_path', config.get('data_path'))

    logger.debug(f"Creating data source for {dataset_name}")

    factory_fn = get_factory(dataset_name)
    return factory_fn(
        dataset_name,
        config,
        base_url=base_url,
        file_extensions=file_extensions,
        output_path=output_path,
        source_config=source_config if isinstance(source_config, dict) else config,
    )

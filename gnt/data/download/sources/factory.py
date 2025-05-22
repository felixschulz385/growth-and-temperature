import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

def validate_required_params(config: Dict[str, Any], required: List[str]):
    """Validate that required parameters are present in config."""
    missing = [param for param in required if param not in config]
    if missing:
        raise ValueError(f"Missing required parameters: {', '.join(missing)}")

def create_data_source(dataset_name: str, config: Dict[str, Any] = None) -> Any:
    """
    Factory function to create appropriate data source based on the dataset name
    
    Args:
        dataset_name: Name of the dataset
        config: Configuration dictionary containing parameters for the data source
        
    Returns:
        Data source instance
    """
    # Default empty config if None provided
    if config is None:
        config = {}
    
    # Import specific data sources here to avoid circular imports
    from gnt.data.download.sources.glass import GlassLSTDataSource
    from gnt.data.download.sources.eog import EOGDataSource
    from gnt.data.download.sources.misc import MiscDataSource
    
    logger.debug(f"Creating data source for {dataset_name}")
    
    # Create data source based on dataset name
    if dataset_name.lower() == 'misc':
        logger.info(f"Creating Misc data source")
        required_params = ["files"]
        validate_required_params(config, required_params)
        return MiscDataSource(
            files=config.get("files", []),
            output_path=config.get("output_path", "misc"),
            timeout=config.get("timeout", 60),
            chunk_size=config.get("chunk_size", 8192)
        )
    
    # For other data sources, require base_url
    if "base_url" not in config:
        raise ValueError(f"base_url must be provided in config for {dataset_name}")
    
    base_url = config.get("base_url")
    file_extensions = config.get("file_extensions", [])
    # Convert string to list if necessary
    if isinstance(file_extensions, str):
        file_extensions = [file_extensions]
        
    output_path = config.get("output_path")
    
    if dataset_name.lower().startswith('glass'):
        logger.info(f"Creating GLASS LST data source")
        return GlassLSTDataSource(
            base_url=base_url,
            file_extensions=file_extensions,
            output_path=output_path
        )
    elif dataset_name.lower().startswith('eog'):
        logger.info(f"Creating EOG data source")
        return EOGDataSource(
            base_url=base_url,
            file_extensions=file_extensions,
            output_path=output_path
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
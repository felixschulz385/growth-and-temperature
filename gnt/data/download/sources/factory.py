import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

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
    
    # Extract common parameters from config
    base_url = config.get("base_url")
    if not base_url:
        raise ValueError(f"base_url must be provided in config for {dataset_name}")
        
    file_extensions = config.get("file_extensions", [])
    # Convert string to list if necessary
    if isinstance(file_extensions, str):
        file_extensions = [file_extensions]
        
    output_path = config.get("output_path")
    
    # Import specific data sources here to avoid circular imports
    from gnt.data.download.sources.glass import GlassLSTDataSource
    from gnt.data.download.sources.eog import EOGDataSource
    
    logger.debug(f"Creating data source for {dataset_name} with base URL {base_url}")
    
    # Create data source based on dataset name
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
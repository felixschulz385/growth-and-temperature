import logging
from typing import Optional, List, Any

logger = logging.getLogger(__name__)

def create_data_source(dataset_name: str, base_url: str, file_extensions: List[str]) -> Any:
    """
    Factory function to create appropriate data source based on the dataset name
    
    Args:
        dataset_name: Name of the dataset
        base_url: Base URL for the data source
        file_extensions: List of file extensions to download
        output_path: Path within the bucket where files should be stored
        download_limit: Optional limit on the number of files to download (0 = no limit)
        
    Returns:
        Data source instance
    """
    # Import specific data sources here to avoid circular imports
    from gnt.data.download.sources.glass import GlassLSTDataSource
    from gnt.data.download.sources.eog import EOGDataSource
    
    # Create data source based on dataset name
    if dataset_name.startswith('glass'):
        logger.info(f"Creating GLASS LST data source")
        return GlassLSTDataSource(
            base_url=base_url,
            file_extensions=file_extensions,
        )
    elif dataset_name.startswith('eog'):
        logger.info(f"Creating EOG data source")
        return EOGDataSource(
            base_url=base_url,
            file_extensions=file_extensions,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
"""
Metadata handling for assembled datasets.

Provides functions for creating and reading assembly metadata files.
"""

import os
import logging
from typing import Dict, Any

import yaml

logger = logging.getLogger(__name__)


def create_assembly_metadata(
    output_path: str,
    assembly_config: Dict[str, Any],
) -> bool:
    """
    Create metadata YAML file for assembled parquet output.
    
    The metadata file documents the assembly configuration and format
    for reproducibility and downstream processing.
    
    Args:
        output_path: Directory path for metadata file
        assembly_config: Assembly configuration to document
        
    Returns:
        True if metadata was written successfully, False otherwise
    """
    try:
        metadata_path = os.path.join(output_path, '_metadata.yaml')
        
        metadata_dict = {
            'assembly_config': assembly_config,
            'output_format': 'parquet',
            'partitioning': 'ix/iy tiles',
            'scaling': 'Applied during zarr read with mask_and_scale=True',
            'description': 'Assembled dataset in tile-partitioned parquet format with automatic scaling'
        }
        
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata_dict, f, default_flow_style=False)
        
        logger.info(f"Metadata written to {metadata_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to create metadata file: {e}")
        return False


def read_assembly_metadata(output_path: str) -> Dict[str, Any]:
    """
    Read metadata YAML file from assembled dataset.
    
    Args:
        output_path: Directory path containing metadata file
        
    Returns:
        Metadata dictionary
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
    """
    metadata_path = os.path.join(output_path, '_metadata.yaml')
    
    with open(metadata_path, 'r') as f:
        return yaml.safe_load(f)

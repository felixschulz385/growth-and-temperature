"""
Metadata utilities for handling packed/unpacked data transformations.

This module provides functions to read assembly metadata and apply
unpacking transformations to pandas DataFrames based on the variable
metadata stored in _metadata.yaml files.
"""

import os
import yaml
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
import sys

logger = logging.getLogger(__name__)

def read_assembly_metadata(metadata_path: str) -> Dict[str, Any]:
    """
    Read assembly metadata from _metadata.yaml file.
    
    Args:
        metadata_path: Path to the _metadata.yaml file
        
    Returns:
        Dictionary containing the metadata
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        logger.debug(f"Successfully loaded metadata from {metadata_path}")
        return metadata
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {metadata_path}: {e}")
    except Exception as e:
        raise Exception(f"Error reading metadata file {metadata_path}: {e}")

def unpack_dataframe(df: pd.DataFrame, metadata_path: str, 
                    dataset_filter: Optional[Union[str, list]] = None,
                    inplace: bool = False) -> pd.DataFrame:
    """
    Transform a pandas DataFrame by unpacking variables according to metadata.
    
    Applies scale_factor and add_offset transformations to convert from
    packed integer values to actual physical values.
    
    Args:
        df: Input DataFrame to transform
        metadata_path: Path to _metadata.yaml file containing variable metadata
        dataset_filter: Optional filter to only process specific datasets.
                       Can be a string (single dataset) or list of strings.
        inplace: Whether to modify the DataFrame in place or return a copy
        
    Returns:
        Transformed DataFrame with unpacked values
        
    Example:
        >>> df_unpacked = unpack_dataframe(df, '/path/to/_metadata.yaml')
        >>> # Only unpack specific datasets
        >>> df_unpacked = unpack_dataframe(df, metadata_path, dataset_filter=['modis', 'viirs'])
    """
    # Read metadata
    try:
        metadata = read_assembly_metadata(metadata_path)
    except Exception as e:
        logger.error(f"Failed to read metadata: {e}")
        raise
    
    # Get variable metadata
    variable_metadata = metadata.get('variable_metadata', {})
    if not variable_metadata:
        logger.warning("No variable_metadata found in metadata file")
        return df if inplace else df.copy()
    
    # Filter datasets if specified
    if dataset_filter is not None:
        if isinstance(dataset_filter, str):
            dataset_filter = [dataset_filter]
        
        filtered_metadata = {}
        for dataset_name in dataset_filter:
            if dataset_name in variable_metadata:
                filtered_metadata[dataset_name] = variable_metadata[dataset_name]
            else:
                logger.warning(f"Dataset '{dataset_name}' not found in metadata")
        variable_metadata = filtered_metadata
    
    # Work on copy if not inplace
    if not inplace:
        df = df.copy()
    
    # Apply transformations with vectorized operations
    transformed_columns = []
    
    # Group columns by transformation parameters for batch processing
    transform_groups = {}
    
    for dataset_name, dataset_vars in variable_metadata.items():
        logger.debug(f"Processing dataset: {dataset_name}")
        
        for var_name, var_metadata in dataset_vars.items():
            if var_name not in df.columns:
                continue
            
            # Extract packing parameters
            scale_factor = var_metadata.get('scale_factor', 1.0)
            add_offset = var_metadata.get('add_offset', 0.0)
            fill_value = var_metadata.get('_FillValue', 0)
            
            # Skip if no transformation needed
            if scale_factor == 1.0 and add_offset == 0.0:
                logger.debug(f"Skipping {var_name} - no packing applied")
                continue
            
            # Group by transformation parameters
            transform_key = (scale_factor, add_offset, fill_value)
            if transform_key not in transform_groups:
                transform_groups[transform_key] = []
            transform_groups[transform_key].append(var_name)
    
    # Apply transformations in batches
    for (scale_factor, add_offset, fill_value), columns in transform_groups.items():
        logger.debug(f"Batch unpacking {len(columns)} columns: scale={scale_factor}, offset={add_offset}")
        
        # Process columns in batch for better cache efficiency
        for col_name in columns:
            # Get column data efficiently
            col_data = df[col_name].values
            
            # Use appropriate dtype for memory efficiency
            if col_data.dtype.kind in ['i', 'u']:  # Integer types
                col_data = col_data.astype(np.float32, copy=False)
            
            # Vectorized fill value masking
            if fill_value is not None:
                fill_mask = (col_data == fill_value)
                # Apply transformation efficiently
                if np.any(~fill_mask):
                    col_data[~fill_mask] = (col_data[~fill_mask] * scale_factor) + add_offset
                # Set fill values to NaN
                if np.any(fill_mask):
                    col_data[fill_mask] = np.nan
            else:
                # Apply transformation to all values
                col_data = (col_data * scale_factor) + add_offset
            
            # Update DataFrame
            df[col_name] = col_data
            transformed_columns.append(col_name)
    
    # if transformed_columns:
    #     logger.info(f"Successfully unpacked {len(transformed_columns)} columns: {transformed_columns}")
    # else:
    #     logger.info("No columns were transformed")
    
    return df

def get_variable_info(metadata_path: str, dataset_name: str = None, 
                     var_name: str = None) -> Dict[str, Any]:
    """
    Get information about variables from metadata file.
    
    Args:
        metadata_path: Path to _metadata.yaml file
        dataset_name: Optional dataset name to filter by
        var_name: Optional variable name to filter by
        
    Returns:
        Dictionary with variable information
    """
    metadata = read_assembly_metadata(metadata_path)
    variable_metadata = metadata.get('variable_metadata', {})
    
    if dataset_name and dataset_name in variable_metadata:
        dataset_vars = variable_metadata[dataset_name]
        if var_name and var_name in dataset_vars:
            return {f"{dataset_name}.{var_name}": dataset_vars[var_name]}
        else:
            return {f"{dataset_name}.{k}": v for k, v in dataset_vars.items()}
    elif var_name:
        # Search for variable across all datasets
        result = {}
        for ds_name, ds_vars in variable_metadata.items():
            if var_name in ds_vars:
                result[f"{ds_name}.{var_name}"] = ds_vars[var_name]
        return result
    else:
        # Return all variables
        result = {}
        for ds_name, ds_vars in variable_metadata.items():
            for v_name, v_info in ds_vars.items():
                result[f"{ds_name}.{v_name}"] = v_info
        return result

def pack_dataframe(df: pd.DataFrame, metadata_path: str,
                  dataset_filter: Optional[Union[str, list]] = None,
                  inplace: bool = False) -> pd.DataFrame:
    """
    Transform a pandas DataFrame by packing variables according to metadata.
    
    Applies inverse of scale_factor and add_offset transformations to convert
    from physical values to packed integer values.
    
    Args:
        df: Input DataFrame to transform
        metadata_path: Path to _metadata.yaml file containing variable metadata  
        dataset_filter: Optional filter to only process specific datasets
        inplace: Whether to modify the DataFrame in place or return a copy
        
    Returns:
        Transformed DataFrame with packed values
    """
    # Read metadata
    try:
        metadata = read_assembly_metadata(metadata_path)
    except Exception as e:
        logger.error(f"Failed to read metadata: {e}")
        raise
    
    # Get variable metadata
    variable_metadata = metadata.get('variable_metadata', {})
    if not variable_metadata:
        logger.warning("No variable_metadata found in metadata file")
        return df if inplace else df.copy()
    
    # Filter datasets if specified
    if dataset_filter is not None:
        if isinstance(dataset_filter, str):
            dataset_filter = [dataset_filter]
        
        filtered_metadata = {}
        for dataset_name in dataset_filter:
            if dataset_name in variable_metadata:
                filtered_metadata[dataset_name] = variable_metadata[dataset_name]
            else:
                logger.warning(f"Dataset '{dataset_name}' not found in metadata")
        variable_metadata = filtered_metadata
    
    # Work on copy if not inplace
    if not inplace:
        df = df.copy()
    
    # Apply transformations
    transformed_columns = []
    
    for dataset_name, dataset_vars in variable_metadata.items():
        logger.debug(f"Processing dataset: {dataset_name}")
        
        for var_name, var_metadata in dataset_vars.items():
            if var_name not in df.columns:
                continue
            
            # Extract packing parameters
            scale_factor = var_metadata.get('scale_factor', 1.0)
            add_offset = var_metadata.get('add_offset', 0.0)
            fill_value = var_metadata.get('_FillValue', 0)
            dtype = var_metadata.get('dtype', 'uint16')
            
            # Skip if no transformation needed
            if scale_factor == 1.0 and add_offset == 0.0:
                logger.debug(f"Skipping {var_name} - no packing applied")
                continue
            
            logger.debug(f"Packing {var_name}: scale={scale_factor}, offset={add_offset}, fill={fill_value}")
            
            # Get column data as numpy array for faster operations
            col_data = df[var_name].values.copy()
            
            # Identify NaN values once
            nan_mask = np.isnan(col_data)
            
            # Apply packing transformation to finite values only
            if np.any(~nan_mask):
                finite_data = col_data[~nan_mask]
                # packed_value = (actual_value - add_offset) / scale_factor
                finite_data = (finite_data - add_offset) / scale_factor
                # Round to nearest integer
                finite_data = np.round(finite_data)
                col_data[~nan_mask] = finite_data
            
            # Set NaN values to fill value
            if fill_value is not None and np.any(nan_mask):
                col_data[nan_mask] = fill_value
            
            # Convert to target dtype efficiently
            try:
                # Handle any remaining non-finite values by replacing with fill_value
                if np.any(~np.isfinite(col_data)):
                    col_data[~np.isfinite(col_data)] = fill_value if fill_value is not None else 0
                
                col_data = col_data.astype(dtype)
            except (ValueError, OverflowError) as e:
                logger.error(f"Error converting {var_name} to {dtype}: {e}")
                # Clip values to valid range for the dtype
                if dtype.startswith('uint'):
                    max_val = np.iinfo(dtype).max
                    col_data = np.clip(col_data, 0, max_val).astype(dtype)
                else:
                    col_data = col_data.astype(dtype)
            
            # Update DataFrame
            df[var_name] = col_data
            transformed_columns.append(var_name)
    
    if transformed_columns:
        logger.info(f"Successfully packed {len(transformed_columns)} columns: {transformed_columns}")
    else:
        logger.info("No columns were transformed")
    
    return df

if __name__ == "__main__":
    # Ad-hoc debugging: run unpack/pack on modis.parquet with its metadata

    metadata_path = "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/assembled/modis.parquet/_metadata.yaml"
    parquet_path = "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/assembled/modis.parquet/ix=0/iy=0/data.parquet"

    try:
        df = pd.read_parquet(parquet_path)
        df_subset = df.query("median>0")
        
        print("Original DataFrame head:")
        print(df_subset.head())
        print(df_subset.dtypes)

        df_unpacked = unpack_dataframe(df_subset, metadata_path)
        print("\nUnpacked DataFrame head:")
        print(df_unpacked.head())
        print(df_unpacked.dtypes)

        df_packed = pack_dataframe(df_unpacked, metadata_path)
        print("\nPacked DataFrame head:")
        print(df_packed.head())
    except Exception as e:
        print(f"Error during ad-hoc debugging: {e}", file=sys.stderr)
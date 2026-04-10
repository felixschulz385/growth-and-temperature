"""
Dataset loading functionality for data assembly.

Handles loading datasets from zarr format with filtering, type conversion,
and coordinate standardization.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.xr import ODCExtensionDa

from gnt.data.assemble.constants import (
    DEFAULT_CRS,
    EXCLUDED_VARIABLES,
    LAND_MASK_RELATIVE_PATHS,
    TIME_COORD,
    YEAR_COORD,
)
from gnt.data.assemble.utils import (
    convert_int_to_float32,
    apply_column_prefix,
)

logger = logging.getLogger(__name__)


def load_land_mask(hpc_root: str, land_mask_path: Optional[str] = None) -> Optional[xr.Dataset]:
    """
    Load land mask from zarr file.
    
    Args:
        hpc_root: HPC root directory
        land_mask_path: Explicit path to land mask, or None to auto-detect
        
    Returns:
        Land mask dataset or None if not found
    """
    try:
        if land_mask_path is None:
            potential_paths = [
                os.path.join(hpc_root, rel_path) 
                for rel_path in LAND_MASK_RELATIVE_PATHS
            ]
        else:
            potential_paths = [land_mask_path]
        
        for path in potential_paths:
            if os.path.exists(path):
                logger.info(f"Loading land mask from: {path}")
                land_mask_ds = xr.open_zarr(path, consolidated=False, chunks='auto')
                land_mask_ds = land_mask_ds.odc.assign_crs(DEFAULT_CRS)
                return land_mask_ds
        
        logger.warning(f"Land mask not found in any of: {potential_paths}")
        return None
        
    except Exception as e:
        logger.warning(f"Error loading land mask: {e}")
        return None


def _standardize_time_coordinate(ds: xr.Dataset, dataset_name: str) -> xr.Dataset:
    """
    Convert time coordinate to integer year coordinate.
    
    Args:
        ds: Dataset with potential time coordinate
        dataset_name: Name for logging
        
    Returns:
        Dataset with standardized year coordinate
    """
    if TIME_COORD in ds.coords:
        ds = ds.rename({TIME_COORD: YEAR_COORD})
        ds.coords[YEAR_COORD] = pd.Series(ds.coords[YEAR_COORD]).dt.year.astype("int16")
        logger.debug(f"Converted 'time' to 'year' coordinate for {dataset_name}")
    return ds


def _apply_year_filter(
    ds: xr.Dataset, 
    year_range: Optional[Tuple[int, int]], 
    dataset_name: str
) -> xr.Dataset:
    """
    Filter dataset to specified year range.
    
    Args:
        ds: Dataset to filter
        year_range: Tuple of (start_year, end_year) or None
        dataset_name: Name for logging
        
    Returns:
        Filtered dataset
    """
    if year_range is None:
        return ds
    
    if YEAR_COORD in ds.coords:
        ds = ds.sel(year=slice(year_range[0], year_range[1]))
        logger.info(f"Filtered {dataset_name} to year range {year_range[0]}-{year_range[1]}")
    
    return ds


def _select_columns(
    ds: xr.Dataset, 
    columns: Optional[List[str]], 
    dataset_name: str
) -> Optional[xr.Dataset]:
    """
    Select specified columns from dataset or all non-excluded variables.
    
    Args:
        ds: Dataset to filter
        columns: List of column names to keep, or None for all
        dataset_name: Name for logging
        
    Returns:
        Dataset with selected columns, or None if no columns available
    """
    if columns:
        available_vars = [var for var in columns if var in ds.data_vars]
        missing_vars = [var for var in columns if var not in ds.data_vars]
        
        if missing_vars:
            logger.warning(f"Dataset {dataset_name} missing specified columns: {missing_vars}")
        
        if not available_vars:
            logger.warning(f"No specified columns found in dataset {dataset_name}")
            return None
        
        return ds[available_vars]
    else:
        vars_to_keep = [var for var in ds.data_vars.keys() if var not in EXCLUDED_VARIABLES]
        if vars_to_keep:
            ds = ds[vars_to_keep]
            logger.debug(f"Dataset {dataset_name}: keeping all variables {vars_to_keep}")
        return ds


def _apply_fillna(
    ds: xr.Dataset,
    dataset_name: str,
    dataset_config: Dict[str, Any],
) -> xr.Dataset:
    """Apply per-dataset NA filling when configured or required by convention."""
    fillna_value = dataset_config.get("fillna")
    if fillna_value is None and dataset_name == "snl_mining":
        fillna_value = 0

    if fillna_value is not None:
        logger.info(f"Filling NA values in dataset {dataset_name} with {fillna_value}")
        ds = ds.fillna(fillna_value)

    return ds


def load_single_dataset(
    dataset_name: str,
    dataset_config: Dict[str, Any],
    year_range: Optional[Tuple[int, int]] = None,
) -> Optional[Tuple[str, xr.Dataset, Dict[str, Any]]]:
    """
    Load a single dataset with all transformations applied.
    
    Processing steps:
    1. Open zarr with automatic scaling
    2. Assign CRS
    3. Standardize time to year coordinate
    4. Apply year range filter
    5. Select columns
    6. Apply column prefix
    7. Convert int to float32
    
    Args:
        dataset_name: Name identifier for the dataset
        dataset_config: Configuration dict for the dataset
        year_range: Optional year range filter (start, end)
        
    Returns:
        Tuple of (name, dataset, config) or None if loading fails
    """
    zarr_path = dataset_config['path']
    resampling_method = dataset_config.get('resampling', 'mode')
    
    if not os.path.exists(zarr_path):
        logger.warning(f"Dataset path does not exist: {zarr_path}, skipping")
        return None
    
    logger.info(f"Loading dataset {dataset_name} from {zarr_path} (resampling: {resampling_method})")
    
    try:
        # Open zarr with automatic scaling and masking
        ds = xr.open_zarr(zarr_path, mask_and_scale=True, consolidated=False, chunks='auto')
        ds = ds.odc.assign_crs(DEFAULT_CRS)
        
        # Standardize coordinates
        ds = _standardize_time_coordinate(ds, dataset_name)
        
        # Apply filters
        ds = _apply_year_filter(ds, year_range, dataset_name)
        
        # Select columns
        ds = _select_columns(ds, dataset_config.get('columns'), dataset_name)
        if ds is None:
            return None

        # Fill missing values for datasets that need explicit defaults during assembly.
        ds = _apply_fillna(ds, dataset_name, dataset_config)
        
        # Apply column prefix
        column_prefix = dataset_config.get('column_prefix')
        if column_prefix:
            logger.info(f"Applying prefix '{column_prefix}' to dataset {dataset_name}")
            ds = apply_column_prefix(ds, column_prefix)
            logger.debug(f"Renamed variables: {list(ds.data_vars.keys())}")
        
        # Type conversions for consistent processing
        ds = convert_int_to_float32(ds)
        
        # Verify geobox metadata for reprojection
        if not hasattr(ds, 'odc') or ds.odc.geobox is None:
            raise ValueError(f"Dataset {dataset_name} lacks geobox metadata for alignment")
        
        # Store metadata
        ds.attrs['dataset_name'] = dataset_name
        ds.attrs['resampling_method'] = resampling_method
        
        logger.info(f"Loaded dataset {dataset_name}: {list(ds.data_vars.keys())}")
        return (dataset_name, ds, dataset_config)
        
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        return None


def load_all_datasets(
    assembly_config: Dict[str, Any],
    target_geobox,
    datasource_filter: Optional[str] = None,
) -> List[Tuple[str, xr.Dataset, Dict[str, Any]]]:
    """
    Load all datasets specified in assembly configuration.
    
    Args:
        assembly_config: Assembly configuration with datasets section
        target_geobox: Target geobox for alignment checks
        datasource_filter: Optional filter to load only specific datasource (for update mode)
        
    Returns:
        List of tuples (dataset_name, dataset_xr, dataset_config)
        
    Raises:
        ValueError: If no datasets could be loaded
    """
    logger.info("Loading datasets (source-by-source)...")
    
    datasets_config = assembly_config['datasets']
    processing_config = assembly_config.get('processing', {})
    target_resolution = processing_config.get('resolution')
    year_range = processing_config.get('year_range')
    
    if target_resolution:
        logger.info(f"Target resolution for assembly: {target_resolution}°")
    if year_range:
        logger.info(f"Year range filter: {year_range[0]} to {year_range[1]}")
    
    # Filter to specific datasource if requested
    if datasource_filter:
        if datasource_filter not in datasets_config:
            raise ValueError(f"Datasource '{datasource_filter}' not found in assembly configuration")
        datasets_config = {datasource_filter: datasets_config[datasource_filter]}
        logger.info(f"Filtering to single datasource: {datasource_filter}")
    
    loaded = []
    
    for dataset_name, dataset_config in datasets_config.items():
        result = load_single_dataset(dataset_name, dataset_config, year_range)
        if result is not None:
            name, ds, config = result
            
            # Log resolution info
            native_res = abs(ds.odc.geobox.resolution.x)
            if target_resolution:
                logger.debug(f"{name}: native res={native_res}, target res={target_resolution}")
            
            loaded.append(result)
    
    if not loaded:
        raise ValueError("No datasets could be loaded successfully")
    
    logger.info(f"Successfully loaded {len(loaded)} datasets")
    return loaded


def prepare_land_mask(land_mask_ds: xr.Dataset) -> xr.Dataset:
    """
    Prepare land mask dataset by standardizing time coordinate.
    
    Args:
        land_mask_ds: Raw land mask dataset
        
    Returns:
        Prepared land mask dataset
    """
    if land_mask_ds is not None and TIME_COORD in land_mask_ds.coords:
        land_mask_ds = land_mask_ds.rename({TIME_COORD: YEAR_COORD})
        land_mask_ds.coords[YEAR_COORD] = pd.Series(land_mask_ds.coords[YEAR_COORD]).dt.year.astype("int16")
        logger.debug("Converted 'time' to 'year' coordinate for land_mask")
    return land_mask_ds

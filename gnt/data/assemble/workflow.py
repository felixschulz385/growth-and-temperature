"""
Data assembly module for the GNT system.

This module provides functionality to merge multiple datasets using a tile-by-tile
approach based on configuration specifications, reading directly from zarr files
and outputting to parquet format with automatic scaling applied.
"""

import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import xarray as xr
from odc.geo import GeoboxTiles
from odc.geo.geobox import GeoBox
from odc.geo.xr import ODCExtensionDa
import math

# Import common utilities
from gnt.data.common.geobox import get_or_create_geobox
from gnt.data.common.dask.client import DaskClientContextManager

logger = logging.getLogger(__name__)

def _load_land_mask(hpc_root: str, land_mask_path: str = None) -> Optional[xr.Dataset]:
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
            # Auto-detect land mask path from misc preprocessor output
            potential_paths = [
                os.path.join(hpc_root, "misc", "processed", "stage_2", "osm", "land_mask.zarr"),
            ]
        else:
            potential_paths = [land_mask_path]
        
        for path in potential_paths:
            if os.path.exists(path):
                logger.info(f"Loading land mask from: {path}")
                land_mask_ds = xr.open_zarr(path, consolidated=False, chunks='auto')
                land_mask_ds = land_mask_ds.odc.assign_crs(4326)
                return land_mask_ds
        
        logger.warning(f"Land mask not found in any of: {potential_paths}")
        return None
        
    except Exception as e:
        logger.warning(f"Error loading land mask: {e}")
        return None

def _load_and_merge_datasets(
    assembly_config: Dict[str, Any],
    land_mask_ds: Optional[xr.Dataset] = None
) -> Tuple[xr.Dataset, Dict[str, List[str]]]:
    """
    Load and merge all zarr datasets into a single large xarray dataset.
    
    Args:
        assembly_config: Assembly configuration
        land_mask_ds: Optional land mask dataset
        
    Returns:
        Tuple of (merged dataset, mapping of resampling method to variable names)
    """
    logger.info("Loading and merging all zarr datasets...")
    
    datasets_config = assembly_config['datasets']
    processing_config = assembly_config.get('processing', {})
    target_resolution = processing_config.get('resolution')
    year_range = processing_config.get('year_range')
    
    if target_resolution:
        logger.info(f"Target resolution for assembly: {target_resolution}° (will be applied during tile extraction)")
    
    if year_range:
        logger.info(f"Year range filter: {year_range[0]} to {year_range[1]}")
    
    merged_datasets = []
    land_mask_for_inner_join = None
    resampling_map = {}  # Maps resampling method to list of {'vars': list, 'config': dict}
    
    # Process land mask separately for inner join
    if land_mask_ds is not None:
        logger.info("Preparing land mask for inner join")
        # Transform time to year if present in land mask
        if "time" in land_mask_ds.coords:
            land_mask_ds.coords["time"] = pd.Series(land_mask_ds.coords["time"]).dt.year.astype("int16")
        
        land_mask_ds.attrs['dataset_name'] = 'land_mask'
        land_mask_for_inner_join = land_mask_ds
    
    for dataset_name, dataset_config in datasets_config.items():
        zarr_path = dataset_config['path']
        resampling_method = dataset_config.get('resampling', 'mode')  # default to mode
        
        if not os.path.exists(zarr_path):
            logger.warning(f"Dataset path does not exist: {zarr_path}, skipping")
            continue
        
        logger.info(f"Loading dataset {dataset_name} from {zarr_path} (resampling: {resampling_method})")
        
        try:
            # Open zarr dataset with automatic scaling and masking
            ds = xr.open_zarr(zarr_path, mask_and_scale=True, consolidated=False, chunks='auto')
            
            # Apply year range filter if time/year dimension exists
            if year_range:
                if 'time' in ds.coords:
                    years = pd.Series(ds.coords['time']).dt.year
                    ds = ds[dict(time=years.between(year_range[0], year_range[1]))]
                    logger.info(f"Filtered {dataset_name} to year range {year_range[0]}-{year_range[1]}")
                elif 'year' in ds.coords:
                    ds = ds.sel(year=slice(year_range[0], year_range[1]))
                    logger.info(f"Filtered {dataset_name} to year range {year_range[0]}-{year_range[1]}")
            
            # Check and subset columns if specified
            columns = dataset_config.get('columns')
            if columns:
                # Check which columns exist in the dataset
                available_vars = [var for var in columns if var in ds.data_vars]
                missing_vars = [var for var in columns if var not in ds.data_vars]
                
                if missing_vars:
                    logger.warning(f"Dataset {dataset_name} missing specified columns: {missing_vars}")
                
                if available_vars:
                    ds = ds[available_vars]
                else:
                    logger.warning(f"No specified columns found in dataset {dataset_name}, skipping")
                    continue
            else:
                # Remove spatial_ref if present, keep all other variables
                vars_to_keep = [var for var in ds.data_vars.keys() if var != 'spatial_ref']
                if vars_to_keep:
                    ds = ds[vars_to_keep]
                    logger.debug(f"Dataset {dataset_name}: keeping all variables {vars_to_keep}")
            
            # Apply column prefix if specified
            column_prefix = dataset_config.get('column_prefix')
            if column_prefix:
                logger.info(f"Applying prefix '{column_prefix}' to dataset {dataset_name}")
                # Rename all data variables with prefix
                rename_dict = {var: f"{column_prefix}{var}" for var in ds.data_vars}
                ds = ds.rename(rename_dict)
                logger.debug(f"Renamed variables: {list(ds.data_vars.keys())}")
                
            # Transform BOOL to float32
            boolean_vars = [x for x, y in ds.dtypes.items() if y == np.dtype("bool")]
            if boolean_vars:
                if len(boolean_vars) == 1:
                   boolean_vars =  boolean_vars[0]
                ds[boolean_vars] = ds[boolean_vars].astype("float32")
            
            # Transform INT to float32
            int_vars = [x for x, y in ds.dtypes.items() if np.issubdtype(y, np.integer)]
            if int_vars:
                if len(int_vars) == 1:
                   int_vars =  int_vars[0]
                ds[int_vars] = ds[int_vars].astype("float32")
            
            # Track which variables belong to this resampling method, with config
            var_names = list(ds.data_vars.keys())
            if resampling_method not in resampling_map:
                resampling_map[resampling_method] = []
            resampling_map[resampling_method].append({'vars': var_names, 'config': dataset_config})
            
            # Add dataset identifier for provenance
            ds.attrs['dataset_name'] = dataset_name
            ds.attrs['resampling_method'] = resampling_method
            merged_datasets.append(ds)
            
            logger.info(f"Loaded dataset {dataset_name}: {var_names}")
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            continue
    
    if not merged_datasets:
        raise ValueError("No datasets could be loaded successfully")
    
    # Merge all datasets using outer join
    logger.info("Merging datasets with outer join...")
    try:
        if len(merged_datasets) == 1:
            merged_ds = merged_datasets[0]
        else:
            # Merge remaining datasets using outer join
            merged_ds = xr.merge(merged_datasets, compat='override', join="outer")
        
        logger.info(f"Successfully outer merged {len(merged_datasets)} datasets")
        logger.info(f"Dataset variables after outer merge: {list(merged_ds.data_vars.keys())}")
        logger.info(f"Dataset dimensions after outer merge: {dict(merged_ds.sizes)}")
        
        # Now apply inner join with land mask if provided
        if land_mask_for_inner_join is not None:
            logger.info("Applying inner join with land mask")
            merged_ds = xr.merge([merged_ds, land_mask_for_inner_join], compat='override', join="inner")
            logger.info(f"Dataset dimensions after inner join with land mask: {dict(merged_ds.sizes)}")
            
        # Add Spatial Reference
        merged_ds = merged_ds.rio.write_crs(4326)
        
        logger.info(f"Final dataset variables: {list(merged_ds.data_vars.keys())}")
        logger.info(f"Final dataset dimensions: {dict(merged_ds.sizes)}")
        logger.info(f"Resampling method mapping: {resampling_map}")
        
        return merged_ds, resampling_map
        
    except Exception as e:
        logger.error(f"Failed to merge datasets: {e}")
        raise
    
def _strip_remote_prefix(path):
    """Remove scp/ssh prefix like user@host: from paths."""
    if isinstance(path, str):
        import re
        return re.sub(r"^[^@]+@[^:]+:", "", path)
    return path

def _derive_hpc_root_from_config(assembly_config: Dict[str, Any], full_config: Dict[str, Any] = None) -> Optional[str]:
    """
    Derive hpc_root from assembly configuration, checking multiple sources.
    
    Args:
        assembly_config: Assembly configuration dictionary
        full_config: Full configuration dictionary containing HPC settings
        
    Returns:
        str: HPC root path or None if not found
    """
    # Check assembly config first for hpc settings
    hpc_config = full_config.get('hpc', {})
    hpc_target = hpc_config.get('target')
    
    if hpc_target:
        return _strip_remote_prefix(hpc_target)
    
    # Check full config for hpc settings
    if full_config:
        hpc_config = full_config.get('hpc', {})
        hpc_target = hpc_config.get('target')
        
        if hpc_target:
            return _strip_remote_prefix(hpc_target)
    
    logger.warning("Could not derive hpc_root from configuration")
    return None


def get_available_tiles(assembly_config: Dict[str, Any], target_geobox) -> List[Tuple[int, int]]:
    """Get all available tile combinations (ix, iy) using geobox tiling."""
    tile_size = assembly_config.get('processing', {}).get('tile_size', 2048)
    tiles = GeoboxTiles(target_geobox, (tile_size, tile_size))
    
    # Return all possible tile indices
    all_tiles = []
    for ix in range(tiles.shape[0]):
        for iy in range(tiles.shape[1]):
            all_tiles.append((ix, iy))
    
    logger.info(f"Generated {len(all_tiles)} tiles from geobox ({tiles.shape[0]}x{tiles.shape[1]})")
    return all_tiles

def create_assembly_metadata(
    output_path: str,
    assembly_config: Dict[str, Any]
) -> bool:
    """Create metadata YAML file for assembled parquet output."""
    try:
        # Write metadata YAML
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

def _load_datasets_for_processing(
    assembly_config: Dict[str, Any],
    target_geobox,
    land_mask_ds: Optional[xr.Dataset] = None
) -> List[Tuple[str, xr.Dataset, Dict[str, Any]]]:
    """
    Load datasets individually, apply basic filtering, and ensure they carry geobox metadata.
    
    This function loads each dataset as a separate xarray.Dataset rather than merging them upfront.
    This source-by-source approach allows for independent reprojection and winsorization per dataset
    before final concatenation at the tile level.
    
    Args:
        assembly_config: Assembly configuration
        target_geobox: Target geobox for alignment checks
        land_mask_ds: Optional land mask dataset (unused here, kept for interface symmetry)
        
    Returns:
        List of tuples (dataset_name, dataset_xr, dataset_config)
    """
    logger.info("Loading datasets (source-by-source)...")
    datasets_config = assembly_config['datasets']
    processing_config = assembly_config.get('processing', {})
    target_resolution = processing_config.get('resolution')
    year_range = processing_config.get('year_range')
    loaded = []

    for dataset_name, dataset_config in datasets_config.items():
        zarr_path = dataset_config['path']
        resampling_method = dataset_config.get('resampling', 'mode')
        if not os.path.exists(zarr_path):
            logger.warning(f"Dataset path does not exist: {zarr_path}, skipping")
            continue
        logger.info(f"Loading dataset {dataset_name} from {zarr_path} (resampling: {resampling_method})")
        try:
            ds = xr.open_zarr(zarr_path, mask_and_scale=True, consolidated=False, chunks='auto')
            ds = ds.odc.assign_crs(4326)
            
            # Convert time coordinate to integer years if present
            # This must happen before year range filtering
            if 'time' in ds.coords:
                ds = ds.rename(time='year')
                ds.coords["year"] = pd.Series(ds.coords["year"]).dt.year.astype("int16")
                logger.debug(f"Converted 'time' to 'year' coordinate for {dataset_name}")
            
            # Apply year range filter if time/year dimension exists
            if year_range:
                if 'year' in ds.coords:
                    ds = ds.sel(year=slice(year_range[0], year_range[1]))
                    logger.info(f"Filtered {dataset_name} to year range {year_range[0]}-{year_range[1]}")
            
            # Check and subset columns if specified
            columns = dataset_config.get('columns')
            if columns:
                # Check which columns exist in the dataset
                available_vars = [var for var in columns if var in ds.data_vars]
                missing_vars = [var for var in columns if var not in ds.data_vars]
                
                if missing_vars:
                    logger.warning(f"Dataset {dataset_name} missing specified columns: {missing_vars}")
                
                if available_vars:
                    ds = ds[available_vars]
                else:
                    logger.warning(f"No specified columns found in dataset {dataset_name}, skipping")
                    continue
            else:
                # Remove spatial_ref if present, keep all other variables
                vars_to_keep = [var for var in ds.data_vars.keys() if var != 'spatial_ref']
                if vars_to_keep:
                    ds = ds[vars_to_keep]
                    logger.debug(f"Dataset {dataset_name}: keeping all variables {vars_to_keep}")
            
            # Apply column prefix if specified
            column_prefix = dataset_config.get('column_prefix')
            if column_prefix:
                logger.info(f"Applying prefix '{column_prefix}' to dataset {dataset_name}")
                # Rename all data variables with prefix
                rename_dict = {var: f"{column_prefix}{var}" for var in ds.data_vars}
                ds = ds.rename(rename_dict)
                logger.debug(f"Renamed variables: {list(ds.data_vars.keys())}")
                
            # Transform BOOL to float32
            boolean_vars = [x for x, y in ds.dtypes.items() if y == np.dtype("bool")]
            if boolean_vars:
                if len(boolean_vars) == 1:
                   boolean_vars =  boolean_vars[0]
                ds[boolean_vars] = ds[boolean_vars].astype("float32")
            
            # Transform INT to float32
            int_vars = [x for x, y in ds.dtypes.items() if np.issubdtype(y, np.integer)]
            if int_vars:
                if len(int_vars) == 1:
                   int_vars =  int_vars[0]
                ds[int_vars] = ds[int_vars].astype("float32")
            
            # Verify dataset has geobox metadata needed for reprojection
            if not hasattr(ds, 'odc') or ds.odc.geobox is None:
                raise ValueError(f"Dataset {dataset_name} lacks geobox metadata for alignment")
            
            native_res = abs(ds.odc.geobox.resolution.x)
            if target_resolution:
                logger.debug(f"{dataset_name}: native res={native_res}, target res={target_resolution}")
            
            ds.attrs['dataset_name'] = dataset_name
            ds.attrs['resampling_method'] = resampling_method
            loaded.append((dataset_name, ds, dataset_config))
            logger.info(f"Loaded dataset {dataset_name}: {list(ds.data_vars.keys())}")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            continue

    if not loaded:
        raise ValueError("No datasets could be loaded successfully")
    return loaded

def winsorize(array):
    """
    Apply winsorization to clip outliers at 0.1% and 99.9% quantiles.
    Preserves NaN values throughout the operation.
    """
    return array.where(array > array.quantile(.001), array.quantile(.001)).where(array < array.quantile(.999), array.quantile(.999)).where(~array.isnull())

def _make_pixel_ids(ix: int, iy: int, tile_geobox) -> pd.DataFrame:
    """
    Generate pixel ID DataFrame with latitude, longitude, and pixel_id columns.
    
    Format: [ix: 16 bits | iy: 16 bits | local_pixel: 32 bits]
    This allows decoding tile coordinates and pixel location from a single integer.
    
    Args:
        ix: Tile x index
        iy: Tile y index
        tile_geobox: Target geobox for tile
    
    Returns:
        DataFrame with columns ['latitude', 'longitude', 'pixel_id']
    """
    h, w = tile_geobox.shape
    local_pixel_ids = np.arange(h * w, dtype="uint32").reshape((h, w))
    pixel_id_matrix = (np.uint64(ix) << 48) | (np.uint64(iy) << 32) | local_pixel_ids.astype(np.uint64)
    
    # Get coordinate arrays from geobox
    lats = tile_geobox.coords['latitude'].values
    lons = tile_geobox.coords['longitude'].values
    
    # Create meshgrid and flatten
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    
    # Create DataFrame with lat/lon/pixel_id
    pixel_id_df = pd.DataFrame({
        'latitude': lat_grid.flatten(),
        'longitude': lon_grid.flatten(),
        'pixel_id': pixel_id_matrix.flatten()
    })
    
    return pixel_id_df

def _extract_and_process_dataset_tile(
    ds: xr.Dataset,
    dataset_config: Dict[str, Any],
    resampling_method: str,
    ix: int,
    iy: int,
    padded_tile_geobox,
    target_geobox_zoomed
) -> Optional[pd.DataFrame]:
    """
    Extract and process a single dataset tile with winsorization and reprojection.
    
    Processing pipeline:
    1. Extract tile from padded bounds
    2. Apply winsorization if configured (clips outliers at 0.1% and 99.9% quantiles)
    3. Reproject to target resolution if needed
    4. Convert to DataFrame (without pixel_id)
    
    Args:
        ds: Source dataset
        dataset_config: Dataset-specific configuration including winsorize flag
        resampling_method: Resampling method for reprojection (bilinear, mode, nearest, etc.)
        ix, iy: Tile indices
        padded_tile_geobox: Padded geobox for extraction
        target_geobox_zoomed: Target geobox at desired resolution (or None if no resolution change)
        
    Returns:
        DataFrame with latitude/longitude coordinates, or None if tile is empty
    """
    try:
        bbox = padded_tile_geobox.boundingbox
        
        # Extract tile data with padding
        if 'latitude' in ds.coords and 'longitude' in ds.coords:
            tile_ds = ds.sel(latitude=slice(bbox.top, bbox.bottom), longitude=slice(bbox.left, bbox.right)).compute()
        else:
            logger.warning(f"Unknown coordinate system in dataset {dataset_config.get('path')}")
            return None
        
        if tile_ds.sizes.get('latitude', 0) == 0 or tile_ds.sizes.get('longitude', 0) == 0:
            return None
        
        # Apply winsorization before reprojection if configured
        # This reduces impact of outliers on resampling operations
        if dataset_config.get('winsorize', False):
            for var in tile_ds.data_vars:
                if np.issubdtype(tile_ds[var].dtype, np.floating):
                    tile_ds[var] = winsorize(tile_ds[var])
        
        # Apply resolution reprojection if target resolution differs from native
        if target_geobox_zoomed is not None and hasattr(tile_ds, 'odc'):
            logger.debug(f"Reprojecting tile [{ix}, {iy}] to target resolution")
            
            # Reproject using specified method (bilinear for continuous, mode for categorical)
            tile_ds = tile_ds.odc.reproject(
                target_geobox_zoomed,
                resampling=resampling_method,
                dst_nodata=np.nan
            )
            
            logger.debug(f"Reprojection complete for tile [{ix}, {iy}]: shape {tile_ds.dims}")
        
        # Convert to DataFrame with lat/lon coordinates preserved
        df = tile_ds.to_dataframe().reset_index()
        df = df.drop(columns=['band', 'spatial_ref'], errors='ignore')
        return df if not df.empty else None
    except Exception as e:
        logger.warning(f"Failed to extract tile [{ix}, {iy}] for dataset {dataset_config.get('path')}: {e}")
        return None

def process_tile(
    datasets: List[Tuple[str, xr.Dataset, Dict[str, Any]]],
    land_mask_ds: Optional[xr.Dataset],
    ix: int,
    iy: int,
    tile_geobox,
    assembly_config: Dict[str, Any],
    output_base_path: str
) -> bool:
    """
    Process a single tile across all datasets, merge on configured index_cols, and write to parquet.
    
    Workflow:
    1. Check if tile already exists (skip if valid)
    2. Determine target resolution and create padded/target geoboxes
    3. Create pixel ID DataFrame with lat/lon coordinates
    4. Extract and process each dataset independently with its configured resampling method
    5. Merge all DataFrames on configured index_cols using iterative merge operations
    6. Extract and apply land mask using right join to implicitly filter to land pixels (if configured)
    7. Write combined result to parquet with pixel_id as index
    
    This source-by-source approach allows different resampling methods for different datasets
    while maintaining proper spatial alignment through configured index columns.
    
    Args:
        datasets: List of (name, xr.Dataset, config) tuples
        land_mask_ds: Optional land mask for filtering to land pixels
        ix, iy: Tile indices
        tile_geobox: Target geobox for tile
        assembly_config: Assembly configuration
        output_base_path: Output directory base path
        
    Returns:
        True if tile was processed successfully, False if no data
    """
    logger.debug(f"Processing tile ix={ix}, iy={iy}")
    tile_output_path = os.path.join(output_base_path, f"ix={ix}", f"iy={iy}")
    output_file = os.path.join(tile_output_path, "data.parquet")

    # Skip tiles that already exist and are valid (resume on failure)
    if os.path.exists(output_file):
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(output_file)
            if parquet_file.metadata.num_rows > 0:
                logger.debug(f"Tile ix={ix}, iy={iy} already exists and is valid, skipping")
                return True
        except Exception as e:
            logger.warning(f"Tile ix={ix}, iy={iy} exists but appears corrupted ({e}), will reprocess")

    processing_config = assembly_config.get('processing', {})
    
    # Determine if resolution reprojection is needed
    target_resolution = processing_config.get('resolution')
    native_res = abs(tile_geobox.resolution.x)
    
    # Create padded tile geobox (64 pixels) to handle reprojection edge effects
    padded_tile_geobox = tile_geobox.pad(64, 64)
    
    # Create target geobox with new resolution if needed
    target_geobox_zoomed = None
    if target_resolution is not None and abs(native_res - target_resolution) >= 1e-10:
        logger.debug(f"Tile [{ix}, {iy}]: will reproject from {native_res}° to {target_resolution}°")
        target_geobox_zoomed = tile_geobox.zoom_to(resolution=target_resolution)
    else:
        logger.debug(f"Tile [{ix}, {iy}]: already at target resolution {target_resolution}°")
        target_geobox_zoomed = tile_geobox
    
    # Create pixel ID DataFrame with lat/lon as base
    pixel_id_df = _make_pixel_ids(ix, iy, target_geobox_zoomed)
    
    if pixel_id_df is None or pixel_id_df.empty:
        logger.warning(f"Failed to create pixel_id DataFrame for tile [{ix}, {iy}]")
        return False

    # Start with pixel_id_df as base (contains lat, lon, pixel_id)
    combined = pixel_id_df
    
    # Extract each dataset independently and merge on configured index_cols
    for dataset_name, ds, dataset_config in datasets:
        logger.debug(f"Tile [{ix}, {iy}]: starting processing of dataset '{dataset_name}'")
        
        resampling_method = dataset_config.get('resampling', 'mode')
        logger.debug(f"Tile [{ix}, {iy}]: dataset '{dataset_name}' - resampling method: {resampling_method}")
        
        # Get index_cols from dataset config, default to ['latitude', 'longitude']
        index_cols = dataset_config.get('index_cols', ['latitude', 'longitude'])
        logger.debug(f"Tile [{ix}, {iy}]: dataset '{dataset_name}' - index_cols: {index_cols}")
        
        df = _extract_and_process_dataset_tile(
            ds, dataset_config, resampling_method, ix, iy,
            padded_tile_geobox, target_geobox_zoomed
        )
        
        if df is not None and not df.empty:
            logger.debug(f"Tile [{ix}, {iy}]: dataset '{dataset_name}' - extracted {len(df)} rows, columns: {list(df.columns)}")
            
            # Find common columns between combined and df for merging
            merge_cols = [col for col in index_cols if col in combined.columns and col in df.columns]
            logger.debug(f"Tile [{ix}, {iy}]: dataset '{dataset_name}' - merge columns available: {merge_cols}")
            
            if not merge_cols:
                logger.warning(f"Tile [{ix}, {iy}]: dataset '{dataset_name}' - no common columns found between index_cols {index_cols} and available columns. Combined cols: {list(combined.columns)}, DF cols: {list(df.columns)}")
                continue
            
            # Merge on configured index columns
            rows_before = len(combined)
            combined = pd.merge(
                combined, df,
                on=merge_cols,
                how='outer'  # Keep all rows from combined (will filter with land mask later)
            )
            rows_after = len(combined)
            logger.debug(f"Tile [{ix}, {iy}]: dataset '{dataset_name}' - merged on {merge_cols}, rows before: {rows_before}, after: {rows_after}")
        else:
            logger.debug(f"Tile [{ix}, {iy}]: dataset '{dataset_name}' - extraction returned no data or empty dataframe")

    # Extract and apply land mask if configured - use RIGHT join as final filter
    if land_mask_ds is not None:
        logger.debug(f"Tile [{ix}, {iy}]: processing land_mask dataset")
        mask_df = _extract_and_process_dataset_tile(
            land_mask_ds, {'winsorize': False}, 'nearest', ix, iy, 
            padded_tile_geobox, target_geobox_zoomed
        )
        if mask_df is not None and 'land_mask' in mask_df.columns:
            # Filter to land pixels only
            mask_df = mask_df[mask_df.land_mask.astype(bool)]
            mask_df = mask_df.drop(columns=['land_mask'], errors='ignore')
            
            # Right join with land mask to implicitly filter combined to land pixels
            rows_before = len(combined)
            combined = pd.merge(
                combined, mask_df[['latitude', 'longitude']], 
                on=['latitude', 'longitude'], 
                how='right'  # Keep only land pixels
            )
            rows_after = len(combined)
            logger.debug(f"Tile [{ix}, {iy}]: land_mask right join applied - rows before: {rows_before}, after land filtering: {rows_after}")
        else:
            logger.debug(f"Tile [{ix}, {iy}]: land_mask extraction returned no data or no land_mask column")

    # Drop lat/lon columns and set pixel_id as index
    combined = combined.drop(columns=['latitude', 'longitude'], errors='ignore')
    
    if combined.empty:
        logger.debug(f"No data remaining in tile ix={ix}, iy={iy} after processing")
        return False

    # Write combined tile to parquet
    os.makedirs(tile_output_path, exist_ok=True)
    compression = processing_config.get('compression', 'snappy')
    logger.debug(f"Tile [{ix}, {iy}]: writing to parquet with compression={compression}, rows={len(combined)}, columns={len(combined.columns)}")
    combined.reset_index().to_parquet(output_file, index=False, compression=compression, engine='pyarrow')
    logger.info(f"Tile [{ix}, {iy}]: successfully written to {output_file}")
    return True

def _adjust_tile_size_for_reprojection(native_resolution: float, target_resolution: Optional[float], tile_size: int) -> int:
    """
    Ensure tile size is large enough to produce at least one output pixel after reprojection.
    
    When target resolution is coarser than native resolution, multiple input pixels map to
    one output pixel. This function ensures the tile is large enough to guarantee at least
    one output pixel after resampling.
    """
    if target_resolution is None:
        return tile_size
    min_tile_pixels = max(1, math.ceil(target_resolution / native_resolution))
    if tile_size < min_tile_pixels:
        logger.info(
            f"Increasing tile_size from {tile_size} to {min_tile_pixels} to cover at least one "
            f"reprojected pixel (native_res={native_resolution}, target_res={target_resolution})."
        )
    return max(tile_size, min_tile_pixels)

def run_assembly(assembly_config: Dict[str, Any], full_config: Dict[str, Any] = None):
    """
    Run the complete data assembly workflow.
    
    Main steps:
    1. Initialize Dask cluster for parallel tile processing
    2. Load all datasets individually (source-by-source approach)
    3. Create metadata YAML for provenance
    4. Process all tiles in parallel, each tile extracting/processing all datasets
    5. Write results as partitioned parquet files (ix/iy directory structure)
    
    The source-by-source approach allows:
    - Independent resampling methods per dataset (e.g., bilinear for temperature, mode for categorical)
    - Per-dataset winsorization configuration
    - Flexible alignment and concatenation at tile level
    """
    logger.info(f"Starting assembly: {assembly_config.get('description', 'Unknown')}")
    
    # Get output path - now expecting parquet directory output
    output_path = assembly_config['output_path']
    processing_config = assembly_config.get('processing', {})
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Output will be written to: {output_path}")
    
    # Try to derive hpc_root from configuration
    hpc_root = _derive_hpc_root_from_config(assembly_config, full_config)
        
    if not hpc_root:
        logger.error("hpc_root must be specified in processing config or derivable from dataset paths")
        return
        
    logger.info(f"Using hpc_root: {hpc_root}")
        
    target_geobox = get_or_create_geobox(hpc_root)

    # Ensure tile size can fit at least one pixel after reprojection
    processing_config.setdefault('tile_size', 2048)
    native_res = abs(target_geobox.resolution.x)
    processing_config['tile_size'] = _adjust_tile_size_for_reprojection(
        native_res,
        processing_config.get('resolution'),
        processing_config['tile_size']
    )

    # Get available tiles
    logger.info("Discovering available tiles...")
    all_tiles = get_available_tiles(assembly_config, target_geobox)
    logger.info(f"Found {len(all_tiles)} tiles to process")
    
    if not all_tiles:
        logger.warning("No tiles found to process")
        return
    
    # Create tile geobox for processing
    tile_size = processing_config.get('tile_size', 2048)
    tiles = GeoboxTiles(target_geobox, (tile_size, tile_size))
    
    # Configure Dask settings and create cluster
    dask_config = processing_config.get('dask', {})
    
    # Set up Dask cluster for all operations with optimized parameters
    dask_kwargs = {
        'threads': dask_config.get('threads'),
        'memory_limit': dask_config.get('memory_limit'),
        'dashboard_port': dask_config.get('dashboard_port', 8787),
        'temp_dir': dask_config.get('temp_dir'),
        'worker_threads_per_cpu': dask_config.get('worker_threads_per_cpu', 2),
        'worker_fraction': dask_config.get('worker_fraction', 0.5)
    }
    
    # Remove None values
    dask_kwargs = {k: v for k, v in dask_kwargs.items() if v is not None}
    
    logger.info("Creating Dask cluster for data loading and processing...")
    
    with DaskClientContextManager(**dask_kwargs) as client:
        logger.info(f"Dask client initialized: {client.dashboard_link}")
        
        # Load land mask if requested - will be inner joined with other datasets
        land_mask_ds = None
        if processing_config.get('apply_land_mask', False):
            land_mask_path = processing_config.get('land_mask_path')
            land_mask_ds = _load_land_mask(hpc_root, land_mask_path)
            
            # Convert time to year for land mask if present
            if land_mask_ds is not None and "time" in land_mask_ds.coords:
                land_mask_ds = land_mask_ds.rename(time='year')
                land_mask_ds.coords["year"] = pd.Series(land_mask_ds.coords["year"]).dt.year.astype("int16")
                logger.debug("Converted 'time' to 'year' coordinate for land_mask")

        logger.info("Step 1: Loading datasets with alignment checks...")
        try:
            # Load each dataset as separate xarray.Dataset, not pre-merged
            # This allows per-dataset configuration (resampling, winsorization) to be applied independently
            # Time-to-year conversion happens during loading, before year range filtering
            datasets = _load_datasets_for_processing(assembly_config, target_geobox, land_mask_ds)
            logger.info(f"Successfully loaded {len(datasets)} datasets")
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            return

        logger.info("Step 2: Creating assembly metadata...")

        if not create_assembly_metadata(output_path, assembly_config):
            logger.warning("Failed to create assembly metadata")

        logger.info("Step 3: Processing tiles (source-by-source)...")
        # Process each tile, extracting all datasets and concatenating results
        processed_count = 0
        skipped_count = 0

        for ix, iy in all_tiles:
            tile_output_path = os.path.join(output_path, f"ix={ix}", f"iy={iy}")
            output_file = os.path.join(tile_output_path, "data.parquet")
            if os.path.exists(output_file):
                try:
                    import pyarrow.parquet as pq
                    parquet_file = pq.ParquetFile(output_file)
                    if parquet_file.metadata.num_rows > 0:
                        skipped_count += 1
                        continue
                except Exception:
                    pass
            tile_geobox = GeoboxTiles(target_geobox, (processing_config.get('tile_size', 2048), processing_config.get('tile_size', 2048)))[ix, iy]
            try:
                success = process_tile(datasets, land_mask_ds, ix, iy, tile_geobox, assembly_config, output_path)
                if success:
                    processed_count += 1
            except Exception as e:
                logger.error(f"Failed to process tile ix={ix}, iy={iy}: {e}")
                continue

        logger.info(f"Dask processing completed. Processed {processed_count}/{len(all_tiles)} tiles, skipped {skipped_count} existing tiles")

def run_workflow_with_config(config: Dict[str, Any]):
    """
    Entry point for assembly workflow with unified configuration.
    
    Applies CLI overrides (Dask settings, tile size, compression) to assembly config
    before running the assembly process.
    """
    assembly_name = config.get('assembly_name', 'main')
    
    # Get assembly configuration
    if 'assemble' not in config or assembly_name not in config['assemble']:
        raise ValueError(f"Assembly configuration '{assembly_name}' not found in config")
    
    assembly_config = config['assemble'][assembly_name]
    
    # Apply CLI overrides to assembly configuration if present in full config
    cli_overrides = config.get('cli_overrides', {})
    if cli_overrides:
        processing_config = assembly_config.setdefault('processing', {})
        dask_config = processing_config.setdefault('dask', {})
        
        # Apply Dask-related CLI overrides
        if 'dask_threads' in cli_overrides:
            dask_config['threads'] = cli_overrides['dask_threads']
            logger.info(f"Overriding dask threads from CLI: {cli_overrides['dask_threads']}")
        if 'dask_memory_limit' in cli_overrides:
            dask_config['memory_limit'] = cli_overrides['dask_memory_limit']
            logger.info(f"Overriding dask memory limit from CLI: {cli_overrides['dask_memory_limit']}")
        if 'temp_dir' in cli_overrides:
            dask_config['temp_dir'] = cli_overrides['temp_dir']
            logger.info(f"Overriding temp dir from CLI: {cli_overrides['temp_dir']}")
        if 'dashboard_port' in cli_overrides:
            dask_config['dashboard_port'] = cli_overrides['dashboard_port']
            logger.info(f"Overriding dashboard port from CLI: {cli_overrides['dashboard_port']}")
        
        # Apply other processing overrides
        if 'tile_size' in cli_overrides:
            processing_config['tile_size'] = cli_overrides['tile_size']
            logger.info(f"Overriding tile size from CLI: {cli_overrides['tile_size']}")
        if 'compression' in cli_overrides:
            processing_config['compression'] = cli_overrides['compression']
            logger.info(f"Overriding compression from CLI: {cli_overrides['compression']}")
    
    # Run the assembly, passing both assembly config and full config
    run_assembly(assembly_config, config)
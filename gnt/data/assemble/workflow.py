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
                land_mask_ds = xr.open_zarr(path, consolidated=False, chunks='auto')["land_mask"]
                return land_mask_ds
        
        logger.warning(f"Land mask not found in any of: {potential_paths}")
        return None
        
    except Exception as e:
        logger.warning(f"Error loading land mask: {e}")
        return None
    
def _load_and_merge_datasets(
    assembly_config: Dict[str, Any],
    land_mask_ds: Optional[xr.Dataset] = None
) -> xr.Dataset:
    """
    Load and merge all zarr datasets into a single large xarray dataset.
    
    Args:
        assembly_config: Assembly configuration
        land_mask_ds: Optional land mask dataset
        
    Returns:
        Large merged xarray dataset with all variables
    """
    logger.info("Loading and merging all zarr datasets...")
    
    datasets_config = assembly_config['datasets']
    merged_datasets = []
    
    # Add land mask as the first dataset if provided
    if land_mask_ds is not None:
        logger.info("Adding land mask as first dataset")
        # Transform time to year if present in land mask
        if "time" in land_mask_ds.coords:
            land_mask_ds.coords["time"] = pd.Series(land_mask_ds.coords["time"]).dt.year.astype("int16")
        
        land_mask_ds.attrs['dataset_name'] = 'land_mask'
        merged_datasets.append(land_mask_ds)
    
    for dataset_name, dataset_config in datasets_config.items():
        zarr_path = dataset_config['path']
        
        if not os.path.exists(zarr_path):
            logger.warning(f"Dataset path does not exist: {zarr_path}, skipping")
            continue
        
        logger.info(f"Loading dataset {dataset_name} from {zarr_path}")
        
        try:
            # Open zarr dataset with automatic scaling and masking
            ds = xr.open_zarr(zarr_path, mask_and_scale=True, consolidated=False, chunks='auto')
            
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
            
            # Add dataset identifier for provenance
            ds.attrs['dataset_name'] = dataset_name
            merged_datasets.append(ds)
            
            logger.info(f"Loaded dataset {dataset_name}: {list(ds.data_vars.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            continue
    
    if not merged_datasets:
        raise ValueError("No datasets could be loaded successfully")
    
    # Merge all datasets using default join behavior
    logger.info("Merging datasets...")
    try:
        if len(merged_datasets) == 1:
            merged_ds = merged_datasets[0]
        else:
            # Start with the first dataset as base
            merged_ds = merged_datasets[0]
            logger.info(f"Base dataset: {merged_ds.attrs.get('dataset_name', 'unknown')}")
            
            # Merge remaining datasets using default xarray merge behavior
            for ds in merged_datasets[1:]:
                dataset_name = ds.attrs.get('dataset_name', 'unknown')
                
                logger.info(f"Merging {dataset_name}")
                merged_ds = xr.merge([merged_ds, ds], compat='override', join="left")
        
        logger.info(f"Successfully merged {len(merged_datasets)} datasets")
        logger.info(f"Final dataset variables: {list(merged_ds.data_vars.keys())}")
        logger.info(f"Dataset dimensions: {dict(merged_ds.sizes)}")
        
        return merged_ds
        
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

def _get_annual_means_path(output_path: str) -> str:
    """Get the path for the annual means YAML file."""
    return os.path.join(output_path, '_annual_means.yaml')

def _save_annual_means(output_path: str, annual_means: Dict[str, Dict[int, float]]):
    """Save annual means to YAML file."""
    annual_means_path = _get_annual_means_path(output_path)
    logger.info(f"Saving annual means to {annual_means_path}")
    
    with open(annual_means_path, 'w') as f:
        yaml.dump(annual_means, f, default_flow_style=False)
    
    logger.info("Successfully saved annual means")

def _load_annual_means(output_path: str) -> Optional[Dict[str, Dict[int, float]]]:
    """Load annual means from YAML file if it exists."""
    annual_means_path = _get_annual_means_path(output_path)
    
    if os.path.exists(annual_means_path):
        logger.info(f"Loading existing annual means from {annual_means_path}")
        try:
            with open(annual_means_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load annual means: {e}")
    
    return None


def _parse_demean_columns_config(demean_config) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Parse demean_columns configuration into column names and per-column demeaning types.
    
    Supports two formats:
    1. Legacy: ["col1", "col2"] - applies all types to all columns
    2. New: [["col1", ["type1", "type2"]], ["col2", ["type3"]]] - per-column types
    
    Args:
        demean_config: demean_columns configuration from assembly config
        
    Returns:
        Tuple of (column_names_list, column_to_types_dict)
    """
    if not demean_config:
        return [], {}
    
    # Check if using new format (list of lists)
    if demean_config and isinstance(demean_config[0], list):
        # New format: [["col1", ["type1", "type2"]], ...]
        columns = []
        column_types = {}
        
        for item in demean_config:
            if not isinstance(item, list) or len(item) != 2:
                logger.warning(f"Invalid demean_columns entry: {item}, expected [column_name, [types]]")
                continue
            
            col_name, types = item
            if not isinstance(types, list):
                logger.warning(f"Invalid types for column {col_name}: {types}, expected list")
                continue
            
            columns.append(col_name)
            column_types[col_name] = types
        
        return columns, column_types
    else:
        # Legacy format: ["col1", "col2", ...]
        # Returns empty dict to indicate "use global settings"
        return list(demean_config), {}

def _apply_demeaning_to_dataframe(
    df: pd.DataFrame,
    columns_to_demean: List[str],
    annual_means: Dict[str, Dict[int, float]],
    column_demeaning_types: Dict[str, List[str]] = None
) -> pd.DataFrame:
    """
    Apply demeaning operations to pandas DataFrame.
    
    Args:
        df: Input pandas DataFrame with 'year' column
        columns_to_demean: List of columns to demean
        annual_means: Pre-computed annual means
        column_demeaning_types: Per-column demeaning types dict {column: [types]}
        
    Returns:
        DataFrame with additional demeaned columns
    """
    df_demeaned = df.copy()
    
    # Determine which columns need which types
    column_types_map = {}
    
    if column_demeaning_types:
        # Use per-column specifications
        for col in columns_to_demean:
            if col in column_demeaning_types:
                column_types_map[col] = column_demeaning_types[col]
            else:
                logger.warning(f"No demeaning types specified for column {col}, skipping")
    else:
        logger.warning("No column_demeaning_types provided, skipping demeaning")
        return df_demeaned
    
    # Validate all types
    valid_types = {'time_demeaned', 'unit_demeaned', 'twoway_demeaned'}
    for col, types in column_types_map.items():
        invalid_types = set(types) - valid_types
        if invalid_types:
            logger.warning(f"Invalid demeaning types for column {col}: {invalid_types}. Valid types: {valid_types}")
            column_types_map[col] = [t for t in types if t in valid_types]
    
    # Check if we have required columns
    missing_cols = [col for col in columns_to_demean if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns for demeaning: {missing_cols}")
        columns_to_demean = [col for col in columns_to_demean if col in df.columns]
    
    if not columns_to_demean or 'year' not in df.columns:
        return df_demeaned
    
    # Determine which operations we need globally
    need_time_demeaned = any('time_demeaned' in types or 'twoway_demeaned' in types 
                             for types in column_types_map.values())
    need_unit_demeaned = any('unit_demeaned' in types for types in column_types_map.values())
    need_twoway_demeaned = any('twoway_demeaned' in types for types in column_types_map.values())
    
    # Compute time-demeaned values for all columns that need them
    time_demeaned_temps = {}
    if need_time_demeaned:
        for col in columns_to_demean:
            col_types = column_types_map.get(col, [])
            if 'time_demeaned' in col_types or 'twoway_demeaned' in col_types:
                if col in annual_means:
                    year_to_mean = annual_means[col]
                    time_demeaned_temps[col] = df_demeaned[col] - df_demeaned['year'].map(year_to_mean)
    
    # Store time-demeaned columns if requested
    for col in columns_to_demean:
        col_types = column_types_map.get(col, [])
        if 'time_demeaned' in col_types and col in time_demeaned_temps:
            df_demeaned[f"{col}_time_demeaned"] = time_demeaned_temps[col]
    
    # Unit and two-way demeaning require pixel grouping
    pixel_group_cols = ['latitude', 'longitude', 'pixel_id']
    available_group_cols = [col for col in pixel_group_cols if col in df.columns]
    
    if available_group_cols:
        # Unit demeaning: subtract pixel means
        if need_unit_demeaned:
            cols_for_unit = [col for col in columns_to_demean 
                           if 'unit_demeaned' in column_types_map.get(col, [])]
            if cols_for_unit:
                pixel_means = df_demeaned.groupby(available_group_cols)[cols_for_unit].transform('mean')
                for col in cols_for_unit:
                    df_demeaned[f"{col}_unit_demeaned"] = df_demeaned[col] - pixel_means[col]
        
        # Two-way demeaning: subtract both annual means and pixel means
        if need_twoway_demeaned:
            cols_for_twoway = [col for col in columns_to_demean 
                              if 'twoway_demeaned' in column_types_map.get(col, [])]
            
            if cols_for_twoway:
                # Collect time-demeaned columns (either stored or temporary)
                time_cols_to_use = {}
                for col in cols_for_twoway:
                    if f"{col}_time_demeaned" in df_demeaned.columns:
                        time_cols_to_use[col] = f"{col}_time_demeaned"
                    elif col in time_demeaned_temps:
                        # Use temporary value
                        temp_col = f"_temp_{col}_time_demeaned"
                        df_demeaned[temp_col] = time_demeaned_temps[col]
                        time_cols_to_use[col] = temp_col
                    else:
                        logger.warning(f"Missing time-demeaned values for {col}, skipping twoway demeaning")
                
                if time_cols_to_use:
                    time_col_names = list(time_cols_to_use.values())
                    pixel_means = df_demeaned.groupby(available_group_cols)[time_col_names].transform('mean')
                    
                    for col, time_col in time_cols_to_use.items():
                        df_demeaned[f"{col}_twoway_demeaned"] = df_demeaned[time_col] - pixel_means[time_col]
                
                # Clean up temporary columns
                temp_cols = [c for c in df_demeaned.columns if c.startswith('_temp_')]
                if temp_cols:
                    df_demeaned = df_demeaned.drop(columns=temp_cols)
    else:
        logger.warning("No pixel grouping columns found, skipping unit and twoway demeaning")
    
    return df_demeaned

def _aggregate_tile_stats(
    tile_stats: Dict[str, Dict[int, Dict[str, float]]],
    annual_sums: Dict[str, Dict[int, float]],
    annual_counts: Dict[str, Dict[int, int]]
):
    """
    Aggregate tile statistics into global accumulators.
    
    Args:
        tile_stats: Statistics from a single tile
        annual_sums: Global sum accumulator
        annual_counts: Global count accumulator
    """
    for col, col_stats in tile_stats.items():
        for year, year_stats in col_stats.items():
            annual_sums[col][year] += year_stats['sum'].compute()
            annual_counts[col][year] += year_stats['count'].compute()
            
def _extract_tile_stats(
    merged_ds: xr.Dataset,
    ix: int,
    iy: int,
    tile_geobox,
    columns_to_demean: List[str],
    land_mask_ds: Optional[xr.Dataset] = None
) -> Optional[Dict[str, Dict[int, Dict[str, float]]]]:
    """
    Extract a single tile from merged dataset and compute its contribution to annual means.
    """
    if not columns_to_demean:
        return None

    try:
        logger.debug(f"Extracting tile [{ix}, {iy}] stats from merged dataset (dask)")

        # Extract tile bounds
        bbox = tile_geobox.boundingbox

        # Slice dataset to tile bounds
        if 'latitude' in merged_ds.coords and 'longitude' in merged_ds.coords:
            tile_ds = merged_ds[columns_to_demean].sel(
                latitude=slice(bbox.top, bbox.bottom),
                longitude=slice(bbox.left, bbox.right)
            ).compute()
        else:
            logger.warning(f"Unknown coordinate system in merged dataset")
            return None

        # Check if tile has data
        if tile_ds.sizes.get('latitude', 0) == 0 or tile_ds.sizes.get('longitude', 0) == 0:
            logger.debug(f"No spatial data in tile [{ix}, {iy}]")
            return None

        # Check if we have time dimension for annual aggregation
        if 'time' not in tile_ds.sizes:
            logger.debug(f"No time dimension in tile [{ix}, {iy}]")
            return None
        
        annual_sum = tile_ds.resample(time='1YE').sum(dim=['latitude', 'longitude'], skipna=True)
        annual_count = tile_ds.resample(time='1YE').count(dim=['latitude', 'longitude'])

        return annual_sum, annual_count

    except Exception as e:
        logger.warning(f"Failed to extract tile [{ix}, {iy}] stats from merged dataset: {e}")
        return None

def _compute_annual_means(
    merged_ds: xr.Dataset,
    all_tiles: List[Tuple[int, int]],
    tiles,
    columns_to_demean: List[str],
    land_mask_ds: Optional[xr.Dataset] = None,
    client=None
) -> Dict[str, Dict[int, float]]:
    """
    Compute global annual means from merged dataset using tile-based processing.
    
    Args:
        merged_ds: Large merged xarray dataset
        all_tiles: List of (ix, iy) tile coordinates
        tiles: GeoboxTiles object for tile coordinate mapping
        columns_to_demean: List of columns to compute annual means for
        land_mask_ds: Optional land mask dataset
        client: Dask client for parallel processing
        
    Returns:
        Dictionary mapping column names to annual means (year -> mean)
    """
    
    logger.info("Computing annual means from merged dataset using sequential tile processing with Dask delayed...")
    
    if not columns_to_demean:
        logger.info("No columns specified for demeaning, skipping annual means computation")
        return {}
    
    logger.info(f"Processing {len(all_tiles)} tiles sequentially for annual means computation")
    
    # Sequentially process each tile and collect statistics
    tile_sum = xr.DataArray(0); tile_count = xr.DataArray(0)
    for ix, iy in all_tiles:
        tile_geobox = tiles[ix, iy]
        ctile_sum, ctile_count = _extract_tile_stats(
            merged_ds, ix, iy, tile_geobox, columns_to_demean, land_mask_ds
        )
        tile_sum = tile_sum + ctile_sum; tile_count = tile_count + ctile_count
    
    years = pd.Series(merged_ds.coords["time"].values).dt.year
    
    # Compute global annual means from aggregated stats
    global_annual_means = {}
    for col in columns_to_demean:
        col_means = {}
        for idx, year in enumerate(years):
            if tile_sum[col].isel(time=idx) > 0:
                col_means[year] = tile_sum[col].isel(time=idx).item() / tile_count[col].isel(time=idx).item()
        global_annual_means[col] = col_means
        logger.info(f"Computed annual means for {col}: {len(col_means)} years")
    
    return global_annual_means

def create_assembly_metadata(
    output_path: str,
    assembly_config: Dict[str, Any],
    annual_means: Optional[Dict[str, Dict[int, float]]] = None,
    column_demeaning_types: Dict[str, List[str]] = None
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
        
        # Add demeaning information if applicable
        processing_config = assembly_config.get('processing', {})
        demean_config = processing_config.get('demean_columns', [])
        
        if demean_config and annual_means:
            # Parse to get column names
            columns, parsed_col_types = _parse_demean_columns_config(demean_config)
            
            # Use parsed column types or provided column types
            effective_col_types = column_demeaning_types or parsed_col_types
            
            if effective_col_types:
                metadata_dict['demeaning'] = {
                    'demeaned_columns': columns,
                    'per_column_demeaning_types': effective_col_types,
                    'demeaning_types_description': {
                        'unit_demeaned': 'Subtracts pixel-level means within tiles (across years)',
                        'time_demeaned': 'Subtracts global annual means (across all pixels)',
                        'twoway_demeaned': 'Applies both time and unit demeaning sequentially'
                    },
                    'description': 'Per-column demeaning specifications define which transformations are stored for each variable.',
                    'annual_means_file': '_annual_means.yaml'
                }
        
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata_dict, f, default_flow_style=False)
        
        logger.info(f"Metadata written to {metadata_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to create metadata file: {e}")
        return False

def _extract_and_process_tile(
    merged_ds: xr.Dataset,
    ix: int,
    iy: int, 
    tile_geobox,
    assembly_config: Dict[str, Any],
    apply_demeaning: bool = False,
    columns_to_demean: List[str] = None,
    annual_means: Optional[Dict[str, Dict[int, float]]] = None,
    column_demeaning_types: Dict[str, List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    Extract and process a single tile from the large merged dataset.
    
    Args:
        merged_ds: Large merged xarray dataset
        ix, iy: Tile coordinates
        tile_geobox: Tile geobox for spatial bounds
        assembly_config: Assembly configuration
        apply_demeaning: Whether to apply demeaning operations
        columns_to_demean: List of columns to demean
        annual_means: Pre-computed annual means
        column_demeaning_types: Per-column demeaning types dict
        
    Returns:
        Processed DataFrame for the tile
    """
    try:
        logger.debug(f"Extracting tile [{ix}, {iy}] from merged dataset")
        
        # Extract tile bounds
        bbox = tile_geobox.boundingbox
        
        # Slice dataset to tile bounds
        if 'latitude' in merged_ds.coords and 'longitude' in merged_ds.coords:
            tile_ds = merged_ds.sel(
                latitude=slice(bbox.top, bbox.bottom),
                longitude=slice(bbox.left, bbox.right)
            ).compute()
        else:
            logger.warning(f"Unknown coordinate system in merged dataset")
            return None
        
        # Check if tile has data
        if tile_ds.sizes.get('latitude', 0) == 0 or tile_ds.sizes.get('longitude', 0) == 0:
            logger.debug(f"No spatial data in tile [{ix}, {iy}]")
            return None

        # Add pixel ID information for tile
        tile_size_actual = (tile_ds.sizes['latitude'], tile_ds.sizes['longitude'])
        pixel_id_matrix = np.arange(tile_size_actual[0] * tile_size_actual[1], dtype="int32").reshape(tile_size_actual)
        tile_ds = tile_ds.assign({
            'pixel_id': (['latitude', 'longitude'], pixel_id_matrix),
            'tile_ix': ix,
            'tile_iy': iy
        })
                
        # Convert to DataFrame
        logger.debug(f"Transforming to dataframe...")
        df = tile_ds.to_dataframe().reset_index()
        
        # Clean
        df = df.drop(columns=['band', 'latitude', 'longitude'])
        
        # Apply land mask filtering if land_mask variable is present
        if 'land_mask' in df.columns:
            df = df[df.land_mask]
            df = df.drop(columns=['land_mask'])
        
        if df.empty:
            logger.debug(f"No data remaining in tile [{ix}, {iy}]")
            return None
        
        # Apply demeaning operations on DataFrame after tabularization
        if apply_demeaning and columns_to_demean and annual_means:
            relevant_cols = [col for col in columns_to_demean if col in df.columns]
            if relevant_cols:
                logger.debug(f"Applying demeaning to {relevant_cols} in tile [{ix}, {iy}]")
                df = _apply_demeaning_to_dataframe(
                    df, relevant_cols, annual_means, column_demeaning_types
                )
        
        return df
        
    except Exception as e:
        logger.warning(f"Failed to extract tile [{ix}, {iy}] from merged dataset: {e}")
        return None

def process_tile(
    merged_ds: xr.Dataset,
    ix: int, 
    iy: int, 
    tile_geobox,
    assembly_config: Dict[str, Any], 
    output_base_path: str,
    annual_means: Optional[Dict[str, Dict[int, float]]] = None,
    column_demeaning_types: Dict[str, List[str]] = None
) -> bool:
    """Process a single tile from the merged dataset and write to parquet."""
    logger.debug(f"Processing tile ix={ix}, iy={iy} from merged dataset")
    
    # Check if tile already exists using fast file existence check
    tile_output_path = os.path.join(output_base_path, f"ix={ix}", f"iy={iy}")
    output_file = os.path.join(tile_output_path, "data.parquet")
    
    if os.path.exists(output_file):
        # Optionally verify the file is valid and not corrupted
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(output_file)
            # Quick check that file has some rows
            if parquet_file.metadata.num_rows > 0:
                logger.debug(f"Tile ix={ix}, iy={iy} already exists and is valid, skipping")
                return True
            else:
                logger.warning(f"Tile ix={ix}, iy={iy} exists but is empty, will reprocess")
        except Exception as e:
            logger.warning(f"Tile ix={ix}, iy={iy} exists but appears corrupted ({e}), will reprocess")
    
    processing_config = assembly_config.get('processing', {})
    demean_config = processing_config.get('demean_columns', [])
    columns, _ = _parse_demean_columns_config(demean_config)
    
    # Extract and process tile data from merged dataset
    apply_demeaning = bool(columns and annual_means)
    merged = _extract_and_process_tile(
        merged_ds, ix, iy, tile_geobox, assembly_config,
        apply_demeaning=apply_demeaning,
        columns_to_demean=columns,
        annual_means=annual_means,
        column_demeaning_types=column_demeaning_types
    )
    
    if merged is None or merged.empty:
        logger.debug(f"No data for tile ix={ix}, iy={iy}, skipping")
        return False
        
    # Write tile to output parquet partition
    os.makedirs(tile_output_path, exist_ok=True)
    
    compression = processing_config.get('compression', 'zstd')
    
    logger.debug(f"Writing tile ix={ix}, iy={iy} to {output_file}")
    merged.to_parquet(output_file, compression=compression, engine='pyarrow')
    
    return True

def run_assembly(assembly_config: Dict[str, Any], full_config: Dict[str, Any] = None):
    """Run the data assembly process based on configuration."""
    logger.info(f"Starting assembly: {assembly_config.get('description', 'Unknown')}")
    
    # Get output path - now expecting parquet directory output
    output_path = assembly_config['output_path']
    processing_config = assembly_config.get('processing', {})
    demean_config = processing_config.get('demean_columns', [])
    
    # Parse demean columns configuration
    demean_cols, column_demeaning_types = _parse_demean_columns_config(demean_config)
    
    # Log which demeaning will be applied
    if demean_cols:
        logger.info(f"Demeaning columns: {demean_cols}")
        if column_demeaning_types:
            logger.info("Using per-column demeaning specifications:")
            for col, types in column_demeaning_types.items():
                logger.info(f"  {col}: {types}")
        else:
            logger.info("No per-column demeaning types specified")
    
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
        
        # Load land mask if requested - will be merged as regular variable
        land_mask_ds = None
        if processing_config.get('apply_land_mask', False):
            land_mask_path = processing_config.get('land_mask_path')
            land_mask_ds = _load_land_mask(hpc_root, land_mask_path)
        
        # Load and merge all datasets into one large xarray dataset (including land mask)
        logger.info("Step 1: Loading and merging all zarr datasets...")
        try:
            merged_ds = _load_and_merge_datasets(assembly_config, land_mask_ds)
            logger.info("Successfully created large merged dataset")
        except Exception as e:
            logger.error(f"Failed to create merged dataset: {e}")
            return
        
        # Step 2: Compute annual means from merged dataset using tile-based approach if demeaning is requested
        annual_means = None
        if demean_cols:
            logger.info("Step 2: Computing annual means from merged dataset using tile-based approach...")
            
            # Try to load existing annual means first
            annual_means = _load_annual_means(output_path)
            
            # Check if we need to recompute
            force_recompute = processing_config.get('force_recompute_annual_means', False)
            if annual_means is None or force_recompute:
                logger.info("Computing annual means from merged dataset using tile processing...")
                annual_means = _compute_annual_means(
                    merged_ds, all_tiles, tiles, demean_cols, land_mask_ds, client
                )
                
                if annual_means:
                    _save_annual_means(output_path, annual_means)
                else:
                    logger.warning("Failed to compute annual means, demeaning will be skipped")
                    demean_cols = []
            else:
                logger.info("Using existing annual means")
                
                # Verify all required columns are present
                missing_cols = [col for col in demean_cols if col not in annual_means]
                if missing_cols:
                    logger.warning(f"Missing annual means for columns {missing_cols}, will recompute")
                    annual_means = _compute_annual_means(
                        merged_ds, all_tiles, tiles, demean_cols, land_mask_ds, client
                    )
                    if annual_means:
                        _save_annual_means(output_path, annual_means)
        
        # Step 2.5: Convert time coordinate to integer years
        logger.info("Step 2.5: Converting time coordinate to integer years...")
        if "time" in merged_ds.coords:
            merged_ds = merged_ds.rename(time='year')
            merged_ds.coords["year"] = pd.Series(merged_ds.coords["year"]).dt.year.astype("int16")
            logger.info(f"Time coordinate converted to year: {merged_ds.coords['year'].values[:5]}...")
        else:
            logger.info("No time coordinate found in merged dataset")
        
        # Create assembly metadata file
        if not create_assembly_metadata(output_path, assembly_config, annual_means, column_demeaning_types):
            logger.warning("Failed to create assembly metadata")
        
        # Step 3: Process tiles from merged dataset
        logger.info("Step 3: Processing tiles from merged dataset...")

        processed_count = 0
        skipped_count = 0

        for ix, iy in all_tiles:
            # Skip if tile already exists and is valid
            tile_output_path = os.path.join(output_path, f"ix={ix}", f"iy={iy}")
            output_file = os.path.join(tile_output_path, "data.parquet")

            if os.path.exists(output_file):
                try:
                    import pyarrow.parquet as pq
                    parquet_file = pq.ParquetFile(output_file)
                    if parquet_file.metadata.num_rows > 0:
                        logger.debug(f"Tile ix={ix}, iy={iy} already exists and is valid, skipping")
                        skipped_count += 1
                        continue
                except Exception:
                    pass  # Will reprocess if file is corrupted

            tile_geobox = tiles[ix, iy]
            try:
                success = process_tile(
                    merged_ds, ix, iy, tile_geobox,
                    assembly_config, output_path, annual_means, column_demeaning_types
                )
                if success:
                    processed_count += 1
                    logger.debug(f"Completed tile ix={ix}, iy={iy}")
                else:
                    logger.debug(f"No data for tile ix={ix}, iy={iy}")
            except Exception as e:
                logger.error(f"Failed to process tile ix={ix}, iy={iy}: {e}")
                continue
            
        logger.info(f"Dask processing completed. Processed {processed_count}/{len(all_tiles)} tiles, skipped {skipped_count} existing tiles")

def run_workflow_with_config(config: Dict[str, Any]):
    """Entry point for running assembly workflow with unified configuration."""
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
"""
Data demeaning module for the GNT system.

This module provides functionality to compute and apply demeaning operations
on assembled datasets using a tile-based approach. It processes individual tiles
that fit in memory and computes global statistics by aggregating across tiles.

The demeaning operations include:
- Unit demeaning: Removes pixel-level fixed effects within each tile
- Time demeaning: Removes global annual fixed effects across all tiles  
- Two-way demeaning: Removes both pixel and time fixed effects sequentially

The module handles packed data by unpacking before processing and repacking
demeaned columns with the same parameters as their source columns.

The module is designed for high-performance computing environments and automatically
detects SLURM resource allocations for optimal parallel processing.
"""

import os
import logging
import yaml
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
import threading
import time

# Import common utilities
from gnt.data.common.geobox import get_or_create_geobox
from gnt.data.common.metadata import unpack_dataframe, pack_dataframe, read_assembly_metadata
from odc.geo import GeoboxTiles

logger = logging.getLogger(__name__)

def _get_representative_dtypes(assembly_path: str) -> Dict[str, np.dtype]:
    """
    Get representative data types by sampling multiple tiles using efficient schema inspection.
    
    This function examines the first few available tiles to determine the data types
    used across the dataset. This information is used to preserve type consistency
    when creating demeaned columns.
    
    Args:
        assembly_path: Path to the assembled parquet directory
        
    Returns:
        Dictionary mapping column names to their representative data types
        
    Note:
        Falls back to empty dict if no tiles are available or readable
    """
    tiles = _get_available_tiles(assembly_path)
    
    for ix, iy in tiles[:3]:  # Check first 3 tiles
        tile_path = os.path.join(assembly_path, f"ix={ix}", f"iy={iy}", "data.parquet")
        
        if not os.path.exists(tile_path):
            continue
            
        try:
            import pyarrow.parquet as pq
            
            # First try to get dtypes from schema
            parquet_file = pq.ParquetFile(tile_path)
            schema = parquet_file.schema_arrow
            
            dtypes = {}
            for i, field in enumerate(schema):
                # Convert PyArrow types to numpy dtypes
                try:
                    pandas_dtype = field.type.to_pandas_dtype()
                    dtypes[field.name] = pandas_dtype
                except Exception:
                    # Fallback for complex types
                    dtypes[field.name] = 'object'
            
            if dtypes:
                logger.info(f"Extracted dtypes from schema of tile ix={ix}, iy={iy}")
                return dtypes
                
        except Exception as e:
            logger.debug(f"Failed to extract dtypes from schema for tile ix={ix}, iy={iy}: {e}")
            # Fallback to loading a small sample
            df = _load_tile_data(assembly_path, ix, iy)
            if df is not None:
                dtypes = {col: df[col].dtype for col in df.columns}
                logger.info(f"Extracted dtypes from sample data of tile ix={ix}, iy={iy}")
                return dtypes
    
    logger.warning("Could not extract dtypes from any tile")
    return {}

def _get_available_tiles(assembly_path: str) -> List[Tuple[int, int]]:
    """
    Discover all available tile partitions in the assembly directory.
    
    This function scans the parquet directory structure to identify all available
    tiles based on the ix=*/iy=* partition naming convention.
    
    Args:
        assembly_path: Path to the assembled parquet directory
        
    Returns:
        Sorted list of (ix, iy) coordinate tuples for available tiles
        
    Note:
        Returns empty list if assembly path does not exist or contains no valid tiles
    """
    tiles = []
    
    if not os.path.exists(assembly_path):
        logger.warning(f"Assembly path does not exist: {assembly_path}")
        return tiles
    
    # Look for ix=*/iy=* partition structure
    for ix_dir in os.listdir(assembly_path):
        if ix_dir.startswith('ix='):
            ix = int(ix_dir.split('=')[1])
            ix_path = os.path.join(assembly_path, ix_dir)
            
            if os.path.isdir(ix_path):
                for iy_dir in os.listdir(ix_path):
                    if iy_dir.startswith('iy='):
                        iy = int(iy_dir.split('=')[1])
                        iy_path = os.path.join(ix_path, iy_dir, 'data.parquet')
                        
                        if os.path.exists(iy_path):
                            tiles.append((ix, iy))
    
    logger.info(f"Found {len(tiles)} available tiles in {assembly_path}")
    return sorted(tiles)

def _load_tile_data(assembly_path: str, ix: int, iy: int, unpack: bool = False) -> Optional[pd.DataFrame]:
    """
    Load data for a specific tile from parquet storage.
    
    Args:
        assembly_path: Path to the assembled parquet directory
        ix: Tile x-coordinate index
        iy: Tile y-coordinate index
        unpack: Whether to unpack data using metadata
        
    Returns:
        DataFrame containing tile data, or None if tile cannot be loaded
        
    Raises:
        Warning: If tile loading fails due to I/O or parsing errors
    """
    tile_path = os.path.join(assembly_path, f"ix={ix}", f"iy={iy}", "data.parquet")
    
    if not os.path.exists(tile_path):
        logger.debug(f"Tile data not found: {tile_path}")
        return None
    
    try:
        df = pd.read_parquet(tile_path)
        logger.debug(f"Loaded tile ix={ix}, iy={iy} with {len(df)} records")
        
        # Unpack data if requested
        if unpack:
            metadata_path = os.path.join(assembly_path, '_metadata.yaml')
            if os.path.exists(metadata_path):
                df = unpack_dataframe(df, metadata_path, inplace=True)
                logger.debug(f"Unpacked data for tile ix={ix}, iy={iy}")
            else:
                logger.warning(f"Metadata file not found for unpacking: {metadata_path}")
        
        return df
    except Exception as e:
        logger.warning(f"Failed to load tile ix={ix}, iy={iy}: {e}")
        return None

def _save_tile_data(assembly_path: str, ix: int, iy: int, df: pd.DataFrame, compression: str = 'zstd'):
    """
    Save updated tile data back to parquet storage.
    
    Args:
        assembly_path: Path to the assembled parquet directory
        ix: Tile x-coordinate index
        iy: Tile y-coordinate index
        df: DataFrame containing updated tile data
        compression: Compression algorithm for parquet storage
        
    Raises:
        Exception: If saving fails due to I/O errors or storage issues
    """
    tile_path = os.path.join(assembly_path, f"ix={ix}", f"iy={iy}", "data.parquet")
    
    try:
        df.to_parquet(tile_path, compression=compression, engine='pyarrow')
        logger.debug(f"Saved updated tile ix={ix}, iy={iy} with {len(df)} records")
    except Exception as e:
        logger.error(f"Failed to save tile ix={ix}, iy={iy}: {e}")
        raise

def _get_annual_means_path(assembly_path: str) -> str:
    """Get the path for the annual means YAML file."""
    return os.path.join(assembly_path, '_annual_means.yaml')

def _compute_annual_means_single_tile(args: Tuple[str, int, int, List[str]]) -> Dict[str, Dict[int, Tuple[float, int]]]:
    """
    Compute annual sums and counts for a single tile using unpacked data.
    
    Args:
        args: Tuple of (assembly_path, ix, iy, columns_to_demean)
        
    Returns:
        Dictionary mapping columns to annual statistics (year -> (sum, count))
    """
    assembly_path, ix, iy, columns_to_demean = args
    
    # Load and unpack tile data
    df = _load_tile_data(assembly_path, ix, iy, unpack=True)
    if df is None:
        return {}
    
    tile_annual_stats = {col: {} for col in columns_to_demean}
    
    for col in columns_to_demean:
        if col in df.columns:
            # Compute temporal sums and counts for global annual mean calculation
            tile_annual = df.groupby('time')[col].agg(['sum', 'count'])
            
            for year in tile_annual.index:
                if not np.isnan(tile_annual.loc[year, 'sum']):
                    tile_annual_stats[col][year] = (
                        tile_annual.loc[year, 'sum'],
                        tile_annual.loc[year, 'count']
                    )
    
    return tile_annual_stats

def _remove_demeaned_columns_single_tile(args: Tuple[str, int, int, List[str]]) -> bool:
    """
    Remove demeaned columns from a single tile.
    
    Args:
        args: Tuple of (assembly_path, ix, iy, columns_to_demean)
        
    Returns:
        True if tile processing succeeded, False otherwise
    """
    assembly_path, ix, iy, columns_to_demean = args
    
    try:
        df = _load_tile_data(assembly_path, ix, iy, unpack=False)
        if df is None:
            return False
        
        # Find demeaned columns to remove
        demeaned_suffixes = ['_time_demeaned', '_unit_demeaned', '_twoway_demeaned']
        columns_to_remove = []
        
        for col in columns_to_demean:
            for suffix in demeaned_suffixes:
                demeaned_col = f"{col}{suffix}"
                if demeaned_col in df.columns:
                    columns_to_remove.append(demeaned_col)
        
        if columns_to_remove:
            logger.debug(f"Removing {len(columns_to_remove)} demeaned columns from tile ix={ix}, iy={iy}: {columns_to_remove}")
            df = df.drop(columns=columns_to_remove)
            _save_tile_data(assembly_path, ix, iy, df)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to remove demeaned columns from tile ix={ix}, iy={iy}: {e}")
        return False

def _remove_existing_demeaned_columns(assembly_path: str, columns_to_demean: List[str], n_workers: Optional[int] = None):
    """
    Remove existing demeaned columns from all tiles.
    
    Args:
        assembly_path: Path to assembled parquet directory
        columns_to_demean: List of columns that have demeaned variants to remove
        n_workers: Number of parallel workers
    """
    logger.info("Removing existing demeaned columns from all tiles...")
    tiles = _get_available_tiles(assembly_path)
    
    if not tiles:
        logger.warning("No tiles found for demeaned column removal")
        return
    
    if n_workers is None:
        n_workers = _get_optimal_worker_count(len(tiles), "io")
    
    logger.info(f"Using {n_workers} workers to remove demeaned columns from {len(tiles)} tiles")
    
    # Prepare arguments for parallel processing
    tile_args = [(assembly_path, ix, iy, columns_to_demean) for ix, iy in tiles]
    
    # Use ThreadPoolExecutor for I/O bound operations
    successful_tiles = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_remove_demeaned_columns_single_tile, args) for args in tile_args]
        
        with tqdm(total=len(tiles), desc="Removing demeaned columns", 
                 position=0, leave=True, dynamic_ncols=True,
                 smoothing=0.1, mininterval=0.5) as pbar:
            
            for future in as_completed(futures):
                try:
                    success = future.result()
                    if success:
                        successful_tiles += 1
                except Exception as e:
                    logger.warning(f"Failed to remove demeaned columns from a tile: {e}")
                finally:
                    pbar.update(1)
    
    logger.info(f"Successfully removed demeaned columns from {successful_tiles}/{len(tiles)} tiles")

def _remove_annual_means_file(assembly_path: str):
    """
    Remove the annual means YAML file.
    
    Args:
        assembly_path: Path to assembled parquet directory
    """
    annual_means_path = _get_annual_means_path(assembly_path)
    
    if os.path.exists(annual_means_path):
        try:
            os.remove(annual_means_path)
            logger.info(f"Removed existing annual means file: {annual_means_path}")
        except Exception as e:
            logger.error(f"Failed to remove annual means file {annual_means_path}: {e}")
    else:
        logger.debug(f"Annual means file does not exist: {annual_means_path}")

def _apply_override_level(assembly_path: str, columns_to_demean: List[str], override_level: int, n_workers: Optional[int] = None):
    """
    Apply the specified override level by removing existing data.
    
    Args:
        assembly_path: Path to assembled parquet directory
        columns_to_demean: List of columns to process
        override_level: Override level (0=none, 1=results, 2=intermediate+results)
        n_workers: Number of parallel workers
    """
    if override_level == 0:
        logger.info("Override level 0: No existing data will be removed")
        return
    elif override_level == 1:
        logger.info("Override level 1: Removing existing demeaned columns only")
        _remove_existing_demeaned_columns(assembly_path, columns_to_demean, n_workers)
    elif override_level == 2:
        logger.info("Override level 2: Removing existing demeaned columns and annual means")
        _remove_existing_demeaned_columns(assembly_path, columns_to_demean, n_workers)
        _remove_annual_means_file(assembly_path)
    else:
        raise ValueError(f"Invalid override level: {override_level}. Must be 0, 1, or 2.")

def compute_and_save_annual_means(
    assembly_path: str, 
    columns_to_demean: List[str],
    n_workers: Optional[int] = None,
    force_recompute: bool = False
) -> Dict[str, Dict[int, float]]:
    """
    Compute global annual means across all tiles and save to YAML file.
    
    Args:
        assembly_path: Path to the assembled parquet directory
        columns_to_demean: List of column names to compute annual means for
        n_workers: Number of parallel workers (auto-detected from SLURM if None)
        force_recompute: If True, recompute even if YAML file exists
        
    Returns:
        Dictionary mapping column names to annual means (year -> mean)
    """
    annual_means_path = _get_annual_means_path(assembly_path)
    
    # Check if annual means already exist
    if not force_recompute and os.path.exists(annual_means_path):
        logger.info(f"Loading existing annual means from {annual_means_path}")
        try:
            with open(annual_means_path, 'r') as f:
                annual_means = yaml.safe_load(f)
            
            # Check if all requested columns are present
            missing_columns = [col for col in columns_to_demean if col not in annual_means]
            if not missing_columns:
                logger.info(f"All {len(columns_to_demean)} columns found in existing annual means")
                return annual_means
            else:
                logger.info(f"Missing columns {missing_columns}, will recompute all annual means")
        except Exception as e:
            logger.warning(f"Failed to load existing annual means: {e}, will recompute")
    
    logger.info("Computing annual means from all tiles using unpacked data...")
    tiles = _get_available_tiles(assembly_path)
    
    if not tiles:
        logger.warning("No tiles found for annual means computation")
        return {}
    
    # Determine number of workers based on SLURM allocation
    if n_workers is None:
        n_workers = _get_optimal_worker_count(len(tiles), "cpu")
    
    logger.info(f"Using {n_workers} workers to process {len(tiles)} tiles")
    
    # Prepare arguments for parallel processing
    tile_args = [(assembly_path, ix, iy, columns_to_demean) for ix, iy in tiles]
    
    # Initialize result containers
    annual_sums = {col: {} for col in columns_to_demean}
    annual_counts = {col: {} for col in columns_to_demean}
    
    # Use ProcessPoolExecutor for CPU-bound statistics computation
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(_compute_annual_means_single_tile, args) for args in tile_args]
        
        # Use thread-safe progress tracking
        with tqdm(total=len(tiles), desc="Computing annual means", 
                 position=0, leave=True, dynamic_ncols=True,
                 smoothing=0.1, mininterval=0.5) as pbar:
            
            for future in as_completed(futures):
                try:
                    tile_annual_stats = future.result()
                    
                    # Aggregate annual statistics
                    for col in columns_to_demean:
                        if col in tile_annual_stats:
                            for year, (sum_val, count_val) in tile_annual_stats[col].items():
                                if year not in annual_sums[col]:
                                    annual_sums[col][year] = 0.0
                                    annual_counts[col][year] = 0
                                
                                annual_sums[col][year] += sum_val
                                annual_counts[col][year] += count_val
                                
                except Exception as e:
                    logger.warning(f"Failed to compute annual statistics for a tile: {e}")
                finally:
                    pbar.update(1)
    
    # Compute global annual means from accumulated sums and counts
    global_annual_means = {}
    for col in columns_to_demean:
        col_means = {}
        for year in annual_sums[col]:
            if annual_counts[col][year] > 0:
                col_means[year] = annual_sums[col][year] / annual_counts[col][year]
        global_annual_means[col] = col_means
        logger.info(f"Computed annual means for {col}: {len(col_means)} years")
    
    # Save annual means to YAML file
    logger.info(f"Saving annual means to {annual_means_path}")
    with open(annual_means_path, 'w') as f:
        yaml.dump(global_annual_means, f, default_flow_style=False)
    
    logger.info(f"Successfully computed and saved annual means for {len(columns_to_demean)} columns")
    return global_annual_means

def _copy_variable_metadata(source_metadata: Dict[str, Any], source_col: str, target_col: str) -> Dict[str, Any]:
    """
    Copy variable metadata from source column to target column for packing.
    
    Args:
        source_metadata: Original variable metadata dictionary
        source_col: Source column name
        target_col: Target column name
        
    Returns:
        Metadata dictionary for the target column
    """
    # Find the source column metadata
    variable_metadata = source_metadata.get('variable_metadata', {})
    for dataset_name, dataset_vars in variable_metadata.items():
        if source_col in dataset_vars:
            source_var_metadata = dataset_vars[source_col].copy()
            # Create metadata for target column with same packing parameters
            target_metadata = source_var_metadata.copy()
            target_metadata['source_column'] = source_col
            target_metadata['description'] = f"Demeaned version of {source_col}"
            logger.debug(f"Copied metadata from {source_col} to {target_col}: {target_metadata}")
            return target_metadata
    
    logger.warning(f"No metadata found for source column {source_col}")
    return {}

def _apply_demeaning_single_tile(args: Tuple[str, int, int, List[str], Dict[str, Dict[int, float]], str, Dict[str, np.dtype]]) -> bool:
    """
    Apply all three types of demeaning to a single tile with packing/unpacking.
    
    Args:
        args: Tuple containing:
            - assembly_path: Path to assembled parquet directory
            - ix, iy: Tile coordinates
            - columns_to_demean: List of columns to process
            - annual_means: Pre-computed global annual means (in unpacked form)
            - compression: Compression format for output
            - original_dtypes: Data types to preserve
        
    Returns:
        True if tile processing succeeded, False otherwise
    """
    assembly_path, ix, iy, columns_to_demean, annual_means, compression, original_dtypes = args
    
    try:
        # Load tile data without unpacking initially to preserve original dtypes
        df_packed = _load_tile_data(assembly_path, ix, iy, unpack=False)
        if df_packed is None:
            return False
        
        # Store original dtypes for all columns
        original_dtypes_all = {col: df_packed[col].dtype for col in df_packed.columns}
        
        # Unpack data for processing
        metadata_path = os.path.join(assembly_path, '_metadata.yaml')
        if not os.path.exists(metadata_path):
            logger.error(f"Metadata file not found: {metadata_path}")
            return False
        
        df = unpack_dataframe(df_packed, metadata_path, inplace=False)
        
        # Check if demeaned columns already exist
        has_demeaned_cols = any(f"{col}_time_demeaned" in df.columns for col in columns_to_demean)
        if has_demeaned_cols:
            logger.debug(f"Demeaned columns already exist in tile ix={ix}, iy={iy}, skipping")
            return True
        
        # Apply demeaning for each column using unpacked data
        for col in columns_to_demean:
            if col not in df.columns:
                logger.debug(f"Column {col} not found in tile ix={ix}, iy={iy}, skipping")
                continue
            
            # 1. Time demeaning: subtract annual means
            if col in annual_means:
                annual_means_col = annual_means[col]
                df[f"{col}_time_demeaned"] = df[col] - df['time'].map(annual_means_col)
            
            # 2. Unit demeaning: subtract pixel means within tile
            pixel_means = df.groupby('pixel_id')[col].transform('mean')
            df[f"{col}_unit_demeaned"] = df[col] - pixel_means
            
            # 3. Two-way demeaning: subtract both annual means and pixel means
            if col in annual_means:
                # First subtract annual means
                time_demeaned = df[col] - df['time'].map(annual_means_col)
                # Then subtract pixel means of the time-demeaned data
                pixel_means_residual = time_demeaned.groupby(df['pixel_id']).transform('mean')
                df[f"{col}_twoway_demeaned"] = time_demeaned - pixel_means_residual
        
        # Pack the demeaned columns back using original metadata
        df = pack_dataframe(df, metadata_path, inplace=True)
        
        # Preserve original dtypes for all columns, including new demeaned ones
        for col in df.columns:
            if col in original_dtypes_all:
                target_dtype = original_dtypes_all[col]
            else:
                # For new demeaned columns, use the dtype of their source column
                for source_col in columns_to_demean:
                    if col.startswith(f"{source_col}_") and col.endswith("_demeaned"):
                        target_dtype = original_dtypes_all.get(source_col, df[col].dtype)
                        break
                else:
                    target_dtype = df[col].dtype
            
            try:
                if df[col].dtype != target_dtype:
                    if (np.issubdtype(df[col].dtype, np.floating) and 
                        np.issubdtype(target_dtype, np.integer)):
                        df[col] = df[col].round().astype(target_dtype, casting='unsafe')
                    else:
                        df[col] = df[col].astype(target_dtype)
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not preserve dtype {target_dtype} for column {col}: {e}")
        
        # Save updated tile data
        _save_tile_data(assembly_path, ix, iy, df, compression)
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply demeaning to tile ix={ix}, iy={iy}: {e}")
        return False

def apply_demeaning_to_all_tiles(
    assembly_path: str, 
    columns_to_demean: List[str],
    annual_means: Dict[str, Dict[int, float]],
    compression: str = 'zstd',
    n_workers: Optional[int] = None
):
    """
    Apply demeaning operations to all tiles using the pre-computed annual means.
    
    Args:
        assembly_path: Path to assembled parquet directory
        columns_to_demean: List of column names to apply demeaning to
        annual_means: Pre-computed global annual means for each column
        compression: Compression format for output parquet files
        n_workers: Number of parallel workers (auto-detected from SLURM if None)
    """
    logger.info("Applying demeaning operations to all tiles with packing/unpacking...")
    tiles = _get_available_tiles(assembly_path)
    
    if not tiles:
        logger.warning("No tiles found for demeaning application")
        return
    
    # Get representative data types to preserve
    original_dtypes = _get_representative_dtypes(assembly_path)
    
    # Determine number of workers for I/O bound operations
    if n_workers is None:
        n_workers = _get_optimal_worker_count(len(tiles), "io")
    
    logger.info(f"Using {n_workers} workers to process {len(tiles)} tiles")
    
    # Prepare arguments for parallel processing
    tile_args = []
    for ix, iy in tiles:
        args = (assembly_path, ix, iy, columns_to_demean, annual_means, compression, original_dtypes)
        tile_args.append(args)
    
    # Use ThreadPoolExecutor for I/O bound operations
    successful_tiles = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(_apply_demeaning_single_tile, args) for args in tile_args]
        
        # Use thread-safe progress tracking
        with tqdm(total=len(tiles), desc="Applying demeaning", 
                 position=0, leave=True, dynamic_ncols=True,
                 smoothing=0.1, mininterval=0.5) as pbar:
            
            for future in as_completed(futures):
                try:
                    success = future.result()
                    if success:
                        successful_tiles += 1
                except Exception as e:
                    logger.warning(f"Failed to process a tile during demeaning: {e}")
                finally:
                    pbar.update(1)
    
    logger.info(f"Successfully applied demeaning to {successful_tiles}/{len(tiles)} tiles")

def _get_optimal_worker_count(total_tiles: int, operation_type: str = "cpu") -> int:
    """
    Calculate optimal worker count based on SLURM allocation and operation characteristics.
    
    This function determines the best number of parallel workers by considering:
    - Available CPU resources from SLURM
    - Operation type (CPU-bound vs I/O-bound)
    - Total number of tiles to process
    
    Args:
        total_tiles: Total number of tiles to be processed
        operation_type: Either "cpu" for CPU-bound or "io" for I/O-bound operations
        
    Returns:
        Optimal number of workers for the specified operation type
        
    Strategy:
        - CPU-bound operations: Use all available CPUs up to tile count
        - I/O-bound operations: Use half of available CPUs to avoid storage bottlenecks
    """
    # Try different SLURM environment variables
    slurm_vars = [
        'SLURM_CPUS_PER_TASK',  # CPUs per task
        'SLURM_NTASKS',         # Number of tasks
        'SLURM_CPUS_ON_NODE',   # CPUs on node
        'SLURM_NPROCS'          # Number of processes
    ]
    
    slurm_cpus = 4  # Default fallback
    for var in slurm_vars:
        value = os.environ.get(var)
        if value:
            try:
                slurm_cpus = int(value)
                logger.info(f"Using {slurm_cpus} CPUs from SLURM environment variable {var}")
                break
            except ValueError:
                logger.warning(f"Invalid value for {var}: {value}")
    else:
        logger.warning("No SLURM CPU environment variables found, using default of 4 CPUs")
    
    if operation_type == "io":
        optimal_workers = min(total_tiles, max(1, slurm_cpus // 2))
    else:
        optimal_workers = min(total_tiles, slurm_cpus)
    
    logger.info(f"Optimal worker count for {operation_type} operation: {optimal_workers} "
                f"(SLURM CPUs: {slurm_cpus}, tiles: {total_tiles})")
    return optimal_workers

def update_assembly_metadata(assembly_path: str, columns_to_demean: List[str]):
    """
    Update assembly metadata to document newly created demeaned columns.
    
    This function modifies the assembly's metadata file to include information about
    the demeaning operations that were performed, including column mappings and
    transformation descriptions. It also copies packing metadata from source columns
    to demeaned columns.
    
    Args:
        assembly_path: Path to assembled parquet directory
        columns_to_demean: List of columns that were demeaned
    """
    metadata_path = os.path.join(assembly_path, '_metadata.yaml')
    
    # Load existing metadata or create new
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            metadata = read_assembly_metadata(metadata_path)
        except Exception as e:
            logger.warning(f"Failed to load existing metadata: {e}")
    
    # Add demeaning information
    metadata['demeaning'] = {
        'demeaned_columns': columns_to_demean,
        'demeaning_types': ['unit_demeaned', 'time_demeaned', 'twoway_demeaned'],
        'description': 'Unit demeaning subtracts pixel-level means within tiles. Time demeaning subtracts global annual means. Two-way demeaning applies both sequentially.',
        'annual_means_file': '_annual_means.yaml'
    }
    
    # Update variable metadata for new columns
    if 'variable_metadata' not in metadata:
        metadata['variable_metadata'] = {}
    
    variable_metadata = metadata['variable_metadata']
    
    # Define demeaning descriptions
    demeaning_descriptions = {
        'time_demeaned': 'Time demeaned version (subtracts annual means)',
        'unit_demeaned': 'Unit demeaned version (subtracts pixel means within tile)',
        'twoway_demeaned': 'Two-way demeaned version (subtracts both annual and pixel means)'
    }
    
    # Copy packing metadata from source columns to demeaned columns
    for col in columns_to_demean:
        # Find the source column's metadata and dataset
        source_var_metadata = None
        source_dataset = None
        
        for dataset_name, dataset_vars in variable_metadata.items():
            if col in dataset_vars:
                source_var_metadata = dataset_vars[col].copy()
                source_dataset = dataset_name
                break
        
        if source_var_metadata is None:
            logger.warning(f"No metadata found for source column {col}, using defaults")
            source_var_metadata = {}
        
        # Create metadata for each type of demeaned column
        for suffix, desc_template in demeaning_descriptions.items():
            demeaned_col = f"{col}_{suffix}"
            
            # Copy packing metadata from source column
            demeaned_metadata = source_var_metadata.copy()
            demeaned_metadata.update({
                'description': f"{desc_template} of {col}",
                'source_column': col,
                'demeaning_type': suffix
            })
            
            # Add to the same dataset as the source column or fallback to 'demeaned'
            target_dataset = source_dataset or 'demeaned'
            if target_dataset not in variable_metadata:
                variable_metadata[target_dataset] = {}
            variable_metadata[target_dataset][demeaned_col] = demeaned_metadata
    
    # Save updated metadata
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    logger.info(f"Updated assembly metadata with demeaning information and packing metadata")

def run_demeaning_workflow(assembly_name: str, config: Dict[str, Any], override_level: int = 0):
    """
    Execute the complete demeaning workflow for a specified assembly.
    
    This is the main entry point for the demeaning process. It orchestrates the
    entire workflow from configuration validation through final metadata updates.
    The workflow consists of three main steps:
    
    1. Apply override level (remove existing data if requested)
    2. Compute and save annual means to YAML file (using unpacked data)
    3. Apply demeaning transformations to all tiles (unpack -> process -> repack)
    4. Update assembly metadata with new column information and packing metadata
    
    Args:
        assembly_name: Name of the assembly configuration to process
        config: Full configuration dictionary containing assembly definitions
        override_level: Override level (0=none, 1=results, 2=intermediate+results)
        
    Raises:
        ValueError: If assembly configuration is not found or invalid override level
        FileNotFoundError: If assembly directory does not exist
    """
    logger.info(f"Starting demeaning workflow for assembly: {assembly_name} (override_level={override_level})")
    
    # Validate override level
    if override_level not in [0, 1, 2]:
        raise ValueError(f"Invalid override level: {override_level}. Must be 0, 1, or 2.")
    
    # Get assembly configuration
    if 'assemble' not in config or assembly_name not in config['assemble']:
        raise ValueError(f"Assembly configuration '{assembly_name}' not found")
    
    assembly_config = config['assemble'][assembly_name]
    assembly_path = assembly_config['output_path']
    
    # Get processing configuration
    processing_config = assembly_config.get('processing', {})
    columns_to_demean = processing_config.get('demean_columns', [])
    compression = processing_config.get('compression', 'zstd')
    n_workers = processing_config.get('demeaning_workers')  # Allow config override
    force_recompute = processing_config.get('force_recompute_annual_means', False)
    
    if not columns_to_demean:
        logger.warning("No columns specified for demeaning")
        return
    
    logger.info(f"Will demean columns: {columns_to_demean}")
    logger.info(f"Assembly path: {assembly_path}")
    
    # Verify assembly exists
    if not os.path.exists(assembly_path):
        raise FileNotFoundError(f"Assembly not found at: {assembly_path}")
    
    # Verify metadata exists for packing/unpacking
    metadata_path = os.path.join(assembly_path, '_metadata.yaml')
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file required for packing/unpacking not found: {metadata_path}")
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Step 0: Apply override level
    if override_level > 0:
        logger.info(f"Step 0: Applying override level {override_level}...")
        _apply_override_level(assembly_path, columns_to_demean, override_level, n_workers)
        
        # For override level 2, force recomputation of annual means
        if override_level == 2:
            force_recompute = True
    
    # Step 1: Compute and save annual means using unpacked data
    logger.info("Step 1: Computing and saving annual means using unpacked data...")
    annual_means = compute_and_save_annual_means(
        assembly_path, 
        columns_to_demean, 
        n_workers,
        force_recompute
    )
    
    if not annual_means:
        logger.error("Failed to compute annual means, cannot proceed with demeaning")
        return
    
    # Step 2: Apply demeaning operations to all tiles with packing/unpacking
    logger.info("Step 2: Applying demeaning operations to all tiles with packing/unpacking...")
    apply_demeaning_to_all_tiles(
        assembly_path, 
        columns_to_demean,
        annual_means, 
        compression,
        n_workers
    )
    
    # Step 3: Update assembly metadata with packing information for demeaned columns
    logger.info("Step 3: Updating metadata with packing information...")
    update_assembly_metadata(assembly_path, columns_to_demean)
    
    logger.info(f"Demeaning workflow completed successfully for assembly: {assembly_name}")

def run_workflow_with_config(config: Dict[str, Any], assembly_name: str = None, override_level: int = 0):
    """
    Entry point for running demeaning workflow with unified configuration interface.
    
    This function provides a standardized interface for the demeaning workflow that
    integrates with the unified GNT configuration system. It handles default
    assembly name resolution and delegates to the main workflow function.
    
    Args:
        config: Full configuration dictionary
        assembly_name: Name of assembly to process (uses 'main' if None)
        override_level: Override level (0=none, 1=results, 2=intermediate+results)
        
    Note:
        This function is designed to be called from the unified run.py interface
        and follows the standard workflow configuration pattern used throughout
        the GNT system.
    """
    if assembly_name is None:
        assembly_name = config.get('assembly_name', 'main')
    
    run_demeaning_workflow(assembly_name, config, override_level)

# Module-level execution guard and usage information
if __name__ == "__main__":
    logger.error("This module should be run through the unified run.py interface")
    logger.error("Usage: python run.py demean --config config.yaml --source assembly_name")
    logger.error("Example: python run.py demean --config configs/data.yaml --source modis")
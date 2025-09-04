"""
Data demeaning module for the GNT system.

This module provides functionality to compute and apply demeaning operations
on assembled datasets using a tile-based approach. It processes individual tiles
that fit in memory and computes global statistics by aggregating across tiles.

The demeaning operations include:
- Unit demeaning: Removes pixel-level fixed effects within each tile
- Time demeaning: Removes global annual fixed effects across all tiles  
- Two-way demeaning: Removes both pixel and time fixed effects sequentially

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
from odc.geo import GeoboxTiles

logger = logging.getLogger(__name__)

def _preserve_dtypes(df: pd.DataFrame, original_dtypes: Dict[str, np.dtype]) -> pd.DataFrame:
    """
    Preserve original data types when creating new columns.
    
    This function ensures that newly created demeaned columns maintain the same
    data types as their source columns to optimize memory usage and storage efficiency.
    Uses rounding for float-to-integer conversions when lossless casting fails.
    
    Args:
        df: DataFrame with new columns to type-cast
        original_dtypes: Mapping of column names to their original data types
        
    Returns:
        DataFrame with preserved data types
        
    Raises:
        Warning: If data type conversion fails for any column
    """
    for col in df.columns:
        if col in original_dtypes:
            try:
                target_dtype = original_dtypes[col]
                current_dtype = df[col].dtype
                
                # If converting from float to integer type, round first
                if (np.issubdtype(current_dtype, np.floating) and 
                    np.issubdtype(target_dtype, np.integer)):
                    logger.debug(f"Rounding {col} from {current_dtype} to {target_dtype}")
                    df[col] = df[col].round().astype(target_dtype, casting='unsafe')
                else:
                    df[col] = df[col].astype(target_dtype)
                    
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not preserve dtype {original_dtypes[col]} for column {col}: {e}")
    return df

def _extract_dtypes_from_tile(assembly_path: str, ix: int, iy: int) -> Optional[Dict[str, np.dtype]]:
    """
    Extract data types from a representative tile.
    
    Args:
        assembly_path: Path to the assembled parquet directory
        ix: Tile x-coordinate index
        iy: Tile y-coordinate index
        
    Returns:
        Dictionary mapping column names to numpy data types, or None if tile cannot be loaded
    """
    df = _load_tile_data(assembly_path, ix, iy)
    if df is not None:
        return {col: df[col].dtype for col in df.columns}
    return None

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
            dtypes = _extract_dtypes_from_tile(assembly_path, ix, iy)
            if dtypes:
                logger.info(f"Extracted dtypes from sample data of tile ix={ix}, iy={iy}")
                return dtypes
    
    logger.warning("Could not extract dtypes from any tile")
    return {}

def _load_assembly_metadata(assembly_path: str) -> Dict[str, Any]:
    """
    Load assembly metadata from the parquet directory.
    
    Args:
        assembly_path: Path to the assembled parquet directory
        
    Returns:
        Dictionary containing assembly metadata, or empty dict if not found
    """
    metadata_path = os.path.join(assembly_path, '_metadata.yaml')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return yaml.safe_load(f)
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

def _load_tile_data(assembly_path: str, ix: int, iy: int) -> Optional[pd.DataFrame]:
    """
    Load data for a specific tile from parquet storage.
    
    Args:
        assembly_path: Path to the assembled parquet directory
        ix: Tile x-coordinate index
        iy: Tile y-coordinate index
        
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

def _compute_pixel_means_single_tile(args: Tuple[str, int, int, List[str]]) -> Tuple[Tuple[int, int], Dict[str, pd.Series]]:
    """
    Compute pixel means for a single tile - designed for parallel processing.
    
    Args:
        args: Tuple of (assembly_path, ix, iy, columns_to_demean)
        
    Returns:
        Tuple of ((ix, iy), {column: pixel_means_series})
    """
    assembly_path, ix, iy, columns_to_demean = args
    
    df = _load_tile_data(assembly_path, ix, iy)
    if df is None:
        return ((ix, iy), {})
    
    tile_pixel_means = {}
    for col in columns_to_demean:
        if col in df.columns:
            # Group by pixel_id and compute mean, handling missing values
            pixel_mean = df.groupby('pixel_id')[col].mean()
            tile_pixel_means[col] = pixel_mean
    
    return ((ix, iy), tile_pixel_means)

def _compute_all_statistics_single_tile(args: Tuple[str, int, int, List[str]]) -> Tuple[Tuple[int, int], Dict[str, pd.Series], Dict[str, Dict[int, Tuple[float, int]]]]:
    """
    Compute comprehensive statistics for a single tile in parallel processing context.
    
    This function performs both pixel-level and temporal aggregations in a single pass
    through the data to maximize I/O efficiency. It computes:
    - Pixel means grouped by pixel_id (for unit demeaning)
    - Annual sums and counts grouped by time (for time demeaning)
    
    Args:
        args: Tuple containing (assembly_path, ix, iy, columns_to_demean)
        
    Returns:
        Tuple containing:
        - Tile coordinates (ix, iy)
        - Dictionary mapping columns to pixel mean Series
        - Dictionary mapping columns to annual statistics (year -> (sum, count))
        
    Note:
        Designed for use with ProcessPoolExecutor to enable parallel processing
    """
    assembly_path, ix, iy, columns_to_demean = args
    
    df = _load_tile_data(assembly_path, ix, iy)
    if df is None:
        return ((ix, iy), {}, {})
    
    tile_pixel_means = {}
    tile_annual_stats = {col: {} for col in columns_to_demean}
    
    for col in columns_to_demean:
        if col in df.columns:
            # Compute pixel-level means for unit demeaning
            # Groups by pixel_id and computes mean, handling NaN values appropriately
            pixel_mean = df.groupby('pixel_id')[col].mean()
            tile_pixel_means[col] = pixel_mean
            
            # Compute temporal sums and counts for global annual mean calculation
            # Groups by time/year and aggregates for later global mean computation
            tile_annual = df.groupby('time')[col].agg(['sum', 'count'])
            
            for year in tile_annual.index:
                if not np.isnan(tile_annual.loc[year, 'sum']):
                    tile_annual_stats[col][year] = (
                        tile_annual.loc[year, 'sum'],
                        tile_annual.loc[year, 'count']
                    )
    
    return ((ix, iy), tile_pixel_means, tile_annual_stats)

def compute_all_statistics_from_tiles(
    assembly_path: str, 
    columns_to_demean: List[str],
    n_workers: Optional[int] = None
) -> Tuple[Dict[Tuple[int, int], Dict[str, pd.Series]], Dict[str, Dict[int, float]]]:
    """
    Compute comprehensive demeaning statistics by processing all tiles in parallel.
    
    This is the main statistics computation function that orchestrates parallel processing
    of all tiles to compute both pixel-level and global temporal statistics in a single
    pass through the data. The function maximizes I/O efficiency by reading each tile
    only once while computing all required statistics.
    
    Args:
        assembly_path: Path to the assembled parquet directory
        columns_to_demean: List of column names to compute demeaning statistics for
        n_workers: Number of parallel workers (auto-detected from SLURM if None)
        
    Returns:
        Tuple containing:
        - pixel_means_dict: Maps (ix, iy) -> {column: pixel_means_series}
        - global_annual_means_dict: Maps column -> {year: global_mean}
        
    Note:
        Uses ProcessPoolExecutor for CPU-bound statistics computation with
        automatic SLURM resource detection for optimal performance
    """
    logger.info("Computing all statistics from tiles using parallel processing...")
    tiles = _get_available_tiles(assembly_path)
    
    if not tiles:
        logger.warning("No tiles found for statistics computation")
        return {}, {}
    
    # Determine number of workers based on SLURM allocation
    if n_workers is None:
        n_workers = _get_optimal_worker_count(len(tiles), "cpu")
    
    logger.info(f"Using {n_workers} workers to process {len(tiles)} tiles")
    
    # Prepare arguments for parallel processing
    tile_args = [(assembly_path, ix, iy, columns_to_demean) for ix, iy in tiles]
    
    # Initialize result containers
    pixel_means = {}
    annual_sums = {col: {} for col in columns_to_demean}
    annual_counts = {col: {} for col in columns_to_demean}
    
    # Use ProcessPoolExecutor for CPU-bound statistics computation
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(_compute_all_statistics_single_tile, args) for args in tile_args]
        
        # Use thread-safe progress tracking
        with tqdm(total=len(tiles), desc="Computing statistics", 
                 position=0, leave=True, dynamic_ncols=True,
                 smoothing=0.1, mininterval=0.5) as pbar:
            
            for future in as_completed(futures):
                try:
                    (ix, iy), tile_pixel_means, tile_annual_stats = future.result()
                    
                    # Store pixel means if available
                    if tile_pixel_means:
                        pixel_means[(ix, iy)] = tile_pixel_means
                    
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
                    logger.warning(f"Failed to compute statistics for a tile: {e}")
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
        logger.info(f"Computed statistics for {col}: {len(pixel_means)} tiles, {len(col_means)} years")
    
    logger.info(f"Computed statistics for {len(pixel_means)} tiles total")
    return pixel_means, global_annual_means

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
    
    optimal_workers = min(total_tiles, slurm_cpus)
    
    logger.info(f"Optimal worker count for {operation_type} operation: {optimal_workers} "
                f"(SLURM CPUs: {slurm_cpus}, tiles: {total_tiles})")
    return optimal_workers

def _check_tile_demeaning_status(assembly_path: str, ix: int, iy: int, columns_to_demean: List[str]) -> Tuple[bool, List[str]]:
    """
    Check demeaning status for a specific tile using fast schema inspection.
    
    Args:
        assembly_path: Path to the assembled parquet directory
        ix: Tile x-coordinate index
        iy: Tile y-coordinate index
        columns_to_demean: List of columns to check for demeaning
        
    Returns:
        Tuple of (all_exist, missing_columns) where:
        - all_exist: True if all demeaned columns exist in this tile
        - missing_columns: List of original columns that don't have all demeaned variants
    """
    tile_path = os.path.join(assembly_path, f"ix={ix}", f"iy={iy}", "data.parquet")
    
    if not os.path.exists(tile_path):
        return False, columns_to_demean
    
    try:
        import pyarrow.parquet as pq
        
        # Read only the schema/metadata, not the data
        parquet_file = pq.ParquetFile(tile_path)
        schema = parquet_file.schema_arrow
        column_names = set(schema.names)
        
        missing_columns = []
        for col in columns_to_demean:
            if col not in column_names:
                missing_columns.append(col)
                continue
                
            # Check if all three types of demeaned columns exist
            demeaned_cols = [f"{col}_unit_demeaned", f"{col}_time_demeaned", f"{col}_twoway_demeaned"]
            if not all(dcol in column_names for dcol in demeaned_cols):
                missing_columns.append(col)
        
        all_exist = len(missing_columns) == 0
        return all_exist, missing_columns
        
    except Exception as e:
        logger.warning(f"Failed to check schema for tile ix={ix}, iy={iy}: {e}")
        return False, columns_to_demean

def _check_if_demeaning_exists(assembly_path: str, columns_to_demean: List[str]) -> Tuple[bool, List[str], Dict[Tuple[int, int], List[str]]]:
    """
    Check if demeaning has already been applied by examining all tiles.
    
    Args:
        assembly_path: Path to the assembled parquet directory
        columns_to_demean: List of columns to check for demeaning
        
    Returns:
        Tuple of (all_tiles_complete, globally_missing_columns, tile_status) where:
        - all_tiles_complete: True if all tiles have all demeaned columns
        - globally_missing_columns: List of columns missing from most tiles
        - tile_status: Dict mapping (ix, iy) to list of missing columns for that tile
    """
    tiles = _get_available_tiles(assembly_path)
    if not tiles:
        return False, columns_to_demean, {}
    
    logger.info(f"Checking demeaning status for {len(tiles)} tiles...")
    
    tile_status = {}
    column_missing_counts = {col: 0 for col in columns_to_demean}
    
    # Check each tile using fast schema inspection
    for ix, iy in tiles:
        all_exist, missing_columns = _check_tile_demeaning_status(assembly_path, ix, iy, columns_to_demean)
        tile_status[(ix, iy)] = missing_columns
        
        # Count how many tiles are missing each column
        for col in missing_columns:
            if col in column_missing_counts:
                column_missing_counts[col] += 1
    
    # Determine globally missing columns (missing from majority of tiles)
    total_tiles = len(tiles)
    globally_missing_columns = []
    for col, missing_count in column_missing_counts.items():
        if missing_count > total_tiles * 0.5:  # Missing from more than 50% of tiles
            globally_missing_columns.append(col)
    
    # Check if all tiles are complete
    all_tiles_complete = all(len(missing) == 0 for missing in tile_status.values())
    
    logger.info(f"Demeaning status check complete:")
    logger.info(f"  - All tiles complete: {all_tiles_complete}")
    logger.info(f"  - Globally missing columns: {globally_missing_columns}")
    logger.info(f"  - Tiles needing processing: {sum(1 for missing in tile_status.values() if missing)}")
    
    return all_tiles_complete, globally_missing_columns, tile_status

def _apply_demeaning_single_tile(args: Tuple[str, int, int, List[str], Dict[str, pd.Series], Dict[str, Dict[int, float]], str, Dict[str, np.dtype], Dict[Tuple[int, int], List[str]]]) -> bool:
    """
    Apply demeaning transformations to a single tile in parallel processing context.
    
    This function performs all three types of demeaning operations on a single tile:
    1. Unit demeaning: Subtracts pixel-level means within the tile
    2. Time demeaning: Subtracts global annual means 
    3. Two-way demeaning: Applies both transformations sequentially
    
    Args:
        args: Tuple containing all necessary parameters for parallel processing:
            - assembly_path: Path to assembled parquet directory
            - ix, iy: Tile coordinates
            - columns_to_demean: List of columns to process
            - tile_pixel_means: Pre-computed pixel means for this tile
            - global_annual_means: Pre-computed global annual means
            - compression: Compression format for output
            - original_dtypes: Data types to preserve
            - tile_status: Dict mapping (ix, iy) to missing columns for that tile
        
    Returns:
        True if tile processing succeeded, False otherwise
        
    Note:
        Uses _preserve_dtypes for all dtype handling to avoid redundancy.
        Designed for use with ThreadPoolExecutor for I/O-bound operations.
    """
    (assembly_path, ix, iy, columns_to_demean, tile_pixel_means, 
     global_annual_means, compression, original_dtypes, tile_status) = args
    
    try:
        # Check if this specific tile needs processing
        tile_missing_columns = tile_status.get((ix, iy), [])
        if not tile_missing_columns:
            logger.debug(f"Tile ix={ix}, iy={iy} already has all demeaned columns, skipping")
            return True
        
        df = _load_tile_data(assembly_path, ix, iy)
        if df is None:
            return False
        
        # Apply demeaning only for columns that are missing in this specific tile
        for col in tile_missing_columns:
            if col not in df.columns:
                logger.debug(f"Column {col} not found in tile ix={ix}, iy={iy}, skipping")
                continue
            
            # Unit-average demeaning (subtract pixel means within tile)
            if col in tile_pixel_means and f"{col}_unit_demeaned" not in df.columns:
                pixel_mean_series = tile_pixel_means[col]
                df[f"{col}_unit_demeaned"] = df[col] - df['pixel_id'].map(pixel_mean_series)
            
            # Annual-average demeaning (subtract global annual means)
            if col in global_annual_means and f"{col}_time_demeaned" not in df.columns:
                annual_means = global_annual_means[col]
                df[f"{col}_time_demeaned"] = df[col] - df['time'].map(annual_means)
            
            # Two-way demeaning (subtract both pixel and annual means)
            if (col in tile_pixel_means and col in global_annual_means and 
                f"{col}_twoway_demeaned" not in df.columns):
                # First subtract annual means, then subtract pixel means of residuals
                annual_means = global_annual_means[col]
                time_demeaned = df[col] - df['time'].map(annual_means)
                
                # Compute pixel means of annual-demeaned data within this tile
                pixel_means_residual = time_demeaned.groupby(df['pixel_id']).transform('mean')
                df[f"{col}_twoway_demeaned"] = time_demeaned - pixel_means_residual
        
        # Apply unified dtype preservation
        df = _preserve_dtypes(df, original_dtypes)
        
        # Save updated tile data
        _save_tile_data(assembly_path, ix, iy, df, compression)
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply demeaning to tile ix={ix}, iy={iy}: {e}")
        return False

def apply_demeaning_to_tiles(
    assembly_path: str, 
    columns_to_demean: List[str],
    pixel_means: Dict[Tuple[int, int], Dict[str, pd.Series]],
    global_annual_means: Dict[str, Dict[int, float]],
    compression: str = 'zstd',
    n_workers: Optional[int] = None
):
    """
    Apply demeaning operations to all tiles using parallel processing.
    
    This function orchestrates the application of demeaning transformations across
    all tiles in the assembly. It uses ThreadPoolExecutor to parallelize I/O-bound
    operations (reading/writing parquet files) while preserving data types and
    providing progress tracking.
    
    Three types of demeaning are applied:
    1. Unit demeaning: {column}_unit_demeaned - removes pixel fixed effects
    2. Time demeaning: {column}_time_demeaned - removes temporal fixed effects  
    3. Two-way demeaning: {column}_twoway_demeaned - removes both effects
    
    Args:
        assembly_path: Path to assembled parquet directory
        columns_to_demean: List of column names to apply demeaning to
        pixel_means: Pre-computed pixel means for each tile and column
        global_annual_means: Pre-computed global annual means for each column
        compression: Compression format for output parquet files
        n_workers: Number of parallel workers (auto-detected from SLURM if None)
        
    Note:
        Uses ThreadPoolExecutor instead of ProcessPoolExecutor since this operation
        is I/O-bound rather than CPU-bound. Worker count is automatically adjusted
        for I/O operations to avoid overwhelming storage systems.
    """
    logger.info("Applying demeaning operations to all tiles using parallel processing...")
    tiles = _get_available_tiles(assembly_path)
    
    if not tiles:
        logger.warning("No tiles found for demeaning application")
        return
    
    # Check demeaning status for each tile
    all_tiles_complete, globally_missing_columns, tile_status = _check_if_demeaning_exists(assembly_path, columns_to_demean)
    
    if all_tiles_complete:
        logger.info("All tiles already have all demeaned columns, skipping demeaning application")
        return
    
    # Count tiles that need processing
    tiles_needing_processing = [(ix, iy) for (ix, iy), missing in tile_status.items() if missing]
    logger.info(f"Will process {len(tiles_needing_processing)} tiles that need demeaning")
    
    if not tiles_needing_processing:
        logger.info("No tiles need demeaning processing")
        return
    
    # Get representative data types to preserve
    original_dtypes = _get_representative_dtypes(assembly_path)
    
    # Determine number of workers (use fewer workers for I/O bound operations)
    if n_workers is None:
        n_workers = _get_optimal_worker_count(len(tiles_needing_processing), "io")
    
    logger.info(f"Using {n_workers} workers to process {len(tiles_needing_processing)} tiles")
    
    # Prepare arguments for parallel processing - only for tiles that need processing
    tile_args = []
    for ix, iy in tiles_needing_processing:
        tile_pixel_means = pixel_means.get((ix, iy), {})
        args = (assembly_path, ix, iy, columns_to_demean, tile_pixel_means, 
                global_annual_means, compression, original_dtypes, tile_status)
        tile_args.append(args)
    
    # Use ThreadPoolExecutor for I/O bound operations (reading/writing parquet files)
    successful_tiles = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(_apply_demeaning_single_tile, args) for args in tile_args]
        
        # Use thread-safe progress tracking with better performance
        with tqdm(total=len(tiles_needing_processing), desc="Applying demeaning", 
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
    
    logger.info(f"Successfully applied demeaning to {successful_tiles}/{len(tiles_needing_processing)} tiles")

def update_assembly_metadata(assembly_path: str, columns_to_demean: List[str]):
    """
    Update assembly metadata to document newly created demeaned columns.
    
    This function modifies the assembly's metadata file to include information about
    the demeaning operations that were performed, including column mappings and
    transformation descriptions.
    
    Args:
        assembly_path: Path to assembled parquet directory
        columns_to_demean: List of columns that were demeaned
        
    Metadata Added:
        - demeaning.demeaned_columns: Original columns that were processed
        - demeaning.demeaning_types: Types of demeaning applied
        - demeaning.description: Explanation of demeaning operations
        - variable_metadata: Detailed information for each new column
        
    Note:
        Creates metadata file if it doesn't exist, otherwise updates existing metadata
    """
    # Use existing metadata loading function
    metadata = _load_assembly_metadata(assembly_path)
    
    # Add demeaning information
    metadata['demeaning'] = {
        'demeaned_columns': columns_to_demean,
        'demeaning_types': ['unit_demeaned', 'time_demeaned', 'twoway_demeaned'],
        'description': 'Unit demeaning subtracts pixel-level means within tiles. Annual demeaning subtracts global annual means. Two-way demeaning applies both sequentially.'
    }
    
    # Update variable metadata for new columns
    if 'variable_metadata' not in metadata:
        metadata['variable_metadata'] = {}
    
    for col in columns_to_demean:
        for suffix in ['unit_demeaned', 'time_demeaned', 'twoway_demeaned']:
            new_col = f"{col}_{suffix}"
            metadata['variable_metadata'][new_col] = {
                'description': f"{suffix.replace('_', ' ').title()} version of {col}",
                'source_column': col,
                'demeaning_type': suffix
            }
    
    # Save updated metadata
    metadata_path = os.path.join(assembly_path, '_metadata.yaml')
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    logger.info(f"Updated assembly metadata with demeaning information")

def run_demeaning_workflow(assembly_name: str, config: Dict[str, Any]):
    """
    Execute the complete demeaning workflow for a specified assembly.
    
    This is the main entry point for the demeaning process. It orchestrates the
    entire workflow from configuration validation through final metadata updates.
    The workflow consists of four main steps:
    
    1. Compute pixel and temporal statistics in parallel
    2. Apply demeaning transformations to all tiles
    3. Update assembly metadata with new column information
    
    Args:
        assembly_name: Name of the assembly configuration to process
        config: Full configuration dictionary containing assembly definitions
        
    Raises:
        ValueError: If assembly configuration is not found
        FileNotFoundError: If assembly directory does not exist
        
    Workflow Steps:
        1-2. Combined statistics computation (pixel means + annual means)
        3. Apply demeaning transformations to all tiles in parallel
        4. Update metadata to document new columns
        
    Note:
        Automatically detects SLURM resource allocation for optimal parallel processing.
        Supports manual worker count override through configuration.
    """
    logger.info(f"Starting demeaning workflow for assembly: {assembly_name}")
    
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
    
    if not columns_to_demean:
        logger.warning("No columns specified for demeaning")
        return
    
    logger.info(f"Will demean columns: {columns_to_demean}")
    logger.info(f"Assembly path: {assembly_path}")
    
    # Verify assembly exists
    if not os.path.exists(assembly_path):
        raise FileNotFoundError(f"Assembly not found at: {assembly_path}")
    
    # Check demeaning status across all tiles
    all_tiles_complete, globally_missing_columns, tile_status = _check_if_demeaning_exists(assembly_path, columns_to_demean)
    
    if all_tiles_complete:
        logger.info("All tiles already have all demeaned columns. Skipping demeaning workflow.")
        return
    
    # Use globally missing columns for statistics computation
    if globally_missing_columns:
        logger.info(f"Will compute statistics for globally missing columns: {globally_missing_columns}")
        stats_columns = globally_missing_columns
    else:
        logger.info("Some tiles have partial demeaning. Computing statistics for all requested columns.")
        stats_columns = columns_to_demean
    
    # Step 1 & 2 Combined: Compute all statistics in one pass
    logger.info("Step 1-2 Combined: Computing all statistics from tiles...")
    pixel_means, global_annual_means = compute_all_statistics_from_tiles(
        assembly_path, stats_columns, n_workers
    )
    
    # Step 3: Apply demeaning operations to all tiles
    logger.info("Step 3: Applying demeaning operations...")
    apply_demeaning_to_tiles(
        assembly_path, 
        columns_to_demean,  # Use original column list for application
        pixel_means, 
        global_annual_means, 
        compression,
        n_workers
    )
    
    # Step 4: Update assembly metadata
    logger.info("Step 4: Updating metadata...")
    # Use original columns list for metadata (not just missing ones)
    original_columns = processing_config.get('demean_columns', [])
    update_assembly_metadata(assembly_path, original_columns)
    
    logger.info(f"Demeaning workflow completed successfully for assembly: {assembly_name}")

def run_workflow_with_config(config: Dict[str, Any], assembly_name: str = None):
    """
    Entry point for running demeaning workflow with unified configuration interface.
    
    This function provides a standardized interface for the demeaning workflow that
    integrates with the unified GNT configuration system. It handles default
    assembly name resolution and delegates to the main workflow function.
    
    Args:
        config: Full configuration dictionary
        assembly_name: Name of assembly to process (uses 'main' if None)
        
    Note:
        This function is designed to be called from the unified run.py interface
        and follows the standard workflow configuration pattern used throughout
        the GNT system.
    """
    if assembly_name is None:
        assembly_name = config.get('assembly_name', 'main')
    
    run_demeaning_workflow(assembly_name, config)

# Module-level execution guard and usage information
if __name__ == "__main__":
    logger.error("This module should be run through the unified run.py interface")
    logger.error("Usage: python run.py demean --config config.yaml --source assembly_name")
    logger.error("Example: python run.py demean --config configs/data.yaml --source modis")
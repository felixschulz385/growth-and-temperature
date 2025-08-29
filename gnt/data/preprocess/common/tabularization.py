import os
import logging
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from odc.geo import GeoboxTiles
from gnt.data.common.geobox.geobox import get_or_create_geobox

logger = logging.getLogger(__name__)

def process_zarr_to_parquet(
    ds: xr.Dataset,
    output_path: str,
    hpc_root: str = None,
    tile_size: int = 2048,
    apply_land_mask: bool = True,
    land_mask_path: str = None,
    drop_na: bool = True,
    na_columns: List[str] = None,
) -> bool:
    """
    Process zarr dataset to parquet using tiled approach with Bodo combination.
    
    Args:
        ds: Input xarray Dataset
        output_path: Final parquet output path
        hpc_root: HPC root directory for geobox
        tile_size: Size of tiles for processing (default 2048)
        apply_land_mask: Whether to filter pixels using land mask
        land_mask_path: Path to land mask zarr file (if None, auto-detect from hpc_root)
        drop_na: Whether to drop rows with NA values
        na_columns: Specific columns to check for NA (if None, check all data columns)
        
    Returns:
        bool: Success status
    """
    try:
        logger.info(f"Starting vectorized zarr to parquet conversion with {tile_size}x{tile_size} tiles")
        
        # Get or create target geobox
        if hpc_root:
            target_geobox = get_or_create_geobox(hpc_root)
        else:
            logger.warning("No geobox provided")
            return False
        
        # Load land mask if requested
        land_mask_ds = None
        if apply_land_mask:
            land_mask_ds = _load_land_mask(hpc_root, land_mask_path)
            if land_mask_ds is None:
                logger.warning("Land mask requested but not found, proceeding without filtering")
        
        # Create tiles
        tiles = GeoboxTiles(target_geobox, (tile_size, tile_size))
        total_tiles = tiles.shape[0] * tiles.shape[1]
        
        logger.info(f"Created {total_tiles} tiles ({tiles.shape[0]}x{tiles.shape[1]}) for processing")
        
        # Fail if output already exists
        if os.path.exists(output_path):
            logger.error(f"Output path already exists: {output_path}")
            return False

        # Setup out directory for tile parquet files
        os.makedirs(output_path, exist_ok=True)
                
        # Process tiles to temporary parquet files
        tile_files = _process_tiles_to_parquet(
            ds, tiles, tile_size, output_path, land_mask_ds, drop_na, na_columns
        )
        
        if not tile_files:
            logger.warning("No tile files were created")
            return False
        
        return True
        
    except Exception as e:
        logger.exception(f"Error in vectorized zarr to parquet conversion: {e}")
        return False

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
                os.path.join(hpc_root, "data", "misc", "processed", "stage_2", "osm", "land_mask.zarr"),
            ]
        else:
            potential_paths = [land_mask_path]
        
        for path in potential_paths:
            if os.path.exists(path):
                logger.info(f"Loading land mask from: {path}")
                land_mask_ds = xr.open_zarr(path)
                return land_mask_ds
        
        logger.warning(f"Land mask not found in any of: {potential_paths}")
        return None
        
    except Exception as e:
        logger.warning(f"Error loading land mask: {e}")
        return None

def _process_tiles_to_parquet(
    ds: xr.Dataset,
    tiles: GeoboxTiles,
    tile_size: int,
    tile_parquet_dir: str,
    land_mask_ds: Optional[xr.Dataset] = None,
    drop_na: bool = True,
    na_columns: List[str] = None,
) -> List[str]:
    """
    Process dataset tiles to individual parquet files sequentially.
    """
    tile_files = []

    # Create list of tile tasks
    tile_tasks = []
    for ix in range(tiles.shape[0]):
        for iy in range(tiles.shape[1]):
            tile_geobox = tiles[ix, iy]
            # Create partitioned directory structure
            tile_subdir = os.path.join(tile_parquet_dir, f"ix={ix:04d}", f"iy={iy:04d}")
            os.makedirs(tile_subdir, exist_ok=True)
            tile_filename = "tile.parquet"
            tile_path = os.path.join(tile_subdir, tile_filename)
            tile_tasks.append((ix, iy, tile_geobox, tile_path))

    logger.info(f"Processing {len(tile_tasks)} tiles sequentially")

    for idx, (ix, iy, tile_geobox, tile_path) in enumerate(tile_tasks):
        try:
            result_path = _process_single_tile(
                ds, ix, iy, tile_size, tile_geobox, tile_path, 
                land_mask_ds, drop_na, na_columns
            )
            if result_path and os.path.exists(result_path):
                tile_files.append(result_path)
        except Exception as e:
            logger.warning(f"Error processing tile [{ix}, {iy}]: {e}")

    logger.info(f"Successfully processed {len(tile_files)} tiles to parquet")
    return tile_files

def _process_single_tile(
    ds: xr.Dataset,
    ix: int,
    iy: int,
    tile_size: int,
    tile_geobox,
    tile_path: str,
    land_mask_ds: Optional[xr.Dataset] = None,
    drop_na: bool = True,
    na_columns: List[str] = None,
) -> Optional[str]:
    """
    Process a single tile to parquet format.
    
    Returns:
        str: Path to created parquet file, or None if no data
    """
    try:
        logger.info(f"Processing tile [{ix}, {iy}]")
        
        # Extract tile bounds
        bbox = tile_geobox.boundingbox
        
        # Slice dataset to tile bounds
        if 'latitude' in ds.coords and 'longitude' in ds.coords:
            tile_ds = ds.sel(
                latitude=slice(bbox.top, bbox.bottom),
                longitude=slice(bbox.left, bbox.right)
            ).compute()
        else:
            logger.warning(f"Unknown coordinate system for tile [{ix}, {iy}]")
            return None
        
        # Check if tile has data
        if tile_ds.sizes.get('latitude', 0) == 0 or tile_ds.sizes.get('longitude', 0) == 0:
            return None
        
        # Apply land mask if provided
        if land_mask_ds is not None:
            tile_ds = _apply_land_mask_to_tile(tile_ds, land_mask_ds, bbox)
            if tile_ds is None:
                logger.debug(f"No land pixels in tile [{ix}, {iy}]")
                return None
        
        # Insert pixel ID information
        pixel_id_matrix = np.arange(tile_size**2, dtype = "int32").reshape((tile_size, tile_size))  # Create index matrix for full tile
        pixel_id_matrix = pixel_id_matrix[:tile_ds.sizes['latitude'], :tile_ds.sizes['longitude']]  # Crop to smaller tile if necessary
        pixel_id_matrix = np.broadcast_to(pixel_id_matrix, tile_ds.sizes.values())
        tile_ds = tile_ds.assign({'pixel_id': (tile_ds.sizes.keys(), pixel_id_matrix)})
        
        # Transform time to year
        if "time" in tile_ds.sizes.keys():
            tile_ds.coords["time"] = pd.Series(tile_ds.coords["time"]).dt.year.astype("int16")
            
        # Convert to DataFrame
        df = tile_ds.to_dataframe().reset_index()
        
        if df.empty:
            logger.warning(f"No data in tile [{ix}, {iy}]")
            return None
        
        # Drop NA rows if requested
        if drop_na:
            initial_rows = len(df)
            df = _drop_na_rows(df, na_columns)
            final_rows = len(df)
            if final_rows < initial_rows:
                logger.debug(f"Dropped {initial_rows - final_rows} NA rows in tile [{ix}, {iy}]")
            
            if df.empty:
                logger.debug(f"No valid data remaining after NA filtering in tile [{ix}, {iy}]")
                return None
        
        # Reorder and drop columns
        col_filter = ['band', 'latitude', 'longitude', 'spatial_ref', 'time', 'pixel_id']
        col_order = ['pixel_id']
        if 'time' in df.columns:
            col_order.append('time')
        col_order += [x for x in df.columns if x not in col_filter]
        df = df.loc[:,col_order]
        
        # Write DataFrame to parquet with optimal compression
        df.to_parquet(
            tile_path, 
            engine='pyarrow', 
            compression='zstd',  # More efficient than snappy
            compression_level=3,  # Balance between compression and speed
            index=False,  # Don't write pandas index
            row_group_size=50000,  # Optimize for query performance
        )
        
        return tile_path
        
    except Exception as e:
        logger.warning(f"Error processing tile [{ix}, {iy}]: {e}")
        return None

def _apply_land_mask_to_tile(tile_ds: xr.Dataset, land_mask_ds: xr.Dataset, bbox) -> Optional[xr.Dataset]:
    """
    Apply land mask to filter out water pixels from tile dataset.
    
    Args:
        tile_ds: Tile dataset to filter
        land_mask_ds: Land mask dataset
        bbox: Tile bounding box
        
    Returns:
        Filtered dataset or None if no land pixels
    """
    try:
        # Extract land mask for this tile
        if 'latitude' in land_mask_ds.coords and 'longitude' in land_mask_ds.coords:
            tile_land_mask = land_mask_ds.sel(
                latitude=slice(bbox.top, bbox.bottom),
                longitude=slice(bbox.left, bbox.right)
            )
        else:
            logger.warning("Land mask coordinates don't match tile coordinates")
            return tile_ds
        
        # Check if land mask has the right variable
        mask_var = None
        for var_name in ['land_mask', 'land', 'mask']:
            if var_name in tile_land_mask.data_vars:
                mask_var = var_name
                break
        
        if mask_var is None:
            logger.warning("No land mask variable found in land mask dataset")
            return tile_ds
        
        # Apply mask - keep only land pixels (mask value = 1)
        land_mask_array = tile_land_mask[mask_var]
        
        # Apply mask to all data variables
        masked_ds = tile_ds.where(land_mask_array == 1)
        
        # Check if any land pixels remain
        if masked_ds.to_dataframe().dropna().empty:
            return None
        
        return masked_ds
        
    except Exception as e:
        logger.warning(f"Error applying land mask: {e}")
        return tile_ds

def _drop_na_rows(df: pd.DataFrame, na_columns: List[str] = None) -> pd.DataFrame:
    """
    Drop rows with NA values from DataFrame.
    
    Args:
        df: Input DataFrame
        na_columns: Specific columns to check for NA (if None, check all data columns)
        
    Returns:
        DataFrame with NA rows removed
    """
    if na_columns is None:
        # Exclude coordinate and metadata columns from NA checking
        exclude_cols = ['pixel_id', 'time', 'latitude', 'longitude', 'spatial_ref', 'band']
        na_columns = [col for col in df.columns if col not in exclude_cols]
    
    if not na_columns:
        return df
    
    # Drop rows where any of the specified columns have NA values
    return df.dropna(subset=na_columns)

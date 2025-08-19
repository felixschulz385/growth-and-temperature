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
) -> bool:
    """
    Process zarr dataset to parquet using tiled approach with Bodo combination.
    
    Args:
        ds: Input xarray Dataset
        output_path: Final parquet output path
        hpc_root: HPC root directory for geobox
        tile_size: Size of tiles for processing (default 2048)
        
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
        tile_files = _process_tiles_to_parquet(ds, tiles, tile_size, output_path)
        
        if not tile_files:
            logger.warning("No tile files were created")
            return False
        
        return True
        
    except Exception as e:
        logger.exception(f"Error in vectorized zarr to parquet conversion: {e}")
        return False

def _process_tiles_to_parquet(
    ds: xr.Dataset,
    tiles: GeoboxTiles,
    tile_size: int,
    tile_parquet_dir: str,
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
            result_path = _process_single_tile(ds, ix, iy, tile_size, tile_geobox, tile_path)
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
    tile_path: str
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
        
        # Insert pixel ID information
        pixel_id_matrix = np.arange(tile_size**2).reshape((tile_size, tile_size))  # Create index matrix for full tile
        pixel_id_matrix = pixel_id_matrix[:tile_ds.sizes['latitude'], :tile_ds.sizes['longitude']]  # Crop to smaller tile if necessary
        pixel_id_matrix = np.broadcast_to(pixel_id_matrix, tile_ds.sizes.values())
        tile_ds = tile_ds.assign({'pixel_id': (tile_ds.sizes.keys(), pixel_id_matrix)})
        
        # Convert to DataFrame
        df = tile_ds.to_dataframe().reset_index()
        
        if df.empty:
            logger.warning(f"No data in tile [{ix}, {iy}]")
            return None
        
        # Reorder and drop columns
        col_filter = ['band', 'latitude', 'longitude', 'spatial_ref', 'time', 'pixel_id']
        col_order = ['pixel_id']
        if 'time' in df.columns:
            col_order.append('time')
        col_order += [x for x in df.columns if x not in col_filter]
        df = df.loc[:,col_order]
        
        # Write DataFrame to parquet
        df.to_parquet(tile_path, engine='pyarrow', compression='snappy')
        
        return tile_path
        
    except Exception as e:
        logger.warning(f"Error processing tile [{ix}, {iy}]: {e}")
        return None

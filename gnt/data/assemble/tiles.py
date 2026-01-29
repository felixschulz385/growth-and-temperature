"""
Tile management functionality for data assembly.

Handles tile generation, indexing, and size calculations for
spatial partitioning of datasets.
"""

import math
import logging
from typing import Dict, Any, List, Tuple

from odc.geo import GeoboxTiles

from gnt.data.assemble.constants import DEFAULT_TILE_SIZE

logger = logging.getLogger(__name__)


def get_available_tiles(
    assembly_config: Dict[str, Any], 
    target_geobox
) -> List[Tuple[int, int]]:
    """
    Get all available tile index combinations (ix, iy) using geobox tiling.
    
    Creates a grid of tiles covering the entire target geobox extent.
    
    Args:
        assembly_config: Assembly configuration with optional tile_size
        target_geobox: Target geobox to tile
        
    Returns:
        List of (ix, iy) tuples for all tiles
    """
    tile_size = assembly_config.get('processing', {}).get('tile_size', DEFAULT_TILE_SIZE)
    tiles = GeoboxTiles(target_geobox, (tile_size, tile_size))
    
    all_tiles = [
        (ix, iy) 
        for ix in range(tiles.shape[0]) 
        for iy in range(tiles.shape[1])
    ]
    
    logger.info(f"Generated {len(all_tiles)} tiles from geobox ({tiles.shape[0]}x{tiles.shape[1]})")
    return all_tiles


def adjust_tile_size_for_reprojection(
    native_resolution: float, 
    target_resolution: float, 
    tile_size: int
) -> int:
    """
    Ensure tile size is large enough to produce at least one output pixel after reprojection.
    
    When target resolution is coarser than native resolution, multiple input pixels
    map to one output pixel. This function ensures the tile is large enough to
    guarantee at least one output pixel after resampling.
    
    Args:
        native_resolution: Native resolution of the source data
        target_resolution: Target resolution for output (or None to skip)
        tile_size: Current tile size in pixels
        
    Returns:
        Adjusted tile size (may be larger than input)
    """
    if target_resolution is None:
        return tile_size
    
    min_tile_pixels = max(1, math.ceil(target_resolution / native_resolution))
    
    if tile_size < min_tile_pixels:
        logger.info(
            f"Increasing tile_size from {tile_size} to {min_tile_pixels} to cover "
            f"at least one reprojected pixel (native_res={native_resolution}, "
            f"target_res={target_resolution})."
        )
        return min_tile_pixels
    
    return tile_size


def create_tile_geobox(
    target_geobox, 
    tile_size: int, 
    ix: int, 
    iy: int
):
    """
    Create a geobox for a specific tile.
    
    Args:
        target_geobox: Full target geobox
        tile_size: Tile size in pixels
        ix, iy: Tile indices
        
    Returns:
        Geobox for the specified tile
    """
    tiles = GeoboxTiles(target_geobox, (tile_size, tile_size))
    return tiles[ix, iy]

"""
Tile management functionality for data assembly.

Handles tile generation, indexing, and size calculations for
spatial partitioning of datasets.
"""

import math
import logging
from typing import Dict, Any, List, Tuple

from src.data.assemble.constants import DEFAULT_TILE_SIZE
from src.data.common import tiling

logger = logging.getLogger(__name__)


def get_available_tiles(
    assembly_config: Dict[str, Any],
    target_geobox
) -> List[Tuple[int, int]]:
    """
    Get all available tile index combinations (ix, iy) using the shared PREPARE
    tile grid (`src.data.common.tiling`), so assembly's tile numbering is
    structurally guaranteed to match whatever tiled-parquet PREPARE output it
    reads -- not just accidentally identical from two independent
    `GeoboxTiles` constructions.

    Args:
        assembly_config: Assembly configuration with optional tile_size
        target_geobox: Target geobox to tile

    Returns:
        List of (ix, iy) tuples for all tiles
    """
    tile_size = assembly_config.get('processing', {}).get('tile_size', DEFAULT_TILE_SIZE)
    n_rows, n_cols = tiling.grid_shape(target_geobox, tile_size)

    all_tiles = [(ix, iy) for ix in range(n_rows) for iy in range(n_cols)]

    logger.info(f"Generated {len(all_tiles)} tiles from geobox ({n_rows}x{n_cols})")
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
    return tiling.build_tile_grid(target_geobox, tile_size)[ix, iy]

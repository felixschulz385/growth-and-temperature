"""
Utility functions for data assembly.

Contains helper functions for path manipulation, data transformations,
and other reusable operations.
"""

import re
import logging
import numpy as np
import xarray as xr
from odc.geo.xr import ODCExtensionDa
from typing import Optional

from gnt.data.assemble.constants import (
    DEFAULT_CRS,
    PIXEL_ID_IX_SHIFT,
    PIXEL_ID_IY_SHIFT,
    LATITUDE_COORD,
    LONGITUDE_COORD,
)

logger = logging.getLogger(__name__)


def strip_remote_prefix(path: str) -> str:
    """
    Remove scp/ssh prefix like user@host: from paths.
    
    Args:
        path: Path string potentially containing remote prefix
        
    Returns:
        Path with remote prefix removed
    """
    if isinstance(path, str):
        return re.sub(r"^[^@]+@[^:]+:", "", path)
    return path


def winsorize(array: xr.DataArray, cutoff: float = 0.001) -> xr.DataArray:
    """
    Apply winsorization to clip outliers at specified quantiles.
    
    Winsorization replaces extreme values with values at specified percentiles,
    reducing the impact of outliers while preserving the data distribution shape.
    
    Args:
        array: xarray DataArray to winsorize
        cutoff: Quantile cutoff on both sides (e.g., 0.001 clips at 0.1% and 99.9%)
    
    Returns:
        Winsorized array with NaN values preserved
        
    Example:
        >>> winsorized = winsorize(temperature_data, cutoff=0.01)  # Clip at 1% and 99%
    """
    lower_quantile = cutoff
    upper_quantile = 1.0 - cutoff
    
    lower_bound = array.quantile(lower_quantile)
    upper_bound = array.quantile(upper_quantile)
    
    return (
        array
        .where(array > lower_bound, lower_bound)
        .where(array < upper_bound, upper_bound)
        .where(~array.isnull())
    )


def make_pixel_ids(ix: int, iy: int, tile_geobox) -> xr.Dataset:
    """
    Generate pixel ID xarray Dataset with pixel_id as a data variable.
    
    Format: [ix: 16 bits | iy: 16 bits | local_pixel: 32 bits]
    This encoding allows decoding tile coordinates and pixel location from a single integer,
    enabling efficient spatial indexing and tile reconstruction.
    
    Args:
        ix: Tile x index (must fit in 16 bits)
        iy: Tile y index (must fit in 16 bits)
        tile_geobox: Target geobox for tile
    
    Returns:
        xarray Dataset with pixel_id as a data variable and lat/lon coordinates
        
    Raises:
        ValueError: If ix or iy exceed 16-bit range
    """
    if ix >= 2**16 or iy >= 2**16:
        raise ValueError(f"Tile indices ({ix}, {iy}) exceed 16-bit range")
    
    h, w = tile_geobox.shape
    local_pixel_ids = np.arange(h * w, dtype="uint32").reshape((h, w))
    
    pixel_id_matrix = (
        (np.uint64(ix) << PIXEL_ID_IX_SHIFT) | 
        (np.uint64(iy) << PIXEL_ID_IY_SHIFT) | 
        local_pixel_ids.astype(np.uint64)
    )
    
    pixel_id_ds = xr.Dataset(
        data_vars={'pixel_id': ([LATITUDE_COORD, LONGITUDE_COORD], pixel_id_matrix)},
        coords={
            LATITUDE_COORD: tile_geobox.coords[LATITUDE_COORD].values,
            LONGITUDE_COORD: tile_geobox.coords[LONGITUDE_COORD].values
        }
    )
    pixel_id_ds = pixel_id_ds.odc.assign_crs(DEFAULT_CRS)
    
    return pixel_id_ds


def decode_pixel_id(pixel_id: np.uint64) -> tuple:
    """
    Decode a pixel_id back into its components.
    
    Args:
        pixel_id: Encoded pixel ID
        
    Returns:
        Tuple of (ix, iy, local_pixel_index)
    """
    ix = int((pixel_id >> PIXEL_ID_IX_SHIFT) & 0xFFFF)
    iy = int((pixel_id >> PIXEL_ID_IY_SHIFT) & 0xFFFF)
    local_pixel = int(pixel_id & 0xFFFFFFFF)
    return ix, iy, local_pixel


def convert_int_to_float32(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert integer variables in dataset to float32.
    
    Args:
        ds: Input dataset
        
    Returns:
        Dataset with integer variables converted to float32
    """
    int_vars = [name for name, dtype in ds.dtypes.items() if np.issubdtype(dtype, np.integer)]
    if int_vars:
        for var in int_vars:
            ds[var] = ds[var].astype("float32")
    return ds


def apply_column_prefix(ds: xr.Dataset, prefix: str) -> xr.Dataset:
    """
    Apply a prefix to all data variable names in the dataset.
    
    Args:
        ds: Input dataset
        prefix: Prefix to apply to variable names
        
    Returns:
        Dataset with renamed variables
    """
    rename_dict = {var: f"{prefix}{var}" for var in ds.data_vars}
    return ds.rename(rename_dict)

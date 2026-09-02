"""
Utility functions for data assembly.

Contains helper functions for path manipulation, data transformations,
and other reusable operations.
"""

import fnmatch
import re
import logging
import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.xr import ODCExtensionDa
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.data.assemble.constants import (
    DEFAULT_RESAMPLING_METHOD,
    PIXEL_ID_IX_SHIFT,
    PIXEL_ID_IY_SHIFT,
    LATITUDE_COORD,
    LONGITUDE_COORD,
    VALID_RESAMPLING_METHODS,
)

logger = logging.getLogger(__name__)


def resolve_resampling(
    config: Any,
    var_names: Iterable[str],
) -> Dict[str, str]:
    """Map each variable name to an ``odc.reproject`` resampling method.

    *config* (a dataset's ``resampling`` value) is one of:

    - ``None`` -> every variable gets ``DEFAULT_RESAMPLING_METHOD``.
    - a method string -> applied to every variable.
    - a mapping: the reserved ``default`` key is the fallback; every other key is
      an :mod:`fnmatch` glob tested against variable names, and the **first**
      matching pattern (in config order) wins. Missing ``default`` ->
      ``DEFAULT_RESAMPLING_METHOD``.

    Example (MODIS: LST means area-averaged, valid-obs counts summed)::

        resampling:
          default: average
          "valid_*count*": sum

    Raises ``ValueError`` on a non-str/non-mapping config or an unknown method.
    """
    names = list(var_names)

    def _check(method: str) -> str:
        if method not in VALID_RESAMPLING_METHODS:
            raise ValueError(
                f"Unknown resampling method {method!r}. "
                f"Valid: {', '.join(sorted(VALID_RESAMPLING_METHODS))}"
            )
        return method

    if config is None:
        return {name: DEFAULT_RESAMPLING_METHOD for name in names}
    if isinstance(config, str):
        return {name: _check(config) for name in names}
    if not isinstance(config, dict):
        raise ValueError(
            f"'resampling' must be a method string or a mapping, got {type(config).__name__}"
        )

    default = _check(str(config.get("default", DEFAULT_RESAMPLING_METHOD)))
    patterns = [(str(k), _check(str(v))) for k, v in config.items() if k != "default"]

    resolved: Dict[str, str] = {}
    for name in names:
        method = default
        for pattern, pattern_method in patterns:
            if fnmatch.fnmatchcase(name, pattern):
                method = pattern_method
                break
        resolved[name] = method
    return resolved

DEFAULT_DERIVED_PIXEL_ID_RESOLUTIONS = {
    "500m": 0.00417,
    "1km": 0.01,
    "5km": 0.05,
    "50km": 0.5,
}


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


def geobox_spatial_dims(geobox) -> Tuple[str, str]:
    """``(y_name, x_name)`` for *geobox* -- ``('latitude', 'longitude')`` for a
    geographic CRS, ``('y', 'x')`` for a projected one (e.g. EASE 6933). The
    assemble stage keys everything off this instead of assuming lat/lon, so it
    works on whichever grid ``pipeline.grid`` selects."""
    return tuple(geobox.dimensions)  # type: ignore[return-value]


def dataset_spatial_dims(obj) -> Optional[Tuple[str, str]]:
    """``(y_name, x_name)`` for a Dataset/DataArray's actual spatial dims, or
    ``None`` if it has neither known pair. Checks the real xarray dim names
    (not ``obj.odc.geobox.dimensions``, which odc canonicalizes to ``y``/``x``
    even when the array's dims are really ``latitude``/``longitude``)."""
    dims = set(getattr(obj, "dims", ()) or ())
    for pair in (("y", "x"), (LATITUDE_COORD, LONGITUDE_COORD)):
        if pair[0] in dims and pair[1] in dims:
            return pair
    return None


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

    dim_y, dim_x = geobox_spatial_dims(tile_geobox)
    pixel_id_ds = xr.Dataset(
        data_vars={'pixel_id': ([dim_y, dim_x], pixel_id_matrix)},
        coords={
            dim_y: tile_geobox.coords[dim_y].values,
            dim_x: tile_geobox.coords[dim_x].values,
        },
    )
    pixel_id_ds = pixel_id_ds.odc.assign_crs(tile_geobox.crs)

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


def encode_pixel_ids(ix: int, iy: int, local_pixels: np.ndarray) -> np.ndarray:
    """Encode local pixel indices into packed uint64 pixel IDs."""
    return (
        (np.uint64(ix) << PIXEL_ID_IX_SHIFT)
        | (np.uint64(iy) << PIXEL_ID_IY_SHIFT)
        | local_pixels.astype(np.uint64)
    )


def normalize_derived_pixel_id_specs(
    config: Optional[Dict[str, object]],
) -> List[Tuple[str, float]]:
    """Normalize a config mapping of derived pixel ID columns to resolutions."""
    if not config:
        return []

    specs: List[Tuple[str, float]] = []
    for column_name, raw_value in config.items():
        if raw_value is None:
            continue
        if isinstance(raw_value, str):
            try:
                resolution = DEFAULT_DERIVED_PIXEL_ID_RESOLUTIONS[raw_value]
            except KeyError as exc:
                available = ", ".join(sorted(DEFAULT_DERIVED_PIXEL_ID_RESOLUTIONS))
                raise ValueError(
                    f"Unknown derived pixel ID grid label {raw_value!r} for {column_name!r}. "
                    f"Available labels: {available}"
                ) from exc
        else:
            resolution = float(raw_value)
        specs.append((str(column_name), float(resolution)))
    return specs


def _centers_to_indices(values: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Map center coordinates to integer pixel indices on an axis."""
    if coords.ndim != 1 or coords.size == 0:
        raise ValueError("Coordinate array must be 1D and non-empty")

    if coords.size == 1:
        return np.zeros(values.shape, dtype=np.int64)

    step = float(coords[1] - coords[0])
    if step == 0:
        raise ValueError("Coordinate step cannot be zero")

    if step > 0:
        edge0 = float(coords[0]) - step / 2.0
        indices = np.floor((values - edge0) / step)
    else:
        cell = abs(step)
        edge0 = float(coords[0]) + cell / 2.0
        indices = np.floor((edge0 - values) / cell)

    indices = indices.astype(np.int64)
    return np.clip(indices, 0, coords.size - 1)


def build_derived_pixel_id_mapping(
    pixel_ids: np.ndarray,
    ix: int,
    iy: int,
    base_tile_geobox,
    source_geobox,
    derived_specs: Iterable[Tuple[str, float]],
) -> pd.DataFrame:
    """Build a dataframe mapping canonical pixel IDs to alternate grid IDs."""
    specs = list(derived_specs)
    if pixel_ids.size == 0:
        return pd.DataFrame(columns=["pixel_id"] + [name for name, _ in specs])

    source_h, source_w = source_geobox.shape
    decoded = np.array([decode_pixel_id(np.uint64(pid)) for pid in pixel_ids], dtype=np.int64)
    local_pixels = decoded[:, 2]
    rows = local_pixels // source_w
    cols = local_pixels % source_w

    src_dim_y, src_dim_x = geobox_spatial_dims(source_geobox)
    source_ys = source_geobox.coords[src_dim_y].values
    source_xs = source_geobox.coords[src_dim_x].values
    ys = source_ys[rows]
    xs = source_xs[cols]

    mapping = pd.DataFrame({"pixel_id": pixel_ids})
    for column_name, resolution in specs:
        target_geobox = base_tile_geobox.zoom_to(resolution=resolution)
        target_h, target_w = target_geobox.shape
        tgt_dim_y, tgt_dim_x = geobox_spatial_dims(target_geobox)
        target_rows = _centers_to_indices(ys, target_geobox.coords[tgt_dim_y].values)
        target_cols = _centers_to_indices(xs, target_geobox.coords[tgt_dim_x].values)
        target_local = target_rows * target_w + target_cols
        if np.any(target_local < 0) or np.any(target_local >= target_h * target_w):
            raise ValueError(
                f"Derived local pixels for column {column_name!r} are out of bounds for tile ({ix}, {iy})"
            )
        mapping[column_name] = encode_pixel_ids(ix, iy, target_local)

    return mapping


def add_derived_pixel_id_columns(
    df: pd.DataFrame,
    ix: int,
    iy: int,
    base_tile_geobox,
    source_geobox,
    derived_specs: Iterable[Tuple[str, float]],
) -> pd.DataFrame:
    """Append derived pixel ID columns to a dataframe that already contains `pixel_id`."""
    specs = list(derived_specs)
    if not specs or df is None or df.empty or "pixel_id" not in df.columns:
        return df

    for column_name, _ in specs:
        df = df.drop(columns=[column_name], errors="ignore")

    unique_ids = pd.Index(df["pixel_id"].dropna().unique()).astype("uint64").to_numpy()
    mapping = build_derived_pixel_id_mapping(
        pixel_ids=unique_ids,
        ix=ix,
        iy=iy,
        base_tile_geobox=base_tile_geobox,
        source_geobox=source_geobox,
        derived_specs=specs,
    )
    if mapping.empty:
        for column_name, _ in specs:
            df[column_name] = pd.Series(dtype="uint64")
        return df

    return df.merge(mapping, on="pixel_id", how="left")


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

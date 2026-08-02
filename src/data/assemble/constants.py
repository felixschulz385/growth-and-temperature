"""
Constants used throughout the assembly module.

Centralizes magic numbers and configuration defaults for maintainability.
"""

from src.data.common.dask.client import DEFAULT_DASHBOARD_PORT

# Default coordinate reference system
DEFAULT_CRS = 4326

# Default tile processing parameters
DEFAULT_TILE_SIZE = 2048
DEFAULT_TILE_PADDING = 64

# Default compression for parquet output
DEFAULT_COMPRESSION = 'snappy'

# Default resampling method for datasets
DEFAULT_RESAMPLING_METHOD = 'mode'

# Default Dask configuration -- kept under this module's own name since
# src/data/assemble/config.py already imports it as such; the value itself
# comes from the single shared constant, not redefined here.
DEFAULT_DASK_DASHBOARD_PORT = DEFAULT_DASHBOARD_PORT
DEFAULT_WORKER_THREADS_PER_CPU = 2
DEFAULT_WORKER_FRACTION = 0.5

# Pixel ID bit layout: [ix: 16 bits | iy: 16 bits | local_pixel: 32 bits]
PIXEL_ID_IX_SHIFT = 48
PIXEL_ID_IY_SHIFT = 32

# Coordinate names
LATITUDE_COORD = 'latitude'
LONGITUDE_COORD = 'longitude'
TIME_COORD = 'time'
YEAR_COORD = 'year'

# Variables to exclude from processing
EXCLUDED_VARIABLES = ['spatial_ref']

# Land mask paths (relative to hpc_root). load_land_mask() tries each in
# order and uses the first that exists, so this list is how it stays
# layout-aware: the legacy per-source path is tried first, then the
# layout:v2 single-source-family path (src/data/sources/layout.py's
# grid_store_path(), v2_family="land_mask") under each grid this repo
# supports -- only one will ever exist for a given run, since layout:v2 is
# only ever run against one grid at a time.
LAND_MASK_RELATIVE_PATHS = [
    "misc/processed/stage_2/osm/land_mask.zarr",
    "grid/legacy_4326/land_mask.zarr",
    "grid/ease6933/land_mask.zarr",
]

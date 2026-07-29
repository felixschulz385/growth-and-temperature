"""
Constants used throughout the assembly module.

Centralizes magic numbers and configuration defaults for maintainability.
"""

# Default coordinate reference system
DEFAULT_CRS = 4326

# Default tile processing parameters
DEFAULT_TILE_SIZE = 2048
DEFAULT_TILE_PADDING = 64

# Default compression for parquet output
DEFAULT_COMPRESSION = 'snappy'

# Default resampling method for datasets
DEFAULT_RESAMPLING_METHOD = 'mode'

# Default Dask configuration
DEFAULT_DASK_DASHBOARD_PORT = 8787
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

# Land mask paths (relative to hpc_root)
LAND_MASK_RELATIVE_PATHS = [
    "misc/processed/stage_2/osm/land_mask.zarr",
]

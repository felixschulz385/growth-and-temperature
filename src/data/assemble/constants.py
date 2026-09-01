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

# `assemble --grid <label>` -> output resolution in metres on the canonical
# EPSG:6933 grid. "1km" is the native canonical resolution (no downsampling);
# every coarser label triggers a downsampling reprojection and makes
# `--shake` meaningful (docs/design/01-grid.md §2, docs/design/04-ingest.md §6).
GRID_RESOLUTIONS_M = {
    "1km": 1000.0,
    "2km": 2000.0,
    "5km": 5000.0,
    "10km": 10000.0,
    "25km": 25000.0,
}
DEFAULT_GRID_LABEL = "1km"

# Partition label for the un-shifted assembled table. Grid-shake variants are
# written as full same-schema sibling tables under shake=s0/s1/... alongside it.
SHAKE_BASE_LABEL = "base"

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
# order and uses the first that exists (src/data/sources/layout.py's
# grid_store_path(), family="land_mask") under each grid this repo supports
# -- only one will ever exist for a given run, since a run only ever targets
# one grid at a time.
LAND_MASK_RELATIVE_PATHS = [
    "grid/legacy_4326/land_mask.zarr",
    "grid/ease6933/land_mask.zarr",
]

"""
Constants used throughout the assembly module.

Centralizes magic numbers and configuration defaults for maintainability.
"""

# Default coordinate reference system
DEFAULT_CRS = 4326

# Default tile processing parameters
DEFAULT_TILE_SIZE = 2048
DEFAULT_TILE_PADDING = 64

# `assemble --grid <label>` -> output resolution in metres on the canonical
# EPSG:6933 grid. "1km" is the native canonical resolution (no downsampling);
# every coarser label triggers an integer block aggregation and makes
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

# Downsampling on the canonical grid is an exact integer block aggregation
# (assemble/sql_engine.py), so every `resampling` method a dataset declares must
# map to a SQL aggregate. `SQL_RESAMPLING_AGGREGATES[m]` is a callable
# `col -> aggregate SQL expression`. `VALID_RESAMPLING_METHODS` is exactly the
# set of keys -- kernel-based methods (nearest/bilinear/cubic/...) have no
# block-aggregate equivalent and are rejected by validate_assembly_config.
SQL_RESAMPLING_AGGREGATES = {
    'average': lambda c: f'avg({c})',
    'sum':     lambda c: f'sum({c})',
    'max':     lambda c: f'max({c})',
    'min':     lambda c: f'min({c})',
    'mode':    lambda c: f'mode({c})',
    'med':     lambda c: f'median({c})',
    'q1':      lambda c: f'quantile_cont({c}, 0.25)',
    'q3':      lambda c: f'quantile_cont({c}, 0.75)',
    'rms':     lambda c: f'sqrt(avg({c} * {c}))',
}
VALID_RESAMPLING_METHODS = frozenset(SQL_RESAMPLING_AGGREGATES)

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

"""Source-path resolution for the DuckDB assembly engine.

The engine reads every GRID-stage source straight from its ``cell_id``-keyed
tiled-parquet directory (``run_tiled_prepare`` output), so there is nothing to
"load" into memory here -- only the land mask needs its on-disk location
resolved, the same way the sources' ``path`` is resolved in
:func:`src.data.assemble.config.resolve_dataset_paths`.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from src.data.assemble.constants import LAND_MASK_RELATIVE_PATHS
from src.data.assemble.parquet_raster import is_tiled_parquet_dataset
from src.data.sources import layout

logger = logging.getLogger(__name__)


def resolve_land_mask_path(data_root: str, grid_id: str) -> Optional[str]:
    """Return the land mask's tiled-parquet directory for *grid_id*, or ``None``.

    Tries the ``layout.grid_store_path`` location for the run's grid first, then
    the historical fixed relative paths. Only a ``run_tiled_prepare``-shaped
    directory is accepted -- the DuckDB engine cannot read a Zarr land mask.
    """
    candidates = [
        layout.grid_store_path(data_root, "misc", grid_id=grid_id, family="land_mask", suffix=""),
        layout.grid_store_path(
            data_root, "misc", grid_id=layout.EASE_GRID_ID, family="land_mask", suffix=""
        ),
    ] + [os.path.join(data_root, rel) for rel in LAND_MASK_RELATIVE_PATHS]

    for path in candidates:
        if is_tiled_parquet_dataset(path):
            logger.info("Using land mask: %s", path)
            return path
        if os.path.exists(path):
            logger.warning(
                "Land mask at %s is not tiled parquet; the DuckDB assembly engine cannot "
                "use it. Skipping.", path,
            )
    logger.warning("No tiled-parquet land mask found under %s", data_root)
    return None

"""Single source of truth for "which geobox does this run's GRID/PREPARE
step target, given ctx.grid_id".

Generalizes the branch that today only `ModisSource._execute_grid`
(src/data/sources/modis/source.py) implements correctly and unconditionally
-- every other source ignores `ctx.grid_id` for the actual reprojection
target, even though `DataSource.output_root()` already picks the right
*directory name* (`stage_2` vs `stage_2_ease6933`) from it. A free function
keyed on `ctx` rather than a `DataSource` method because `snl_mining` needs
the same branch during PREPARE (to pick the CRS baked into a DuckDB
`ST_Transform`), not just during GRID.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from src.data.sources.layout import EASE_GRID_ID

from .canonical import get_or_create_canonical_geobox
from .geobox import get_or_create_geobox

if TYPE_CHECKING:
    from src.data.pipeline.context import PipelineContext

#: Shared cache filename for the canonical EASE geobox, codifying the
#: convention ModisSource already used as a literal string.
CANONICAL_GEOBOX_CACHE_FILENAME = "canonical_geobox.pkl"


def get_target_geobox(ctx: "PipelineContext"):
    """Return the geobox a source should reproject/rasterize onto for `ctx`.

    `ctx.grid_id == EASE_GRID_ID` -> canonical EPSG:6933 EASE-Grid, cached at
    `<ctx.data_root>/canonical_geobox.pkl`. Otherwise -> the legacy
    VIIRS-derived EPSG:4326 grid (today's default, unchanged).
    """
    if ctx.grid_id == EASE_GRID_ID:
        cache_path = os.path.join(ctx.data_root, CANONICAL_GEOBOX_CACHE_FILENAME)
        return get_or_create_canonical_geobox(cache_path)
    return get_or_create_geobox(ctx.data_root)

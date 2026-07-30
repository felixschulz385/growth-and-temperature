"""Deprecated import path -- moved to src.data.common.raster.spatial.

docs/design/09-integrated-pipeline.md §4/§10 step 2: reprojection onto the
canonical geobox is shared library code used by every source's GRID step, not
owned by the preprocess subsystem being retired. This shim keeps every
`src.data.preprocess.sources.*` module working unmodified during the
migration; delete it in step 10's cutover alongside the rest of
`src/data/preprocess/`.
"""

from src.data.common.raster.spatial import *  # noqa: F401,F403
from src.data.common.raster.spatial import SpatialProcessor  # noqa: F401

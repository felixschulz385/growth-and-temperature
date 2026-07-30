"""Deprecated import path -- moved to src.data.sources.modis.tiles.

docs/design/09-integrated-pipeline.md §4/§10 step 6: MODIS sinusoidal tile
math is now owned by the merged `src/data/sources/modis/` package. This shim
keeps the not-yet-deleted `src/data/preprocess/sources/modis.py` working
unmodified; delete it in step 10's cutover alongside the rest of
`src/data/preprocess/`.
"""

from src.data.sources.modis.tiles import *  # noqa: F401,F403

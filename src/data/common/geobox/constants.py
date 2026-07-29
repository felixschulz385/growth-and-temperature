"""Constants for canonical grid construction.

See docs/design/01-grid.md.
"""

# EASE-Grid 2.0 Global (Lambert cylindrical equal-area, standard parallel 30 deg).
DEFAULT_CANONICAL_CRS = "EPSG:6933"

# Canonical grid's latitude clip -- confirmed with the user
# (docs/design/00-backbone-overview.md, docs/design/01-grid.md §1).
DEFAULT_LAT_CLIP_DEG = 60.0

# Canonical grid resolution in metres -- confirmed with the user
# (docs/design/00-backbone-overview.md, docs/design/01-grid.md §2).
DEFAULT_RESOLUTION_M = 1000.0

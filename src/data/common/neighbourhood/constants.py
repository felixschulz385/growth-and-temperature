"""Constants for the neighbourhood (ring/annulus) convolution engine.

See docs/design/01-grid.md and docs/design/02-storage.md for the reasoning
behind these defaults.
"""

# Standard parallel of the canonical EPSG:6933 (EASE-Grid 2.0 Global) grid.
DEFAULT_STANDARD_PARALLEL_DEG = 30.0

# Canonical grid's latitude clip (docs/design/01-grid.md §1).
DEFAULT_LAT_CLIP_DEG = 60.0

# Disc-radius ladder in km (docs/design/02-storage.md §4). Dense near the
# origin where most identifying variation sits, coarsening toward R_max.
DEFAULT_DISC_LADDER_KM = [1, 2, 3, 4, 5, 7, 10, 14, 20, 30]

# Top of the ladder == R_max, confirmed with the user (docs/design/00-backbone-overview.md).
DEFAULT_R_MAX_KM = 30

# Fractional change in the anisotropy ratio cos^2(phi_s)/cos^2(phi) tolerated
# within a single latitude band (docs/design/01-grid.md §6).
DEFAULT_BAND_TOLERANCE = 0.02

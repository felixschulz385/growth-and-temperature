"""
MODIS-specific helpers: sinusoidal tile grid geometry and QC decoding.

Kept out of `modis.py` so the tile-list/QC logic (and its unit-testable
UNVERIFIED assumptions -- docs/design/07a-modis-band-reference.md) is
isolated from the preprocessor's STAC/IO plumbing.
"""

import logging
import math
from typing import List, Optional, Set

import xarray as xr

logger = logging.getLogger(__name__)

# Standard MODIS Sinusoidal Tile Grid constants (LP DAAC), computed here
# rather than trusting a single hardcoded published figure, mirroring
# docs/design/01-grid.md §2's "compute, don't hardcode" convention -- these
# four numbers are the grid's defining constants, everything else (tile
# bounds, per-tile latitude range) is derived from them below.
SPHERE_RADIUS_M = 6371007.181
GRID_XMIN = -20015109.354
GRID_XMAX = 20015109.354
GRID_YMIN = -10007554.677
GRID_YMAX = 10007554.677
N_H = 36
N_V = 18
TILE_SIZE_M = (GRID_XMAX - GRID_XMIN) / N_H

SINUSOIDAL_PROJ4 = (
    f"+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R={SPHERE_RADIUS_M} +units=m +no_defs"
)


def tile_bounds_m(h: int, v: int) -> tuple:
    """Sinusoidal-projection (x0, y0, x1, y1) bounds of tile (h, v)."""
    x0 = GRID_XMIN + h * TILE_SIZE_M
    x1 = x0 + TILE_SIZE_M
    y1 = GRID_YMAX - v * TILE_SIZE_M
    y0 = y1 - TILE_SIZE_M
    return x0, y0, x1, y1


def tile_lat_range_deg(v: int) -> tuple:
    """Latitude range (lat0, lat1) of tile row *v*.

    Exact, not approximate: the sinusoidal projection's defining property is
    y = R * phi (independent of longitude), so a tile row's latitude range
    is determined purely by its y-extent -- no pyproj transform needed.
    """
    _, y0, _, y1 = tile_bounds_m(0, v)
    return math.degrees(y0 / SPHERE_RADIUS_M), math.degrees(y1 / SPHERE_RADIUS_M)


def get_modis_sinusoidal_tiles(
    lat_clip_deg: float = 60.0,
    land_tiles: Optional[Set[str]] = None,
) -> List[str]:
    """Sinusoidal tile ids (``hXXvYY``) intersecting |lat| <= lat_clip_deg.

    docs/design/07-modis-ingest.md §3: restrict the tile list to tiles
    intersecting the clip at build time, rather than ingesting and later
    discarding high-latitude tiles.

    `land_tiles`, if given, additionally restricts to a caller-supplied
    allowlist of land-covering tile ids. This module does not itself
    determine land/ocean membership -- LP DAAC's officially published
    land-tile list (or a coastline mask) is the correct source for that and
    is not wired up in this repo; the ~317-land-tile figure in
    docs/design/07a-modis-band-reference.md is explicitly flagged
    UNVERIFIED for exactly this reason. Without `land_tiles`, this returns
    every tile (ocean-only included) within the latitude clip.
    """
    tiles = []
    for v in range(N_V):
        lat0, lat1 = tile_lat_range_deg(v)
        if lat1 < -lat_clip_deg or lat0 > lat_clip_deg:
            continue
        for h in range(N_H):
            tile_id = f"h{h:02d}v{v:02d}"
            if land_tiles is not None and tile_id not in land_tiles:
                continue
            tiles.append(tile_id)
    return tiles


# Literature-cited (not primary-source-verified) 8-bit QC layout for the
# MOD11/MYD11-family L3 gridded QC_Day/QC_Night field -- see
# docs/design/07a-modis-band-reference.md, "QC bit layout -- the single most
# important unresolved item in this document". `qc_max_lst_error_k` is
# deliberately a caller-supplied threshold (docs/design/07-modis-ingest.md
# §6: "implement the threshold as a configurable parameter ... do not
# hardcode a threshold until the bit layout is confirmed") so a wrong
# assumed layout is a one-line config fix, not a silently wrong mask baked
# into a completed ingest run.
_LST_ERROR_K_BY_BITS = {0: 1.0, 1: 2.0, 2: 3.0, 3: float("inf")}

_QC_LAYOUT_WARNED = False


def decode_qc_valid_mask(qc: xr.DataArray, max_lst_error_k: float = 2.0) -> xr.DataArray:
    """Boolean valid-observation mask from a QC_Day/QC_Night band.

    UNVERIFIED bit layout -- confirm against the MOD11 V6.1 user guide's PDF
    QC table (not machine-extractable this session) or a peer-reviewed
    methods paper before a production run. Requires mandatory QA bits == 00
    ("good") and the LST error category <= `max_lst_error_k`.
    """
    global _QC_LAYOUT_WARNED
    if not _QC_LAYOUT_WARNED:
        logger.warning(
            "decode_qc_valid_mask: QC bit layout is UNVERIFIED from a primary source "
            "for the L3 gridded product -- see docs/design/07a-modis-band-reference.md. "
            "Confirm before a production run."
        )
        _QC_LAYOUT_WARNED = True

    qc_uint = qc.astype("uint8")
    mandatory_qa = qc_uint & 0b00000011
    error_bits = (qc_uint >> 6) & 0b11

    error_k = xr.zeros_like(qc_uint, dtype="float32")
    for bits, k in _LST_ERROR_K_BY_BITS.items():
        error_k = xr.where(error_bits == bits, k, error_k)

    return (mandatory_qa == 0) & (error_k <= max_lst_error_k)

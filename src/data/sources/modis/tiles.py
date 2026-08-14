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
    allowlist of land-covering tile ids -- see `compute_land_tiles()` below
    for how to derive that allowlist from a land-polygon layer (e.g. the
    `osm` source's `land_polygons_simplified.gpkg`). Without `land_tiles`,
    this returns every tile (ocean-only included) within the latitude clip.
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


def compute_land_tiles(land_polygons_path: str, lat_clip_deg: float = 60.0) -> Set[str]:
    """Sinusoidal tile ids (within |lat| <= lat_clip_deg) overlapping land.

    Derives the `land_tiles` allowlist `get_modis_sinusoidal_tiles()` accepts
    from a real land-polygon layer -- e.g. the `osm` source's PREPARE output,
    `misc/prepared/osm/land_polygons_simplified.gpkg` -- rather than trusting
    the ~317-land-tile figure docs/design/07a-modis-band-reference.md flags
    UNVERIFIED. Answers exactly the question that figure was never checked
    against: which of the 36x18 sinusoidal tiles actually intersect land.

    Mirrors gadm.py's per-tile overlap pre-filter (reproject the vector layer
    to the tile grid's CRS once, up front, then a plain shapely
    `intersects()` per tile) via the same shared
    `reproject_for_tile_overlap()` helper gadm/ecoregions use -- see its
    docstring for the CRS-mismatch pitfall (comparing un-reprojected WGS84
    degrees against projected-meter tile boxes silently finds ~no overlap)
    that reprojecting first, rather than per tile, avoids. Unlike a GRID
    step, this isn't a zarr rasterization: it is a one-time, 36x18-tile
    bounding-box overlap check against a vector layer, producing a plain
    tile-id list for a STAC query filter, so there's no zarr chunk size to
    align processing tiles to.
    """
    import geopandas as gpd
    import shapely.geometry

    from src.data.common.raster.spatial import reproject_for_tile_overlap

    gdf = gpd.read_file(land_polygons_path, engine="pyogrio")
    gdf = reproject_for_tile_overlap(gdf, SINUSOIDAL_PROJ4)

    land_tiles: Set[str] = set()
    for v in range(N_V):
        lat0, lat1 = tile_lat_range_deg(v)
        if lat1 < -lat_clip_deg or lat0 > lat_clip_deg:
            continue
        for h in range(N_H):
            tile_box = shapely.geometry.box(*tile_bounds_m(h, v))
            if gdf.geometry.intersects(tile_box).any():
                land_tiles.add(f"h{h:02d}v{v:02d}")
    return land_tiles


# 8-bit QC layout for the MOD11/MYD11-family and MOD21/MYD21-family L3
# gridded QC_Day/QC_Night fields. Bits 1&0 (mandatory QA, 00=good) and the
# overall 8-bit shape are the same across both families -- confirmed from
# two separate primary sources:
#   - MOD11A1: "Collection-6 MODIS Land Surface Temperature Products Users'
#     Guide" (Wan, ERI/UCSB, June 2019), Table 13.
#   - MOD21A2: "MODIS Land Surface Temperature and Emissivity Product (MxD21)
#     User Guide, Collection-6" (Hulley et al., JPL, March 2019), Table 12
#     ("Bit flags defined in the QC_Day and QC_Night SDS in the MxD21A2
#     8-day product").
#
# Bits 7&6 sit at the same position in both ("LST error"/"LST accuracy") but
# **mean the opposite thing** -- not assumed to match, and they don't:
#   MOD11A1  (increasing bit value = worse):  00 <=1K, 01 <=2K, 10 <=3K, 11 >3K
#   MOD21A2  (increasing bit value = better): 00 >2K,  01 1.5-2K, 10 1-1.5K, 11 <1K
# Applying MOD11's mapping to MOD21A2 data (as this module did before this
# was checked against Table 12) would silently invert the quality filter --
# keeping the worst-quality pixels and discarding the best. Each product's
# category is mapped here to that category's upper-bound error in K (the
# same "assign the category's worst-case value" convention MOD11's mapping
# already used), so `max_lst_error_k` keeps the same meaning for both.
_LST_ERROR_K_BY_BITS = {
    "11A1": {0: 1.0, 1: 2.0, 2: 3.0, 3: float("inf")},
    "21A2": {0: float("inf"), 1: 2.0, 2: 1.5, 3: 1.0},
}
_DEFAULT_LST_ERROR_K_BY_BITS = _LST_ERROR_K_BY_BITS["11A1"]

_QC_LAYOUT_WARNED = False

_QC_LAYOUT_CONFIRMED_PRODUCTS = frozenset(_LST_ERROR_K_BY_BITS)


def decode_qc_valid_mask(
    qc: xr.DataArray,
    max_lst_error_k: float = 2.0,
    product: Optional[str] = None,
    *,
    lst: Optional[xr.DataArray] = None,
    min_lst_k: float = 150.0,
    max_lst_k: float = 350.0,
) -> xr.DataArray:
    """Boolean valid-observation mask from a QC_Day/QC_Night band.

    Bit layout confirmed for `product="11A1"` and `product="21A2"` -- see the
    module comment above for the two primary sources and the bit-value
    inversion between them. Requires mandatory QA bits == 00 ("good") and the
    LST error category <= `max_lst_error_k`.

    `lst`, when given, additionally requires the *decoded* (already
    scale/offset-applied, Kelvin) LST value to fall within
    `[min_lst_k, max_lst_k]`. This is a real, observed gap in the QC bits
    alone, not a hypothetical: a small fraction of raw MODIS pixels (single-
    observation, i.e. one bad granule reading) carry a "good" QC flag
    (mandatory QA 00, low error category) alongside a corrupted encoded
    value that decodes to a physically impossible temperature -- e.g. 744K,
    791K observed in tile h09v02/2002. `composite_to_annual`'s averaging
    dilutes but doesn't remove one of these from an annual composite still
    diluted from a single bad monthly reading. The default bounds mirror the
    codebase's existing `lst_night` GRID-verification range
    (`ModisSource._discover_prepare`'s `value_range=(150, 350)`), so a pixel
    that would already fail post-hoc verification is excluded from the
    composite in the first place instead of merely being flagged after the
    fact by a coarse, easily-missed sample check. A NaN `lst` value (already
    masked as the asset's fill value) compares False against both bounds, so
    it's excluded here too -- consistent with, not an addition to, the
    existing fill-masking in `ModisSource._load_tile_year`.
    """
    global _QC_LAYOUT_WARNED
    if product not in _QC_LAYOUT_CONFIRMED_PRODUCTS and not _QC_LAYOUT_WARNED:
        logger.warning(
            "decode_qc_valid_mask: QC bit layout is UNVERIFIED from a primary source "
            "for product=%s -- confirmed only for %s (see docs/design/07a-modis-"
            "band-reference.md). Confirm before a production run.",
            product, sorted(_QC_LAYOUT_CONFIRMED_PRODUCTS),
        )
        _QC_LAYOUT_WARNED = True

    error_k_by_bits = _LST_ERROR_K_BY_BITS.get(product, _DEFAULT_LST_ERROR_K_BY_BITS)

    qc_uint = qc.astype("uint8")
    mandatory_qa = qc_uint & 0b00000011
    error_bits = (qc_uint >> 6) & 0b11

    error_k = xr.zeros_like(qc_uint, dtype="float32")
    for bits, k in error_k_by_bits.items():
        error_k = xr.where(error_bits == bits, k, error_k)

    mask = (mandatory_qa == 0) & (error_k <= max_lst_error_k)
    if lst is not None:
        mask = mask & (lst >= min_lst_k) & (lst <= max_lst_k)
    return mask

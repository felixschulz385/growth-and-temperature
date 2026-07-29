"""Canonical EPSG:6933 (EASE-Grid 2.0 Global) GeoBox construction.

This is the additive replacement grid described in docs/design/01-grid.md --
metric, equal-area, clipped to |lat| <= 60 deg -- kept alongside the existing
EPSG:4326 `get_or_create_geobox()` grid in `geobox.py`, not replacing it
(docs/design/05-migration.md §1).
"""

import logging
import pickle
from pathlib import Path

from odc.geo.geobox import GeoBox
from pyproj import Transformer

from .constants import (
    DEFAULT_CANONICAL_CRS,
    DEFAULT_LAT_CLIP_DEG,
    DEFAULT_RESOLUTION_M,
)

logger = logging.getLogger(__name__)


def compute_ease_bbox(
    lat_clip_deg: float = DEFAULT_LAT_CLIP_DEG, crs: str = DEFAULT_CANONICAL_CRS
) -> tuple:
    """Compute the canonical grid's projected bounding box from the installed PROJ database.

    Decision in docs/design/01-grid.md §2: compute the extent programmatically
    at build time (`pyproj.Transformer`) rather than hand-copying a published
    constant -- a direct check against the installed PROJ database found
    NSIDC's published EASE2_G polar extent disagrees with a direct ellipsoidal
    transform by ~0.38% (docs/design/06-open-questions.md item 1). That
    discrepancy is specific to the *polar* (lat=90) extent; it is not blocking
    here since `lat_clip_deg` (60 deg, confirmed) is far from the pole.

    x is the full antimeridian-to-antimeridian width at lat=0 -- under a
    cylindrical projection x does not depend on latitude, so any latitude
    would give the same x extent; lat=0 is used for clarity, not because it
    matters. y is evaluated at `lat_clip_deg`, not 90 deg, since the canonical
    grid is clipped there.
    """
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x_max, _ = transformer.transform(180.0, 0.0)
    _, y_max = transformer.transform(0.0, lat_clip_deg)
    return (-x_max, -y_max, x_max, y_max)


def canonical_ease_geobox(
    resolution_m: float = DEFAULT_RESOLUTION_M,
    lat_clip_deg: float = DEFAULT_LAT_CLIP_DEG,
    crs: str = DEFAULT_CANONICAL_CRS,
) -> GeoBox:
    """Build the canonical EPSG:6933 GeoBox: 1 km resolution, |lat| <= 60 deg clip.

    At the confirmed parameters (1 km, 60 deg clip -- docs/design/00-backbone-
    overview.md), this produces a grid within a pixel of the
    ~34,735 x 12,703 px (~441.2M px) figure worked out in
    docs/design/01-grid.md §2, and tiles into 17 x 7 = 119 tiles at
    `DEFAULT_TILE_SIZE=2048` (docs/design/01-grid.md §5) -- both cross-checked
    in tests/data/common/geobox/test_canonical.py.
    """
    bbox = compute_ease_bbox(lat_clip_deg, crs)
    return GeoBox.from_bbox(bbox, crs=crs, resolution=resolution_m)


def get_or_create_canonical_geobox(
    cache_path,
    resolution_m: float = DEFAULT_RESOLUTION_M,
    lat_clip_deg: float = DEFAULT_LAT_CLIP_DEG,
    crs: str = DEFAULT_CANONICAL_CRS,
    force_regenerate: bool = False,
) -> GeoBox:
    """Load a cached canonical GeoBox if present, else build and cache it.

    Mirrors the caching pattern of `get_or_create_geobox` (the existing
    EPSG:4326/VIIRS-derived grid) in `src/data/common/geobox/geobox.py` --
    cheap to build (a handful of PROJ transforms), but cached anyway so every
    consumer of the canonical grid is guaranteed byte-identical shape/affine
    without re-deriving it, and so the exact extent used for a given run is
    recorded on disk rather than implicitly recomputed from
    whatever PROJ/pyproj happens to be installed at read time.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_regenerate:
        logger.info("Loading canonical EASE geobox from %s", cache_path)
        with open(cache_path, "rb") as f:
            geobox = pickle.load(f)
        if not isinstance(geobox, GeoBox):
            raise TypeError(f"{cache_path} does not contain a GeoBox")
        return geobox

    logger.info(
        "Building canonical EASE geobox: crs=%s resolution=%sm lat_clip=%s deg",
        crs,
        resolution_m,
        lat_clip_deg,
    )
    geobox = canonical_ease_geobox(resolution_m, lat_clip_deg, crs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(geobox, f)
    logger.info("Saved canonical EASE geobox to %s (shape=%s)", cache_path, geobox.shape)
    return geobox

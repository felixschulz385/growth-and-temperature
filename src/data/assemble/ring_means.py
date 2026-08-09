"""Ring means from disc sum/count: the tabularization handoff interface.

docs/design/04-ingest.md §6: tabularize exactly once, at the very end of the
raster pipeline, after every raster-space geometry operation (reprojection,
temporal compositing, disc convolution) is complete, and only for cells that
will actually enter estimation. Ring means are a division of differences of
S_d/N_d, computed here at read time -- never stored directly
(docs/design/02-storage.md §4).

Per the backbone design's hard constraint, this module specifies only the
handoff interface consumed by `src/analysis/` (pixel_id + year + country id +
ring-mean columns with S_d/N_d provenance); pixel-FE/country-year-FE
absorption is an estimation-time concern, out of scope here
(docs/design/04-ingest.md §6, docs/design/05-migration.md §2 on
`demean_columns`).
"""

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.xr import xr_coords

from src.data.assemble.constants import PIXEL_ID_IX_SHIFT, PIXEL_ID_IY_SHIFT

logger = logging.getLogger(__name__)


def make_canonical_pixel_ids(ix: int, iy: int, tile_geobox) -> xr.DataArray:
    """Generic pixel_id construction for any canonical GeoBox.

    Same `[ix:16 | iy:16 | local_pixel:32]` bit layout as
    `src.data.assemble.utils.make_pixel_ids`, and decodable by its
    `decode_pixel_id` unchanged (docs/design/01-grid.md §5: the scheme itself
    survives the CRS migration unmodified). Not a drop-in replacement for
    `make_pixel_ids`, though: that function hardcodes `latitude`/`longitude`
    dim names and `EPSG:4326` (`DEFAULT_CRS`) for the existing pipeline, and
    editing it in place would touch the legacy grid this migration must keep
    running untouched (docs/design/05-migration.md §1). This version reads
    dims/coords/CRS directly off whatever `tile_geobox` it's given -- `y`/`x`
    under EPSG:6933, but equally correct for any other GeoBox.
    """
    if ix >= 2**16 or iy >= 2**16:
        raise ValueError(f"Tile indices ({ix}, {iy}) exceed 16-bit range")

    h, w = tile_geobox.shape
    local_pixel_ids = np.arange(h * w, dtype="uint32").reshape((h, w))
    pixel_id_matrix = (
        (np.uint64(ix) << PIXEL_ID_IX_SHIFT)
        | (np.uint64(iy) << PIXEL_ID_IY_SHIFT)
        | local_pixel_ids.astype(np.uint64)
    )
    coords = dict(xr_coords(tile_geobox))
    return xr.DataArray(pixel_id_matrix, dims=tile_geobox.dims, coords=coords, name="pixel_id")


def ring_means_from_discs(S_d: xr.DataArray, N_d: xr.DataArray) -> xr.DataArray:
    """Ring (annulus) means from cumulative disc sums/counts.

    `L^(0) = S_{d_0}/N_{d_0}` (the innermost disc itself); for k >= 1,
    `L^(k) = (S_{d_k} - S_{d_{k-1}}) / (N_{d_k} - N_{d_{k-1}})`
    (docs/design/02-storage.md §4). Rings whose annulus contains no valid
    cells (ring_N == 0, e.g. fully outside the land mask) come out NaN, not
    a divide-by-zero error or a silent zero -- both ring_S and ring_N are 0
    in that annulus by construction (mask-aware convolution;
    docs/design/03-neighbourhood-engine.md §2), so 0/0 is the correct,
    unambiguous "undefined" signal.
    """
    radius_km = list(S_d.coords["radius_km"].values)
    if radius_km != list(N_d.coords["radius_km"].values):
        raise ValueError("S_d/N_d radius_km ladders do not match")
    if S_d.shape != N_d.shape:
        raise ValueError(f"S_d/N_d shape mismatch: {S_d.shape} vs {N_d.shape}")
    # `ring_S[1:] = S[1:] - S[:-1]` below assumes the ladder is ascending --
    # nothing upstream (convolve_discs/write_disc_tile) sorts or validates
    # it, so an out-of-order radius_km list would silently produce
    # nonsensical/negative annulus sums with no error.
    if radius_km != sorted(radius_km):
        raise ValueError(f"radius_km must be sorted ascending, got {radius_km}")

    S = np.asarray(S_d.values)
    N = np.asarray(N_d.values)
    ring_S = np.empty_like(S)
    ring_N = np.empty_like(N)
    ring_S[0] = S[0]
    ring_N[0] = N[0]
    if S.shape[0] > 1:
        ring_S[1:] = S[1:] - S[:-1]
        ring_N[1:] = N[1:] - N[:-1]

    with np.errstate(invalid="ignore", divide="ignore"):
        ring_mean = ring_S / ring_N

    return xr.DataArray(ring_mean, dims=S_d.dims, coords=S_d.coords, name="ring_mean")


def tabularize_tile(
    S_d: xr.DataArray,
    N_d: xr.DataArray,
    mask: xr.DataArray,
    tile_geobox,
    ix: int,
    iy: int,
    year: int,
    country_raster: Optional[xr.DataArray] = None,
) -> pd.DataFrame:
    """One tile's worth of the handoff interface: convolved rasters -> rows.

    Rows are emitted only for cells where `mask` is True (docs/design/04-
    ingest.md §6: only cells that will actually enter estimation) -- the
    land/validity mask and the |phi|<=60 clip are both expected to already be
    folded into `mask` and the extent of `tile_geobox` respectively by the
    caller, not re-applied here.

    Columns: `pixel_id`, `year`, optionally `country_id`, then per ladder
    radius `r` (in km): `L_<r>km` (ring mean), `S_<r>km`/`N_<r>km` (disc sum/
    count provenance, for weighting/diagnostics -- docs/design/04-ingest.md
    §6). Thinning, grid-shake, and FE absorption are explicitly out of scope
    here; see module docstring.
    """
    if S_d.shape != N_d.shape or S_d.shape[1:] != mask.shape:
        raise ValueError(
            f"S_d/N_d/mask shapes must agree: S_d={S_d.shape}, N_d={N_d.shape}, mask={mask.shape}"
        )

    ladder_km: Sequence = list(S_d.coords["radius_km"].values)
    ring_means = ring_means_from_discs(S_d, N_d)
    pixel_ids = make_canonical_pixel_ids(ix, iy, tile_geobox)

    mask_flat = np.asarray(mask.values).astype(bool).reshape(-1)
    valid = np.flatnonzero(mask_flat)

    columns = ["pixel_id", "year"]
    if country_raster is not None:
        columns.append("country_id")
    for r in ladder_km:
        columns += [f"L_{r}km", f"S_{r}km", f"N_{r}km"]

    if valid.size == 0:
        return pd.DataFrame(columns=columns)

    data = {
        "pixel_id": pixel_ids.values.reshape(-1)[valid],
        "year": np.full(valid.size, year, dtype="int64"),
    }
    if country_raster is not None:
        data["country_id"] = np.asarray(country_raster.values).reshape(-1)[valid]

    ring_flat = ring_means.values.reshape(len(ladder_km), -1)
    S_flat = np.asarray(S_d.values).reshape(len(ladder_km), -1)
    N_flat = np.asarray(N_d.values).reshape(len(ladder_km), -1)
    for k, r in enumerate(ladder_km):
        data[f"L_{r}km"] = ring_flat[k][valid]
        data[f"S_{r}km"] = S_flat[k][valid]
        data[f"N_{r}km"] = N_flat[k][valid]

    return pd.DataFrame(data, columns=columns)

"""Disc-ladder orchestration: (variable, mask) -> (S_d, N_d).

Public entry point of the neighbourhood engine. See
docs/design/03-neighbourhood-engine.md.
"""

import logging
from typing import Sequence, Tuple

import numpy as np
import xarray as xr
from pyproj import Transformer

from .convolve import convolve_band_aware, padded_tile_geobox, read_padded_tile, trim_halo
from .kernels import EllipticalKernelRegistry

logger = logging.getLogger(__name__)


def row_latitudes_deg(geobox) -> np.ndarray:
    """Latitude of each row's cell-center, via the installed PROJ database.

    Uses an exact ``pyproj`` transform of the row centers rather than a
    hand-derived spherical inverse, consistent with docs/design/01-grid.md
    §2's preference for computing from the installed PROJ database rather
    than a hardcoded/approximated constant.
    """
    transformer = Transformer.from_crs(geobox.crs, "EPSG:4326", always_xy=True)
    y = geobox.coordinates["y"].values
    x = np.zeros_like(y)
    _, lat = transformer.transform(x, y)
    return lat


def convolve_discs(
    variable: xr.DataArray,
    mask: xr.DataArray,
    ladder_km: Sequence[float],
    kernel_registry: EllipticalKernelRegistry,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Core engine function: variable + mask -> disc sum/count at each radius.

    ``variable`` and ``mask`` are same-shape DataArrays on the canonical grid,
    carrying an ``.odc.geobox`` (typically a halo-padded tile read via
    ``read_padded_tile``, but any canonical-grid array works). Returns
    ``S_d``/``N_d`` -- cumulative disc sum and disc valid-count, never ring
    means (docs/design/02-storage.md §4) -- stacked along a new
    ``radius_km`` dimension, same y/x shape and coordinates as the input.
    Halo is not trimmed here; see ``convolve_tile`` for the halo-read +
    convolve + trim orchestration used by a real tile pipeline.

    This is the one code path all four consumers (lights, mediators,
    rasterized mine points, the validity mask itself) are meant to share
    (docs/design/03-neighbourhood-engine.md §4) -- convolving the mask
    against itself, ``convolve_discs(mask, mask, ladder, registry)``, is the
    degenerate case where S_d == N_d, and doubles as this engine's own unit
    test (see tests/data/common/neighbourhood/test_discs.py).
    """
    if variable.shape != mask.shape:
        raise ValueError(f"variable/mask shape mismatch: {variable.shape} vs {mask.shape}")

    lat_deg = row_latitudes_deg(variable.odc.geobox)
    field = np.asarray(variable.values)
    mask_arr = np.asarray(mask.values).astype(bool)

    S_layers, N_layers = [], []
    for radius_km in ladder_km:
        S, N = convolve_band_aware(field, mask_arr, lat_deg, radius_km, kernel_registry)
        S_layers.append(S)
        N_layers.append(N)

    coords = dict(variable.coords)
    dims = ("radius_km",) + variable.dims
    out_coords = {**coords, "radius_km": list(ladder_km)}
    S_d = xr.DataArray(np.stack(S_layers), dims=dims, coords=out_coords, name="S_d")
    N_d = xr.DataArray(np.stack(N_layers), dims=dims, coords=out_coords, name="N_d")
    return S_d, N_d


def convolve_tile(
    source_variable: xr.DataArray,
    source_mask: xr.DataArray,
    tile_geobox,
    r_max_m: float,
    ladder_km: Sequence[float],
    kernel_registry: EllipticalKernelRegistry,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Halo-read, convolve, and trim back to one tile's core region.

    ``source_variable``/``source_mask`` must cover at least ``tile_geobox``
    buffered by ``r_max_m`` -- typically the full canonical-grid array (or a
    Dask-backed view of it), not pre-sliced to the tile's core region. This
    is the per-tile orchestration a SLURM-per-stage tile worker would call
    (docs/design/03-neighbourhood-engine.md §3, docs/design/05-migration.md
    §5 step 4).
    """
    padded_var = read_padded_tile(source_variable, tile_geobox, r_max_m, fill_value=0.0)
    padded_mask = read_padded_tile(source_mask, tile_geobox, r_max_m, fill_value=False).astype(bool)

    S_d, N_d = convolve_discs(padded_var, padded_mask, ladder_km, kernel_registry)

    halo_gbox = padded_tile_geobox(tile_geobox, r_max_m)
    halo_y = (halo_gbox.shape.y - tile_geobox.shape.y) // 2
    halo_x = (halo_gbox.shape.x - tile_geobox.shape.x) // 2

    return trim_halo(S_d, halo_y, halo_x), trim_halo(N_d, halo_y, halo_x)


def apply_own_country_mask(
    mask: xr.DataArray, country_raster: xr.DataArray, own_country_id
) -> xr.DataArray:
    """Combine a validity mask with an own-country restriction.

    Implements the cross-border ring decision (docs/design/03-neighbourhood-
    engine.md §5): pass the result as ``mask`` to ``convolve_discs``/
    ``convolve_tile`` to get the own-country-masked S_d/N_d variant. Applied
    to lights (the regressor) only, not every mediator -- a robustness check
    on whether cross-border neighbour lights still identify the effect, not a
    change to the baseline specification.
    """
    return mask & (country_raster == own_country_id)

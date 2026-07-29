"""FFT convolution and halo-read primitives.

See docs/design/03-neighbourhood-engine.md.
"""

import logging
from typing import Tuple

import numpy as np
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_zeros
from scipy.signal import fftconvolve

logger = logging.getLogger(__name__)


def padded_tile_geobox(tile_geobox: GeoBox, r_max_m: float) -> GeoBox:
    """Tile geobox buffered by the disc ladder's top radius.

    First real use of ``GeoBox.buffered()`` in this codebase, replacing the
    ad hoc fixed-pixel ``tile_geobox.pad(...)`` pattern for this specific
    purpose -- convolution halo, not resampling-edge padding
    (docs/design/01-grid.md §4).
    """
    return tile_geobox.buffered(r_max_m)


def read_padded_tile(
    source: xr.DataArray, tile_geobox: GeoBox, r_max_m: float, fill_value=np.nan
) -> xr.DataArray:
    """Read a halo-padded tile from a canonical-grid source array.

    ``source`` and ``tile_geobox`` must share the same canonical grid
    (resolution, CRS, pixel alignment) -- this is a pixel-aligned slice, never
    a reprojection, since the convolution engine only ever runs on data
    already regridded onto the canonical GeoBox (docs/design/02-storage.md
    §1). Cells of the padded window that fall outside ``source``'s own extent
    (edge tiles, e.g. near the |phi|<=60 clip or a source's own missing-data
    boundary) are filled with ``fill_value`` rather than raising.
    """
    padded_gbox = padded_tile_geobox(tile_geobox, r_max_m)
    source_gbox = source.odc.geobox

    dst_roi = padded_gbox.overlap_roi(source_gbox)
    src_roi = source_gbox.overlap_roi(padded_gbox)

    out = xr_zeros(padded_gbox, dtype=source.dtype.name)
    out.values[...] = fill_value
    out.values[dst_roi] = np.asarray(source.values)[src_roi]
    return out


def trim_halo(da: xr.DataArray, halo_y: int, halo_x: int) -> xr.DataArray:
    """Trim a convolution result back down from a padded tile to its core region."""
    y_stop = -halo_y if halo_y else None
    x_stop = -halo_x if halo_x else None
    return da.isel(y=slice(halo_y, y_stop), x=slice(halo_x, x_stop))


def mask_aware_fft_convolve(
    field: np.ndarray, mask: np.ndarray, kernel: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convolve ``field`` and ``mask`` with the same kernel in one call.

    This is the single most important correctness property of the engine
    (docs/design/03-neighbourhood-engine.md §2): missing cells must never be
    silently treated as zero without simultaneously tracking how many valid
    cells actually contributed. Convolving the mask alongside the variable,
    and leaving division to read/tabularization time
    (docs/design/02-storage.md §4), makes missing-data handling exact instead
    of an approximation biased near coastlines, cloud gaps, sensor-swath
    edges, and the |phi|<=60 clip edge.

    Returns raw ``(S, N)`` -- disc sum and disc valid-count -- undivided.
    """
    if field.shape != mask.shape:
        raise ValueError(f"field/mask shape mismatch: {field.shape} vs {mask.shape}")
    mask_f = mask.astype(np.float64)
    filled = np.where(mask, field, 0.0).astype(np.float64)
    kernel_f = kernel.astype(np.float64)
    S = fftconvolve(filled, kernel_f, mode="same")
    N = fftconvolve(mask_f, kernel_f, mode="same")
    return S, N


def convolve_band_aware(
    field: np.ndarray,
    mask: np.ndarray,
    row_lat_deg: np.ndarray,
    radius_km: float,
    registry,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mask-aware disc convolution using a per-row (latitude-band) kernel.

    A tile straddling a latitude-band boundary needs a different elliptical
    kernel for different output rows within the same padded input
    (docs/design/01-grid.md §6). This loops over the bands intersecting
    ``row_lat_deg``'s range, convolving the *full* padded input once per band
    and keeping only that band's output rows -- still O(N log N) per
    band-slice, per docs/design/03-neighbourhood-engine.md §3's note that
    this is the piece of the engine most likely to have surprising
    performance characteristics.
    """
    if field.shape[0] != len(row_lat_deg):
        raise ValueError(
            f"row_lat_deg length {len(row_lat_deg)} does not match field's row count {field.shape[0]}"
        )
    band_idx_per_row = registry.band_index_for_lat(row_lat_deg)
    S = np.full(field.shape, np.nan, dtype=np.float64)
    N = np.full(field.shape, np.nan, dtype=np.float64)
    for band_idx in np.unique(band_idx_per_row):
        kernel = registry.kernel_for(int(band_idx), radius_km)
        S_band, N_band = mask_aware_fft_convolve(field, mask, kernel)
        rows = band_idx_per_row == band_idx
        S[rows, :] = S_band[rows, :]
        N[rows, :] = N_band[rows, :]
    return S, N

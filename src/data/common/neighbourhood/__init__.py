"""Neighbourhood (ring/annulus) convolution engine.

Computes disc sum/count fields (``S_d``, ``N_d``) at a ladder of radii via FFT
convolution in raster space, using a latitude-banded elliptical kernel to
correct for EPSG:6933's one-dimensional (latitude-only) anisotropy. This is
new capability -- there is no prior neighbourhood/convolution code elsewhere
in this repository to extend. See docs/design/03-neighbourhood-engine.md.
"""

from .constants import (
    DEFAULT_BAND_TOLERANCE,
    DEFAULT_DISC_LADDER_KM,
    DEFAULT_LAT_CLIP_DEG,
    DEFAULT_R_MAX_KM,
    DEFAULT_STANDARD_PARALLEL_DEG,
)
from .convolve import (
    mask_aware_fft_convolve,
    padded_tile_geobox,
    read_padded_tile,
    trim_halo,
)
from .discs import (
    apply_own_country_mask,
    convolve_discs,
    convolve_tile,
    row_latitudes_deg,
)
from .kernels import (
    EllipticalKernelRegistry,
    anisotropy_scales,
    compute_band_edges,
    disc_kernel_pixels,
    get_or_create_registry,
)
from .store import (
    create_empty_disc_count_store,
    create_empty_disc_sum_store,
    write_disc_tile,
)

__all__ = [
    "create_empty_disc_count_store",
    "create_empty_disc_sum_store",
    "write_disc_tile",
    "DEFAULT_BAND_TOLERANCE",
    "DEFAULT_DISC_LADDER_KM",
    "DEFAULT_LAT_CLIP_DEG",
    "DEFAULT_R_MAX_KM",
    "DEFAULT_STANDARD_PARALLEL_DEG",
    "EllipticalKernelRegistry",
    "anisotropy_scales",
    "apply_own_country_mask",
    "compute_band_edges",
    "convolve_discs",
    "convolve_tile",
    "disc_kernel_pixels",
    "get_or_create_registry",
    "mask_aware_fft_convolve",
    "padded_tile_geobox",
    "read_padded_tile",
    "row_latitudes_deg",
    "trim_halo",
]

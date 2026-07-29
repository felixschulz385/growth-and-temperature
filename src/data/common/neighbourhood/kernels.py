"""Latitude-band elliptical kernel registry.

Under EPSG:6933 (Lambert cylindrical equal-area, standard parallel phi_s), the
projection has zero shear, so its distortion is a closed-form function of
latitude alone. A ground-circle of radius d therefore maps to a pixel-space
*ellipse* whose semi-axes depend only on the latitude band it's centered in --
correctable with one 2-D kernel per (latitude band, radius) pair rather than a
per-pixel or per-tile correction. See docs/design/01-grid.md §6 and
docs/design/03-neighbourhood-engine.md.
"""

import logging
import math
import pickle
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

from .constants import DEFAULT_STANDARD_PARALLEL_DEG

logger = logging.getLogger(__name__)


def anisotropy_scales(
    lat_deg: float, standard_parallel_deg: float = DEFAULT_STANDARD_PARALLEL_DEG
) -> Tuple[float, float]:
    """East-west / north-south linear-scale factors at a given latitude.

    For the Lambert cylindrical equal-area forward projection
    ``x = R*lambda*cos(phi_s)``, ``y = R*sin(phi)/cos(phi_s)``, the ground
    distance represented by one unit of map distance is
    ``g_ew = cos(phi)/cos(phi_s)`` (east-west, along a parallel) and
    ``g_ns = cos(phi_s)/cos(phi)`` (north-south, along a meridian), with
    ``g_ew * g_ns == 1`` (equal area, zero shear -- docs/design/01-grid.md §1).

    A ground-circle of radius ``d`` maps to a pixel-space ellipse with
    semi-axis ``a_ew = d / g_ew = d * scale_ew`` in the map's x-direction and
    ``a_ns = d / g_ns = d * scale_ns`` in the map's y-direction, where

        scale_ew = cos(phi_s) / cos(phi)
        scale_ns = cos(phi) / cos(phi_s)

    This is the exact spherical/authalic form used to define EASE-Grid 2.0;
    it is not a small-angle or first-order approximation of the projection's
    scale factors, only of treating the local Jacobian as constant across one
    kernel's footprint (reasonable at ladder radii <= tens of km).

    Verified numerically in
    ``tests/data/common/neighbourhood/test_kernels.py`` (rasterize a
    known-radius geographic circle at several latitudes, confirm the
    disc-mean recovers the true circle mean) -- resolving
    docs/design/06-open-questions.md item 6, the axis-orientation sign
    convention.
    """
    phi = math.radians(lat_deg)
    phi_s = math.radians(standard_parallel_deg)
    cos_phi = math.cos(phi)
    cos_phi_s = math.cos(phi_s)
    scale_ew = cos_phi_s / cos_phi
    scale_ns = cos_phi / cos_phi_s
    return scale_ew, scale_ns


def compute_band_edges(
    lat_max_deg: float = 60.0,
    tolerance: float = 0.02,
    standard_parallel_deg: float = DEFAULT_STANDARD_PARALLEL_DEG,
    min_step_deg: float = 0.02,
    max_step_deg: float = 5.0,
) -> np.ndarray:
    """Non-uniform latitude band edges from 0 to ``lat_max_deg``.

    Anisotropy ratio ``cos^2(phi_s)/cos^2(phi)`` changes at rate
    ``d(ln ratio)/d(phi) = 2*tan(phi)`` -- ~0 near the equator, largest near
    the clip edge. Step size per band solves ``ln(1+tolerance) = 2*tan(phi)*step``
    (docs/design/01-grid.md §6). Since the step diverges as phi -> 0, it is
    clamped to ``max_step_deg``; ``min_step_deg`` guards against pathologically
    small steps from floating-point error near a band edge.

    Ratio depends on latitude only through ``cos(phi)``, so it is symmetric in
    the sign of phi -- bands are only computed for [0, lat_max_deg] and the
    caller mirrors them for the southern hemisphere via ``abs(lat)``
    (see ``EllipticalKernelRegistry.band_index_for_lat``).
    """
    edges = [0.0]
    phi = 0.0
    while phi < lat_max_deg:
        tan_phi = math.tan(math.radians(max(phi, 1e-6)))
        if tan_phi <= 1e-12:
            step = max_step_deg
        else:
            step = math.degrees(math.log(1.0 + tolerance) / (2.0 * tan_phi))
        step = min(max(step, min_step_deg), max_step_deg)
        phi = min(phi + step, lat_max_deg)
        edges.append(phi)
    return np.array(edges)


def disc_kernel_pixels(
    radius_m: float, resolution_m: float, scale_ew: float = 1.0, scale_ns: float = 1.0
) -> np.ndarray:
    """Boolean elliptical disc kernel in pixel space for a ground-radius disc.

    Semi-axes ``a_ew = radius_m*scale_ew``, ``a_ns = radius_m*scale_ns``
    (see ``anisotropy_scales``), converted to a pixel-index ellipse test at
    ``resolution_m`` per pixel. Kernel is odd-sized in both dimensions so it
    has an unambiguous center pixel for ``scipy.signal.fftconvolve(...,
    mode="same")``.
    """
    a_ew = radius_m * scale_ew
    a_ns = radius_m * scale_ns
    half_x = max(1, math.ceil(a_ew / resolution_m))
    half_y = max(1, math.ceil(a_ns / resolution_m))
    iy, ix = np.mgrid[-half_y : half_y + 1, -half_x : half_x + 1]
    dx = ix * resolution_m
    dy = iy * resolution_m
    return ((dx / a_ew) ** 2 + (dy / a_ns) ** 2) <= 1.0


class EllipticalKernelRegistry:
    """Precomputed disc kernels, one per (latitude band, ladder radius).

    Built once per (canonical grid, disc ladder) pair -- expensive to build,
    depends only on the grid and ladder, never on data -- and intended to be
    cached to disk the same way ``viirs_geobox.pkl`` is in
    ``src/data/common/geobox/geobox.py`` (see ``get_or_create_registry``).
    """

    def __init__(
        self,
        band_edges_deg: Sequence[float],
        radii_km: Sequence[float],
        resolution_m: float,
        standard_parallel_deg: float = DEFAULT_STANDARD_PARALLEL_DEG,
    ):
        self.band_edges_deg = np.asarray(band_edges_deg, dtype=float)
        if self.band_edges_deg.ndim != 1 or len(self.band_edges_deg) < 2:
            raise ValueError("band_edges_deg must have at least 2 entries")
        self.radii_km = list(radii_km)
        self.resolution_m = float(resolution_m)
        self.standard_parallel_deg = float(standard_parallel_deg)
        self._kernels: Dict[Tuple[int, float], np.ndarray] = {}
        self._built = False

    @property
    def n_bands(self) -> int:
        return len(self.band_edges_deg) - 1

    def band_centers_deg(self) -> np.ndarray:
        return (self.band_edges_deg[:-1] + self.band_edges_deg[1:]) / 2.0

    def build(self) -> "EllipticalKernelRegistry":
        centers = self.band_centers_deg()
        for band_idx, lat_center in enumerate(centers):
            scale_ew, scale_ns = anisotropy_scales(lat_center, self.standard_parallel_deg)
            for radius_km in self.radii_km:
                kernel = disc_kernel_pixels(radius_km * 1000.0, self.resolution_m, scale_ew, scale_ns)
                self._kernels[(band_idx, radius_km)] = kernel
        self._built = True
        logger.info(
            "Built elliptical kernel registry: %d bands x %d radii = %d kernels",
            self.n_bands,
            len(self.radii_km),
            len(self._kernels),
        )
        return self

    def band_index_for_lat(self, lat_deg) -> np.ndarray:
        """Map latitude(s) (either hemisphere) to a band index."""
        lat_abs = np.abs(np.asarray(lat_deg, dtype=float))
        idx = np.searchsorted(self.band_edges_deg, lat_abs, side="right") - 1
        return np.clip(idx, 0, self.n_bands - 1)

    def kernel_for(self, band_idx: int, radius_km: float) -> np.ndarray:
        if not self._built:
            raise RuntimeError("EllipticalKernelRegistry.build() has not been called")
        return self._kernels[(int(band_idx), radius_km)]

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> "EllipticalKernelRegistry":
        with open(path, "rb") as f:
            registry = pickle.load(f)
        if not isinstance(registry, cls):
            raise TypeError(f"{path} does not contain an EllipticalKernelRegistry")
        return registry


def get_or_create_registry(
    cache_path,
    band_edges_deg: Sequence[float],
    radii_km: Sequence[float],
    resolution_m: float,
    standard_parallel_deg: float = DEFAULT_STANDARD_PARALLEL_DEG,
    force_regenerate: bool = False,
) -> EllipticalKernelRegistry:
    """Load a cached kernel registry if present, else build and cache it.

    Mirrors the caching pattern of ``get_or_create_geobox`` in
    ``src/data/common/geobox/geobox.py``.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_regenerate:
        logger.info("Loading kernel registry from %s", cache_path)
        return EllipticalKernelRegistry.load(cache_path)

    registry = EllipticalKernelRegistry(band_edges_deg, radii_km, resolution_m, standard_parallel_deg).build()
    registry.save(cache_path)
    logger.info("Saved kernel registry to %s", cache_path)
    return registry

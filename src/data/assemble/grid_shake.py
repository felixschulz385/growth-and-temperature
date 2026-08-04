"""Grid-shake / block-reduce robustness check for downsampling assemble operations.

docs/design/04-ingest.md §6: grid-shake re-runs the coarse-grid reprojection with the
output grid's origin shifted by a fraction of a pixel, to test how sensitive assembled
values are to where the coarse cell boundaries fall relative to the native pixels. Only
meaningful when downsampling (target resolution coarser than the source/native resolution)
-- shifting the origin when upsampling doesn't change which native pixel a target cell maps
to. This module only builds the origin-shifted geoboxes; the actual re-reprojection reuses
the existing `odc.reproject` call in `src.data.assemble.processors.TileProcessor`.
"""

from typing import Any, Dict, List, Tuple, Union

from affine import Affine
from odc.geo.geobox import GeoBox

DEFAULT_GRID_SHAKE_PRESETS: Dict[str, List[Tuple[float, float]]] = {
    "quad": [(0.5, 0.0), (0.0, 0.5), (0.5, 0.5)],
}


def normalize_grid_shake_offsets(
    config: Union[str, Dict[str, Any], List[Any], None],
) -> List[Tuple[str, float, float]]:
    """Normalize a `processing.grid_shake` config value into (label, dx_frac, dy_frac) triples.

    Accepts a preset name (e.g. "quad"), a dict with an "offsets" list, or a raw list of
    (dx, dy) pairs. Each offset is a fraction of one target pixel in [0, 1); label is the
    offset's index as a string, matching the `{column}__shake_{label}` naming used downstream.
    """
    if not config:
        return []

    if isinstance(config, str):
        try:
            offsets = DEFAULT_GRID_SHAKE_PRESETS[config]
        except KeyError as exc:
            available = ", ".join(sorted(DEFAULT_GRID_SHAKE_PRESETS))
            raise ValueError(
                f"Unknown grid_shake preset {config!r}. Available presets: {available}"
            ) from exc
    elif isinstance(config, dict):
        offsets = config.get("offsets")
        if not offsets:
            raise ValueError("'processing.grid_shake' dict form requires a non-empty 'offsets' list")
    elif isinstance(config, list):
        offsets = config
    else:
        raise ValueError(
            f"'processing.grid_shake' must be a preset name, a dict with 'offsets', or a list "
            f"of offsets, got {type(config)}"
        )

    specs: List[Tuple[str, float, float]] = []
    for index, offset in enumerate(offsets):
        if len(offset) != 2:
            raise ValueError(f"grid_shake offset {offset!r} must be a (dx, dy) pair")
        dx, dy = float(offset[0]), float(offset[1])
        if not (0.0 <= dx < 1.0) or not (0.0 <= dy < 1.0):
            raise ValueError(f"grid_shake offset {offset!r} must have components in [0, 1)")
        specs.append((str(index), dx, dy))

    return specs


def shift_geobox_origin(geobox: GeoBox, dx_frac: float, dy_frac: float) -> GeoBox:
    """Return a geobox with identical shape/resolution/crs, origin shifted by a pixel fraction."""
    shifted_affine = geobox.affine * Affine.translation(dx_frac, dy_frac)
    return GeoBox(geobox.shape, shifted_affine, geobox.crs)

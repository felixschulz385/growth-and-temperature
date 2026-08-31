"""Grid-shake / block-reduce robustness check for downsampling assemble operations.

docs/design/04-ingest.md §6: grid-shake re-runs the coarse-grid reprojection with the
output grid's origin shifted by a fraction of a pixel, to test how sensitive assembled
values are to where the coarse cell boundaries fall relative to the native pixels. Only
meaningful when downsampling (target resolution coarser than the source/native resolution)
-- shifting the origin when upsampling doesn't change which native pixel a target cell maps
to. This module only builds the origin-shifted geoboxes; the actual re-reprojection reuses
the existing `odc.reproject` call in `src.data.assemble.processors.TileProcessor`.
"""

import re
from typing import Any, Dict, List, Tuple, Union

from affine import Affine
from odc.geo.geobox import GeoBox

DEFAULT_GRID_SHAKE_PRESETS: Dict[str, List[Tuple[float, float]]] = {
    "quad": [(0.5, 0.0), (0.0, 0.5), (0.5, 0.5)],
}

#: Partition label for the un-shifted table (kept in sync with
#: `src.data.assemble.constants.SHAKE_BASE_LABEL`, duplicated here to keep this
#: module import-light).
SHAKE_BASE_LABEL = "base"


def resolve_shake_selection(
    name: Union[str, None],
) -> List[Tuple[str, float, float]]:
    """Resolve a CLI ``--shake`` value into the ordered list of variants to build.

    Each entry is ``(partition_label, dx_frac, dy_frac)`` -- one full assembly pass
    per entry, writing ``shake=<partition_label>/`` under the grid's output root.

    - ``None`` / ``"none"`` / ``""`` -> just the un-shifted table, ``[("base", 0, 0)]``.
    - a preset name (e.g. ``"quad"``) -> the base table plus one ``s0``/``s1``/...
      partition per offset in the preset.
    - a single ``"s<N>"`` label -> only that one shifted partition (a cheap add-on
      run that leaves ``shake=base`` untouched), where ``N`` indexes the ``"quad"``
      preset's offsets.
    """
    if not name or str(name).lower() == "none":
        return [(SHAKE_BASE_LABEL, 0.0, 0.0)]

    name = str(name)

    if name in DEFAULT_GRID_SHAKE_PRESETS:
        offsets = DEFAULT_GRID_SHAKE_PRESETS[name]
        return [(SHAKE_BASE_LABEL, 0.0, 0.0)] + [
            (f"s{i}", float(dx), float(dy)) for i, (dx, dy) in enumerate(offsets)
        ]

    if re.fullmatch(r"s\d+", name):
        index = int(name[1:])
        preset = DEFAULT_GRID_SHAKE_PRESETS["quad"]
        if index >= len(preset):
            raise ValueError(
                f"grid-shake label {name!r} is out of range for the 'quad' preset "
                f"({len(preset)} offsets: s0..s{len(preset) - 1})"
            )
        dx, dy = preset[index]
        return [(name, float(dx), float(dy))]

    available = ", ".join(sorted(DEFAULT_GRID_SHAKE_PRESETS))
    raise ValueError(
        f"Unknown --shake value {name!r}. Use 'none', a preset ({available}), or an "
        f"'s<N>' offset label."
    )


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

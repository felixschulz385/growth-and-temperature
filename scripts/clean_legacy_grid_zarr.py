"""Find and remove leftover `<family>.zarr` GRID stores left behind by
sources that have since switched to writing `cell_id`-keyed parquet parts
instead (`src.data.common.prepare.driver.run_tiled_prepare`,
`layout.grid_store_path(..., suffix="")` -- see docs/design/02-storage.md
§1's "point 1 is now parquet for every pixel-grid source" addendum).

Every pixel-grid source's *current* GRID output path is
`prepared/<data_path>/crs/<grid_id>/<family><suffix>` (`layout.grid_store_
path()`). Before this migration `suffix` was always `.zarr`; now every
pixel-grid source passes `suffix=""` and writes a directory of `ix=<row>/
iy=<col>/part[-<year>].parquet` files instead. The path prefix and `family`
name are otherwise unchanged, so a pre-migration `<family>.zarr` sibling
sitting next to the new parquet-parts directory (same `data_path`/`grid_id`/
`family`) is pure dead weight -- nothing in the current codebase reads it.

`family` is read directly off each live source instance (`self.variant`/
`self.product`/`self.source_type` where a source's family name depends on
config, e.g. GLASS-MODIS's lst/ta split or MODIS's per-product/variant
suffix) via PIXEL_GRID_FAMILY below, rather than a fully independent
hand-copied table, so it stays anchored to the same logic each source's own
`_output_path()`/`_prepare_output_path()`/`_grid_output_path()` uses.
`modis`/`modis_robustness_11a1` always force `grid_id="ease6933"`
(`MODIS_FORCED_GRID_IDS`, mirroring `migrate_legacy_layout.py`), independent
of `--grid-id`.

Only ever touches a path matching `<family>.zarr` exactly at the current
`crs/<grid_id>/` tier -- never the new parquet-parts directory itself (no
`.zarr` suffix, so it can never collide), and never anything under the
legacy/interim layout eras (`clean_prepare_scratch.py`/
`migrate_legacy_layout.py` already cover those separately).

Dry-run by default -- pass --execute to actually delete.

Usage:
    python scripts/clean_legacy_grid_zarr.py --config orchestration/configs/data.yaml
    python scripts/clean_legacy_grid_zarr.py --config orchestration/configs/data.yaml --execute
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.cli.config import load_config_with_env_vars  # noqa: E402
from src.config.runtime import get_paths_config  # noqa: E402
from src.data.sources import layout  # noqa: E402
from src.data.sources.layout import EASE_GRID_ID, LEGACY_GRID_ID  # noqa: E402
from migrate_legacy_layout import MODIS_FORCED_GRID_IDS, build_source  # noqa: E402

logger = logging.getLogger(__name__)


def _modis_family(source) -> str:
    variant_suffix = "" if source.variant == "main" else "_extended"
    return f"modis_lst_{source.product.lower()}{variant_suffix}"


def _glass_modis_family(source) -> str:
    return f"glass_modis_{source.variant}"


def _eog_family(source) -> str:
    return f"eog_{source.source_type}"


#: (source id) -> callable(live source instance) -> current `family` name,
#: for every source whose GRID output moved from `<family>.zarr` to
#: `<family>/` (cell_id-keyed parquet parts) this migration. Only sources
#: with a real pixel-grid GRID step belong here -- `plad`,
#: `country_classifications`, `commodity_prices` never wrote a pixel-grid
#: zarr at all (small GID/lookup-keyed parquet tables, no GRID step).
PIXEL_GRID_FAMILY = {
    "acag": lambda source: "pm25",
    "esacci": lambda source: "land_cover",
    "ntl_harm": lambda source: "ntl_harm",
    "eog_dmsp": _eog_family,
    "eog_viirs": _eog_family,
    "eog_dvnl": _eog_family,
    "osm": lambda source: "land_mask",
    "gadm": lambda source: "country_id",
    "ecoregions": lambda source: "ecoregions",
    "snl_mining": lambda source: "snl_mining",
    "berman_mining": lambda source: "berman_mining",
    "glass_avhrr": lambda source: "glass_avhrr_lst",
    "modis": _modis_family,
    "modis_robustness_11a1": _modis_family,
    "glass_modis": _glass_modis_family,
    "glass_ta_modis": _glass_modis_family,
}


def dir_size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}PB"


def legacy_zarr_path(source_key: str, source, *, grid_id: str) -> str | None:
    family_fn = PIXEL_GRID_FAMILY.get(source_key)
    if family_fn is None:
        return None
    effective_grid_id = EASE_GRID_ID if source_key in MODIS_FORCED_GRID_IDS else grid_id
    # path_prefix-keyed sources (glass_avhrr/glass_modis, modis) build their
    # own output root off a fixed constant, not cfg.data_path -- same
    # fallback clean_prepare_scratch.py's candidate_prepare_dirs() uses.
    data_path = getattr(source, "path_prefix", source.cfg.data_path)
    family = family_fn(source)
    return layout.grid_store_path(
        source.ctx.data_root, data_path, grid_id=effective_grid_id, family=family, suffix=".zarr"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to the unified pipeline config (data.yaml)")
    parser.add_argument("--data-root", default=None, help="Override paths.data_root from the config")
    parser.add_argument(
        "--grid-id", default=LEGACY_GRID_ID, choices=[LEGACY_GRID_ID, EASE_GRID_ID],
        help="Grid id to check (default: legacy_4326). Ignored for modis/modis_robustness_11a1, which always use ease6933.",
    )
    parser.add_argument("--source", default=None, help="Scan only this source (default: every source in the config)")
    parser.add_argument("--execute", action="store_true", help="Actually delete (default: dry run, logs only)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    config = load_config_with_env_vars(args.config)
    data_root = args.data_root or get_paths_config(config).get("data_root")
    if not data_root:
        raise SystemExit("No data_root found -- pass --data-root or set paths.data_root in the config")

    source_keys = [args.source] if args.source else sorted((config.get("sources") or {}).keys())

    logger.info("Scanning for leftover legacy-format GRID zarr stores, data_root=%s, grid_id=%s, execute=%s", data_root, args.grid_id, args.execute)
    logger.info("Sources: %s", ", ".join(source_keys))

    total_count = 0
    total_bytes = 0
    failed = 0
    for source_key in source_keys:
        if source_key not in PIXEL_GRID_FAMILY:
            logger.debug("[%s] not a pixel-grid source (or not migrated to parquet) -- skipping", source_key)
            continue

        try:
            source = build_source(source_key, config, data_root, args.grid_id)
        except Exception:
            logger.exception("[%s] failed to construct source -- skipping entirely", source_key)
            failed += 1
            continue

        try:
            path = legacy_zarr_path(source_key, source, grid_id=args.grid_id)
        except Exception:
            logger.exception("[%s] failed to resolve legacy zarr path -- skipping", source_key)
            failed += 1
            continue

        if path is None or not os.path.exists(path):
            continue

        size = dir_size(path)
        total_count += 1
        total_bytes += size
        if args.execute:
            logger.info("[%s] deleting %s (%s)", source_key, path, human(size))
            try:
                shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
            except Exception:
                logger.exception("[%s] failed to delete %s", source_key, path)
                failed += 1
        else:
            logger.info("[DRY RUN] [%s] would delete %s (%s)", source_key, path, human(size))

    logger.info("Total: %d entries, %s (failed=%d)", total_count, human(total_bytes), failed)
    if not args.execute:
        logger.info("This was a DRY RUN -- pass --execute to actually delete.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

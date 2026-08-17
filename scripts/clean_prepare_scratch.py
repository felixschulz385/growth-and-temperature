"""Find and remove leftover GLASS-AVHRR per-year "annual stats" scratch
zarrs (`<year>.zarr` / `<year>_monthly.zarr`, e.g. `1992.zarr`) sitting
directly inside a source's PREPARE-stage output directory.

These are same-run-only intermediates -- `_annual_zarr_path()`/
`_ensure_annual_zarr()` (`src/data/sources/glass/avhrr.py`) build one per
year, consumed within the same PREPARE run to produce the final reprojected
GRID-family store, and never read again afterwards by anything else. They
were never meant to be a durable pipeline artifact (flagged as such in
`avhrr.py`'s own comments, pending a future GLASS rework), but real runs
across all three layout eras this repo has had left them on disk:

- legacy:  <data_root>/<path_prefix>/processed/stage_1[/<namespace>]/<year>.zarr
- interim: <data_root>/prepared/<path_prefix>[/<namespace>]/<year>.zarr
- current: <data_root>/prepared/<path_prefix>/crs[/<namespace>]/<year>.zarr

Only ever touches entries directly inside one of those three PREPARE
directories whose name matches `^\\d{4}(_monthly)?\\.zarr$` -- nothing else
in the directory (e.g. the final family.zarr store, if it happens to share
the "current" directory) is ever considered, let alone deleted.

Dry-run by default -- pass --execute to actually delete.

Usage:
    python scripts/clean_prepare_scratch.py --config orchestration/configs/data.yaml
    python scripts/clean_prepare_scratch.py --config orchestration/configs/data.yaml --execute
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.cli.config import load_config_with_env_vars  # noqa: E402
from src.config.runtime import get_paths_config  # noqa: E402
from src.data.sources.layout import LEGACY_GRID_ID  # noqa: E402
from src.data.sources.steps import PipelineStep  # noqa: E402
from migrate_legacy_layout import build_source, interim_output_root, legacy_output_root  # noqa: E402

logger = logging.getLogger(__name__)

#: Only GLASS-AVHRR's `_annual_zarr_path()` ever built PREPARE output as a
#: bare `<year>[.zarr|_monthly.zarr]` file directly in the PREPARE directory
#: -- no other registered source uses this naming, so this is a static,
#: intentionally narrow list rather than something auto-discovered.
SCRATCH_SOURCES = ("glass_avhrr",)

YEAR_ZARR_RE = re.compile(r"^\d{4}(_monthly)?\.zarr$")


def find_scratch_entries(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if YEAR_ZARR_RE.match(name)
    ]


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


def remove_entry(path: str, *, execute: bool) -> None:
    if not execute:
        return
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to the unified pipeline config (data.yaml)")
    parser.add_argument("--data-root", default=None, help="Override paths.data_root from the config")
    parser.add_argument("--grid-id", default=LEGACY_GRID_ID, help="Grid id used to construct sources (does not affect PREPARE paths, kept for symmetry with migrate_legacy_layout.py)")
    parser.add_argument("--execute", action="store_true", help="Actually delete (default: dry run, logs only)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    config = load_config_with_env_vars(args.config)
    data_root = args.data_root or get_paths_config(config).get("data_root")
    if not data_root:
        raise SystemExit("No data_root found -- pass --data-root or set paths.data_root in the config")

    logger.info("Scanning for PREPARE-stage year-zarr scratch, data_root=%s, execute=%s", data_root, args.execute)

    total_count = 0
    total_bytes = 0
    for source_key in SCRATCH_SOURCES:
        try:
            source = build_source(source_key, config, data_root, args.grid_id)
        except Exception:
            logger.exception("[%s] failed to construct source -- skipping entirely", source_key)
            continue

        path_prefix = getattr(source, "path_prefix", source.cfg.data_path)
        candidates = {
            "legacy": legacy_output_root(data_root, path_prefix, PipelineStep.PREPARE, namespace=source.cfg.namespace),
            "interim": interim_output_root(data_root, path_prefix, PipelineStep.PREPARE, namespace=source.cfg.namespace),
            "current": source.output_root(PipelineStep.PREPARE),
        }

        for layout_name, directory in candidates.items():
            entries = find_scratch_entries(directory)
            for entry in entries:
                size = dir_size(entry)
                total_count += 1
                total_bytes += size
                label = f"{source_key}/{layout_name}"
                if args.execute:
                    logger.info("[%s] deleting %s (%s)", label, entry, human(size))
                    try:
                        remove_entry(entry, execute=True)
                    except Exception:
                        logger.exception("[%s] failed to delete %s", label, entry)
                else:
                    logger.info("[DRY RUN] [%s] would delete %s (%s)", label, entry, human(size))

    logger.info("Total: %d entries, %s", total_count, human(total_bytes))
    if not args.execute:
        logger.info("This was a DRY RUN -- pass --execute to actually delete.")


if __name__ == "__main__":
    main()

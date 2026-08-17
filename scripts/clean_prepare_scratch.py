"""Find and remove leftover per-year "annual zarr" scratch (`<year>.zarr` /
`<year>_monthly.zarr`, e.g. `2000.zarr`) sitting directly inside a source's
PREPARE-stage output directory.

Two distinct origins, both safe to delete:

- **Dead leftovers (almost every source)**: before commit `2d3bf1a`
  ("Merge the PREPARE and GRID pipeline steps into one ledger-free PREPARE
  step across every remaining source"), each source's old PREPARE step wrote
  one bare `<year>.zarr` per year (e.g. `acag.py`'s old `_write_annual_zarr`/
  `_list_annual_zarrs`, parsing filenames via `int(os.path.splitext(fname)[0])`
  -- any all-digit basename), later reprojected by a separate GRID step into
  one combined timeseries store. `2d3bf1a` rewrote PREPARE to reproject
  straight into the shared store via `run_tiled_prepare()`
  (`src/data/common/prepare/driver.py:69`), so nothing in the current
  codebase writes or reads a bare `<year>.zarr` anymore for these sources --
  any that still exist on disk are pure dead weight from the old pipeline
  generation.

- **Live-but-disposable scratch (glass_avhrr only)**: `_annual_zarr_path()`/
  `_ensure_annual_zarr()` (`src/data/sources/glass/avhrr.py`) still build one
  `<year>.zarr` (or `<year>_monthly.zarr`) per year *today*, consumed within
  the same PREPARE run to produce the final reprojected GRID-family store
  and also used as a same-run resumability marker. Deleting these is safe at
  any time -- worst case, a future PREPARE run recomputes that year's stats
  instead of reusing the cached intermediate -- but note it's a real,
  currently-used mechanism, not dead code (flagged for a future GLASS
  rework, see `avhrr.py`'s own comments).

Every configured source is scanned, across all three layout eras this repo
has had:

    legacy:  <data_root>/<path_prefix>/processed/stage_1[/<namespace>]/<year>.zarr
    interim: <data_root>/prepared/<path_prefix>[/<namespace>]/<year>.zarr
    current: <data_root>/prepared/<path_prefix>/<agg>[/<grid_id>]/<year>.zarr

Only ever touches entries directly inside one of those PREPARE directories
whose name matches `^\\d{4}(_monthly)?\\.zarr$` -- nothing else in the
directory (e.g. a real family.zarr store, if it happens to share the
"current" directory) is ever considered, let alone deleted. A source whose
"current" PREPARE root can't be resolved generically (e.g. it requires an
`agg=` this script doesn't know, or has no PREPARE step at all) is skipped
for that one candidate rather than aborting the whole scan.

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
from src.data.sources import layout  # noqa: E402
from src.data.sources.layout import LEGACY_GRID_ID  # noqa: E402
from src.data.sources.steps import PipelineStep  # noqa: E402
from migrate_legacy_layout import (  # noqa: E402
    PREPARE_AGG_OVERRIDES,
    PREPARE_EXCEPTIONS,
    build_source,
    interim_output_root,
    legacy_output_root,
)

logger = logging.getLogger(__name__)

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


def remove_entry(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def candidate_prepare_dirs(source_key: str, source, data_root: str) -> dict[str, str]:
    # path_prefix-keyed sources (glass_avhrr/glass_modis, modis) build their
    # own output root off a fixed constant, not cfg.data_path -- same
    # fallback migrate_legacy_layout.py's per-source special-casing relies
    # on implicitly via source.output_root() for "current".
    path_prefix = getattr(source, "path_prefix", source.cfg.data_path)
    candidates = {
        "legacy": legacy_output_root(data_root, path_prefix, PipelineStep.PREPARE, namespace=source.cfg.namespace),
        "interim": interim_output_root(data_root, path_prefix, PipelineStep.PREPARE, namespace=source.cfg.namespace),
    }
    if source_key not in PREPARE_EXCEPTIONS:
        try:
            agg = PREPARE_AGG_OVERRIDES.get(source_key, layout.CRS_AGG)
            candidates["current"] = source.output_root(PipelineStep.PREPARE, agg=agg)
        except Exception:
            logger.debug("[%s] could not resolve current PREPARE root -- skipping that candidate", source_key, exc_info=True)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to the unified pipeline config (data.yaml)")
    parser.add_argument("--data-root", default=None, help="Override paths.data_root from the config")
    parser.add_argument("--grid-id", default=LEGACY_GRID_ID, help="Grid id used to construct sources (does not affect PREPARE paths, kept for symmetry with migrate_legacy_layout.py)")
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

    logger.info("Scanning for PREPARE-stage year-zarr scratch, data_root=%s, execute=%s", data_root, args.execute)
    logger.info("Sources: %s", ", ".join(source_keys))

    total_count = 0
    total_bytes = 0
    failed = 0
    for source_key in source_keys:
        try:
            source = build_source(source_key, config, data_root, args.grid_id)
        except Exception:
            logger.exception("[%s] failed to construct source -- skipping entirely", source_key)
            failed += 1
            continue

        for layout_name, directory in candidate_prepare_dirs(source_key, source, data_root).items():
            entries = find_scratch_entries(directory)
            for entry in entries:
                size = dir_size(entry)
                total_count += 1
                total_bytes += size
                label = f"{source_key}/{layout_name}"
                if args.execute:
                    logger.info("[%s] deleting %s (%s)", label, entry, human(size))
                    try:
                        remove_entry(entry)
                    except Exception:
                        logger.exception("[%s] failed to delete %s", label, entry)
                        failed += 1
                else:
                    logger.info("[DRY RUN] [%s] would delete %s (%s)", label, entry, human(size))

    logger.info("Total: %d entries, %s (failed=%d)", total_count, human(total_bytes), failed)
    if not args.execute:
        logger.info("This was a DRY RUN -- pass --execute to actually delete.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Rename columns in already-materialized tile-partitioned parquet, in place.

Two source columns were renamed in code for clarity (see git history around
2026-09-07):
  * MODIS `valid_period_count_annual`/`valid_month_count_annual` (night LST)
    -> `valid_period_count_night_annual`/`valid_month_count_night_annual`,
    to mirror the existing `_day_` suffix on the paired day-LST columns
    (`src/data/sources/modis/source.py`).
  * EOG VIIRS `viirs_annual` (the masked-mean-radiance variant)
    -> `viirs_annual_avg`, to make explicit it is the mean, parallel to the
    `_median`/`_cf_cvg` siblings it's always written alongside
    (`src/data/sources/eog/source.py`).

Renaming those in code only changes what a *future* PREPARE/ASSEMBLE run
writes -- it does nothing to parquet already sitting on disk from before the
change. This script patches that existing parquet directly (read with
pandas, rename columns, write back), without re-running PREPARE or ASSEMBLE.

Targets (found automatically under `--data-root`, skipped if absent):
  * `prepared/modis/21A2/crs/ease6933/modis_lst_21a2/ix=*/iy=*/part-*.parquet`
  * `prepared/eog/viirs/crs/ease6933/eog_viirs_annual/ix=*/iy=*/part-*.parquet`
  * every already-assembled `assembled/grid=*/shake=*/ix=*/iy=*/data_*.parquet`
    (gets the union of both rename maps, since either source's columns may
    be joined into it)

Each of those trees is `ix=<row>/iy=<col>/`-tile-partitioned, so the unit of
work here is one tile: every `*.parquet` file directly inside one `ix=/iy=`
directory. `--task-index` selects one tile out of the full, deterministically
-ordered manifest (stable across repeated invocations as long as the set of
tiles on disk hasn't changed) -- this is what
`orchestration/slurm/rename_processed_columns.sbatch` drives via
`$SLURM_ARRAY_TASK_ID`, one SLURM array task per tile.

A file already carrying the new names (no old name present) is left alone,
so this is safe to re-run / re-submit over partially-completed work.

Dry-run by default -- pass --execute to actually rewrite files.

Usage:
    # see how many tiles there are (sizes the SLURM array)
    python scripts/rename_processed_columns.py --data-root /path/to/data_nobackup --print-manifest | wc -l

    # do one tile (what the SLURM array actually calls)
    python scripts/rename_processed_columns.py --data-root /path/to/data_nobackup --task-index 0 --execute

    # do every tile serially -- review the dry-run log, THEN add --execute
    python scripts/rename_processed_columns.py --data-root /path/to/data_nobackup --all
    python scripts/rename_processed_columns.py --data-root /path/to/data_nobackup --all --execute
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

MODIS_RENAME: Dict[str, str] = {
    "valid_period_count_annual": "valid_period_count_night_annual",
    "valid_month_count_annual": "valid_month_count_night_annual",
}
VIIRS_RENAME: Dict[str, str] = {
    "viirs_annual": "viirs_annual_avg",
}
COMBINED_RENAME: Dict[str, str] = {**MODIS_RENAME, **VIIRS_RENAME}


@dataclass(frozen=True)
class Target:
    label: str
    root: Path
    rename: Dict[str, str]


def discover_targets(data_root: Path) -> List[Target]:
    """The PREPARE-stage family roots plus every already-assembled
    grid/shake tree, each paired with the rename map applicable to it.
    Silently skips any root that doesn't exist (e.g. a grid/shake
    combination never assembled)."""
    targets: List[Target] = []

    modis_root = data_root / "prepared" / "modis" / "21A2" / "crs" / "ease6933" / "modis_lst_21a2"
    if modis_root.is_dir():
        targets.append(Target("modis", modis_root, MODIS_RENAME))

    viirs_root = data_root / "prepared" / "eog" / "viirs" / "crs" / "ease6933" / "eog_viirs_annual"
    if viirs_root.is_dir():
        targets.append(Target("eog_viirs", viirs_root, VIIRS_RENAME))

    assembled_root = data_root / "assembled"
    for grid_dir in sorted(assembled_root.glob("grid=*")):
        for shake_dir in sorted(grid_dir.glob("shake=*")):
            if shake_dir.is_dir():
                label = f"assembled/{grid_dir.name}/{shake_dir.name}"
                targets.append(Target(label, shake_dir, COMBINED_RENAME))

    return targets


def discover_tiles(root: Path) -> List[Tuple[int, int]]:
    """Sorted `(ix, iy)` pairs for every `ix=*/iy=*` tile directory under
    *root* -- sorted so the manifest built from these is stable across
    repeated invocations (required for `--task-index` to keep meaning the
    same tile across the lifetime of one SLURM array submission)."""
    tiles = []
    for ix_dir in root.glob("ix=*"):
        ix = int(ix_dir.name.split("=", 1)[1])
        for iy_dir in ix_dir.glob("iy=*"):
            if iy_dir.is_dir():
                iy = int(iy_dir.name.split("=", 1)[1])
                tiles.append((ix, iy))
    return sorted(tiles)


@dataclass(frozen=True)
class ManifestEntry:
    label: str
    tile_dir: Path
    rename: Dict[str, str]
    ix: int
    iy: int


def build_manifest(data_root: Path) -> List[ManifestEntry]:
    manifest: List[ManifestEntry] = []
    for target in discover_targets(data_root):
        for ix, iy in discover_tiles(target.root):
            tile_dir = target.root / f"ix={ix}" / f"iy={iy}"
            manifest.append(ManifestEntry(target.label, tile_dir, target.rename, ix, iy))
    return manifest


def rename_file_columns(path: Path, rename: Dict[str, str], *, dry_run: bool) -> str:
    """Rename whichever of *rename*'s keys are actually present as columns
    in the parquet file at *path*, writing it back in place. Returns a short
    status string for logging. No-op (returns without touching the file) if
    none of the old names are present -- makes this safe to re-run."""
    pf = pq.ParquetFile(path)
    existing = set(pf.schema_arrow.names)
    applicable = {old: new for old, new in rename.items() if old in existing}
    if not applicable:
        return "skip (no matching columns)"

    if dry_run:
        return f"would rename {applicable}"

    compression = "snappy"
    if pf.metadata.num_row_groups and pf.metadata.row_group(0).num_columns:
        codec = pf.metadata.row_group(0).column(0).compression
        if codec:
            compression = codec.lower()

    df = pd.read_parquet(path)
    df = df.rename(columns=applicable)

    tmp_path = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        df.to_parquet(tmp_path, index=False, compression=compression)
        os.replace(tmp_path, path)  # atomic -- never leaves a half-written file at `path`
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return f"renamed {applicable}"


def process_tile(entry: ManifestEntry, *, dry_run: bool) -> bool:
    """Processes every `*.parquet` file directly inside one tile directory.
    Returns True iff every file in the tile was handled without error (a
    missing tile directory -- already moved/cleaned up by something else --
    is logged and treated as success, not a failure, since there's nothing
    left to rename)."""
    if not entry.tile_dir.is_dir():
        logger.warning("[%s] ix=%d iy=%d: tile directory missing, skipping", entry.label, entry.ix, entry.iy)
        return True

    files = sorted(entry.tile_dir.glob("*.parquet"))
    if not files:
        logger.info("[%s] ix=%d iy=%d: no parquet files", entry.label, entry.ix, entry.iy)
        return True

    ok = True
    for path in files:
        try:
            status = rename_file_columns(path, entry.rename, dry_run=dry_run)
            logger.info("[%s] ix=%d iy=%d: %s -- %s", entry.label, entry.ix, entry.iy, path.name, status)
        except Exception:
            logger.exception("[%s] ix=%d iy=%d: %s -- FAILED", entry.label, entry.ix, entry.iy, path.name)
            ok = False
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, type=Path, help="e.g. .../growth-and-temperature/data_nobackup")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-manifest", action="store_true", help="list tiles (label, ix, iy) and exit -- pipe to `wc -l` to size the SLURM array")
    mode.add_argument("--task-index", type=int, help="process manifest[N] only -- what the SLURM array passes")
    mode.add_argument("--all", action="store_true", help="process every tile serially, in this process (no SLURM)")
    parser.add_argument(
        "--execute", action="store_true",
        help="actually rewrite files -- default is dry-run (log what would be renamed, touch nothing)",
    )
    args = parser.parse_args()
    dry_run = not args.execute

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    manifest = build_manifest(args.data_root)
    if not manifest:
        logger.error("No targets found under --data-root=%s (expected prepared/ and/or assembled/ subtrees)", args.data_root)
        sys.exit(1)

    if args.print_manifest:
        for i, entry in enumerate(manifest):
            print(f"{i}\t{entry.label}\tix={entry.ix}\tiy={entry.iy}")
        return

    if args.task_index is not None:
        if not 0 <= args.task_index < len(manifest):
            logger.error("--task-index %d out of range (manifest has %d tiles)", args.task_index, len(manifest))
            sys.exit(1)
        ok = process_tile(manifest[args.task_index], dry_run=dry_run)
        sys.exit(0 if ok else 1)

    # --all
    failures = 0
    for entry in manifest:
        if not process_tile(entry, dry_run=dry_run):
            failures += 1
    logger.info("Done: %d/%d tiles had a failure", failures, len(manifest))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

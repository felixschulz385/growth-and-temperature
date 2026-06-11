#!/usr/bin/env python3
"""Move CSV-only duckreg subdirectories into a cleanup area."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gnt.analysis.core.config import RESULTS_DIR  # noqa: E402


def _iter_candidate_dirs(duckreg_root: Path, cleanup_root: Path) -> List[Path]:
    """Return top-level CSV-only subdirectories under ``duckreg_root``."""
    candidates: List[Path] = []

    for directory in sorted(path for path in duckreg_root.rglob("*") if path.is_dir()):
        if directory == cleanup_root or cleanup_root in directory.parents:
            continue

        files = [path for path in directory.rglob("*") if path.is_file()]
        if not files:
            continue
        if any(path.suffix.lower() != ".csv" for path in files):
            continue
        if any(parent in candidates for parent in directory.parents):
            continue

        candidates.append(directory)

    return candidates


def _collect_conflicts(
    candidates: Iterable[Path],
    duckreg_root: Path,
    cleanup_root: Path,
) -> List[tuple[Path, Path]]:
    conflicts: List[tuple[Path, Path]] = []
    for source_dir in candidates:
        target_dir = cleanup_root / source_dir.relative_to(duckreg_root)
        if target_dir.exists():
            conflicts.append((source_dir, target_dir))
    return conflicts


def move_stale_csv_dirs(duckreg_root: Path, apply: bool = False) -> int:
    if not duckreg_root.exists():
        raise FileNotFoundError(f"Duckreg directory not found: {duckreg_root}")

    cleanup_root = duckreg_root / "cleanup"
    candidates = _iter_candidate_dirs(duckreg_root, cleanup_root)
    conflicts = _collect_conflicts(candidates, duckreg_root, cleanup_root)

    for source_dir in candidates:
        target_dir = cleanup_root / source_dir.relative_to(duckreg_root)
        action = "MOVE" if apply and not any(source_dir == src for src, _ in conflicts) else "PLAN"
        print(f"{action} {source_dir} -> {target_dir}")

    if conflicts:
        print("\nConflicts:")
        for source_dir, target_dir in conflicts:
            print(f"  {source_dir}")
            print(f"    exists: {target_dir}")
        if apply:
            return 1

    if not apply:
        print(f"\nDry run complete: {len(candidates)} CSV-only directorie(s) identified.")
        return 0 if not conflicts else 1

    moved = 0
    for source_dir in candidates:
        if any(source_dir == src for src, _ in conflicts):
            continue
        target_dir = cleanup_root / source_dir.relative_to(duckreg_root)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_dir), str(target_dir))
        moved += 1

    print(f"\nMove complete: moved {moved} CSV-only directorie(s).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move duckreg subdirectories containing only CSV files into duckreg/cleanup."
    )
    parser.add_argument(
        "--duckreg-root",
        default=str(RESULTS_DIR / "duckreg"),
        help="Duckreg results root (default: output/analysis/duckreg)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply filesystem moves. Default is dry-run.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return move_stale_csv_dirs(
        Path(args.duckreg_root),
        apply=args.apply,
    )


if __name__ == "__main__":
    raise SystemExit(main())

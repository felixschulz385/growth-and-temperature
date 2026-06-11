#!/usr/bin/env python3
"""Append coarse-grid pixel ID columns to assembled parquet partitions.

This script is intended for existing grid-partitioned assembly outputs such as
``data_nobackup/assembled/500m.parquet``. It reads each ``ix=*/iy=*/data.parquet``
partition, derives alternate grid IDs from the canonical ``pixel_id`` column,
and writes the updated partition back in place.

Default output columns use user-facing metric labels:
    - ``pixel_id_1km``
    - ``pixel_id_5km``

The canonical ``pixel_id`` column is left unchanged.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gnt.data.assemble.tiles import create_tile_geobox
from gnt.data.assemble.utils import (
    DEFAULT_DERIVED_PIXEL_ID_RESOLUTIONS,
    add_derived_pixel_id_columns,
)
from gnt.data.common.geobox.geobox import get_or_create_geobox


LOGGER = logging.getLogger("backfill_grid_ids")


PIXEL_ID_COL = "pixel_id"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Partitioned assembled parquet root, e.g. data_nobackup/assembled/500m.parquet",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Project data root used to load the canonical VIIRS geobox cache",
    )
    parser.add_argument(
        "--source-grid",
        default="500m",
        help="Source grid label used for the existing pixel_id column (default: 500m)",
    )
    parser.add_argument(
        "--target-grid",
        action="append",
        default=[],
        help=(
            "Target grid label to add, e.g. --target-grid 1km --target-grid 5km. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--grid-resolution",
        action="append",
        default=[],
        metavar="LABEL=RESOLUTION",
        help=(
            "Override or add grid resolutions in degrees, e.g. "
            "--grid-resolution 10km=0.1"
        ),
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=2048,
        help="Assembly tile size used by the dataset (default: 2048)",
    )
    parser.add_argument(
        "--column-prefix",
        default="pixel_id_",
        help="Prefix for derived ID columns (default: pixel_id_)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite derived columns if they already exist",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N partitions",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect partitions and planned columns without writing changes",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def parse_grid_resolutions(overrides: Iterable[str]) -> Dict[str, float]:
    resolutions = dict(DEFAULT_DERIVED_PIXEL_ID_RESOLUTIONS)
    for item in overrides:
        if "=" not in item:
            raise ValueError(
                f"Invalid --grid-resolution value {item!r}; expected LABEL=RESOLUTION"
            )
        label, raw_resolution = item.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Invalid empty grid label in {item!r}")
        resolutions[label] = float(raw_resolution)
    return resolutions


def normalized_resolution(label: str, resolutions: Dict[str, float]) -> float:
    try:
        return resolutions[label]
    except KeyError as exc:
        available = ", ".join(sorted(resolutions))
        raise KeyError(
            f"Unknown grid label {label!r}. Available labels: {available}"
        ) from exc


def iter_partition_files(dataset_root: Path) -> Iterable[Tuple[int, int, Path]]:
    pattern = re.compile(r"ix=(?P<ix>\d+)$")
    for path in sorted(dataset_root.glob("ix=*/iy=*/data.parquet")):
        ix_match = pattern.match(path.parent.parent.name)
        if ix_match is None:
            continue
        iy_name = path.parent.name
        if not iy_name.startswith("iy="):
            continue
        ix = int(ix_match.group("ix"))
        iy = int(iy_name.split("=", 1)[1])
        yield ix, iy, path


def update_partition(
    path: Path,
    ix: int,
    iy: int,
    base_tile_geobox,
    source_resolution: float,
    derived_specs: List[Tuple[str, float]],
    overwrite: bool,
    dry_run: bool,
) -> Tuple[int, int]:
    df = pd.read_parquet(path)
    if PIXEL_ID_COL not in df.columns:
        raise ValueError(f"Partition {path} does not contain column {PIXEL_ID_COL!r}")
    original_columns = list(df.columns)

    target_labels = [name for name, _ in derived_specs]
    added = 0
    replaced = 0
    specs_to_write: List[Tuple[str, float]] = []
    for label in target_labels:
        column = label
        resolution = next(res for name, res in derived_specs if name == label)
        if column in df.columns:
            if not overwrite:
                LOGGER.info("Skipping existing column %s in %s", column, path)
                continue
            replaced += 1
        else:
            added += 1
        df = df.drop(columns=[column], errors="ignore")
        specs_to_write.append((column, resolution))

    if not specs_to_write:
        return 0, 0

    source_geobox = base_tile_geobox.zoom_to(resolution=source_resolution)
    df = add_derived_pixel_id_columns(
        df=df,
        ix=ix,
        iy=iy,
        base_tile_geobox=base_tile_geobox,
        source_geobox=source_geobox,
        derived_specs=specs_to_write,
    )
    df = reorder_columns(df, original_columns, [name for name, _ in derived_specs])

    if dry_run:
        LOGGER.info(
            "Dry-run: would write %s with %d added and %d replaced columns",
            path,
            added,
            replaced,
        )
        return added, replaced

    tmp_path = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp_path, index=False, compression="snappy", engine="pyarrow")
    os.replace(tmp_path, path)
    LOGGER.info(
        "Updated %s (ix=%d, iy=%d): added=%d replaced=%d",
        path,
        ix,
        iy,
        added,
        replaced,
    )
    return added, replaced


def reorder_columns(
    df: pd.DataFrame,
    original_columns: List[str],
    derived_columns: List[str],
) -> pd.DataFrame:
    """Match assembly ordering: index columns, derived IDs, then remaining columns."""
    ordered: List[str] = []

    for column in original_columns:
        if column in df.columns and column not in derived_columns and column not in ordered:
            ordered.append(column)
        if column == PIXEL_ID_COL:
            for derived in derived_columns:
                if derived in df.columns and derived not in ordered:
                    ordered.append(derived)

    for derived in derived_columns:
        if derived in df.columns and derived not in ordered:
            ordered.append(derived)

    for column in df.columns:
        if column not in ordered:
            ordered.append(column)

    return df[ordered]


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)

    dataset_root = args.dataset_root.resolve()
    data_root = args.data_root.resolve()
    if not dataset_root.exists():
        parser.error(f"Dataset root does not exist: {dataset_root}")
    if not data_root.exists():
        parser.error(f"Data root does not exist: {data_root}")

    resolutions = parse_grid_resolutions(args.grid_resolution)
    source_resolution = normalized_resolution(args.source_grid, resolutions)
    target_grids = args.target_grid or ["1km", "5km"]
    target_specs = [(label, normalized_resolution(label, resolutions)) for label in target_grids]

    derived_specs = [(f"{args.column_prefix}{label}", resolution) for label, resolution in target_specs]
    target_columns = [name for name, _ in derived_specs]
    LOGGER.info("Dataset root: %s", dataset_root)
    LOGGER.info("Source grid: %s (%s°)", args.source_grid, source_resolution)
    LOGGER.info(
        "Target grids: %s",
        ", ".join(f"{label}={resolution}°" for label, resolution in target_specs),
    )
    LOGGER.info("Derived columns: %s", ", ".join(target_columns))

    target_geobox = get_or_create_geobox(str(data_root))
    partitions = list(iter_partition_files(dataset_root))
    if args.limit is not None:
        partitions = partitions[: args.limit]
    if not partitions:
        LOGGER.warning("No partition files found under %s", dataset_root)
        return 0

    total_partitions = 0
    total_added = 0
    total_replaced = 0

    for ix, iy, path in partitions:
        base_tile_geobox = create_tile_geobox(target_geobox, args.tile_size, ix, iy)
        added, replaced = update_partition(
            path=path,
            ix=ix,
            iy=iy,
            base_tile_geobox=base_tile_geobox,
            source_resolution=source_resolution,
            derived_specs=derived_specs,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        total_partitions += 1
        total_added += added
        total_replaced += replaced

    LOGGER.info(
        "Done. partitions=%d added=%d replaced=%d dry_run=%s",
        total_partitions,
        total_added,
        total_replaced,
        args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

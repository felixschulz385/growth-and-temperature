"""Shape detectors for a ``run_tiled_prepare``-produced tiled-parquet directory.

Directory shape: ``<parquet_dir>/ix=<tile.row>/iy=<tile.col>/part[-<year>].parquet``,
one ``cell_id``-keyed part per (tile, year) unit (or per tile for static sources).
``cell_id = row * full_width + col`` against the FULL canonical grid
(``src.data.common.geobox.cell_id``).

The DuckDB assembly engine (``src.data.assemble.sql_engine``) reads these parts
directly with ``read_parquet``; these helpers only classify a path and read its
schema/year coverage.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from typing import List, Optional, Sequence

import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

_YEAR_PART_RE = re.compile(r"^part-(\d+)\.parquet$")


def is_tiled_parquet_dataset(path: str) -> bool:
    """True if *path* is a ``run_tiled_prepare``-shaped directory of
    ``ix=<row>/iy=<col>/part[-<year>].parquet`` files, rather than a Zarr store
    or a single parquet file."""
    if not os.path.isdir(path):
        return False
    if any(os.path.exists(os.path.join(path, name)) for name in ("zarr.json", ".zmetadata", ".zgroup")):
        return False
    return bool(_partitioned_parquet_files(path))


def _partitioned_parquet_files(path: str) -> List[str]:
    return glob.glob(os.path.join(path, "ix=*", "iy=*", "part*.parquet"))


def _detect_years(part_files: Sequence[str]) -> Optional[List[int]]:
    """``None`` for a static (year-independent) dataset (``part.parquet`` files);
    otherwise the sorted, deduplicated set of years found across every tile's
    ``part-<year>.parquet`` files."""
    years = set()
    static = False
    for f in part_files:
        name = os.path.basename(f)
        if name == "part.parquet":
            static = True
            continue
        m = _YEAR_PART_RE.match(name)
        if m:
            years.add(int(m.group(1)))
    if static and years:
        raise ValueError(f"Mixed static and temporal parquet parts found under one dataset: {part_files[:2]}...")
    return sorted(years) if years else None


def _detect_variables(part_files: Sequence[str]) -> List[str]:
    schema_names = pq.ParquetFile(part_files[0]).schema_arrow.names
    return [c for c in schema_names if c not in ("cell_id", "year")]

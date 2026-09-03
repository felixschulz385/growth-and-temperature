"""Shared helpers for the DuckDB assembly-engine tests.

A synthetic ``run_tiled_prepare``-shaped tiled-parquet fixture on a small
hand-built canonical grid (``GridFacts`` is injected, bypassing the real
~34k x 12k EASE geobox), so the block-aggregation / merge / join / pixel_id
math can be asserted against values computed by hand.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import duckdb
import pandas as pd
import pytest

from src.data.assemble import sql_engine as se


@pytest.fixture
def tiny_grid() -> se.GridFacts:
    """W=20, H=16, tile size 8 -> 2 tile-rows x 3 tile-cols; caller picks F/DR/DC."""
    return se.GridFacts(W=20, H=16, TS=8, F=1, DR=0, DC=0)


def grid(F: int = 1, DR: int = 0, DC: int = 0, *, W=20, H=16, TS=8) -> se.GridFacts:
    return se.GridFacts(W=W, H=H, TS=TS, F=F, DR=DR, DC=DC)


def write_tiled_source(
    root: str,
    name: str,
    *,
    W: int,
    H: int,
    years: Optional[Sequence[int]],
    value_fn,               # (r, c, year|None) -> dict[str, float|str]
    keep: Optional[set] = None,  # native (r, c) to include; None = all
) -> str:
    """Write ``<root>/<name>/ix=0/iy=0/part[-<year>].parquet`` with a global
    row-major ``cell_id = r*W + c``."""
    path = os.path.join(root, name)
    sub = os.path.join(path, "ix=0", "iy=0")
    os.makedirs(sub, exist_ok=True)
    for yr in (list(years) if years else [None]):
        rows = []
        for r in range(H):
            for c in range(W):
                if keep is not None and (r, c) not in keep:
                    continue
                rec = {"cell_id": r * W + c}
                if yr is not None:
                    rec["year"] = yr
                rec.update(value_fn(r, c, yr))
                rows.append(rec)
        fn = "part.parquet" if yr is None else f"part-{yr}.parquet"
        pd.DataFrame(rows).to_parquet(os.path.join(sub, fn), index=False)
    return path


def write_land_mask(root: str, *, W: int, H: int, land) -> str:
    """``land(r, c) -> bool``. Returns the land-mask tiled-parquet dir."""
    path = os.path.join(root, "land_mask")
    sub = os.path.join(path, "ix=0", "iy=0")
    os.makedirs(sub, exist_ok=True)
    rows = [{"cell_id": r * W + c, "land_mask": bool(land(r, c))} for r in range(H) for c in range(W)]
    pd.DataFrame(rows).to_parquet(os.path.join(sub, "part.parquet"), index=False)
    return path


def run_create(g: se.GridFacts, datasets: Dict[str, dict], out: str, *,
               land_mask_path: Optional[str] = None,
               derived: Optional[dict] = None,
               shake_offset=(0.0, 0.0)) -> pd.DataFrame:
    sources, join_specs = se._build_sources(datasets, datasource_filter=None)
    derived_specs = se.normalize_derived_pixel_id_specs(derived)
    se._check_no_column_collisions(sources, derived_specs)
    con = se._connect(se.DuckDBConfig())
    try:
        se._run_create(
            con, sources, join_specs, g, shake_offset, out,
            land_scan=se._land_scan_sql(land_mask_path),
            year_range=None,
            derived_specs=derived_specs,
            compression="zstd",
        )
    finally:
        con.close()
    return read_panel(out)


def run_update(g: se.GridFacts, datasets: Dict[str, dict], datasource: str, out: str,
               *, land_mask_path: Optional[str] = None) -> pd.DataFrame:
    sources, _join_specs = se._build_sources(datasets, datasource_filter=datasource)
    con = se._connect(se.DuckDBConfig())
    try:
        se._run_update(
            con, sources[0], g, out,
            land_scan=se._land_scan_sql(land_mask_path), year_range=None, compression="zstd",
        )
    finally:
        con.close()
    return read_panel(out)


def read_panel(out: str) -> pd.DataFrame:
    files = os.path.join(out, "**", "*.parquet")
    df = duckdb.sql(f"SELECT * FROM read_parquet('{files}', hive_partitioning=true)").df()
    sort_cols = [c for c in ("pixel_id", "year") if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)

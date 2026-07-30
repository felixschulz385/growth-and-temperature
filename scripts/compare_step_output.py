#!/usr/bin/env python3
"""Diff two versions of a source's output artefact -- the migration-
equivalence tool docs/design/09-integrated-pipeline.md §10/§13 calls for at
the hard validation gate before cutover (step 9): for one target per
archetype (bulk-raster-with-composite, MODIS streaming, vector, tabular-join,
point), run the same target through the old code and the new code, then use
this tool to confirm the outputs are equivalent before deleting the old code.

Supports Zarr stores, single-band/multi-band GeoTIFFs, and parquet tables --
the three artefact shapes this migration's sources produce. Read-only: never
modifies either input.

Usage:
    python scripts/compare_step_output.py OLD_PATH NEW_PATH [--rtol 1e-6] [--atol 1e-6]

Exit code 0 if equivalent, 1 if not (with a diff explaining why), 2 on error
(e.g. unreadable file, unsupported format).

Also worth keeping permanently, not just for this migration: the same tool
is exactly what the future `legacy_4326 -> ease6933` grid cutover
(docs/design/05-migration.md) will need to validate its own dual-grid
overlap period.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _is_zarr(path: Path) -> bool:
    return path.suffix == ".zarr" or (path / ".zattrs").exists() or (path / "zarr.json").exists()


def _is_geotiff(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in (".tif", ".tiff")


def _is_parquet(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".parquet"


def compare_zarr(old_path: Path, new_path: Path, rtol: float, atol: float) -> list[str]:
    import xarray as xr

    problems = []
    old_ds = xr.open_zarr(str(old_path), consolidated=False)
    new_ds = xr.open_zarr(str(new_path), consolidated=False)

    old_vars, new_vars = set(old_ds.data_vars), set(new_ds.data_vars)
    if old_vars != new_vars:
        problems.append(f"variables differ: only-in-old={old_vars - new_vars} only-in-new={new_vars - old_vars}")

    for var in sorted(old_vars & new_vars):
        old_da, new_da = old_ds[var], new_ds[var]
        if old_da.dims != new_da.dims:
            problems.append(f"{var}: dims differ: old={old_da.dims} new={new_da.dims}")
            continue
        if old_da.shape != new_da.shape:
            problems.append(f"{var}: shape differs: old={old_da.shape} new={new_da.shape}")
            continue
        if old_da.dtype != new_da.dtype:
            problems.append(f"{var}: dtype differs: old={old_da.dtype} new={new_da.dtype}")

        try:
            import numpy as np

            old_vals, new_vals = old_da.values, new_da.values
            if np.issubdtype(old_vals.dtype, np.floating):
                both_nan = np.isnan(old_vals) & np.isnan(new_vals)
                close = np.isclose(old_vals, new_vals, rtol=rtol, atol=atol, equal_nan=True)
                mismatch_frac = 1.0 - (close | both_nan).mean()
            else:
                mismatch_frac = 1.0 - (old_vals == new_vals).mean()
            if mismatch_frac > 0:
                problems.append(f"{var}: {mismatch_frac:.4%} of values differ beyond tolerance")
        except Exception as exc:  # pragma: no cover -- diagnostic path
            problems.append(f"{var}: could not compare values ({exc})")

    for coord in sorted(set(old_ds.coords) & set(new_ds.coords)):
        old_vals, new_vals = old_ds.coords[coord].values, new_ds.coords[coord].values
        # 0-d coords (e.g. rioxarray's `spatial_ref` CRS grid-mapping variable)
        # aren't iterable -- `list()` on them raises "iteration over a 0-d
        # array" rather than comparing, found while validating this tool
        # against a real ACAG PREPARE target during the migration's hard gate.
        if old_vals.ndim == 0:
            if old_vals.item() != new_vals.item():
                problems.append(f"coord '{coord}' differs between old and new")
        elif list(old_vals) != list(new_vals):
            problems.append(f"coord '{coord}' differs between old and new")

    return problems


def compare_geotiff(old_path: Path, new_path: Path, rtol: float, atol: float) -> list[str]:
    import numpy as np
    import rasterio

    problems = []
    with rasterio.open(old_path) as old_src, rasterio.open(new_path) as new_src:
        if old_src.count != new_src.count:
            problems.append(f"band count differs: old={old_src.count} new={new_src.count}")
        if (old_src.width, old_src.height) != (new_src.width, new_src.height):
            problems.append(f"shape differs: old={(old_src.width, old_src.height)} new={(new_src.width, new_src.height)}")
        if old_src.crs != new_src.crs:
            problems.append(f"CRS differs: old={old_src.crs} new={new_src.crs}")

        old_descriptions, new_descriptions = list(old_src.descriptions), list(new_src.descriptions)
        if old_descriptions != new_descriptions:
            problems.append(f"band descriptions differ: old={old_descriptions} new={new_descriptions}")

        for i in range(1, min(old_src.count, new_src.count) + 1):
            old_band, new_band = old_src.read(i), new_src.read(i)
            if old_band.shape != new_band.shape:
                problems.append(f"band {i}: shape differs")
                continue
            if np.issubdtype(old_band.dtype, np.floating):
                both_nan = np.isnan(old_band) & np.isnan(new_band)
                close = np.isclose(old_band, new_band, rtol=rtol, atol=atol, equal_nan=True)
                mismatch_frac = 1.0 - (close | both_nan).mean()
            else:
                mismatch_frac = 1.0 - (old_band == new_band).mean()
            if mismatch_frac > 0:
                problems.append(f"band {i}: {mismatch_frac:.4%} of pixels differ beyond tolerance")
    return problems


def compare_parquet(old_path: Path, new_path: Path, rtol: float, atol: float) -> list[str]:
    import pandas as pd

    problems = []
    old_df, new_df = pd.read_parquet(old_path), pd.read_parquet(new_path)

    if list(old_df.columns) != list(new_df.columns):
        problems.append(f"columns differ: old={list(old_df.columns)} new={list(new_df.columns)}")
    if len(old_df) != len(new_df):
        problems.append(f"row count differs: old={len(old_df)} new={len(new_df)}")
        return problems

    common_cols = [c for c in old_df.columns if c in new_df.columns]
    sort_keys = [c for c in ("iso3", "pixel_id", "year", "time") if c in common_cols]
    if sort_keys:
        old_df = old_df.sort_values(sort_keys).reset_index(drop=True)
        new_df = new_df.sort_values(sort_keys).reset_index(drop=True)

    for col in common_cols:
        try:
            pd.testing.assert_series_equal(old_df[col], new_df[col], check_exact=False, rtol=rtol, atol=atol, check_names=False)
        except AssertionError as exc:
            problems.append(f"column '{col}' differs: {exc}")
    return problems


def compare(old_path: Path, new_path: Path, rtol: float, atol: float) -> list[str]:
    if not old_path.exists():
        return [f"old path does not exist: {old_path}"]
    if not new_path.exists():
        return [f"new path does not exist: {new_path}"]

    if _is_zarr(old_path) or _is_zarr(new_path):
        return compare_zarr(old_path, new_path, rtol, atol)
    if _is_geotiff(old_path) and _is_geotiff(new_path):
        return compare_geotiff(old_path, new_path, rtol, atol)
    if _is_parquet(old_path) and _is_parquet(new_path):
        return compare_parquet(old_path, new_path, rtol, atol)
    return [f"unsupported or mismatched artefact types: old={old_path} new={new_path}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("old_path", type=Path, help="Artefact produced by the old (pre-migration) code")
    parser.add_argument("new_path", type=Path, help="Artefact produced by the new (migrated) code")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for float comparisons")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for float comparisons")
    args = parser.parse_args()

    try:
        problems = compare(args.old_path, args.new_path, args.rtol, args.atol)
    except Exception as exc:
        print(f"ERROR comparing {args.old_path} vs {args.new_path}: {exc}", file=sys.stderr)
        return 2

    if problems:
        print(f"NOT EQUIVALENT: {args.old_path} vs {args.new_path}")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"EQUIVALENT: {args.old_path} vs {args.new_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Diff two versions of a source's output artefact -- the migration-
equivalence tool docs/design/09-integrated-pipeline.md §10/§13 calls for at
the hard validation gate before cutover (step 9): for one target per
archetype (bulk-raster-with-composite, MODIS streaming, vector, tabular-join,
point), run the same target through the old code and the new code, then use
this tool to confirm the outputs are equivalent before deleting the old code.

Supports Zarr stores, single-band/multi-band GeoTIFFs, parquet tables,
GeoPackage vector layers, and DuckDB databases -- the five artefact shapes
this migration's sources produce. Read-only: never modifies either input
(DuckDB files are opened `read_only=True`).

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


def _is_geopackage(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".gpkg"


def _is_duckdb(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in (".duckdb", ".db")


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


def compare_geopackage(old_path: Path, new_path: Path, rtol: float, atol: float) -> list[str]:
    import geopandas as gpd
    import pandas as pd

    problems = []
    old_layers = set(gpd.list_layers(str(old_path)).name)
    new_layers = set(gpd.list_layers(str(new_path)).name)
    if old_layers != new_layers:
        problems.append(f"layers differ: only-in-old={old_layers - new_layers} only-in-new={new_layers - old_layers}")

    for layer in sorted(old_layers & new_layers):
        old_gdf = gpd.read_file(str(old_path), layer=layer, engine="pyogrio")
        new_gdf = gpd.read_file(str(new_path), layer=layer, engine="pyogrio")

        old_cols = [c for c in old_gdf.columns if c != "geometry"]
        new_cols = [c for c in new_gdf.columns if c != "geometry"]
        if set(old_cols) != set(new_cols):
            problems.append(f"layer '{layer}': columns differ: old={old_cols} new={new_cols}")
        if len(old_gdf) != len(new_gdf):
            problems.append(f"layer '{layer}': row count differs: old={len(old_gdf)} new={len(new_gdf)}")
            continue

        common_cols = [c for c in old_cols if c in new_cols]
        # Row order isn't guaranteed to be stable across two independent
        # reads/writes of the same source file, so sort by whatever looks
        # like a natural id column before comparing row-for-row -- otherwise
        # an identical layer with merely reordered rows would falsely report
        # as different.
        sort_key = next((c for c in ("GID_0", "GID_1", "GID_2", "GID_3", "id", "ID") if c in common_cols), None)
        if sort_key:
            old_gdf = old_gdf.sort_values(sort_key).reset_index(drop=True)
            new_gdf = new_gdf.sort_values(sort_key).reset_index(drop=True)
        else:
            old_gdf = old_gdf.reset_index(drop=True)
            new_gdf = new_gdf.reset_index(drop=True)

        for col in common_cols:
            try:
                pd.testing.assert_series_equal(
                    old_gdf[col], new_gdf[col], check_exact=False, rtol=rtol, atol=atol, check_names=False
                )
            except AssertionError as exc:
                problems.append(f"layer '{layer}' column '{col}' differs: {exc}")

        geom_matches = old_gdf.geometry.geom_equals_exact(new_gdf.geometry, tolerance=max(atol, 1e-9))
        mismatch_frac = 1.0 - geom_matches.mean()
        if mismatch_frac > 0:
            problems.append(f"layer '{layer}': {mismatch_frac:.4%} of geometries differ beyond tolerance")

    return problems


def compare_duckdb(old_path: Path, new_path: Path, rtol: float, atol: float) -> list[str]:
    import duckdb
    import pandas as pd

    problems = []
    old_con = duckdb.connect(str(old_path), read_only=True)
    new_con = duckdb.connect(str(new_path), read_only=True)
    try:
        for con in (old_con, new_con):
            try:
                con.execute("INSTALL spatial; LOAD spatial;")
            except Exception:
                pass  # geometry columns, if any, are skipped below regardless

        table_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        old_tables = {r[0] for r in old_con.execute(table_query).fetchall()}
        new_tables = {r[0] for r in new_con.execute(table_query).fetchall()}
        if old_tables != new_tables:
            problems.append(f"tables differ: only-in-old={old_tables - new_tables} only-in-new={new_tables - old_tables}")

        for table in sorted(old_tables & new_tables):
            old_df = old_con.execute(f"SELECT * FROM {table}").fetchdf()
            new_df = new_con.execute(f"SELECT * FROM {table}").fetchdf()

            old_cols, new_cols = list(old_df.columns), list(new_df.columns)
            if set(old_cols) != set(new_cols):
                problems.append(f"table '{table}': columns differ: old={old_cols} new={new_cols}")
                continue
            if len(old_df) != len(new_df):
                problems.append(f"table '{table}': row count differs: old={len(old_df)} new={len(new_df)}")
                continue

            # DuckDB spatial geometry columns are opaque WKB blobs -- compare
            # every other column's values, but only that a geometry column
            # is present/absent (already covered by the column-set check
            # above), not its byte content.
            geometry_cols = {c for c in old_cols if "geometry" in c.lower()}
            comparable_cols = [c for c in old_cols if c not in geometry_cols]

            sort_key = [c for c in comparable_cols if c.lower() in ("id", "property_id", "year", "gid_1", "gid_2")] or comparable_cols
            if sort_key:
                old_df = old_df.sort_values(sort_key).reset_index(drop=True)
                new_df = new_df.sort_values(sort_key).reset_index(drop=True)

            for col in comparable_cols:
                try:
                    pd.testing.assert_series_equal(
                        old_df[col], new_df[col], check_exact=False, rtol=rtol, atol=atol, check_names=False
                    )
                except AssertionError as exc:
                    problems.append(f"table '{table}' column '{col}' differs: {exc}")
    finally:
        old_con.close()
        new_con.close()
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
    if _is_geopackage(old_path) and _is_geopackage(new_path):
        return compare_geopackage(old_path, new_path, rtol, atol)
    if _is_duckdb(old_path) and _is_duckdb(new_path):
        return compare_duckdb(old_path, new_path, rtol, atol)
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

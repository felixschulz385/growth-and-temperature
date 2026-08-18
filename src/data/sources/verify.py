"""Cheap, format-generic sanity checks for GRID-stage outputs.

Not a full data QA pass -- opens the store, confirms the expected
variables/columns are present, has a CRS (rasters only), and that a strided
sample of the data isn't degenerate (all-nodata, or outside a declared
physical value range). Catches the two failure modes a killed or logic-bug'd
GRID job produces (empty output, all-nodata output) without reading full
arrays.

Results are cached in a `_verification/<name>.json` manifest sibling to the
output (see `manifest_path()`), keyed by a cheap stat-based fingerprint of
the output (`_fingerprint()`) -- so `data summary` (which calls this on
every complete GRID target on every invocation) and the assembly gate don't
re-open/re-sample a store that hasn't changed since it was last verified.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    detail: str


def verification_meta(
    raw: dict,
    *,
    expected_vars: Sequence[str] | None = None,
    value_range: tuple[float, float] | None = None,
    range_vars: Sequence[str] | None = None,
) -> dict:
    """Build the `expected_vars`/`value_range`/`range_vars` `StepTarget.meta`
    entries for one source's GRID target, letting that source's own
    `verification:` config block (in e.g. orchestration/configs/data.yaml)
    override the Python-side defaults passed in by the caller -- so a
    deployment can loosen/tighten a range or variable list per source
    without a code change, while every source still works out of the box
    with no `verification:` block at all.

    *raw* is a source's `SourceConfig.raw` (the source's own config dict).
    Config shape:

        sources:
          acag:
            verification:
              expected_vars: [pm25]
              value_range: [0, 500]

    An explicit `value_range: null` in config disables the range check
    entirely even if the caller passed one in.
    """
    cfg = raw.get("verification") or {}

    result: dict = {}
    ev = cfg["expected_vars"] if "expected_vars" in cfg else expected_vars
    if ev is not None:
        result["expected_vars"] = tuple(ev)

    vr = cfg["value_range"] if "value_range" in cfg else value_range
    if vr is not None:
        result["value_range"] = tuple(vr)

    rv = cfg["range_vars"] if "range_vars" in cfg else range_vars
    if rv is not None:
        result["range_vars"] = tuple(rv)

    return result


def manifest_path(path: str) -> str:
    """Where the cached verification result for *path* lives: a
    `_verification/<name>.json` file sibling to *path* itself, never inside
    it -- fingerprinting a zarr store's own directory must not see the
    manifest as one of its own files, or writing the manifest would
    invalidate the very fingerprint it just recorded."""
    trimmed = path.rstrip(os.sep)
    return os.path.join(os.path.dirname(trimmed), "_verification", f"{os.path.basename(trimmed)}.json")


def _fingerprint(path: str) -> str:
    """Cheap, single-`stat()` fingerprint for cache invalidation. Prefers (in
    order): the MARKER-completion sibling file (`steps.marker_path`, touched
    only on a fresh successful write); a zarr store's own root metadata file
    (`zarr.json` / `.zmetadata` / `.zgroup`, rewritten by every
    `to_zarr(mode="w", ...)` call) -- stat'ing one file beats walking an
    entire chunk tree; the path's own mtime+size otherwise (single-file
    outputs, or a last-resort fallback for a directory with neither -- note
    a bare directory mtime only reflects direct-child adds/removes, so it
    won't always catch a nested chunk being rewritten in place)."""
    from src.data.sources.steps import marker_path

    marker = marker_path(path)
    if os.path.exists(marker):
        st = os.stat(marker)
        return f"{st.st_mtime_ns}:{st.st_size}"

    if os.path.isdir(path):
        for meta_name in ("zarr.json", ".zmetadata", ".zgroup"):
            meta_path = os.path.join(path, meta_name)
            if os.path.exists(meta_path):
                st = os.stat(meta_path)
                return f"{st.st_mtime_ns}:{st.st_size}"

    st = os.stat(path)
    return f"{st.st_mtime_ns}:{st.st_size}"


def _params_fingerprint(
    expected_vars: Sequence[str] | None,
    value_range: tuple[float, float] | None,
    range_vars: Sequence[str] | None,
) -> str:
    """Cache entries must be scoped to *both* the output's own state and the
    parameters it was checked against -- two callers checking the same store
    with different `expected_vars`/`value_range` (e.g. a source's own
    `expected_vars` vs. an assembly config's narrower `columns`) must not
    silently reuse each other's cached verdict."""
    return json.dumps(
        {
            "expected_vars": list(expected_vars) if expected_vars else None,
            "value_range": list(value_range) if value_range else None,
            "range_vars": list(range_vars) if range_vars else None,
        },
        sort_keys=True,
    )


def _read_cached(path: str, fingerprint: str) -> VerificationResult | None:
    mpath = manifest_path(path)
    try:
        with open(mpath, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("fingerprint") != fingerprint:
        return None
    return VerificationResult(ok=bool(data["ok"]), detail=str(data["detail"]))


def _write_cache(path: str, fingerprint: str, result: VerificationResult) -> None:
    mpath = manifest_path(path)
    try:
        os.makedirs(os.path.dirname(mpath), exist_ok=True)
        tmp_path = mpath + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "fingerprint": fingerprint,
                    "ok": result.ok,
                    "detail": result.detail,
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                },
                fh,
                indent=2,
            )
        os.replace(tmp_path, mpath)
    except OSError:
        pass  # caching is a speed optimization, not required for correctness


def verify_grid_output(
    path: str,
    *,
    expected_vars: Sequence[str] | None = None,
    value_range: tuple[float, float] | None = None,
    range_vars: Sequence[str] | None = None,
    force: bool = False,
) -> VerificationResult:
    """Verify a GRID-stage (or assembly-input) output at *path*, using a
    fingerprint-checked cache (see module docstring) unless *force* is set.

    Dispatches on file type: zarr stores (directories with zarr root
    metadata) and GeoTIFFs get a full raster check (variables/bands, CRS,
    sampled value sanity); a single parquet file (e.g. per-GID sidecar
    outputs, which aren't gridded and have no CRS/pixel values) gets a schema
    + row-count check; a directory of `cell_id`-keyed parquet parts (the
    `run_tiled_prepare` driver's own output shape, `ix=<row>/iy=<col>/
    part[-<year>].parquet`) gets the same variable/range checks a zarr store
    would, sampled across a bounded spread of part files instead of chunks.
    Anything else is reported ok on existence alone -- an unrecognized format
    shouldn't hard-fail verification just because this checker doesn't know
    how to open it.

    *range_vars* narrows which of *expected_vars* actually get the
    *value_range* check (all of them, if omitted) -- for sources where one
    GRID target bundles variables on different physical scales (e.g. GLASS's
    absolute-Kelvin `mean`/`max`/`min` alongside a `std` spread that isn't on
    the same scale), so a real value outside the *absolute* range doesn't
    make an unrelated variable look like a false failure.
    """
    if not os.path.exists(path):
        return VerificationResult(False, f"output path does not exist: {path}")

    fingerprint = _fingerprint(path) + "|" + _params_fingerprint(expected_vars, value_range, range_vars)
    if not force:
        cached = _read_cached(path, fingerprint)
        if cached is not None:
            return cached

    result = _run_verification(path, expected_vars=expected_vars, value_range=value_range, range_vars=range_vars)
    _write_cache(path, fingerprint, result)
    return result


def _is_zarr_store(path: str) -> bool:
    """Same root-metadata-file check `_fingerprint()` uses -- a directory
    only counts as a zarr store if it actually carries zarr's own root
    metadata, not just because it's a directory (the `run_tiled_prepare`
    driver's `cell_id`-keyed parquet-parts output is also a directory)."""
    return any(os.path.exists(os.path.join(path, name)) for name in ("zarr.json", ".zmetadata", ".zgroup"))


def _partitioned_parquet_files(path: str) -> list[str]:
    import glob

    return sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))


def _run_verification(
    path: str,
    *,
    expected_vars: Sequence[str] | None,
    value_range: tuple[float, float] | None,
    range_vars: Sequence[str] | None,
) -> VerificationResult:
    try:
        if path.endswith(".parquet"):
            return _verify_table(path, expected_vars=expected_vars)
        if os.path.isdir(path):
            if _is_zarr_store(path):
                return _verify_zarr(path, expected_vars=expected_vars, value_range=value_range, range_vars=range_vars)
            part_files = _partitioned_parquet_files(path)
            if part_files:
                return _verify_partitioned_table(
                    part_files, expected_vars=expected_vars, value_range=value_range, range_vars=range_vars
                )
            if not os.listdir(path):
                return VerificationResult(False, f"{path} is empty")
        if path.lower().endswith((".tif", ".tiff")):
            return _verify_geotiff(path, expected_vars=expected_vars, value_range=value_range, range_vars=range_vars)
    except Exception as exc:  # noqa: BLE001 -- verification must never crash the caller
        return VerificationResult(False, f"failed to open/read {path}: {exc}")

    return VerificationResult(True, "exists (unrecognized format, existence-only check)")


def _stride_sample(da, target_size: int = 200_000, max_chunks_per_dim: int = 4):
    """A small sample spread across the *entire* extent of every dim, not a
    fixed central crop. Several sources (e.g. mine-count grids) are globally
    sparse -- a center-of-array window would find nothing but nodata on
    legitimately good output. Striding across the full extent instead keeps
    the sample cheap while still being representative.

    Samples in *chunk*-index space, not element-index space: a handful of
    chunk positions per dim (spread across the chunk grid), each contributing
    a dense run of elements, rather than one element position per dim spread
    across every chunk. Same full-extent coverage, but bounds the number of
    distinct chunks opened to roughly `max_chunks_per_dim ** ndim` instead of
    one chunk per sampled element -- the difference between dozens of chunk
    reads and thousands on a store chunked as finely as one (year, tile) per
    chunk (see src/data/sources/snl_mining/source.py's `to_zarr` encoding)."""
    import numpy as np

    dims = list(da.sizes.items())
    if not dims:
        return da.compute()

    per_dim_target = max(1, round(target_size ** (1 / len(dims))))
    chunks = da.chunks  # dask block-size tuples per dim, in `dims` order; None if not dask-backed
    if chunks is None:
        indexers = {dim: slice(None, None, max(1, size // per_dim_target)) for dim, size in dims}
        return da.isel(indexers).compute()

    indexers = {}
    for (dim, size), block_sizes in zip(dims, chunks):
        n_chunks = len(block_sizes)
        starts = np.cumsum((0,) + block_sizes[:-1])
        n_pick = min(max_chunks_per_dim, n_chunks)
        picked_starts = starts[np.linspace(0, n_chunks - 1, n_pick, dtype=int)]
        run_len = max(1, per_dim_target // n_pick)
        indexers[dim] = np.concatenate(
            [np.arange(start, min(start + run_len, size)) for start in picked_starts]
        )
    return da.isel(indexers).compute()


def _check_sample_range(values, *, value_range: tuple[float, float] | None, label: str):
    """Shared finite/range check for one variable's/band's sample. Returns a
    `VerificationResult` on failure, `None` if the sample is fine."""
    import numpy as np

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return VerificationResult(False, f"{label}: sample is entirely nodata/NaN")
    if value_range is not None:
        lo, hi = value_range
        out_of_range = finite[(finite < lo) | (finite > hi)]
        if out_of_range.size:
            frac = out_of_range.size / finite.size
            return VerificationResult(
                False,
                f"{label}: {frac:.1%} of sampled values outside expected range [{lo}, {hi}] "
                f"(e.g. {out_of_range[0]})",
            )
    return None


def _verify_zarr(
    path: str,
    *,
    expected_vars: Sequence[str] | None,
    value_range: tuple[float, float] | None,
    range_vars: Sequence[str] | None,
) -> VerificationResult:
    import numpy as np
    import xarray as xr

    # decode_coords="all" promotes the CF grid_mapping variable (e.g.
    # "spatial_ref", written by `.rio.write_crs()` before every source's own
    # `to_zarr()`) to a coordinate -- without it, rioxarray's `.rio.crs`
    # can't find it and always reports None even on a store with a real CRS
    # (confirmed: every other reader of these same stores in this codebase,
    # e.g. src/data/assemble/loaders.py, re-assigns a CRS explicitly rather
    # than trusting a bare `open_zarr()`'s `.rio.crs`).
    ds = xr.open_zarr(path, consolidated=False, decode_coords="all")
    try:
        if expected_vars:
            missing = [v for v in expected_vars if v not in ds.data_vars]
            if missing:
                return VerificationResult(False, f"missing expected variable(s): {missing}")
            check_vars = list(expected_vars)
        else:
            if not ds.data_vars:
                return VerificationResult(False, "zarr store has no data variables")
            check_vars = list(ds.data_vars)

        crs = ds.rio.crs if hasattr(ds, "rio") else None
        if crs is None and not ds.attrs.get("crs"):
            return VerificationResult(False, "no CRS found (neither rio.crs nor a 'crs' attr)")

        range_var_set = set(range_vars) if range_vars is not None else None
        for var in check_vars:
            sample = _stride_sample(ds[var])
            values = np.asarray(sample.values, dtype="float64").ravel()
            applies_range = range_var_set is None or var in range_var_set
            failure = _check_sample_range(
                values, value_range=value_range if applies_range else None, label=f"variable '{var}'"
            )
            if failure is not None:
                return failure
        return VerificationResult(True, f"ok: {len(check_vars)} variable(s) sampled, values sane")
    finally:
        ds.close()


def _verify_geotiff(
    path: str,
    *,
    expected_vars: Sequence[str] | None,
    value_range: tuple[float, float] | None,
    range_vars: Sequence[str] | None,
) -> VerificationResult:
    import numpy as np
    import rioxarray  # noqa: F401 -- registers the .rio accessor

    da = rioxarray.open_rasterio(path)
    try:
        if da.rio.crs is None:
            return VerificationResult(False, "no CRS found on GeoTIFF")
        sample = _stride_sample(da)
        values = np.asarray(sample.values, dtype="float64").ravel()
        failure = _check_sample_range(values, value_range=value_range, label="sample")
        if failure is not None:
            return failure
        return VerificationResult(True, "ok: sampled values sane")
    finally:
        da.close()


def _verify_table(path: str, *, expected_vars: Sequence[str] | None) -> VerificationResult:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    columns = set(pf.schema_arrow.names)
    if expected_vars:
        missing = [v for v in expected_vars if v not in columns]
        if missing:
            return VerificationResult(False, f"missing expected column(s): {missing}")
    if pf.metadata.num_rows == 0:
        return VerificationResult(False, "table has zero rows")
    return VerificationResult(True, f"ok: {pf.metadata.num_rows} row(s), {len(columns)} column(s)")


def _verify_partitioned_table(
    part_files: Sequence[str],
    *,
    expected_vars: Sequence[str] | None,
    value_range: tuple[float, float] | None,
    range_vars: Sequence[str] | None,
) -> VerificationResult:
    """Verify a `run_tiled_prepare`-produced output: a directory of
    `cell_id`-keyed parquet parts (`ix=<row>/iy=<col>/part[-<year>].parquet`),
    not a single file -- so unlike `_verify_table`, this reads a bounded
    sample of *files*, not one file's own row sample.

    Row-count and column presence come cheaply from each file's own parquet
    footer metadata (no data read). The value-range check reads a stride
    sample of up to 8 part files spread across the *entire* file list (same
    "spread across the extent, bound the file count" spirit as
    `_stride_sample` -- one tile is not representative of a globally sparse
    or regionally-biased source any more than one chunk was for a zarr
    store), same `range_vars` narrowing `_verify_zarr` supports.
    """
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    schema_names = set(pq.ParquetFile(part_files[0]).schema_arrow.names)
    if expected_vars:
        missing = [v for v in expected_vars if v not in schema_names]
        if missing:
            return VerificationResult(False, f"missing expected column(s): {missing}")
        check_cols = list(expected_vars)
    else:
        check_cols = sorted(schema_names - {"cell_id", "year"})
        if not check_cols:
            return VerificationResult(False, "no data columns found (only cell_id/year)")

    total_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in part_files)
    if total_rows == 0:
        return VerificationResult(False, "table has zero rows across all part files")

    n_pick = min(8, len(part_files))
    picked = [part_files[i] for i in np.linspace(0, len(part_files) - 1, n_pick, dtype=int)]
    sample_df = pd.concat([pd.read_parquet(f, columns=check_cols) for f in picked], ignore_index=True)

    range_var_set = set(range_vars) if range_vars is not None else None
    for col in check_cols:
        values = pd.to_numeric(sample_df[col], errors="coerce").to_numpy(dtype="float64")
        applies_range = range_var_set is None or col in range_var_set
        failure = _check_sample_range(
            values, value_range=value_range if applies_range else None, label=f"column '{col}'"
        )
        if failure is not None:
            return failure

    return VerificationResult(
        True,
        f"ok: {len(check_cols)} column(s) sampled across {len(picked)}/{len(part_files)} part file(s), "
        f"{total_rows} row(s), values sane",
    )

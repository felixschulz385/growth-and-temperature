"""Reconstruct a lazy, dask-backed `xr.Dataset` from a `run_tiled_prepare`-produced
tiled-parquet directory (`src.data.common.prepare.driver`), so the rest of the
assembly pipeline (`TileProcessor._extract_dataset_tile`'s `.sel()`/`.odc.reproject()`
raster logic) can keep treating a PREPARE dataset as a lazy raster the same way it
already treats a Zarr store -- only the loading layer differs.

Directory shape: `<parquet_dir>/ix=<tile.row>/iy=<tile.col>/part[-<year>].parquet`,
one `cell_id`-keyed part per (tile, year) unit (or per tile for static sources).
`cell_id = row * full_width + col` against the FULL canonical grid
(`src.data.common.geobox.cell_id`), decodable back to tile-local pixel positions
with plain integer arithmetic.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from typing import Dict, List, Optional, Sequence

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr

from src.data.common import tiling
from src.data.common.geobox.cell_id import decode_cell_id

logger = logging.getLogger(__name__)

_YEAR_PART_RE = re.compile(r"^part-(\d+)\.parquet$")


def is_tiled_parquet_dataset(path: str) -> bool:
    """True if *path* is a `run_tiled_prepare`-shaped directory of
    `ix=<row>/iy=<col>/part[-<year>].parquet` files, rather than a Zarr store or a
    single parquet file."""
    if not os.path.isdir(path):
        return False
    if any(os.path.exists(os.path.join(path, name)) for name in ("zarr.json", ".zmetadata", ".zgroup")):
        return False
    return bool(_partitioned_parquet_files(path))


def _partitioned_parquet_files(path: str) -> List[str]:
    return glob.glob(os.path.join(path, "ix=*", "iy=*", "part*.parquet"))


def _detect_years(part_files: Sequence[str]) -> Optional[List[int]]:
    """`None` for a static (year-independent) dataset (`part.parquet` files);
    otherwise the sorted, deduplicated set of years found across every tile's
    `part-<year>.parquet` files -- a coverage hole in one tile/year (the
    source's `raw_getter` returned nothing for that unit) does not shrink this
    set, since other tiles still declare that year."""
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


def _read_tile_block(
    part_path: str,
    y_slice_start: int,
    x_slice_start: int,
    shape: tuple,
    full_width: int,
    variables: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Read one tile's part file (or return an all-NaN block if the unit was
    never written, e.g. a globally sparse source with no data in this tile)
    and scatter its rows into dense `shape`-sized arrays keyed by variable."""
    h, w = shape
    if not os.path.exists(part_path):
        return {v: np.full((h, w), np.nan, dtype=np.float32) for v in variables}

    df = pd.read_parquet(part_path, columns=["cell_id", *variables])
    row, col = decode_cell_id(df["cell_id"].to_numpy(), full_width)
    local_row = row - y_slice_start
    local_col = col - x_slice_start

    out: Dict[str, np.ndarray] = {}
    for v in variables:
        arr = np.full((h, w), np.nan, dtype=np.float32)
        arr[local_row, local_col] = df[v].to_numpy(dtype=np.float32)
        out[v] = arr
    return out


def open_tiled_parquet_dataset(
    parquet_dir: str,
    target_geobox,
    tile_size: int = tiling.DEFAULT_TILE_SIZE,
    variables: Optional[Sequence[str]] = None,
) -> xr.Dataset:
    """Open a `run_tiled_prepare`-produced tiled-parquet directory as a lazy
    `xr.Dataset` shaped like the equivalent Zarr store would be: dims
    `target_geobox.dimensions` (e.g. `latitude`/`longitude`), a leading `year`
    dim for temporal datasets, one data_var per column, dask-backed so
    `.sel()`/`.odc.reproject()` only touch the tile blocks a caller's slice
    actually intersects.

    `target_geobox` must be the SAME geobox (tile boundaries) used to write
    *parquet_dir* -- assembly's own target_geobox and PREPARE's currently
    resolve to the same shared geobox for the one grid_id this repo produces
    (`src.data.common.geobox.get_or_create_geobox` /
    `src.data.common.geobox.target.get_target_geobox`), so callers should pass
    that same geobox through rather than deriving a new one.
    """
    tiles = list(tiling.iter_tiles(target_geobox, tile_size))
    part_files = _partitioned_parquet_files(parquet_dir)
    if not part_files:
        raise ValueError(f"No tiled-parquet part files found under {parquet_dir}")

    years = _detect_years(part_files)
    if variables is None:
        variables = _detect_variables(part_files)
    variables = list(variables)

    full_width = target_geobox.shape[1]
    dim_y, dim_x = target_geobox.dimensions
    n_rows, n_cols = tiling.grid_shape(target_geobox, tile_size)

    def block_for(tile, year: Optional[int]):
        filename = "part.parquet" if year is None else f"part-{year}.parquet"
        part_path = os.path.join(parquet_dir, f"ix={tile.row}", f"iy={tile.col}", filename)
        return dask.delayed(_read_tile_block)(
            part_path, tile.y_slice.start, tile.x_slice.start, tile.geobox.shape, full_width, variables
        )

    def var_block(delayed_dict, var: str, shape):
        return da.from_delayed(
            dask.delayed(lambda d: d[var])(delayed_dict), shape=shape, dtype=np.float32
        )

    tiles_by_rc = {(t.row, t.col): t for t in tiles}

    def var_array_for_year(var: str, year: Optional[int]):
        rows = []
        for row in range(n_rows):
            row_blocks = []
            for col in range(n_cols):
                tile = tiles_by_rc[(row, col)]
                delayed_dict = block_for(tile, year)
                row_blocks.append(var_block(delayed_dict, var, tile.geobox.shape))
            rows.append(row_blocks)
        return da.block(rows)

    data_vars = {}
    coords = {
        dim_y: target_geobox.coords[dim_y].values.round(5),
        dim_x: target_geobox.coords[dim_x].values.round(5),
    }

    if years is None:
        for var in variables:
            data_vars[var] = ((dim_y, dim_x), var_array_for_year(var, None))
    else:
        coords["year"] = np.asarray(years, dtype="int16")
        for var in variables:
            year_arrays = [var_array_for_year(var, y) for y in years]
            data_vars[var] = (("year", dim_y, dim_x), da.stack(year_arrays, axis=0))

    ds = xr.Dataset(data_vars, coords=coords)
    logger.debug(
        f"Opened tiled-parquet dataset at {parquet_dir}: {len(variables)} variable(s), "
        f"{'static' if years is None else f'{len(years)} year(s)'}, {n_rows}x{n_cols} tiles"
    )
    return ds

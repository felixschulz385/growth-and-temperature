"""Bbox math and DuckDB matrix-building for EASE6933 `cell_id`-keyed PREPARE
parquet output (`ix=<row>/iy=<col>/part[-<year>].parquet` trees written by
`src.data.common.prepare.driver.run_tiled_prepare`).

No matplotlib import here on purpose -- this module is the query/aggregation
half of the plotting engine (see `src/viz/plot.py` for rendering), kept
testable without a display backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import duckdb
import numpy as np
import pyproj

from src.data.assemble.constants import DEFAULT_TILE_SIZE
from src.data.common.geobox.canonical import canonical_ease_geobox
from src.data.common.geobox.cell_id import cell_tile_indices

_AGG_WHITELIST = {"mean", "median", "sum", "min", "max"}


@dataclass
class MatrixResult:
    """One year's downsampled grid, ready to hand to `imshow`."""

    matrix: np.ndarray  # (n_rows, n_cols), float64, NaN where no data
    n_obs: np.ndarray  # (n_rows, n_cols) int, raw-pixel count per output cell
    extent_lonlat: tuple[float, float, float, float]  # (min_lon, max_lon, min_lat, max_lat)


def bbox_to_row_col(bbox: tuple[float, float, float, float], geobox=None) -> tuple[int, int, int, int]:
    """Convert a WGS84 `(min_lon, min_lat, max_lon, max_lat)` bbox to a pixel
    `(row0, row1, col0, col1)` range on the canonical EASE6933 grid (`row1`/
    `col1` exclusive).

    EASE6933 is a non-sheared cylindrical projection (docs/design/01-grid.md
    §1): `x` depends only on longitude, `y` only on latitude -- so, like
    `canonical.compute_ease_bbox()`, longitude and latitude are projected
    independently rather than as paired corners.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError(f"Invalid bbox (min must be < max): {bbox}")

    if geobox is None:
        geobox = canonical_ease_geobox()

    transformer = pyproj.Transformer.from_crs("EPSG:4326", geobox.crs, always_xy=True)
    x0, _ = transformer.transform(min_lon, 0.0)
    x1, _ = transformer.transform(max_lon, 0.0)
    _, y0 = transformer.transform(0.0, max_lat)  # max lat -> smaller row (top)
    _, y1 = transformer.transform(0.0, min_lat)  # min lat -> larger row (bottom)

    inv = ~geobox.affine
    col0f, row0f = inv * (x0, y0)
    col1f, row1f = inv * (x1, y1)

    height, width = geobox.shape
    row0 = max(0, math.floor(min(row0f, row1f)))
    row1 = min(height, math.ceil(max(row0f, row1f)))
    col0 = max(0, math.floor(min(col0f, col1f)))
    col1 = min(width, math.ceil(max(col0f, col1f)))

    if row0 >= row1 or col0 >= col1:
        raise ValueError(f"bbox {bbox} does not intersect the canonical EASE6933 grid")

    return row0, row1, col0, col1


def _row_col_to_lonlat(row: float, col: float, geobox) -> tuple[float, float]:
    x, y = geobox.affine * (col, row)
    transformer = pyproj.Transformer.from_crs(geobox.crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lon, lat


def _validate_variable(con: duckdb.DuckDBPyConnection, glob: str, variable: str) -> None:
    columns = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{glob}') LIMIT 0").fetchall()
    names = {row[0] for row in columns}
    if variable not in names:
        raise ValueError(f"Column {variable!r} not found in {glob!r}; available columns: {sorted(names)}")


def load_matrix(
    input_dir: str,
    variable: str,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    years: list[int] | None = None,
    max_pixels: int = 800,
    agg: str = "mean",
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[int, MatrixResult]:
    """Query and downsample one variable from a PREPARE `cell_id` parquet
    tree into a dict of `{year: MatrixResult}`, sized to at most
    `max_pixels` on the longer axis.

    `input_dir` is the PREPARE output root for one source, e.g. what
    `src.data.sources.layout.grid_store_path(data_root, data_path,
    grid_id=EASE_GRID_ID, family=..., suffix="")` resolves to.
    """
    if agg not in _AGG_WHITELIST:
        raise ValueError(f"agg must be one of {sorted(_AGG_WHITELIST)}, got {agg!r}")

    geobox = canonical_ease_geobox()
    height, width = geobox.shape

    if bbox is not None:
        row0, row1, col0, col1 = bbox_to_row_col(bbox, geobox=geobox)
    else:
        row0, row1, col0, col1 = 0, height, 0, width

    (ix0, iy0), (ix1, iy1) = (
        cell_tile_indices(row0, col0, DEFAULT_TILE_SIZE),
        cell_tile_indices(row1 - 1, col1 - 1, DEFAULT_TILE_SIZE),
    )

    factor = max(1, math.ceil(max(row1 - row0, col1 - col0) / max_pixels))

    glob = f"{input_dir.rstrip('/')}/ix=*/iy=*/part*.parquet"
    owns_con = con is None
    if owns_con:
        con = duckdb.connect()

    try:
        _validate_variable(con, glob, variable)

        year_filter = ""
        if years is not None:
            year_list = ", ".join(str(int(y)) for y in years)
            year_filter = f"AND year IN ({year_list})"

        query = f"""
            SELECT
                (cell_id // {width}) // {factor} AS coarse_row,
                (cell_id % {width}) // {factor} AS coarse_col,
                year,
                {agg}({variable}) AS value,
                count(*) AS n
            FROM read_parquet('{glob}', hive_partitioning=true)
            WHERE ix BETWEEN {ix0} AND {ix1}
              AND iy BETWEEN {iy0} AND {iy1}
              AND (cell_id // {width}) BETWEEN {row0} AND {row1 - 1}
              AND (cell_id % {width}) BETWEEN {col0} AND {col1 - 1}
              {year_filter}
            GROUP BY 1, 2, 3
        """
        df = con.execute(query).fetchdf()
    finally:
        if owns_con:
            con.close()

    n_coarse_rows = math.ceil((row1 - row0) / factor)
    n_coarse_cols = math.ceil((col1 - col0) / factor)
    coarse_row0 = row0 // factor
    coarse_col0 = col0 // factor

    lon0, lat1 = _row_col_to_lonlat(row0, col0, geobox)
    lon1, lat0 = _row_col_to_lonlat(row1, col1, geobox)
    extent = (lon0, lon1, lat0, lat1)

    results: dict[int, MatrixResult] = {}
    for year, group in df.groupby("year"):
        matrix = np.full((n_coarse_rows, n_coarse_cols), np.nan)
        n_obs = np.zeros((n_coarse_rows, n_coarse_cols), dtype=np.int64)
        r = (group["coarse_row"].to_numpy() - coarse_row0).astype(int)
        c = (group["coarse_col"].to_numpy() - coarse_col0).astype(int)
        matrix[r, c] = group["value"].to_numpy()
        n_obs[r, c] = group["n"].to_numpy()
        results[int(year)] = MatrixResult(matrix=matrix, n_obs=n_obs, extent_lonlat=extent)

    return results

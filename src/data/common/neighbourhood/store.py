"""Zarr disc-ladder store: skeleton creation + per-tile region writes.

See docs/design/02-storage.md. This is the second of the design's two Zarr
write points (the first, analysis-ready per-source grids, is out of this
module's scope -- see docs/design/05-migration.md §5 step 6 vs. step 2). New
pattern for this repo's assembly stage: today's stage_3 output is parquet,
and Zarr is otherwise only used for intermediate stage_2 arrays.

Also holds `write_disc_tile_parquet`, a scaffolding-only sibling to
`write_disc_tile` that writes the same per-tile convolution output as a
`cell_id`-keyed parquet part instead of a Zarr region write -- see its
docstring for scope.
"""

import logging
import os
from typing import Sequence

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.xr import xr_coords
from zarr.codecs import BloscCodec

from src.data.common.geobox.cell_id import encode_cell_ids

logger = logging.getLogger(__name__)

# blosc/zstd level ~5, shuffle enabled -- starting default per
# docs/design/02-storage.md §6. Flagged there as unmeasured against real
# data (docs/design/06-open-questions.md item 7); not a final tuning.
DEFAULT_COMPRESSOR = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle", blocksize=0)


def _year_time_coords(years: Sequence[int]) -> pd.DatetimeIndex:
    return pd.to_datetime([f"{year}-12-31" for year in sorted(years)])


def _empty_disc_dataset(
    geobox,
    years: Sequence[int],
    ladder_km: Sequence[float],
    variable: str,
    dtype,
    tile_size: int,
    fill_value,
) -> xr.Dataset:
    time = _year_time_coords(years)
    ny, nx = geobox.shape
    coords = dict(xr_coords(geobox))
    coords["time"] = time
    coords["radius_km"] = list(ladder_km)
    dims = ("time", "radius_km") + geobox.dims
    shape = (len(time), len(ladder_km), ny, nx)
    chunks = (1, len(ladder_km), tile_size, tile_size)
    data = da.full(shape, fill_value, dtype=dtype, chunks=chunks)
    ds = xr.Dataset({variable: (dims, data)}, coords=coords)
    return ds


def _disc_encoding(variable: str, dtype, tile_size: int, ladder_km: Sequence[float], fill_value) -> dict:
    # `fill_value` here is the zarr-native uninitialized-chunk sentinel, set
    # via `encoding` rather than a CF `_FillValue` attribute deliberately --
    # the latter makes xarray's default `mask_and_scale=True` read path
    # silently decode integer N_d back as float and replace `fill_value`
    # (0, a legitimate "no valid neighbours" count) with NaN, which is both
    # the wrong dtype and semantically wrong for N_d.
    return {
        variable: {
            "chunks": (1, len(ladder_km), tile_size, tile_size),
            "compressors": (DEFAULT_COMPRESSOR,),
            "dtype": dtype,
            "fill_value": fill_value,
        }
    }


def _create_empty_disc_store(
    output_path,
    geobox,
    years: Sequence[int],
    ladder_km: Sequence[float],
    variable: str,
    tile_size: int,
    dtype,
    fill_value,
) -> bool:
    try:
        ds = _empty_disc_dataset(geobox, years, ladder_km, variable, dtype, tile_size, fill_value)
        encoding = _disc_encoding(variable, dtype, tile_size, ladder_km, fill_value)
        ds.to_zarr(
            str(output_path), mode="w", compute=False, encoding=encoding, zarr_format=3, consolidated=False
        )
        logger.info("Created empty disc store at %s (shape=%s)", output_path, ds[variable].shape)
        return True
    except Exception:
        logger.exception("Error creating disc store at %s", output_path)
        return False


def create_empty_disc_sum_store(
    output_path,
    geobox,
    years: Sequence[int],
    ladder_km: Sequence[float],
    variable: str = "S_d",
    tile_size: int = 2048,
    dtype: str = "float32",
    fill_value=np.nan,
) -> bool:
    """Initialize a one-store-per-variable-family S_d skeleton.

    One call per convolved variable (lights, each mediator, mine density,
    ...) -- `output_path` should be that variable's own store (e.g.
    `disc_sums/lights.zarr`), never shared across variables
    (docs/design/02-storage.md §2).
    """
    return _create_empty_disc_store(output_path, geobox, years, ladder_km, variable, tile_size, dtype, fill_value)


def create_empty_disc_count_store(
    output_path,
    geobox,
    years: Sequence[int],
    ladder_km: Sequence[float],
    variable: str = "N_d",
    tile_size: int = 2048,
    dtype: str = "uint16",
    fill_value: int = 0,
) -> bool:
    """Initialize a shared-per-mask-family N_d skeleton.

    One call per *mask family*, not per variable -- N_d depends only on the
    validity mask, so every convolved variable sharing a mask
    (docs/design/03-neighbourhood-engine.md §5's unmasked vs.
    own-country-masked split) should point at the same store rather than
    each writing its own copy (docs/design/02-storage.md §5).
    """
    return _create_empty_disc_store(output_path, geobox, years, ladder_km, variable, tile_size, dtype, fill_value)


def _stored_dtype(output_path, variable: str):
    ds = xr.open_zarr(str(output_path), consolidated=False, mask_and_scale=False)
    return ds[variable].dtype


def write_disc_tile(output_path, tile_data: xr.DataArray, year: int, variable: str, dtype=None) -> bool:
    """Write one tile's convolved output into its store region.

    `tile_data` has dims (radius_km, y, x) -- the trimmed, core-region
    output of `discs.convolve_tile` -- with coordinate values matching a
    contiguous slice of the full store's `radius_km`/y/x coordinates (true
    for any tile carved from the same canonical GeoBox via `GeoboxTiles`).
    `region="auto"` -- the same pattern
    `SpatialProcessor.write_year_to_zarr` already uses for the (different)
    analysis-ready-grid write point -- locates that slice from the
    coordinate labels alone, so no manual chunk-index bookkeeping is needed
    here.

    `convolve_discs`/`convolve_tile` always produce float64 S_d/N_d (that's
    what `scipy.signal.fftconvolve` returns, regardless of the eventual
    storage dtype) -- so this casts to the store's own declared dtype (e.g.
    `uint16` for a count store, docs/design/02-storage.md §6) before writing.
    Without this, xarray silently writes float bit patterns into an integer
    zarr array, corrupting values on read-back rather than raising.

    *dtype*, if given, skips the `_stored_dtype()` lookup (an `xr.open_zarr()`
    call that parses the whole store's metadata) entirely -- it never
    changes across tiles of the same store, so a caller writing many tiles
    in a loop should resolve it once (e.g. from `_stored_dtype()` or the
    dtype it originally created the store with) and pass it to every call,
    rather than paying for a redundant metadata parse per tile.
    """
    try:
        target_dtype = dtype if dtype is not None else _stored_dtype(output_path, variable)
        data = tile_data
        if np.dtype(target_dtype) != data.dtype:
            if np.issubdtype(np.dtype(target_dtype), np.integer):
                data = data.round()
            data = data.astype(target_dtype)

        time = pd.Timestamp(f"{year}-12-31")
        tile_out = data.expand_dims(time=[time])
        ds_out = tile_out.to_dataset(name=variable).drop_vars("spatial_ref", errors="ignore")
        ds_out.to_zarr(str(output_path), region="auto", align_chunks=True, zarr_format=3, consolidated=False)
        return True
    except Exception:
        logger.exception("Error writing tile to disc store at %s (year=%s)", output_path, year)
        return False


def write_disc_tile_parquet(
    output_dir,
    S_d: xr.DataArray,
    N_d: xr.DataArray,
    tile,
    year: int,
    full_width: int,
    compression: str = "snappy",
) -> str:
    """SCAFFOLDING ONLY -- write one tile's raw convolution output as a
    parquet part, keyed on the new global `cell_id`
    (`src/data/common/geobox/cell_id.py`), instead of a Zarr region write.

    Unlike `write_disc_tile`, this is not the production disc-store writer:
    it stores raw `S_d`/`N_d` (no ring-mean division -- `ring_means.
    ring_means_from_discs` stays unwired), the column layout below is
    provisional, and this function is not called from any pipeline CLI step
    -- full per-source PREPARE wiring and the final production schema are
    tracked as follow-up work, not done here (docs/design/03-neighbourhood-
    engine.md, docs/design/09-integrated-pipeline.md).

    Args:
        output_dir: Root directory for this convolved variable's parquet
            output (analogous to `write_disc_tile`'s `output_path`, but a
            directory of Hive-partitioned parts rather than one Zarr store).
        S_d, N_d: `convolve_tile`'s trimmed output for this tile, dims
            `(radius_km, y, x)` matching `tile.geobox.shape`.
        tile: `src.data.common.tiling.Tile` for this tile -- supplies the
            global pixel offset (`y_slice.start`/`x_slice.start`, used for
            `cell_id` encoding) and the `row`/`col` tile indices used for
            the output path (`row` -> `ix`, `col` -> `iy`, matching
            `src/data/assemble/processors.py::_get_output_path`'s
            convention -- see `cell_id.py` module docstring).
        full_width: Pixel width `W` of the full canonical grid, passed to
            `encode_cell_ids` (never hardcoded/memoized -- see `cell_id.py`).

    Returns:
        The written file path, `<output_dir>/ix=<tile.row>/iy=<tile.col>/
        part-<year>.parquet`, sorted by `(cell_id, radius_km)`.
    """
    row0, col0 = tile.y_slice.start, tile.x_slice.start
    cell_ids_2d = encode_cell_ids(row0, col0, tile.geobox, full_width)
    h, w = cell_ids_2d.shape

    radii = np.asarray(S_d.coords["radius_km"].values)
    n_r = radii.shape[0]
    S_vals = np.asarray(S_d.values).reshape(n_r, h, w)
    N_vals = np.asarray(N_d.values).reshape(n_r, h, w)

    df = pd.DataFrame(
        {
            "cell_id": np.broadcast_to(cell_ids_2d, (n_r, h, w)).reshape(-1),
            "year": year,
            "radius_km": np.repeat(radii, h * w),
            "S_d": S_vals.reshape(-1),
            "N_d": N_vals.reshape(-1),
        }
    ).sort_values(["cell_id", "radius_km"]).reset_index(drop=True)

    out_path = os.path.join(output_dir, f"ix={tile.row}", f"iy={tile.col}", f"part-{year}.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False, compression=compression, engine="pyarrow")
    return out_path

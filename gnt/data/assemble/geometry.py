from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.converters import from_geopandas
from odc.geo.geom import Geometry
from odc.geo.xr import rasterize
from shapely.geometry import box

from gnt.data.assemble.constants import (
    DEFAULT_TILE_PADDING,
    EXCLUDED_VARIABLES,
    LATITUDE_COORD,
    LONGITUDE_COORD,
    YEAR_COORD,
)
from gnt.data.assemble.tiles import (
    adjust_tile_size_for_reprojection,
    create_tile_geobox,
    get_available_tiles,
)
from gnt.data.assemble.utils import apply_column_prefix, convert_int_to_float32, winsorize

logger = logging.getLogger(__name__)


_XR_REDUCERS = {
    "mean": lambda da, dims, **kw: da.mean(dim=dims, skipna=True, **kw),
    "min": lambda da, dims, **kw: da.min(dim=dims, skipna=True, **kw),
    "max": lambda da, dims, **kw: da.max(dim=dims, skipna=True, **kw),
    "sum": lambda da, dims, **kw: da.sum(dim=dims, skipna=True, **kw),
    "median": lambda da, dims, **kw: da.median(dim=dims, skipna=True, **kw),
    "std": lambda da, dims, **kw: da.std(dim=dims, skipna=True, **kw),
    "var": lambda da, dims, **kw: da.var(dim=dims, skipna=True, **kw),
    "count": lambda da, dims, **kw: da.count(dim=dims, **kw),
}

_SUPPORTED_PARTIAL_AGGS = {"mean", "sum", "count", "min", "max", "first"}


def zonal_reduce_odc(
    ds: xr.Dataset,
    gdf: gpd.GeoDataFrame,
    agg: str | Callable[..., xr.DataArray] = "mean",
    *,
    geometry_col: str | None = None,
    id_col: str | None = None,
    keep_columns: Sequence[str] | None = None,
    all_touched: bool = True,
    crop: bool = True,
    reproject_geometries: bool = True,
    spatial_dims: tuple[str, str] | None = None,
    preserve_geometry: bool = True,
    reducer_kwargs: dict[str, Any] | None = None,
    prefer_xvec: bool = True,
) -> gpd.GeoDataFrame:
    """
    Aggregate each variable in a geo-registered xarray.Dataset over each geometry.

    This helper is kept available for ad-hoc or single-window zonal reductions.
    The assembly backend below uses tile-local partials so geometry spans across
    multiple tiles can be combined correctly after processing.
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError("`ds` must be an xarray.Dataset.")
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("`gdf` must be a geopandas.GeoDataFrame.")
    if not hasattr(ds, "odc") or ds.odc.crs is None:
        raise ValueError("`ds` must be geo-registered and expose `.odc.crs`.")
    if gdf.crs is None:
        raise ValueError("`gdf` must have a CRS.")
    if gdf.empty:
        out = gdf.copy()
        return out if preserve_geometry else pd.DataFrame(out.drop(columns=[gdf.geometry.name]))

    reducer_kwargs = dict(reducer_kwargs or {})
    geometry_col = geometry_col or gdf.geometry.name
    keep_columns = list(keep_columns or [])

    if spatial_dims is None:
        spatial_dims = ds.odc.spatial_dims
    if spatial_dims is None or len(spatial_dims) != 2:
        raise ValueError("Could not infer spatial dimensions; pass `spatial_dims=('y', 'x')`.")
    y_dim, x_dim = spatial_dims

    gdf_work = gdf.set_geometry(geometry_col)
    if gdf_work.crs != ds.odc.crs:
        if reproject_geometries:
            gdf_work = gdf_work.to_crs(ds.odc.crs)
        else:
            raise ValueError("CRS mismatch between `gdf` and `ds`.")

    if prefer_xvec:
        try:
            import xvec  # noqa: F401

            if callable(agg):
                stats = [("custom", agg, reducer_kwargs)]
                zonal = ds.xvec.zonal_stats(
                    gdf_work.geometry,
                    x_coords=x_dim,
                    y_coords=y_dim,
                    stats=stats,
                    all_touched=all_touched,
                )
            else:
                zonal = ds.xvec.zonal_stats(
                    gdf_work.geometry,
                    x_coords=x_dim,
                    y_coords=y_dim,
                    stats=agg,
                    all_touched=all_touched,
                    **reducer_kwargs,
                )

            df = zonal.to_dataframe().reset_index()
            value_cols = [c for c in df.columns if c != "geometry"]

            rename_map = {}
            for c in value_cols:
                if c.endswith("_custom"):
                    rename_map[c] = c[:-7]
            df = df.rename(columns=rename_map)

            extra_dims = [
                c
                for c in df.columns
                if c not in {"geometry"} and c not in [id_col, *keep_columns]
                and c not in ds.data_vars
                and c.replace("_custom", "") not in ds.data_vars
            ]
            if extra_dims:
                raise ValueError(
                    "xvec returned extra non-spatial dimensions. Reduce those dims first."
                )

            out = pd.DataFrame(index=gdf.index)
            if id_col is not None:
                out[id_col] = gdf[id_col].values
            for col in keep_columns:
                out[col] = gdf[col].values
            for col in df.columns:
                if col != "geometry":
                    out[col] = df[col].values

            if preserve_geometry:
                return gpd.GeoDataFrame(out, geometry=gdf[geometry_col], crs=gdf.crs)
            return out
        except ImportError:
            pass

    if isinstance(agg, str):
        if agg not in _XR_REDUCERS:
            raise ValueError(f"Unsupported agg='{agg}'. Supported: {sorted(_XR_REDUCERS)}.")
        reducer = _XR_REDUCERS[agg]
    elif callable(agg):
        reducer = lambda da, dims, **kw: agg(da, dims=dims, **kw)
    else:
        raise TypeError("`agg` must be a string or callable.")

    odc_geoms = from_geopandas(gdf_work)
    rows: list[dict[str, Any]] = []
    for idx, geom in zip(gdf.index, odc_geoms):
        subset = ds.odc.crop(geom, apply_mask=False, all_touched=all_touched) if crop else ds
        masked = subset.odc.mask(geom, all_touched=all_touched)

        row: dict[str, Any] = {}
        if id_col is not None:
            row[id_col] = gdf.loc[idx, id_col]
        for col in keep_columns:
            row[col] = gdf.loc[idx, col]

        for var_name, da in masked.data_vars.items():
            reduced = reducer(da, dims=(y_dim, x_dim), **reducer_kwargs)
            if reduced.ndim != 0:
                raise ValueError(
                    f"Reducer for variable '{var_name}' did not return a scalar."
                )
            value = reduced.item()
            if isinstance(value, np.generic):
                value = value.item()
            row[var_name] = value

        rows.append(row)

    out = pd.DataFrame(rows, index=gdf.index)
    if preserve_geometry:
        return gpd.GeoDataFrame(out, geometry=gdf[geometry_col], crs=gdf.crs)
    return out


def _select_columns(ds: xr.Dataset, columns: Optional[list[str]]) -> xr.Dataset:
    if columns:
        available = [var for var in columns if var in ds.data_vars]
        if not available:
            raise ValueError("Requested columns not found in dataset")
        return ds[available]

    vars_to_keep = [var for var in ds.data_vars.keys() if var not in EXCLUDED_VARIABLES]
    return ds[vars_to_keep]


def _extract_spatial_tile(
    ds: xr.Dataset,
    dataset_config: dict[str, Any],
    padded_tile_geobox,
    target_geobox_zoomed,
    land_mask: Optional[xr.DataArray],
) -> Optional[xr.Dataset]:
    bbox = padded_tile_geobox.boundingbox

    tile_ds = ds.sel(
        latitude=slice(bbox.top, bbox.bottom),
        longitude=slice(bbox.left, bbox.right),
    ).compute()

    if tile_ds.sizes.get(LATITUDE_COORD, 0) == 0 or tile_ds.sizes.get(LONGITUDE_COORD, 0) == 0:
        return None

    winsorize_cutoff = dataset_config.get("winsorize")
    if winsorize_cutoff is not None and winsorize_cutoff > 0:
        for var in tile_ds.data_vars:
            if np.issubdtype(tile_ds[var].dtype, np.floating):
                tile_ds[var] = winsorize(tile_ds[var], cutoff=winsorize_cutoff)

    if land_mask is not None:
        for var in tile_ds.data_vars:
            tile_ds[var] = tile_ds[var].where(land_mask)

    tile_ds = tile_ds.odc.reproject(
        target_geobox_zoomed,
        resampling=dataset_config.get("resampling", "average"),
        dst_nodata=np.nan,
    )
    tile_ds = convert_int_to_float32(tile_ds)

    if tile_ds.sizes.get(LATITUDE_COORD, 0) == 0 or tile_ds.sizes.get(LONGITUDE_COORD, 0) == 0:
        return None

    return tile_ds


def _load_land_mask_tile(
    land_mask_ds: Optional[xr.Dataset],
    padded_tile_geobox,
) -> Optional[xr.DataArray]:
    if land_mask_ds is None:
        return None

    bbox = padded_tile_geobox.boundingbox
    mask_tile = land_mask_ds.sel(
        latitude=slice(bbox.top, bbox.bottom),
        longitude=slice(bbox.left, bbox.right),
    ).compute()
    if "land_mask" not in mask_tile.data_vars:
        return None

    land_mask = mask_tile["land_mask"].astype(bool)
    if not bool(land_mask.any()):
        return None
    return land_mask


def _get_geometry_agg(dataset_config: dict[str, Any]) -> str:
    agg = dataset_config.get("geometry_agg", "mean")
    if agg not in _SUPPORTED_PARTIAL_AGGS:
        raise ValueError(
            f"Unsupported geometry_agg '{agg}'. Supported: {sorted(_SUPPORTED_PARTIAL_AGGS)}"
        )
    return agg


def _rasterize_geometry_labels(tile_geobox, tile_geometries: gpd.GeoDataFrame) -> xr.DataArray:
    labels = np.zeros(tile_geobox.shape, dtype=np.int32)
    for label_value, (_, row) in enumerate(tile_geometries.iterrows(), start=1):
        geom = Geometry(row.geometry, crs=str(tile_geometries.crs))
        mask = rasterize(geom, tile_geobox)
        labels = np.where(mask, label_value, labels)

    return xr.DataArray(
        labels,
        dims=[LATITUDE_COORD, LONGITUDE_COORD],
        coords={
            LATITUDE_COORD: tile_geobox.coords[LATITUDE_COORD].values,
            LONGITUDE_COORD: tile_geobox.coords[LONGITUDE_COORD].values,
        },
    )


def _iter_dataset_slices(ds: xr.Dataset):
    if YEAR_COORD in ds.coords:
        for year in ds.coords[YEAR_COORD].values:
            yield int(year), ds.sel({YEAR_COORD: year}).squeeze(drop=True)
    else:
        yield None, ds


def _partial_column_names(var_name: str, agg: str) -> list[str]:
    if agg == "mean":
        return [f"{var_name}__sum", f"{var_name}__weight"]
    if agg == "sum":
        return [f"{var_name}__sum"]
    if agg == "count":
        return [f"{var_name}__count"]
    if agg == "min":
        return [f"{var_name}__min"]
    if agg == "max":
        return [f"{var_name}__max"]
    if agg == "first":
        return [f"{var_name}__first"]
    raise ValueError(f"Unsupported agg '{agg}'")


def _compute_tile_partials(
    tile_ds: xr.Dataset,
    label_da: xr.DataArray,
    tile_geometries: gpd.GeoDataFrame,
    id_col: str,
    dataset_config: dict[str, Any],
) -> pd.DataFrame:
    agg = _get_geometry_agg(dataset_config)
    label_values = label_da.values.reshape(-1)
    valid_labels = label_values > 0
    label_to_id = {i + 1: row[id_col] for i, (_, row) in enumerate(tile_geometries.iterrows())}

    partial_frames: list[pd.DataFrame] = []
    for year, ds_slice in _iter_dataset_slices(tile_ds):
        series_map: dict[str, pd.Series] = {}

        for var_name, da in ds_slice.data_vars.items():
            values = da.values.reshape(-1)
            valid = valid_labels & np.isfinite(values)
            if not valid.any():
                continue

            frame = pd.DataFrame(
                {
                    "_label": label_values[valid],
                    "_value": values[valid],
                }
            )
            grouped = frame.groupby("_label")["_value"]

            if agg == "mean":
                series_map[f"{var_name}__sum"] = grouped.sum()
                series_map[f"{var_name}__weight"] = grouped.size().astype("float64")
            elif agg == "sum":
                series_map[f"{var_name}__sum"] = grouped.sum()
            elif agg == "count":
                series_map[f"{var_name}__count"] = grouped.size().astype("float64")
            elif agg == "min":
                series_map[f"{var_name}__min"] = grouped.min()
            elif agg == "max":
                series_map[f"{var_name}__max"] = grouped.max()
            elif agg == "first":
                series_map[f"{var_name}__first"] = grouped.first()

        if not series_map:
            continue

        partial = pd.DataFrame(series_map)
        partial[id_col] = partial.index.map(label_to_id)
        if year is not None:
            partial[YEAR_COORD] = year
        partial_frames.append(partial.reset_index(drop=True))

    if not partial_frames:
        cols = [id_col]
        if YEAR_COORD in tile_ds.coords:
            cols.append(YEAR_COORD)
        return pd.DataFrame(columns=cols)

    return pd.concat(partial_frames, ignore_index=True)


def _finalize_dataset_partials(
    partials: pd.DataFrame,
    dataset_name: str,
    dataset_config: dict[str, Any],
    loaded_ds: xr.Dataset,
    id_col: str,
) -> pd.DataFrame:
    agg = _get_geometry_agg(dataset_config)
    index_cols = [id_col]
    if YEAR_COORD in loaded_ds.coords:
        index_cols.append(YEAR_COORD)

    if partials.empty:
        out_cols = index_cols + list(loaded_ds.data_vars.keys())
        return pd.DataFrame(columns=out_cols)

    group = partials.groupby(index_cols, dropna=False, sort=False)
    finalized = pd.DataFrame(index=group.size().index)

    for var_name in loaded_ds.data_vars.keys():
        if agg == "mean":
            sum_col = f"{var_name}__sum"
            weight_col = f"{var_name}__weight"
            if sum_col in partials.columns and weight_col in partials.columns:
                sums = group[sum_col].sum(min_count=1)
                weights = group[weight_col].sum(min_count=1)
                finalized[var_name] = sums / weights.replace({0: np.nan})
        elif agg == "sum":
            sum_col = f"{var_name}__sum"
            if sum_col in partials.columns:
                finalized[var_name] = group[sum_col].sum(min_count=1)
        elif agg == "count":
            count_col = f"{var_name}__count"
            if count_col in partials.columns:
                finalized[var_name] = group[count_col].sum(min_count=1)
        elif agg == "min":
            min_col = f"{var_name}__min"
            if min_col in partials.columns:
                finalized[var_name] = group[min_col].min()
        elif agg == "max":
            max_col = f"{var_name}__max"
            if max_col in partials.columns:
                finalized[var_name] = group[max_col].max()
        elif agg == "first":
            first_col = f"{var_name}__first"
            if first_col in partials.columns:
                finalized[var_name] = group[first_col].first()

    finalized = finalized.reset_index()
    logger.info(
        "Finalized geometry aggregation for dataset '%s': %s rows, %s columns",
        dataset_name,
        len(finalized),
        len(finalized.columns),
    )
    return finalized


def _merge_dataset_results(
    results: list[tuple[str, pd.DataFrame, dict[str, Any]]],
    id_col: str,
) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()

    has_year = any(YEAR_COORD in df.columns for _, df, _ in results)
    all_index_cols = [id_col] + ([YEAR_COORD] if has_year else [])

    combined = pd.DataFrame(columns=all_index_cols)
    for dataset_name, df, _dataset_config in results:
        merge_cols = [col for col in all_index_cols if col in combined.columns and col in df.columns]
        if not merge_cols:
            logger.warning("Skipping dataset '%s' because no merge columns were found", dataset_name)
            continue
        combined = pd.merge(combined, df, on=merge_cols, how="outer")

    ordered_cols = [col for col in all_index_cols if col in combined.columns]
    for _, df, _ in results:
        for col in df.columns:
            if col not in ordered_cols:
                ordered_cols.append(col)
    return combined[ordered_cols]


def _existing_dataset_columns(dataset_name: str, assembly_config: dict[str, Any]) -> list[str]:
    dataset_config = assembly_config["datasets"][dataset_name]
    columns = dataset_config.get("columns")
    column_prefix = dataset_config.get("column_prefix", "")

    ds = xr.open_zarr(dataset_config["path"], consolidated=False, chunks="auto")
    ds = _select_columns(ds, columns)
    if column_prefix:
        ds = apply_column_prefix(ds, column_prefix)
    return list(ds.data_vars.keys())


def _geometry_output_index_cols(id_col: str, ds: xr.Dataset) -> list[str]:
    cols = [id_col]
    if YEAR_COORD in ds.coords:
        cols.append(YEAR_COORD)
    return cols


def assemble_geometry_weighted(
    *,
    datasets: list[tuple[str, xr.Dataset, dict[str, Any]]],
    geometry_source: dict[str, Any],
    assembly_config: dict[str, Any],
    output_path: str,
    target_geobox,
    land_mask_ds: Optional[xr.Dataset] = None,
    hpc_root: Optional[str] = None,
    full_config: Optional[dict[str, Any]] = None,
    dask_client=None,
):
    """
    Aggregate datasets along geometries by accumulating tile-local partials.

    Geometries may span multiple tiles. Each tile contributes partial sums,
    counts, or extrema per geometry, and the final dataset is computed only
    after all tiles have been processed.
    """
    del hpc_root, full_config, dask_client

    geometry_path = geometry_source["path"]
    id_col = geometry_source["id_column"]
    keep_columns = list(geometry_source.get("columns", []))
    all_touched = geometry_source.get("all_touched", True)

    gdf = gpd.read_file(geometry_path)
    if id_col not in gdf.columns:
        raise ValueError(f"Geometry source id column '{id_col}' not found in {geometry_path}")
    gdf = gdf[[id_col, *[c for c in keep_columns if c in gdf.columns], gdf.geometry.name]].copy()

    target_crs = str(target_geobox.crs)
    if gdf.crs is None:
        raise ValueError("Geometry source must have a CRS")
    if str(gdf.crs) != target_crs:
        gdf = gdf.to_crs(target_crs)

    processing_config = assembly_config.get("processing", {})
    tile_size = processing_config.get("tile_size", 2048)
    native_res = abs(target_geobox.resolution.x)
    tile_size = adjust_tile_size_for_reprojection(
        native_res,
        processing_config.get("resolution"),
        tile_size,
    )

    all_tiles = get_available_tiles(
        {"processing": {"tile_size": tile_size}},
        target_geobox,
    )

    dataset_results: list[tuple[str, pd.DataFrame, dict[str, Any]]] = []
    geometry_bounds = gdf.geometry.bounds

    for dataset_name, ds, dataset_config in datasets:
        logger.info("Geometry assembly: processing dataset '%s'", dataset_name)
        dataset_partials: list[pd.DataFrame] = []

        for ix, iy in all_tiles:
            tile_geobox = create_tile_geobox(target_geobox, tile_size, ix, iy)
            bbox = tile_geobox.boundingbox
            tile_polygon = box(bbox.left, bbox.bottom, bbox.right, bbox.top)

            overlap_mask = (
                (geometry_bounds.maxx >= bbox.left)
                & (geometry_bounds.minx <= bbox.right)
                & (geometry_bounds.maxy >= bbox.bottom)
                & (geometry_bounds.miny <= bbox.top)
            )
            overlapping = gdf.loc[overlap_mask]
            if overlapping.empty:
                continue

            overlapping = overlapping[overlapping.geometry.intersects(tile_polygon)]
            if overlapping.empty:
                continue

            padded_tile_geobox = tile_geobox.pad(DEFAULT_TILE_PADDING, DEFAULT_TILE_PADDING)
            target_resolution = processing_config.get("resolution")
            if target_resolution is not None and abs(native_res - target_resolution) >= 1e-10:
                target_geobox_zoomed = tile_geobox.zoom_to(resolution=target_resolution)
            else:
                target_geobox_zoomed = tile_geobox

            land_mask = _load_land_mask_tile(land_mask_ds, padded_tile_geobox)
            if land_mask_ds is not None and land_mask is None:
                continue

            tile_ds = _extract_spatial_tile(
                ds,
                dataset_config,
                padded_tile_geobox,
                target_geobox_zoomed,
                land_mask,
            )
            if tile_ds is None:
                continue

            tile_geometries = overlapping.copy()
            if str(tile_geometries.crs) != str(target_geobox_zoomed.crs):
                tile_geometries = tile_geometries.to_crs(str(target_geobox_zoomed.crs))

            label_da = _rasterize_geometry_labels(target_geobox_zoomed, tile_geometries)
            if not bool((label_da > 0).any().item()):
                continue

            partial = _compute_tile_partials(
                tile_ds=tile_ds,
                label_da=label_da,
                tile_geometries=tile_geometries,
                id_col=id_col,
                dataset_config=dataset_config,
            )
            if not partial.empty:
                dataset_partials.append(partial)

        if dataset_partials:
            partials_df = pd.concat(dataset_partials, ignore_index=True)
        else:
            partials_df = pd.DataFrame(columns=_geometry_output_index_cols(id_col, ds))

        finalized_df = _finalize_dataset_partials(
            partials=partials_df,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            loaded_ds=ds,
            id_col=id_col,
        )
        dataset_results.append((dataset_name, finalized_df, dataset_config))

    combined = _merge_dataset_results(dataset_results, id_col=id_col)

    base_geom = gdf[[id_col, *[c for c in keep_columns if c in gdf.columns], gdf.geometry.name]].drop_duplicates(id_col)
    combined = pd.merge(base_geom, combined, on=id_col, how="right")
    combined = gpd.GeoDataFrame(combined, geometry=gdf.geometry.name, crs=gdf.crs)

    output_file = os.path.join(output_path, "data.parquet")
    assembly_mode = processing_config.get("assembly_mode", "create")
    target_datasource = processing_config.get("datasource")

    if assembly_mode == "update" and os.path.exists(output_file):
        existing = gpd.read_parquet(output_file)
        if not target_datasource:
            raise ValueError("Geometry update mode requires 'datasource'")

        dataset_columns = _existing_dataset_columns(target_datasource, assembly_config)
        cols_to_drop = [col for col in dataset_columns if col in existing.columns]
        if cols_to_drop:
            existing = existing.drop(columns=cols_to_drop)

        merge_cols = [id_col]
        if YEAR_COORD in combined.columns and YEAR_COORD in existing.columns:
            merge_cols.append(YEAR_COORD)
        value_columns = [col for col in combined.columns if col not in {gdf.geometry.name, *merge_cols}]
        combined = existing.merge(combined[merge_cols + value_columns], on=merge_cols, how="left")
        combined = gpd.GeoDataFrame(combined, geometry=gdf.geometry.name, crs=gdf.crs)

    combined.to_parquet(output_file, index=False)
    logger.info("Geometry assembly written to %s", output_file)

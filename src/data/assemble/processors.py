"""
Tile processing functionality for data assembly.

Handles extraction, transformation, and merging of dataset tiles
with support for different resampling methods and winsorization.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import xarray as xr
import pyarrow.parquet as pq

from src.data.assemble.constants import (
    DEFAULT_RESAMPLING_METHOD,
    DEFAULT_TILE_PADDING,
    LATITUDE_COORD,
    LONGITUDE_COORD,
    EXCLUDED_VARIABLES,
)
from src.data.assemble.parquet_raster import _partitioned_parquet_files, is_tiled_parquet_dataset
from src.data.assemble.utils import (
    add_derived_pixel_id_columns,
    dataset_spatial_dims,
    make_pixel_ids,
    normalize_derived_pixel_id_specs,
    resolve_resampling,
    winsorize,
)

logger = logging.getLogger(__name__)


def _reproject_per_variable(ds: xr.Dataset, target_geobox, resampling_cfg) -> xr.Dataset:
    """Reproject *ds* onto *target_geobox*, grouping its data variables by the
    resampling method each resolves to (`resolve_resampling`), so one store can
    mix methods -- e.g. MODIS LST means via ``average`` and valid-observation
    counts via ``sum`` -- in a single downsampling pass.
    """
    var_names = [v for v in ds.data_vars if v not in EXCLUDED_VARIABLES]
    method_by_var = resolve_resampling(resampling_cfg, var_names)

    groups: Dict[str, List[str]] = {}
    for var, method in method_by_var.items():
        groups.setdefault(method, []).append(var)

    if len(groups) <= 1:
        method = next(iter(groups), DEFAULT_RESAMPLING_METHOD)
        return ds.odc.reproject(target_geobox, resampling=method, dst_nodata=np.nan)

    parts = [
        ds[group_vars].odc.reproject(target_geobox, resampling=method, dst_nodata=np.nan)
        for method, group_vars in groups.items()
    ]
    # Every part is on `target_geobox`, so coords are identical -- `override`
    # (take-first) is the correct and unambiguous combine here.
    return xr.merge(parts, compat="override", combine_attrs="override")


def get_dataset_columns(
    dataset_path: str,
    columns: Optional[List[str]] = None,
    column_prefix: str = '',
) -> List[str]:
    """
    Get column (data variable) names from a GRID-stage dataset, dispatching on
    format: a Zarr store or a `run_tiled_prepare`-produced tiled-parquet
    directory (see `src.data.assemble.parquet_raster`).

    Args:
        dataset_path: Path to the dataset (Zarr store or tiled-parquet directory)
        columns: Optional list of specific columns to select
        column_prefix: Prefix to apply to column names

    Returns:
        List of column names (with prefix applied if specified)
    """
    try:
        if is_tiled_parquet_dataset(dataset_path):
            part_files = _partitioned_parquet_files(dataset_path)
            schema_names = pq.ParquetFile(part_files[0]).schema_arrow.names
            all_vars = [
                c for c in schema_names
                if c not in EXCLUDED_VARIABLES and c not in ("cell_id", "year")
            ]
        else:
            # Open zarr to inspect variables (don't load data)
            ds = xr.open_zarr(dataset_path, consolidated=False, chunks='auto')
            all_vars = [var for var in ds.data_vars.keys() if var not in EXCLUDED_VARIABLES]
            ds.close()

        # Filter to requested columns if specified
        if columns:
            selected_vars = [var for var in columns if var in all_vars]
        else:
            selected_vars = all_vars

        # Apply prefix
        if column_prefix:
            prefixed_vars = [f"{column_prefix}{var}" for var in selected_vars]
        else:
            prefixed_vars = selected_vars

        return prefixed_vars

    except Exception as e:
        logger.warning(f"Failed to load columns from {dataset_path}: {e}")
        return []


class TileProcessor:
    """
    Processes individual tiles across multiple datasets.
    
    Handles the extraction, transformation, and merging of data for a single tile,
    supporting per-dataset resampling methods and winsorization.
    """
    
    def __init__(
        self,
        assembly_config: Dict[str, Any],
        output_base_path: str,
        target_geobox: Optional[Any] = None,
    ):
        """
        Initialize tile processor.

        Args:
            assembly_config: Assembly configuration
            output_base_path: Base path for output files (already scoped to one
                grid=<label>/shake=<label> variant)
            target_geobox: The run's target geobox -- already coarsened to the
                requested `--grid` resolution and already origin-shifted for the
                requested `--shake` variant, if any. Grid-shake is a whole-run
                origin shift now, not a per-column operation, so there is nothing
                shake-specific to do here.
        """
        self.assembly_config = assembly_config
        self.output_base_path = output_base_path
        self.processing_config = assembly_config.get('processing', {})
        self.compression = self.processing_config.get('compression', 'snappy')
        self.column_order_map = {}  # Track {dataset_name: [col1, col2, ...]}
        self.all_index_cols = self._get_all_index_cols()  # Cache all index columns
        # One in-memory DuckDB connection reused for every tile's merges --
        # this instance is already reused across the whole tile loop
        # (_process_all_tiles), so opening a fresh connection per merge call
        # would add avoidable overhead across hundreds of tiles.
        self._duckdb_con = duckdb.connect()
        self.derived_pixel_id_specs = normalize_derived_pixel_id_specs(
            self.processing_config.get("derived_pixel_ids")
        )

        # Pre-build column order map from dataset configs (raster datasets only --
        # join_on datasets are handled below, never reprojected).
        self._build_column_order_map_from_config()

        # join_on datasets: small GID-keyed tables merged directly onto assembled
        # rows by an existing GID_N column (e.g. from gadm's own GID grid dataset),
        # instead of being reprojected onto the pixel grid -- their value is
        # constant within a GID and rasterizing them would be pure waste.
        self.join_tables: Dict[str, Tuple[str, pd.DataFrame]] = self._load_join_tables()
        for dataset_name, (join_col, table) in self.join_tables.items():
            self.column_order_map[dataset_name] = [c for c in table.columns if c != join_col]

        # Log index column configuration
        logger.info(f"Unified index columns for merging: {self.all_index_cols}")
        for dataset_name, dataset_config in self.assembly_config.get('datasets', {}).items():
            idx_cols = dataset_config.get('index_cols', ['pixel_id'])
            logger.debug(f"Dataset '{dataset_name}' index_cols: {idx_cols}")
        if self.derived_pixel_id_specs:
            logger.info(
                "Derived pixel ID columns enabled: %s",
                ", ".join(f"{name}@{resolution}" for name, resolution in self.derived_pixel_id_specs),
            )
        if self.join_tables:
            logger.info(
                "Join-on datasets enabled: %s",
                ", ".join(f"{name}@{join_col}" for name, (join_col, _) in self.join_tables.items()),
            )

    def _build_column_order_map_from_config(self) -> None:
        """
        Build column_order_map by loading all datasets from config.

        This reads zarr files to get actual variable names, ensuring
        the column order map is consistent and complete.
        """
        datasets_config = self.assembly_config.get('datasets', {})

        for dataset_name, dataset_config in datasets_config.items():
            if dataset_config.get('join_on'):
                continue  # handled by _load_join_tables instead
            dataset_path = dataset_config.get('path')
            if not dataset_path:
                logger.warning(f"No path specified for dataset '{dataset_name}'")
                continue

            if not os.path.exists(dataset_path):
                logger.warning(f"Dataset path does not exist: {dataset_path}")
                continue

            # Get columns from the dataset (zarr store or tiled-parquet directory)
            columns = dataset_config.get('columns')
            column_prefix = dataset_config.get('column_prefix', '')

            dataset_cols = get_dataset_columns(dataset_path, columns, column_prefix)
            
            if dataset_cols:
                self.column_order_map[dataset_name] = dataset_cols
                logger.debug(f"Initialized column order for '{dataset_name}': {dataset_cols}")
            else:
                logger.warning(f"No columns found for dataset '{dataset_name}'")
    
    def _get_output_path(self, ix: int, iy: int) -> str:
        """Get output file path for a tile."""
        return os.path.join(
            self.output_base_path, 
            f"ix={ix}", 
            f"iy={iy}", 
            "data.parquet"
        )
    
    def _tile_exists_and_valid(self, output_file: str) -> bool:
        """Check if tile output already exists and is valid."""
        if not os.path.exists(output_file):
            return False
        
        try:
            parquet_file = pq.ParquetFile(output_file)
            return parquet_file.metadata.num_rows > 0
        except Exception as e:
            logger.warning(f"Tile exists but appears corrupted ({e}), will reprocess")
            return False

    def _get_fillna_config(self, dataset_name: str, dataset_config: Dict[str, Any]) -> Optional[Any]:
        """Return configured fill behavior, including legacy defaults."""
        fillna_config = dataset_config.get("fillna")
        if fillna_config is None and dataset_name == "snl_mining":
            fillna_config = 0
        return fillna_config

    def _fill_dataset_columns(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        dataset_config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Fill configured datasource columns after dataframe alignment/merges."""
        fillna_config = self._get_fillna_config(dataset_name, dataset_config)
        if fillna_config is None or df is None or df.empty:
            return df

        data_cols = [
            col for col in self.column_order_map.get(dataset_name, [])
            if col in df.columns
        ]
        if not data_cols:
            return df

        if isinstance(fillna_config, dict):
            column_prefix = dataset_config.get("column_prefix", "")
            for var_name, fill_value in fillna_config.items():
                candidate_cols = []
                if column_prefix:
                    candidate_cols.append(f"{column_prefix}{var_name}")
                candidate_cols.append(var_name)
                target_cols = [
                    col for col in candidate_cols
                    if col in data_cols and col in df.columns
                ]
                if target_cols:
                    df.loc[:, target_cols] = df.loc[:, target_cols].fillna(fill_value)
            return df

        df.loc[:, data_cols] = df.loc[:, data_cols].fillna(fillna_config)
        return df
    
    def _create_tile_geoboxes(self, tile_geobox) -> Tuple[Any, Optional[Any]]:
        """
        Create padded and target resolution geoboxes for tile processing.
        
        Returns:
            Tuple of (padded_geobox, target_geobox_zoomed or None)
        """
        target_resolution = self.processing_config.get('resolution')
        native_res = abs(tile_geobox.resolution.x)
        
        # Create padded geobox for edge handling during reprojection
        padded_tile_geobox = tile_geobox.pad(DEFAULT_TILE_PADDING, DEFAULT_TILE_PADDING)
        
        # Create target resolution geobox if needed
        if target_resolution is not None and abs(native_res - target_resolution) >= 1e-10:
            logger.debug(f"Will reproject from {native_res} to {target_resolution} (target units)")
            target_geobox_zoomed = tile_geobox.zoom_to(resolution=target_resolution)
        else:
            target_geobox_zoomed = tile_geobox

        return padded_tile_geobox, target_geobox_zoomed

    def _load_join_tables(self) -> Dict[str, Tuple[str, pd.DataFrame]]:
        """Load every `join_on`-configured dataset as a small in-memory table.

        Returns {dataset_name: (join_column, table)}. These tables are tiny
        (one row per GID, not per pixel) so loading them fully at init time is
        cheap and lets every tile's merge be a plain in-memory pandas join.
        """
        join_tables: Dict[str, Tuple[str, pd.DataFrame]] = {}
        for dataset_name, dataset_config in self.assembly_config.get('datasets', {}).items():
            join_col = dataset_config.get('join_on')
            if not join_col:
                continue

            path = dataset_config.get('path')
            if not path or not os.path.exists(path):
                logger.warning(f"join_on dataset '{dataset_name}': path not found: {path}")
                continue

            table = pd.read_parquet(path, columns=dataset_config.get('columns'))
            if join_col not in table.columns:
                logger.warning(
                    f"join_on dataset '{dataset_name}': table at {path} has no '{join_col}' "
                    f"column, skipping"
                )
                continue

            column_prefix = dataset_config.get('column_prefix')
            if column_prefix:
                table = table.rename(
                    columns={c: f"{column_prefix}{c}" for c in table.columns if c != join_col}
                )

            if table[join_col].duplicated().any():
                logger.warning(
                    f"join_on dataset '{dataset_name}': duplicate '{join_col}' values, "
                    f"keeping the first occurrence of each"
                )
                table = table.drop_duplicates(subset=[join_col], keep='first')

            join_tables[dataset_name] = (join_col, table)
            logger.info(
                f"Loaded join table '{dataset_name}': {len(table)} rows keyed by '{join_col}', "
                f"columns: {[c for c in table.columns if c != join_col]}"
            )
        return join_tables

    def _apply_join_tables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge every configured join_on dataset onto assembled rows by an
        existing GID_N column, instead of reprojecting them onto the pixel
        grid -- their value is constant within a GID (docs/design/04-ingest.md).
        """
        for dataset_name, (join_col, table) in self.join_tables.items():
            if join_col not in df.columns:
                logger.warning(
                    f"join_on dataset '{dataset_name}': column '{join_col}' not present in "
                    f"assembled rows -- add a dataset that provides it (e.g. gadm's GID grid) "
                    f"to this assembly config. Skipping."
                )
                continue

            value_cols = [c for c in table.columns if c != join_col]
            drop_cols = [c for c in value_cols if c in df.columns]
            if drop_cols:
                df = df.drop(columns=drop_cols)

            df = df.merge(table, on=join_col, how='left')
            dataset_config = self.assembly_config['datasets'][dataset_name]
            df = self._fill_dataset_columns(df, dataset_name, dataset_config)
        return df

    def _extract_dataset_tile(
        self,
        ds: xr.Dataset,
        dataset_config: Dict[str, Any],
        ix: int,
        iy: int,
        padded_tile_geobox,
        target_geobox_zoomed,
        pixel_id_ds: Optional[xr.Dataset],
        land_mask: Optional[xr.DataArray] = None,
        keep_spatial_coords: bool = False,
        include_pixel_id: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Extract and process a single dataset tile.

        Processing pipeline:
        1. Extract tile from padded bounds
        2. Apply winsorization if configured
        3. Apply land mask at native resolution (using xarray .where())
        4. Reproject to the target geobox (already coarsened/origin-shifted for
           the run's --grid / --shake variant)
        5. Assign pixel_id variable
        6. Convert to DataFrame
        7. Drop NaN rows

        Args:
            land_mask: Optional boolean DataArray for masking pixels at native resolution (True=land, False=ocean)

        Returns:
            DataFrame with pixel_id, or None if tile is empty
        """
        resampling_cfg = dataset_config.get('resampling')

        try:
            bbox = padded_tile_geobox.boundingbox

            # Extract tile data with padding. Spatial dim names come from the
            # dataset's own grid (`latitude`/`longitude` on a geographic CRS,
            # `y`/`x` on EASE 6933), not a hardcoded assumption.
            dim_y, dim_x = dataset_spatial_dims(ds) or (None, None)
            if dim_y is None:
                logger.warning("Dataset %s has no recognizable spatial dims", ds.attrs.get('dataset_name', '?'))
                return None
            tile_ds = ds.sel({
                dim_y: slice(bbox.top, bbox.bottom),
                dim_x: slice(bbox.left, bbox.right),
            }).compute()

            # Check for empty tile
            if tile_ds.sizes.get(dim_y, 0) == 0 or tile_ds.sizes.get(dim_x, 0) == 0:
                return None
            
            # Apply winsorization before reprojection
            winsorize_cutoff = dataset_config.get('winsorize')
            if winsorize_cutoff is not None and winsorize_cutoff > 0:
                for var in tile_ds.data_vars:
                    if np.issubdtype(tile_ds[var].dtype, np.floating):
                        tile_ds[var] = winsorize(tile_ds[var], cutoff=winsorize_cutoff)
                logger.debug(f"Applied winsorization with cutoff={winsorize_cutoff}")
            
            # Apply land mask at native resolution before reprojection
            if land_mask is not None:
                for var in tile_ds.data_vars:
                    tile_ds[var] = tile_ds[var].where(land_mask)
            
            # Reproject to the target geobox if it differs from native. The
            # target geobox already carries the run's --grid coarsening and the
            # --shake origin shift (applied once, up front, in run_assembly), so
            # there is nothing shake-specific to do per column here. Variables
            # are grouped by their resolved resampling method (per-variable via
            # the dataset's `resampling` map) so e.g. means and counts in one
            # store downsample correctly in the same pass.
            if target_geobox_zoomed is not None and hasattr(tile_ds, 'odc'):
                tile_ds = _reproject_per_variable(tile_ds, target_geobox_zoomed, resampling_cfg)

            # Assign pixel_id when requested for pixel-partitioned assemblies.
            if include_pixel_id:
                if pixel_id_ds is None:
                    raise ValueError("pixel_id dataset is required when include_pixel_id=True")
                tile_ds = tile_ds.assign(pixel_id=pixel_id_ds['pixel_id'])
            
            # Convert to DataFrame, preserving all coordinates as columns (including year)
            df = tile_ds.to_dataframe().reset_index()
            
            # Drop spatial coordinate columns (their names depend on the grid's CRS).
            drop_cols = ['band', 'spatial_ref', LATITUDE_COORD, LONGITUDE_COORD, 'y', 'x']
            if keep_spatial_coords:
                drop_cols = ['band', 'spatial_ref']
            df = df.drop(columns=drop_cols, errors='ignore')
            
            # Drop rows where all data columns are NaN (from land mask filtering)
            if land_mask is not None:
                # Keep index columns, drop rows where all non-index columns are NaN
                data_cols = [col for col in df.columns if col not in ['pixel_id', 'year']]
                if data_cols:
                    df = df.dropna(subset=data_cols, how='all')
            
            return df if not df.empty else None
            
        except Exception as e:
            logger.warning(f"Failed to extract tile [{ix}, {iy}]: {e}")
            return None
    
    def _duckdb_join(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        merge_cols: List[str],
        how: str,
    ) -> pd.DataFrame:
        """Join two DataFrames on `merge_cols` via DuckDB, matching
        `pd.merge(left, right, on=merge_cols, how=how)`'s row-order and
        merge-key semantics (`how` is 'outer' or 'left').

        Column order is NOT reproduced here -- callers of this helper always
        feed into `_reorder_columns`, which is applied once to the final
        combined result before writing (not after every pairwise merge), so
        intermediate column order is immaterial.
        """
        con = self._duckdb_con
        row_ord_col = "__row_ord__"
        left_ordered = left.reset_index(drop=True)
        left_ordered = left_ordered.assign(**{row_ord_col: np.arange(len(left_ordered))})

        con.register("_left", left_ordered)
        con.register("_right", right)
        try:
            # `IS NOT DISTINCT FROM`, not `=`: a plain `=` join drops rows
            # with a NULL merge key on either side, but pandas' outer/left
            # merges keep them (NaN keys round-trip as unmatched rows).
            on_clause = " AND ".join(
                f'_left."{c}" IS NOT DISTINCT FROM _right."{c}"' for c in merge_cols
            )
            if how == "outer":
                join_kw = "FULL OUTER JOIN"
                key_select = [f'COALESCE(_left."{c}", _right."{c}") AS "{c}"' for c in merge_cols]
            elif how == "left":
                join_kw = "LEFT JOIN"
                key_select = [f'_left."{c}" AS "{c}"' for c in merge_cols]
            else:
                raise ValueError(f"Unsupported join type: {how!r}")

            left_extra = [c for c in left.columns if c not in merge_cols]
            right_extra = [c for c in right.columns if c not in merge_cols]
            select_cols = (
                key_select
                + [f'_left."{c}"' for c in left_extra]
                + [f'_right."{c}"' for c in right_extra]
                + [f'_left."{row_ord_col}"']
            )
            query = f'SELECT {", ".join(select_cols)} FROM _left {join_kw} _right ON {on_clause}'
            result = con.sql(query).df()
        finally:
            con.unregister("_left")
            con.unregister("_right")

        # Preserve left-frame row order (SQL joins don't guarantee output
        # order the way pd.merge does). Rows that only exist on the right
        # side (outer join) have a NULL row-order and sort after every
        # left-originated row -- pd.merge(how='outer') doesn't guarantee a
        # specific relative order for those either, only that left rows
        # keep their original relative order.
        result = (
            result.sort_values(row_ord_col, kind="stable", na_position="last")
            .drop(columns=[row_ord_col])
            .reset_index(drop=True)
        )

        # DuckDB's Arrow round-trip can shift a column's dtype at the
        # margins (e.g. int64 -> float64) differently than pandas would --
        # cast back to the pre-join dtype where the values still fit and no
        # nulls were introduced, so callers see the same dtype contract as
        # the pre-swap pandas merge.
        for col in result.columns:
            prior_dtype = left[col].dtype if col in left.columns else (
                right[col].dtype if col in right.columns else None
            )
            if prior_dtype is not None and result[col].dtype != prior_dtype and not result[col].isna().any():
                try:
                    result[col] = result[col].astype(prior_dtype)
                except (TypeError, ValueError):
                    pass

        return result

    def _merge_dataframes(
        self,
        combined: pd.DataFrame,
        df: pd.DataFrame,
        dataset_name: str,
        ix: int,
        iy: int,
    ) -> pd.DataFrame:
        """
        Merge a new DataFrame into the combined result.

        Uses all available common index columns from the unified set for merging.
        Uses outer join to preserve all rows, with land mask filtering applied later.
        """
        # Find common merge columns from the unified index set
        # Use all index columns that are present in both dataframes
        merge_cols = [col for col in self.all_index_cols if col in combined.columns and col in df.columns]

        if not merge_cols:
            logger.warning(
                f"Tile [{ix}, {iy}]: {dataset_name} - no common columns found for merge. "
                f"all_index_cols: {self.all_index_cols}, combined: {list(combined.columns)}, df: {list(df.columns)}"
            )
            return combined

        # _duckdb_join has no pd.merge-style _x/_y suffixing for colliding
        # non-key column names -- drop any non-merge column `df` also
        # provides from `combined` first so the two frames never collide.
        df_cols = [col for col in df.columns if col not in merge_cols]
        cols_to_drop = [col for col in df_cols if col in combined.columns]
        if cols_to_drop:
            logger.debug(f"Tile [{ix}, {iy}]: {dataset_name} - dropping existing columns: {cols_to_drop}")
            combined = combined.drop(columns=cols_to_drop)

        rows_before = len(combined)
        combined = self._duckdb_join(combined, df, merge_cols, how="outer")
        logger.debug(
            f"Tile [{ix}, {iy}]: {dataset_name} - merged on {merge_cols}, "
            f"rows: {rows_before} -> {len(combined)}"
        )

        return combined

    def _combine_dataset_tables(
        self,
        tables: List[Tuple[str, pd.DataFrame]],
        ix: Optional[int] = None,
        iy: Optional[int] = None,
    ) -> pd.DataFrame:
        """Combine dataset tables using the shared index-column merge policy."""
        combined: Optional[pd.DataFrame] = None
        pending_empty_tables: List[Tuple[str, pd.DataFrame]] = []

        def merge_into_combined(name: str, table: pd.DataFrame) -> None:
            nonlocal combined
            if combined is None:
                combined = table.copy()
                return
            tile_ix = -1 if ix is None else ix
            tile_iy = -1 if iy is None else iy
            combined = self._merge_dataframes(combined, table, name, tile_ix, tile_iy)

        for dataset_name, df in tables:
            if df is None:
                continue

            if df.empty and combined is None:
                pending_empty_tables.append((dataset_name, df))
                continue

            merge_into_combined(dataset_name, df)
            for empty_dataset_name, empty_df in pending_empty_tables:
                merge_into_combined(empty_dataset_name, empty_df)
            pending_empty_tables = []

        if combined is None:
            return self._combine_empty_dataset_tables(pending_empty_tables)
        return combined

    def _combine_empty_dataset_tables(
        self,
        tables: List[Tuple[str, pd.DataFrame]],
    ) -> pd.DataFrame:
        """Return a schema-only table when every datasource table is empty."""
        columns = list(self.all_index_cols)
        for _, df in tables:
            for col in df.columns:
                if col not in columns:
                    columns.append(col)
        return pd.DataFrame(columns=columns)

    def _merge_update_table(
        self,
        existing_df: pd.DataFrame,
        update_df: pd.DataFrame,
        update_index_cols: List[str],
        context: str,
    ) -> Optional[pd.DataFrame]:
        """Replace updated datasource columns in an existing table and left-merge the new values."""
        merge_cols = [
            col for col in update_index_cols
            if col in existing_df.columns and col in update_df.columns
        ]
        if not merge_cols:
            logger.error(
                f"{context}: no common index columns found for merge. "
                f"index_cols: {update_index_cols}, existing: {list(existing_df.columns)}, "
                f"new: {list(update_df.columns)}"
            )
            return None

        update_cols = [col for col in update_df.columns if col not in update_index_cols]
        cols_to_drop = [col for col in update_cols if col in existing_df.columns]
        if cols_to_drop:
            logger.debug(f"{context}: dropping existing columns: {cols_to_drop}")
            existing_df = existing_df.drop(columns=cols_to_drop)

        logger.info(f"{context}: merging on index columns: {merge_cols}")
        return self._duckdb_join(existing_df, update_df, merge_cols, how="left")
    
    def _reorder_columns(
        self,
        df: pd.DataFrame,
        index_cols: List[str],
        dataset_order: List[str],
    ) -> pd.DataFrame:
        """
        Reorder DataFrame columns based on dataset order in config.
        
        Order: index columns first, then data columns by dataset order (as in config),
        with within-dataset order preserved from zarr.
        
        Args:
            df: DataFrame to reorder
            index_cols: List of index column names (e.g., ['pixel_id', 'year'])
            dataset_order: List of dataset names in config order
            
        Returns:
            DataFrame with reordered columns
        """
        # Start with index columns that exist in the dataframe
        ordered_cols = [col for col in index_cols if col in df.columns]
        derived_cols = [col for col, _ in self.derived_pixel_id_specs if col in df.columns]
        ordered_cols.extend([col for col in derived_cols if col not in ordered_cols])
        
        # Add data columns by dataset order
        for dataset_name in dataset_order:
            if dataset_name in self.column_order_map:
                # Add columns from this dataset in their original order
                for col in self.column_order_map[dataset_name]:
                    if col in df.columns and col not in ordered_cols:
                        ordered_cols.append(col)
        
        # Don't add any remaining columns not tracked (shouldn't happen, but be safe)
        for col in df.columns:
            if col not in ordered_cols:
                ordered_cols.append(col)
                logger.debug(f"Found untracked column: {col}")
        
        return df[ordered_cols]

    def _apply_derived_pixel_id_columns(
        self,
        df: pd.DataFrame,
        ix: int,
        iy: int,
        tile_geobox,
        source_geobox,
    ) -> pd.DataFrame:
        """Append configured derived pixel ID columns to a tile dataframe."""
        if not self.derived_pixel_id_specs:
            return df
        return add_derived_pixel_id_columns(
            df=df,
            ix=ix,
            iy=iy,
            base_tile_geobox=tile_geobox,
            source_geobox=source_geobox,
            derived_specs=self.derived_pixel_id_specs,
        )
    
    def _get_all_index_cols(self) -> List[str]:
        """
        Get unified list of all index columns from all datasets in config.
        
        Returns:
            List of unique index column names across all datasets
        """
        datasets_config = self.assembly_config.get('datasets', {})
        all_index_cols = []
        
        for dataset_config in datasets_config.values():
            index_cols = dataset_config.get('index_cols', ['pixel_id'])
            for col in index_cols:
                if col not in all_index_cols:
                    all_index_cols.append(col)
        
        return all_index_cols
    
    def _load_land_mask_as_dataarray(
        self,
        land_mask_ds: Optional[xr.Dataset],
        ix: int,
        iy: int,
        padded_tile_geobox,
        target_geobox_zoomed,
    ) -> Optional[xr.DataArray]:
        """
        Load and prepare land mask as a boolean xarray DataArray.
        
        Extracts the land mask tile at native resolution and returns as boolean DataArray.
        The mask is NOT reprojected; masking is applied before reprojection in _extract_dataset_tile.
        Returns None if mask cannot be loaded. Logs and skips tile if no land pixels are found.
        
        Args:
            land_mask_ds: Land mask xarray Dataset
            ix, iy: Tile indices for logging
            padded_tile_geobox: Padded geobox for tile extraction
            target_geobox_zoomed: Target resolution geobox (not used for reprojection here)
            
        Returns:
            Boolean xarray DataArray at native resolution, or None if loading fails or no land pixels
        """
        if land_mask_ds is None:
            return None
        
        logger.debug(f"Tile [{ix}, {iy}]: loading land_mask as boolean raster")
        
        try:
            bbox = padded_tile_geobox.boundingbox

            # Extract land mask tile (spatial dim names follow the mask's own grid).
            dim_y, dim_x = dataset_spatial_dims(land_mask_ds) or (None, None)
            if dim_y is None:
                logger.warning(f"Tile [{ix}, {iy}]: land_mask has unknown coordinate system")
                return None
            mask_tile = land_mask_ds.sel({
                dim_y: slice(bbox.top, bbox.bottom),
                dim_x: slice(bbox.left, bbox.right),
            }).compute()

            if mask_tile is None or mask_tile.sizes.get(dim_y, 0) == 0:
                logger.debug(f"Tile [{ix}, {iy}]: land_mask tile is empty")
                return None
            
            # Extract boolean land mask DataArray at native resolution
            if 'land_mask' not in mask_tile.data_vars:
                logger.warning(f"Tile [{ix}, {iy}]: 'land_mask' variable not found")
                return None
            
            land_mask = mask_tile['land_mask'].astype(bool)
            
            # Quick check: if no land pixels, skip tile entirely
            if not land_mask.any():
                logger.debug(f"Tile [{ix}, {iy}]: no land pixels found, skipping tile")
                return None
            
            land_pixel_count = int(land_mask.sum())
            logger.debug(f"Tile [{ix}, {iy}]: found {land_pixel_count} land pixels")
            
            return land_mask
            
        except Exception as e:
            logger.warning(f"Tile [{ix}, {iy}]: failed to load land_mask: {e}")
            return None
    
    def _process_pixel_tile_update(
        self,
        datasets: List[Tuple[str, xr.Dataset, Dict[str, Any]]],
        land_mask_ds: Optional[xr.Dataset],
        ix: int,
        iy: int,
        tile_geobox,
        output_file: str,
    ) -> bool:
        """
        Process tile in UPDATE mode: load existing, merge new datasource data, write back.
        
        Args:
            datasets: List of (name, xr.Dataset, config) tuples (should contain only target datasource)
            land_mask_ds: Optional land mask (not used in update mode)
            ix, iy: Tile indices
            tile_geobox: Target geobox for tile
            output_file: Path to existing tile file
            
        Returns:
            True if tile was updated successfully, False otherwise
        """
        if not os.path.exists(output_file):
            logger.warning(f"Tile ix={ix}, iy={iy} does not exist, cannot update")
            return False
        
        # Load existing tile
        try:
            existing_df = pd.read_parquet(output_file)
            logger.debug(f"Tile [{ix}, {iy}]: loaded existing data with {len(existing_df)} rows")
        except Exception as e:
            logger.error(f"Failed to load existing tile [{ix}, {iy}]: {e}")
            return False
        
        # Note: all_index_cols and column_order_map are already built in __init__
        
        # Create geoboxes
        padded_tile_geobox, target_geobox_zoomed = self._create_tile_geoboxes(tile_geobox)

        # Create pixel IDs
        pixel_id_ds = make_pixel_ids(ix, iy, target_geobox_zoomed)
        if pixel_id_ds is None or min(pixel_id_ds.sizes.values(), default=0) == 0:
            logger.warning(f"Failed to create pixel_id for tile [{ix}, {iy}]")
            return False

        # Process only the target datasource
        if not datasets:
            logger.error(f"No datasource provided for update mode")
            return False

        dataset_name, ds, dataset_config = datasets[0]
        logger.info(f"Tile [{ix}, {iy}]: updating datasource '{dataset_name}'")

        # Get index_cols for this specific dataset for logging
        dataset_index_cols = dataset_config.get('index_cols', ['pixel_id'])
        logger.debug(f"Tile [{ix}, {iy}]: '{dataset_name}' configured index_cols: {dataset_index_cols}")

        df = self._extract_dataset_tile(
            ds, dataset_config, ix, iy,
            padded_tile_geobox, target_geobox_zoomed, pixel_id_ds,
        )
        
        if df is None or df.empty:
            logger.warning(f"Tile [{ix}, {iy}]: no data extracted for '{dataset_name}'")
            return False
        
        logger.debug(
            f"Tile [{ix}, {iy}]: '{dataset_name}' - "
            f"extracted {len(df)} rows, {len(df.columns)} columns"
        )
        
        context = f"Tile [{ix}, {iy}] update '{dataset_name}'"
        combined = self._merge_update_table(existing_df, df, self.all_index_cols, context)
        if combined is None:
            return False
        combined = self._fill_dataset_columns(combined, dataset_name, dataset_config)
        combined = self._apply_join_tables(combined)
        combined = self._apply_derived_pixel_id_columns(
            combined,
            ix=ix,
            iy=iy,
            tile_geobox=tile_geobox,
            source_geobox=target_geobox_zoomed,
        )
        
        rows_before = len(existing_df)
        logger.info(
            f"Tile [{ix}, {iy}]: updated '{dataset_name}' - "
            f"rows: {rows_before} -> {len(combined)}, columns: {len(existing_df.columns)} -> {len(combined.columns)}"
        )
        
        # Reorder columns based on complete dataset order from config
        datasets_config = self.assembly_config.get('datasets', {})
        dataset_order = list(datasets_config.keys())
        combined = self._reorder_columns(combined, self.all_index_cols, dataset_order)
        logger.debug(f"Tile [{ix}, {iy}]: reordered columns to: {list(combined.columns)}")
        
        # Write updated tile
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        combined.reset_index(drop=True).to_parquet(
            output_file, 
            index=False, 
            compression=self.compression, 
            engine='pyarrow'
        )
        logger.info(f"Tile [{ix}, {iy}]: updated tile written to {output_file}")
        
        return True

    def _process_pixel_tile_create(
        self,
        datasets: List[Tuple[str, xr.Dataset, Dict[str, Any]]],
        land_mask_ds: Optional[xr.Dataset],
        ix: int,
        iy: int,
        tile_geobox,
        output_file: str,
    ) -> bool:
        """
        Process tile in CREATE mode: extract all datasets, merge, and write.
        
        Args:
            datasets: List of (name, xr.Dataset, config) tuples
            land_mask_ds: Optional land mask for filtering
            ix, iy: Tile indices
            tile_geobox: Target geobox for tile
            output_file: Path to output tile file
            
        Returns:
            True if tile was processed successfully, False if no data
        """
        # Create geoboxes
        padded_tile_geobox, target_geobox_zoomed = self._create_tile_geoboxes(tile_geobox)

        # Create pixel IDs
        pixel_id_ds = make_pixel_ids(ix, iy, target_geobox_zoomed)
        if pixel_id_ds is None or min(pixel_id_ds.sizes.values(), default=0) == 0:
            logger.warning(f"Failed to create pixel_id for tile [{ix}, {iy}]")
            return False

        # Returns None if tile has no land pixels (early exit)
        land_mask = self._load_land_mask_as_dataarray(
            land_mask_ds, ix, iy,
            padded_tile_geobox, target_geobox_zoomed
        )
        
        # If land mask was needed but not found, skip tile
        if land_mask_ds is not None and land_mask is None:
            return False
        
        dataset_tables: List[Tuple[str, pd.DataFrame]] = []
        
        # Process each dataset
        for dataset_name, ds, dataset_config in datasets:
            logger.debug(f"Tile [{ix}, {iy}]: processing '{dataset_name}'")
            
            df = self._extract_dataset_tile(
                ds, dataset_config, ix, iy,
                padded_tile_geobox, target_geobox_zoomed, pixel_id_ds,
                land_mask=land_mask,
            )

            # If no data, create skeleton with NaN columns
            if df is None or df.empty:
                df = pd.DataFrame(columns=self.all_index_cols + self.column_order_map[dataset_name])
                logger.debug(
                    f"Tile [{ix}, {iy}]: '{dataset_name}' - "
                    f"no data, created skeleton with {len(self.column_order_map[dataset_name])} NaN columns"
                )
            else:
                df = self._fill_dataset_columns(df, dataset_name, dataset_config)
                logger.debug(
                    f"Tile [{ix}, {iy}]: '{dataset_name}' - "
                    f"extracted {len(df)} rows, {len(df.columns)} columns"
                )
            
            dataset_tables.append((dataset_name, df))

        combined = self._combine_dataset_tables(dataset_tables, ix, iy)
        for dataset_name, _, dataset_config in datasets:
            combined = self._fill_dataset_columns(combined, dataset_name, dataset_config)
        combined = self._apply_join_tables(combined)
        combined = self._apply_derived_pixel_id_columns(
            combined,
            ix=ix,
            iy=iy,
            tile_geobox=tile_geobox,
            source_geobox=target_geobox_zoomed,
        )

        # Check for empty result
        if combined is None or combined.empty:
            logger.debug(f"No data in tile ix={ix}, iy={iy}")
            return False

        # Reorder columns based on dataset order in config
        dataset_order = [name for name, _, _ in datasets] + list(self.join_tables.keys())
        combined = self._reorder_columns(combined, self.all_index_cols, dataset_order)
        logger.debug(f"Tile [{ix}, {iy}]: reordered columns to: {list(combined.columns)}")
        
        # Write to parquet
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        combined.reset_index(drop=True).to_parquet(
            output_file, 
            index=False, 
            compression=self.compression, 
            engine='pyarrow'
        )
        logger.info(f"Tile [{ix}, {iy}]: written {len(combined)} rows to {output_file}")
        
        return True
    
    def process_tile(
        self,
        datasets: List[Tuple[str, xr.Dataset, Dict[str, Any]]],
        land_mask_ds: Optional[xr.Dataset],
        ix: int,
        iy: int,
        tile_geobox,
    ) -> bool:
        """
        Process a single tile across all datasets and write to parquet.
        
        Workflow (CREATE mode - default):
        1. Create geoboxes for processing
        2. Generate pixel IDs
        3. Extract and merge each dataset
        4. Apply land mask filter
        5. Write result to parquet
        
        Workflow (UPDATE mode):
        1. Load existing tile from parquet
        2. Create geoboxes for processing
        3. Generate pixel IDs
        4. Extract only the specified datasource
        5. Merge/replace columns in existing data
        6. Write updated result back to parquet
        
        Args:
            datasets: List of (name, xr.Dataset, config) tuples
            land_mask_ds: Optional land mask for filtering
            ix, iy: Tile indices
            tile_geobox: Target geobox for tile
            
        Returns:
            True if tile was processed successfully, False if no data
        """
        logger.debug(f"Processing tile ix={ix}, iy={iy}")

        output_file = self._get_output_path(ix, iy)
        assembly_mode = self.processing_config.get('assembly_mode', 'create')

        # Route to appropriate mode handler
        if assembly_mode == 'update':
            return self._process_pixel_tile_update(
                datasets, land_mask_ds, ix, iy, tile_geobox, output_file
            )
        else:
            return self._process_pixel_tile_create(
                datasets, land_mask_ds, ix, iy, tile_geobox, output_file
            )

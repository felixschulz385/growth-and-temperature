"""
Tile processing functionality for data assembly.

Handles extraction, transformation, and merging of dataset tiles
with support for different resampling methods and winsorization.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import pyarrow.parquet as pq

from gnt.data.assemble.constants import (
    DEFAULT_TILE_PADDING,
    LATITUDE_COORD,
    LONGITUDE_COORD,
    EXCLUDED_VARIABLES,
)
from gnt.data.assemble.utils import winsorize, make_pixel_ids

logger = logging.getLogger(__name__)


def uses_geometry_aggregation(assembly_config: Dict[str, Any]) -> bool:
    """Return True when the assembly writes geometry-level output instead of ix/iy pixel tiles."""
    processing_config = assembly_config.get('processing', {})
    return bool(
        assembly_config.get('geometry_aggregator')
        or assembly_config.get('geometry_source')
        or processing_config.get('spatial_partition') == 'geometry'
    )


def get_dataset_columns_from_zarr(
    zarr_path: str,
    columns: Optional[List[str]] = None,
    column_prefix: str = '',
) -> List[str]:
    """
    Get column names from a zarr file.
    
    Args:
        zarr_path: Path to zarr file
        columns: Optional list of specific columns to select
        column_prefix: Prefix to apply to column names
        
    Returns:
        List of column names (with prefix applied if specified)
    """
    try:
        # Open zarr to inspect variables (don't load data)
        ds = xr.open_zarr(zarr_path, consolidated=False, chunks='auto')
        
        # Get all data variables (exclude coordinates and excluded vars)
        all_vars = [var for var in ds.data_vars.keys() if var not in EXCLUDED_VARIABLES]
        
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
        
        ds.close()
        return prefixed_vars
        
    except Exception as e:
        logger.warning(f"Failed to load columns from {zarr_path}: {e}")
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
    ):
        """
        Initialize tile processor.
        
        Args:
            assembly_config: Assembly configuration
            output_base_path: Base path for output files
        """
        self.assembly_config = assembly_config
        self.output_base_path = output_base_path
        self.processing_config = assembly_config.get('processing', {})
        self.compression = self.processing_config.get('compression', 'snappy')
        self.uses_geometry_aggregation = uses_geometry_aggregation(assembly_config)
        self.column_order_map = {}  # Track {dataset_name: [col1, col2, ...]}
        self.all_index_cols = self._get_all_index_cols()  # Cache all index columns
        
        # Pre-build column order map from dataset configs
        self._build_column_order_map_from_config()
        
        # Log index column configuration
        logger.info(f"Unified index columns for merging: {self.all_index_cols}")
        for dataset_name, dataset_config in self.assembly_config.get('datasets', {}).items():
            idx_cols = dataset_config.get('index_cols', ['pixel_id'])
            logger.debug(f"Dataset '{dataset_name}' index_cols: {idx_cols}")
    
    def _build_column_order_map_from_config(self) -> None:
        """
        Build column_order_map by loading all datasets from config.
        
        This reads zarr files to get actual variable names, ensuring
        the column order map is consistent and complete.
        """
        datasets_config = self.assembly_config.get('datasets', {})
        
        for dataset_name, dataset_config in datasets_config.items():
            zarr_path = dataset_config.get('path')
            if not zarr_path:
                logger.warning(f"No path specified for dataset '{dataset_name}'")
                continue
                
            if not os.path.exists(zarr_path):
                logger.warning(f"Dataset path does not exist: {zarr_path}")
                continue
            
            # Get columns from zarr file
            columns = dataset_config.get('columns')
            column_prefix = dataset_config.get('column_prefix', '')
            
            dataset_cols = get_dataset_columns_from_zarr(zarr_path, columns, column_prefix)
            
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
            logger.debug(f"Will reproject from {native_res}° to {target_resolution}°")
            target_geobox_zoomed = tile_geobox.zoom_to(resolution=target_resolution)
        else:
            target_geobox_zoomed = tile_geobox
        
        return padded_tile_geobox, target_geobox_zoomed
    
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
        4. Reproject to target resolution if needed
        5. Assign pixel_id variable
        6. Convert to DataFrame
        7. Drop NaN rows
        
        Args:
            land_mask: Optional boolean DataArray for masking pixels at native resolution (True=land, False=ocean)
        
        Returns:
            DataFrame with pixel_id, or None if tile is empty
        """
        resampling_method = dataset_config.get('resampling', 'mode')
        
        try:
            bbox = padded_tile_geobox.boundingbox
            
            # Extract tile data with padding
            if LATITUDE_COORD in ds.coords and LONGITUDE_COORD in ds.coords:
                tile_ds = ds.sel(
                    latitude=slice(bbox.top, bbox.bottom),
                    longitude=slice(bbox.left, bbox.right)
                ).compute()
            else:
                logger.warning(f"Unknown coordinate system in dataset")
                return None
            
            # Check for empty tile
            if tile_ds.sizes.get(LATITUDE_COORD, 0) == 0 or tile_ds.sizes.get(LONGITUDE_COORD, 0) == 0:
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
            
            # Reproject to target resolution if needed
            if target_geobox_zoomed is not None and hasattr(tile_ds, 'odc'):
                tile_ds = tile_ds.odc.reproject(
                    target_geobox_zoomed,
                    resampling=resampling_method,
                    dst_nodata=np.nan
                )
            
            # Assign pixel_id when requested for pixel-partitioned assemblies.
            if include_pixel_id:
                if pixel_id_ds is None:
                    raise ValueError("pixel_id dataset is required when include_pixel_id=True")
                tile_ds = tile_ds.assign(pixel_id=pixel_id_ds['pixel_id'])
            
            # Convert to DataFrame, preserving all coordinates as columns (including year)
            df = tile_ds.to_dataframe().reset_index()
            
            # Drop spatial coordinates unless a geometry-aggregation path needs them.
            drop_cols = ['band', 'spatial_ref']
            if not keep_spatial_coords:
                drop_cols.extend([LATITUDE_COORD, LONGITUDE_COORD])
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
        
        rows_before = len(combined)
        combined = pd.merge(combined, df, on=merge_cols, how='outer')
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
        return pd.merge(existing_df, update_df, on=merge_cols, how='left')
    
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
                #ordered_cols.append(col)
                logger.debug(f"Found untracked column: {col}")
        
        return df[ordered_cols]
    
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
            
            # Extract land mask tile
            if LATITUDE_COORD in land_mask_ds.coords and LONGITUDE_COORD in land_mask_ds.coords:
                mask_tile = land_mask_ds.sel(
                    latitude=slice(bbox.top, bbox.bottom),
                    longitude=slice(bbox.left, bbox.right)
                ).compute()
            else:
                logger.warning(f"Tile [{ix}, {iy}]: land_mask has unknown coordinate system")
                return None
            
            if mask_tile is None or mask_tile.sizes.get(LATITUDE_COORD, 0) == 0:
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
            if self.uses_geometry_aggregation:
                logger.error(
                    f"Tile [{ix}, {iy}]: geometry-aggregated update should use the top-level parquet "
                    f"update path, not ix/iy tile files"
                )
                return False
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
        if pixel_id_ds is None or pixel_id_ds.sizes.get(LATITUDE_COORD, 0) == 0:
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
            padded_tile_geobox, target_geobox_zoomed, pixel_id_ds
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

    def _get_geometry_output_path(self) -> str:
        """Return the top-level parquet file used by geometry-aggregated assemblies."""
        return os.path.join(self.output_base_path, "data.parquet")

    def _get_geometry_update_output_path(self) -> str:
        """Backward-compatible alias for the geometry output path."""
        return self._get_geometry_output_path()

    def _get_geometry_group_columns(self, dataset_config: Dict[str, Any], geometry_id_column: str) -> List[str]:
        dataset_index_cols = dataset_config.get('index_cols', [geometry_id_column])
        group_cols = [geometry_id_column]
        for col in dataset_index_cols:
            if col != geometry_id_column and col not in group_cols:
                group_cols.append(col)
        return group_cols

    def _get_geometry_agg_func(self, dataset_config: Dict[str, Any]) -> str:
        agg = str(dataset_config.get('geometry_agg', 'mean')).lower()
        return {'avg': 'mean', 'average': 'mean', 'med': 'median'}.get(agg, agg)

    def _can_preaggregate_geometry(self, agg_func: str) -> bool:
        """Return True when a tile-level partial aggregation can be combined exactly."""
        return agg_func in {'mean', 'sum', 'min', 'max', 'first', 'last', 'count'}

    def _load_geometry_source(self) -> Tuple[gpd.GeoDataFrame, str]:
        """Load geometry polygons for geometry-aggregated assemblies."""
        geometry_source = self.assembly_config.get('geometry_source', {})
        geometry_path = geometry_source.get('path')
        geometry_id_column = geometry_source.get('id_column')

        if not geometry_path or not geometry_id_column:
            raise ValueError("Geometry-aggregated assembly requires geometry_source.path and geometry_source.id_column")

        geometry_gdf = gpd.read_file(geometry_path)
        if geometry_id_column not in geometry_gdf.columns:
            raise ValueError(f"Geometry source missing id column '{geometry_id_column}'")

        geometry_gdf = geometry_gdf[[geometry_id_column, 'geometry']].copy()
        if geometry_gdf.crs is None:
            geometry_gdf = geometry_gdf.set_crs("EPSG:4326")
        else:
            geometry_gdf = geometry_gdf.to_crs("EPSG:4326")
        return geometry_gdf, geometry_id_column

    def _create_geometry_grid_cell_dataframe(
        self,
        tile_df: pd.DataFrame,
        dataset_config: Dict[str, Any],
        geometry_gdf: gpd.GeoDataFrame,
        geometry_id_column: str,
        variable_cols: List[str],
        ix: int,
        iy: int,
    ) -> Optional[pd.DataFrame]:
        """Spatially join tile grid-cell rows to polygons and keep index/data columns."""
        if tile_df is None or tile_df.empty:
            return None

        if LATITUDE_COORD not in tile_df.columns or LONGITUDE_COORD not in tile_df.columns:
            raise ValueError("Geometry aggregation requires latitude/longitude columns in extracted tile data")

        geometry_group_cols = self._get_geometry_group_columns(dataset_config, geometry_id_column)
        data_cols = [
            col for col in variable_cols
            if col in tile_df.columns and col not in geometry_group_cols
        ]
        if not data_cols:
            return None

        points = gpd.GeoDataFrame(
            tile_df,
            geometry=gpd.points_from_xy(tile_df[LONGITUDE_COORD], tile_df[LATITUDE_COORD]),
            crs="EPSG:4326",
        )

        joined = gpd.sjoin(
            points,
            geometry_gdf,
            how='inner',
            predicate='within',
        ).drop(columns=['geometry', 'index_right'], errors='ignore')

        if joined.empty:
            logger.debug(f"Tile [{ix}, {iy}]: no geometry matches found for aggregated datasource")
            return None

        keep_cols = geometry_group_cols + data_cols
        missing_group_cols = [col for col in geometry_group_cols if col not in joined.columns]
        if missing_group_cols:
            raise ValueError(
                f"Geometry aggregation missing index columns after spatial join: {missing_group_cols}"
            )

        grid_cells = joined[keep_cols].copy()
        logger.debug(
            f"Tile [{ix}, {iy}]: joined {len(grid_cells)} grid-cell rows to geometry "
            f"with columns {keep_cols}"
        )
        return grid_cells

    def _preaggregate_geometry_tile_dataframe(
        self,
        grid_cell_df: pd.DataFrame,
        geometry_group_cols: List[str],
        data_cols: List[str],
        agg_func: str,
    ) -> pd.DataFrame:
        """Reduce a tile's geometry-matched grid-cell rows before appending them."""
        if agg_func == 'mean':
            grouped = grid_cell_df.groupby(geometry_group_cols, dropna=False)
            sums = grouped[data_cols].sum(min_count=1)
            counts = grouped[data_cols].count()
            sums.columns = [f"{col}__sum" for col in sums.columns]
            counts.columns = [f"{col}__count" for col in counts.columns]
            return pd.concat([sums, counts], axis=1).reset_index()

        return (
            grid_cell_df
            .groupby(geometry_group_cols, dropna=False)[data_cols]
            .agg(agg_func)
            .reset_index()
        )

    def _finalize_geometry_preaggregation(
        self,
        tile_tables: List[pd.DataFrame],
        geometry_group_cols: List[str],
        data_cols: List[str],
        agg_func: str,
    ) -> pd.DataFrame:
        """Combine pre-aggregated tile tables into final geometry-level datasource values."""
        rows = pd.concat(tile_tables, ignore_index=True)

        if agg_func == 'mean':
            sum_cols = [f"{col}__sum" for col in data_cols]
            count_cols = [f"{col}__count" for col in data_cols]
            grouped = rows.groupby(geometry_group_cols, dropna=False)
            summed = grouped[sum_cols + count_cols].sum(min_count=1).reset_index()
            for col in data_cols:
                summed[col] = summed[f"{col}__sum"] / summed[f"{col}__count"]
            return summed[geometry_group_cols + data_cols]

        if agg_func == 'count':
            return (
                rows
                .groupby(geometry_group_cols, dropna=False)[data_cols]
                .sum(min_count=1)
                .reset_index()
            )

        return (
            rows
            .groupby(geometry_group_cols, dropna=False)[data_cols]
            .agg(agg_func)
            .reset_index()
        )

    def _aggregate_geometry_rows(
        self,
        grid_cell_tables: List[pd.DataFrame],
        geometry_group_cols: List[str],
        data_cols: List[str],
        agg_func: str,
    ) -> pd.DataFrame:
        """Aggregate geometry rows, using tile pre-aggregation when exact for the aggregation."""
        grid_cell_rows = pd.concat(grid_cell_tables, ignore_index=True)
        return (
            grid_cell_rows
            .groupby(geometry_group_cols, dropna=False)[data_cols]
            .agg(agg_func)
            .reset_index()
        )

    def _build_geometry_dataset_table(
        self,
        dataset_name: str,
        ds: xr.Dataset,
        dataset_config: Dict[str, Any],
        geometry_gdf: gpd.GeoDataFrame,
        geometry_id_column: str,
        land_mask_ds: Optional[xr.Dataset],
        all_tiles: List[Tuple[int, int]],
        target_geobox,
    ) -> Tuple[Optional[pd.DataFrame], int, int]:
        """Build one geometry-level datasource table from all grid-cell rows."""
        from gnt.data.assemble.tiles import create_tile_geobox

        geometry_group_cols = self._get_geometry_group_columns(dataset_config, geometry_id_column)
        tile_size = self.processing_config.get('tile_size')
        variable_cols = self.column_order_map.get(dataset_name, list(ds.data_vars.keys()))
        agg_func = self._get_geometry_agg_func(dataset_config)
        can_preaggregate = self._can_preaggregate_geometry(agg_func)
        grid_cell_tables: List[pd.DataFrame] = []
        preaggregated_tables: List[pd.DataFrame] = []
        processed_count = 0
        skipped_count = 0

        for ix, iy in all_tiles:
            logger.debug(f"Geometry aggregation for '{dataset_name}': processing tile ix={ix}, iy={iy}")
            tile_geobox = create_tile_geobox(target_geobox, tile_size, ix, iy)
            padded_tile_geobox, target_geobox_zoomed = self._create_tile_geoboxes(tile_geobox)

            land_mask = self._load_land_mask_as_dataarray(
                land_mask_ds, ix, iy, padded_tile_geobox, target_geobox_zoomed
            )
            if land_mask_ds is not None and land_mask is None:
                logger.debug(
                    f"Geometry aggregation tile [{ix}, {iy}] for '{dataset_name}': "
                    f"skipped because no usable land-mask data was available"
                )
                skipped_count += 1
                continue

            df = self._extract_dataset_tile(
                ds,
                dataset_config,
                ix,
                iy,
                padded_tile_geobox,
                target_geobox_zoomed,
                pixel_id_ds=None,
                land_mask=land_mask,
                keep_spatial_coords=True,
                include_pixel_id=False,
            )
            if df is None or df.empty:
                logger.debug(
                    f"Geometry aggregation tile [{ix}, {iy}] for '{dataset_name}': no source rows extracted"
                )
                skipped_count += 1
                continue

            grid_cell_df = self._create_geometry_grid_cell_dataframe(
                df, dataset_config, geometry_gdf, geometry_id_column, variable_cols, ix, iy
            )
            if grid_cell_df is None or grid_cell_df.empty:
                logger.debug(
                    f"Geometry aggregation tile [{ix}, {iy}] for '{dataset_name}': no joined grid-cell rows"
                )
                skipped_count += 1
                continue

            data_cols = [
                col for col in variable_cols
                if col in grid_cell_df.columns and col not in geometry_group_cols
            ]
            if not data_cols:
                logger.debug(
                    f"Geometry aggregation tile [{ix}, {iy}] for '{dataset_name}': no data columns"
                )
                skipped_count += 1
                continue

            if can_preaggregate:
                tile_table = self._preaggregate_geometry_tile_dataframe(
                    grid_cell_df, geometry_group_cols, data_cols, agg_func
                )
                preaggregated_tables.append(tile_table)
                output_row_count = len(tile_table)
            else:
                grid_cell_tables.append(grid_cell_df)
                output_row_count = len(grid_cell_df)

            processed_count += 1
            logger.info(
                f"Geometry aggregation tile [{ix}, {iy}] for '{dataset_name}': "
                f"{len(grid_cell_df)} grid-cell rows -> {output_row_count} "
                f"{'pre-aggregated' if can_preaggregate else 'raw'} rows"
            )

        if not preaggregated_tables and not grid_cell_tables:
            logger.warning(f"No geometry grid-cell rows produced for datasource '{dataset_name}'")
            return None, processed_count, skipped_count

        source_rows = pd.concat(preaggregated_tables or grid_cell_tables, ignore_index=True)
        data_cols = [
            col for col in variable_cols
            if col in source_rows.columns and col not in geometry_group_cols
        ]
        if agg_func == 'mean' and can_preaggregate:
            data_cols = [
                col for col in variable_cols
                if f"{col}__sum" in source_rows.columns and f"{col}__count" in source_rows.columns
            ]
        if not data_cols:
            logger.warning(f"No data columns available for geometry datasource '{dataset_name}'")
            return None, processed_count, skipped_count

        if can_preaggregate:
            aggregated = self._finalize_geometry_preaggregation(
                preaggregated_tables, geometry_group_cols, data_cols, agg_func
            )
        else:
            aggregated = self._aggregate_geometry_rows(
                grid_cell_tables, geometry_group_cols, data_cols, agg_func
            )
        logger.info(
            f"Geometry aggregation complete for '{dataset_name}': aggregated "
            f"{len(source_rows)} {'pre-aggregated' if can_preaggregate else 'grid-cell'} rows "
            f"to {len(aggregated)} rows using {geometry_group_cols}"
        )
        return aggregated, processed_count, skipped_count

    def process_geometry_output(
        self,
        datasets: List[Tuple[str, xr.Dataset, Dict[str, Any]]],
        land_mask_ds: Optional[xr.Dataset],
        all_tiles: List[Tuple[int, int]],
        target_geobox,
    ) -> Tuple[int, int]:
        """
        Create or update geometry-aggregated output.

        CREATE aggregates all configured datasources and writes the top-level table.
        UPDATE aggregates the requested datasource and merges it into the existing table.
        """
        assembly_mode = self.processing_config.get('assembly_mode', 'create')
        if not datasets:
            raise ValueError("Geometry assembly requires at least one datasource to be loaded")
        if assembly_mode == 'update' and len(datasets) != 1:
            raise ValueError("Geometry update mode requires exactly one datasource to be loaded")

        output_file = self._get_geometry_output_path()
        if assembly_mode == 'update' and not os.path.exists(output_file):
            raise FileNotFoundError(
                f"Geometry-aggregated update target not found: {output_file}"
            )
        if assembly_mode == 'create' and os.path.exists(output_file) and not self.processing_config.get('overwrite', True):
            logger.info(f"Geometry output already exists, skipping because overwrite=False: {output_file}")
            return 0, len(all_tiles)

        geometry_gdf, geometry_id_column = self._load_geometry_source()
        dataset_tables: List[Tuple[str, pd.DataFrame]] = []
        processed_count = 0
        skipped_count = 0

        for dataset_name, ds, dataset_config in datasets:
            table, dataset_processed, dataset_skipped = self._build_geometry_dataset_table(
                dataset_name,
                ds,
                dataset_config,
                geometry_gdf,
                geometry_id_column,
                land_mask_ds,
                all_tiles,
                target_geobox,
            )
            processed_count += dataset_processed
            skipped_count += dataset_skipped
            if table is not None and not table.empty:
                dataset_tables.append((dataset_name, table))

        if not dataset_tables:
            logger.warning("No geometry-level rows produced")
            return 0, skipped_count

        if assembly_mode == 'update':
            dataset_name, _, dataset_config = datasets[0]
            if len(dataset_tables) != 1:
                raise ValueError("Geometry update mode requires exactly one datasource table")
            existing_df = pd.read_parquet(output_file)
            update_df = dataset_tables[0][1]
            update_index_cols = self._get_geometry_group_columns(dataset_config, geometry_id_column)
            merged = self._merge_update_table(
                existing_df,
                update_df,
                update_index_cols,
                f"Geometry update '{dataset_name}'",
            )
            if merged is None:
                return 0, skipped_count
        else:
            dataset_order = [name for name, _ in dataset_tables]
            merged = self._combine_dataset_tables(dataset_tables)
            merged = self._reorder_columns(merged, self.all_index_cols, dataset_order)

        os.makedirs(self.output_base_path, exist_ok=True)
        merged.to_parquet(
            output_file,
            index=False,
            compression=self.compression,
            engine='pyarrow',
        )
        logger.info(
            f"Geometry {assembly_mode} complete: wrote {len(merged)} rows to top-level table {output_file}"
        )
        return processed_count, skipped_count

    def update_geometry_aggregated_output(
        self,
        datasets: List[Tuple[str, xr.Dataset, Dict[str, Any]]],
        land_mask_ds: Optional[xr.Dataset],
        all_tiles: List[Tuple[int, int]],
        target_geobox,
    ) -> Tuple[int, int]:
        """Backward-compatible wrapper for the geometry update entry point."""
        return self.process_geometry_output(datasets, land_mask_ds, all_tiles, target_geobox)
    
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
        if pixel_id_ds is None or pixel_id_ds.sizes.get(LATITUDE_COORD, 0) == 0:
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
                land_mask=land_mask
            )
            
            # If no data, create skeleton with NaN columns
            if df is None or df.empty:
                df = pd.DataFrame(columns=self.all_index_cols + self.column_order_map[dataset_name])
                logger.debug(
                    f"Tile [{ix}, {iy}]: '{dataset_name}' - "
                    f"no data, created skeleton with {len(self.column_order_map[dataset_name])} NaN columns"
                )
            else:
                logger.debug(
                    f"Tile [{ix}, {iy}]: '{dataset_name}' - "
                    f"extracted {len(df)} rows, {len(df.columns)} columns"
                )
            
            dataset_tables.append((dataset_name, df))

        combined = self._combine_dataset_tables(dataset_tables, ix, iy)
        
        # Check for empty result
        if combined is None or combined.empty:
            logger.debug(f"No data in tile ix={ix}, iy={iy}")
            return False
        
        # Reorder columns based on dataset order in config
        dataset_order = [name for name, _, _ in datasets]
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

        if self.uses_geometry_aggregation:
            raise ValueError(
                "Geometry-aggregated assemblies must use process_geometry_output, "
                "not per-pixel tile processing"
            )
        
        # Route to appropriate mode handler
        if assembly_mode == 'update':
            return self._process_pixel_tile_update(
                datasets, land_mask_ds, ix, iy, tile_geobox, output_file
            )
        else:
            return self._process_pixel_tile_create(
                datasets, land_mask_ds, ix, iy, tile_geobox, output_file
            )

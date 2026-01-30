"""
Tile processing functionality for data assembly.

Handles extraction, transformation, and merging of dataset tiles
with support for different resampling methods and winsorization.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from odc.geo.xr import ODCExtensionDa
import pyarrow.parquet as pq

from gnt.data.assemble.constants import (
    DEFAULT_TILE_PADDING,
    LATITUDE_COORD,
    LONGITUDE_COORD,
    EXCLUDED_VARIABLES,
)
from gnt.data.assemble.utils import winsorize, make_pixel_ids

logger = logging.getLogger(__name__)


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
        pixel_id_ds: xr.Dataset,
    ) -> Optional[pd.DataFrame]:
        """
        Extract and process a single dataset tile.
        
        Processing pipeline:
        1. Extract tile from padded bounds
        2. Apply winsorization if configured
        3. Reproject to target resolution if needed
        4. Assign pixel_id variable
        5. Convert to DataFrame
        
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
            
            # Reproject to target resolution if needed
            if target_geobox_zoomed is not None and hasattr(tile_ds, 'odc'):
                tile_ds = tile_ds.odc.reproject(
                    target_geobox_zoomed,
                    resampling=resampling_method,
                    dst_nodata=np.nan
                )
            
            # Assign pixel_id
            tile_ds = tile_ds.assign(pixel_id=pixel_id_ds['pixel_id'])
            
            # Convert to DataFrame, preserving all coordinates as columns (including year)
            df = tile_ds.to_dataframe().reset_index()
            
            # Drop spatial coordinates but keep index coordinates like year
            df = df.drop(
                columns=['band', 'spatial_ref', LATITUDE_COORD, LONGITUDE_COORD], 
                errors='ignore'
            )
            
            return df if not df.empty else None
            
        except Exception as e:
            logger.warning(f"Failed to extract tile [{ix}, {iy}]: {e}")
            return None
    
    def _merge_dataframes(
        self,
        combined: Optional[pd.DataFrame],
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
        if combined is None:
            logger.debug(f"Tile [{ix}, {iy}]: {dataset_name} - initialized combined DataFrame")
            return df
        
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
    
    def _apply_land_mask(
        self,
        combined: pd.DataFrame,
        land_mask_ds: Optional[xr.Dataset],
        ix: int,
        iy: int,
        padded_tile_geobox,
        target_geobox_zoomed,
        pixel_id_ds: xr.Dataset,
    ) -> pd.DataFrame:
        """
        Apply land mask filter using right join.
        
        Filters combined data to only include land pixels.
        """
        if land_mask_ds is None:
            return combined
        
        logger.debug(f"Tile [{ix}, {iy}]: processing land_mask")
        
        mask_df = self._extract_dataset_tile(
            land_mask_ds, 
            {'winsorize': None, 'resampling': 'nearest'}, 
            ix, iy,
            padded_tile_geobox, 
            target_geobox_zoomed, 
            pixel_id_ds
        )
        
        if mask_df is None or 'land_mask' not in mask_df.columns:
            logger.debug(f"Tile [{ix}, {iy}]: land_mask extraction failed")
            return combined
        
        # Filter to land pixels only
        mask_df = mask_df[mask_df.land_mask.astype(bool)]
        mask_df = mask_df.drop(columns=['land_mask'], errors='ignore')
        
        rows_before = len(combined)
        combined = pd.merge(
            combined, 
            mask_df[['pixel_id']], 
            on=['pixel_id'], 
            how='right'
        )
        logger.debug(
            f"Tile [{ix}, {iy}]: land_mask filter applied, "
            f"rows: {rows_before} -> {len(combined)}"
        )
        
        return combined
    
    def _process_tile_update_mode(
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
        
        # Use unified index columns for merge (intersection of what's available)
        merge_cols = [col for col in self.all_index_cols if col in existing_df.columns and col in df.columns]
        if not merge_cols:
            logger.error(
                f"Tile [{ix}, {iy}]: no common index columns found for merge. "
                f"all_index_cols: {self.all_index_cols}, existing: {list(existing_df.columns)}, new: {list(df.columns)}"
            )
            return False
        
        logger.info(f"Tile [{ix}, {iy}]: merging on index columns: {merge_cols}")
        
        # Identify columns to drop from existing data (excluding index columns)
        cols_to_drop = [col for col in df.columns if col not in all_index_cols and col in existing_df.columns]
        if cols_to_drop:
            logger.debug(f"Tile [{ix}, {iy}]: dropping existing columns: {cols_to_drop}")
            existing_df = existing_df.drop(columns=cols_to_drop)
        
        # Merge new data with existing (left join to preserve all existing rows)
        rows_before = len(existing_df)
        combined = pd.merge(existing_df, df, on=merge_cols, how='left')
        
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
    
    def _process_tile_create_mode(
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
        
        # Process each dataset
        combined = None
        for dataset_name, ds, dataset_config in datasets:
            logger.debug(f"Tile [{ix}, {iy}]: processing '{dataset_name}'")
            
            df = self._extract_dataset_tile(
                ds, dataset_config, ix, iy,
                padded_tile_geobox, target_geobox_zoomed, pixel_id_ds
            )
            
            if df is not None and not df.empty:
                logger.debug(
                    f"Tile [{ix}, {iy}]: '{dataset_name}' - "
                    f"extracted {len(df)} rows, {len(df.columns)} columns"
                )
                combined = self._merge_dataframes(
                    combined, df, dataset_name, ix, iy
                )
        
        # Apply land mask filter
        if combined is not None:
            combined = self._apply_land_mask(
                combined, land_mask_ds, ix, iy,
                padded_tile_geobox, target_geobox_zoomed, pixel_id_ds
            )
        
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
        
        # Route to appropriate mode handler
        if assembly_mode == 'update':
            return self._process_tile_update_mode(
                datasets, land_mask_ds, ix, iy, tile_geobox, output_file
            )
        else:
            return self._process_tile_create_mode(
                datasets, land_mask_ds, ix, iy, tile_geobox, output_file
            )


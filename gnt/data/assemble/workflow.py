"""
Data assembly module for the GNT system.

This module provides functionality to merge multiple datasets using a tile-by-tile
approach based on configuration specifications, reading directly from zarr files
and outputting to parquet format with integer packing preservation.
"""

import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import xarray as xr
from odc.geo import GeoboxTiles
from zarr.codecs import BloscCodec

# Import common utilities
from gnt.data.common.geobox import get_or_create_geobox

logger = logging.getLogger(__name__)

def _load_land_mask(hpc_root: str, land_mask_path: str = None) -> Optional[xr.Dataset]:
    """
    Load land mask from zarr file.
    
    Args:
        hpc_root: HPC root directory
        land_mask_path: Explicit path to land mask, or None to auto-detect
        
    Returns:
        Land mask dataset or None if not found
    """
    try:
        if land_mask_path is None:
            # Auto-detect land mask path from misc preprocessor output
            potential_paths = [
                os.path.join(hpc_root, "misc", "processed", "stage_2", "osm", "land_mask.zarr"),
            ]
        else:
            potential_paths = [land_mask_path]
        
        for path in potential_paths:
            if os.path.exists(path):
                logger.info(f"Loading land mask from: {path}")
                land_mask_ds = xr.open_zarr(path, consolidated=False)["land_mask"]
                return land_mask_ds
        
        logger.warning(f"Land mask not found in any of: {potential_paths}")
        return None
        
    except Exception as e:
        logger.warning(f"Error loading land mask: {e}")
        return None

def _apply_land_mask_to_tile(tile_ds: xr.Dataset, land_mask_ds: xr.Dataset, bbox) -> Optional[xr.Dataset]:
    """
    Apply land mask to filter out water pixels from tile dataset.
    
    Args:
        tile_ds: Tile dataset to filter
        land_mask_ds: Land mask dataset
        bbox: Tile bounding box
        
    Returns:
        Filtered dataset or None if no land pixels
    """
    try:
        # Extract land mask for this tile
        if 'latitude' in land_mask_ds.coords and 'longitude' in land_mask_ds.coords:
            tile_land_mask = land_mask_ds.sel(
                latitude=slice(bbox.top, bbox.bottom),
                longitude=slice(bbox.left, bbox.right)
            )
            tile_land_mask = tile_land_mask.compute()
        else:
            logger.warning("Land mask coordinates don't match tile coordinates")
            return tile_ds
        
        # Check if there are any land pixels
        if tile_land_mask.isnull().all():
            return None
        
        # Add mask to data variables
        tile_ds = tile_ds.merge(tile_land_mask, join='exact')
        
        return tile_ds
        
    except Exception as e:
        logger.warning(f"Error applying land mask: {e}")
        return tile_ds

def get_available_tiles(assembly_config: Dict[str, Any], target_geobox) -> List[Tuple[int, int]]:
    """Get all available tile combinations (ix, iy) using geobox tiling."""
    tile_size = assembly_config.get('processing', {}).get('tile_size', 2048)
    tiles = GeoboxTiles(target_geobox, (tile_size, tile_size))
    
    # Return all possible tile indices
    all_tiles = []
    for ix in range(tiles.shape[0]):
        for iy in range(tiles.shape[1]):
            all_tiles.append((ix, iy))
    
    logger.info(f"Generated {len(all_tiles)} tiles from geobox ({tiles.shape[0]}x{tiles.shape[1]})")
    return all_tiles

def load_tile_data_from_zarr(
    dataset_config: Dict[str, Any], 
    ix: int, 
    iy: int, 
    tile_geobox,
    land_mask_ds: Optional[xr.Dataset] = None
) -> Optional[pd.DataFrame]:
    """Load data for a specific tile from zarr dataset."""
    zarr_path = dataset_config['path']
    
    if not os.path.exists(zarr_path):
        logger.warning(f"Zarr path does not exist: {zarr_path}")
        return None
    
    try:
        logger.debug(f"Loading tile [{ix}, {iy}] from {zarr_path}")
        
        # Open zarr dataset
        ds = xr.open_zarr(zarr_path, mask_and_scale=False, consolidated=False)
        
        columns = dataset_config.get('columns')
        if not columns:
            columns = [var for var in ds.data_vars.keys() if var != 'spatial_ref']
        
        # Select specific variables
        ds = ds[columns]
        
        # Extract tile bounds
        bbox = tile_geobox.boundingbox
        
        # Slice dataset to tile bounds
        if 'latitude' in ds.coords and 'longitude' in ds.coords:
            tile_ds = ds.sel(
                latitude=slice(bbox.top, bbox.bottom),
                longitude=slice(bbox.left, bbox.right)
            ).compute()
        else:
            logger.warning(f"Unknown coordinate system for {zarr_path}")
            return None
        
        # Check if tile has data
        if tile_ds.sizes.get('latitude', 0) == 0 or tile_ds.sizes.get('longitude', 0) == 0:
            logger.debug(f"No spatial data in tile [{ix}, {iy}] for {zarr_path}")
            return None
        
        # Apply land mask if provided
        if land_mask_ds is not None:
            tile_ds = _apply_land_mask_to_tile(tile_ds, land_mask_ds, bbox)
            if tile_ds is None:
                logger.debug(f"No land pixels in tile [{ix}, {iy}] for {zarr_path}")
                return None
        
        # Add pixel ID information for tile
        tile_size_actual = (tile_ds.sizes['latitude'], tile_ds.sizes['longitude'])
        pixel_id_matrix = np.arange(tile_size_actual[0] * tile_size_actual[1], dtype="int32").reshape(tile_size_actual)
        tile_ds = tile_ds.assign({
            'pixel_id': (['latitude', 'longitude'], pixel_id_matrix),
            'tile_ix': ix,
            'tile_iy': iy
        })
        
        # Transform time to year if present
        if "time" in tile_ds.coords:
            tile_ds.coords["time"] = pd.Series(tile_ds.coords["time"]).dt.year.astype("int16")
            
        # Convert to DataFrame
        df = tile_ds.to_dataframe().reset_index()
        
        # Apply land mask filtering if present
        if 'land_mask' in df.columns:
            df = df[df.land_mask]
            df = df.drop(columns=["land_mask"])
        
        if df.empty:
            logger.debug(f"No data remaining in tile [{ix}, {iy}] for {zarr_path}")
            return None
        
        # Set index columns as specified in config
        index_cols = dataset_config.get('index_cols', ['pixel_id'])
        if 'time' in df.columns and 'time' not in index_cols:
            index_cols = ['time'] + index_cols
            
        df = df.set_index(index_cols)
        df = df.loc[:,columns]
        
        return df
        
    except Exception as e:
        logger.warning(f"Failed to load tile [{ix}, {iy}] from {zarr_path}: {e}")
        return None

def extract_metadata_from_datasets(datasets_config: Dict[str, Dict]) -> Dict[str, Dict]:
    """Extract scale_factor and add_offset metadata from zarr datasets."""
    metadata = {}
    
    for dataset_name, config in datasets_config.items():
        zarr_path = config['path']
        if not os.path.exists(zarr_path):
            continue
            
        try:
            ds = xr.open_zarr(zarr_path, mask_and_scale=False, consolidated=False)
            dataset_metadata = {}
            
            columns = [x for x in ds.data_vars.keys() if x != "spatial_ref"]                
            if 'columns' in config:
                columns = [var for var in ds.data_vars if var in config['columns']]
            
            for var_name in columns:
                var_attrs = ds[var_name].attrs
                var_metadata = {
                    'dtype': str(ds[var_name].dtype),
                    'scale_factor': var_attrs.get('scale_factor', 1.0),
                    'add_offset': var_attrs.get('add_offset', 0.0),
                    '_FillValue': var_attrs.get('_FillValue', 0)
                }
                dataset_metadata[var_name] = var_metadata
            
            metadata[dataset_name] = dataset_metadata
            ds.close()
            
        except Exception as e:
            logger.warning(f"Could not extract metadata from {zarr_path}: {e}")
    
    return metadata

def demean_columns(df: pd.DataFrame, columns_to_demean: List[str]) -> pd.DataFrame:
    """Apply year and pixel demeaning to specified columns."""
    for col in columns_to_demean:
        if col in df.columns:
            logger.debug(f"Demeaning column: {col}")
            
            # Two-way demeaning: subtract year means and pixel means
            year_means = df.groupby('time')[col].transform('mean')
            year_demeaned = df[col] - year_means
            pixel_means = year_demeaned.groupby('pixel_id').transform('mean')
            df[f"{col}_demeaned"] = year_demeaned - pixel_means
    
    return df

def process_tile(
    ix: int, 
    iy: int, 
    tile_geobox,
    assembly_config: Dict[str, Any], 
    output_base_path: str,
    land_mask_ds: Optional[xr.Dataset] = None
) -> bool:
    """Process a single tile according to assembly configuration and write to parquet."""
    logger.debug(f"Processing tile ix={ix}, iy={iy}")
    
    # Check if tile already exists using fast file existence check
    tile_output_path = os.path.join(output_base_path, f"ix={ix}", f"iy={iy}")
    output_file = os.path.join(tile_output_path, "data.parquet")
    
    if os.path.exists(output_file):
        # Optionally verify the file is valid and not corrupted
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(output_file)
            # Quick check that file has some rows
            if parquet_file.metadata.num_rows > 0:
                logger.debug(f"Tile ix={ix}, iy={iy} already exists and is valid, skipping")
                return True
            else:
                logger.warning(f"Tile ix={ix}, iy={iy} exists but is empty, will reprocess")
        except Exception as e:
            logger.warning(f"Tile ix={ix}, iy={iy} exists but appears corrupted ({e}), will reprocess")
    
    datasets_config = assembly_config['datasets']
    processing_config = assembly_config.get('processing', {})
    demean_cols = processing_config.get('demean_columns', [])
    
    # Load and merge datasets for this tile
    merged = None
    dataset_names = list(datasets_config.keys())
    
    for i, dataset_name in enumerate(dataset_names):
        dataset_config = datasets_config[dataset_name]
        dataset = load_tile_data_from_zarr(dataset_config, ix, iy, tile_geobox, land_mask_ds)
        
        if dataset is not None:
            logger.debug(f"Loading dataset {dataset_name} for tile ix={ix}, iy={iy}")
            
            if merged is None:
                # First dataset becomes the base
                merged = dataset
                logger.debug(f"Started with dataset: {dataset_name}")
            else:
                # Join with existing data
                join_type = dataset_config.get('join_type', 'left')
                logger.debug(f"Joining {dataset_name} with {join_type} join for tile ix={ix}, iy={iy}")
                
                # Determine join strategy based on index structure
                if len(dataset_config['index_cols']) == len(merged.index.names):
                    # Same index structure - join on index
                    merged = merged.join(dataset, how=join_type)
                else:
                    # Different index structure - reset and merge on columns
                    merged_reset = merged.reset_index()
                    dataset_reset = dataset.reset_index()
                    merge_cols = [col for col in dataset_config['index_cols'] if col in merged_reset.columns]
                    merged = pd.merge(merged_reset, dataset_reset, on=merge_cols, how=join_type)
        else:
            logger.debug(f"No data found for {dataset_name} in tile ix={ix}, iy={iy}")
    
    if merged is None or merged.empty:
        logger.debug(f"No data for tile ix={ix}, iy={iy}, skipping")
        return False
    
    # Write tile to output parquet partition
    os.makedirs(tile_output_path, exist_ok=True)
    
    compression = processing_config.get('compression', 'zstd')
    
    logger.debug(f"Writing tile ix={ix}, iy={iy} to {output_file}")
    merged.to_parquet(output_file, compression=compression, engine='pyarrow')
    
    return True

def create_assembly_metadata(
    output_path: str,
    metadata: Dict[str, Dict],
    assembly_config: Dict[str, Any]
) -> bool:
    """Create metadata YAML file for assembled parquet output."""
    try:
        # Write metadata YAML
        metadata_path = os.path.join(output_path, '_metadata.yaml')
        with open(metadata_path, 'w') as f:
            yaml.dump({
                'assembly_config': assembly_config,
                'variable_metadata': metadata,
                'output_format': 'parquet',
                'partitioning': 'ix/iy tiles',
                'description': 'Assembled dataset in tile-partitioned parquet format'
            }, f, default_flow_style=False)
        
        logger.info(f"Created assembly metadata at {metadata_path}")
        return True
        
    except Exception as e:
        logger.exception(f"Error creating assembly metadata: {e}")
        return False

def _strip_remote_prefix(path):
    """Remove scp/ssh prefix like user@host: from paths."""
    if isinstance(path, str):
        import re
        return re.sub(r"^[^@]+@[^:]+:", "", path)
    return path

def _derive_hpc_root_from_config(assembly_config: Dict[str, Any], full_config: Dict[str, Any] = None) -> Optional[str]:
    """
    Derive hpc_root from assembly configuration, checking multiple sources.
    
    Args:
        assembly_config: Assembly configuration dictionary
        full_config: Full configuration dictionary containing HPC settings
        
    Returns:
        str: HPC root path or None if not found
    """
    # Check assembly config first for hpc settings
    hpc_config = full_config.get('hpc', {})
    hpc_target = hpc_config.get('target')
    
    if hpc_target:
        return _strip_remote_prefix(hpc_target)
    
    # Check full config for hpc settings
    if full_config:
        hpc_config = full_config.get('hpc', {})
        hpc_target = hpc_config.get('target')
        
        if hpc_target:
            return _strip_remote_prefix(hpc_target)
    
    logger.warning("Could not derive hpc_root from configuration")
    return None


def run_assembly(assembly_config: Dict[str, Any], full_config: Dict[str, Any] = None):
    """Run the data assembly process based on configuration."""
    logger.info(f"Starting assembly: {assembly_config.get('description', 'Unknown')}")
    
    # Get output path - now expecting parquet directory output
    output_path = assembly_config['output_path']
    processing_config = assembly_config.get('processing', {})
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Output will be written to: {output_path}")
    
    # Try to derive hpc_root from configuration
    hpc_root = _derive_hpc_root_from_config(assembly_config, full_config)
        
    if not hpc_root:
        logger.error("hpc_root must be specified in processing config or derivable from dataset paths")
        return
        
    logger.info(f"Using hpc_root: {hpc_root}")
        
    target_geobox = get_or_create_geobox(hpc_root)
    
    # Load land mask if requested
    land_mask_ds = None
    if processing_config.get('apply_land_mask', False):
        land_mask_path = processing_config.get('land_mask_path')
        land_mask_ds = _load_land_mask(hpc_root, land_mask_path)
    
    # Extract metadata from datasets
    metadata = extract_metadata_from_datasets(assembly_config['datasets'])
    
    # Create assembly metadata file
    if not create_assembly_metadata(output_path, metadata, assembly_config):
        logger.warning("Failed to create assembly metadata")
    
    # Get available tiles
    logger.info("Discovering available tiles...")
    all_tiles = get_available_tiles(assembly_config, target_geobox)
    logger.info(f"Found {len(all_tiles)} tiles to process")
    
    if not all_tiles:
        logger.warning("No tiles found to process")
        return
    
    # Create tile geobox for processing
    tile_size = processing_config.get('tile_size', 2048)
    tiles = GeoboxTiles(target_geobox, (tile_size, tile_size))
    
    # Process each tile
    processed_count = 0
    skipped_count = 0
    for i, (ix, iy) in enumerate(all_tiles):
        logger.info(f"Processing tile {i+1}/{len(all_tiles)}: ix={ix}, iy={iy}")
        
        # Fast check if tile already exists before expensive processing
        tile_output_path = os.path.join(output_path, f"ix={ix}", f"iy={iy}")
        output_file = os.path.join(tile_output_path, "data.parquet")
        
        if os.path.exists(output_file):
            # Quick validation that the file is not corrupted
            try:
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(output_file)
                if parquet_file.metadata.num_rows > 0:
                    logger.debug(f"Tile ix={ix}, iy={iy} already exists and is valid, skipping")
                    skipped_count += 1
                    continue
                else:
                    logger.warning(f"Tile ix={ix}, iy={iy} exists but is empty, will reprocess")
            except Exception as e:
                logger.warning(f"Tile ix={ix}, iy={iy} exists but appears corrupted ({e}), will reprocess")
        
        try:
            tile_geobox = tiles[ix, iy]
            success = process_tile(ix, iy, tile_geobox, assembly_config, output_path, land_mask_ds)
            
            if success:
                processed_count += 1
                
        except Exception as e:
            logger.error(f"Failed to process tile ix={ix}, iy={iy}: {e}", exc_info=True)
            continue
    
    logger.info(f"Assembly process completed successfully. Processed {processed_count}/{len(all_tiles)} tiles, skipped {skipped_count} existing tiles")

def run_workflow_with_config(config: Dict[str, Any]):
    """Entry point for running assembly workflow with unified configuration."""
    assembly_name = config.get('assembly_name', 'main')
    
    # Get assembly configuration
    if 'assemble' not in config or assembly_name not in config['assemble']:
        raise ValueError(f"Assembly configuration '{assembly_name}' not found in config")
    
    assembly_config = config['assemble'][assembly_name]
    
    # Run the assembly, passing both assembly config and full config
    run_assembly(assembly_config, config)

if __name__ == "__main__":
    logger.error("This module should be run through the unified run.py interface")
    logger.error("Usage: python run.py assemble --config config.yaml --source assembly_name")
"""
Data assembly module for the GNT system.

This module provides functionality to merge multiple datasets using a tile-by-tile
approach based on configuration specifications, reading directly from zarr files
and outputting to parquet format with automatic scaling applied.
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

# Import common utilities
from gnt.data.common.geobox import get_or_create_geobox
from gnt.data.common.dask.client import DaskClientContextManager

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
                land_mask_ds = xr.open_zarr(path, consolidated=False, chunks='auto')["land_mask"]
                return land_mask_ds
        
        logger.warning(f"Land mask not found in any of: {potential_paths}")
        return None
        
    except Exception as e:
        logger.warning(f"Error loading land mask: {e}")
        return None
    
def _load_and_merge_datasets(
    assembly_config: Dict[str, Any],
    land_mask_ds: Optional[xr.Dataset] = None
) -> xr.Dataset:
    """
    Load and merge all zarr datasets into a single large xarray dataset.
    
    Args:
        assembly_config: Assembly configuration
        land_mask_ds: Optional land mask dataset
        
    Returns:
        Large merged xarray dataset with all variables
    """
    logger.info("Loading and merging all zarr datasets...")
    
    datasets_config = assembly_config['datasets']
    merged_datasets = []
    
    # Add land mask as the first dataset if provided
    if land_mask_ds is not None:
        logger.info("Adding land mask as first dataset")
        # Transform time to year if present in land mask
        if "time" in land_mask_ds.coords:
            land_mask_ds.coords["time"] = pd.Series(land_mask_ds.coords["time"]).dt.year.astype("int16")
        
        land_mask_ds.attrs['dataset_name'] = 'land_mask'
        merged_datasets.append(land_mask_ds)
    
    for dataset_name, dataset_config in datasets_config.items():
        zarr_path = dataset_config['path']
        
        if not os.path.exists(zarr_path):
            logger.warning(f"Dataset path does not exist: {zarr_path}, skipping")
            continue
        
        logger.info(f"Loading dataset {dataset_name} from {zarr_path}")
        
        try:
            # Open zarr dataset with automatic scaling and masking
            ds = xr.open_zarr(zarr_path, mask_and_scale=True, consolidated=False, chunks='auto')
            
            # Check and subset columns if specified
            columns = dataset_config.get('columns')
            if columns:
                # Check which columns exist in the dataset
                available_vars = [var for var in columns if var in ds.data_vars]
                missing_vars = [var for var in columns if var not in ds.data_vars]
                
                if missing_vars:
                    logger.warning(f"Dataset {dataset_name} missing specified columns: {missing_vars}")
                
                if available_vars:
                    ds = ds[available_vars]
                else:
                    logger.warning(f"No specified columns found in dataset {dataset_name}, skipping")
                    continue
            else:
                # Remove spatial_ref if present, keep all other variables
                vars_to_keep = [var for var in ds.data_vars.keys() if var != 'spatial_ref']
                if vars_to_keep:
                    ds = ds[vars_to_keep]
                    logger.debug(f"Dataset {dataset_name}: keeping all variables {vars_to_keep}")
            
            # Apply column prefix if specified
            column_prefix = dataset_config.get('column_prefix')
            if column_prefix:
                logger.info(f"Applying prefix '{column_prefix}' to dataset {dataset_name}")
                # Rename all data variables with prefix
                rename_dict = {var: f"{column_prefix}{var}" for var in ds.data_vars}
                ds = ds.rename(rename_dict)
                logger.debug(f"Renamed variables: {list(ds.data_vars.keys())}")
            
            # Add dataset identifier for provenance
            ds.attrs['dataset_name'] = dataset_name
            merged_datasets.append(ds)
            
            logger.info(f"Loaded dataset {dataset_name}: {list(ds.data_vars.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            continue
    
    if not merged_datasets:
        raise ValueError("No datasets could be loaded successfully")
    
    # Merge all datasets using default join behavior
    logger.info("Merging datasets...")
    try:
        if len(merged_datasets) == 1:
            merged_ds = merged_datasets[0]
        else:
            # Start with the first dataset as base
            merged_ds = merged_datasets[0]
            logger.info(f"Base dataset: {merged_ds.attrs.get('dataset_name', 'unknown')}")
            
            # Merge remaining datasets using default xarray merge behavior
            for ds in merged_datasets[1:]:
                dataset_name = ds.attrs.get('dataset_name', 'unknown')
                
                logger.info(f"Merging {dataset_name}")
                merged_ds = xr.merge([merged_ds, ds], compat='override', join="left")
        
        logger.info(f"Successfully merged {len(merged_datasets)} datasets")
        logger.info(f"Final dataset variables: {list(merged_ds.data_vars.keys())}")
        logger.info(f"Dataset dimensions: {dict(merged_ds.sizes)}")
        
        return merged_ds
        
    except Exception as e:
        logger.error(f"Failed to merge datasets: {e}")
        raise
    
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

def create_assembly_metadata(
    output_path: str,
    assembly_config: Dict[str, Any]
) -> bool:
    """Create metadata YAML file for assembled parquet output."""
    try:
        # Write metadata YAML
        metadata_path = os.path.join(output_path, '_metadata.yaml')
        
        metadata_dict = {
            'assembly_config': assembly_config,
            'output_format': 'parquet',
            'partitioning': 'ix/iy tiles',
            'scaling': 'Applied during zarr read with mask_and_scale=True',
            'description': 'Assembled dataset in tile-partitioned parquet format with automatic scaling'
        }
        
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata_dict, f, default_flow_style=False)
        
        logger.info(f"Metadata written to {metadata_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to create metadata file: {e}")
        return False

def _extract_and_process_tile(
    merged_ds: xr.Dataset,
    ix: int,
    iy: int, 
    tile_geobox,
    assembly_config: Dict[str, Any]
) -> Optional[pd.DataFrame]:
    """
    Extract and process a single tile from the large merged dataset.
    
    Args:
        merged_ds: Large merged xarray dataset
        ix, iy: Tile coordinates
        tile_geobox: Tile geobox for spatial bounds
        assembly_config: Assembly configuration
        
    Returns:
        Processed DataFrame for the tile
    """
    try:
        logger.debug(f"Extracting tile [{ix}, {iy}] from merged dataset")
        
        # Extract tile bounds
        bbox = tile_geobox.boundingbox
        
        # Slice dataset to tile bounds
        if 'latitude' in merged_ds.coords and 'longitude' in merged_ds.coords:
            tile_ds = merged_ds.sel(
                latitude=slice(bbox.top, bbox.bottom),
                longitude=slice(bbox.left, bbox.right)
            ).compute()
        else:
            logger.warning(f"Unknown coordinate system in merged dataset")
            return None
        
        # Check if tile has data
        if tile_ds.sizes.get('latitude', 0) == 0 or tile_ds.sizes.get('longitude', 0) == 0:
            logger.debug(f"No spatial data in tile [{ix}, {iy}]")
            return None

        # Create bit-packed pixel ID: combines ix (16 bits), iy (16 bits), and local pixel index (32 bits)
        # Format: [ix: 16 bits | iy: 16 bits | local_pixel: 32 bits] = 64-bit integer
        tile_size_actual = (tile_ds.sizes['latitude'], tile_ds.sizes['longitude'])
        local_pixel_ids = np.arange(tile_size_actual[0] * tile_size_actual[1], dtype="uint32").reshape(tile_size_actual)
        
        # Bit-pack: (ix << 48) | (iy << 32) | local_pixel_id
        pixel_id_matrix = (
            (np.uint64(ix) << 48) | 
            (np.uint64(iy) << 32) | 
            local_pixel_ids.astype(np.uint64)
        )
        
        tile_ds = tile_ds.assign({
            'pixel_id': (['latitude', 'longitude'], pixel_id_matrix)
        })
                
        # Convert to DataFrame
        logger.debug(f"Transforming to dataframe...")
        df = tile_ds.to_dataframe().reset_index()
        
        # Clean
        df = df.drop(columns=['band', 'latitude', 'longitude'])
        
        # Apply land mask filtering if land_mask variable is present
        if 'land_mask' in df.columns:
            df = df[df.land_mask]
            df = df.drop(columns=['land_mask'])
        
        if df.empty:
            logger.debug(f"No data remaining in tile [{ix}, {iy}]")
            return None
        
        return df
        
    except Exception as e:
        logger.warning(f"Failed to extract tile [{ix}, {iy}] from merged dataset: {e}")
        return None

def process_tile(
    merged_ds: xr.Dataset,
    ix: int, 
    iy: int, 
    tile_geobox,
    assembly_config: Dict[str, Any], 
    output_base_path: str
) -> bool:
    """Process a single tile from the merged dataset and write to parquet."""
    logger.debug(f"Processing tile ix={ix}, iy={iy} from merged dataset")
    
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
    
    processing_config = assembly_config.get('processing', {})
    
    # Extract and process tile data from merged dataset
    merged = _extract_and_process_tile(
        merged_ds, ix, iy, tile_geobox, assembly_config
    )
    
    if merged is None or merged.empty:
        logger.debug(f"No data for tile ix={ix}, iy={iy}, skipping")
        return False
        
    # Write tile to output parquet partition
    os.makedirs(tile_output_path, exist_ok=True)
    
    compression = processing_config.get('compression', 'snappy')
    
    logger.debug(f"Writing tile ix={ix}, iy={iy} to {output_file}")
    merged.to_parquet(output_file, index = False, compression=compression, engine='pyarrow')
    
    return True

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
    
    # Configure Dask settings and create cluster
    dask_config = processing_config.get('dask', {})
    
    # Set up Dask cluster for all operations with optimized parameters
    dask_kwargs = {
        'threads': dask_config.get('threads'),
        'memory_limit': dask_config.get('memory_limit'),
        'dashboard_port': dask_config.get('dashboard_port', 8787),
        'temp_dir': dask_config.get('temp_dir'),
        'worker_threads_per_cpu': dask_config.get('worker_threads_per_cpu', 2),
        'worker_fraction': dask_config.get('worker_fraction', 0.5)
    }
    
    # Remove None values
    dask_kwargs = {k: v for k, v in dask_kwargs.items() if v is not None}
    
    logger.info("Creating Dask cluster for data loading and processing...")
    
    with DaskClientContextManager(**dask_kwargs) as client:
        logger.info(f"Dask client initialized: {client.dashboard_link}")
        
        # Load land mask if requested - will be merged as regular variable
        land_mask_ds = None
        if processing_config.get('apply_land_mask', False):
            land_mask_path = processing_config.get('land_mask_path')
            land_mask_ds = _load_land_mask(hpc_root, land_mask_path)
        
        # Load and merge all datasets into one large xarray dataset (including land mask)
        logger.info("Step 1: Loading and merging all zarr datasets...")
        try:
            merged_ds = _load_and_merge_datasets(assembly_config, land_mask_ds)
            logger.info("Successfully created large merged dataset")
        except Exception as e:
            logger.error(f"Failed to create merged dataset: {e}")
            return
        
        # Convert time coordinate to integer years
        logger.info("Step 2: Converting time coordinate to integer years...")
        if "time" in merged_ds.coords:
            merged_ds = merged_ds.rename(time='year')
            merged_ds.coords["year"] = pd.Series(merged_ds.coords["year"]).dt.year.astype("int16")
            logger.info(f"Time coordinate converted to year: {merged_ds.coords['year'].values[:5]}...")
        else:
            logger.info("No time coordinate found in merged dataset")
        
        # Create assembly metadata file
        if not create_assembly_metadata(output_path, assembly_config):
            logger.warning("Failed to create assembly metadata")
        
        # Step 3: Process tiles from merged dataset
        logger.info("Step 3: Processing tiles from merged dataset...")

        processed_count = 0
        skipped_count = 0

        for ix, iy in all_tiles:
            # Skip if tile already exists and is valid
            tile_output_path = os.path.join(output_path, f"ix={ix}", f"iy={iy}")
            output_file = os.path.join(tile_output_path, "data.parquet")

            if os.path.exists(output_file):
                try:
                    import pyarrow.parquet as pq
                    parquet_file = pq.ParquetFile(output_file)
                    if parquet_file.metadata.num_rows > 0:
                        logger.debug(f"Tile ix={ix}, iy={iy} already exists and is valid, skipping")
                        skipped_count += 1
                        continue
                except Exception:
                    pass  # Will reprocess if file is corrupted

            tile_geobox = tiles[ix, iy]
            try:
                success = process_tile(
                    merged_ds, ix, iy, tile_geobox,
                    assembly_config, output_path
                )
                if success:
                    processed_count += 1
                    logger.debug(f"Completed tile ix={ix}, iy={iy}")
                else:
                    logger.debug(f"No data for tile ix={ix}, iy={iy}")
            except Exception as e:
                logger.error(f"Failed to process tile ix={ix}, iy={iy}: {e}")
                continue
            
        logger.info(f"Dask processing completed. Processed {processed_count}/{len(all_tiles)} tiles, skipped {skipped_count} existing tiles")

def run_workflow_with_config(config: Dict[str, Any]):
    """Entry point for running assembly workflow with unified configuration."""
    assembly_name = config.get('assembly_name', 'main')
    
    # Get assembly configuration
    if 'assemble' not in config or assembly_name not in config['assemble']:
        raise ValueError(f"Assembly configuration '{assembly_name}' not found in config")
    
    assembly_config = config['assemble'][assembly_name]
    
    # Apply CLI overrides to assembly configuration if present in full config
    cli_overrides = config.get('cli_overrides', {})
    if cli_overrides:
        processing_config = assembly_config.setdefault('processing', {})
        dask_config = processing_config.setdefault('dask', {})
        
        # Apply Dask-related CLI overrides
        if 'dask_threads' in cli_overrides:
            dask_config['threads'] = cli_overrides['dask_threads']
            logger.info(f"Overriding dask threads from CLI: {cli_overrides['dask_threads']}")
        if 'dask_memory_limit' in cli_overrides:
            dask_config['memory_limit'] = cli_overrides['dask_memory_limit']
            logger.info(f"Overriding dask memory limit from CLI: {cli_overrides['dask_memory_limit']}")
        if 'temp_dir' in cli_overrides:
            dask_config['temp_dir'] = cli_overrides['temp_dir']
            logger.info(f"Overriding temp dir from CLI: {cli_overrides['temp_dir']}")
        if 'dashboard_port' in cli_overrides:
            dask_config['dashboard_port'] = cli_overrides['dashboard_port']
            logger.info(f"Overriding dashboard port from CLI: {cli_overrides['dashboard_port']}")
        
        # Apply other processing overrides
        if 'tile_size' in cli_overrides:
            processing_config['tile_size'] = cli_overrides['tile_size']
            logger.info(f"Overriding tile size from CLI: {cli_overrides['tile_size']}")
        if 'compression' in cli_overrides:
            processing_config['compression'] = cli_overrides['compression']
            logger.info(f"Overriding compression from CLI: {cli_overrides['compression']}")
    
    # Run the assembly, passing both assembly config and full config
    run_assembly(assembly_config, config)
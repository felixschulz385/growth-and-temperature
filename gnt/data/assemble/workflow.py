"""
Main workflow orchestration for data assembly.

This module provides the high-level workflow functions that coordinate
dataset loading, tile processing, and output generation. Implementation
details are delegated to specialized modules.
"""

import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

import logging
from typing import Dict, Any, List, Optional, Tuple

from odc.geo import GeoboxTiles

# Import common utilities
from gnt.data.common.geobox import get_or_create_geobox
from gnt.data.common.dask.client import DaskClientContextManager

# Import assembly submodules
from gnt.data.assemble.config import (
    derive_hpc_root,
    apply_cli_overrides,
    validate_assembly_config,
    ProcessingConfig,
)
from gnt.data.assemble.loaders import (
    load_land_mask,
    load_all_datasets,
    prepare_land_mask,
)
from gnt.data.assemble.processors import TileProcessor
from gnt.data.assemble.metadata import create_assembly_metadata
from gnt.data.assemble.tiles import (
    get_available_tiles,
    adjust_tile_size_for_reprojection,
    create_tile_geobox,
)
from gnt.data.assemble.constants import DEFAULT_TILE_SIZE

logger = logging.getLogger(__name__)


def _setup_dask_cluster(processing_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create Dask cluster configuration from processing config.
    
    Args:
        processing_config: Processing configuration with dask settings
        
    Returns:
        Dictionary of kwargs for DaskClientContextManager
    """
    from gnt.data.assemble.config import DaskConfig
    
    dask_dict = processing_config.get('dask', {})
    dask_config = DaskConfig(
        threads=dask_dict.get('threads'),
        memory_limit=dask_dict.get('memory_limit'),
        dashboard_port=dask_dict.get('dashboard_port', 8787),
        temp_dir=dask_dict.get('temp_dir'),
        worker_threads_per_cpu=dask_dict.get('worker_threads_per_cpu', 2),
        worker_fraction=dask_dict.get('worker_fraction', 0.5),
    )
    return dask_config.to_kwargs()


def _process_all_tiles(
    datasets: List[Tuple],
    land_mask_ds,
    all_tiles: List[Tuple[int, int]],
    target_geobox,
    assembly_config: Dict[str, Any],
    output_path: str,
) -> Tuple[int, int]:
    """
    Process all tiles and return counts of processed and skipped tiles.
    
    Args:
        datasets: Loaded datasets
        land_mask_ds: Optional land mask dataset
        all_tiles: List of (ix, iy) tile indices
        target_geobox: Target geobox
        assembly_config: Assembly configuration
        output_path: Output directory path
        
    Returns:
        Tuple of (processed_count, skipped_count)
    """
    processing_config = assembly_config.get('processing', {})
    tile_size = processing_config.get('tile_size', DEFAULT_TILE_SIZE)
    assembly_mode = processing_config.get('assembly_mode', 'create')
    
    processor = TileProcessor(assembly_config, output_path)
    processed_count = 0
    skipped_count = 0
    
    for ix, iy in all_tiles:
        tile_output_path = os.path.join(output_path, f"ix={ix}", f"iy={iy}")
        output_file = os.path.join(tile_output_path, "data.parquet")
        
        # Default behavior: recreate all tiles (no skipping)
        # Only skip in update mode if tile doesn't exist yet
        if assembly_mode == 'update' and not os.path.exists(output_file):
            logger.warning(f"Tile ix={ix}, iy={iy} does not exist, skipping in update mode")
            skipped_count += 1
            continue
        
        tile_geobox = create_tile_geobox(target_geobox, tile_size, ix, iy)
        
        try:
            success = processor.process_tile(
                datasets, land_mask_ds, ix, iy, tile_geobox
            )
            if success:
                processed_count += 1
        except Exception as e:
            logger.error(f"Failed to process tile ix={ix}, iy={iy}: {e}")
            continue
    
    return processed_count, skipped_count


def run_assembly(assembly_config: Dict[str, Any], full_config: Optional[Dict[str, Any]] = None):
    """
    Run the complete data assembly workflow.
    
    Main steps:
    1. Validate configuration
    2. Initialize Dask cluster for parallel processing
    3. Load all datasets individually (source-by-source approach)
    4. Create metadata YAML for provenance
    5. Process all tiles, each extracting/processing all datasets
    6. Write results as partitioned parquet files (ix/iy directory structure)
    
    The source-by-source approach enables:
    - Independent resampling methods per dataset (e.g., bilinear for continuous, mode for categorical)
    - Per-dataset winsorization configuration
    - Flexible alignment and concatenation at tile level
    
    Args:
        assembly_config: Assembly-specific configuration
        full_config: Full configuration dictionary (optional, for HPC settings)
    """
    logger.info(f"Starting assembly: {assembly_config.get('description', 'Unknown')}")
    
    # Validate configuration
    errors = validate_assembly_config(assembly_config)
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        return
    
    output_path = assembly_config['output_path']
    processing_config = assembly_config.get('processing', {})
    
    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Output will be written to: {output_path}")
    
    # Derive HPC root from configuration
    hpc_root = derive_hpc_root(assembly_config, full_config)
    if not hpc_root:
        logger.error("hpc_root must be specified in config or derivable from HPC settings")
        return
    
    logger.info(f"Using hpc_root: {hpc_root}")
    
    # Get target geobox and adjust tile size for reprojection
    target_geobox = get_or_create_geobox(hpc_root)
    processing_config.setdefault('tile_size', DEFAULT_TILE_SIZE)
    native_res = abs(target_geobox.resolution.x)
    processing_config['tile_size'] = adjust_tile_size_for_reprojection(
        native_res,
        processing_config.get('resolution'),
        processing_config['tile_size']
    )
    
    # Discover tiles
    logger.info("Discovering available tiles...")
    all_tiles = get_available_tiles(assembly_config, target_geobox)
    logger.info(f"Found {len(all_tiles)} tiles to process")
    
    if not all_tiles:
        logger.warning("No tiles found to process")
        return
    
    # Set up Dask cluster
    dask_kwargs = _setup_dask_cluster(processing_config)
    logger.info("Creating Dask cluster for data loading and processing...")
    
    with DaskClientContextManager(**dask_kwargs) as client:
        logger.info(f"Dask client initialized: {client.dashboard_link}")
        
        # Load land mask if requested
        land_mask_ds = None
        if processing_config.get('apply_land_mask', False):
            land_mask_path = processing_config.get('land_mask_path')
            land_mask_ds = load_land_mask(hpc_root, land_mask_path)
            if land_mask_ds is not None:
                land_mask_ds = prepare_land_mask(land_mask_ds)
        
        # Step 1: Load datasets
        logger.info("Step 1: Loading datasets with alignment checks...")
        try:
            assembly_mode = processing_config.get('assembly_mode', 'create')
            target_datasource = processing_config.get('datasource')
            
            if assembly_mode == 'update':
                if not target_datasource:
                    logger.error("Update mode requires datasource to be specified")
                    return
                logger.info(f"UPDATE mode: Loading only datasource '{target_datasource}'")
                # In update mode, only load the specified datasource
                datasets = load_all_datasets(assembly_config, target_geobox, datasource_filter=target_datasource)
            else:
                logger.info("CREATE mode: Loading all datasets")
                datasets = load_all_datasets(assembly_config, target_geobox)
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            return
        
        # Step 2: Create metadata
        logger.info("Step 2: Creating assembly metadata...")
        if not create_assembly_metadata(output_path, assembly_config):
            logger.warning("Failed to create assembly metadata")
        
        # Step 3: Process tiles
        logger.info("Step 3: Processing tiles (source-by-source)...")
        processed_count, skipped_count = _process_all_tiles(
            datasets, land_mask_ds, all_tiles, target_geobox, 
            assembly_config, output_path
        )
        
        logger.info(
            f"Dask processing completed. Processed {processed_count}/{len(all_tiles)} tiles, "
            f"skipped {skipped_count} existing tiles"
        )


def run_workflow_with_config(config: Dict[str, Any]):
    """
    Entry point for assembly workflow with unified configuration.
    
    Applies CLI overrides (Dask settings, tile size, compression) to assembly config
    before running the assembly process.
    
    Args:
        config: Full configuration dictionary including:
            - 'assembly_name': Name of assembly config to use (default: 'main')
            - 'assemble': Dict of assembly configurations
            - 'cli_overrides': Optional CLI override values
            
    Raises:
        ValueError: If specified assembly configuration not found
    """
    assembly_name = config.get('assembly_name', 'main')
    
    if 'assemble' not in config or assembly_name not in config['assemble']:
        raise ValueError(f"Assembly configuration '{assembly_name}' not found in config")
    
    assembly_config = config['assemble'][assembly_name]
    
    # Apply CLI overrides
    cli_overrides = config.get('cli_overrides', {})
    apply_cli_overrides(assembly_config, cli_overrides)
    
    # Run the assembly
    run_assembly(assembly_config, config)
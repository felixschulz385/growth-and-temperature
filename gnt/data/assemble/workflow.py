"""
Main workflow orchestration for data assembly.

This module provides the high-level workflow functions that coordinate
dataset loading, tile processing, and output generation. Implementation
details are delegated to specialized modules.
"""

import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

import logging
import importlib
from typing import Dict, Any, List, Optional, Tuple

# Import common utilities
from gnt.data.common.geobox import get_or_create_geobox
from gnt.data.common.dask.client import DaskClientContextManager

# Import assembly submodules
from gnt.data.assemble.config import (
    derive_data_root,
    apply_cli_overrides,
    validate_assembly_config,
)
from gnt.data.assemble.loaders import (
    load_land_mask,
    load_all_datasets,
    prepare_land_mask,
)
from gnt.data.assemble.processors import TileProcessor, uses_geometry_aggregation
from gnt.data.assemble.metadata import create_assembly_metadata
from gnt.data.assemble.tiles import (
    get_available_tiles,
    adjust_tile_size_for_reprojection,
    create_tile_geobox,
)
from gnt.data.assemble.constants import DEFAULT_TILE_SIZE

logger = logging.getLogger(__name__)


def _load_geometry_aggregator(import_path: str):
    """
    Load a geometry aggregation callable from an import path.

    Supported formats:
    - ``package.module:function``
    - ``package.module.function``
    """
    if ":" in import_path:
        module_name, attr_name = import_path.split(":", 1)
    else:
        module_name, attr_name = import_path.rsplit(".", 1)

    module = importlib.import_module(module_name)
    aggregator = getattr(module, attr_name)

    if not callable(aggregator):
        raise TypeError(f"Geometry aggregator '{import_path}' is not callable")

    return aggregator


def _run_geometry_assembly(
    assembly_config: Dict[str, Any],
    full_config: Optional[Dict[str, Any]],
    hpc_root: str,
    target_geobox,
    output_path: str,
):
    """
    Run geometry-based assembly instead of tiled grid assembly.

    The heavy lifting is delegated to a user-specified aggregator callable so
    geometry assembly can evolve independently from the grid backend.
    """
    processing_config = assembly_config.get("processing", {})
    geometry_source = assembly_config["geometry_source"]
    aggregator_path = assembly_config["geometry_aggregator"]

    logger.info(
        "GEOMETRY mode: aggregating datasets to %s using %s",
        geometry_source.get("path"),
        aggregator_path,
    )

    dask_kwargs = _setup_dask_cluster(processing_config)
    logger.info("Creating Dask cluster for geometry assembly...")

    with DaskClientContextManager(**dask_kwargs) as client:
        logger.info(f"Dask client initialized: {client.dashboard_link}")

        land_mask_ds = None
        if processing_config.get("apply_land_mask", False):
            land_mask_path = processing_config.get("land_mask_path")
            land_mask_ds = load_land_mask(hpc_root, land_mask_path)
            if land_mask_ds is not None:
                land_mask_ds = prepare_land_mask(land_mask_ds)

        assembly_mode = processing_config.get("assembly_mode", "create")
        target_datasource = processing_config.get("datasource")

        if assembly_mode == "update":
            if not target_datasource:
                logger.error("Update mode requires datasource to be specified")
                return
            logger.info(
                "UPDATE mode: Loading only datasource '%s' for geometry assembly",
                target_datasource,
            )
            datasets = load_all_datasets(
                assembly_config,
                target_geobox,
                datasource_filter=target_datasource,
            )
        else:
            logger.info("CREATE mode: Loading all datasets for geometry assembly")
            datasets = load_all_datasets(assembly_config, target_geobox)

        logger.info("Creating assembly metadata...")
        if not create_assembly_metadata(output_path, assembly_config):
            logger.warning("Failed to create assembly metadata")

        aggregator = _load_geometry_aggregator(aggregator_path)
        aggregator(
            datasets=datasets,
            geometry_source=geometry_source,
            assembly_config=assembly_config,
            output_path=output_path,
            target_geobox=target_geobox,
            land_mask_ds=land_mask_ds,
            hpc_root=hpc_root,
            full_config=full_config,
            dask_client=client,
        )
        logger.info("Geometry assembly completed successfully")


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
    uses_geometry_output = uses_geometry_aggregation(assembly_config)
    
    processor = TileProcessor(assembly_config, output_path)
    processed_count = 0
    skipped_count = 0
    overwrite = processing_config.get('overwrite', True)  # Default to True for backward compatibility
    
    for ix, iy in all_tiles:
        tile_output_path = os.path.join(output_path, f"ix={ix}", f"iy={iy}")
        output_file = os.path.join(tile_output_path, "data.parquet")
        
        # Geometry-aggregated assemblies do not materialize ix/iy parquet files on disk,
        # so update mode cannot use file existence as a skip criterion for them.
        if (
            assembly_mode == 'update'
            and not uses_geometry_output
            and not os.path.exists(output_file)
        ):
            logger.warning(f"Tile ix={ix}, iy={iy} does not exist, skipping in update mode")
            skipped_count += 1
            continue
        
        # In create mode, skip existing tiles if overwrite=False
        if assembly_mode == 'create' and not overwrite and os.path.exists(output_file):
            logger.info(f"Tile ix={ix}, iy={iy} already exists, skipping (overwrite=False)")
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
    spatial_partition = processing_config.get("spatial_partition", "grid")

    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Output will be written to: {output_path}")
    
    # Derive local project data root from configuration
    data_root = derive_data_root(assembly_config, full_config)
    if not data_root:
        logger.error("data_root must be specified in config or derivable from runtime settings")
        return
    
    logger.info(f"Using data_root: {data_root}")
    
    # Get target geobox and adjust tile size for reprojection
    target_geobox = get_or_create_geobox(data_root)

    if spatial_partition == "geometry":
        _run_geometry_assembly(
            assembly_config=assembly_config,
            full_config=full_config,
            hpc_root=data_root,
            target_geobox=target_geobox,
            output_path=output_path,
        )
        return

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
            land_mask_ds = load_land_mask(data_root, land_mask_path)
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
        
        # Step 3: Process pixel tiles or geometry-aggregated output
        processor = TileProcessor(assembly_config, output_path)
        if uses_geometry_aggregation(assembly_config):
            logger.info(
                f"Step 3: Geometry-aggregated {assembly_mode.upper()} mode: building grid-cell tables, "
                "aggregating appended rows, and merging into the top-level output table"
            )
            processed_count, skipped_count = processor.process_geometry_output(
                datasets, land_mask_ds, all_tiles, target_geobox
            )
            logger.info(
                f"Geometry {assembly_mode} completed. Processed {processed_count} tile-dataset chunks and "
                f"skipped {skipped_count} chunks without usable source rows"
            )
        else:
            logger.info("Step 3: Processing tiles (source-by-source)...")
            processed_count, skipped_count = _process_all_tiles(
                datasets, land_mask_ds, all_tiles, target_geobox,
                assembly_config, output_path
            )
            logger.info(
                f"Dask processing completed. Processed {processed_count}/{len(all_tiles)} tiles, "
                f"skipped {skipped_count} tiles"
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

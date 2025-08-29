"""
Data assembly module for the GNT system.

This module provides functionality to merge multiple datasets using a tile-by-tile
approach based on configuration specifications.

Note: Requires Java 21.0.2 - run 'module load Java/21.0.2' before executing
"""

import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

import pyspark.pandas as ps
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import logging
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import common Spark utilities
from gnt.data.common.spark import create_spark_session, SparkSessionContextManager

logger = logging.getLogger(__name__)

# Configure Spark-related logging to reduce verbosity
def _configure_logging():
    """Configure logging to reduce Spark/py4j noise."""
    # Set specific loggers to WARNING level to reduce debug output
    noisy_loggers = [
        "py4j",
        "py4j.java_gateway", 
        "py4j.clientserver",
        "pyspark",
        "org.apache.spark",
        "org.eclipse.jetty",
        "org.sparkproject.jetty"
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

# Configure logging when module is imported
_configure_logging()

def get_available_tiles(dataset_paths: List[str]) -> List[tuple]:
    """Get all available tile combinations (ix, iy) from all datasets."""
    all_tiles = set()
    
    for path in dataset_paths:
        if os.path.exists(path):
            ix_dirs = glob.glob(os.path.join(path, "ix=*"))
            for ix_dir in ix_dirs:
                ix_value = os.path.basename(ix_dir).split("=")[1]
                iy_dirs = glob.glob(os.path.join(ix_dir, "iy=*"))
                for iy_dir in iy_dirs:
                    iy_value = os.path.basename(iy_dir).split("=")[1]
                    parquet_files = glob.glob(os.path.join(iy_dir, "*.parquet"))
                    if parquet_files:
                        all_tiles.add((ix_value, iy_value))
    
    return sorted(list(all_tiles))

def load_tile_data(dataset_config: Dict[str, Any], ix: str, iy: str) -> Optional[ps.DataFrame]:
    """Load data for a specific tile from a dataset."""
    base_path = dataset_config['path']
    tile_path = os.path.join(base_path, f"ix={ix}", f"iy={iy}", "tile.parquet")
    
    if not os.path.exists(tile_path):
        return None
    
    try:
        index_cols = dataset_config['index_cols']
        columns = dataset_config.get('columns')
        
        if columns:
            df = ps.read_parquet(tile_path, index_col=index_cols, columns=columns)
        else:
            df = ps.read_parquet(tile_path, index_col=index_cols)
        return df
    except Exception as e:
        logger.warning(f"Failed to load tile ix={ix}, iy={iy} from {base_path}: {e}")
        return None

def demean_columns(df: ps.DataFrame, columns_to_demean: List[str]) -> ps.DataFrame:
    """Apply year and pixel demeaning to specified columns using pandas syntax."""
    for col in columns_to_demean:
        if col in df.columns:
            logger.debug(f"Demeaning column: {col}")
            
            # Two-way demeaning: subtract year means and pixel means
            year_means = df.groupby('time')[col].transform('mean')
            year_demeaned = df[col] - year_means
            pixel_means = year_demeaned.groupby('pixel_id').transform('mean')
            df[f"{col}_demeaned"] = year_demeaned - pixel_means
    
    return df

def process_tile(ix: str, iy: str, assembly_config: Dict[str, Any], output_base_path: str):
    """Process a single tile according to assembly configuration."""
    logger.debug(f"Processing tile ix={ix}, iy={iy}")
    
    datasets_config = assembly_config['datasets']
    processing_config = assembly_config.get('processing', {})
    demean_cols = processing_config.get('demean_columns', [])
    
    # Load first dataset to start with
    merged = None
    dataset_names = list(datasets_config.keys())
    
    for i, dataset_name in enumerate(dataset_names):
        dataset_config = datasets_config[dataset_name]
        dataset = load_tile_data(dataset_config, ix, iy)
        
        if dataset is not None:
            logger.debug(f"Loading dataset {dataset_name} for tile ix={ix}, iy={iy}")
            
            # Apply demeaning before merging if configured
            if demean_cols:
                dataset = demean_columns(dataset, demean_cols)
            
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
                    merged = ps.merge(merged_reset, dataset_reset, on=merge_cols, how=join_type)
        else:
            logger.debug(f"No data found for {dataset_name} in tile ix={ix}, iy={iy}")
    
    if merged is None:
        logger.debug(f"No data for tile ix={ix}, iy={iy}, skipping")
        return
    
    # Apply land mask filtering if configured
    if processing_config.get('filter_land_only', False) and 'land_mask' in merged.columns:
        logger.debug(f"Filtering to land pixels for tile ix={ix}, iy={iy}")
        merged = merged[merged["land_mask"]]
        merged = merged.drop(columns=["land_mask"])
    
    # Write tile to output
    tile_output_path = os.path.join(output_base_path, f"ix={ix}", f"iy={iy}")
    os.makedirs(tile_output_path, exist_ok=True)
    
    compression = processing_config.get('compression', 'zstd')
    output_file = os.path.join(tile_output_path, "data.parquet")
    
    logger.debug(f"Writing tile ix={ix}, iy={iy} to {output_file}")
    merged.to_parquet(output_file, compression=compression)

def run_assembly(assembly_config: Dict[str, Any]):
    """Run the data assembly process based on configuration using context manager."""
    logger.info(f"Starting assembly: {assembly_config.get('description', 'Unknown')}")
    
    spark_config = assembly_config.get('spark', {})
    
    with SparkSessionContextManager(
        config={'spark': spark_config},
        app_name=spark_config.get('app_name', 'DataAssembly')
    ) as spark:
        # Get output path and create directory
        output_path = assembly_config['output_path']
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Output will be written to: {output_path}")
        
        # Get all dataset paths for tile discovery
        dataset_paths = [config['path'] for config in assembly_config['datasets'].values()]
        
        # Discover available tiles
        logger.info("Discovering available tiles...")
        all_tiles = get_available_tiles(dataset_paths)
        logger.info(f"Found {len(all_tiles)} tiles to process")
        
        if not all_tiles:
            logger.warning("No tiles found to process")
            return
        
        # Process each tile
        for i, (ix, iy) in enumerate(all_tiles):
            logger.info(f"Processing tile {i+1}/{len(all_tiles)}: ix={ix}, iy={iy}")
            try:
                process_tile(ix, iy, assembly_config, output_path)
            except Exception as e:
                logger.error(f"Failed to process tile ix={ix}, iy={iy}: {e}", exc_info=True)
                continue
        
        logger.info("Assembly process completed successfully")

def run_workflow_with_config(config: Dict[str, Any]):
    """Entry point for running assembly workflow with unified configuration."""
    assembly_name = config.get('assembly_name', 'main')
    
    # Get assembly configuration
    if 'assemble' not in config or assembly_name not in config['assemble']:
        raise ValueError(f"Assembly configuration '{assembly_name}' not found in config")
    
    assembly_config = config['assemble'][assembly_name]
    
    # Run the assembly
    run_assembly(assembly_config)

if __name__ == "__main__":
    logger.error("This module should be run through the unified run.py interface")
    logger.error("Usage: python run.py assemble --config config.yaml --source assembly_name")
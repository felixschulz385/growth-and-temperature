import argparse
import os
import shutil
import zarr
import xarray as xr
from pathlib import Path
import glob
import logging
from datetime import datetime

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def rechunk_zarr_file(zarr_path, chunk_size=256, logger=None):
    """
    Rechunk a single zarr file to specified chunk size and rename original.
    
    Args:
        zarr_path: Path to zarr file
        chunk_size: Size for spatial chunks (default 256)
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    zarr_path = Path(zarr_path)
    
    # Create backup name
    backup_path = zarr_path.with_name(f"{zarr_path.name}_original")
    temp_path = zarr_path.with_name(f"{zarr_path.name}_temp")
    
    # Skip if already processed (backup exists)
    if backup_path.exists():
        logger.info(f"Skipping {zarr_path} - already processed (backup exists)")
        return
    
    logger.info(f"Processing: {zarr_path}")
    
    try:
        ds = xr.open_zarr(zarr_path)
        
        # Log dataset info
        logger.info(f"  Dataset shape: {dict(ds.sizes)}")
        logger.info(f"  Data variables: {list(ds.data_vars.keys())}")
        
        # Determine chunk dimensions
        chunks = {}
        for dim in ds.dims:
            if dim in ['x', 'y', 'lon', 'lat', 'longitude', 'latitude']:
                chunks[dim] = chunk_size
            else:
                # Keep time and other dimensions unchunked or small
                chunks[dim] = ds.sizes[dim] if ds.sizes[dim] <= 100 else 100
        
        logger.info(f"  Rechunking to: {chunks}")
        
        # Rechunk the dataset
        ds_rechunked = ds.chunk(chunks)
        
        # Clear existing chunk encodings to avoid conflicts
        for var_name in ds_rechunked.data_vars:
            if 'chunks' in ds_rechunked[var_name].encoding:
                del ds_rechunked[var_name].encoding['chunks']
        
        # Clear coordinate encodings as well
        for coord_name in ds_rechunked.coords:
            if 'chunks' in ds_rechunked[coord_name].encoding:
                del ds_rechunked[coord_name].encoding['chunks']
        
        logger.info(f"  Writing rechunked data to: {temp_path}")
        ds_rechunked.to_zarr(temp_path, mode='w')
        
        # Close datasets
        ds.close()
        ds_rechunked.close()
        
        # Rename original to backup
        logger.info(f"  Creating backup: {backup_path}")
        shutil.move(str(zarr_path), str(backup_path))
        
        # Move temp to original location
        logger.info(f"  Moving rechunked data to: {zarr_path}")
        shutil.move(str(temp_path), str(zarr_path))
        
        logger.info(f"  Successfully completed: {zarr_path}")
        
    except Exception as e:
        logger.error(f"  Error processing {zarr_path}: {e}")
        # Cleanup temp file if it exists
        if temp_path.exists():
            logger.info(f"  Cleaning up temporary files: {temp_path}")
            shutil.rmtree(temp_path)
        raise

def find_stage1_directories(base_path):
    """
    Find all stage_1 directories in the project structure.
    
    Args:
        base_path: Base project path
        
    Returns:
        List of stage_1 directory paths
    """
    base_path = Path(base_path)
    stage1_dirs = []
    
    # Look for stage_1 directories in the data structure
    data_dirs = [
        base_path / "data_nobackup",
        base_path / "processed"
    ]
    
    for data_dir in data_dirs:
        if data_dir.exists():
            # Find all stage_1 directories recursively
            for stage1_path in data_dir.rglob("stage_1"):
                if stage1_path.is_dir():
                    stage1_dirs.append(stage1_path)
    
    return stage1_dirs

def rechunk_annual_zarrs(base_path, chunk_size=256):
    """
    Rechunk all zarr files in processed/stage_1/[YYYY] directories.
    
    Args:
        base_path: Base project path
        chunk_size: Size for spatial chunks (default 256)
    """
    logger = setup_logging()
    
    base_path = Path(base_path)
    
    logger.info(f"Starting zarr rechunking process")
    logger.info(f"Base path: {base_path}")
    logger.info(f"Target chunk size: {chunk_size}")
    
    # Find all stage_1 directories
    stage1_dirs = find_stage1_directories(base_path)
    
    if not stage1_dirs:
        logger.warning(f"No stage_1 directories found in {base_path}")
        return
    
    logger.info(f"Found {len(stage1_dirs)} stage_1 directories:")
    for stage1_path in stage1_dirs:
        logger.info(f"  {stage1_path}")
    
    total_files = 0
    processed_files = 0
    skipped_files = 0
    failed_files = 0
    
    for stage1_path in stage1_dirs:
        logger.info(f"Processing stage_1 directory: {stage1_path}")
        
        # Check for two different structures:
        # 1. YYYY.zarr files directly in stage_1 (new structure)
        # 2. YYYY/ subdirectories containing *.zarr files (old structure)
        
        # Find yearly zarr files directly in stage_1
        yearly_zarr_files = [f for f in stage1_path.glob("*.zarr") 
                           if f.stem.isdigit() and len(f.stem) == 4]
        
        # Find year directories containing zarr files
        year_dirs = [d for d in stage1_path.iterdir() 
                     if d.is_dir() and d.name.isdigit() and len(d.name) == 4]
        
        # Process yearly zarr files directly in stage_1
        if yearly_zarr_files:
            logger.info(f"  Found {len(yearly_zarr_files)} yearly zarr files in {stage1_path}")
            total_files += len(yearly_zarr_files)
            
            for zarr_file in sorted(yearly_zarr_files):
                try:
                    # Check if already processed
                    backup_path = zarr_file.with_name(f"{zarr_file.name}_original")
                    if backup_path.exists():
                        logger.info(f"Skipping {zarr_file} - already processed (backup exists)")
                        skipped_files += 1
                        continue
                        
                    rechunk_zarr_file(zarr_file, chunk_size, logger)
                    processed_files += 1
                except Exception as e:
                    logger.error(f"  Failed to process {zarr_file}: {e}")
                    failed_files += 1
        
        # Process year directories containing zarr files
        for year_dir in sorted(year_dirs):
            logger.info(f"  Processing year directory: {year_dir}")
            
            # Find all zarr files in this year directory
            zarr_files = list(year_dir.glob("*.zarr"))
            
            if not zarr_files:
                logger.info(f"    No zarr files found in {year_dir}")
                continue
            
            logger.info(f"    Found {len(zarr_files)} zarr files")
            total_files += len(zarr_files)
            
            for zarr_file in sorted(zarr_files):
                try:
                    # Check if already processed
                    backup_path = zarr_file.with_name(f"{zarr_file.name}_original")
                    if backup_path.exists():
                        logger.info(f"Skipping {zarr_file} - already processed (backup exists)")
                        skipped_files += 1
                        continue
                        
                    rechunk_zarr_file(zarr_file, chunk_size, logger)
                    processed_files += 1
                except Exception as e:
                    logger.error(f"    Failed to process {zarr_file}: {e}")
                    failed_files += 1
    
    logger.info(f"Rechunking process completed")
    logger.info(f"Summary: {processed_files} processed, {skipped_files} skipped, {failed_files} failed out of {total_files} total files")

def main():
    parser = argparse.ArgumentParser(description="Rechunk all zarr files in annual directories")
    parser.add_argument("--base-path", 
                       default="/scicore/home/meiera/schulz0022/projects/growth-and-temperature",
                       help="Base project path")
    parser.add_argument("--chunk-size", type=int, default=256, 
                       help="Chunk size for spatial dimensions (default: 256)")
    
    args = parser.parse_args()
    
    try:
        rechunk_annual_zarrs(args.base_path, args.chunk_size)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Fatal error: {e}")
        exit(1)

if __name__ == "__main__":
    main()

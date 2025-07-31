import os
import logging
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from typing import Dict, Any, List
from pathlib import Path

# Add PyArrow for efficient parquet operations
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    pa = None
    pq = None

logger = logging.getLogger(__name__)

def process_zarr_to_parquet_vectorized(ds: xr.Dataset, output_path: str, batch_size: int = 32, 
                                     hpc_root: str = None) -> bool:
    """
    Generalized zarr to parquet processing using vectorized operations and efficient memory management.
    
    Args:
        ds: xarray Dataset
        output_path: Output parquet file path
        batch_size: Number of spatial rows to process at once
        hpc_root: HPC root path for land mask lookup
    """
    try:
        # Remove temporary output directory if it exists
        temp_output_dir = f"{output_path}_temp"
        if os.path.exists(temp_output_dir):
            import shutil
            shutil.rmtree(temp_output_dir)
        os.makedirs(temp_output_dir, exist_ok=True)
        
        # Get dimensions
        n_lat, n_lon = ds.sizes['latitude'], ds.sizes['longitude']
        n_time = ds.sizes['time']
        
        logger.info(f"Processing {n_lat}x{n_lon} spatial grid with {n_time} time steps")
        logger.info(f"Using optimized batch size: {batch_size}")
        logger.info(f"Dataset chunks: lat={ds.chunks.get('latitude')}, lon={ds.chunks.get('longitude')}")
        
        # Pre-compute global arrays once using efficient operations
        time_years = pd.DatetimeIndex(ds.time.values).year - 1900
        lats = ds.latitude.values
        lons = ds.longitude.values
        
        # Create coordinate grids once
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Create global index matrix with chunking that matches the dataset
        global_index_matrix = create_index_matrix_optimized(ds)
        
        # Get land mask if available
        land_mask = None
        if hpc_root:
            try:
                land_mask = get_or_create_land_mask_optimized(ds, hpc_root)
                # Apply land mask
                ds = apply_land_mask_optimized(ds, land_mask)
                logger.info("Applied land mask to dataset")
            except Exception as e:
                logger.warning(f"Could not apply land mask: {e}")
        
        # Define the schema for consistency based on available variables
        schema = get_parquet_schema_flexible(ds)
        
        # Configure Dask to minimize chunk operations
        import dask
        with dask.config.set({
            'array.slicing.split_large_chunks': False,
            'array.chunk-size': '1GB',
            'optimization.fuse.active': True,
        }):
            
            # Process batches with optimized approach
            batch_files = []
            total_rows_written = 0
            
            # Use Dask delayed for parallel processing
            from dask import delayed
            
            # Create delayed tasks for each batch
            delayed_tasks = []
            for lat_start in range(0, n_lat, batch_size):
                lat_end = min(lat_start + batch_size, n_lat)
                
                task = delayed(process_batch_vectorized_optimized)(
                    ds,
                    lat_start,
                    lat_end,
                    temp_output_dir,
                    time_years,
                    lat_grid[lat_start:lat_end, :],
                    lon_grid[lat_start:lat_end, :],
                    global_index_matrix,
                    schema,
                    len(delayed_tasks)
                )
                delayed_tasks.append(task)
            
            logger.info(f"Created {len(delayed_tasks)} optimized batch tasks")
            
            batch_results = dask.compute(*delayed_tasks)
        
        # Collect successful batch files
        for result in batch_results:
            if result['success'] and result['batch_file']:
                batch_files.append(result['batch_file'])
                total_rows_written += result['rows_written']
        
        if not batch_files:
            logger.warning("No data to write - all batches were empty")
            return False
        
        # Combine all batch files into final parquet file
        logger.info(f"Combining {len(batch_files)} batch files into final parquet file")
        combine_parquet_files_optimized(batch_files, output_path, schema)
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_output_dir)
        
        logger.info(f"Successfully wrote {total_rows_written:,} rows to {output_path}")
        return True
        
    except Exception as e:
        logger.exception(f"Error in vectorized parquet processing: {e}")
        return False

def create_index_matrix_optimized(ds: xr.Dataset) -> da.Array:
    """Create index matrix with optimized chunking to match dataset structure."""
    # Get dataset dimensions and chunking
    n_lat, n_lon = ds.sizes['latitude'], ds.sizes['longitude']
    
    # Use chunking that matches the dataset to prevent rechunking operations
    lat_chunks = ds.chunks.get('latitude', (n_lat,))
    lon_chunks = ds.chunks.get('longitude', (n_lon,))
    
    # Create index matrix as a dask array with proper chunking
    total_size = n_lat * n_lon
    id_vector = da.arange(total_size, dtype=np.uint32, chunks=(np.prod(lat_chunks[0]) * lon_chunks[0],))
    
    # Reshape first, then rechunk to match the dataset structure
    id_matrix = da.reshape(id_vector, (n_lat, n_lon))
    id_matrix = id_matrix.rechunk((lat_chunks, lon_chunks))
    
    logger.debug(f"Created optimized index matrix with chunks: {id_matrix.chunks}")
    return id_matrix

def get_or_create_land_mask_optimized(ds: xr.Dataset, hpc_root: str) -> xr.DataArray:
    """Get land mask with optimized chunking to match the dataset."""
    # Construct the expected land mask path relative to hpc_root
    land_mask_path = os.path.join(
        hpc_root, 
        "misc", 
        "processed", 
        "stage_2", 
        "osm", 
        "land_mask.zarr"
    )
    
    if not os.path.exists(land_mask_path):
        raise FileNotFoundError(
            f"Land mask not found at expected location: {land_mask_path}. "
            "Please ensure the misc data has been processed through stage 2 "
            "to generate the land mask from OpenStreetMap data."
        )
    
    try:
        # Load land mask with native chunking first
        logger.info("Loading land mask with native chunking")
        land_mask_ds = xr.open_zarr(land_mask_path)
        
        # Extract the land mask array - assume it's the first data variable
        if len(land_mask_ds.data_vars) == 0:
            raise ValueError("Land mask zarr file contains no data variables")
        
        land_mask_var = list(land_mask_ds.data_vars)[0]
        land_mask = land_mask_ds[land_mask_var]
        
        # Ensure the land mask coordinates match the dataset exactly
        if not coordinates_match(ds, land_mask):
            raise ValueError(
                "Land mask coordinates don't match dataset coordinates. "
                "The land mask must be on the same grid as the reprojected data. "
                "Please ensure the misc data was processed with the same target geobox."
            )
        
        # Rechunk to exactly match dataset chunking
        target_chunks = {
            'latitude': ds.chunks['latitude'] if 'latitude' in ds.chunks else (ds.sizes['latitude'],),
            'longitude': ds.chunks['longitude'] if 'longitude' in ds.chunks else (ds.sizes['longitude'],)
        }
        
        land_mask = land_mask.chunk(target_chunks)
        
        logger.info(f"Successfully loaded and optimized land mask from: {land_mask_path}")
        return land_mask
        
    except Exception as e:
        raise RuntimeError(f"Failed to load land mask from {land_mask_path}: {e}")

def apply_land_mask_optimized(ds: xr.Dataset, land_mask: xr.DataArray) -> xr.Dataset:
    """Apply land mask with optimized operations to minimize chunk proliferation."""
    try:
        logger.info("Applying land mask with optimized chunking")
        
        # Convert land_mask to boolean explicitly
        land_mask_bool = land_mask.astype(bool)
        
        # Apply mask to each variable in the dataset
        masked_vars = {}
        for var_name, var_data in ds.data_vars.items():
            logger.debug(f"Applying land mask to variable: {var_name}")
            masked_vars[var_name] = var_data.where(land_mask_bool)
        
        # Create new dataset with masked variables
        ds_masked = xr.Dataset(masked_vars, coords=ds.coords, attrs=ds.attrs)
        
        logger.info("Land mask applied successfully with optimized chunking")
        return ds_masked
        
    except Exception as e:
        logger.error(f"Error applying land mask: {e}")
        # Fallback to original method if optimized version fails
        return ds.where(land_mask)

def coordinates_match(ds: xr.Dataset, land_mask: xr.DataArray) -> bool:
    """Check if dataset and land mask have matching coordinates."""
    try:
        # Check spatial coordinates
        ds_x = ds.coords.get('x', ds.coords.get('longitude'))
        ds_y = ds.coords.get('y', ds.coords.get('latitude'))
        mask_x = land_mask.coords.get('x', land_mask.coords.get('longitude'))
        mask_y = land_mask.coords.get('y', land_mask.coords.get('latitude'))
        
        if ds_x is None or ds_y is None or mask_x is None or mask_y is None:
            return False
        
        # Check if coordinate arrays are approximately equal
        x_match = np.allclose(ds_x.values, mask_x.values, rtol=1e-6)
        y_match = np.allclose(ds_y.values, mask_y.values, rtol=1e-6)
        
        return x_match and y_match
        
    except Exception:
        return False

def get_parquet_schema_flexible(ds: xr.Dataset) -> pa.Schema:
    """Define PyArrow schema based on available variables in the dataset."""
    if not PYARROW_AVAILABLE:
        raise ImportError("PyArrow is required for parquet operations")
    
    # Base schema fields
    schema_fields = [
        ('id', pa.uint32()),
        ('latitude', pa.float32()),
        ('longitude', pa.float32()),
        ('time', pa.uint8()),
    ]
    
    # Add fields for each data variable in the dataset
    for var_name in ds.data_vars:
        # Determine appropriate data type based on variable characteristics
        var_data = ds[var_name]
        
        # Check if variable has scale_factor (indicating it's scaled integer data)
        if hasattr(var_data, 'attrs') and 'scale_factor' in var_data.attrs:
            schema_fields.append((var_name, pa.uint16()))
        else:
            # For other variables, use uint16 as default (can be adjusted per source)
            schema_fields.append((var_name, pa.uint16()))
    
    return pa.schema(schema_fields)

def process_batch_vectorized_optimized(ds: xr.Dataset, lat_start: int, lat_end: int, 
                                     temp_output_dir: str, time_years: np.ndarray,
                                     lat_grid_batch: np.ndarray, lon_grid_batch: np.ndarray,
                                     global_index_matrix: da.Array, schema: pa.Schema, 
                                     batch_idx: int) -> Dict[str, Any]:
    """Optimized batch processing with reduced chunk operations and memory efficiency."""
    try:
        logger.debug(f"Processing optimized batch {batch_idx}: latitude {lat_start}:{lat_end}")
        
        # Extract spatial subset of the index matrix for this batch
        index_matrix_batch = global_index_matrix[lat_start:lat_end, :].compute()
        
        # Use more efficient data extraction approach
        batch_data = {}
        
        # Extract data for all variables at once to minimize zarr reads
        for var in ds.data_vars:
            # Use isel with computed=False to create lazy selection, then compute once
            var_subset = ds[var].isel(latitude=slice(lat_start, lat_end))
            batch_data[var] = var_subset.compute().values  # Compute once and extract values
        
        if not batch_data:
            logger.warning(f"No valid variables found in batch {batch_idx}")
            return {'success': True, 'batch_file': None, 'rows_written': 0, 'batch_idx': batch_idx}
        
        # Get batch dimensions from the first variable
        first_var_data = next(iter(batch_data.values()))
        n_times, n_lats_batch, n_lons = first_var_data.shape
        
        # Validate index matrix dimensions
        if index_matrix_batch.shape != (n_lats_batch, n_lons):
            logger.error(f"Index matrix shape mismatch: {index_matrix_batch.shape} vs ({n_lats_batch}, {n_lons})")
            return {'success': False, 'batch_file': None, 'rows_written': 0, 'batch_idx': batch_idx}
        
        # Create arrays for DataFrame construction using vectorized operations
        total_records = n_times * n_lats_batch * n_lons
        
        # Pre-allocate arrays for all columns with proper data types
        df_data = {
            'id': np.zeros(total_records, dtype=np.uint32),
            'latitude': np.zeros(total_records, dtype=np.float32),
            'longitude': np.zeros(total_records, dtype=np.float32),
            'time': np.zeros(total_records, dtype=np.uint8)
        }
        
        # Add variable arrays
        for var in batch_data.keys():
            df_data[var] = np.zeros(total_records, dtype=np.uint16)
        
        # Use vectorized operations to fill arrays efficiently
        time_broadcast = np.repeat(time_years, n_lats_batch * n_lons)
        id_broadcast = np.tile(index_matrix_batch.flatten(), n_times)
        lat_broadcast = np.tile(lat_grid_batch.flatten(), n_times)
        lon_broadcast = np.tile(lon_grid_batch.flatten(), n_times)
        
        # Fill coordinate data
        df_data['time'][:] = time_broadcast
        df_data['id'][:] = id_broadcast
        df_data['latitude'][:] = lat_broadcast
        df_data['longitude'][:] = lon_broadcast
        
        # Fill variable data using efficient reshaping
        for var, var_values in batch_data.items():
            # Reshape from (time, lat, lon) to (total_records,) using C-order
            df_data[var][:] = var_values.reshape(-1, order='C')
        
        # Create DataFrame from pre-allocated arrays
        df = pd.DataFrame(df_data)
        
        # Apply data type optimization
        df = optimize_dataframe_dtypes(df)
        
        result = {
            'success': False,
            'batch_file': None,
            'rows_written': 0,
            'batch_idx': batch_idx
        }
        
        if len(df) > 0:
            # Write batch to temporary parquet file
            batch_file = os.path.join(temp_output_dir, f"batch_{lat_start:06d}_{lat_end:06d}.parquet")
            
            # Convert to PyArrow table with consistent schema
            table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
            pq.write_table(table, batch_file, compression='snappy')
            
            result.update({
                'success': True,
                'batch_file': batch_file,
                'rows_written': len(df)
            })
            
            logger.debug(f"Completed optimized batch {batch_idx}: {lat_start}:{lat_end} with {len(df):,} rows")
        else:
            result['success'] = True  # Still successful, just empty
        
        return result
        
    except Exception as e:
        logger.exception(f"Error processing optimized batch {batch_idx}: {e}")
        return {
            'success': False,
            'batch_file': None,
            'rows_written': 0,
            'batch_idx': batch_idx,
            'error': str(e)
        }

def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame data types for memory efficiency."""
    # Convert to optimal dtypes
    df['id'] = df['id'].astype('uint32')
    df['latitude'] = df['latitude'].astype('float32')
    df['longitude'] = df['longitude'].astype('float32')
    df['time'] = df['time'].astype('uint8')
    
    # Convert other variables to uint16 (can be customized per source)
    for col in df.columns:
        if col not in ['id', 'latitude', 'longitude', 'time']:
            df[col] = df[col].astype('uint16')
    
    return df

def combine_parquet_files_optimized(batch_files: List[str], output_path: str, schema: pa.Schema):
    """Optimized parquet file combination with streaming and memory management."""
    try:
        logger.info("Starting optimized parquet file combination")
        
        # Use PyArrow's optimized concatenation with streaming
        dataset = pq.ParquetDataset(batch_files, schema=schema)
        
        # Write with optimized settings for large files
        pq.write_table(
            dataset.read(),
            output_path,
            compression='snappy',
            use_dictionary=['id'],  # Only use dictionary encoding for ID column
            write_statistics=True,
            row_group_size=100000,  # Larger row groups for better compression
            use_compliant_nested_type=False,  # Faster writing
            coerce_timestamps='ms'  # Consistent timestamp handling
        )
        
        logger.info("Optimized parquet combination completed")
        
    except Exception as e:
        logger.warning(f"Optimized combination failed, falling back to standard method: {e}")
        # Fallback to original method
        combine_parquet_files(batch_files, output_path, schema)

def combine_parquet_files(batch_files: List[str], output_path: str, schema: pa.Schema):
    """Combine multiple parquet files into a single file."""
    # Read all batch files and combine
    tables = []
    for batch_file in batch_files:
        table = pq.read_table(batch_file)
        tables.append(table)
    
    # Concatenate all tables
    combined_table = pa.concat_tables(tables)
    
    # Write final parquet file with compression
    pq.write_table(
        combined_table, 
        output_path,
        compression='snappy',
        use_dictionary=True,
        write_statistics=True,
        row_group_size=50000
    )

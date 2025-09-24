import os
import logging
import tempfile
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from zarr.codecs import BloscCodec
from odc.geo import GeoboxTiles
from odc.geo.xr import xr_reproject
from gnt.data.common.geobox.geobox import get_or_create_geobox

logger = logging.getLogger(__name__)

class SpatialProcessor:
    """
    Common spatial processing utilities for reprojecting data to unified grids.
    
    This class provides shared functionality for spatial stage processing while
    allowing source-specific customization through callback functions.
    """
    
    def __init__(self, hpc_root: str, temp_dir: str = None, dask_client=None):
        """
        Initialize spatial processor.
        
        Args:
            hpc_root: HPC root directory for geobox
            temp_dir: Temporary directory for processing
            dask_client: Optional Dask client context manager
        """
        self.hpc_root = hpc_root
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="spatial_processor_")
        self.dask_client = dask_client
        
    def get_target_geobox(self):
        """Get or create the target geobox for reprojection."""
        try:
            target_geobox = get_or_create_geobox(self.hpc_root)
            logger.info(f"Using target geobox for reprojection: {target_geobox.shape}")
            return target_geobox
        except Exception as e:
            logger.error(f"Failed to get target geobox: {e}")
            raise
    
    def create_empty_target_zarr(
        self,
        output_path: str,
        target_geobox,
        years: List[int],
        variables: List[str],
        sample_attrs: Dict[str, Any] = None,
        variable_attrs_func: Callable[[str, Dict], Dict] = None
    ) -> bool:
        """
        Create empty zarr file with target dimensions and metadata.
        
        Args:
            output_path: Path for output zarr
            target_geobox: Target geobox for spatial dimensions
            years: List of years for time dimension
            variables: List of variable names
            sample_attrs: Global attributes for the dataset
            variable_attrs_func: Function to get variable-specific attributes
            
        Returns:
            bool: Success status
        """
        try:
            logger.info("Creating empty target zarr file")
            
            # Create time coordinates
            time_coords = pd.to_datetime([f"{year}-12-31" for year in sorted(years)])
            
            # Create empty dataset with target geobox dimensions
            ny, nx = target_geobox.shape
            lat_coords = target_geobox.coords['latitude'].values.round(5)
            lon_coords = target_geobox.coords['longitude'].values.round(5)
            
            # Create data variables with fill values and band dimension
            data_vars = {}
            
            default_attrs = {"_FillValue": 65535}
            packaging_attrs = {
                "scale_factor": 0.01,
                "add_offset": 0.0
            }
            
            for var in variables:
                # Get variable-specific attributes
                if variable_attrs_func:
                    var_attrs = variable_attrs_func(var, default_attrs.copy())
                else:
                    var_attrs = default_attrs.copy()
                    var_attrs.update(packaging_attrs)
                    
                data_vars[var] = xr.DataArray(
                    da.zeros((len(time_coords), 1, ny, nx), dtype=np.uint16, chunks=(1, 1, 512, 512)),
                    dims=['time', 'band', 'latitude', 'longitude'],
                    coords={
                        'time': time_coords,
                        'band': [1],
                        'latitude': lat_coords,
                        'longitude': lon_coords
                    },
                    attrs=var_attrs
                )
            
            # Create empty dataset and copy global attributes
            empty_ds = xr.Dataset(data_vars, attrs=sample_attrs or {})
            
            # Set CRS
            empty_ds = empty_ds.rio.write_crs(target_geobox.crs)
            
            # Set up compression for Zarr output
            compressor = BloscCodec(cname="zstd", clevel=3, shuffle='bitshuffle', blocksize=0)
            encoding = {
                var: {
                    "chunks": (1, 1, 512, 512), 
                    "compressors": (compressor,),
                    "dtype": "uint16"
                } 
                for var in variables
            }
            
            # Write empty zarr structure
            logger.info(f"Writing empty zarr structure to: {output_path}")
            empty_ds.to_zarr(
                output_path, 
                mode="w",
                encoding=encoding,
                compute=False,
                zarr_format = 3,
                consolidated = False
            )
            
            logger.info("Empty target zarr created successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Error creating empty target zarr: {e}")
            return False
    
    def setup_dask_config(self):
        """Set up Dask configuration for large array operations."""
        import dask
        return dask.config.set({
            'array.slicing.split_large_chunks': True,
            'array.chunk-size': '512MB',
            'optimization.fuse.active': False,
            'distributed.comm.compression': 'lz4',
        })
    
    def group_files_by_year(self, source_files: List[str], year_pattern_func: Callable[[str], Optional[int]]) -> Dict[int, List[str]]:
        """
        Group source files by year using a custom pattern function.
        
        Args:
            source_files: List of source file paths
            year_pattern_func: Function to extract year from file path
            
        Returns:
            Dict mapping year to list of file paths
        """
        files_by_year = {}
        
        for file_path in source_files:
            year = year_pattern_func(file_path)
            if year is not None:
                if year not in files_by_year:
                    files_by_year[year] = []
                files_by_year[year].append(file_path)
        
        return files_by_year
    
    def write_year_to_zarr(
        self,
        year_ds: xr.Dataset,
        output_path: str,
        year: int,
        target_geobox,
        preprocess_func: Callable[[xr.Dataset], xr.Dataset] = None
    ) -> bool:
        """
        Write a year's worth of data to zarr with reprojection.
        
        Args:
            year_ds: Source dataset for the year
            output_path: Output zarr path
            year: Year being processed
            target_geobox: Target geobox for reprojection
            preprocess_func: Optional preprocessing function
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Processing year {year} for spatial reprojection")
            
            # Apply preprocessing if provided
            if preprocess_func:
                year_ds = preprocess_func(year_ds)
            
            # Reproject to target geobox
            reprojected_ds = xr_reproject(year_ds, target_geobox, resampling="nearest")
            
            # Clean up dataset
            reprojected_ds = reprojected_ds.drop_vars(['spatial_ref'], errors='ignore').drop_attrs()
            
            # Transform coordinates
            reprojected_ds.coords['longitude'] = reprojected_ds.coords['longitude'].round(5)
            reprojected_ds.coords['latitude'] = reprojected_ds.coords['latitude'].round(5)
            
            # Rechunk for zarr writing
            reprojected_ds = reprojected_ds.chunk({'time': 1, 'band': 1, 'latitude': 512, 'longitude': 512})
            
            # Write to zarr
            reprojected_ds.to_zarr(
                output_path,
                region='auto',
                align_chunks=True,
                zarr_format=3,
                consolidated=False
            )
            
            logger.info(f"Successfully wrote year {year} to zarr")
            return True
            
        except Exception as e:
            logger.exception(f"Error writing year {year} to zarr: {e}")
            return False
    
    def process_spatial_standard(
        self,
        source_files: List[str],
        output_path: str,
        years_to_process: List[int],
        year_pattern_func: Callable[[str], Optional[int]],
        preprocess_func: Callable[[xr.Dataset], xr.Dataset] = None,
        get_variables_func: Callable[[str], Tuple[List[str], Dict]] = None
    ) -> bool:
        """
        Standard spatial processing workflow for simple cases.
        
        This handles the common case where each year has one file and minimal
        aggregation is needed.
        
        Args:
            source_files: List of source zarr files
            output_path: Output zarr path
            years_to_process: List of years to include
            year_pattern_func: Function to extract year from file path
            preprocess_func: Optional preprocessing function for each dataset
            get_variables_func: Function to get variables and attrs from sample file
            
        Returns:
            bool: Success status
        """
        try:
            # Get target geobox
            target_geobox = self.get_target_geobox()
            
            # Group files by year
            files_by_year = self.group_files_by_year(source_files, year_pattern_func)
            
            # Get variables and attributes from sample file
            if get_variables_func:
                variables, sample_attrs = get_variables_func(source_files[0])
            else:
                sample_ds = xr.open_zarr(source_files[0], mask_and_scale=False, chunks='auto')
                variables = list(sample_ds.data_vars.keys())
                sample_attrs = sample_ds.attrs.copy()
                sample_ds.close()
            
            # Create empty target zarr
            if not self.create_empty_target_zarr(output_path, target_geobox, years_to_process, variables, sample_attrs):
                return False
            
            # Process each year
            for year in sorted(years_to_process):
                if year not in files_by_year:
                    logger.warning(f"No files found for year {year}")
                    continue
                
                year_files = files_by_year[year]
                if len(year_files) > 1:
                    logger.warning(f"Multiple files found for year {year}, using first: {year_files[0]}")
                
                # Open year dataset
                year_ds = xr.open_zarr(year_files[0], consolidated=False, decode_coords='all')
                
                # Write to output zarr
                success = self.write_year_to_zarr(
                    year_ds, output_path, year, target_geobox, preprocess_func
                )
                
                year_ds.close()
                
                if not success:
                    logger.error(f"Failed to process year {year}")
                    return False
            
            logger.info("Standard spatial processing completed successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Error in standard spatial processing: {e}")
            return False


def create_zarr_encoding(variables: List[str], chunks: Tuple[int, ...] = (1, 1, 512, 512)) -> Dict[str, Dict]:
    """Create standard zarr encoding configuration."""
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle='bitshuffle', blocksize=0)
    return {
        var: {
            "chunks": chunks, 
            "compressors": (compressor,),
            "dtype": "uint16"
        } 
        for var in variables
    }

import os
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rxr
from datetime import datetime
import dask
from dask.distributed import Client, LocalCluster
import dask.array as da
import zarr
import re

from sources.base import AbstractPreprocessor
from gcs.client import GCSClient

logger = logging.getLogger(__name__)

class EOGPreprocessor(AbstractPreprocessor):
    """
    Preprocessor for Earth Observation Group (EOG) nighttime lights data.
    
    This class handles the preprocessing of both DMSP-OLS and VIIRS nighttime lights data.
    The preprocessing involves:
    1. Downloading raw data files from cloud storage
    2. Processing them into annual composites
    3. Saving the results as zarr files for efficient access and reprojection
    
    The preprocessing is organized in two stages:
    - Stage 1: Create annual zarr files from raw data
    - Stage 2: Reproject the data to a unified grid for analysis
    """
    
    # Cloud storage information
    BUCKET_NAME = "growthandheat"
    
    # Path prefixes in cloud storage
    DMSP_PATH_PREFIX = "glass/EOG/DMSP/Daily/1KM/"
    VIIRS_PATH_PREFIX = "glass/EOG/VIIRS/Daily/1KM/"
    
    # Variable name for the data
    DMSP_VARIABLE_NAME = "avg_vis"  # The primary variable in DMSP data
    VIIRS_VARIABLE_NAME = "DNB_BRDF_Corrected_NTL"  # The primary variable in VIIRS data
    
    # Chunking configuration for Dask processing
    DEFAULT_CHUNK_SIZE = {
        'time': 1,
        'band': 1,
        'y': 1000,
        'x': 1000
    }
    
    def __init__(self, **kwargs):
        """
        Initialize the EOG preprocessor.
        
        Args:
            **kwargs: Configuration parameters including:
                input_path (str): Path to input data
                output_path (str): Path for final output
                intermediate_path (str): Path for intermediate files
                data_source (str): "DMSP" or "VIIRS"
                years (list or tuple): Years to process
                grid_cells (list, optional): Grid cells to process (for VIIRS)
                override (bool): Whether to reprocess existing outputs
                stage (str): Processing stage ("stage1", "stage2", or "all")
                dask_threads (int): Number of threads for Dask processing
                dask_memory_limit (str): Memory limit for Dask
        """
        super().__init__(**kwargs)
        
        # Set data source (DMSP or VIIRS)
        self.data_source = kwargs.get('data_source', 'DMSP').upper()
        if self.data_source not in ['DMSP', 'VIIRS']:
            raise ValueError(f"Unsupported data source: {self.data_source}. Use 'DMSP' or 'VIIRS'.")
        
        # Set years to process
        years = kwargs.get('years', None)
        if years is None:
            # Default to processing all years
            self.years = list(range(1992, 2014)) if self.data_source == 'DMSP' else list(range(2012, 2024))
        elif isinstance(years, list):
            self.years = years
        elif isinstance(years, tuple) and len(years) == 2:
            self.years = list(range(years[0], years[1] + 1))
        else:
            raise ValueError("Years must be a list of years or a tuple (start_year, end_year)")
            
        # Whether to override existing processed files
        self.override = kwargs.get('override', False)
        
        # Path prefix based on data source
        self.path_prefix = self.DMSP_PATH_PREFIX if self.data_source == 'DMSP' else self.VIIRS_PATH_PREFIX
        self.variable_name = self.DMSP_VARIABLE_NAME if self.data_source == 'DMSP' else self.VIIRS_VARIABLE_NAME
        
        # Set directory paths for outputs
        self.intermediate_dir = kwargs.get('intermediate_dir', 'intermediate')
        self.output_dir = kwargs.get('output_dir', 'output')
        
        # Initialize GCS client
        self.gcs_client = GCSClient(self.BUCKET_NAME)
        
        # Dask configuration
        self.chunk_size = kwargs.get('chunk_size', self.DEFAULT_CHUNK_SIZE)
        self.dask_threads = kwargs.get('dask_threads', None)
        self.dask_memory_limit = kwargs.get('dask_memory_limit', None)
        
        logger.info(f"Initialized EOGPreprocessor for {self.data_source} data")
        logger.info(f"Processing years: {min(self.years)}-{max(self.years)}")
        
    def validate_input(self) -> bool:
        """
        Validate that the input data exists and is in an expected format.
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            # Check if source files exist in cloud storage
            file_pattern = "*.tif" if self.data_source == "DMSP" else "*.tif"
            files = self.gcs_client.list_blobs_with_limit(
                prefix=self.path_prefix,
                delimiter="/", 
                limit=5
            )
            
            # Ensure we have at least some files
            if not files:
                logger.error(f"No files found in {self.path_prefix}")
                return False
                
            logger.info(f"Found input files in {self.path_prefix}, validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating input: {str(e)}")
            return False
            
    def summarize_annual_means(self) -> None:
        """
        Process raw data into annual statistics (Stage 1).
        
        This processes all the raw data files into annual composites and saves
        them as zarr files, which are efficient for cloud storage and access.
        """
        if self.data_source == "DMSP":
            self._process_dmsp_data()
        else:
            self._process_viirs_data()
            
    def project_to_unified_grid(self) -> None:
        """
        Project processed data to a unified grid (Stage 2).
        
        This takes the processed zarr files from Stage 1 and reprojects them
        to a standard grid projection for analysis.
        """
        import warnings
        warnings.warn("Stage 2 projection not yet implemented for EOG data")
        logger.warning("Stage 2 projection not yet implemented for EOG data")
            
    def _process_dmsp_data(self) -> None:
        """
        Process DMSP-OLS nighttime lights data.
        
        Lists all DMSP files, groups them by year and satellite, and processes each group.
        """
        # List all DMSP files from cloud storage
        all_files = self._list_files_from_gcs()
        
        # Parse filenames to extract metadata
        files_df = self._parse_dmsp_filenames(all_files)
        
        # Filter by years
        files_df = files_df[files_df['year'].isin(self.years)]
        
        # Group by year and satellite
        grouped = files_df.groupby(['year', 'satellite'])
        
        total_groups = len(grouped)
        logger.info(f"Found {total_groups} year-satellite combinations to process")
        
        # Process each group
        for i, ((year, satellite), group) in enumerate(grouped):
            output_path = f"{self.data_source.lower()}/{year}/{satellite}.zarr"
            
            # Skip if already processed and override is False
            if not self.override and self.gcs_client.check_if_exists(
                f"{self.path_prefix}{self.intermediate_dir}/{self.data_source.lower()}_{year}_{satellite}_annual.zarr/.zmetadata"):
                logger.info(f"Skipping existing output for {year} {satellite} ({i+1}/{total_groups})")
                continue
                
            logger.info(f"Processing {year} {satellite} ({i+1}/{total_groups})")
            
            try:
                # Process this group of files
                self._process_file_group(group, year, satellite, output_path)
                
            except Exception as e:
                logger.error(f"Error processing {year} {satellite}: {str(e)}")
                
    def _process_viirs_data(self) -> None:
        """
        Process VIIRS nighttime lights data.
        
        NOTE: This is a placeholder for future implementation.
        """
        logger.info("VIIRS preprocessing not yet implemented")
        pass
        
    def _list_files_from_gcs(self) -> List[str]:
        """
        List all relevant files from Google Cloud Storage.
        
        Returns:
            List[str]: List of file paths
        """
        search_path = self.path_prefix
        
        if self.data_source == "DMSP":
            # Search in the DMSP directory structure for .tif files
            file_pattern = "*.tif"
            logger.info(f"Listing DMSP files from {search_path}")
            files = self.gcs_client.list_files_with_pattern(search_path, file_pattern)
        else:
            # Search in the VIIRS directory structure for .tif files
            file_pattern = "*.tif"
            logger.info(f"Listing VIIRS files from {search_path}")
            files = self.gcs_client.list_files_with_pattern(search_path, file_pattern)
            
        logger.info(f"Found {len(files)} files matching pattern {file_pattern}")
        return files
        
    def _parse_dmsp_filenames(self, filenames: List[str]) -> pd.DataFrame:
        """
        Parse DMSP filenames to extract metadata.
        
        DMSP filename format is typically like: F101992.v4b.global.avg_vis.tif
        where F10 is the satellite, 1992 is the year.
        
        Args:
            filenames: List of filenames to parse
            
        Returns:
            pd.DataFrame: DataFrame with extracted metadata
        """
        result = []
        
        # Regex pattern to match DMSP filenames
        pattern = r'F(\d+)(\d{4})\.v4b\.global\.([^.]+)\.tif$'
        
        for filename in filenames:
            try:
                # Extract base filename from path
                basename = os.path.basename(filename)
                
                match = re.search(pattern, basename)
                if not match:
                    continue
                    
                satellite = f"F{match.group(1)}"
                year = int(match.group(2))
                product_type = match.group(3)
                
                # Only include stable_lights or avg_vis products
                if product_type in ['avg_vis', 'stable_lights']:
                    result.append({
                        'path': filename,
                        'year': year,
                        'satellite': satellite,
                        'product_type': product_type
                    })
            except (IndexError, ValueError) as e:
                logger.warning(f"Could not parse filename {filename}: {str(e)}")
                
        return pd.DataFrame(result)
        
    def _process_file_group(self, files_df: pd.DataFrame, year: int, satellite: str, output_path: str) -> None:
        """
        Process a group of files for a specific year and satellite.
        
        Args:
            files_df: DataFrame with file information
            year: Year of the data
            satellite: Satellite identifier
            output_path: Path for output zarr file
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Processing {len(files_df)} files for {year} {satellite}")
            
            # Try to get a sample file to determine metadata
            sample_data = None
            for _, file_info in files_df.iterrows():
                try:
                    file_path = file_info['path']
                    local_path = os.path.join(temp_dir, os.path.basename(file_path))
                    
                    # Download file
                    self.gcs_client.download_file(file_path, local_path)
                    
                    # Open with rioxarray to get metadata
                    sample_data = rxr.open_rasterio(local_path)
                    break
                except Exception as e:
                    logger.warning(f"Error getting sample data from {file_info['path']}: {str(e)}")
            
            # Check if we successfully got sample data
            if sample_data is None:
                logger.warning(f"No valid sample data found for {year} {satellite}")
                return
            
            # Process each file and store in a list for later concatenation
            data_arrays = []
            for _, file_info in files_df.iterrows():
                file_path = file_info['path']
                local_path = os.path.join(temp_dir, os.path.basename(file_path))
                
                # Download file if not already present
                if not os.path.exists(local_path):
                    self.gcs_client.download_file(file_path, local_path)
                    
                # Open the data with chunking for Dask processing
                data = rxr.open_rasterio(
                    local_path,
                    chunks={k:self.chunk_size[k] for k in ('band','y','x') if k in self.chunk_size},
                )
                
                data_arrays.append(data)
            
            # Combine the time series into a single dataset
            # This is a placeholder where you would implement the logic for merging 
            # multiple files per year (if needed)
            combined_data = self._combine_time_series(data_arrays, files_df, year)
            
            # Calculate statistics
            annual_stats = self._calculate_statistics(combined_data)
            
            # Create output paths for cloud storage
            output_basename = f"{self.data_source.lower()}_{year}_{satellite}"
            annual_cloud_path = f"{self.path_prefix}{self.intermediate_dir}/{output_basename}_annual.zarr"
            
            # Save the data with compression
            compressor = zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle)
            encoding = {var: {'compressors': compressor} for var in annual_stats.data_vars}
            
            # Create temporary paths for zarr storage
            annual_output_path = Path(temp_dir) / f"{output_basename}_annual.zarr"
            
            # Save in smaller chunks to avoid large graphs
            logger.info(f"Saving annual statistics to temporary location with compression")
            annual_stats.to_zarr(
                str(annual_output_path),
                zarr_format=3,
                mode="w",
                consolidated=False,
                encoding=encoding
            )
            
            # Upload to cloud storage
            logger.info(f"Uploading annual statistics to cloud: {annual_cloud_path}")
            self._upload_to_cloud(annual_output_path, annual_cloud_path)
            
            logger.info(f"Completed processing for {year} {satellite}")
            
    def _combine_time_series(self, data_arrays: List[xr.DataArray], files_df: pd.DataFrame, year: int) -> xr.DataArray:
        """
        Combine multiple data arrays into a single time series.
        
        For DMSP, this might involve selecting the best quality data or averaging overlapping products.
        
        Args:
            data_arrays: List of data arrays
            files_df: DataFrame with file metadata
            year: Year of data
            
        Returns:
            xr.DataArray: Combined data array
        """
        # In a full implementation, you would:
        # 1. Properly handle multiple satellites for the same year
        # 2. Apply quality filters
        # 3. Handle temporal overlaps
        
        # For now, we'll use a simple approach of just taking the first array
        # and adding a time dimension to it
        if not data_arrays:
            raise ValueError(f"No data arrays to combine for year {year}")
            
        # Create a time coordinate for January 1 of the year
        time_coord = pd.to_datetime(f"{year}-01-01")
        
        # Take the first array and add time dimension
        combined = data_arrays[0].expand_dims({"time": [time_coord]})
        
        # In a more complex implementation, you would merge/average/select from multiple arrays here
        
        return combined
        
    def _calculate_statistics(self, data: xr.DataArray) -> xr.Dataset:
        """
        Calculate annual statistics from the time series data.
        
        For nighttime lights, we typically want:
        - Annual mean
        - Maximum value
        - Frequency of detection
        
        Args:
            data: Input data array with time dimension
            
        Returns:
            xr.Dataset: Dataset with calculated statistics
        """
        # For now, create a simple dataset with the mean value
        # In a full implementation, you would compute multiple statistics
        
        stats = xr.Dataset()
        
        # Add the mean as the primary statistic
        stats['mean'] = data.mean(dim='time')
        
        # Add other useful statistics
        if len(data.time) > 1:  # Only calculate these if we have multiple time points
            stats['max'] = data.max(dim='time')
            stats['min'] = data.min(dim='time')
            stats['std'] = data.std(dim='time')
            
            # Calculate frequency of valid observations
            valid_obs = (~np.isnan(data)).sum(dim='time')
            total_obs = len(data.time)
            stats['valid_fraction'] = valid_obs / total_obs
        
        return stats
        
    def _upload_to_cloud(self, local_path: Path, cloud_path: str) -> None:
        """
        Upload processed data to cloud storage.
        
        Args:
            local_path: Local path to the zarr directory
            cloud_path: Cloud storage path
        """
        try:
            # Upload directory contents
            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_path)
                    destination = f"{cloud_path}/{relative_path}"
                    self.gcs_client.upload_file(file_path, destination)
            
            logger.info(f"Uploads completed successfully to {cloud_path}")
                    
        except Exception as e:
            logger.error(f"Error uploading to cloud: {str(e)}")
            
    def finalize_stage1(self) -> None:
        """
        Finalize Stage 1 processing.
        
        Creates a manifest of processed files in the cloud.
        """
        logger.info("Finalizing Stage 1 processing")
        
        # Determine the cloud manifest path
        cloud_manifest_path = f"{self.path_prefix}{self.intermediate_dir}/{self.data_source.lower()}_manifest.csv"
        
        processed_files = []
        
        # Query the cloud storage for processed files
        annual_prefix = f"{self.path_prefix}{self.intermediate_dir}/{self.data_source.lower()}"
        try:
            # List all annual stats files in the cloud
            cloud_files = self.gcs_client.list_existing_files(prefix=annual_prefix)
            
            # Process the list of files
            for file_path in cloud_files:
                if "_annual.zarr/.zmetadata" in file_path:
                    # Extract metadata from path
                    filename = os.path.basename(file_path.split('/.zmetadata')[0])
                    parts = filename.split('_')
                    
                    if len(parts) >= 3:
                        year = parts[1]
                        satellite = parts[2]
                        processed_files.append({
                            "data_source": self.data_source,
                            "year": year,
                            "satellite": satellite,
                            "path": file_path.replace("/.zmetadata", "")
                        })
            
            # Save manifest to a temporary file and upload to cloud
            if processed_files:
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as temp_file:
                    manifest_df = pd.DataFrame(processed_files)
                    manifest_df.to_csv(temp_file.name, index=False)
                    
                    # Upload the manifest to cloud
                    self.gcs_client.upload_file(temp_file.name, cloud_manifest_path)
                    
                    logger.info(f"Created and uploaded manifest with {len(processed_files)} processed files to {cloud_manifest_path}")
                    
                    # Clean up the temporary file
                    os.unlink(temp_file.name)
            else:
                logger.warning("No processed files found in cloud storage to include in manifest")
                    
        except Exception as e:
            logger.error(f"Error creating cloud manifest: {str(e)}")
            
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EOGPreprocessor":
        """
        Create a preprocessor instance from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            EOGPreprocessor: Initialized preprocessor
        """
        # Extract required paths
        input_path = config.get("input_path")
        output_path = config.get("output_path")
        intermediate_path = config.get("intermediate_path")
        
        # Extract other configuration values with defaults
        data_source = config.get("data_source", "DMSP")
        years = config.get("years", None)
        override = config.get("override", False)
        stage = config.get("stage", "all")
        
        # Create and return the preprocessor instance
        return cls(
            input_path=input_path,
            output_path=output_path,
            intermediate_path=intermediate_path,
            data_source=data_source,
            years=years,
            override=override,
            stage=stage,
            # Pass through any other configs
            **{k: v for k, v in config.items() if k not in [
                "input_path", "output_path", "intermediate_path", 
                "data_source", "years", "override", "stage"
            ]}
        )
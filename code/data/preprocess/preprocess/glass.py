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

from preprocess.base import AbstractPreprocessor
from gcs.client import GCSClient

logger = logging.getLogger(__name__)

class GlassPreprocessor(AbstractPreprocessor):
    """
    Preprocessor for GLASS (Global LAnd Surface Satellite) LST data.
    
    Handles both MODIS (multiple files per day in grid cells) and AVHRR (one file per day) datasets.
    Uses Dask for distributed processing to handle datasets larger than memory.
    """
    
    # Class constants
    BUCKET_NAME = "growthandheat"
    MODIS_PATH_PREFIX = "glass/LST/MODIS/Daily/1KM/"
    AVHRR_PATH_PREFIX = "glass/LST/AVHRR/0.05D/"
    VARIABLE_NAME = "LST"
    
    def __init__(self, input_path: Union[str, Path], output_path: Union[str, Path],
                 intermediate_path: Optional[Union[str, Path]] = None, **kwargs):
        """
        Initialize the Glass preprocessor.
        
        Args:
            input_path: Path or prefix in Google Cloud Storage bucket
            output_path: Path where processed data will be stored
            intermediate_path: Path where Stage 1 output will be stored
            **kwargs: Additional configuration parameters including:
                - data_source: 'MODIS' or 'AVHRR'
                - years: List of years to process or range (start, end)
                - grid_cells: List of grid cells to process (for MODIS)
                - override: Whether to override existing processed files
                - dask_threads: Number of threads for Dask (default: number of CPU cores)
                - dask_memory_limit: Memory limit for Dask workers in GB (default: 75% of system memory)
                - chunk_size: Size of chunks for Dask arrays (default: {"time": 1, "x": 500, "y": 500})
        """
        super().__init__(input_path, output_path, intermediate_path, **kwargs)
        
        # Set data source-specific attributes
        self.data_source = kwargs.get('data_source', 'MODIS').upper()
        if self.data_source not in ['MODIS', 'AVHRR']:
            raise ValueError(f"Unsupported data source: {self.data_source}. Use 'MODIS' or 'AVHRR'.")
        
        # Set years to process
        years = kwargs.get('years', None)
        if years is None:
            # Default to processing all years
            self.years = list(range(2000, 2021)) if self.data_source == 'MODIS' else list(range(1982, 2021))
        elif isinstance(years, list):
            self.years = years
        elif isinstance(years, tuple) and len(years) == 2:
            self.years = list(range(years[0], years[1] + 1))
        else:
            raise ValueError("Years must be a list of years or a tuple (start_year, end_year)")
            
        # Set grid cells to process (for MODIS)
        self.grid_cells = kwargs.get('grid_cells', None)
        
        # Whether to override existing processed files
        self.override = kwargs.get('override', False)
        
        # Path prefix based on data source
        self.path_prefix = self.MODIS_PATH_PREFIX if self.data_source == 'MODIS' else self.AVHRR_PATH_PREFIX
        
        # Initialize GCS client
        self.gcs_client = GCSClient(self.BUCKET_NAME)
        
        # Dask configuration
        self.dask_threads = kwargs.get('dask_threads', None)  # None means use all available cores
        self.dask_memory_limit = kwargs.get('dask_memory_limit', None)  # None means 75% of system memory
        self.chunk_size = kwargs.get('chunk_size', {"band": 1, "x": 500, "y": 500})
        
        # Initialize Dask client
        self._init_dask_client()
    
    def _init_dask_client(self):
        """Initialize a Dask client for distributed processing."""
        try:
            # Set up a LocalCluster with desired resources
            # Using processes=False to avoid issues with multiprocessing and GCS client
            cluster = LocalCluster(
                n_workers=1,  # Use a single worker to avoid memory duplication
                threads_per_worker=self.dask_threads,  # Use multiple threads per worker
                memory_limit=self.dask_memory_limit,
                processes=False,  # Use threads instead of processes
            )
            self.client = Client(cluster)
            
            # Log Dask dashboard URL for monitoring
            logger.info(f"Dask dashboard available at: {self.client.dashboard_link}")
        except Exception as e:
            logger.warning(f"Failed to initialize Dask client: {str(e)}. Using default threading.")
            self.client = None
    
    def __del__(self):
        """Clean up Dask client when the object is garbage collected."""
        if hasattr(self, 'client') and self.client:
            try:
                self.client.close()
                logger.debug("Closed Dask client")
            except:
                pass
            
    def summarize_annual_means(self) -> None:
        """
        Summarize GLASS daily data into annual means.
        
        Stage 1 processing:
        - Lists all available files for the specified data source
        - Groups files by year and grid cell
        - For each group, downloads the files, calculates annual statistics
        - Saves the results to the intermediate location using Dask for out-of-memory processing
        """
        logger.info(f"Starting summarization of {self.data_source} data for years {min(self.years)}-{max(self.years)}")
        
        if self.data_source == 'MODIS':
            self._process_modis_data()
        else:
            self._process_avhrr_data()
            
        logger.info(f"Completed summarization of {self.data_source} data")
    
    def _process_modis_data(self):
        """Process MODIS data which is organized in grid cells."""
        # List all MODIS files
        all_files = self._list_files_from_gcs()
        
        # Parse filenames to extract metadata
        files_df = self._parse_modis_filenames(all_files)
        
        # Filter by years
        files_df = files_df[files_df['year'].isin(self.years)]
        
        # Filter by grid cells if specified
        if self.grid_cells:
            grid_filter = files_df.apply(lambda row: f"h{row['h']:02d}v{row['v']:02d}" in self.grid_cells, axis=1)
            files_df = files_df[grid_filter]
        
        # Group by year and grid cell
        grouped = files_df.groupby(['year', 'h', 'v'])
        
        total_groups = len(grouped)
        logger.info(f"Found {total_groups} year-gridcell combinations to process")
        
        for i, ((year, h, v), group) in enumerate(grouped):
            grid_cell = f"h{h:02d}v{v:02d}"
            output_path = self.intermediate_path / f"{self.data_source.lower()}" / f"{year}" / f"{grid_cell}.zarr"
            
            # Skip if already processed and override is False
            if output_path.exists() and not self.override:
                logger.info(f"Skipping existing output for {year} {grid_cell} ({i+1}/{total_groups})")
                continue
                
            logger.info(f"Processing {year} {grid_cell} ({i+1}/{total_groups})")
            
            try:
                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Process this group of files
                self._process_file_group(group, year, output_path)
                
            except Exception as e:
                logger.error(f"Error processing {year} {grid_cell}: {str(e)}")
                
    def _process_avhrr_data(self):
        """Process AVHRR data which has one file per day globally."""
        # List all AVHRR files
        all_files = self._list_files_from_gcs()
        
        # Parse filenames to extract metadata
        files_df = self._parse_avhrr_filenames(all_files)
        
        # Filter by years
        files_df = files_df[files_df['year'].isin(self.years)]
        # TODO: remove
        files_df = files_df.head(10)
        
        # Group by year
        grouped = files_df.groupby('year')
        
        total_groups = len(grouped)
        logger.info(f"Found {total_groups} years to process")
        
        for i, (year, group) in enumerate(grouped):
            # Define cloud storage paths instead of local paths
            annual_cloud_path = f"{self.path_prefix}intermediate/{self.data_source.lower()}_{year}_annual.zarr"
            monthly_cloud_path = f"{self.path_prefix}intermediate/{self.data_source.lower()}_{year}_monthly.zarr"
            output_path = self.intermediate_path / f"{self.data_source.lower()}" / f"{year}.zarr"
            
            # Skip if already processed in cloud storage and override is False
            if not self.override and (
                self.gcs_client.check_if_exists(f"{annual_cloud_path}/.zmetadata") or 
                self.gcs_client.check_if_exists(f"{annual_cloud_path}/zgroup")
            ):
                logger.info(f"Skipping existing cloud output for year {year} ({i+1}/{total_groups})")
                continue
                
            logger.info(f"Processing year {year} ({i+1}/{total_groups})")
            
            try:
                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Process this group of files
                self._process_file_group(group, year, output_path)
                
            except Exception as e:
                logger.error(f"Error processing year {year}: {str(e)}")
    
    def _list_files_from_gcs(self) -> List[str]:
        """List all files from the Google Cloud Storage bucket with the specified prefix."""
        logger.info(f"Listing files with prefix {self.path_prefix} from bucket {self.BUCKET_NAME}")
        try:
            files = self.gcs_client.list_existing_files(prefix=self.path_prefix)
            return list(files)
        except Exception as e:
            logger.error(f"Error listing files from GCS: {str(e)}")
            raise
    
    def _parse_modis_filenames(self, filenames: List[str]) -> pd.DataFrame:
        """
        Parse MODIS filenames to extract metadata.
        
        Expected format: GLASS06A01.V01.A2000055.h00v10.2022021.hdf
        where:
        - 2000055 is year (2000) and day of year (055)
        - h00v10 is the grid cell (h=0, v=10)
        """
        result = []
        
        for filename in filenames:
            try:
                basename = os.path.basename(filename)
                if not basename.endswith('.hdf'):
                    continue
                    
                # Extract year and day
                year_day_match = basename.split('.')[2]
                if not (year_day_match.startswith('A') and len(year_day_match) == 8):
                    continue
                    
                year = int(year_day_match[1:5])
                day = int(year_day_match[5:8])
                
                # Extract grid cell
                grid_match = basename.split('.')[3]
                if not (grid_match.startswith('h') and 'v' in grid_match):
                    continue
                    
                h = int(grid_match[1:].split('v')[0])
                v = int(grid_match.split('v')[1])
                
                result.append({
                    'path': filename,
                    'year': year,
                    'day': day,
                    'h': h,
                    'v': v
                })
            except (IndexError, ValueError) as e:
                logger.warning(f"Could not parse filename {filename}: {str(e)}")
                
        return pd.DataFrame(result)
    
    def _parse_avhrr_filenames(self, filenames: List[str]) -> pd.DataFrame:
        """
        Parse AVHRR filenames to extract metadata.
        
        Expected format: GLASS08B31.V40.A1982001.2021259.hdf
        where:
        - 1982001 is year (1982) and day of year (001)
        """
        result = []
        
        for filename in filenames:
            try:
                basename = os.path.basename(filename)
                if not basename.endswith('.hdf'):
                    continue
                    
                # Extract year and day
                year_day_match = basename.split('.')[2]
                if not (year_day_match.startswith('A') and len(year_day_match) == 8):
                    continue
                    
                year = int(year_day_match[1:5])
                day = int(year_day_match[5:8])
                
                result.append({
                    'path': filename,
                    'year': year,
                    'day': day
                })
            except (IndexError, ValueError) as e:
                logger.warning(f"Could not parse filename {filename}: {str(e)}")
                
        return pd.DataFrame(result)

    def _process_file_group(self, files_df: pd.DataFrame, year: int, output_path: Path) -> None:
        """
        Process a group of files for a specific year (and grid cell for MODIS).
        All local processing uses temporary directories, with only compressed results uploaded to cloud.
        """
        
        # Create temporary directory for downloads and processing
        with tempfile.TemporaryDirectory() as temp_dir:
            files_df = files_df.sort_values(by='day')  # Ensure files are in chronological order
            
            # First pass: get metadata and coordinates without loading all data
            # This helps Dask set up the arrays more efficiently
            sample_data = None
            for _, file_info in files_df.iloc[:1].iterrows():
                try:
                    file_path = file_info['path']
                    local_path = os.path.join(temp_dir, os.path.basename(file_path))
                    
                    # Download from GCS using the client
                    self.gcs_client.download_file(file_path, local_path)
                    
                    # Open with rioxarray but only get metadata
                    sample_data = rxr.open_rasterio(
                        local_path, 
                        variable=self.VARIABLE_NAME, 
                        decode_coords="all",
                    )
                    break
                except Exception as e:
                    logger.warning(f"Error getting sample data from {file_info['path']}: {str(e)}")
            
            if sample_data is None:
                logger.warning(f"No valid sample data found for {year}")
                return
            
            # Initialize empty dask array with proper dimensions
            shape = (len(files_df), *sample_data.sizes.values())
            
            # Create a template array with proper coordinates
            coords = {
                'time': pd.to_datetime([f"{year}{day:03d}" for day in files_df['day'].values], format="%Y%j"),
                'y': sample_data.y.values,
                'x': sample_data.x.values
            }
            
            # Use template array dimensions and dtypes for the empty array
            template = xr.DataArray(
                da.zeros(shape, dtype=sample_data.dtypes[self.VARIABLE_NAME], chunks=self.chunk_size),
                dims=['time', 'band', 'x', 'y'],
                coords=coords
            )
            
            # Fill the template array with actual data
            logger.info(f"Processing {len(files_df)} files with Dask")
            for i, (_, file_info) in enumerate(files_df.iterrows()):
                try:
                    file_path = file_info['path']
                    local_path = os.path.join(temp_dir, os.path.basename(file_path))
                    
                    # Download from GCS using the client (or use cached file from first pass)
                    if not os.path.exists(local_path):
                        self.gcs_client.download_file(file_path, local_path)
                    
                    # Open with rioxarray and update template
                    data = rxr.open_rasterio(
                        local_path, 
                        variable=self.VARIABLE_NAME, 
                        decode_coords="all",
                    )
                    
                    # Update the template at the specific time index
                    # We use dask's optimized methods to avoid loading everything into memory
                    template[i, :, :] = data[self.VARIABLE_NAME]  # First band
                    
                    # Log progress periodically
                    if (i + 1) % 20 == 0 or i == len(files_df) - 1:
                        logger.debug(f"Processed {i + 1}/{len(files_df)} files")
                        
                except Exception as e:
                    logger.warning(f"Error processing file {file_info['path']}: {str(e)}")
                    # Continue with NaN values for this time step
            
            # Convert the template to a Dataset with LST variable
            combined_data = xr.Dataset({'LST': template})
            
            # Calculate statistics using Dask
            logger.info("Calculating statistics with Dask (this may take a while)")
            annual_stats, monthly_stats = self._calculate_statistics(combined_data)
            
            # Create temporary output paths within the temp directory
            annual_output_path = Path(temp_dir) / f"{output_path.stem}_annual.zarr"
            monthly_output_path = Path(temp_dir) / f"{output_path.stem}_monthly.zarr"
            
            # Save to temporary Zarr files with compression
            compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
            encoding = {var: {'compressor': compressor} for var in annual_stats.data_vars}
            
            logger.info(f"Saving annual statistics to temporary location with compression")
            annual_stats_task = annual_stats.to_zarr(
                str(annual_output_path), 
                mode="w", 
                consolidated=True,
                compute=False,
                encoding=encoding
            )
            
            logger.info(f"Saving monthly statistics to temporary location with compression")
            monthly_stats_task = monthly_stats.to_zarr(
                str(monthly_output_path), 
                mode="w", 
                consolidated=True,
                compute=False,
                encoding=encoding
            )
            
            # Execute both saving operations in parallel
            logger.info("Executing Dask tasks for saving statistics")
            dask.compute(annual_stats_task, monthly_stats_task)
            
            # Upload to cloud storage - this is the only persistent storage
            self._upload_to_cloud(annual_output_path, monthly_output_path, year)

    def _calculate_statistics(self, data: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Calculate annual and monthly statistics from daily data using Dask.
        
        Args:
            data: Dataset with time dimension
            
        Returns:
            Tuple of (annual_stats, monthly_stats)
        """
        # Create a mask for invalid values
        mask = ~data[self.VARIABLE_NAME].isnull()
        
        # Calculate annual statistics
        annual_stats = xr.Dataset({
            "mean": data[self.VARIABLE_NAME].resample(time="1YE").mean(),
            "median": data[self.VARIABLE_NAME].resample(time="1YE").median(),
            "std": data[self.VARIABLE_NAME].resample(time="1YE").std(),
            "count": mask.resample(time="1YE").sum()
        })
        
        # Calculate monthly statistics
        monthly_stats = xr.Dataset({
            "mean": data[self.VARIABLE_NAME].resample(time="1ME").mean(),
            "median": data[self.VARIABLE_NAME].resample(time="1ME").median(),
            "std": data[self.VARIABLE_NAME].resample(time="1ME").std(),
            "count": mask.resample(time="1ME").sum()
        })
        
        return annual_stats, monthly_stats
    
    def _upload_to_cloud(self, annual_path: Path, monthly_path: Path, year: int) -> None:
        """
        Upload compressed processed data to cloud storage.
        
        Args:
            annual_path: Local path to annual statistics
            monthly_path: Local path to monthly statistics
            year: Year being processed
        """
        try:
            # Determine the cloud storage paths
            data_type = self.data_source.lower()
            
            grid_cell = ""
            if self.data_source == "MODIS" and annual_path.stem.split('_')[-2:-1]:
                grid_cell = f"_{annual_path.stem.split('_')[-2]}"
            
            # Create cloud paths with appropriate naming
            annual_cloud_path = f"{self.path_prefix}intermediate/{data_type}_{year}{grid_cell}_annual.zarr"
            monthly_cloud_path = f"{self.path_prefix}intermediate/{data_type}_{year}{grid_cell}_monthly.zarr"
            
            logger.info(f"Uploading annual statistics to cloud: {annual_cloud_path}")
            # Upload annual statistics (directory)
            for file_path in annual_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(annual_path)
                    destination = f"{annual_cloud_path}/{relative_path}"
                    self.gcs_client.upload_file(file_path, destination)
            
            logger.info(f"Uploading monthly statistics to cloud: {monthly_cloud_path}")
            # Upload monthly statistics (directory)
            for file_path in monthly_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(monthly_path)
                    destination = f"{monthly_cloud_path}/{relative_path}"
                    self.gcs_client.upload_file(file_path, destination)
            
            logger.info("Uploads completed successfully")
                    
        except Exception as e:
            logger.error(f"Error uploading to cloud: {str(e)}")
    
    def _combine_time_series(self, data_arrays: List[xr.DataArray], files_df: pd.DataFrame, year: int) -> Optional[xr.DataArray]:
        """
        Combine multiple DataArrays into a time series.
        
        Args:
            data_arrays: List of DataArrays to combine
            files_df: DataFrame with metadata about the files
            year: The year being processed
            
        Returns:
            Combined DataArray with proper time dimension
        """
        try:
            if len(data_arrays) == 0:
                return None
                
            # Concatenate along a new dimension (will become time)
            concat_dim = 'day'
            combined = xr.concat(data_arrays, dim=pd.Index(files_df['day'].values, name=concat_dim))
            
            # Convert day of year to datetime
            times = pd.to_datetime([f"{year}{day:03d}" for day in combined[concat_dim].values], format="%Y%j").values
            combined = combined.assign_coords({concat_dim: times}).rename({concat_dim: "time"})
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining time series data: {str(e)}")
            return None

    
    def finalize_stage1(self) -> None:
        """
        Finalize Stage 1 processing.
        
        Creates a manifest of processed files in the cloud.
        """
        logger.info("Finalizing Stage 1 processing")
        
        # Determine the cloud manifest path
        cloud_manifest_path = f"{self.path_prefix}intermediate/{self.data_source.lower()}_manifest.csv"
        
        processed_files = []
        
        # Query the cloud storage for processed files instead of local directory
        annual_prefix = f"{self.path_prefix}intermediate/{self.data_source.lower()}"
        try:
            # List all annual stats files in the cloud
            cloud_files = self.gcs_client.list_existing_files(prefix=annual_prefix)
            
            # Process the list of files
            for file_path in cloud_files:
                if "_annual.zarr/.zmetadata" in file_path:
                    # Extract year and possibly grid cell from path
                    path_parts = file_path.split('/')
                    filename = path_parts[-2].split('_annual.zarr')[0]  # Remove the suffix
                    
                    if self.data_source == "MODIS":
                        # For MODIS: path contains year and grid cell (e.g., modis_2000_h08v05_annual.zarr)
                        parts = filename.split('_')
                        if len(parts) >= 3:
                            year = parts[1]
                            grid_cell = parts[2]
                            processed_files.append({
                                "data_source": self.data_source,
                                "year": year,
                                "grid_cell": grid_cell,
                                "path": file_path.replace("/.zmetadata", "")
                            })
                    else:
                        # For AVHRR: path contains year (e.g., avhrr_2000_annual.zarr)
                        parts = filename.split('_')
                        if len(parts) >= 2:
                            year = parts[1]
                            processed_files.append({
                                "data_source": self.data_source,
                                "year": year,
                                "grid_cell": "global",
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

    def validate_input(self) -> bool:
        """
        Validate that the input data exists and is accessible.
        
        For GCS input, checks that the bucket and prefix exist and are accessible
        by trying to list just one file, rather than all files.
        """
        try:
            # Check if the GCS client can access the bucket by listing just one file
            # This is much more efficient than listing all files
            for _ in self.gcs_client.list_blobs_with_limit(prefix=self.path_prefix, limit=1):
                # If we can iterate even one file, access is working
                return True
                
            logger.warning(f"No files found with prefix {self.path_prefix} in bucket {self.BUCKET_NAME}")
            return False
                
        except Exception as e:
            logger.error(f"Error validating input: {str(e)}")
            return False
    
    def project_to_unified_grid(self) -> None:
        """
        Stage 2: Project data onto a unified grid.
        
        This is a placeholder for the Stage 2 implementation, which will be added later.
        It will read data from Stage 1 and reproject it to a standard grid.
        """
        logger.warning("Stage 2 (project_to_unified_grid) not yet implemented for GLASS data")
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GlassPreprocessor":
        """
        Create a GlassPreprocessor from a configuration dictionary.
        
        Args:
            config: Dictionary containing configuration parameters
            
        Returns:
            GlassPreprocessor instance
        """
        input_path = config.pop("input_path", None)  # For GCS, may be None
        output_path = config.pop("output_path")
        intermediate_path = config.pop("intermediate_path", None)
        
        return cls(input_path, output_path, intermediate_path, **config)
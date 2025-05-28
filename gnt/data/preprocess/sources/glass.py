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
import numcodecs
import json  # Add missing import at the top of the file

from gnt.data.preprocess.sources.base import AbstractPreprocessor
from gnt.data.common.gcs.client import GCSClient
from gnt.data.common.dask.client import DaskClientContextManager
from gnt.data.common.index.preprocessing_index import PreprocessingIndex

logger = logging.getLogger(__name__)

class GlassPreprocessor(AbstractPreprocessor):
    """
    Preprocessor for GLASS (Global LAnd Surface Satellite) LST data.
    
    Handles both MODIS (multiple files per day in grid cells) and AVHRR (one file per day) datasets.
    Uses Dask for distributed processing to handle datasets larger than memory.
    
    Integrates with PreprocessingIndex for tracking files during the preprocessing
    stages to enable data transfer between Kubernetes and University cluster.
    """
    
    # Class constants
    BUCKET_NAME = "growthandheat"
    MODIS_PATH_PREFIX = "glass/LST/MODIS/Daily/1KM/"
    AVHRR_PATH_PREFIX = "glass/LST/AVHRR/0.05D/"
    VARIABLE_NAME = "LST"
    
    def __init__(self, **kwargs):
        """
        Initialize the Glass preprocessor.
        
        Args:
            **kwargs: Additional configuration parameters including:
                - data_source: 'MODIS' or 'AVHRR'
                - years: List of years to process or range (start, end)
                - grid_cells: List of grid cells to process (for MODIS)
                - override: Whether to override existing processed files
                - dask_threads: Number of threads for Dask (default: number of CPU cores)
                - dask_memory_limit: Memory limit for Dask workers in GB (default: 75% of system memory)
                - chunk_size: Size of chunks for Dask arrays (default: {"time": 1, "x": 500, "y": 500})
                - intermediate_dir: Directory for intermediate results (default: "intermediate")
                - output_dir: Directory for final output (default: "output")
                - base_url: Base URL for the data source (used for index lookup)
                - file_extensions: List of file extensions to look for
                - version: Version of processing (default: "v1")
        """
        super().__init__(**kwargs)
        
        # Set data source-specific attributes
        self.data_source = kwargs.get('data_source', 'MODIS').upper()
        if self.data_source not in ['MODIS', 'AVHRR']:
            raise ValueError(f"Unsupported data source: {self.data_source}. Use 'MODIS' or 'AVHRR'.")
        
        # Set years to process
        years = kwargs.get('years', None)
        if years is None:
            # Default to processing all years
            self.years = list(range(2000, 2021)) if self.data_source == 'MODIS' else list(range(1982, 2021))
        elif isinstance(years, list) and len(years) == 2:
            self.years = list(range(years[0], years[1] + 1))
        elif isinstance(years, list):
            self.years = years
        else:
            raise ValueError("Years must be a list of years or a tuple (start_year, end_year)")
            
        # Set grid cells to process (for MODIS)
        self.grid_cells = kwargs.get('grid_cells', None)
        
        # Whether to override existing processed files
        self.override = kwargs.get('override', False)
        
        # Path prefix based on data source
        self.path_prefix = self.MODIS_PATH_PREFIX if self.data_source == 'MODIS' else self.AVHRR_PATH_PREFIX
        
        # Set directory paths
        self.intermediate_dir = kwargs.get('intermediate_dir', 'intermediate')
        self.output_dir = kwargs.get('output_dir', 'output')
        
        # Set base_url and file_extensions (used for index lookup)
        self.base_url = kwargs.get('base_url', "https://glass-data.bnu.edu.cn/download/")
        self.file_extensions = kwargs.get('file_extensions', [".hdf"])
        self.version = kwargs.get('version', 'v1')
        
        # Initialize GCS client
        self.gcs_client = GCSClient(self.BUCKET_NAME)
        
        # Initialize preprocessing index
        self.data_path = f"glass/LST/{self.data_source}"
        self.preprocessing_index = PreprocessingIndex(
            bucket_name=self.BUCKET_NAME,
            data_path=self.data_path,
            version=self.version,
            client=self.gcs_client.client if hasattr(self.gcs_client, 'client') else None
        )
        
        # Dask configuration parameters - store them but don't create client yet
        self.dask_threads = kwargs.get('dask_threads', None)  # None means use all available cores
        self.dask_memory_limit = kwargs.get('dask_memory_limit', None)  # None means 75% of system memory
        self.chunk_size = kwargs.get('chunk_size', {"band": 1, "x": 500, "y": 500})
        self.dashboard_port = kwargs.get('dashboard_port', 8787)
    
    def _get_dask_client_params(self):
        """
        Return parameters for creating a Dask client.
        """
        return {
            'threads': self.dask_threads,
            'memory_limit': self.dask_memory_limit,
            'dashboard_port': self.dashboard_port,
            'temp_dir': "/tmp/dask_glass",
        }
    
    def summarize_annual_means(self) -> None:
        """
        Summarize GLASS daily data into annual means.
        
        Stage 1 processing:
        - Lists all available files for the specified data source
        - Groups files by year and grid cell
        - For each group, downloads the files, calculates annual statistics
        - Saves the results to the intermediate location using Dask for out-of-memory processing
        - Updates the preprocessing index with file information
        """
        logger.info(f"Starting summarization of {self.data_source} data for years {min(self.years)}-{max(self.years)}")
        
        # Create a Dask client specifically for this task
        with DaskClientContextManager(**self._get_dask_client_params()) as client:
            logger.info(f"Created Dask client for annual means task: {client.dashboard_link}")
            
            if self.data_source == 'MODIS':
                self._process_modis_data(client)
            else:
                self._process_avhrr_data(client)
                
        logger.info(f"Completed summarization of {self.data_source} data")
    
    def _process_modis_data(self, client: Client):
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
            output_path = f"{self.data_source.lower()}/{year}/{grid_cell}.zarr"
            
            # Check if output exists using our helper method
            if not self.override and self._check_output_exists(year, grid_cell):
                logger.info(f"Skipping existing output for {year} {grid_cell} ({i+1}/{total_groups})")
                continue
            
            logger.info(f"Processing {year} {grid_cell} ({i+1}/{total_groups})")
            
            try:
                # First, create an entry in the index for tracking
                annual_file_hash = self._register_annual_file(year, grid_cell)
                
                # Process this group of files
                success = self._process_file_group(group, year, output_path, client, grid_cell=grid_cell)
                
                # Update index with success/failure status
                if success:
                    self.preprocessing_index.update_file_status(
                        file_hash=annual_file_hash, 
                        status=PreprocessingIndex.STATUS_COMPLETED,
                        metadata={
                            "processing_time": datetime.now().isoformat(),
                            "input_files": len(group),
                            "source": self.data_source
                        }
                    )
                    # Mark for transfer to the cluster
                    self.preprocessing_index.mark_for_transfer(annual_file_hash, "cluster")
                else:
                    self.preprocessing_index.update_file_status(
                        file_hash=annual_file_hash, 
                        status=PreprocessingIndex.STATUS_FAILED
                    )
                    
            except Exception as e:
                # Use our centralized error handler
                self._handle_processing_error(year, grid_cell, e)

    def _process_avhrr_data(self, client: Client):
        """Process AVHRR data which has one file per day globally."""
        # List all AVHRR files
        all_files = self._list_files_from_gcs()
        
        # Parse filenames to extract metadata
        files_df = self._parse_avhrr_filenames(all_files)
        
        # Filter by years
        files_df = files_df[files_df['year'].isin(self.years)]
        
        # Group by year
        grouped = files_df.groupby('year')
        
        total_groups = len(grouped)
        logger.info(f"Found {total_groups} years to process")
        
        for i, (year, group) in enumerate(grouped):
            # Define cloud storage paths
            annual_cloud_path = f"{self.path_prefix}{self.intermediate_dir}/{self.data_source.lower()}_{year}_annual.zarr"
            output_path = f"{self.data_source.lower()}/{year}.zarr"
            
            # For AVHRR, grid cell is "global"
            grid_cell = "global"
            
            # Check using get_files method from the index
            annual_files = []
            try:
                cursor = self.preprocessing_index._get_connection().cursor()
                cursor.execute(
                    "SELECT * FROM files WHERE stage = ? AND year = ? AND grid_cell = ? AND status = ?",
                    (PreprocessingIndex.STAGE_ANNUAL, year, grid_cell, PreprocessingIndex.STATUS_COMPLETED)
                )
                columns = [description[0] for description in cursor.description]
                for row in cursor.fetchall():
                    file_info = dict(zip(columns, row))
                    # Parse metadata if it exists
                    if file_info.get('metadata'):
                        try:
                            file_info['metadata'] = json.loads(file_info['metadata'])
                        except:
                            pass  # Keep as string if not valid JSON
                    annual_files.append(file_info)
            except Exception as e:
                logger.warning(f"Error querying index: {e}")
                
            # Skip if already processed and override is False
            if not self.override and (annual_files or 
                self.gcs_client.file_exists(f"{annual_cloud_path}/.zmetadata") or 
                self.gcs_client.file_exists(f"{annual_cloud_path}/zgroup")):
                logger.info(f"Skipping existing cloud output for year {year} ({i+1}/{total_groups})")
                continue
                
            logger.info(f"Processing year {year} ({i+1}/{total_groups})")
            
            try:
                # First, create an entry in the index for tracking
                annual_file_hash = self._register_annual_file(year, grid_cell)
                
                # Process this group of files
                success = self._process_file_group(group, year, output_path, client)
                
                # Update index with success/failure status
                if success:
                    self.preprocessing_index.update_file_status(
                        file_hash=annual_file_hash, 
                        status=PreprocessingIndex.STATUS_COMPLETED,
                        metadata={
                            "processing_time": datetime.now().isoformat(),
                            "input_files": len(group),
                            "source": self.data_source
                        }
                    )
                    # Mark for transfer to the cluster
                    self.preprocessing_index.mark_for_transfer(annual_file_hash, "cluster")
                else:
                    self.preprocessing_index.update_file_status(
                        file_hash=annual_file_hash, 
                        status=PreprocessingIndex.STATUS_FAILED
                    )
                    
            except Exception as e:
                logger.error(f"Error processing year {year}: {str(e)}")
                # Try to update any processing files for this year/grid_cell
                try:
                    cursor = self.preprocessing_index._get_connection().cursor()
                    cursor.execute(
                        "SELECT file_hash FROM files WHERE stage = ? AND year = ? AND grid_cell = ? AND status = ?",
                        (PreprocessingIndex.STAGE_ANNUAL, year, grid_cell, PreprocessingIndex.STATUS_PROCESSING)
                    )
                    processing_files = [{"file_hash": row[0]} for row in cursor.fetchall()]
                    if processing_files:
                        self.preprocessing_index.update_file_status(
                            file_hash=processing_files[0]["file_hash"], 
                            status=PreprocessingIndex.STATUS_FAILED,
                            metadata={"error": str(e)}
                        )
                except Exception as ex:
                    logger.error(f"Error updating file status: {ex}")

    def _register_annual_file(self, year: int, grid_cell: str) -> str:
        """
        Register an annual file in the preprocessing index.
        
        Args:
            year: The year of the data
            grid_cell: The grid cell identifier
            
        Returns:
            The file hash for the registered file
        """
        # Create blob path for the annual file using the preprocessing_index.annual_path
        if grid_cell == "global":
            blob_path = f"{self.preprocessing_index.annual_path}{year}.zarr"
        else:
            blob_path = f"{self.preprocessing_index.annual_path}{year}/{grid_cell}.zarr"
        
        # Create metadata
        metadata = {
            "filename": os.path.basename(blob_path),
            "data_source": self.data_source,
            "creation_started": datetime.now().isoformat()
        }
        
        # Create file entry in index
        file_hash = self.preprocessing_index.add_file(
            stage=PreprocessingIndex.STAGE_ANNUAL,
            year=year,
            grid_cell=grid_cell,
            status=PreprocessingIndex.STATUS_PROCESSING,
            blob_path=blob_path,
            metadata=metadata
        )
        
        return file_hash
    
    def _list_files_from_gcs(self) -> List[str]:
        """
        List all files from the Google Cloud Storage bucket using the download index.
        Falls back to direct GCS listing if the index is not available or empty.
        """
        logger.info(f"Attempting to list GLASS files with prefix {self.path_prefix} from download index")
        
        try:
            from gnt.data.common.index.download_index import DataDownloadIndex
            from gnt.data.download.sources.factory import create_data_source
            
            # Create a data source instance for GLASS with parameters passed from the class
            data_source_name = "glass"  # Use appropriate source name
            data_source = create_data_source(
                dataset_name=data_source_name,
                config = {
                    "base_url": self.base_url, 
                    "file_extensions": self.file_extensions},
            )
            
            # Initialize download index without auto-indexing (we just want to query it)
            index = DataDownloadIndex(
                bucket_name=self.BUCKET_NAME,
                data_source=data_source,
                client=self.gcs_client.client if hasattr(self.gcs_client, 'client') else None
            )
            
            # List files
            files = index.list_successful_files(prefix=self.path_prefix)
            
            logger.info(f"Found {len(files)} indexed files matching prefix {self.path_prefix}")
            
            # If no files found in index, fall back to direct listing
            if not files:
                logger.warning(f"No files found in index, falling back to direct GCS listing")
                files = list(self.gcs_client.list_existing_files(prefix=self.path_prefix))
                
            return files
            
        except ImportError as e:
            logger.warning(f"Download index not available, falling back to direct GCS listing: {str(e)}")
        except Exception as e:
            logger.error(f"Error listing files from index: {str(e)}")
            logger.warning("Falling back to direct listing from GCS")
        
        # Fall back to direct GCS listing in case of any issues
        logger.info(f"Listing files directly from GCS with prefix {self.path_prefix}")
        files = list(self.gcs_client.list_existing_files(prefix=self.path_prefix))
        return files
    
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

    def _process_file_group(self, files_df: pd.DataFrame, year: int, output_path: str, 
                         client: Client, grid_cell: str = None) -> bool:
        """
        Process a group of files for a specific year (and grid cell for MODIS).
        All local processing uses temporary directories, with only compressed results uploaded to cloud.
        
        Args:
            files_df: DataFrame with file information
            year: Year being processed
            output_path: Path for output data
            client: Dask client for distributed processing
            grid_cell: Optional grid cell identifier
        
        Returns:
            Boolean indicating success
        """
        
        # Create temporary directory for downloads and processing
        with tempfile.TemporaryDirectory(dir='/tmp') as temp_dir:
            files_df = files_df.sort_values(by = 'day')  # Ensure files are in chronological order
            #files_df = files_df.iloc[:3]  # TODO: remove; For testing, only process the first three files
            
            # First pass: get metadata and coordinates without loading all data
            sample_data = None
            for _, file_info in files_df.iloc[:1].iterrows():
                try:
                    file_path = file_info['path']
                    local_path = os.path.join(temp_dir, os.path.basename(file_path))
                    
                    # Download from GCS using the client
                    self.gcs_client.download_file(file_path, local_path)
                    
                    # Open with rioxarray but only get metadata
                    # Decode coordinates if MODIS
                    sample_data = (
                        rxr.open_rasterio(local_path, variable=self.VARIABLE_NAME, decode_coords="all") 
                        if self.data_source == 'MODIS'
                        else rxr.open_rasterio(local_path, variable=self.VARIABLE_NAME)
                    )
                    
                    break
                except Exception as e:
                    logger.warning(f"Error getting sample data from {file_info['path']}: {str(e)}")
            
            # Check if we successfully got sample data
            if sample_data is None:
                logger.warning(f"No valid sample data found for {year}")
                return False
            
            # Process each file and store in a list for later concatenation
            array_list = []
            for _, file_info in files_df.iterrows():
                file_path = file_info['path']
                local_path = os.path.join(temp_dir, os.path.basename(file_path))
                
                if not os.path.exists(local_path):
                    self.gcs_client.download_file(file_path, local_path)
                    
                t_data = (
                    rxr.open_rasterio(local_path, variable=self.VARIABLE_NAME, decode_coords="all", chunks={k:self.chunk_size[k] for k in ('band','y','x') if k in self.chunk_size})
                    if self.data_source == 'MODIS'
                    else rxr.open_rasterio(local_path, variable=self.VARIABLE_NAME, chunks={k:self.chunk_size[k] for k in ('band','y','x') if k in self.chunk_size})
                )
                
                array_list.append(t_data[self.VARIABLE_NAME])
            
            # Concatenate all arrays along time dimension
            combined_data = xr.concat(array_list, dim='day')
            
            # Rename day dimension to time
            combined_data = combined_data.rename({"day": "time"})
            
            # Convert day-of-year to proper datetime coordinates
            combined_data = combined_data.assign_coords({
                'time': pd.to_datetime([f"{year}{day:03d}" for day in files_df['day'].values], format="%Y%j"),
            })
            
            # Convert to dataset with proper variable name
            combined_data = combined_data.to_dataset(name=self.VARIABLE_NAME)
            
            # Calculate annual and monthly statistics using Dask
            logger.info("Calculating statistics with Dask")
            annual_stats, monthly_stats = self._calculate_statistics(combined_data)
            
            # Set up compression for Zarr output
            compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=2)#compressor = zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle) #
            encoding = {var: {'compressors': compressor} for var in annual_stats.data_vars}# edit: 'compressors':
            
            # Create temporary paths for zarr storage
            output_basename = os.path.basename(output_path.rstrip('.zarr'))
            annual_output_path = Path(temp_dir) / f"{output_basename}_annual.zarr"
            monthly_output_path = Path(temp_dir) / f"{output_basename}_monthly.zarr"
            
            # Save in smaller chunks to avoid large graphs
            logger.info(f"Saving annual statistics to temporary location with compression")
            annual_stats.to_zarr(
                str(annual_output_path),
                mode="w",
                encoding=encoding,
                zarr_version=2,
                consolidated=True
            )
        
            
            logger.info(f"Saving monthly statistics to temporary location with compression")
            monthly_stats.to_zarr(
                str(monthly_output_path),
                mode="w",
                encoding=encoding,
                zarr_version=2,
                consolidated=True
            )
            
            # Upload to cloud storage - this is the only persistent storage
            upload_success = self._upload_to_cloud(annual_output_path, monthly_output_path, year, grid_cell)
            
            return upload_success

    def _calculate_statistics(self, data: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Calculate annual and monthly statistics from daily data using Dask.
        """
        # Create a mask for invalid values - using dask arrays
        mask = da.logical_and(data[self.VARIABLE_NAME] >= 20000, 
                          data[self.VARIABLE_NAME] <= 35000)
        masked = data.where(mask)
        
        # Set better chunk sizes for time-based resampling
        # This is critical for resampling performance
        rechunked = masked.chunk({'time': -1, 'x': self.chunk_size.get('x', 500), 
                             'y': self.chunk_size.get('y', 500)})
        
        # Calculate annual statistics
        annual_stats = xr.Dataset({
            "mean": rechunked[self.VARIABLE_NAME].resample(time="1YE").mean(),
            "median": rechunked[self.VARIABLE_NAME].resample(time="1YE").median(),
            "std": rechunked[self.VARIABLE_NAME].resample(time="1YE").std(),
            "valid_count": mask.resample(time="1YE").sum().astype(np.int32)
        })
        
        # Calculate monthly statistics
        monthly_stats = xr.Dataset({
            "mean": data[self.VARIABLE_NAME].resample(time="1ME").mean(),
            "median": data[self.VARIABLE_NAME].resample(time="1ME").median(),
            "std": data[self.VARIABLE_NAME].resample(time="1ME").std(),
            "valid_count": mask.resample(time="1ME").sum().astype(np.int32)
        })
        
        # Rechunk using the same chunk sizes defined in the class initialization
        # For spatial dimensions (x and y), use the values from self.chunk_size
        # For time dimension, always use 1 to ensure each timestep is in its own chunk
        chunk_dict = {"time": 1}
        
        # Add x and y dimensions from the original chunk_size if they exist
        if 'x' in self.chunk_size:
            chunk_dict['x'] = self.chunk_size['x']
        if 'y' in self.chunk_size:
            chunk_dict['y'] = self.chunk_size['y']
        
        # Apply the chunking to both datasets
        annual_stats = annual_stats.chunk(chunk_dict)
        monthly_stats = monthly_stats.chunk(chunk_dict)
        
        return annual_stats, monthly_stats
    
    def _upload_to_cloud(self, annual_path: Path, monthly_path: Path, year: int, grid_cell: str = None) -> bool:
        """
        Upload compressed processed data to cloud storage.
        
        Args:
            annual_path: Local path to annual statistics
            monthly_path: Local path to monthly statistics
            year: Year being processed
            grid_cell: Optional grid cell identifier
            
        Returns:
            Boolean indicating success
        """
        try:
            # Determine the cloud storage paths
            data_type = self.data_source.lower()
            
            if grid_cell is None and self.data_source == "MODIS":
                # For MODIS, try to extract grid_cell from the annual_path filename if not provided
                if annual_path.stem.split('_')[-2:-1]:
                    grid_cell = annual_path.stem.split('_')[-2]
                else:
                    grid_cell = "unknown"
            elif grid_cell is None:
                # For AVHRR, use "global" as the grid cell identifier
                grid_cell = "global"
            
            # Create cloud paths using preprocessing_index paths (like EOG preprocessor)
            # For annual data
            if grid_cell == "global":
                annual_cloud_path = f"{self.preprocessing_index.annual_path}{year}.zarr"
                monthly_cloud_path = f"{self.preprocessing_index.annual_path}{year}_monthly.zarr"
            else:
                annual_cloud_path = f"{self.preprocessing_index.annual_path}{year}/{grid_cell}.zarr"
                monthly_cloud_path = f"{self.preprocessing_index.annual_path}{year}/{grid_cell}_monthly.zarr"
            
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
            return True
                
        except Exception as e:
            logger.error(f"Error uploading to cloud: {str(e)}")
            return False
    
    def finalize_stage1(self) -> None:
        """
        Finalize Stage 1 processing.
        
        Creates a manifest of processed files in the cloud and ensures the index is up to date.
        """
        logger.info("Finalizing Stage 1 processing")
        
        # Create a new Dask client just for this task
        with DaskClientContextManager(**self._get_dask_client_params()) as client:
            logger.info(f"Created Dask client for finalizing stage 1: {client.dashboard_link}")
            
            # Determine the manifest file path using the preprocessing_index path
            cloud_manifest_path = f"{self.preprocessing_index.annual_path}_manifest.csv"
            
            processed_files = []
            
            # Query the index for all completed annual files
            try:
                cursor = self.preprocessing_index._get_connection().cursor()
                cursor.execute(
                    "SELECT * FROM files WHERE stage = ? AND status = ? AND data_path = ?",
                    (PreprocessingIndex.STAGE_ANNUAL, PreprocessingIndex.STATUS_COMPLETED, self.data_path)
                )
                columns = [description[0] for description in cursor.description]
                for row in cursor.fetchall():
                    file_info = dict(zip(columns, row))
                    # Parse metadata if it exists
                    if file_info.get('metadata'):
                        try:
                            file_info['metadata'] = json.loads(file_info['metadata'])
                        except:
                            pass  # Keep as string if not valid JSON
                
                blob_path = file_info.get("blob_path", "")
                year = file_info.get("year")
                grid_cell = file_info.get("grid_cell", "global")
                
                processed_files.append({
                    "data_source": self.data_source,
                    "year": year,
                    "grid_cell": grid_cell,
                    "path": blob_path,
                    "file_hash": file_info.get("file_hash")
                })
            except Exception as e:
                logger.error(f"Error querying index: {e}")
        
        logger.info(f"Found {len(processed_files)} completed annual files in the index")
        
        # If the index doesn't have files (though it should), fall back to cloud listing
        if not processed_files:
            # Query the cloud storage for processed files using the preprocessing_index.annual_path
            annual_prefix = self.preprocessing_index.annual_path
            try:
                # List all annual stats files in the cloud
                cloud_files = self.gcs_client.list_existing_files(prefix=annual_prefix)
                logger.info(f"Found {len(cloud_files)} annual files in cloud storage at {annual_prefix}")
                
                # Process the list of files
                for file_path in cloud_files:
                    # Only process zarr files
                    if file_path.endswith(".zarr/.zmetadata") or file_path.endswith(".zarr/zgroup"):
                        # Extract path without metadata file
                        base_path = file_path.rsplit("/.zmetadata", 1)[0] if ".zmetadata" in file_path else file_path.rsplit("/zgroup", 1)[0]
                        
                        # Extract year and grid cell from path structure
                        path_parts = base_path.split('/')
                        
                        # Check if filename contains a year
                        year = None
                        grid_cell = "global"
                        
                        # Check for year in the path structure
                        for part in path_parts:
                            if part.isdigit() and 1980 <= int(part) <= 2030:
                                year = int(part)
                                break
                                
                        # For paths like year/gridcell.zarr
                        if year is not None and len(path_parts) > 2:
                            potential_grid_cell = path_parts[-1].replace(".zarr", "")
                            if potential_grid_cell.startswith("h") and "v" in potential_grid_cell:
                                grid_cell = potential_grid_cell
                        
                        # Skip if we couldn't extract year
                        if year is None:
                            logger.warning(f"Could not extract year from path: {file_path}")
                            continue
                            
                        processed_files.append({
                            "data_source": self.data_source,
                            "year": year,
                            "grid_cell": grid_cell,
                            "path": base_path
                        })
                
                logger.warning(f"Found {len(processed_files)} files in cloud not in index; adding them to index")
                
                # Add these files to the index if they weren't there already
                for file_info in processed_files:
                    # Only add if it has year and other required info
                    if "year" in file_info and file_info["year"]:
                        try:
                            year = int(file_info["year"])
                            grid_cell = file_info["grid_cell"]
                            blob_path = file_info["path"]
                            
                            # Check if this file already exists in the index
                            cursor.execute(
                                "SELECT COUNT(*) FROM files WHERE stage = ? AND year = ? AND grid_cell = ? AND blob_path = ?",
                                (PreprocessingIndex.STAGE_ANNUAL, year, grid_cell, blob_path)
                            )
                            count = cursor.fetchone()[0]
                            
                            if count == 0:
                                # Add to index with completed status
                                file_hash = self.preprocessing_index.add_file(
                                    stage=PreprocessingIndex.STAGE_ANNUAL,
                                    year=year,
                                    grid_cell=grid_cell,
                                    status=PreprocessingIndex.STATUS_COMPLETED,
                                    blob_path=blob_path,
                                    metadata={
                                        "filename": os.path.basename(blob_path),
                                        "data_source": self.data_source,
                                        "imported_from_cloud": datetime.now().isoformat()
                                    }
                                )
                                logger.info(f"Added file to index: {year}/{grid_cell} at {blob_path}")
                        except Exception as e:
                            logger.error(f"Error adding cloud file to index: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error listing cloud files: {str(e)}")
        
        # Save index to make sure all changes are persisted
        self.preprocessing_index.save()
        
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
            logger.warning("No processed files found to include in manifest")

    def project_to_unified_grid(self) -> None:
        """
        Stage 2: Project data onto a unified grid.
        
        This method would be run on the university cluster after stage 1 files are
        transferred. It reads from the preprocessing index to determine which files
        need processing.
        """
        # Create a new Dask client just for this task
        with DaskClientContextManager(**self._get_dask_client_params()) as client:
            logger.info(f"Created Dask client for projection task: {client.dashboard_link}")
            logger.warning("Stage 2 (project_to_unified_grid) not yet implemented for GLASS data")
            
            # Get files that are ready for stage 2 processing
            files_to_process = []
            try:
                # Use direct SQL query instead of the missing method
                conn = self.preprocessing_index._get_connection()
                cursor = conn.cursor()
                
                # Query for annual files that are completed but don't have spatial counterparts
                query = """
                SELECT a.* 
                FROM files a
                WHERE a.stage = ? AND a.status = ?
                AND NOT EXISTS (
                    SELECT 1 FROM files b 
                    WHERE b.stage = ? AND b.year = a.year AND b.grid_cell = a.grid_cell
                    AND (b.status = ? OR b.status = ?)
                )
                LIMIT 10
                """
                cursor.execute(query, (
                    PreprocessingIndex.STAGE_ANNUAL, 
                    PreprocessingIndex.STATUS_COMPLETED,
                    PreprocessingIndex.STAGE_SPATIAL,
                    PreprocessingIndex.STATUS_COMPLETED, 
                    PreprocessingIndex.STATUS_PROCESSING
                ))
                
                columns = [description[0] for description in cursor.description]
                for row in cursor.fetchall():
                    file_info = dict(zip(columns, row))
                    # Parse metadata if it exists
                    if file_info.get('metadata'):
                        try:
                            file_info['metadata'] = json.loads(file_info['metadata'])
                        except:
                            pass
                    files_to_process.append(file_info)
                    
            except Exception as e:
                logger.error(f"Error querying for files to process: {e}")
            
            logger.info(f"Found {len(files_to_process)} files ready for Stage 2 processing")
            
            # Process each file
            for file_info in files_to_process:
                try:
                    year = file_info["year"]
                    grid_cell = file_info["grid_cell"]
                    
                    # Register a new spatial file in the index
                    spatial_file_hash = self.preprocessing_index.add_file(
                        stage=PreprocessingIndex.STAGE_SPATIAL,
                        year=year,
                        grid_cell=grid_cell,
                        status=PreprocessingIndex.STATUS_PROCESSING,
                        parent_hash=file_info["file_hash"],
                        metadata={
                            "processing_started": datetime.now().isoformat(),
                            "source": self.data_source,
                            "parent_blob": file_info["blob_path"]
                        }
                    )
                    
                    # In a real implementation, process the file here
                    # This is just a placeholder
                    
                    # Update the status to completed
                    self.preprocessing_index.update_file_status(
                        file_hash=spatial_file_hash,
                        status=PreprocessingIndex.STATUS_COMPLETED,
                        metadata={
                            "processing_completed": datetime.now().isoformat(),
                            "note": "This is a placeholder - no actual processing was performed"
                        }
                    )
                    
                except Exception as e:
                    logger.error(f"Error in stage 2 processing for {file_info.get('year')}/{file_info.get('grid_cell')}: {str(e)}")

    def preprocess(self) -> bool:
        """
        Main preprocessing method that performs all necessary preprocessing steps.
        
        Returns:
            Boolean indicating overall success
        """
        try:
            # Validate input data
            if not self.validate_input():
                logger.error("Input validation failed")
                return False
                
            # Stage 1: Create annual summaries
            logger.info("Starting Stage 1: Creating annual summaries")
            self.summarize_annual_means()
            
            # Finalize Stage 1 and prepare for Stage 2
            self.finalize_stage1()
            
            # Save the index for transfer between environments
            self.preprocessing_index.save()
            
            return True
            
        except Exception as e:
            logger.exception(f"Error in preprocessing: {str(e)}")
            return False
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GlassPreprocessor":
        """
        Create a GlassPreprocessor from a configuration dictionary.
        
        Args:
            config: Dictionary containing configuration parameters
            
        Returns:
            GlassPreprocessor instance
        """
        return cls(**config)

    def _get_cloud_paths(self, year: int, grid_cell: str = None) -> Tuple[str, str]:
        """
        Return standardized cloud paths for a given year and grid cell.
        
        Args:
            year: Year being processed
            grid_cell: Optional grid cell identifier (defaults to "global" for AVHRR)
            
        Returns:
            Tuple of (annual_path, monthly_path)
        """
        # Use global as default for AVHRR or if not specified
        if grid_cell is None:
            grid_cell = "global"
            
        # Generate paths based on grid cell
        if grid_cell == "global":
            annual_path = f"{self.preprocessing_index.annual_path}{year}.zarr"
            monthly_path = f"{self.preprocessing_index.annual_path}{year}_monthly.zarr"
        else:
            annual_path = f"{self.preprocessing_index.annual_path}{year}/{grid_cell}.zarr"
            monthly_path = f"{self.preprocessing_index.annual_path}{year}/{grid_cell}_monthly.zarr"
            
        return annual_path, monthly_path
    
    def _check_output_exists(self, year: int, grid_cell: str) -> bool:
        """
        Check if output already exists for a given year and grid cell.
        
        Args:
            year: Year to check
            grid_cell: Grid cell to check
        
        Returns:
            Boolean indicating whether output exists
        """
        # Get the annual cloud path
        annual_cloud_path, _ = self._get_cloud_paths(year, grid_cell)
        
        # First check using the preprocessing index (preferred method)
        annual_files = self.preprocessing_index.get_files(
            stage=PreprocessingIndex.STAGE_ANNUAL,
            year=year,
            grid_cell=grid_cell,
            status=PreprocessingIndex.STATUS_COMPLETED
        )
        
        # Check if files exist in index or in GCS
        return bool(annual_files or self.preprocessing_index.is_blob_exists(f"{annual_cloud_path}/.zmetadata"))

    def _handle_processing_error(self, year: int, grid_cell: str, error: Exception) -> None:
        """
        Handle errors during processing by updating the index.
        
        Args:
            year: Year being processed
            grid_cell: Grid cell being processed
            error: Exception that occurred
        """
        logger.error(f"Error processing {year} {grid_cell}: {str(error)}")
        
        # Try to update any processing files for this year/grid_cell
        try:
            # Get files that are in processing state for this year/grid_cell
            processing_files = self.preprocessing_index.get_files(
                stage=PreprocessingIndex.STAGE_ANNUAL,
                year=year,
                grid_cell=grid_cell,
                status=PreprocessingIndex.STATUS_PROCESSING
            )
            
            # Update the first processing file with failure status
            if processing_files:
                self.preprocessing_index.update_file_status(
                    file_hash=processing_files[0]["file_hash"], 
                    status=PreprocessingIndex.STATUS_FAILED,
                    metadata={"error": str(error)}
                )
                logger.info(f"Updated file status to failed for {year} {grid_cell}")
        except Exception as ex:
            logger.error(f"Error updating file status: {ex}")
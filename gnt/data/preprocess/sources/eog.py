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
import re
import json

from gnt.data.preprocess.sources.base import AbstractPreprocessor
from gnt.data.common.gcs.client import GCSClient
from gnt.data.common.dask.client import init_dask_client, close_client, DaskClientContextManager
from gnt.data.common.index.preprocessing_index import PreprocessingIndex
from gnt.data.common.index.hpc_download_index import DataDownloadIndex
from gnt.data.download.sources.eog import EOGDataSource

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
    
    # Default cloud storage information
    BUCKET_NAME = "growthandheat"
    
    # Default data source URLs
    DMSP_BASE_URL = "https://eogdata.mines.edu/wwwdata/dmsp/v4composites_rearrange/"
    VIIRS_ANNUAL_BASE_URL = "https://eogdata.mines.edu/nighttime_light/annual/v21/"
    VIIRS_DVNL_BASE_URL = "https://eogdata.mines.edu/wwwdata/viirs_products/dvnl/"
    
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
                stage (str): Processing stage ("annual", "spatial")
                year (int, optional): Specific year to process
                year_range (list, optional): [start_year, end_year] range to process
                grid_cell (str, optional): Grid cell to process (for spatial stage)
                bucket_name (str): GCS bucket name
                
                # Data source parameters (matching download config)
                base_url (str): Base URL for the specific EOG data source to use
                output_path (str): GCS path where source data is stored
                file_extensions (list, optional): File extensions to filter by
                
                # Other parameters
                version (str): Processing version
                temp_dir (str): Directory for temporary files
                override (bool): Whether to reprocess existing outputs
                dask_threads (int): Number of threads for Dask processing
                dask_memory_limit (str): Memory limit for Dask
        """
        super().__init__(**kwargs)
        
        # Set processing stage
        self.stage = kwargs.get('stage', 'annual')
        if self.stage not in ['annual', 'spatial']:
            raise ValueError(f"Unsupported stage: {self.stage}. Use 'annual' or 'spatial'.")
            
        # Set year or year range
        self.year = kwargs.get('year')
        self.year_range = kwargs.get('year_range')
        
        # Validate year parameters
        if self.year is None and self.year_range is None:
            raise ValueError("Either 'year' or 'year_range' must be specified")
        
        # Process the year parameters
        if self.year is not None:
            # Single year mode
            self.year_start = self.year
            self.year_end = self.year
            self.years_to_process = [self.year]
        elif self.year_range is not None:
            # Year range mode
            if not isinstance(self.year_range, list) or len(self.year_range) != 2:
                raise ValueError("'year_range' must be a list with exactly two elements [start_year, end_year]")
            
            self.year_start = self.year_range[0]
            self.year_end = self.year_range[1]
            
            # Validate range values
            if not isinstance(self.year_start, int) or not isinstance(self.year_end, int):
                raise ValueError("Year range values must be integers")
            
            # Ensure year_start <= year_end
            if self.year_start > self.year_end:
                raise ValueError(f"year_range start ({self.year_start}) must be less than or equal to year_range end ({self.year_end})")
            
            # Calculate the list of years to process
            self.years_to_process = list(range(self.year_start, self.year_end + 1))
            
            logger.info(f"Processing year range: {self.year_start}-{self.year_end} ({len(self.years_to_process)} years)")
        
        self.grid_cell = kwargs.get('grid_cell')
        if self.stage == 'spatial' and not self.grid_cell:
            raise ValueError("Grid cell must be specified for spatial stage")
        
        # Settings
        self.override = kwargs.get('override', False)
        self.bucket_name = kwargs.get('bucket_name', self.BUCKET_NAME)
        
        # Get data source parameters
        base_url = kwargs.get('base_url')
        output_path = kwargs.get('output_path')
        file_extensions = kwargs.get('file_extensions')
        
        # Validate required parameters
        if not base_url:
            raise ValueError("base_url parameter must be specified")
        if not output_path:
            raise ValueError("output_path parameter must be specified")
        
        # Create the download data source
        self.data_source = EOGDataSource(
            base_url=base_url,
            file_extensions=file_extensions,
            output_path=output_path
        )
            
        # Initialize download index for finding source files
        self._init_download_index(kwargs)
        
        # Derive paths for input/output
        self.input_path = self.data_source.data_path
        
        # Initialize preprocessing index
        self._init_preprocessing_index(kwargs)
        
        # Derive source_type from data source properties
        self._derive_source_type()
        
        # Log paths for verification
        logger.info(f"Data source: {self.source_type} from {self.data_source.base_url}")
        logger.info(f"Input path: {self.input_path}")
        logger.info(f"Output paths: annual={self.preprocessing_index.annual_path}, spatial={self.preprocessing_index.spatial_path}")
        
        # Initialize GCS client
        self.gcs_client = GCSClient(self.bucket_name)
        
        # Dask configuration
        self.chunk_size = kwargs.get('chunk_size', self.DEFAULT_CHUNK_SIZE)
        self.dask_threads = kwargs.get('dask_threads', None)
        self.dask_memory_limit = kwargs.get('dask_memory_limit', None)
        
        # Status cache to reduce database access
        self._status_cache = {}
        self._max_cache_size = 10000
        
        # Track operations for periodic saves
        self._operations_since_save = 0
            
        if len(self.years_to_process) == 1:
            logger.info(f"Initialized EOGPreprocessor for year {self.years_to_process[0]}")
        else:
            logger.info(f"Initialized EOGPreprocessor for years {self.year_start}-{self.year_end}")
            
        if self.grid_cell:
            logger.info(f"Processing grid cell: {self.grid_cell}")
        
    def _init_preprocessing_index(self, config):
        """Initialize the preprocessing index."""
        version = config.get("version", "v1")
        temp_dir = config.get("temp_dir")
        
        self.preprocessing_index = PreprocessingIndex(
            bucket_name=self.bucket_name,
            data_path=self.input_path,
            version=version,
            temp_dir=temp_dir
        )
    
    def _init_download_index(self, config):
        """Initialize the download index to query for downloaded files."""
        # Load the index from GCS
        try:
            temp_dir = config.get("temp_dir")
        
            # Use one the existing data source for the download index
            data_source = self.data_source
            
            # Create the download index
            self.download_index = DataDownloadIndex(
                bucket_name=self.bucket_name,
                data_source=data_source,
                temp_dir=temp_dir
            )

            logger.info("Download index loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load download index: {e}")
    
    def find_data_files_for_year(self, year: int) -> List[str]:
        """
        Find data files for a specific year.
        
        This method returns all available files for a given year without type prioritization,
        since the preprocessor is initialized for a specific data type.
        
        Args:
            year: Year to find data for
            
        Returns:
            List of file paths matching the year
        """
        year_str = str(year)
        
        # Get all successfully downloaded files from the download index
        all_files = []
        
        # First, try to get files from the download index
        if hasattr(self, 'download_index') and self.download_index:
            logger.info(f"Querying download index for EOG data files for year {year}")
            try:
                prefix = f"{self.input_path}"
                downloaded_files = self.download_index.list_successful_files(prefix=prefix)
                
                # Filter by year string in filename
                all_files = [f for f in downloaded_files if year_str in f]
                logger.info(f"Found {len(all_files)} files that include year {year} in download index")
                
            except Exception as e:
                logger.warning(f"Error querying download index: {e}")

        # If download index failed or returned no results, try GCS directly
        if not all_files:
            logger.info(f"Listing EOG files directly from GCS for year {year}")
            gcs_files = self.gcs_client.list_existing_files(self.input_path)
            all_files = [f for f in gcs_files if year_str in f]
            logger.info(f"Found {len(all_files)} files in GCS for year {year}")

        # No files found for this year
        if not all_files:
            logger.warning(f"No data files found for year {year}")
            
        return all_files
    
    def _extract_dmsp_sensor(self, filename: str) -> int:
        """
        Extract the sensor number from a DMSP filename.
        Used for sorting to find the latest sensor.
        
        Args:
            filename: DMSP filename
            
        Returns:
            Sensor number as integer, or 0 if not found
        """
        filename = os.path.basename(filename)
        match = re.search(r'F(\d+)(\d{4})', filename)
        if match:
            return int(match.group(1))
        return 0
    
    def _create_annual_zarr(self, data_array: xr.DataArray, year: int) -> str:
        """
        Create an annual .zarr file from the dataset.
        
        Args:
            year: Year for the data
            dataarray: xarray DataArray to save
            
        Returns:
            GCS path to created zarr file
        """
        # Existing implementation kept
        # Define output path
        output_path = f"{self.preprocessing_index.annual_path}{year}.zarr"
        
        # Define chunking strategy (adjust based on your data characteristics)
        chunks = {"x": 1000, "y": 1000}
        
        # Create local temp directory for zarr
        import tempfile
        temp_dir = tempfile.mkdtemp()
        local_zarr = os.path.join(temp_dir, f"{year}.zarr")
        
        try:
            # Turn into Dataset if not already
            dataset = data_array.to_dataset(name=self.source_type)
                
            # Write to local zarr with compression
            # Set up compression for Zarr output
            compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=2)#compressor = zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle) #
            encoding = {var: {'compressors': compressor} for var in dataset.data_vars}# edit: 'compressors':
            
            dataset.chunk(chunks).to_zarr(
                local_zarr, 
                mode="w", 
                encoding=encoding, 
                zarr_version=2, 
                consolidated=True
                )
            
            # Upload to GCS
            self.upload_directory(local_zarr, output_path)
            
            logger.info(f"Created annual zarr file for {year} at {output_path}")
            return output_path
            
        finally:
            # Clean up local files
            import shutil
            shutil.rmtree(temp_dir)
    
    def process_annual_for_year(self, year: int) -> str:
        """
        Process data for a specific year and create annual zarr file.
        
        Args:
            year: Year to process
            
        Returns:
            GCS path to created zarr file or None if failed
        """
        logger.info(f"Processing annual data for year {year}")
        
        # Check if this year is already processed
        if hasattr(self, 'preprocessing_index') and self.preprocessing_index:
            existing = self.preprocessing_index.get_files(
                stage=PreprocessingIndex.STAGE_ANNUAL,
                year=year,
                status=PreprocessingIndex.STATUS_COMPLETED
            )
            
            if existing and not self.override:
                logger.info(f"Annual data for year {year} already exists, skipping")
                return existing[0]['blob_path']
        
        # Register the file we're about to create in the index
        file_hash = None
        if hasattr(self, 'preprocessing_index') and self.preprocessing_index:
            file_hash = self.preprocessing_index.add_file(
                stage=PreprocessingIndex.STAGE_ANNUAL,
                year=year,
                grid_cell=None,  # Not using grid cells for annual files
                status=PreprocessingIndex.STATUS_PROCESSING,
                blob_path=f"{self.preprocessing_index.annual_path}{year}.zarr",
                metadata={
                    "processing_start": datetime.now().isoformat()
                }
            )

        # Get files for this year
        all_files = self.find_data_files_for_year(year)
        
        if not all_files:
            logger.warning(f"No data found for year {year}")
            
            # Update index if available
            if hasattr(self, 'preprocessing_index') and self.preprocessing_index and file_hash:
                self.preprocessing_index.update_file_status(
                    file_hash=file_hash,
                    status=PreprocessingIndex.STATUS_FAILED,
                    metadata={
                        "error": "No source data found",
                        "processing_end": datetime.now().isoformat()
                    }
                )
            
            return None
            
        # Filter files based on source type
        filtered_file = self.filter_files_for_processing(all_files)
        
        if not filtered_file:
            logger.warning(f"No files selected after filtering for year {year}")
            
            # Update index if available
            if hasattr(self, 'preprocessing_index') and self.preprocessing_index and file_hash:
                self.preprocessing_index.update_file_status(
                    file_hash=file_hash,
                    status=PreprocessingIndex.STATUS_FAILED,
                    metadata={
                        "error": "No files selected after filtering",
                        "processing_end": datetime.now().isoformat()
                    }
                )
            
            return None
        
        # Process the filtered files
        try:
            data_array = self._process_data_files(filtered_file, year)
            
            if data_array is None:
                logger.error(f"Failed to create DataArray for year {year}")
                
                # Update index if available
                if hasattr(self, 'preprocessing_index') and self.preprocessing_index and file_hash:
                    self.preprocessing_index.update_file_status(
                        file_hash=file_hash,
                        status=PreprocessingIndex.STATUS_FAILED,
                        metadata={
                            "error": "Failed to create DataArray",
                            "processing_end": datetime.now().isoformat()
                        }
                    )
                
                return None
                
            # Create annual zarr file
            output_path = self._create_annual_zarr(data_array, year)
            
            # Update index if available
            if hasattr(self, 'preprocessing_index') and self.preprocessing_index and file_hash:
                self.preprocessing_index.update_file_status(
                    file_hash=file_hash,
                    status=PreprocessingIndex.STATUS_COMPLETED,
                    metadata={
                        "source_files": filtered_file,
                        "original_files_count": len(all_files),
                        "processing_end": datetime.now().isoformat()
                    }
                )
                
                # Mark for transfer to university cluster
                self.preprocessing_index.mark_for_transfer(file_hash, destination="cluster")

            return output_path
            
        except Exception as e:
            logger.exception(f"Error processing annual data for year {year}: {e}")
            
            # Update index if available
            if hasattr(self, 'preprocessing_index') and self.preprocessing_index and file_hash:
                self.preprocessing_index.update_file_status(
                    file_hash=file_hash,
                    status=PreprocessingIndex.STATUS_FAILED,
                    metadata={
                        "error": str(e),
                        "processing_end": datetime.now().isoformat()
                    }
                )
            
            return None
    
    def process_annual(self) -> bool:
        """
        Process annual data for all years in the configured range.
        
        Returns:
            True if all years were processed successfully, False otherwise
        """
        results = []
        
        for year in self.years_to_process:
            # Set current year for processing (for compatibility)
            self.year = year
            
            # Process the year
            result = self.process_annual_for_year(year)
            results.append(result is not None)
            
            # Save index after each year
            if hasattr(self, 'preprocessing_index'):
                self.preprocessing_index.save()
        
        # Return True only if all years were processed successfully
        return all(results)
    
    def process_spatial_for_year_cell(self, year: int, grid_cell: str) -> str:
        """
        Process spatial transformation for a specific year and grid cell.
        
        Args:
            year: Year to process
            grid_cell: Grid cell to process
            
        Returns:
            Path to created spatial file or None if failed
        """
        logger.info(f"Processing spatial data for year {year}, grid cell {grid_cell}")
        
        # Check if this combination is already processed
        if hasattr(self, 'preprocessing_index') and self.preprocessing_index:
            existing = self.preprocessing_index.get_files(
                stage=PreprocessingIndex.STAGE_SPATIAL,
                year=year,
                grid_cell=grid_cell,
                status=PreprocessingIndex.STATUS_COMPLETED
            )
            
            if existing and not self.override:
                logger.info(f"Spatial data for year {year}, grid cell {grid_cell} already exists, skipping")
                return existing[0]['blob_path']
        
        # Find the annual file for this year
        annual_files = []
        if hasattr(self, 'preprocessing_index') and self.preprocessing_index:
            annual_files = self.preprocessing_index.get_files(
                stage=PreprocessingIndex.STAGE_ANNUAL,
                year=year,
                status=PreprocessingIndex.STATUS_COMPLETED
            )
        
        if not annual_files:
            logger.error(f"No annual data found for year {year}")
            return None
        
        # Get the annual file location
        annual_file = annual_files[0]
        annual_blob_path = annual_file['blob_path']
        
        # Register the spatial file we're about to create
        output_filename = f"eog_{year}_grid{grid_cell}.parquet"
        output_blob_path = f"{self.preprocessing_index.spatial_path}{year}/gridcell{grid_cell}/{output_filename}"
        
        file_hash = None
        if hasattr(self, 'preprocessing_index') and self.preprocessing_index:
            file_hash = self.preprocessing_index.add_file(
                stage=PreprocessingIndex.STAGE_SPATIAL,
                year=year,
                grid_cell=grid_cell,
                status=PreprocessingIndex.STATUS_PROCESSING,
                blob_path=output_blob_path,
                parent_hash=annual_file['file_hash'],
                metadata={
                    "processing_start": datetime.now().isoformat()
                }
            )
        
        try:
            # Download the annual zarr file to a temporary location
            temp_dir = tempfile.mkdtemp()
            local_zarr = os.path.join(temp_dir, f"{year}.zarr")
            
            # Download zarr directory from GCS
            self.download_directory(annual_blob_path, local_zarr)
            
            # Open annual data
            annual_ds = xr.open_zarr(local_zarr)
            
            # Extract grid cell based on coordinates
            cell_bounds = self._get_grid_cell_bounds(grid_cell)
            
            # Slice dataset to the grid cell bounds
            cell_ds = annual_ds.sel(
                x=slice(cell_bounds['lon_min'], cell_bounds['lon_max']),
                y=slice(cell_bounds['lat_min'], cell_bounds['lat_max'])
            )
            
            # Convert to tabular format
            var_name = list(cell_ds.data_vars)[0]
            data_array = cell_ds[var_name].values
            coords_x = cell_ds.x.values
            coords_y = cell_ds.y.values
            
            # Create meshgrid of coordinates
            xx, yy = np.meshgrid(coords_x, coords_y)
            
            # Create DataFrame
            df = pd.DataFrame({
                'x': xx.flatten(),
                'y': yy.flatten(),
                'value': data_array.flatten()
            })
            
            # Remove NaN values
            df = df.dropna()
            
            # Save to temporary parquet file
            temp_parquet = os.path.join(temp_dir, output_filename)
            df.to_parquet(temp_parquet, index=False)
            
            # Upload to GCS
            self.gcs_client.upload_file(temp_parquet, output_blob_path)
            
            # Update index
            if hasattr(self, 'preprocessing_index') and self.preprocessing_index and file_hash:
                self.preprocessing_index.update_file_status(
                    file_hash=file_hash,
                    status=PreprocessingIndex.STATUS_COMPLETED,
                    metadata={
                        "annual_source": annual_file['blob_path'],
                        "processing_end": datetime.now().isoformat(),
                        "grid_cell": grid_cell,
                        "row_count": len(df)
                    }
                )
            
            logger.info(f"Created spatial file for year {year}, grid cell {grid_cell} at {output_blob_path}")
            
            # Clean up
            shutil.rmtree(temp_dir)
            return output_blob_path
            
        except Exception as e:
            logger.exception(f"Error processing spatial data for year {year}, grid cell {grid_cell}: {e}")
            
            # Update index if available
            if hasattr(self, 'preprocessing_index') and self.preprocessing_index and file_hash:
                self.preprocessing_index.update_file_status(
                    file_hash=file_hash,
                    status=PreprocessingIndex.STATUS_FAILED,
                    metadata={
                        "error": str(e),
                        "processing_end": datetime.now().isoformat()
                    }
                )
            
            # Clean up any temp files
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            return None
    
    def process_spatial(self) -> bool:
        """
        Process spatial transformation for all years in the configured range.
        
        Returns:
            True if all years and grid cells were processed successfully, False otherwise
        """
        results = []
        
        for year in self.years_to_process:
            # Set current year for processing (for compatibility)
            self.year = year
            
            # Process the year with the current grid cell
            result = self.process_spatial_for_year_cell(year, self.grid_cell)
            results.append(result is not None)
            
            # Save index after each year/cell combination
            if hasattr(self, 'preprocessing_index'):
                self.preprocessing_index.save()
        
        # Return True only if all combinations were processed successfully
        return all(results)
    
    def _get_grid_cell_bounds(self, grid_cell: str) -> Dict[str, float]:
        """
        Get the geographical bounds for a grid cell.
        This is a placeholder - replace with actual grid cell definitions.
        
        Args:
            grid_cell: Grid cell identifier
            
        Returns:
            Dictionary with lat/lon bounds
        """
        # Existing implementation kept
        # This is just a placeholder - implement your actual grid cell logic
        # For example, parsing grid cell "N30E100" to get bounds
        match = re.match(r'([NS])(\d+)([EW])(\d+)', grid_cell)
        if not match:
            raise ValueError(f"Invalid grid cell format: {grid_cell}")
            
        ns, lat, ew, lon = match.groups()
        lat = int(lat)
        lon = int(lon)
        
        if ns == 'S':
            lat = -lat
        if ew == 'W':
            lon = -lon
            
        # Define bounds as 10-degree cells
        return {
            'lat_min': lat,
            'lat_max': lat + 10,
            'lon_min': lon,
            'lon_max': lon + 10
        }
    
    def preprocess(self) -> bool:
        """
        Main preprocessing method called by workflow.
        Dispatches to appropriate processing method based on stage.
        
        Returns:
            True if processing was successful, False otherwise
        """
        try:
            if self.stage == "annual":
                return self.process_annual()
                
            elif self.stage == "spatial":
                return self.process_spatial()
                
            else:
                logger.error(f"Unknown stage: {self.stage}")
                return False
                
        except Exception as e:
            logger.exception(f"Error in preprocessing: {e}")
            return False
    
    def _derive_source_type(self):
        """
        Derive the source type from the data source's base URL and output path.
        Sets self.source_type to one of: "dmsp", "viirs_annual", or "viirs_dvnl"
        """
        base_url = self.data_source.base_url
        output_path = self.data_source.data_path
        
        # First check the output path which often has descriptive directory names
        if "dmsp" in output_path.lower():
            self.source_type = "dmsp"
        elif "annual" in output_path.lower() or "stable_lights" in output_path.lower():
            self.source_type = "viirs_annual"
        elif "dvnl" in output_path.lower():
            self.source_type = "viirs_dvnl"
        # If output path doesn't help, check the base URL
        elif "dmsp" in base_url.lower():
            self.source_type = "dmsp"
        elif "annual" in base_url.lower():
            self.source_type = "viirs_annual"
        elif "dvnl" in base_url.lower() or "viirs_products" in base_url.lower():
            self.source_type = "viirs_dvnl"
        else:
            # Default to examining the file extensions
            if any("stable_lights" in ext for ext in self.data_source.file_extensions):
                self.source_type = "dmsp"
            elif any("median_masked" in ext for ext in self.data_source.file_extensions):
                self.source_type = "viirs_annual"
            else:
                # Default case
                logger.warning(f"Could not determine source_type from data source. Defaulting to viirs_dvnl")
                self.source_type = "viirs_dvnl"
        
        logger.info(f"Determined source type: {self.source_type}")
    
    def summarize_annual_means(self) -> None:
        """
        Stage 1: Summarize EOG data into annual mean files.
        
        For EOG data, this is equivalent to the process_annual method since
        the data is already organized annually. This method provides compatibility
        with the AbstractPreprocessor interface.
        """
        logger.info(f"Running annual summarization for EOG data ({self.source_type})")
        
        # For EOG, process_annual does the equivalent of summarizing annual means
        success = self.process_annual()
        
        if success:
            logger.info(f"Successfully created annual files for years {self.year_start}-{self.year_end}")
        else:
            logger.error(f"Failed to create annual files for some years in range {self.year_start}-{self.year_end}")

    def project_to_unified_grid(self) -> None:
        """
        Stage 2: Project EOG data onto a unified grid.
        
        For EOG data, this is equivalent to the process_spatial method.
        This method provides compatibility with the AbstractPreprocessor interface.
        """
        logger.info(f"Projecting EOG data to unified grid for years {self.year_start}-{self.year_end}")
        
        if not self.grid_cell:
            logger.error("Cannot project to unified grid without a grid_cell parameter")
            return
        
        # For EOG, process_spatial does the equivalent of projecting to a unified grid
        success = self.process_spatial()
        
        if success:
            logger.info(f"Successfully projected data to unified grid for years {self.year_start}-{self.year_end}")
        else:
            logger.error(f"Failed to project data for some years in range {self.year_start}-{self.year_end}")
    
    def filter_files_for_processing(self, files: List[str]) -> str:
        """
        Filter the list of files before processing based on source type.
        
        For DMSP: Select the file with the highest sensor number
        For other types: Select the most recent file
        
        Args:
            files: List of file paths to filter
            
        Returns:
            Filtered file path
        """
        if not files:
            return []
            
        logger.info(f"Filtering {len(files)} files before processing")
        
        if self.source_type == "dmsp":
            # Sort files by sensor number (descending)
            sensor_files = []
            for file_path in files:
                filename = os.path.basename(file_path)
                match = re.search(r'F(\d+)(\d{4})', filename)
                if match:
                    sensor = int(match.group(1))
                    sensor_files.append((sensor, file_path))
            
            # Sort by sensor number (descending)
            sensor_files.sort(reverse=True)
            
            if sensor_files:
                # Take the file with highest sensor number
                filtered = sensor_files[0][1]
                logger.info(f"Selected DMSP file with highest sensor {sensor_files[0][0]}: {os.path.basename(filtered[0])}")
                return filtered
        else:
            # For VIIRS products, take the most recent file based on filename
            # Assuming files are named with dates in a sortable format
            sorted_files = sorted(files)
            if sorted_files:
                # Take the last file (most recent)
                filtered = sorted_files[-1]
                logger.info(f"Selected most recent file for {self.source_type}: {os.path.basename(filtered[0])}")
                return filtered
    
        return ""

    def _process_data_files(self, file_path: str, year: int) -> xr.Dataset:
        """
        Unified processing function for EOG data files.
        Handles DMSP, VIIRS DVNL, and VIIRS annual data formats.
        
        Args:
            file: File path to process
            year: Year for the data
            
        Returns:
            xarray Dataset or None if processing failed
        """
        if not file_path:
            return None
            
        try:
            # Download file from GCS
            local_file = self.download_to_temp(file_path)
                
            # Uncompress if needed (.gz files, typically for VIIRS annual)
            if file_path.endswith(".gz"):
                import gzip
                import shutil
                uncompressed = local_file[:-3]  # Remove .gz
                with gzip.open(local_file, 'rb') as f_in:
                    with open(uncompressed, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(local_file)
                local_file = uncompressed
            
            # Open as xarray with rioxarray for geospatial metadata
            ds = rxr.open_rasterio(local_file)
            
            # Extract metadata based on source type and filename
            filename = os.path.basename(file_path)
            
            # Add t8je coordinate
            # Expand dimensions to include time and assign the coordinate
            ds = ds.expand_dims(
                dim={"time": 1}
            ).assign_coords(
                {"time": [pd.Timestamp(f"{year}-12-31")]}
            )
            
            
            if self.source_type == "dmsp":
                # Extract satellite and year from DMSP filename
                match = re.search(r'F(\d+)(\d{4})', filename)
                if match:
                    satellite = f"F{match.group(1)}"
                    
                    ds = ds.assign_attrs(
                        satellite=satellite,
                    )
            
            # Clean up the local file
            os.remove(local_file)
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None
        
        finally: 
            return ds
    
    def download_to_temp(self, file_path: str) -> str:
        """
        Download a file from GCS to a temporary local file.
        
        Args:
            file_path: Path to the file in GCS
            
        Returns:
            Path to the local temporary file
        """
        import tempfile
        import os
        
        # Create a temporary file with the same extension
        _, extension = os.path.splitext(file_path)
        fd, local_path = tempfile.mkstemp(suffix=extension)
        os.close(fd)
        
        # Download the file
        success = self.gcs_client.download_file(file_path, local_path)
        
        if not success:
            raise Exception(f"Failed to download file {file_path}")
        
        return local_path

    def download_directory(self, source_path: str, destination_path: str) -> bool:
        """
        Download a directory from GCS to a local path.
        
        Args:
            source_path: Path to the directory in GCS
            destination_path: Local path to download to
            
        Returns:
            True if successful, False otherwise
        """
        import os
        
        # Ensure the source path ends with a slash
        if not source_path.endswith('/'):
            source_path = source_path + '/'
        
        # List all files in the directory
        try:
            files = [blob.name for blob in self.gcs_client.client.list_blobs(
                self.gcs_client.bucket_name, prefix=source_path)]
            
            if not files:
                logger.warning(f"No files found at {source_path}")
                return False
            
            # Create the destination directory
            os.makedirs(destination_path, exist_ok=True)
            
            # Download each file
            for file_path in files:
                # Skip directory markers
                if file_path.endswith('/'):
                    continue
                    
                # Calculate the relative path
                rel_path = file_path[len(source_path):]
                local_path = os.path.join(destination_path, rel_path)
                
                # Create subdirectories if needed
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                # Download the file
                if not self.gcs_client.download_file(file_path, local_path):
                    logger.error(f"Failed to download {file_path}")
                    return False
            
            return True
        except Exception as e:
            logger.exception(f"Error downloading directory {source_path}: {e}")
            return False

    def upload_directory(self, local_path: str, destination_path: str) -> bool:
        """
        Upload a local directory to GCS.
        
        Args:
            local_path: Local path to upload from
            destination_path: Path in GCS to upload to
            
        Returns:
            True if successful, False otherwise
        """
        import os
        
        # Ensure the destination path ends with a slash
        if not destination_path.endswith('/'):
            destination_path = destination_path + '/'
        
        try:
            # Walk the local directory
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    # Calculate the relative path
                    local_file = os.path.join(root, file)
                    rel_path = os.path.relpath(local_file, local_path)
                    gcs_path = os.path.join(destination_path, rel_path).replace('\\', '/')
                    
                    # Upload the file
                    if not self.gcs_client.upload_file(local_file, gcs_path):
                        logger.error(f"Failed to upload {local_file}")
                        return False
                    
            
            return True
        except Exception as e:
            logger.exception(f"Error uploading directory {local_path}: {e}")
            return False
    
    def get_preprocessing_targets(self, stage: str, year_range: Tuple[int, int] = None) -> List[Dict[str, Any]]:
        """
        Generate list of preprocessing targets from unified download index.
        
        Args:
            stage: Processing stage ('annual' or 'spatial')
            year_range: Optional year range filter
            
        Returns:
            List of target dictionaries with source files and output specifications
        """
        if not hasattr(self, 'download_index') or not self.download_index:
            raise ValueError("Download index not available")
        
        # Query successfully downloaded files
        downloaded_files = self.download_index.query_pending_files(limit=None)  # Get all completed
        completed_files = [f for f in downloaded_files if f.get('download_status') == 'completed']
        
        if stage == 'annual':
            return self._generate_annual_targets(completed_files, year_range)
        elif stage == 'spatial':
            return self._generate_spatial_targets(completed_files, year_range)
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def _generate_annual_targets(self, files: List[Dict], year_range: Tuple[int, int] = None) -> List[Dict]:
        """Generate annual processing targets."""
        targets = []
        
        # Group files by year
        files_by_year = {}
        for file_info in files:
            year = self._extract_year_from_path(file_info['relative_path'])
            if year and (not year_range or year_range[0] <= year <= year_range[1]):
                if year not in files_by_year:
                    files_by_year[year] = []
                files_by_year[year].append(file_info)
        
        # Create targets for each year
        for year, year_files in files_by_year.items():
            # Filter to best file for the year (e.g., latest sensor for DMSP)
            selected_file = self._select_best_file_for_year(year_files)
            
            target = {
                'year': year,
                'stage': 'annual',
                'source_files': [selected_file],
                'output_path': f"{self.get_hpc_output_path('annual')}/{year}.zarr",
                'dependencies': [],
                'metadata': {
                    'source_type': self.source_type,
                    'total_candidates': len(year_files)
                }
            }
            targets.append(target)
        
        return targets
    
    def _generate_spatial_targets(self, files: List[Dict], year_range: Tuple[int, int] = None) -> List[Dict]:
        """Generate spatial processing targets."""
        # First, find completed annual files
        annual_targets = self._get_completed_annual_files(year_range)
        
        targets = []
        for annual_file in annual_targets:
            year = annual_file['year']
            
            # Generate targets for each grid cell
            for grid_cell in self.get_grid_cells():
                target = {
                    'year': year,
                    'grid_cell': grid_cell,
                    'stage': 'spatial',
                    'source_files': [annual_file],
                    'output_path': f"{self.get_hpc_output_path('spatial')}/{year}/grid_{grid_cell}.parquet",
                    'dependencies': [annual_file['output_path']],
                    'metadata': {
                        'grid_cell': grid_cell
                    }
                }
                targets.append(target)
        
        return targets
    
    def get_hpc_output_path(self, stage: str) -> str:
        """Get HPC output path for a given stage."""
        base_path = f"/cluster/work/climate/fschulz/preprocessing_outputs/{self.source_type}"
        return f"{base_path}/{stage}"
    
    # Add this class method to the EOGPreprocessor class
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'EOGPreprocessor':
        """
        Create an instance of the EOGPreprocessor from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Initialized EOGPreprocessor instance
        """
        return cls(**config)
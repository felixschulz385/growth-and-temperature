import os
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rxr
import dask
from dask.distributed import Client, LocalCluster
import dask.array as da
import numcodecs
import re
from functools import partial
from odc.geo import CRS
from odc.geo.xr import ODCExtensionDa, assign_crs, xr_reproject

from gnt.data.preprocess.sources.base import AbstractPreprocessor
from gnt.data.common.dask.client import DaskClientContextManager
from gnt.data.common.index.preprocessing_index import PreprocessingIndex
from gnt.data.common.geobox import get_or_create_geobox

# Add pandas for reading parquet files directly
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

logger = logging.getLogger(__name__)

class GlassPreprocessor(AbstractPreprocessor):
    """
    HPC-mode preprocessor for GLASS (Global LAnd Surface Satellite) LST data.
    
    Handles both MODIS (multiple files per day in grid cells) and AVHRR (one file per day) datasets.
    Uses Dask for distributed processing to handle datasets larger than memory.
    
    The preprocessing is organized in two stages:
    - Stage 1: Create annual zarr files from raw data
    - Stage 2: Reproject the data to a unified grid for analysis
    
    Requires HPC mode configuration with parquet index.
    """
    
    # Class constants
    BUCKET_NAME = "growthandheat"
    MODIS_PATH_PREFIX = "glass/LST/MODIS/Daily/1KM/"
    AVHRR_PATH_PREFIX = "glass/LST/AVHRR/0.05D/"
    VARIABLE_NAME = "LST"
    
    def __init__(self, **kwargs):
        """
        Initialize the Glass preprocessor in HPC mode.
        
        Args:
            **kwargs: Configuration parameters including:
                stage (str): Processing stage ("annual", "spatial")
                year (int, optional): Specific year to process
                year_range (list, optional): [start_year, end_year] range to process
                grid_cells (list, optional): List of grid cells to process (for MODIS)
                data_source (str): 'MODIS' or 'AVHRR'
                
                # Data source parameters
                base_url (str): Base URL for the data source
                data_path (str, optional): Data path where source data is stored
                file_extensions (list, optional): File extensions to filter by
                
                # HPC mode parameters
                hpc_target (str): Required - HPC root path
                
                # Other parameters
                temp_dir (str): Directory for temporary files
                override (bool): Whether to reprocess existing outputs
                dask_threads (int): Number of Dask threads
                dask_memory_limit (str): Dask memory limit
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
            self.year_start = self.year
            self.year_end = self.year
            self.years_to_process = [self.year]
        elif self.year_range is not None:
            if not isinstance(self.year_range, list) or len(self.year_range) != 2:
                raise ValueError("'year_range' must be a list with exactly two elements [start_year, end_year]")
            
            self.year_start = self.year_range[0]
            self.year_end = self.year_range[1]
            
            if not isinstance(self.year_start, int) or not isinstance(self.year_end, int):
                raise ValueError("Year range values must be integers")
            
            if self.year_start > self.year_end:
                raise ValueError(f"year_range start ({self.year_start}) must be less than or equal to year_range end ({self.year_end})")
            
            self.years_to_process = list(range(self.year_start, self.year_end + 1))
            logger.info(f"Processing year range: {self.year_start}-{self.year_end} ({len(self.years_to_process)} years)")
            
        # Set data source-specific attributes - FIXED LOGIC
        # Get the source name for fallback detection
        source_name = kwargs.get('type', '').lower()
        
        # Determinedata source
        if 'avhrr' in source_name:
            self.data_source = "AVHRR"
        elif 'modis' in source_name:
            self.data_source = "MODIS"
        
        # Set grid cells to process (for MODIS) - moved here before it's used
        self.grid_cells = kwargs.get('grid_cells', None)
        
        # Whether to override existing processed files
        self.override = kwargs.get('override', False)
        
        # Path prefix based on data source - MUST BE SET BEFORE data_path derivation
        self.path_prefix = self.MODIS_PATH_PREFIX if self.data_source == 'MODIS' else self.AVHRR_PATH_PREFIX
        
        # Get data source parameters
        base_url = kwargs.get('base_url')
        data_path = kwargs.get('data_path') or kwargs.get('output_path')
        
        # If data_path is not provided, derive it from the path_prefix - FIXED
        if not data_path:
            # Remove trailing slash to get base data path
            data_path = self.path_prefix.rstrip('/')
            logger.info(f"Derived data_path from path_prefix: {data_path}")
        
        file_extensions = kwargs.get('file_extensions', [".hdf"])
        
        # Validate required parameters
        if not base_url:
            raise ValueError("base_url parameter must be specified")
        
        # HPC mode parameters
        hpc_target = kwargs.get('hpc_target')
        self.hpc_root = self._strip_remote_prefix(hpc_target)
        if not self.hpc_root:
            raise ValueError("hpc_target is required for HPC mode")
        self.index_dir = os.path.join(self.hpc_root, "hpc_data_index")
        
        # Store the data path for later use
        self.data_path = data_path
        
        # Set base_url and file_extensions (used for index lookup)
        self.base_url = base_url
        self.file_extensions = file_extensions
        self.version = kwargs.get('version', 'v1')
        
        # Dask configuration
        self.dask_threads = kwargs.get('dask_threads', None)
        self.dask_memory_limit = kwargs.get('dask_memory_limit', None)
        self.chunk_size = kwargs.get('chunk_size', {"band": 1, "x": 500, "y": 500})
        self.dashboard_port = kwargs.get('dashboard_port', 8787)
        
        # Initialize parquet index path
        self._init_parquet_index_path()
        
        # Setup temporary directory
        self.temp_dir = kwargs.get('temp_dir')
        if not self.temp_dir:
            self.temp_dir = tempfile.mkdtemp(prefix=f"glass_{self.data_source}_processor_")
        else:
            os.makedirs(self.temp_dir, exist_ok=True)
            
        logger.info(f"Initialized GlassPreprocessor for {self.data_source}")
        logger.info(f"HPC root: {self.hpc_root}")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Years to process: {len(self.years_to_process)}")
        if self.grid_cells:
            logger.info(f"Grid cells: {self.grid_cells}")

    def _strip_remote_prefix(self, path):
        """Remove scp/ssh prefix like user@host: from paths."""
        if isinstance(path, str):
            return re.sub(r"^[^@]+@[^:]+:", "", path)
        return path

    def _init_parquet_index_path(self):
        """Initialize parquet index path based on data source."""
        safe_data_path = self.data_path.replace("/", "_").replace("\\", "_")
        os.makedirs(self.index_dir, exist_ok=True)
        # Default expected path
        default_path = os.path.join(
            self.index_dir,
            f"parquet_{safe_data_path}.parquet"
        )
        # Also try legacy/remote naming convention if available
        # e.g. parquet_glass_LST_MODIS_Daily_1KM.parquet
        alt_path = os.path.join(
            self.index_dir,
            f"parquet_{self.path_prefix.strip('/').replace('/','_')}.parquet"
        )
        # Use the one that exists, or default
        if os.path.exists(default_path):
            self.parquet_index_path = default_path
        elif os.path.exists(alt_path):
            self.parquet_index_path = alt_path
        else:
            self.parquet_index_path = default_path
        logger.debug(f"Parquet index path: {self.parquet_index_path}")

    def get_preprocessing_targets(self, stage: str, year_range: Tuple[int, int] = None) -> List[Dict[str, Any]]:
        """Generate list of preprocessing targets by reading directly from parquet index."""
        if not os.path.exists(self.parquet_index_path):
            logger.warning("Parquet index file not found - cannot generate preprocessing targets")
            return []
        
        if not PANDAS_AVAILABLE:
            logger.error("Pandas not available - cannot read parquet index")
            return []
        
        try:
            df = pd.read_parquet(self.parquet_index_path)
            
            # Filter for completed files
            if 'status_category' in df.columns:
                completed_files = df[df['status_category'] == 'completed']
            elif 'download_status' in df.columns:
                completed_files = df[df['download_status'] == 'completed']
            else:
                logger.warning("No status column found in parquet index")
                return []
            
            if completed_files.empty:
                logger.warning("No completed files found in parquet index")
                return []
            
            # Get file paths using relative_path
            if 'relative_path' in completed_files.columns:
                file_paths = completed_files['relative_path'].tolist()
            else:
                logger.warning("No path column found in parquet index")
                return []
            
            logger.info(f"Found {len(file_paths)} completed files from parquet index")
            
            if stage == 'annual':
                return self._generate_annual_targets(file_paths, year_range)
            elif stage == 'spatial':
                return self._generate_spatial_targets(file_paths, year_range)
            else:
                raise ValueError(f"Unknown stage: {stage}")
                
        except Exception as e:
            logger.error(f"Error reading parquet index: {e}")
            return []

    def _generate_annual_targets(self, files: List[str], year_range: Tuple[int, int] = None) -> List[Dict]:
        """Generate annual processing targets."""
        targets = []
        
        # Parse filenames to extract metadata
        files_df = self._parse_filenames(files)
        
        # Filter by years
        if year_range:
            files_df = files_df[files_df['year'].between(year_range[0], year_range[1])]
        else:
            files_df = files_df[files_df['year'].isin(self.years_to_process)]
        
        if self.data_source == 'MODIS':
            # Filter by grid cells if specified
            if self.grid_cells:
                grid_filter = files_df.apply(lambda row: f"h{row['h']:02d}v{row['v']:02d}" in self.grid_cells, axis=1)
                files_df = files_df[grid_filter]
            
            # Group by year and grid cell
            grouped = files_df.groupby(['year', 'h', 'v'])
            
            for (year, h, v), group in grouped:
                grid_cell = f"h{h:02d}v{v:02d}"
                
                target = {
                    'year': year,
                    'grid_cell': grid_cell,
                    'stage': 'annual',
                    'source_files': group['path'].tolist(),
                    'output_path': f"{self.get_hpc_output_path('annual')}/{year}/{grid_cell}.zarr",
                    'dependencies': [],
                    'metadata': {
                        'source_type': self.data_source.lower(),
                        'total_files': len(group),
                        'data_type': 'glass_annual'
                    }
                }
                targets.append(target)
        else:
            # AVHRR - group by year only
            grouped = files_df.groupby('year')
            
            for year, group in grouped:
                target = {
                    'year': year,
                    'grid_cell': 'global',
                    'stage': 'annual',
                    'source_files': group['path'].tolist(),
                    'output_path': f"{self.get_hpc_output_path('annual')}/{year}.zarr",
                    'dependencies': [],
                    'metadata': {
                        'source_type': self.data_source.lower(),
                        'total_files': len(group),
                        'data_type': 'glass_annual'
                    }
                }
                targets.append(target)
        
        return targets

    def _generate_spatial_targets(self, files: List[str], year_range: Tuple[int, int] = None) -> List[Dict]:
        """Generate spatial processing targets."""
        targets = []
        
        # Check if all required annual files are available
        annual_files = self._get_all_annual_files()
        
        if not annual_files:
            logger.warning("No annual files available for spatial processing")
            return targets
        
        missing_years = set(self.years_to_process) - {f['year'] for f in annual_files}
        if missing_years:
            logger.warning(f"Missing annual files for years: {sorted(missing_years)}")
        
        if self.data_source == 'MODIS':
            # Group annual files by grid cell
            from collections import defaultdict
            files_by_grid = defaultdict(list)
            for f in annual_files:
                files_by_grid[f['grid_cell']].append(f)
            
            # Create spatial target for each grid cell
            for grid_cell, grid_files in files_by_grid.items():
                target = {
                    'grid_cell': grid_cell,
                    'data_type': f'{self.data_source.lower()}_spatial',
                    'stage': 'spatial',
                    'source_files': [f['zarr_path'] for f in grid_files],
                    'output_path': f"{self.get_hpc_output_path('spatial')}/{grid_cell}_timeseries_reprojected.zarr",
                    'dependencies': [f['zarr_path'] for f in grid_files],
                    'metadata': {
                        'source_type': self.data_source.lower(),
                        'data_type': f'{self.data_source.lower()}_spatial',
                        'processing_type': 'reproject_timeseries',
                        'years_available': [f['year'] for f in grid_files],
                        'years_requested': self.years_to_process,
                        'missing_years': sorted(missing_years) if missing_years else []
                    }
                }
                targets.append(target)
        else:
            # AVHRR - single global target
            target = {
                'grid_cell': 'global',
                'data_type': f'{self.data_source.lower()}_spatial',
                'stage': 'spatial',
                'source_files': [f['zarr_path'] for f in annual_files],
                'output_path': f"{self.get_hpc_output_path('spatial')}/{self.data_source.lower()}_timeseries_reprojected.zarr",
                'dependencies': [f['zarr_path'] for f in annual_files],
                'metadata': {
                    'source_type': self.data_source.lower(),
                    'data_type': f'{self.data_source.lower()}_spatial',
                    'processing_type': 'reproject_timeseries',
                    'years_available': [f['year'] for f in annual_files],
                    'years_requested': self.years_to_process,
                    'missing_years': sorted(missing_years) if missing_years else []
                }
            }
            targets.append(target)
        
        return targets

    def _parse_filenames(self, filenames: List[str]) -> pd.DataFrame:
        """Parse filenames to extract metadata for both MODIS and AVHRR."""
        if self.data_source == 'MODIS':
            return self._parse_modis_filenames(filenames)
        else:
            return self._parse_avhrr_filenames(filenames)

    def _parse_modis_filenames(self, filenames: List[str]) -> pd.DataFrame:
        """
        Parse MODIS filenames to extract metadata.
        
        Expected format: GLASS06A01.V01.A2000055.h00v10.2022021.hdf
        """
        result = []
        
        for filename in filenames:
            try:
                clean_filename = self._strip_remote_prefix(filename)
                basename = os.path.basename(clean_filename)
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
                    'path': clean_filename,
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
        """
        result = []
        
        for filename in filenames:
            try:
                clean_filename = self._strip_remote_prefix(filename)
                basename = os.path.basename(clean_filename)
                if not basename.endswith('.hdf'):
                    continue
                    
                # Extract year and day
                year_day_match = basename.split('.')[2]
                if not (year_day_match.startswith('A') and len(year_day_match) == 8):
                    continue
                    
                year = int(year_day_match[1:5])
                day = int(year_day_match[5:8])
                
                result.append({
                    'path': clean_filename,
                    'year': year,
                    'day': day
                })
            except (IndexError, ValueError) as e:
                logger.warning(f"Could not parse filename {filename}: {str(e)}")
                
        return pd.DataFrame(result)

    def _get_all_annual_files(self) -> List[Dict]:
        """
        Return all available annual zarr files in the annual output directory.
        Each dict contains 'year', 'grid_cell', and 'zarr_path'.
        """
        annual_dir = self.get_hpc_output_path('annual')
        if not os.path.exists(annual_dir):
            return []
        
        files = []
        
        if self.data_source == 'MODIS':
            # MODIS files are organized as year/gridcell.zarr
            for year_dir in os.listdir(annual_dir):
                year_path = os.path.join(annual_dir, year_dir)
                if not os.path.isdir(year_path):
                    continue
                
                try:
                    year = int(year_dir)
                except ValueError:
                    continue
                
                for fname in os.listdir(year_path):
                    if fname.endswith('.zarr'):
                        grid_cell = os.path.splitext(fname)[0]
                        files.append({
                            'year': year,
                            'grid_cell': grid_cell,
                            'zarr_path': os.path.join(year_path, fname)
                        })
        else:
            # AVHRR files are organized as year.zarr
            for fname in os.listdir(annual_dir):
                if fname.endswith('.zarr'):
                    try:
                        year = int(os.path.splitext(fname)[0])
                        files.append({
                            'year': year,
                            'grid_cell': 'global',
                            'zarr_path': os.path.join(annual_dir, fname)
                        })
                    except ValueError:
                        continue
        
        return files

    def get_grid_cells(self) -> List[str]:
        """Get list of grid cells to process."""
        if self.data_source == 'MODIS':
            return self.grid_cells or ["h25v06", "h26v06"]  # Default example grid cells
        else:
            return ["global"]

    def get_hpc_output_path(self, stage: str) -> str:
        """Get HPC output path for a given stage."""
        if stage == "annual":
            base_path = os.path.join(self.hpc_root, self.path_prefix, "processed", "stage_1")
        elif stage == "spatial":
            base_path = os.path.join(self.hpc_root, self.path_prefix, "processed", "stage_2")
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        return self._strip_remote_prefix(base_path)

    def process_target(self, target: Dict[str, Any]) -> bool:
        """Process a single preprocessing target."""
        stage = target.get('stage')
        year = target.get('year', None)
        grid_cell = target.get('grid_cell', 'global')
        
        logger.info(f"Processing target: {stage}" + (f" - year {year}" if year is not None else "") + f" - grid_cell {grid_cell}")
        
        try:
            if stage == 'annual':
                return self._process_annual_target(target)
            elif stage == 'spatial':
                return self._process_spatial_target(target)
            else:
                logger.error(f"Unknown stage: {stage}")
                return False
                
        except Exception as e:
            logger.exception(f"Error processing target {stage}/{year}/{grid_cell}: {e}")
            return False

    def _process_annual_target(self, target: Dict[str, Any]) -> bool:
        """Process annual target."""
        year = target['year']
        grid_cell = target.get('grid_cell', 'global')
        source_files = target['source_files']
        output_path = self._strip_remote_prefix(target['output_path'])
        
        if not self.override and os.path.exists(output_path):
            logger.info(f"Skipping annual processing for {year}/{grid_cell}, output already exists: {output_path}")
            return True
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # Resolve source file paths
            resolved_files = [self._resolve_source_file_path(f) for f in source_files]
            
            # Process the files for this year/grid_cell
            success = self._process_file_group_hpc(resolved_files, year, output_path, grid_cell)
            
            if success:
                logger.info(f"Annual processing complete for {year}/{grid_cell}: {output_path}")
            
            return success
            
        except Exception as e:
            logger.exception(f"Error in GLASS annual processing: {e}")
            return False

    def _resolve_source_file_path(self, file_path: str) -> str:
        """Resolve the full path to a source file using path_prefix and relative_path."""
        # If already absolute or already under hpc_root, return as is
        if os.path.isabs(file_path) or (self.hpc_root and file_path.startswith(self.hpc_root)):
            return file_path
        # Use path_prefix for correct subdirectory structure
        return os.path.join(self.hpc_root, self.path_prefix, "raw", file_path)

    def _initialize_dask_client(self):
        """Initialize Dask client for parallel processing using the context manager."""
        dask_params = {
            'threads': self.dask_threads,
            'memory_limit': self.dask_memory_limit,
            'dashboard_port': self.dashboard_port,
            'temp_dir': os.path.join(self.temp_dir, "dask_workspace")
        }
        return DaskClientContextManager(**dask_params)

    def _process_file_group_hpc(self, files: List[str], year: int, output_path: str, grid_cell: str = None) -> bool:
        """
        Process a group of files for a specific year (and grid cell for MODIS) in HPC mode.
        """
        try:
            # Initialize Dask client for parallel processing
            with self._initialize_dask_client() as client:
                if client is None:
                    logger.warning("Failed to initialize Dask client, proceeding without it")
                else:
                    dashboard_link = getattr(client, "dashboard_link", None)
                    if dashboard_link:
                        logger.info(f"Created Dask client for annual processing: {dashboard_link}")
                
                # Sort files chronologically by extracting day of year
                files_with_day = []
                for file_path in files:
                    basename = os.path.basename(file_path)
                    if self.data_source == 'MODIS':
                        # Extract day from MODIS filename
                        year_day_match = basename.split('.')[2]
                        if year_day_match.startswith('A') and len(year_day_match) == 8:
                            day = int(year_day_match[5:8])
                            files_with_day.append((day, file_path))
                    else:
                        # Extract day from AVHRR filename
                        year_day_match = basename.split('.')[2]
                        if year_day_match.startswith('A') and len(year_day_match) == 8:
                            day = int(year_day_match[5:8])
                            files_with_day.append((day, file_path))
                
                # Sort by day of year
                files_with_day.sort(key=lambda x: x[0])
                sorted_files = [f[1] for f in files_with_day]
                
                # Process each file and store in a list for later concatenation
                array_list = []
                days = []
                
                for file_path in sorted_files:
                    if not os.path.exists(file_path):
                        logger.warning(f"File does not exist: {file_path}")
                        continue
                    
                    # Extract day of year for this file
                    basename = os.path.basename(file_path)
                    year_day_match = basename.split('.')[2]
                    day = int(year_day_match[5:8])
                    days.append(day)
                    
                    # Open file with rioxarray and handle different data structures
                    logger.debug(f"Opening GLASS file: {file_path}")
                    
                    if self.data_source == 'MODIS':
                        # MODIS files return a DataArray - access LST directly
                        ds = rxr.open_rasterio(file_path, decode_coords="all", 
                                               chunks={k: self.chunk_size[k] for k in ('band', 'y', 'x') if k in self.chunk_size})
                        # For MODIS, the DataArray itself contains the LST data
                        lst_data = ds
                    else:
                        # AVHRR files return a Dataset - access LST as a data variable
                        ds = rxr.open_rasterio(file_path, decode_coords="all", 
                                               chunks={k: self.chunk_size[k] for k in ('band', 'y', 'x') if k in self.chunk_size})
                        # For AVHRR, LST is a data variable within the dataset
                        if hasattr(ds, 'data_vars') and self.VARIABLE_NAME in ds.data_vars:
                            lst_data = ds[self.VARIABLE_NAME]
                        elif hasattr(ds, self.VARIABLE_NAME):
                            lst_data = getattr(ds, self.VARIABLE_NAME)
                        else:
                            logger.error(f"Could not find {self.VARIABLE_NAME} variable in {file_path}")
                            continue
                    
                    array_list.append(lst_data)
                
                if not array_list:
                    logger.error(f"No valid files found for {year}/{grid_cell}")
                    return False
                
                # Concatenate all arrays along time dimension
                combined_data = xr.concat(array_list, dim='day')
                
                # Rename day dimension to time
                combined_data = combined_data.rename({"day": "time"})
                
                # Convert day-of-year to proper datetime coordinates
                combined_data = combined_data.assign_coords({
                    'time': pd.to_datetime([f"{year}{day:03d}" for day in days], format="%Y%j"),
                })
                
                # Convert to dataset with proper variable name
                combined_data = combined_data.to_dataset(name=self.VARIABLE_NAME)
                
                # Calculate annual and monthly statistics
                logger.info("Calculating statistics with Dask")
                annual_stats, monthly_stats = self._calculate_statistics(combined_data)
                
                # Create zarr file with optimized settings
                success = self._create_annual_zarr_hpc(annual_stats, monthly_stats, output_path)
                
                return success
            
        except Exception as e:
            logger.error(f"Error processing file group for {year}/{grid_cell}: {e}")
            return False

    def _calculate_statistics(self, data: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
        """Calculate annual and monthly statistics from daily data using Dask."""
        # Create a mask for invalid values - using dask arrays
        mask = da.logical_and(data[self.VARIABLE_NAME] >= 20000, 
                          data[self.VARIABLE_NAME] <= 35000)
        masked = data.where(mask)
        
        # Set better chunk sizes for time-based resampling
        rechunked = masked.chunk({'time': -1, 'x': self.chunk_size.get('x', 500), 
                             'y': self.chunk_size.get('y', 500)})
        # Define output attributes for all statistics
        attrs = {
            "_FillValue": 0,
            "scale_factor": 0.01,
            "add_offset": 0.0
        }

        # Helper to format output arrays: fill NaNs, set attrs, cast to uint16
        def format_output(xarray):
            return xarray.fillna(0).assign_attrs(attrs).astype(np.uint16, casting="unsafe")
        
        # Helper to format output arrays: fill NaNs, set attrs, cast to uint16
        def format_output_count(xarray):
            return xarray.fillna(0).astype(np.uint16, casting="unsafe")
        
        # Calculate annual statistics with comments and format_output where relevant
        annual_stats = xr.Dataset({
            # Mean LST
            "mean": format_output(rechunked[self.VARIABLE_NAME].resample(time="1YE").mean()),
            # Median LST
            "median": format_output(rechunked[self.VARIABLE_NAME].resample(time="1YE").median()),
            # Standard deviation
            "std": format_output(rechunked[self.VARIABLE_NAME].resample(time="1YE").std()),
            # Maximum value
            "max": format_output(rechunked[self.VARIABLE_NAME].resample(time="1YE").max()),
            # Minimum value
            "min": format_output(rechunked[self.VARIABLE_NAME].resample(time="1YE").min()),
            # 5-day rolling max
            "rollmax3": format_output(rechunked[self.VARIABLE_NAME].rolling(time=3, center=True).mean().resample(time="1YE").max()),
            # 5-day rolling min
            "rollmin3": format_output(rechunked[self.VARIABLE_NAME].rolling(time=3, center=True).mean().resample(time="1YE").min()),
            # Count of days > 30°C (303.15K)
            "gt30C": format_output_count((rechunked[self.VARIABLE_NAME] > 30315).resample(time="1YE").sum()),
            # Count of days < 0°C (273.15K)
            "lt0C": format_output_count((rechunked[self.VARIABLE_NAME] < 27315).resample(time="1YE").sum()),
            # Count of valid values
            "valid_count": format_output_count(mask.resample(time="1YE").sum())
        })
        
        # Calculate monthly statistics
        monthly_stats = xr.Dataset({
            "mean": format_output(data[self.VARIABLE_NAME].resample(time="1ME").mean()),
            "median": format_output(data[self.VARIABLE_NAME].resample(time="1ME").median()),
            "std": format_output(data[self.VARIABLE_NAME].resample(time="1ME").std()),
            "valid_count": mask.resample(time="1ME").sum().fillna(0).astype(np.uint16, casting="unsafe")
        })
        
        # Rechunk for optimal zarr storage
        chunk_dict = {"time": 1}
        if 'x' in self.chunk_size:
            chunk_dict['x'] = self.chunk_size['x']
        if 'y' in self.chunk_size:
            chunk_dict['y'] = self.chunk_size['y']
        
        annual_stats = annual_stats.chunk(chunk_dict)
        monthly_stats = monthly_stats.chunk(chunk_dict)
        
        return annual_stats, monthly_stats

    def _create_annual_zarr_hpc(self, annual_stats: xr.Dataset, monthly_stats: xr.Dataset, output_path: str) -> bool:
        """Create annual zarr file for HPC mode with Dask optimization."""
        try:
            # Define chunking strategy optimized for zarr storage
            chunks = {"x": 1000, "y": 1000, "time": 1}
            
            # Rechunk the datasets for optimal zarr storage
            logger.info(f"Rechunking datasets for zarr storage: {chunks}")
            annual_stats = annual_stats.chunk(chunks)
            monthly_stats = monthly_stats.chunk(chunks)
            
            # Set up compression for Zarr output
            compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=2)
            encoding = {var: {'compressor': compressor} for var in annual_stats.data_vars}
            
            # Create separate paths for annual and monthly data
            annual_output_path = output_path
            monthly_output_path = output_path.replace('.zarr', '_monthly.zarr')
            
            logger.info(f"Writing GLASS annual dataset to zarr: {annual_output_path}")
            annual_stats.to_zarr(
                annual_output_path, 
                mode="w", 
                encoding=encoding, 
                consolidated=True,
                compute=False
            ).compute()
            
            logger.info(f"Writing GLASS monthly dataset to zarr: {monthly_output_path}")
            monthly_encoding = {var: {'compressor': compressor} for var in monthly_stats.data_vars}
            monthly_stats.to_zarr(
                monthly_output_path, 
                mode="w", 
                encoding=monthly_encoding, 
                consolidated=True,
                compute=False
            ).compute()
            
            logger.info(f"Created annual zarr files at {annual_output_path} and {monthly_output_path}")
            return True
            
        except Exception as e:
            logger.exception(f"Error creating zarr file: {e}")
            return False

    def _process_spatial_target(self, target: Dict[str, Any]) -> bool:
        """Process spatial stage with optimized memory management."""
        logger.info("Starting spatial stage processing")
        
        output_path = self._strip_remote_prefix(target['output_path'])
        source_files = target.get('source_files', [])
        grid_cell = target.get('grid_cell', 'global')
        
        if not self.override and os.path.exists(output_path):
            logger.info(f"Skipping spatial processing, output already exists: {output_path}")
            return True
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            with self._initialize_dask_client() as client:
                if client is None:
                    logger.error("Failed to initialize Dask client")
                    return False
                    
                dashboard_link = getattr(client, "dashboard_link", None)
                if dashboard_link:
                    logger.info(f"Created Dask client for spatial processing: {dashboard_link}")
                
                # Configure Dask for large array operations
                import dask
                with dask.config.set({
                    'array.slicing.split_large_chunks': True,
                    'array.chunk-size': '512MB',
                    'optimization.fuse.active': False,
                }):
                    # Extract years from source files
                    years = []
                    for f in source_files:
                        try:
                            if self.data_source == 'MODIS':
                                # For MODIS: extract year from path like .../2000/h25v06.zarr
                                year = int(os.path.basename(os.path.dirname(f)))
                            else:
                                # For AVHRR: extract year from filename like 2000.zarr
                                year = int(os.path.splitext(os.path.basename(f))[0])
                            years.append(year)
                        except (ValueError, IndexError):
                            continue
                    
                    if not years:
                        logger.error("No valid years found in source files")
                        return False
                    
                    # Sort years for consistent processing
                    years.sort()
                    logger.info(f"Processing {len(years)} years for grid cell {grid_cell}: {years}")
                    
                    # Get the target geobox for reprojection
                    try:
                        target_geobox = get_or_create_geobox(self.hpc_root)
                        logger.info(f"Using target geobox for reprojection: {target_geobox.shape}")
                    except Exception as e:
                        logger.error(f"Failed to get target geobox: {e}")
                        return False
                    
                    # Load, reproject, and combine all years
                    combined_ds = self._load_and_reproject_years_mfdataset(source_files, years, target_geobox)
                    
                    # Export to zarr
                    self._export_to_zarr(combined_ds, Path(output_path))
                    
                    logger.info("Spatial stage processing completed successfully")
                    return True
                    
        except Exception as e:
            logger.exception(f"Error in GLASS spatial processing: {e}")
            return False

    def _load_and_reproject_years_mfdataset(self, source_files: List[str], years: List[int], target_geobox) -> xr.Dataset:
        """Load and reproject years using open_mfdataset for maximum efficiency."""
        logger.info(f"Loading {len(years)} years using open_mfdataset with preprocess (including reprojection)")
        
        # Verify all files exist
        missing_files = [f for f in source_files if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"Missing zarr files: {missing_files}")
        
        # Get CRS and transform from the first dataset
        first_ds = xr.open_zarr(source_files[0], chunks=True)
        crs = getattr(first_ds.odc, "crs", None)
        transform = getattr(first_ds.odc, "transform", None)
        first_ds.close()

        preprocess_func = partial(_preprocess_glass, crs=crs, transform=transform, geobox=target_geobox)

        # The MODIS data requires an intermediate processing step. Data from all grid cells within a year should be combined first. Then these should be concatenated along the year axis. AVHRR is already global and does not require this step
        ds = xr.open_mfdataset(
            source_files,
            engine='zarr',
            chunks={"time": 1, "x": 2048, "y": 2048},
            parallel=True,
            preprocess=preprocess_func
        )

        logger.info(f"Created lazy reprojected dataset with shape: {ds.dims}")
        return ds


    def _export_to_zarr(self, ds: xr.Dataset, output_path: Path) -> None:
        """Export dataset to zarr file with optimized settings."""
        logger.info(f"Exporting to zarr: {output_path}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up compression and chunking for Zarr output
        compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=2)
        encoding = {
            var: {
                "chunks": {"time": 1, "x": 512, "y": 512}, 
                "compressor": compressor,
                "dtype": "uint16"
            } 
            for var in ds.data_vars
        }
        
        # Export to zarr
        logger.info("Starting zarr write computation")
        ds.to_zarr(
            output_path, 
            mode="w",
            consolidated=True,
            encoding=encoding,
            compute=False
        ).compute()
        
        logger.info(f"Successfully exported to {output_path}")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GlassPreprocessor":
        """Create a GlassPreprocessor from a configuration dictionary."""
        return cls(**config)

def _preprocess_glass(ds, crs=None, transform=None, drop_vars=None, geobox=None):
    """Preprocess function for GLASS data similar to EOG preprocessing."""
    # Set CRS if needed
    if crs and getattr(ds.rio, "crs", None) is None:
        ds = ds.rio.write_crs(crs)
    if transform and getattr(ds.rio, "transform", None) is None:
        ds = ds.rio.write_transform(transform)
    if drop_vars:
        ds = ds.drop_vars(drop_vars, errors="ignore")
    # Reproject if geobox is provided
    if geobox is not None:
        ds = xr_reproject(ds, geobox, resampling="nearest")
    return ds
import os
import tempfile
import logging
import re  # Added for _strip_remote_prefix regex
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dask.distributed import Client, LocalCluster
from functools import partial
import xarray as xr
import numpy as np
import dask.array as da
from zarr.codecs import BloscCodec
import rioxarray as rxr  # Added for rasterio-based xarray reading

# Add pandas for reading parquet files directly
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Add PyArrow for efficient parquet operations
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    pa = None
    pq = None

from gnt.data.preprocess.sources.base import AbstractPreprocessor
from gnt.data.preprocess.common.spatial import SpatialProcessor, create_zarr_encoding
from gnt.data.common.dask.client import DaskClientContextManager

from odc.geo import CRS
from odc.geo.xr import ODCExtensionDa, assign_crs, xr_reproject

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
        if self.stage not in ['annual', 'spatial', 'tabular']:
            raise ValueError(f"Unsupported stage: {self.stage}. Use 'annual', 'spatial', or 'tabular'.")
            
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
        
        # Tabular processing configuration - optimized defaults
        self.tabular_batch_size = kwargs.get('tabular_batch_size', 32)  # Increased default batch size
        
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
            elif stage == 'tabular':
                return self._generate_tabular_targets(file_paths, year_range)
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
            # For MODIS: Create single target combining all grid cells and years
            target = {
                'grid_cell': 'all_cells',
                'data_type': f'{self.data_source.lower()}_spatial',
                'stage': 'spatial',
                'source_files': [f['zarr_path'] for f in annual_files],
                'output_path': f"{self.get_hpc_output_path('spatial')}/{self.data_source.lower()}_timeseries_reprojected.zarr",
                'dependencies': [f['zarr_path'] for f in annual_files],
                'metadata': {
                    'source_type': self.data_source.lower(),
                    'data_type': f'{self.data_source.lower()}_spatial',
                    'processing_type': 'reproject_timeseries_combined',
                    'years_available': [f['year'] for f in annual_files],
                    'years_requested': self.years_to_process,
                    'missing_years': sorted(missing_years) if missing_years else [],
                    'grid_cells': list(set(f['grid_cell'] for f in annual_files))
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
    
    def _generate_tabular_targets(self, files: List[str], year_range: Tuple[int, int] = None) -> List[Dict]:
        """Generate tabular processing targets."""
        targets = []
        
        # Check if spatial zarr files are available
        spatial_files = self._get_all_spatial_files()
        
        if not spatial_files:
            logger.warning("No spatial files available for tabular processing")
            return targets
        
        # Create target for each spatial file
        for spatial_file in spatial_files:
            target = {
                'grid_cell': spatial_file['grid_cell'],
                'data_type': f'{self.data_source.lower()}_tabular',
                'stage': 'tabular',
                'source_files': [spatial_file['zarr_path']],
                'output_path': f"{self.get_hpc_output_path('tabular')}/{self.data_source.lower()}_tabular.parquet",
                'dependencies': [spatial_file['zarr_path']],
                'metadata': {
                    'source_type': self.data_source.lower(),
                    'data_type': f'{self.data_source.lower()}_tabular',
                    'processing_type': 'zarr_to_parquet',
                    'source_file': spatial_file['zarr_path']
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
                    if fname.endswith('.zarr') and not fname.endswith('_monthly.zarr'):
                        grid_cell = os.path.splitext(fname)[0]
                        files.append({
                            'year': year,
                            'grid_cell': grid_cell,
                            'zarr_path': os.path.join(year_path, fname)
                        })
        else:
            # AVHRR files are organized as year.zarr
            for fname in os.listdir(annual_dir):
                if fname.endswith('.zarr') and not fname.endswith('_monthly.zarr'):
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

    def _get_all_spatial_files(self) -> List[Dict]:
        """
        Return all available spatial zarr files in the spatial output directory.
        Each dict contains 'grid_cell' and 'zarr_path'.
        """
        spatial_dir = self.get_hpc_output_path('spatial')
        if not os.path.exists(spatial_dir):
            return []
        
        files = []
        
        for fname in os.listdir(spatial_dir):
            if fname.endswith('.zarr'):
                if self.data_source == 'MODIS' and 'timeseries_reprojected' in fname:
                    grid_cell = fname.replace('_timeseries_reprojected.zarr', '')
                    files.append({
                        'grid_cell': grid_cell,
                        'zarr_path': os.path.join(spatial_dir, fname)
                    })
                elif self.data_source == 'AVHRR' and 'timeseries_reprojected' in fname:
                    files.append({
                        'grid_cell': 'global',
                        'zarr_path': os.path.join(spatial_dir, fname)
                    })
        
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
        elif stage == "tabular":
            base_path = os.path.join(self.hpc_root, self.path_prefix, "processed", "stage_3")
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
            elif stage == 'tabular':
                return self._process_tabular_target(target)
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
            compressor = BloscCodec(cname="zstd", clevel=3, shuffle=2)
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
        """Process spatial stage with chunked aggregation and reprojection."""
        logger.info("Starting spatial stage processing with chunked approach")
        
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
                    'distributed.comm.compression': 'lz4',
                }):
                    
                    # Get the target geobox for reprojection
                    try:
                        target_geobox = get_or_create_geobox(self.hpc_root)
                        logger.info(f"Using target geobox for reprojection: {target_geobox.shape}")
                    except Exception as e:
                        logger.error(f"Failed to get target geobox: {e}")
                        return False
                    
                    # Step 1: Create empty zarr file with target dimensions
                    if not self._create_empty_target_zarr(output_path, target_geobox, source_files):
                        return False
                    
                    # Step 2: Process by year with aggregation and reprojection
                    success = self._process_years_chunked(source_files, output_path, target_geobox)
                    
                    if success:
                        logger.info("Spatial stage processing completed successfully")
                    
                    return success
                        
        except Exception as e:
            logger.exception(f"Error in GLASS spatial processing: {e}")
            return False

    def _create_empty_target_zarr(self, output_path: str, target_geobox, source_files: List[str]) -> bool:
        """Create empty zarr file with target dimensions and metadata."""
        try:
            logger.info("Creating empty target zarr file")
            
            # Get sample dataset to determine variables and time dimension
            sample_ds = xr.open_zarr(source_files[0], mask_and_scale=False, chunks='auto', consolidated=False)
            variables = list(sample_ds.data_vars.keys())
            sample_attrs = sample_ds.attrs.copy()
            
            # Use years from year range setting to determine time dimension
            years = sorted(self.years_to_process)
            
            logger.info(f"Creating zarr for {len(years)} years: {min(years)}-{max(years)}")
            
            # Create time coordinates
            time_coords = pd.to_datetime([f"{year}-12-31" for year in years])
            
            # Create empty dataset with target geobox dimensions
            ny, nx = target_geobox.shape
            lat_coords = target_geobox.coords['latitude'].values.round(5)
            lon_coords = target_geobox.coords['longitude'].values.round(5)
            
            # Create data variables with fill values and band dimension
            data_vars = {}
            
            default_attrs = {"_FillValue": 0}
            packaging_attrs = {
                "scale_factor": 0.01,
                "add_offset": 0.0
            }
            
            for var in variables:
                var_attrs = sample_ds[var].attrs.copy() | default_attrs
                if "float" in str(sample_ds[var].dtype):
                    var_attrs |= packaging_attrs
                    
                data_vars[var] = xr.DataArray(
                    da.zeros((len(years), 1, ny, nx), dtype=np.uint16, chunks=(1, 1, 512, 512)),
                    dims=['time', 'band', 'latitude', 'longitude'],
                    coords={
                        'time': time_coords,
                        'band': [1],
                        'latitude': lat_coords,
                        'longitude': lon_coords
                    },
                    attrs=var_attrs
                )
            sample_ds.close()
            # Create empty dataset and copy global attributes
            empty_ds = xr.Dataset(data_vars, attrs=sample_attrs)
            
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
            
            # Write empty zarr structure (compute=False)
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

    def _process_years_chunked(self, source_files: List[str], output_path: str, target_geobox) -> bool:
        """Process files by year with chunked aggregation and reprojection."""
        try:
            # Group files by year
            files_by_year = self._group_files_by_year(source_files)
            
            # Import GeoboxTiles for chunked processing
            from odc.geo import GeoboxTiles
            
            # Create tiles for processing 
            tile_size = 2048
            tiles = GeoboxTiles(target_geobox, (tile_size, tile_size))
            
            # Process each year
            for year in sorted(files_by_year.keys()):
                year_files = files_by_year[year]
                logger.info(f"Processing year {year} with {len(year_files)} files")
                
                # Step 2a: Aggregate year files if multiple exist
                if len(year_files) > 1:
                    annual_temp_path = f'{output_path.split("stage_2")[0]}stage_1/{year}/temp_combined.tzarr'
                    if not self._aggregate_year_files(year_files, annual_temp_path, year):
                        logger.error(f"Failed to aggregate files for year {year}")
                        return False
                    year_source = annual_temp_path
                else:
                    year_source = year_files[0]
                
                # Step 2b: Process tiles for this year
                if not self._process_year_tiles(year_source, output_path, target_geobox, tiles, year, tile_size):
                    logger.error(f"Failed to process tiles for year {year}")
                    return False
            
            return True
            
        except Exception as e:
            logger.exception(f"Error in chunked year processing: {e}")
            return False

    def _group_files_by_year(self, source_files: List[str]) -> Dict[int, List[str]]:
        """Group source files by year."""
        files_by_year = {}
        
        for file_path in source_files:
            if self.data_source == 'MODIS':
                # Extract year from path like: .../2020/h25v06.zarr
                year_match = re.search(r'/(\d{4})/', file_path)
                if year_match:
                    year = int(year_match.group(1))
                    if year not in files_by_year:
                        files_by_year[year] = []
                    files_by_year[year].append(file_path)
            else:
                # Extract year from filename like: 2020.zarr
                basename = os.path.basename(file_path)
                year_match = re.search(r'(\d{4})\.zarr', basename)
                if year_match:
                    year = int(year_match.group(1))
                    if year not in files_by_year:
                        files_by_year[year] = []
                    files_by_year[year].append(file_path)
        
        return files_by_year

    def _aggregate_year_files(self, year_files: List[str], temp_output_path: str, year: int) -> bool:
        """Aggregate multiple files for a year into a temporary zarr."""
        try:
            
            if os.path.exists(temp_output_path):
                logger.info(f"Temporary output path already exists")
                return True
                
            else: 
                logger.info(f"Aggregating {len(year_files)} files for year {year}")
                
                # Load all files to determine combined spatial extent
                datasets = []
                for file_path in year_files:
                    ds = xr.open_zarr(file_path, decode_coords='all', chunks='auto')
                    ds.coords['x'] = ds.coords['x'].astype('int')
                    ds.coords['y'] = ds.coords['y'].astype('int')
                    datasets.append(ds)
                
                combined = xr.combine_by_coords(datasets, combine_attrs='drop_conflicts', join='outer')
                
                # Create coordinate arrays
                x_coords = combined.coords["x"]; y_coords = combined.coords["y"]
                nx = len(x_coords); ny = len(y_coords)
                
                # Create empty combined dataset
                variables = list(ds.data_vars.keys())
                coordinates = list(ds.coords.keys())
                data_vars = {}
                
                default_attrs = {"_FillValue": 0}
                packaging_attrs = {
                        "scale_factor": 0.01,
                        "add_offset": 0.0
                }
                
                for var in variables:
                    var_attrs = combined[var].attrs.copy() | default_attrs
                    if "float" in str(datasets[0][var].dtype):
                        var_attrs |= packaging_attrs
                    
                    data_vars[var] = xr.DataArray(
                        da.zeros((1, 1, ny, nx), dtype=np.uint16, chunks=(1, 1, 300, 300)),
                        dims=['band', 'time', 'y', 'x'],
                        coords={
                            'band': [1],
                            'time': [pd.to_datetime(f"{year}-12-31")],
                            'y': y_coords,
                            'x': x_coords
                        },
                        attrs=var_attrs
                    )
                
                combined_ds = xr.Dataset(data_vars)
                
                if self.data_source == 'MODIS':
                    crs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
                else:
                    crs = 4326
                
                combined_ds = combined_ds.rio.write_crs(crs)
                
                # Set up encoding
                compressor = BloscCodec(cname="zstd", clevel=3, shuffle='bitshuffle', blocksize=0)
                encoding = {
                    var: {
                        "chunks": (1, 1, 300, 300), 
                        "compressors": (compressor,),
                        "dtype": "uint16"
                    } 
                    for var in variables
                } | {
                    coord: {
                        "compressors": (compressor,)
                        } 
                    for coord in coordinates
                    }
                
                # Write empty structure
                combined_ds.to_zarr(temp_output_path, mode="w", encoding=encoding, zarr_format=3, consolidated=False, compute=False)
                
                # Now fill regions iteratively
                for i, ds in enumerate(datasets):
                    
                    try:
                        
                        logger.debug(f"Writing region {i+1}/{len(datasets)} for year {year}")
                        
                        # Write CRS
                        ds_clean = ds.rio.write_crs(crs)
                        # Drop spatial ref
                        ds_clean = ds_clean.drop_vars(['spatial_ref']).drop_attrs()
                        # Rechunk
                        ds_clean = ds_clean.chunk({'band': 1, 'time': 1, 'y': 300, 'x': 300})
                        # # Flip y axis
                        ds_clean = ds_clean.isel(y=slice(None,None,-1))
                        
                        if ds_clean.sizes['x'] == 0 or ds_clean.sizes['y'] == 0:
                                # No data in this tile, skip
                                continue
                        
                        # slice_x, slice_y = get_indices(ds_clean, x_coords, y_coords)
                        
                        ds_clean.\
                            to_zarr(
                                temp_output_path,
                                region='auto',
                                # region={
                                #     'band': 'auto',
                                #     'time': 'auto',
                                #     'y': slice_y,
                                #     'x': slice_x
                                # },
                                align_chunks=True
                            )
                            
                    except Exception as e:
                        logger.warning(f"Error processing region {i+1}/{len(datasets)} for year {year}: {e}")
                        continue
            
            # Close datasets
            combined_ds.close()
            for ds in datasets:
                ds.close()
            
            logger.info(f"Successfully aggregated year {year} to: {temp_output_path}")
            return True
        
        except Exception as e:
            logger.exception(f"Error aggregating year files for {year}: {e}")
            return False

    def _process_year_tiles(self, year_source: str, output_path: str, target_geobox, tiles, year: int, tile_size: int) -> bool:
        """Process tiles for a specific year with reprojection."""
        try:            
            # Open year source
            year_ds = xr.open_zarr(year_source, consolidated = False, decode_coords='all')
            
            if self.data_source == "AVHRR":
                year_ds = year_ds.rio.write_crs(4326)
                year_ds = year_ds.sel(y=slice(None, None, -1))
            elif self.data_source == "MODIS":
                year_ds = year_ds.rio.write_crs("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs")
            
            if year_ds.rio.crs is None:
                try:
                    year_ds = year_ds.rio.write_crs(year_ds.spatial_ref.crs_wkt)
                except Exception as e:
                    logger.warning(f"Error setting crs on dataset: {e}")
            
            # Process each tile
            for ix in range(tiles.shape[0]):
                for iy in range(tiles.shape[1]):

                    try:
                        logger.info(f"Reprojecting tile [{ix}, {iy}] for year {year} to: {year_source}")
                        
                        # Get the tile index
                        tile_geobox = tiles[ix, iy]
                        
                        def extract_slice(ds, tile_geobox):
                            """Extract slice for the given tile using native xarray slicing."""
                            # Calculate the slice bounds in the source dataset
                            
                            bbox = tile_geobox.footprint(year_ds.rio.crs).boundingbox
                        
                            return ds.sel(
                                y = slice(bbox.bottom, bbox.top),
                                x = slice(bbox.left, bbox.right)
                                )
                        
                        # Clip source data to tile bounds
                        clipped_ds = extract_slice(year_ds, tile_geobox).compute()
                        
                        if clipped_ds.sizes['x'] == 0 or clipped_ds.sizes['y'] == 0:
                            # No data in this tile, skip
                            continue
                        
                        # Reproject to target geobox for this tile
                        reprojected_ds = xr_reproject(clipped_ds, tile_geobox)
                        
                        # Remove attributes and spatial reference
                        reprojected_ds = reprojected_ds.drop_vars(['spatial_ref']).drop_attrs()
                        
                        # Transform coordinates
                        reprojected_ds.coords['longitude'] = reprojected_ds.coords['longitude'].round(5)
                        reprojected_ds.coords['latitude'] = reprojected_ds.coords['latitude'].round(5)
                        
                        # Rechunk
                        reprojected_ds = reprojected_ds.chunk({'band': 1, 'time': 1, 'latitude': 512, 'longitude': 512})
                        
                        # Write to zarr region
                        reprojected_ds.to_zarr(
                            output_path,
                            region='auto',
                            align_chunks=True,
                            zarr_format=3,
                            consolidated=False
                        )
                        
                        reprojected_ds.close()
                        
                        
                    except Exception as e:
                        logger.warning(f"Error processing tile [{ix}, {iy}] for year {year}: {e}")
                        continue
            
            year_ds.close()
            logger.info(f"Completed processing tiles for year {year}")
            return True
            
        except Exception as e:
            logger.exception(f"Error processing tiles for year {year}: {e}")
            return False
        
        
    def _process_tabular_target(self, target: Dict[str, Any]) -> bool:
        return super()._process_tabular_target(target)

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

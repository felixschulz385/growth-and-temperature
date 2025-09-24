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
import zarr
import re
import dask.array as da
from odc.geo import CRS
from odc.geo.xr import ODCExtensionDa, assign_crs, xr_reproject
from gnt.data.common.geobox import get_or_create_geobox
from functools import partial
from zarr.codecs import BloscCodec

from gnt.data.preprocess.sources.base import AbstractPreprocessor
from gnt.data.preprocess.common.spatial import SpatialProcessor
from gnt.data.download.sources.eog import EOGDataSource

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

logger = logging.getLogger(__name__)

class EOGPreprocessor(AbstractPreprocessor):
    """
    HPC-mode preprocessor for Earth Observation Group (EOG) nighttime lights data.
    
    This class handles the preprocessing of both DMSP-OLS and VIIRS nighttime lights data.
    The preprocessing involves:
    1. Processing raw data files into annual composites
    2. Saving the results as zarr files for efficient access
    
    The preprocessing is organized in two stages:
    - Stage 1: Create annual zarr files from raw data
    - Stage 2: Reproject the data to a unified grid for analysis
    
    Requires HPC mode configuration with parquet index.
    """
    
    # Variable name for the data
    DMSP_VARIABLE_NAME = "avg_vis"
    VIIRS_VARIABLE_NAME = "DNB_BRDF_Corrected_NTL"
    
    def __init__(self, **kwargs):
        """
        Initialize the EOG preprocessor in HPC mode.
        
        Args:
            **kwargs: Configuration parameters including:
                stage (str): Processing stage ("annual", "spatial")
                year (int, optional): Specific year to process
                year_range (list, optional): [start_year, end_year] range to process
                grid_cell (str, optional): Grid cell to process (for spatial stage)
                
                # Data source parameters
                base_url (str): Base URL for the specific EOG data source to use
                data_path (str): Data path where source data is stored
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
        
        self.grid_cell = kwargs.get('grid_cell')
        
        # Settings
        self.override = kwargs.get('override', False)
        
        # Get data source parameters - handle both data_path and output_path
        base_url = kwargs.get('base_url')
        data_path = kwargs.get('data_path') or kwargs.get('output_path')
        file_extensions = kwargs.get('file_extensions')
        
        # Validate required parameters
        if not base_url:
            raise ValueError("base_url parameter must be specified")
        if not data_path:
            raise ValueError("data_path or output_path parameter must be specified")
        
        # HPC mode parameters
        hpc_target = kwargs.get('hpc_target')
        self.hpc_root = self._strip_remote_prefix(hpc_target)
        if not self.hpc_root:
            raise ValueError("hpc_target is required for HPC mode")
        self.index_dir = os.path.join(self.hpc_root, "hpc_data_index")
        
        # Store the data path for later use
        self.data_path = data_path
        
        # Dask configuration
        self.dask_threads = kwargs.get('dask_threads', None)
        self.dask_memory_limit = kwargs.get('dask_memory_limit', None)
        
        # Tabular processing configuration - optimized defaults
        self.tabular_batch_size = kwargs.get('tabular_batch_size', 32)  # Increased default batch size
        
        # Create the download data source
        self.data_source = EOGDataSource(
            base_url=base_url,
            file_extensions=file_extensions,
            output_path=data_path
        )
            
        # Derive paths for input/output
        self.input_path = self._strip_remote_prefix(self.data_source.data_path)
        
        # Initialize parquet index path
        self._init_parquet_index_path()
        
        # Derive source_type from data source properties
        self._derive_source_type()
        
        # Setup temporary directory
        self.temp_dir = kwargs.get('temp_dir')
        if not self.temp_dir:
            self.temp_dir = tempfile.mkdtemp(prefix=f"eog_{self.source_type}_processor_")
        else:
            os.makedirs(self.temp_dir, exist_ok=True)
            
        logger.info(f"Initialized EOGPreprocessor for {self.source_type}")
        logger.info(f"HPC root: {self.hpc_root}")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Years to process: {len(self.years_to_process)}")
        if self.grid_cell:
            logger.info(f"Grid cell: {self.grid_cell}")

    def _strip_remote_prefix(self, path):
        """Remove scp/ssh prefix like user@host: from paths."""
        if isinstance(path, str):
            return re.sub(r"^[^@]+@[^:]+:", "", path)
        return path

    def _init_parquet_index_path(self):
        """Initialize parquet index path based on data source."""
        safe_data_path = self.data_path.replace("/", "_").replace("\\", "_")
        os.makedirs(self.index_dir, exist_ok=True)
        self.parquet_index_path = os.path.join(
            self.index_dir,
            f"parquet_{safe_data_path}.parquet"
        )
        logger.debug(f"Parquet index path: {self.parquet_index_path}")
    
    def _derive_source_type(self):
        """Derive the source type from the data source's base URL and output path."""
        base_url = self.data_source.base_url
        output_path = self.data_source.data_path
        
        if "dmsp" in output_path.lower():
            self.source_type = "dmsp"
        elif "annual" in output_path.lower() or "stable_lights" in output_path.lower():
            self.source_type = "viirs_annual"
        elif "dvnl" in output_path.lower():
            self.source_type = "viirs_dvnl"
        elif "dmsp" in base_url.lower():
            self.source_type = "dmsp"
        elif "annual" in base_url.lower():
            self.source_type = "viirs_annual"
        elif "dvnl" in base_url.lower() or "viirs_products" in base_url.lower():
            self.source_type = "viirs_dvnl"
        else:
            if hasattr(self.data_source, 'file_extensions') and self.data_source.file_extensions:
                if any("stable_lights" in ext for ext in self.data_source.file_extensions):
                    self.source_type = "dmsp"
                elif any("median_masked" in ext for ext in self.data_source.file_extensions):
                    self.source_type = "viirs_annual"
                else:
                    self.source_type = "viirs_dvnl"
            else:
                logger.warning("Could not determine source_type from data source. Defaulting to viirs_dvnl")
                self.source_type = "viirs_dvnl"
        
        logger.info(f"Determined source type: {self.source_type}")
    
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
        
        # Group files by year
        files_by_year = {}
        for file_path in files:
            clean_file_path = self._strip_remote_prefix(file_path)
            year = self._extract_year_from_path(clean_file_path)
            if year and (not year_range or year_range[0] <= year <= year_range[1]):
                if year not in files_by_year:
                    files_by_year[year] = []
                files_by_year[year].append(clean_file_path)
        
        # Create targets for each year
        for year, year_files in files_by_year.items():
            selected_file = self._select_best_file_for_year(year_files)
            
            target = {
                'year': year,
                'stage': 'annual',
                'source_files': [selected_file],
                'output_path': f"{self.get_hpc_output_path('annual')}/{year}.zarr",
                'dependencies': [],
                'metadata': {
                    'source_type': self.source_type,
                    'total_candidates': len(year_files),
                    'data_type': 'eog_annual'
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
        
        # Create single spatial target for the entire time series
        target = {
            'data_type': f'{self.source_type}_spatial',
            'stage': 'spatial',
            'source_files': [f['zarr_path'] for f in annual_files],
            'output_path': f"{self.get_hpc_output_path('spatial')}/{self.source_type}_timeseries_reprojected.zarr",
            'dependencies': [f['zarr_path'] for f in annual_files],
            'metadata': {
                'source_type': self.source_type,
                'data_type': f'{self.source_type}_spatial',
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
                'grid_cell': spatial_file.get('grid_cell', 'global'),
                'data_type': f'{self.source_type}_tabular',
                'stage': 'tabular',
                'source_files': [spatial_file['zarr_path']],
                'output_path': f"{self.get_hpc_output_path('tabular')}/{self.source_type}_tabular.parquet",
                'dependencies': [spatial_file['zarr_path']],
                'metadata': {
                    'source_type': self.source_type,
                    'data_type': f'{self.source_type}_tabular',
                    'processing_type': 'zarr_to_parquet',
                    'source_file': spatial_file['zarr_path']
                }
            }
            targets.append(target)
        
        return targets

    def _process_tabular_target(self, target: Dict[str, Any]) -> bool:
        """Process tabular stage using the common implementation."""
        # Use the base class implementation with enhanced tabularization
        return super()._process_tabular_target(target)

    def _get_all_annual_files(self) -> List[Dict]:
        """
        Return all available annual zarr files in the annual output directory.
        Each dict contains 'year' and 'zarr_path'.
        """
        annual_dir = self.get_hpc_output_path('annual')
        if not os.path.exists(annual_dir):
            return []
        
        files = []
        
        for fname in os.listdir(annual_dir):
            if fname.endswith('.zarr'):
                try:
                    year = int(os.path.splitext(fname)[0])
                    files.append({
                        'year': year,
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
            if fname.endswith('.zarr') and 'timeseries_reprojected' in fname:
                files.append({
                    'grid_cell': 'global',
                    'zarr_path': os.path.join(spatial_dir, fname)
                })
        
        return files

    def get_hpc_output_path(self, stage: str) -> str:
        """Get HPC output path for a given stage."""
        if stage == "annual":
            base_path = os.path.join(self.hpc_root, self.data_path, "processed", "stage_1")
        elif stage == "spatial":
            base_path = os.path.join(self.hpc_root, self.data_path, "processed", "stage_2")
        elif stage == "tabular":
            base_path = os.path.join(self.hpc_root, self.data_path, "processed", "stage_3")
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
            logger.exception(f"Error processing target {stage}" + (f"/{year}" if year is not None else "") + f"/{grid_cell}: {e}")
            return False

    def _process_annual_target(self, target: Dict[str, Any]) -> bool:
        """Process annual target - now without Dask context manager."""
        year = target['year']
        source_files = target['source_files']
        output_path = self._strip_remote_prefix(target['output_path'])
        
        if not self.override and os.path.exists(output_path):
            logger.info(f"Skipping annual processing for {year}, output already exists: {output_path}")
            return True
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            source_file = self._resolve_source_file_path(source_files[0])
            data_array = self._process_data_files(source_file, year)
            
            if data_array is None:
                logger.error(f"Failed to process data for year {year}")
                return False
            
            success = self._create_annual_zarr_hpc(data_array, output_path)
            
            if success:
                logger.info(f"Annual processing complete for {year}: {output_path}")
            
            return success
            
        except Exception as e:
            logger.exception(f"Error in EOG annual processing: {e}")
            return False

    def _resolve_source_file_path(self, file_path: str) -> str:
        """Resolve the full path to a source file."""
        if os.path.isabs(file_path) or (self.hpc_root and file_path.startswith(self.hpc_root)):
            return file_path
        return os.path.join(self.hpc_root, self.data_path, "raw", file_path)

    def _initialize_dask_client(self):
        """Initialize Dask client for parallel processing using the context manager."""
        from gnt.data.common.dask.client import DaskClientContextManager

        dask_params = {
            'threads': self.dask_threads,
            'memory_limit': self.dask_memory_limit,
            'dashboard_port': 8787,
            'temp_dir': os.path.join(self.temp_dir, "dask_workspace")
        }
        return DaskClientContextManager(**dask_params)

    def _create_annual_zarr_hpc(self, data_array: xr.DataArray, output_path: str) -> bool:
        """Create annual zarr file for HPC mode with Dask optimization."""
        cleanup_file = None
        
        try:
            # Check if there's a file to clean up after processing
            cleanup_file = data_array.attrs.get('_cleanup_file')
            
            # Define chunking strategy optimized for zarr storage
            chunks = {"x": 1000, "y": 1000}
            
            # Turn into Dataset if not already
            dataset = data_array.to_dataset(name=self.source_type)
            
            # Remove cleanup info from dataset attributes before saving
            if '_cleanup_file' in dataset.attrs:
                del dataset.attrs['_cleanup_file']
            
            # Rechunk the dataset for optimal zarr storage
            logger.info(f"Rechunking dataset for zarr storage: {chunks}")
            dataset = dataset.chunk(chunks)
            
            
            # Set up compression for Zarr output
            compressor = BloscCodec(cname="zstd", clevel=3, shuffle='bitshuffle', blocksize=0)
            encoding = {var: {'compressors': (compressor,)} for var in dataset.data_vars}
            
            logger.info(f"Writing EOG dataset to zarr with Dask: {output_path}")
            dataset.to_zarr(
                output_path, 
                mode="w", 
                encoding=encoding, 
                zarr_version=3, 
                consolidated=False
            )
            
            logger.info(f"Created annual zarr file at {output_path}")
            
            # Now it's safe to clean up the uncompressed file
            if cleanup_file and os.path.exists(cleanup_file):
                os.remove(cleanup_file)
                logger.info(f"Cleaned up uncompressed file: {cleanup_file}")
            
            return True
            
        except Exception as e:
            logger.exception(f"Error creating zarr file: {e}")
            
            # Clean up uncompressed file if error occurred
            if cleanup_file and os.path.exists(cleanup_file):
                os.remove(cleanup_file)
                logger.info(f"Cleaned up uncompressed file after error: {cleanup_file}")
            
            return False

    def _process_data_files(self, file_path: str, year: int) -> xr.DataArray:
        """Process EOG data files into xarray format using Dask."""
        if not file_path:
            return None
            
        uncompressed_file_to_delete = None  # Track uncompressed file for later cleanup
        
        try:
            local_file = file_path
            
            # Uncompress if needed
            if file_path.endswith(".gz"):
                import gzip
                import shutil
                uncompressed = local_file[:-3]
                with gzip.open(local_file, 'rb') as f_in:
                    with open(uncompressed, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                local_file = uncompressed
                uncompressed_file_to_delete = uncompressed  # Mark for deletion after zarr export
            
            # Check if file exists before opening
            if not os.path.exists(local_file):
                logger.error(f"File does not exist: {local_file}")
                return None
            
            # Open as xarray with rioxarray and Dask chunks
            logger.info(f"Opening EOG file with Dask chunks: {local_file}")
            ds = rxr.open_rasterio(local_file, chunks="auto")
            
            # Add time coordinate
            ds = ds.expand_dims(
                dim={"time": 1}
            ).assign_coords(
                {"time": [pd.Timestamp(f"{year}-12-31")]}
            )
            
            if self.source_type == "dmsp":
                filename = os.path.basename(file_path)
                match = re.search(r'F(\d+)(\d{4})', filename)
                if match:
                    satellite = f"F{match.group(1)}"
                    ds = ds.assign_attrs(satellite=satellite)
            
            # Store cleanup info in dataset attributes for later use
            if uncompressed_file_to_delete:
                ds.attrs['_cleanup_file'] = uncompressed_file_to_delete
                                    
            logger.info(f"Successfully loaded EOG data with shape {ds.shape} and chunks {ds.chunks}")
            return ds
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'EOGPreprocessor':
        """Create an instance from configuration dictionary."""
        return cls(**config)

    def _process_spatial_target(self, target: Dict[str, Any]) -> bool:
        """
        Process spatial stage using common spatial processing utilities.
        """
        logger.info("Starting spatial stage processing")
        
        output_path = self._strip_remote_prefix(target['output_path'])
        source_files = target.get('source_files', [])
        
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
                
                # Initialize spatial processor
                spatial_processor = SpatialProcessor(
                    hpc_root=self.hpc_root,
                    temp_dir=self.temp_dir,
                    dask_client=client
                )
                
                with spatial_processor.setup_dask_config():
                    
                    # Define EOG-specific functions
                    def extract_year_from_eog_path(file_path: str) -> Optional[int]:
                        """Extract year from EOG zarr file path."""
                        try:
                            return int(os.path.splitext(os.path.basename(file_path))[0])
                        except ValueError:
                            return None
                    
                    def preprocess_eog_dataset(ds: xr.Dataset) -> xr.Dataset:
                        """Preprocess EOG dataset for reprojection."""
                        # Set CRS if needed
                        if ds.rio.crs is None:
                            ds = ds.rio.write_crs(4326)
                        return ds
                    
                    def get_eog_variables_and_attrs(file_path: str) -> Tuple[List[str], Dict]:
                        """Get variables and attributes from EOG sample file."""
                        sample_ds = xr.open_zarr(file_path, mask_and_scale=False, chunks='auto', consolidated=False)
                        variables = list(sample_ds.data_vars.keys())
                        sample_attrs = sample_ds.attrs.copy()
                        sample_ds.close()
                        return variables, sample_attrs
                    
                    # Use standard spatial processing workflow
                    success = spatial_processor.process_spatial_standard(
                        source_files=source_files,
                        output_path=output_path,
                        years_to_process=self.years_to_process,
                        year_pattern_func=extract_year_from_eog_path,
                        preprocess_func=preprocess_eog_dataset,
                        get_variables_func=get_eog_variables_and_attrs
                    )
                    
                    if success:
                        logger.info("EOG spatial stage processing completed successfully")
                    
                    return success
                        
        except Exception as e:
            logger.exception(f"Error in EOG spatial processing: {e}")
            return False
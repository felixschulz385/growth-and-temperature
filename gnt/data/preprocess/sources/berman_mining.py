import os
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
import re
import dask.array as da
from odc.geo import CRS
from gnt.data.common.geobox import get_or_create_geobox
from zarr.codecs import BloscCodec
from odc.geo.xr import xr_reproject

from gnt.data.preprocess.sources.base import AbstractPreprocessor

logger = logging.getLogger(__name__)

class BermanMiningPreprocessor(AbstractPreprocessor):
    """
    Preprocessor for Berman mining data.
    
    This class handles the preprocessing of mining location data to create gridded
    representations of mining activities. The preprocessing involves:
    1. Loading mining data from Stata file
    2. Creating xarray dataset with mine locations and attributes
    3. Rasterizing to create gridded representations
    
    The preprocessing goes directly to spatial stage (no stage_1/vector processing).
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Berman mining preprocessor.
        
        Args:
            **kwargs: Configuration parameters including:
                stage (str): Processing stage ("spatial")
                year_range (list): [start_year, end_year] range to process
                hpc_target (str): Required - HPC root path
                temp_dir (str): Directory for temporary files
                override (bool): Whether to reprocess existing outputs
                data_path (str): Data path for outputs
                mining_data_path (str): Path to mining data file
        """
        super().__init__(**kwargs)
        
        # Set processing stage
        self.stage = kwargs.get('stage', 'spatial')
        if self.stage not in ['spatial']:
            raise ValueError(f"Unsupported stage: {self.stage}. Use 'spatial'.")
        
        # Set year range - will be determined from data if not provided
        self.year_range = kwargs.get('year_range')
        
        # HPC mode parameters
        hpc_target = kwargs.get('hpc_target')
        self.hpc_root = self._strip_remote_prefix(hpc_target)
        if not self.hpc_root:
            raise ValueError("hpc_target is required for HPC mode")
        self.index_dir = os.path.join(self.hpc_root, "hpc_data_index")
        
        # Data path configuration
        self.data_path = kwargs.get('data_path', 'berman_mining')
        
        # Settings
        self.override = kwargs.get('override', False)
        
        # Mining data file path
        self.mining_data_path = kwargs.get('mining_data_path')
        if not self.mining_data_path:
            # Default to standard location
            self.mining_data_path = os.path.join(
                self.hpc_root, 
                self.data_path, 
                "raw", 
                "baseline",
                "BCRT_baseline.dta"
            )
        
        # Dask configuration
        self.dask_threads = kwargs.get('dask_threads', None)
        self.dask_memory_limit = kwargs.get('dask_memory_limit', None)
        
        # Initialize parquet index path
        self._init_parquet_index_path()
        
        # Setup temporary directory
        self.temp_dir = kwargs.get('temp_dir')
        if not self.temp_dir:
            self.temp_dir = tempfile.mkdtemp(prefix="berman_mining_processor_")
        else:
            os.makedirs(self.temp_dir, exist_ok=True)
        
        logger.info(f"Initialized BermanMiningPreprocessor")
        logger.info(f"HPC root: {self.hpc_root}")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Mining data: {self.mining_data_path}")
    
    def _init_parquet_index_path(self):
        """Initialize parquet index path based on data path."""
        safe_data_path = self.data_path.replace("/", "_").replace("\\", "_")
        os.makedirs(self.index_dir, exist_ok=True)
        self.parquet_index_path = os.path.join(
            self.index_dir,
            f"parquet_{safe_data_path}.parquet"
        )
        logger.debug(f"Parquet index path: {self.parquet_index_path}")
    
    def _strip_remote_prefix(self, path):
        """Remove scp/ssh prefix like user@host: from paths."""
        if isinstance(path, str):
            return re.sub(r"^[^@]+@[^:]+:", "", path)
        return path
    
    def get_preprocessing_targets(self, stage: str, year_range: Tuple[int, int] = None) -> List[Dict[str, Any]]:
        """Generate list of preprocessing targets."""
        if stage == 'spatial':
            return self._generate_spatial_targets()
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def _generate_spatial_targets(self) -> List[Dict]:
        """Generate spatial processing targets - one target for all years."""
        target = {
            'stage': 'spatial',
            'output_path': f"{self.get_hpc_output_path('spatial')}/berman_mining_timeseries_reprojected.zarr",
            'dependencies': [],
            'metadata': {
                'data_type': 'berman_mining_spatial',
                'year_range': self.year_range
            }
        }
        return [target]
    
    def process_target(self, target: Dict[str, Any]) -> bool:
        """Process a single preprocessing target."""
        stage = target.get('stage')
        
        logger.info(f"Processing Berman mining target: {stage}")
        
        try:
            if stage == 'spatial':
                return self._process_spatial_target(target)
            else:
                logger.error(f"Unknown stage: {stage}")
                return False
                
        except Exception as e:
            logger.exception(f"Error processing Berman mining target {stage}: {e}")
            return False
    
    def _process_spatial_target(self, target: Dict[str, Any]) -> bool:
        """Process spatial target - create gridded mining dataset."""
        output_path = self._strip_remote_prefix(target['output_path'])

        if not self.override and os.path.exists(output_path):
            logger.info(f"Skipping spatial processing, output already exists: {output_path}")
            return True

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            # Load and process mining data
            mines_ds = self._create_mining_dataset()

            if mines_ds is None:
                logger.error("Failed to create mining dataset")
                return False
            
            # Get the standard geobox
            geobox = self._get_or_create_geobox()
            # from odc.geo import GeoboxTiles
            # geobox = GeoboxTiles(geobox, (512, 512))[10,10]
            logger.info(f"Using geobox with shape: {geobox.shape}")
            
            # Convert all variables to uint8 
            for var in mines_ds.data_vars:
                mines_ds[var] = mines_ds[var].fillna(255).astype(np.uint8, casting='unsafe')
                
            # Reproject the entire dataset to the target geobox
            logger.info("Reprojecting mining data to target geobox...")
            reprojected_ds = xr_reproject(mines_ds, geobox, resampling="nearest", dst_nodata=255)
            
            # Get years for time dimension
            years = sorted(reprojected_ds.year.values)
            
            # Prepare dataset for writing
            # Rename year dimension to time and add band dimension
            reprojected_ds = reprojected_ds.rename({'year': 'time'})
            
            reprojected_ds['time'] = pd.to_datetime([f"{year}-12-31" for year in years])
            
            # Add band dimension
            reprojected_ds = reprojected_ds.expand_dims('band').assign_coords(band=[1])
            
            # Round spatial coordinates
            reprojected_ds = reprojected_ds.assign_coords({
                'latitude': geobox.coords['latitude'].values.round(5),
                'longitude': geobox.coords['longitude'].values.round(5)
            })
            
            # Drop spatial_ref if present
            reprojected_ds = reprojected_ds.drop_vars(['spatial_ref'], errors='ignore')
            
            # Rechunk
            reprojected_ds = reprojected_ds.chunk({'time': 1, 'band': 1, 'latitude': 512, 'longitude': 512})
            
            # Set up compression
            compressor = BloscCodec(cname="zstd", clevel=3, shuffle='bitshuffle', blocksize=0)
            encoding = {}
            for var in reprojected_ds.data_vars:
                encoding[var] = {
                    'chunks': (1, 512, 512, 1),
                    'compressors': (compressor,),
                    'dtype': 'uint8',
                    'fill_value': 255
                }
            
            # Write to zarr
            logger.info(f"Writing reprojected mining data to {output_path}")
            reprojected_ds.to_zarr(
                output_path,
                mode="w",
                encoding=encoding,
                zarr_format=3,
                consolidated=False
            )
            
            logger.info(f"Spatial Berman mining processing complete: {output_path}")
            return True

        except Exception as e:
            logger.exception(f"Error in Berman mining spatial processing: {e}")
            return False
    
    def _create_mining_dataset(self) -> Optional[xr.Dataset]:
        """Create mining dataset following the notebook logic."""
        try:
            # Load mining data
            logger.info(f"Loading mining data from: {self.mining_data_path}")
            mines = pd.read_stata(self.mining_data_path)
            
            logger.info(f"Loaded mining data with {len(mines)} records and columns: {list(mines.columns)}")
            
            # Select only the two numeric variables we need
            variables = ["nb_mines_a", "nb_diamond"]
            
            # Create xarray dataset
            logger.info("Creating xarray dataset from mining data...")
            mines_ds = xr.Dataset.from_dataframe(
                mines.set_index(["latitude", "longitude", "year"])[variables]
            )
            
            # Set CRS
            mines_ds = mines_ds.rio.write_crs(4326)
            
            logger.info(f"Created mining dataset with {len(mines_ds.year)} years")
            
            # Apply year range filter if specified
            if self.year_range:
                logger.info(f"Filtering to year range: {self.year_range}")
                mines_ds = mines_ds.sel(year=slice(self.year_range[0], self.year_range[1]))
            
            return mines_ds
            
        except Exception as e:
            logger.exception(f"Error creating mining dataset: {e}")
            return None
    
    def _get_or_create_geobox(self):
        """Get or create geobox using the common utility - same location as misc preprocessor."""
        misc_level1_dir = os.path.join(self.hpc_root, "misc", "processed", "stage_1", "misc")
        os.makedirs(misc_level1_dir, exist_ok=True)
        return get_or_create_geobox(self.hpc_root, misc_level1_dir)
    
    def get_hpc_output_path(self, stage: str) -> str:
        """Get output path for a given stage."""
        if stage == "spatial":
            base_path = os.path.join(self.hpc_root, self.data_path, "processed", "stage_2")
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        return self._strip_remote_prefix(base_path)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BermanMiningPreprocessor':
        """Create an instance from configuration dictionary."""
        return cls(**config)

    def get_hpc_output_path(self, stage: str) -> str:
        """Get output path for a given stage."""
        if stage == "spatial":
            base_path = os.path.join(self.hpc_root, self.data_path, "processed", "stage_2")
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        return self._strip_remote_prefix(base_path)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BermanMiningPreprocessor':
        """Create an instance from configuration dictionary."""
        return cls(**config)

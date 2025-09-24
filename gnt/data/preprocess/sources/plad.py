import os
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from itertools import product
from datetime import datetime
import zarr
import re
import dask.array as da
from odc.geo import CRS
from gnt.data.common.geobox import get_or_create_geobox
from functools import partial
from zarr.codecs import BloscCodec
from odc.geo.xr import rasterize, xr_zeros
from odc.geo.geom import Geometry
import shapely

from gnt.data.preprocess.sources.base import AbstractPreprocessor
from gnt.data.preprocess.common.spatial import SpatialProcessor
from gnt.data.common.geobox import get_or_create_geobox

logger = logging.getLogger(__name__)

class PLADPreprocessor(AbstractPreprocessor):
    """
    Preprocessor for Political Leaders and Development (PLAD) data.
    
    This class handles the preprocessing of PLAD data to create gridded boolean 
    matrices showing regional favoritism by country and year. The preprocessing involves:
    1. Loading PLAD data and administrative boundaries
    2. Creating country-year panels with regional favoritism indicators
    3. Rasterizing to create gridded boolean representations
    
    The preprocessing is organized in two stages:
    - Stage 1 (annual): Create annual zarr files with boolean matrices by country-year
    - Stage 2 (spatial): Reproject the data to a unified grid for analysis
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the PLAD preprocessor.
        
        Args:
            **kwargs: Configuration parameters including:
                stage (str): Processing stage ("spatial", "tabular")
                year_range (list): [start_year, end_year] range to process
                hpc_target (str): Required - HPC root path
                temp_dir (str): Directory for temporary files
                override (bool): Whether to reprocess existing outputs
                admin_level (int): Administrative level (1 or 2)
                data_path (str): Data path for outputs
        """
        super().__init__(**kwargs)
        
        # Set processing stage
        self.stage = kwargs.get('stage', 'spatial')
        if self.stage not in ['spatial', 'tabular']:
            raise ValueError(f"Unsupported stage: {self.stage}. Use 'spatial' or 'tabular'.")
        
        # Set year range - default to 1980-2022 based on notebook
        self.year_range = kwargs.get('year_range', [1980, 2022])
        if not isinstance(self.year_range, list) or len(self.year_range) != 2:
            raise ValueError("'year_range' must be a list with exactly two elements [start_year, end_year]")
        
        self.year_start = self.year_range[0]
        self.year_end = self.year_range[1]
        self.years_to_process = list(range(self.year_start, self.year_end + 1))
        
        # HPC mode parameters
        hpc_target = kwargs.get('hpc_target')
        self.hpc_root = self._strip_remote_prefix(hpc_target)
        if not self.hpc_root:
            raise ValueError("hpc_target is required for HPC mode")
        self.index_dir = os.path.join(self.hpc_root, "hpc_data_index")
        
        # Data path configuration
        self.data_path = kwargs.get('data_path', 'plad')
        
        # Settings
        self.override = kwargs.get('override', False)
        
        # Administrative level to process (1 or 2)
        self.admin_level = kwargs.get('admin_level', 1)
        if self.admin_level not in [1, 2]:
            raise ValueError("admin_level must be 1 or 2")
        
        # Dask configuration
        self.dask_threads = kwargs.get('dask_threads', None)
        self.dask_memory_limit = kwargs.get('dask_memory_limit', None)
        
        # Initialize parquet index path
        self._init_parquet_index_path()
        
        # Setup temporary directory
        self.temp_dir = kwargs.get('temp_dir')
        if not self.temp_dir:
            self.temp_dir = tempfile.mkdtemp(prefix="plad_processor_")
        else:
            os.makedirs(self.temp_dir, exist_ok=True)
        
        logger.info(f"Initialized PLADPreprocessor for administrative level {self.admin_level}")
        logger.info(f"HPC root: {self.hpc_root}")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Years to process: {len(self.years_to_process)} ({self.year_start}-{self.year_end})")
    
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
        """Generate list of preprocessing targets by reading from parquet index when available."""
        if stage == 'spatial':
            return self._generate_spatial_targets()
        elif stage == 'tabular':
            return self._generate_tabular_targets()
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def _generate_spatial_targets(self) -> List[Dict]:
        """Generate spatial processing targets - one target for all years."""
        target = {
            'stage': 'spatial',
            'output_path': f"{self.get_hpc_output_path('spatial')}/plad_adm{self.admin_level}_panel.zarr",
            'dependencies': [],
            'metadata': {
                'data_type': 'plad_spatial',
                'admin_level': self.admin_level,
                'year_range': self.year_range
            }
        }
        return [target]
    
    def _generate_tabular_targets(self) -> List[Dict]:
        """Generate tabular processing targets."""
        spatial_files = self._get_all_spatial_files()
        
        if not spatial_files:
            logger.warning("No spatial files available for tabular processing")
            return []
        
        target = {
            'stage': 'tabular',
            'source_files': [f['zarr_path'] for f in spatial_files],
            'output_path': f"{self.get_hpc_output_path('tabular')}/plad_adm{self.admin_level}_tabular.parquet",
            'dependencies': [f['zarr_path'] for f in spatial_files],
            'metadata': {
                'data_type': 'plad_tabular',
                'admin_level': self.admin_level,
                'processing_type': 'zarr_to_parquet'
            }
        }
        return [target]
    
    def _get_all_spatial_files(self) -> List[Dict]:
        """Return all available spatial zarr files."""
        spatial_dir = self.get_hpc_output_path('spatial')
        if not os.path.exists(spatial_dir):
            return []
        
        files = []
        for fname in os.listdir(spatial_dir):
            if fname.endswith('.zarr') and 'panel' in fname:
                files.append({
                    'admin_level': self.admin_level,
                    'zarr_path': os.path.join(spatial_dir, fname)
                })
        
        return files
    
    def get_hpc_output_path(self, stage: str) -> str:
        """Get HPC output path for a given stage."""
        if stage == "spatial":
            base_path = os.path.join(self.hpc_root, "plad", "processed", "stage_2")
        elif stage == "tabular":
            base_path = os.path.join(self.hpc_root, "plad", "processed", "stage_3")
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        return self._strip_remote_prefix(base_path)
    
    def process_target(self, target: Dict[str, Any]) -> bool:
        """Process a single preprocessing target."""
        stage = target.get('stage')
        
        logger.info(f"Processing PLAD target: {stage}")
        
        try:
            if stage == 'spatial':
                return self._process_spatial_target(target)
            elif stage == 'tabular':
                return self._process_tabular_target(target)
            else:
                logger.error(f"Unknown stage: {stage}")
                return False
                
        except Exception as e:
            logger.exception(f"Error processing PLAD target {stage}: {e}")
            return False
    
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

    def _process_spatial_target(self, target: Dict[str, Any]) -> bool:
        """Process spatial target - create boolean panel dataset."""
        output_path = self._strip_remote_prefix(target['output_path'])

        if not self.override and os.path.exists(output_path):
            logger.info(f"Skipping spatial processing, output already exists: {output_path}")
            return True

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            # Load and process PLAD data following notebook logic
            panel_gdf = self._create_plad_panel()

            if panel_gdf is None or panel_gdf.empty:
                logger.error("Failed to create PLAD panel")
                return False
            
            # Get the standard geobox
            geobox = self._get_or_create_geobox()
            logger.info(f"Using geobox with shape: {geobox.shape}")
            
            # Get unique years
            years = sorted(panel_gdf['year'].unique())
            
            # Create empty zarr structure
            if not self._create_empty_plad_zarr(output_path, geobox, years):
                return False

            # Rasterize the panel to xarray dataset
            success = self._rasterize_panel(panel_gdf, output_path, geobox, years)
            
            if success:
                logger.info(f"Spatial PLAD processing complete: {output_path}")

            return success

        except Exception as e:
            logger.exception(f"Error in PLAD spatial processing: {e}")
            return False
    
    def _resolve_data_files_from_index(self) -> Dict[str, str]:
        """Resolve data file paths from parquet index and known preprocessed locations."""
        data_files = {}
        
        # Try PLAD-specific index first
        if os.path.exists(self.parquet_index_path):
            plad_files = self._resolve_plad_files_from_index()
            data_files.update(plad_files)
        
        # For GADM files, use the known preprocessed locations instead of parquet index
        gadm_files = self._resolve_gadm_files_from_preprocessed()
        data_files.update(gadm_files)
        
        if not data_files:
            logger.warning("No data files found")
        
        return data_files
    
    def _resolve_plad_files_from_index(self) -> Dict[str, str]:
        """Resolve PLAD data files from PLAD-specific parquet index."""
        data_files = {}
        
        try:
            df = pd.read_parquet(self.parquet_index_path)
            
            # Filter for completed files
            if 'status_category' in df.columns:
                completed_files = df[df['status_category'] == 'completed']
            elif 'download_status' in df.columns:
                completed_files = df[df['download_status'] == 'completed']
            else:
                logger.warning("No status column found in PLAD parquet index")
                return data_files
            
            if completed_files.empty:
                logger.warning("No completed files found in PLAD parquet index")
                return data_files
            
            # Get file paths using relative_path
            if 'relative_path' in completed_files.columns:
                file_paths = completed_files['relative_path'].tolist()
            else:
                logger.warning("No path column found in PLAD parquet index")
                return data_files
            
            # Look for PLAD data files (.dta files that are actually whitespace-delimited)
            for file_path in file_paths:
                clean_file_path = self._strip_remote_prefix(file_path)
                filename = os.path.basename(clean_file_path).lower()
                
                if 'plad' in filename and filename.endswith('.dta'):
                    data_files['plad_data'] = clean_file_path
                    logger.info(f"Found PLAD data file: {clean_file_path}")
                    break
            
        except Exception as e:
            logger.error(f"Error reading PLAD parquet index: {e}")
            
        return data_files
    
    def _resolve_gadm_files_from_preprocessed(self) -> Dict[str, str]:
        """Resolve GADM files from known preprocessed locations in misc/processed/stage_1."""
        data_files = {}
        
        try:
            # Known location of preprocessed GADM files
            gadm_base_path = os.path.join(self.hpc_root, "misc", "processed", "stage_1", "gadm")
            
            # Look for the preprocessed GADM files
            gadm_adm1_path = os.path.join(gadm_base_path, "gadm_levelADM_1_simplified.gpkg")
            gadm_adm2_path = os.path.join(gadm_base_path, "gadm_levelADM_2_simplified.gpkg")
            
            if os.path.exists(gadm_adm1_path):
                data_files['gadm_adm1'] = gadm_adm1_path
                logger.info(f"Found preprocessed GADM ADM1 file: {gadm_adm1_path}")
            else:
                logger.warning(f"Preprocessed GADM ADM1 file not found: {gadm_adm1_path}")
            
            if os.path.exists(gadm_adm2_path):
                data_files['gadm_adm2'] = gadm_adm2_path
                logger.info(f"Found preprocessed GADM ADM2 file: {gadm_adm2_path}")
            else:
                logger.warning(f"Preprocessed GADM ADM2 file not found: {gadm_adm2_path}")
            
            logger.info(f"Found {len(data_files)} preprocessed GADM files")
            
        except Exception as e:
            logger.error(f"Error resolving preprocessed GADM files: {e}")
            
        return data_files
    
    def _resolve_gadm_files_from_misc_index(self, misc_index_path: str) -> Dict[str, str]:
        """Legacy method - now unused since we use preprocessed files directly."""
        logger.debug("Skipping misc index lookup - using preprocessed GADM files directly")
        return {}
    
    def _get_data_file_paths(self) -> Dict[str, str]:
        """Get data file paths, either from parquet index or fallback to manual paths."""
        # First try to resolve from parquet index
        data_files = self._resolve_data_files_from_index()
        
        # If parquet index didn't provide all files, use manual configuration
        if not data_files.get('plad_data'):
            # Use the original approach with explicit path parameters
            if hasattr(self, 'plad_data_path') and self.plad_data_path:
                data_files['plad_data'] = self.plad_data_path
        
        if not data_files.get('gadm_adm1'):
            if hasattr(self, 'gadm_adm1_path') and self.gadm_adm1_path:
                data_files['gadm_adm1'] = self.gadm_adm1_path
        
        if not data_files.get('gadm_adm2'):
            if hasattr(self, 'gadm_adm2_path') and self.gadm_adm2_path:
                data_files['gadm_adm2'] = self.gadm_adm2_path
        
        # Resolve full paths if they're relative
        for key, path in data_files.items():
            if path and not os.path.isabs(path):
                data_files[key] = os.path.join(self.hpc_root, self.data_path, "raw", path)
        
        return data_files
    
    def _create_plad_panel(self) -> Optional[gpd.GeoDataFrame]:
        """Create PLAD panel following the notebook logic."""
        try:
            # Get data file paths
            data_files = self._get_data_file_paths()
            
            plad_data_path = data_files.get('plad_data')
            gadm_adm1_path = data_files.get('gadm_adm1')
            gadm_adm2_path = data_files.get('gadm_adm2')
            
            if not plad_data_path:
                raise ValueError("PLAD data file not found")
            
            # Load PLAD data - handle .dta files as whitespace-delimited
            logger.info(f"Loading PLAD data from: {plad_data_path}")
            plad = pd.read_table(plad_data_path)
            
            logger.info(f"Loaded PLAD data with {len(plad)} records and columns: {list(plad.columns)}")
            
            # Load administrative boundaries
            if self.admin_level == 1:
                if not gadm_adm1_path:
                    raise ValueError("GADM ADM1 file not found - ensure misc preprocessing has been run")
                logger.info(f"Loading ADM1 boundaries from: {gadm_adm1_path}")
                adm_gdf = gpd.read_file(gadm_adm1_path)
                gid_col = "GID_1"
                reg_fav_col = "reg_fav_adm_1"
            else:
                if not gadm_adm2_path:
                    raise ValueError("GADM ADM2 file not found - ensure misc preprocessing has been run")
                logger.info(f"Loading ADM2 boundaries from: {gadm_adm2_path}")
                adm_gdf = gpd.read_file(gadm_adm2_path)
                gid_col = "GID_2"
                reg_fav_col = "reg_fav_adm_2"
            
            logger.info(f"Loaded {len(adm_gdf)} administrative boundaries with columns: {list(adm_gdf.columns)}")
            
            # Create administrative panel (following notebook logic)
            logger.info("Creating administrative panel...")
            adm_panel = pd.DataFrame(
                list(product(adm_gdf[gid_col].unique(), self.years_to_process)),
                columns=[gid_col, "year"]
            )
            
            # Merge with geometry
            adm_panel = pd.merge(adm_panel, adm_gdf[[gid_col, "geometry"]])
            
            # Create PLAD panel
            logger.info("Creating PLAD panel...")
            plad_panel = pd.DataFrame(
                list(product(plad.gid_0.unique(), self.years_to_process)),
                columns=["gid_0", "year"]
            )
            
            # Process PLAD data (following notebook processor function)
            def processor(row):
                qresults = plad.loc[
                    (plad.startyear <= row["year"]) & 
                    (plad.endyear >= row["year"]) & 
                    (plad.gid_0 == row["gid_0"]),
                    ["gid_1", "gid_2"]
                ]
                if qresults.empty:
                    return pd.Series()
                else:
                    return qresults.iloc[0]
            
            plad_panel[["reg_fav_adm_1", "reg_fav_adm_2"]] = plad_panel.apply(processor, axis=1)
            
            # Merge with administrative panel
            logger.info("Merging PLAD with administrative boundaries...")
            reg_fav_panel = pd.merge(
                adm_panel, plad_panel,
                left_on=[gid_col, "year"],
                right_on=[reg_fav_col, "year"],
                how="left"
            )
            
            # Create boolean regional favoritism indicator
            reg_fav_panel["reg_fav"] = (~reg_fav_panel[reg_fav_col].isna()).astype(int)
            
            # Clean up columns
            columns_to_drop = ["gid_0", "reg_fav_adm_1", "reg_fav_adm_2"]
            reg_fav_panel = reg_fav_panel.drop(columns=[col for col in columns_to_drop if col in reg_fav_panel.columns])
            
            # Convert to GeoDataFrame
            panel_gdf = gpd.GeoDataFrame(reg_fav_panel)
            
            logger.info(f"Created PLAD panel with {len(panel_gdf)} records")
            
            return panel_gdf
            
        except Exception as e:
            logger.exception(f"Error creating PLAD panel: {e}")
            return None
    
    def _get_or_create_geobox(self):
        """Get or create geobox using the common utility - same location as misc preprocessor."""
        misc_level1_dir = os.path.join(self.hpc_root, "misc", "processed", "stage_1", "misc")
        os.makedirs(misc_level1_dir, exist_ok=True)
        return get_or_create_geobox(self.hpc_root, misc_level1_dir)
    
    def _create_empty_plad_zarr(self, output_path: str, geobox, years: List[int]) -> bool:
        """Create empty zarr file with target dimensions for PLAD data."""
        try:
            logger.info("Creating empty PLAD zarr structure")
            
            # Create time coordinates
            time_coords = pd.to_datetime([f"{year}-12-31" for year in years])
            
            # Get geobox dimensions and coordinates
            ny, nx = geobox.shape
            lat_coords = geobox.coords['latitude'].values.round(5)
            lon_coords = geobox.coords['longitude'].values.round(5)
            
            # Create data variable for regional favoritism
            data_var = xr.DataArray(
                da.zeros((len(years), 1, ny, nx), dtype=bool),
                dims=['time', 'band', 'latitude', 'longitude'],
                coords={
                    'time': time_coords,
                    'band': [1],
                    'latitude': lat_coords,
                    'longitude': lon_coords
                },
                attrs={
                    'long_name': 'Regional Favoritism Indicator',
                    'description': f'Boolean indicator for regional favoritism at ADM{self.admin_level} level',
                    'admin_level': self.admin_level,
                    'dtype': 'bool'
                }
            )
            
            # Create dataset
            empty_ds = xr.Dataset({
                'reg_fav': data_var
            })
            
            # Set CRS
            empty_ds = empty_ds.rio.write_crs(geobox.crs)
            
            # Set up compression for boolean data
            compressor = BloscCodec(cname="zstd", clevel=3, shuffle='bitshuffle', blocksize=0)
            encoding = {
                'reg_fav': {
                    'chunks': (1, 1, 512, 512),
                    'compressors': (compressor,),
                    'dtype': 'bool'
                }
            }
            
            # Write empty zarr structure
            logger.info(f"Writing empty PLAD zarr structure to: {output_path}")
            empty_ds.to_zarr(
                output_path,
                mode="w",
                encoding=encoding,
                compute=False,
                zarr_format=3,
                consolidated=False
            )
            
            logger.info("Empty PLAD zarr structure created successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Error creating empty PLAD zarr: {e}")
            return False

    def _rasterize_panel(self, panel_gdf: gpd.GeoDataFrame, output_path: str, geobox, years: List[int]) -> bool:
        """Rasterize the PLAD panel to create xarray dataset using geobox."""
        try:
            logger.info("Rasterizing PLAD panel using geobox...")
            
            # Create data arrays for each year
            data_arrays = []
            
            for year in years:
                logger.debug(f"Rasterizing year {year}")
                year_data = panel_gdf[panel_gdf['year'] == year]
                
                if year_data.empty:
                    # Create empty raster with geobox dimensions - use boolean type
                    data_array = xr_zeros(geobox, dtype=bool)
                else:
                    # Filter to only regions with favoritism
                    favoritism_regions = year_data[year_data['reg_fav'] == 1]
                    
                    if favoritism_regions.empty:
                        data_array = xr_zeros(geobox, dtype=bool)
                    else:
                        # Unpack existing multipolygons
                        geom_list = [geom for mgeom in favoritism_regions.geometry for geom in mgeom.geoms]
                        # Create MultiPolygon from favoritism regions
                        favoritism_polygons = shapely.MultiPolygon(geom_list)
                        geom = Geometry(favoritism_polygons, crs=str(year_data.crs))
                        
                        # Rasterize using geobox - results in boolean mask
                        data_array = rasterize(geom, geobox).astype(bool)
                        
                # Add band dimension
                data_array = data_array.expand_dims('time').assign_coords(time=[pd.Timestamp(f"{year}-12-31")])     
                data_array = data_array.expand_dims('band').assign_coords(band=[1])
                
                # Round spatial dimensions
                data_array = data_array.assign_coords(
                    {
                        'latitude': geobox.coords['latitude'].values.round(5),
                        'longitude': geobox.coords['longitude'].values.round(5)
                        }
                    )
                
                # Set CRS using geobox CRS
                data_array = data_array.drop_vars(['spatial_ref'])
                
                # Create dataset for this year
                dataset = data_array.to_dataset(name='reg_fav')
                
                try:        
                    # Set up compression - use different settings for boolean data
                    compressor = BloscCodec(cname="zstd", clevel=3, shuffle='bitshuffle', blocksize=0)
                    encoding = {}
                    for var in dataset.data_vars:
                        encoding[var] = {
                            'compressors': (compressor,),
                            'dtype': 'bool'
                        }
                        
                    # Write to zarr using region parameter
                    dataset.to_zarr(
                        output_path,
                        region="auto",
                        consolidated=False
                    )
                                
                    logger.info(f"Successfully wrote year {year} data to zarr")
                                
                except Exception as e:
                    logger.exception(f"Error writing year {year} to zarr: {e}")
                    return False
            
        except Exception as e:
            logger.exception(f"Error rasterizing PLAD panel: {e}")
            return False
        
        return True
    
    def _process_tabular_target(self, target: Dict[str, Any]) -> bool:
        """Process tabular stage using the common implementation."""
        return super()._process_tabular_target(target)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'PLADPreprocessor':
        """Create an instance from configuration dictionary."""
        return cls(**config)
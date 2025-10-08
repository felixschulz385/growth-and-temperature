import os
import tempfile
import logging
import zipfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import geopandas as gpd
from datetime import datetime
import json
import re
import pickle
import rioxarray as rxr
import pandas as pd
import numpy as np
import dask.array as da
from zarr.codecs import BloscCodec

from gnt.data.preprocess.sources.base import AbstractPreprocessor
from gnt.data.download.sources.misc import MiscDataSource
from gnt.data.common.geobox import get_or_create_geobox

# Add pandas for reading parquet files directly
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

logger = logging.getLogger(__name__)

class MiscPreprocessor(AbstractPreprocessor):
    """
    HPC-mode preprocessor for auxiliary boundary data.
    
    This class handles preprocessing of auxiliary data sources like:
    - OpenStreetMap land polygons
    - GADM administrative boundaries
    
    It uses a single-stage workflow to:
    1. Extract data from source files
    2. Simplify geometries and standardize formats
    3. Create derived products (like rasterized versions)
    
    Requires HPC mode configuration with parquet index.
    """
    
    # Default grid parameters
    DEFAULT_GRID_RESOLUTION = 0.008333  # ~1km at equator
    DEFAULT_CRS = "EPSG:4326"           # WGS84
    
    def __init__(self, **kwargs):
        """
        Initialize the misc data preprocessor in HPC mode.
        
        Args:
            **kwargs: Configuration parameters including:
                subsource (str): Optional - Subsource name (e.g., 'osm', 'gadm', 'country_classifications') to process specific subsource only
                sources (Dict): Required - Sources configuration from data.yaml
                output_path (str): Required - Path for processed outputs
                hpc_target (str): Required - Base path for HPC outputs
                local_index_dir (str): Required - Directory for parquet index
                simplify_tolerance (float): Tolerance for polygon simplification
                rasterize (bool): Whether to create raster versions
                temp_dir (str): Directory for temporary files
                dask_threads (int): Number of Dask threads
                dask_memory_limit (int): Dask memory limit in GB
                grid_resolution (float): Grid resolution for rasterization
                grid_bounds (List[float]): Grid bounds [west, south, east, north]
        """
        super().__init__(**kwargs)
        
        # Extract source information
        source_name = kwargs.get('name', kwargs.get('preprocessor', 'misc'))
        subsource_name = kwargs.get('subsource')
        sources_config = kwargs.get('sources', {})
        
        if source_name not in sources_config:
            raise ValueError(f"Source '{source_name}' not found in sources configuration")
        
        source_config = sources_config[source_name]
        
        # Extract files configuration from the source
        if source_config.get('type') != 'misc':
            raise ValueError(f"Source '{source_name}' is not of type 'misc'")
        
        # Convert sources substructure to files list for MiscDataSource
        files_config = []
        available_subsources = source_config.get('sources', {})
        
        # Define subsource groups - HDI and WB are always processed together
        subsource_groups = {
            'osm': ['osm'],
            'gadm': ['gadm'],
            'country_classifications': ['hdi', 'wb']  # Always process together
        }
        
        # Map individual source keys to their group
        source_to_group = {}
        for group, sources in subsource_groups.items():
            for src in sources:
                source_to_group[src] = group
        
        # If subsource is specified, resolve to its group
        target_group = None
        if subsource_name:
            # Handle both individual names and group names
            if subsource_name in subsource_groups:
                target_group = subsource_name
            elif subsource_name in source_to_group:
                target_group = source_to_group[subsource_name]
            else:
                raise ValueError(f"Subsource '{subsource_name}' not found. Available: {list(subsource_groups.keys())} or {list(source_to_group.keys())}")
            
            # Get all sources in the target group
            sources_to_process = subsource_groups[target_group]
            logger.info(f"Processing subsource group '{target_group}' which includes: {sources_to_process}")
        else:
            # Process all subsources
            sources_to_process = list(available_subsources.keys())
            logger.info(f"Processing all subsources: {sources_to_process}")
        
        # Build files_config for the selected sources
        for source_key in sources_to_process:
            if source_key in available_subsources:
                source_info = available_subsources[source_key]
                files_config.append({
                    'url': source_info['url'],
                    'name': source_info['name'],
                    'description': source_info.get('description', ''),
                    'subfolder': source_info.get('subfolder', source_key)
                })
        
        # Store subsource filter as the group name
        self.subsource_filter = target_group if subsource_name else None
        
        # Required parameters
        self.data_path = source_config.get('data_path') or kwargs.get('output_path') or "misc"
        
        # HPC mode parameters - similar to glass.py implementation
        hpc_target = kwargs.get('hpc_target')
        self.hpc_root = self._strip_remote_prefix(hpc_target)
        if not self.hpc_root:
            raise ValueError("hpc_target is required for HPC mode")
        self.index_dir = os.path.join(self.hpc_root, "hpc_data_index")
        
        # Processing parameters
        self.simplify_tolerance = kwargs.get('simplify_tolerance', 0.001)  # ~100m at equator
        self.rasterize = kwargs.get('rasterize', True)
        self.grid_resolution = kwargs.get('grid_resolution', self.DEFAULT_GRID_RESOLUTION)
        self.grid_bounds = kwargs.get('grid_bounds', [-180, -90, 180, 90])  # Global by default
        
        # Setup temporary directory
        self.temp_dir = kwargs.get('temp_dir')
        if not self.temp_dir:
            self.temp_dir = tempfile.mkdtemp(prefix="misc_processor_")
        else:
            os.makedirs(self.temp_dir, exist_ok=True)
        
        # Dask configuration
        self.dask_threads = kwargs.get('dask_threads', None)
        self.dask_memory_limit = kwargs.get('dask_memory_limit', None)
        self.overwrite = kwargs.get('overwrite', False)
        
        # Initialize HPC workflow
        self._init_hpc_workflow(files_config, self.data_path)

        # Store the data source instance for the workflow to use
        self.data_source_instance = self.data_source

        logger.info(f"Initialized MiscPreprocessor for source '{source_name}'{f' (subsource: {subsource_name})' if subsource_name else ''} in HPC mode")
        logger.info(f"HPC root: {self.hpc_root}")
        logger.info(f"Input path: {self.input_path}")
        logger.info(f"Index dir: {self.index_dir}")
        logger.info(f"Parquet index: {self.parquet_index_path}")

    def _strip_remote_prefix(self, path):
        """Remove scp/ssh prefix like user@host: from paths."""
        if isinstance(path, str):
            return re.sub(r"^[^@]+@[^:]+:", "", path)
        return path

    def _init_hpc_workflow(self, files_config: List[Dict], data_path: str):
        """Initialize HPC workflow components."""
        try:
            # All paths are relative to the HPC root
            self.data_source = MiscDataSource(
                files=files_config,  # Pass as positional argument
                output_path=os.path.join(self.hpc_root, data_path)
            )
            # Strip remote prefix from input_path if present
            self.input_path = self._strip_remote_prefix(self.data_source.data_path)
            self._init_parquet_index_path()

            logger.info(f"HPC workflow initialized successfully")
            logger.info(f"Configured {len(files_config)} file sources")

        except Exception as e:
            logger.error(f"Failed to initialize HPC workflow: {e}")
            raise

    def _init_parquet_index_path(self):
        """Always use hpc_data_index under the HPC root for the parquet index."""
        # Use the actual data_path for the index filename, not the data_source.data_path
        safe_data_path = self.data_path.replace("/", "_").replace("\\", "_")
        os.makedirs(self.index_dir, exist_ok=True)
        self.parquet_index_path = os.path.join(
            self.index_dir,
            f"parquet_{safe_data_path}.parquet"
        )

        logger.debug(f"Parquet index path: {self.parquet_index_path}")

    def get_preprocessing_targets(self, stage: str, year_range: Tuple[int, int] = None) -> List[Dict[str, Any]]:
        """
        Generate list of preprocessing targets by reading directly from parquet index.
        
        Args:
            stage: Processing stage ('vector', 'spatial', 'tabular')  
            year_range: Optional year range filter (not used for misc data)
            
        Returns:
            List of target dictionaries with source files and output specifications
        """
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
            
            # Get file paths - use destination_blob for the actual file location
            if 'destination_blob' in completed_files.columns:
                file_paths = completed_files['destination_blob'].tolist()
            elif 'relative_path' in completed_files.columns:
                # Fallback to relative_path and construct full path
                file_paths = [
                    f"{self.input_path}/{path}" if not path.startswith(self.input_path) 
                    else path 
                    for path in completed_files['relative_path'].tolist()
                ]
            else:
                logger.warning("No path column found in parquet index")
                return []
            
            logger.info(f"Found {len(file_paths)} completed files from parquet index")
            
            if stage == 'vector':
                return self._generate_vector_targets(file_paths)
            elif stage == 'spatial':
                return self._generate_spatial_targets()
            else:
                raise ValueError(f"Unknown stage: {stage}")
                
        except Exception as e:
            logger.error(f"Error reading parquet index: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return []

    def _generate_vector_targets(self, files: List[str]) -> List[Dict]:
        """Generate vector processing targets for misc data."""
        targets = []
        
        # Group files by data type
        osm_files = []
        gadm_files = []
        hdi_files = []
        wb_files = []
        
        for file_path in files:
            # Strip remote prefix from file paths
            clean_file_path = self._strip_remote_prefix(file_path)
            filename = os.path.basename(clean_file_path)
            
            if any(pattern in filename.lower() for pattern in ['osm', 'land-polygons']):
                osm_files.append(clean_file_path)
            elif 'gadm' in filename.lower():
                gadm_files.append(clean_file_path)
            elif 'hdi' in filename.lower() or 'hdr' in filename.lower():
                hdi_files.append(clean_file_path)
            elif 'dr0095334' in filename.lower() or 'world_bank' in filename.lower():
                wb_files.append(clean_file_path)
        
        logger.info(f"Categorized files: {len(osm_files)} OSM, {len(gadm_files)} GADM, {len(hdi_files)} HDI, {len(wb_files)} WB")
        
        # OSM land polygons target (only if not filtered or if filtering for osm)
        if osm_files and (not self.subsource_filter or self.subsource_filter == 'osm'):
            source_file = osm_files[0]
            target = {
                'data_type': 'osm',
                'stage': 'vector',
                'source_files': [source_file],
                'output_path': f"{self.get_hpc_output_path('vector')}/osm/land_polygons_simplified.gpkg",
                'dependencies': [],
                'metadata': {
                    'data_type': 'osm',
                    'processing_type': 'simplify_polygons',
                    'source_count': len(osm_files),
                    'source_file': source_file
                }
            }
            targets.append(target)
        
        # GADM boundaries target (only if not filtered or if filtering for gadm)
        if gadm_files and (not self.subsource_filter or self.subsource_filter == 'gadm'):
            source_file = gadm_files[0]
            target = {
                'data_type': 'gadm',
                'stage': 'vector',
                'source_files': [source_file],
                'output_path': f"{self.get_hpc_output_path('vector')}/gadm/gadm_levels_simplified.gpkg",
                'dependencies': [],
                'metadata': {
                    'data_type': 'gadm',
                    'processing_type': 'simplify_boundaries',
                    'source_count': len(gadm_files),
                    'source_file': source_file
                }
            }
            targets.append(target)
        
        # Country classifications target (HDI + World Bank income groups)
        # Always process HDI and WB together
        if (hdi_files or wb_files) and (not self.subsource_filter or self.subsource_filter == 'country_classifications'):
            source_files = []
            if hdi_files:
                source_files.append(hdi_files[0])
            if wb_files:
                source_files.append(wb_files[0])
            
            if source_files:
                target = {
                    'data_type': 'country_classifications',
                    'stage': 'vector',
                    'source_files': source_files,
                    'output_path': f"{self.get_hpc_output_path('vector')}/country_classifications/classifications.parquet",
                    'dependencies': [],
                    'metadata': {
                        'data_type': 'country_classifications',
                        'processing_type': 'classify_countries',
                        'source_count': len(source_files),
                        'has_hdi': len(hdi_files) > 0,
                        'has_wb': len(wb_files) > 0
                    }
                }
                targets.append(target)
        
        return targets
    
    def _generate_spatial_targets(self) -> List[Dict]:
        """Generate spatial processing targets for processed vector data."""
        targets = []
        
        if not self.rasterize:
            return targets
        
        # OSM land mask rasterization (only if not filtered or if filtering for osm)
        if not self.subsource_filter or self.subsource_filter == 'osm':
            osm_vector_path = f"{self.get_hpc_output_path('vector')}/osm/land_polygons_simplified.gpkg"
            if os.path.exists(osm_vector_path):
                target = {
                    'data_type': 'osm',
                    'stage': 'spatial',
                    'source_files': [osm_vector_path],
                    'output_path': f"{self.get_hpc_output_path('spatial')}/osm/land_mask.zarr",
                    'dependencies': [osm_vector_path],
                    'metadata': {
                        'data_type': 'osm',
                        'processing_type': 'rasterize_land_mask'
                    }
                }
                targets.append(target)
        
        # GADM countries grid rasterization (only if not filtered or if filtering for gadm)
        if not self.subsource_filter or self.subsource_filter == 'gadm':
            gadm_vector_path = f"{self.get_hpc_output_path('vector')}/gadm/gadm_levelADM_0_simplified.gpkg"
            if os.path.exists(gadm_vector_path):
                target = {
                    'data_type': 'gadm',
                    'stage': 'spatial',
                    'source_files': [gadm_vector_path],
                    'output_path': f"{self.get_hpc_output_path('spatial')}/gadm/countries_grid.zarr",
                    'dependencies': [gadm_vector_path],
                    'metadata': {
                        'data_type': 'gadm',
                        'processing_type': 'rasterize_countries'
                    }
                }
                targets.append(target)
        
        # Country classifications grid rasterization (only if not filtered or if filtering for hdi)
        if not self.subsource_filter or self.subsource_filter == 'country_classifications':
            classifications_parquet = f"{self.get_hpc_output_path('vector')}/country_classifications/classifications.parquet"
            gadm_zarr = f"{self.get_hpc_output_path('spatial')}/gadm/countries_grid.zarr"
            
            if os.path.exists(classifications_parquet) and os.path.exists(gadm_zarr):
                target = {
                    'data_type': 'country_classifications',
                    'stage': 'spatial',
                    'source_files': [classifications_parquet, gadm_zarr],
                    'output_path': f"{self.get_hpc_output_path('spatial')}/country_classifications/classifications_grid.zarr",
                    'dependencies': [classifications_parquet, gadm_zarr],
                    'metadata': {
                        'data_type': 'country_classifications',
                        'processing_type': 'rasterize_classifications'
                    }
                }
                targets.append(target)
        
        return targets
    
    def get_hpc_output_path(self, stage: str) -> str:
        """Get output path for a given stage, relative to the HPC root."""
        if stage == "vector":
            base_path = os.path.join(self.hpc_root, self.data_path, "processed", "stage_1")
        elif stage == "spatial":
            base_path = os.path.join(self.hpc_root, self.data_path, "processed", "stage_2")
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        # Ensure the output path is also stripped of any remote prefixes
        return self._strip_remote_prefix(base_path)

    def process_target(self, target: Dict[str, Any]) -> bool:
        """Process a single preprocessing target."""
        data_type = target['metadata']['data_type']
        processing_type = target['metadata']['processing_type']
        stage = target.get('stage')
        
        logger.info(f"Processing target: {data_type} - {processing_type} - {stage}")
        
        try:
            if stage == 'vector':
                if data_type == 'osm' and processing_type == 'simplify_polygons':
                    return self._process_osm_target(target)
                elif data_type == 'gadm' and processing_type == 'simplify_boundaries':
                    return self._process_gadm_target(target)
                elif data_type == 'country_classifications' and processing_type == 'classify_countries':
                    return self._process_country_classifications_target(target)
            elif stage == 'spatial':
                if data_type == 'osm' and processing_type == 'rasterize_land_mask':
                    return self._rasterize_osm_target(target)
                elif data_type == 'gadm' and processing_type == 'rasterize_countries':
                    return self._rasterize_gadm_target(target)
                elif data_type == 'country_classifications' and processing_type == 'rasterize_classifications':
                    return self._rasterize_country_classifications_target(target)
            else:
                logger.error(f"Unknown target type: {data_type} - {processing_type} - {stage}")
                return False
                
        except Exception as e:
            logger.exception(f"Error processing target {data_type}: {e}")
            return False
    
    def _resolve_source_file_path(self, file_path: str, subfolder: str = None) -> str:
        """
        Resolve the full path to a source file, prepending hpc_root and subfolder if necessary.
        """
        # If already absolute or starts with hpc_root, return as is
        if os.path.isabs(file_path) or (self.hpc_root and file_path.startswith(self.hpc_root)):
            return file_path
        # Otherwise, join with hpc_root
        if subfolder:
            return os.path.join(self.hpc_root, self.data_path, "raw", subfolder, file_path)
        else:
            return os.path.join(self.hpc_root, self.data_path, "raw", file_path)

    def _process_osm_target(self, target: Dict[str, Any]) -> bool:
        """Process OSM land polygons target."""
        # Use subfolder 'osm' for OSM files
        source_file_rel = os.path.basename(self._strip_remote_prefix(target['source_files'][0]))
        source_file = self._resolve_source_file_path(source_file_rel, subfolder="osm")
        output_path = self._strip_remote_prefix(target['output_path'])

        # Check for existence unless overwrite is True
        if not self.overwrite and os.path.exists(output_path):
            logger.info(f"Skipping OSM processing, output already exists: {output_path}")
            return True

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Extract and process OSM data
        extract_dir = os.path.join(self.temp_dir, "osm_extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extract zip file
        with zipfile.ZipFile(source_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find the shapefile
        shapefiles = list(Path(extract_dir).glob("**/*.shp"))
        if not shapefiles:
            raise Exception("No shapefiles found in OSM extract")
        
        shapefile_path = str(shapefiles[0])
        logger.info(f"Found OSM shapefile: {shapefile_path}")
        
        # Read and simplify
        gdf = gpd.read_file(shapefile_path, engine="pyogrio")
        logger.info(f"Simplifying {len(gdf)} OSM polygons with tolerance {self.simplify_tolerance}")
        
        gdf_simplified = gdf.copy()
        gdf_simplified['geometry'] = gdf_simplified.geometry.simplify(
            tolerance=self.simplify_tolerance,
            preserve_topology=True
        )
        
        # Save processed version
        gdf_simplified.to_file(output_path, driver="GPKG")
        
        logger.info(f"OSM processing complete: {output_path}")
        return True
    
    def _process_gadm_target(self, target: Dict[str, Any]) -> bool:
        """Process GADM boundaries target."""
        # Use subfolder 'gadm' for GADM files
        source_file_rel = os.path.basename(self._strip_remote_prefix(target['source_files'][0]))
        source_file = self._resolve_source_file_path(source_file_rel, subfolder="gadm")
        output_base = os.path.dirname(self._strip_remote_prefix(target['output_path']))

        # Check for existence unless overwrite is True
        # Check for all expected output files for each level
        expected_outputs = []
        # Try to find geopackage layers if possible, else fallback to previous logic
        if not self.overwrite and os.path.exists(output_base):
            # Check if any GADM level files exist
            existing_files = [f for f in os.listdir(output_base) if f.startswith('gadm_level') and f.endswith('_simplified.gpkg')]
            if existing_files:
                logger.info(f"Skipping GADM processing, outputs already exist in: {output_base}")
                return True

        # Ensure output directory exists
        os.makedirs(output_base, exist_ok=True)
        
        # Extract and process GADM data
        extract_dir = os.path.join(self.temp_dir, "gadm_extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            # Extract zip file
            with zipfile.ZipFile(source_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                
            # Find the geopackage
            geopackages = list(Path(extract_dir).glob("*.gpkg"))
            if not geopackages:
                logger.error("No geopackage found in GADM extract")
                return False
            
            geopackage_path = str(geopackages[0])
            logger.info(f"Found GADM geopackage: {geopackage_path}")
            
            # Get layers
            layers = gpd.list_layers(geopackage_path)
            
            # Process each level
            for level in layers.name.tolist():
                logger.info(f"Processing GADM level {level}")
                
                # Read the layer
                gdf = gpd.read_file(geopackage_path, engine="pyogrio", layer=level)

                gdf_simplified = gdf.copy()
                gdf_simplified['geometry'] = gdf_simplified.geometry.simplify(
                    tolerance=self.simplify_tolerance,
                    preserve_topology=True
                )
                
                # Save processed version
                output_path = f"{output_base}/gadm_level{level}_simplified.gpkg"
                gdf_simplified.to_file(output_path, driver="GPKG")
                
                logger.info(f"GADM level {level} processing complete: {output_path}")
            
            return True
            
        except Exception as e:
            logger.exception(f"Error processing GADM target: {e}")
            return False
        
    def _process_country_classifications_target(self, target: Dict[str, Any]) -> bool:
        """Process country classifications (HDI + World Bank income groups) target."""
        output_path = self._strip_remote_prefix(target['output_path'])

        # Check for existence unless overwrite is True
        if not self.overwrite and os.path.exists(output_path):
            logger.info(f"Skipping country classifications processing, output already exists: {output_path}")
            return True

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            has_hdi = target['metadata'].get('has_hdi', False)
            has_wb = target['metadata'].get('has_wb', False)
            
            result_df = None
            
            # Process HDI data if available
            if has_hdi:
                hdi_file = None
                for source_file in target['source_files']:
                    clean_path = self._strip_remote_prefix(source_file)
                    if 'hdi' in os.path.basename(clean_path).lower() or 'hdr' in os.path.basename(clean_path).lower():
                        hdi_file = self._resolve_source_file_path(os.path.basename(clean_path), subfolder="hdi")
                        break
                
                if hdi_file:
                    logger.info(f"Processing HDI data from {hdi_file}")
                    
                    # Define HDI columns for years 1990-2023
                    hdi_cols = [f"hdi_{y}" for y in range(1990, 2024)]
                    
                    # Read HDI data
                    hdi = pd.read_csv(
                        hdi_file,
                        encoding="latin1",
                        usecols=["iso3"] + hdi_cols,
                    )
                    
                    # Turn into long format
                    hdi = hdi.melt(id_vars=["iso3"], var_name="year", value_name="hdi")
                    
                    # Make year integer
                    hdi.loc[:, "year"] = hdi["year"].str[4:].astype(int)
                    
                    # Classify HDI values into UNDP categories
                    def assign_group(x) -> str:
                        """Classify a single HDI value into UNDP categories."""
                        if pd.isna(x):
                            return None
                        elif x < 0.550:
                            return "Low"
                        elif x < 0.700:
                            return "Medium"
                        elif x < 0.800:
                            return "High"
                        else:
                            return "Very High"
                    
                    logger.info("Classifying HDI values into groups")
                    hdi["hdi_group"] = hdi["hdi"].apply(assign_group)
                    
                    # Get modal value from 1990-2023
                    logger.info("Aggregating to modal HDI group per country")
                    hdi = (
                        hdi.groupby("iso3")["hdi_group"]
                        .agg(lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else None)
                        .reset_index()
                    )
                    
                    # Convert to boolean dummy variables
                    logger.info("Creating HDI class dummy variables")
                    hdi["HDI_ME"] = hdi.hdi_group.isin(["Medium"])
                    hdi["HDI_HI"] = hdi.hdi_group.isin(["High"])
                    hdi["HDI_VH"] = hdi.hdi_group.isin(["Very High"])
                    
                    # Drop string classification
                    hdi.drop(columns=["hdi_group"], inplace=True)
                    
                    result_df = hdi
                    logger.info(f"HDI processing complete: {len(hdi)} countries classified")
            
            # Process World Bank income group data if available
            if has_wb:
                wb_file = None
                for source_file in target['source_files']:
                    clean_path = self._strip_remote_prefix(source_file)
                    if 'dr0095334' in os.path.basename(clean_path).lower() or 'world_bank' in os.path.basename(clean_path).lower():
                        wb_file = self._resolve_source_file_path(os.path.basename(clean_path), subfolder="hdi")
                        break
                
                if wb_file:
                    logger.info(f"Processing World Bank income groups from {wb_file}")
                    
                    # Read World Bank data
                    wb = pd.read_excel(
                        wb_file, 
                        sheet_name="Country Analytical History", 
                        header=4
                    )
                    
                    # Rename country code column
                    wb.rename({wb.columns[0]: "iso3"}, axis=1, inplace=True)
                    
                    # Drop full country names
                    wb.drop(columns=[wb.columns[1]], inplace=True)
                    
                    # Cut head and tail
                    wb = wb.iloc[6:-2,]
                    
                    # Melt to long format
                    wb = wb.melt(id_vars=["iso3"], var_name="year", value_name="classification")
                    
                    # Mutate year to int
                    wb.loc[:, "year"] = wb.year.str[2:].apply(
                        lambda x: int("19" + x) if int(x) > 50 else int("20" + x)
                    )
                    
                    # Drop NAs
                    wb = wb.query("classification!='..'").dropna()
                    
                    # Get modal classification
                    wb = wb.dropna().groupby("iso3").classification.agg(pd.Series.mode).reset_index()
                    
                    # Convert to booleans
                    logger.info("Creating World Bank income group dummy variables")
                    wb["WB_LM"] = wb.classification.isin(["LM", "LM*"])
                    wb["WB_UM"] = wb.classification.isin(["UM"])
                    wb["WB_HI"] = wb.classification.isin(["H"])
                    
                    # Drop string classification
                    wb.drop(columns=["classification"], inplace=True)
                    
                    logger.info(f"World Bank processing complete: {len(wb)} countries classified")
                    
                    # Merge with HDI if both available
                    if result_df is not None:
                        result_df = result_df.merge(wb, on="iso3", how="outer")
                        logger.info("Merged HDI and World Bank classifications")
                    else:
                        result_df = wb
            
            if result_df is None:
                logger.error("No data to process")
                return False
            
            # Save to parquet
            logger.info(f"Saving country classifications to {output_path}")
            result_df.to_parquet(output_path, index=False)
            
            logger.info(f"Country classifications processing complete: {len(result_df)} countries")
            return True
            
        except Exception as e:
            logger.exception(f"Error processing country classifications target: {e}")
            return False

    def _rasterize_osm_target(self, target: Dict[str, Any]) -> bool:
        """Rasterize OSM land polygons target."""
        try:
            from odc.geo.xr import rasterize
            from odc.geo.geom import Geometry
            import shapely
            import xarray as xr
            import zarr

            source_file = self._strip_remote_prefix(target['source_files'][0])
            output_path = self._strip_remote_prefix(target['output_path'])

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Load the source vector file
            gdf = gpd.read_file(source_file, engine="pyogrio")
            logger.info(f"Loaded {len(gdf)} polygons from {source_file}")

            geobox = self._get_or_create_geobox()

            logger.info("Creating MultiPolygon from all land geometries")
            land_polygons = shapely.MultiPolygon(gdf.geometry.tolist())
            geom = Geometry(land_polygons, crs=str(gdf.crs))

            logger.info(f"Rasterizing land polygons to grid of shape {geobox.shape}")
            land_mask = rasterize(geom, geobox)
            
            land_mask.coords['latitude'] = land_mask.coords['latitude'].values.round(5)
            land_mask.coords['longitude'] = land_mask.coords['longitude'].values.round(5)
                            
            ds = xr.Dataset(
                    data_vars={'land_mask': land_mask},
                    attrs={
                        'description': 'Land/water mask (1=land, 0=water)',
                        'source': 'OpenStreetMap land polygons',
                        'date_created': datetime.now().isoformat(),
                        'crs': str(geobox.crs)
                    }
                )

            compressor = BloscCodec(cname="zstd", clevel=3, shuffle='bitshuffle', blocksize=0)
            encoding = {'land_mask': {'compressors': (compressor,)}}

            logger.info(f"Writing land mask to zarr file at {output_path}")
            ds.to_zarr(output_path, encoding=encoding, mode="w")

            logger.info("OSM land mask rasterization complete")
            return True

        except Exception as e:
            logger.exception(f"Error in OSM rasterization: {e}")
            return False

    def _rasterize_gadm_target(self, target: Dict[str, Any]) -> bool:
        """Rasterize GADM countries and subdivisions target using tiling approach."""
        try:
            from odc.geo.xr import rasterize
            from odc.geo.geom import Geometry
            from odc.geo import GeoboxTiles
            import xarray as xr
            import zarr
            import numpy as np
            import shapely

            source_file = self._strip_remote_prefix(target['source_files'][0])
            output_path = self._strip_remote_prefix(target['output_path'])
            output_dir = os.path.dirname(output_path)

            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Find both ADM_0 and ADM_1 files
            gadm_dir = os.path.dirname(source_file)
            adm0_file = os.path.join(gadm_dir, "gadm_levelADM_0_simplified.gpkg")
            adm1_file = os.path.join(gadm_dir, "gadm_levelADM_1_simplified.gpkg")
            
            if not os.path.exists(adm0_file):
                logger.error(f"ADM_0 file not found: {adm0_file}")
                return False
            
            adm1_exists = os.path.exists(adm1_file)
            if not adm1_exists:
                logger.warning(f"ADM_1 file not found: {adm1_file}, will only process ADM_0")

            # Load the source vector files and prepare geometries
            gdf_adm0 = gpd.read_file(adm0_file, engine="pyogrio")
            logger.info(f"Loaded {len(gdf_adm0)} country polygons from {adm0_file}")

            # Prepare country codes and mapping
            country_codes = sorted(gdf_adm0['GID_0'].unique())
            country_code_to_id = {code: i+1 for i, code in enumerate(country_codes)}

            # Load ADM_1 if available
            gdf_adm1 = None
            subdivision_code_to_id = {}
            if adm1_exists:
                gdf_adm1 = gpd.read_file(adm1_file, engine="pyogrio")
                logger.info(f"Loaded {len(gdf_adm1)} subdivision polygons from {adm1_file}")
                
                # Create subdivision mapping
                subdivision_codes = sorted(gdf_adm1['GID_1'].unique())
                subdivision_code_to_id = {code: i+1 for i, code in enumerate(subdivision_codes)}

            # Use DaskClientContextManager as a context manager
            with self._initialize_dask_client() as client:
                try:
                    dashboard_link = getattr(client, "dashboard_link", None)
                    if dashboard_link:
                        logger.info(f"Created Dask client for GADM rasterization: {dashboard_link}")

                    geobox = self._get_or_create_geobox()
                    logger.info(f"Using geobox with shape: {geobox.shape}")

                    # Create tiles for processing
                    tile_size = 2048  # Adjust based on memory constraints
                    tiles = GeoboxTiles(geobox, (tile_size, tile_size))
                    logger.info(f"Created {tiles.shape[0]}x{tiles.shape[1]} tiles of size {tile_size}")

                    # Step 1: Create empty zarr file with target dimensions
                    if not self._create_empty_gadm_zarr(output_path, geobox, gdf_adm1 is not None):
                        return False

                    # Step 2: Process tiles iteratively
                    success = self._process_gadm_tiles(
                        tiles, output_path, geobox, gdf_adm0, gdf_adm1, 
                        country_code_to_id, subdivision_code_to_id
                    )

                    if not success:
                        return False

                except Exception as e:
                    logger.exception(f"Error in GADM rasterization (Dask context): {e}")
                    return False

            # Save mapping files in output directory
            country_mapping_file = os.path.join(output_dir, "country_code_mapping.json")
            with open(country_mapping_file, 'w') as f:
                json.dump(country_code_to_id, f, indent=2)
                
            if subdivision_code_to_id:
                subdivision_mapping_file = os.path.join(output_dir, "subdivision_code_mapping.json")
                with open(subdivision_mapping_file, 'w') as f:
                    json.dump(subdivision_code_to_id, f, indent=2)

            logger.info("GADM rasterization complete")
            return True

        except Exception as e:
            logger.exception(f"Error in GADM rasterization: {e}")
            return False

    def _create_empty_gadm_zarr(self, output_path: str, geobox, include_subdivisions: bool) -> bool:
        """Create empty zarr file with target dimensions for GADM data."""
        try:
            import xarray as xr
            import dask.array as da
            import numpy as np
            from zarr.codecs import BloscCodec

            logger.info("Creating empty GADM zarr file")
            
            # Get dimensions
            ny, nx = geobox.shape
            lat_coords = geobox.coords['latitude'].values.round(5)
            lon_coords = geobox.coords['longitude'].values.round(5)

            # Create data variables
            data_vars = {}
            
            # Countries variable (ADM_0)
            data_vars['countries'] = xr.DataArray(
                da.zeros((ny, nx), dtype=np.uint16, chunks=(512, 512)),
                dims=['latitude', 'longitude'],
                coords={
                    'latitude': lat_coords,
                    'longitude': lon_coords
                },
                attrs={
                    'description': 'Country ID grid (0=no country)',
                    '_FillValue': 0
                }
            )
            
            # Subdivisions variable (ADM_1) if available
            if include_subdivisions:
                data_vars['subdivisions'] = xr.DataArray(
                    da.zeros((ny, nx), dtype=np.uint16, chunks=(512, 512)),
                    dims=['latitude', 'longitude'],
                    coords={
                        'latitude': lat_coords,
                        'longitude': lon_coords
                    },
                    attrs={
                        'description': 'Subdivision ID grid (0=no subdivision)',
                        '_FillValue': 0
                    }
                )

            # Create dataset
            dataset_attrs = {
                'description': 'GADM administrative boundaries grid',
                'source': 'GADM administrative boundaries',
                'date_created': datetime.now().isoformat(),
                'crs': str(geobox.crs),
                'levels_included': 'ADM_0 (countries)' + (' and ADM_1 (subdivisions)' if include_subdivisions else '')
            }
            
            ds = xr.Dataset(data_vars, attrs=dataset_attrs)
            ds = ds.rio.write_crs(geobox.crs)

            # Setup encoding
            compressor = BloscCodec(cname="zstd", clevel=3, shuffle='bitshuffle', blocksize=0)
            encoding = {}
            for var_name in data_vars.keys():
                encoding[var_name] = {
                    "chunks": (512, 512), 
                    "compressors": compressor,
                    "dtype": "uint16"
                }

            # Write empty zarr structure (compute=False)
            logger.info(f"Writing empty GADM zarr structure to: {output_path}")
            ds.to_zarr(
                output_path, 
                mode="w",
                encoding=encoding,
                compute=False,
                consolidated=False
            )
            
            logger.info("Empty GADM zarr created successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Error creating empty GADM zarr: {e}")
            return False

    def _process_gadm_tiles(self, tiles, output_path: str, geobox, gdf_adm0, gdf_adm1, 
                           country_code_to_id: dict, subdivision_code_to_id: dict) -> bool:
        """Process GADM tiles iteratively with overlap detection."""
        try:
            from odc.geo.xr import rasterize
            from odc.geo.geom import Geometry
            import xarray as xr
            import numpy as np
            import shapely.geometry

            total_tiles = tiles.shape[0] * tiles.shape[1]
            processed_tiles = 0

            # Process each tile
            for ix in range(tiles.shape[0]):
                for iy in range(tiles.shape[1]):
                    try:
                        # Get the tile geobox
                        tile_geobox = tiles[ix, iy]
                        tile_bounds = tile_geobox.boundingbox
                        
                        # Create tile polygon for intersection testing
                        tile_polygon = shapely.geometry.box(
                            tile_bounds.left, tile_bounds.bottom,
                            tile_bounds.right, tile_bounds.top
                        )

                        # Find overlapping geometries for ADM_0
                        overlapping_countries = gdf_adm0[gdf_adm0.geometry.intersects(tile_polygon)]
                        
                        # Find overlapping geometries for ADM_1 if available
                        overlapping_subdivisions = None
                        if gdf_adm1 is not None:
                            overlapping_subdivisions = gdf_adm1[gdf_adm1.geometry.intersects(tile_polygon)]

                        # Skip tile if no overlapping geometries
                        if len(overlapping_countries) == 0 and (gdf_adm1 is None or len(overlapping_subdivisions) == 0):
                            processed_tiles += 1
                            continue

                        logger.debug(f"Processing tile [{ix}, {iy}] with {len(overlapping_countries)} countries" + 
                                   (f" and {len(overlapping_subdivisions)} subdivisions" if overlapping_subdivisions is not None else ""))

                        # Create empty arrays for this tile
                        tile_shape = tile_geobox.shape
                        countries_tile = np.zeros(tile_shape, dtype=np.uint16)
                        subdivisions_tile = np.zeros(tile_shape, dtype=np.uint16) if gdf_adm1 is not None else None

                        # Rasterize overlapping countries
                        for _, row in overlapping_countries.iterrows():
                            code = row['GID_0']
                            value = country_code_to_id[code]
                            
                            geom = Geometry(row.geometry, crs=str(gdf_adm0.crs))
                            country_mask = rasterize(geom, tile_geobox)
                            countries_tile = np.where(country_mask, value, countries_tile)

                        # Rasterize overlapping subdivisions if available
                        if overlapping_subdivisions is not None and len(overlapping_subdivisions) > 0:
                            for _, row in overlapping_subdivisions.iterrows():
                                code = row['GID_1']
                                value = subdivision_code_to_id[code]
                                
                                geom = Geometry(row.geometry, crs=str(gdf_adm1.crs))
                                subdivision_mask = rasterize(geom, tile_geobox)
                                subdivisions_tile = np.where(subdivision_mask, value, subdivisions_tile)

                        # Create dataset for this tile
                        tile_data_vars = {}
                        
                        tile_data_vars['countries'] = xr.DataArray(
                            countries_tile,
                            dims=['latitude', 'longitude'],
                            coords={
                                'latitude': tile_geobox.coords['latitude'].values.round(5),
                                'longitude': tile_geobox.coords['longitude'].values.round(5)
                            }
                        )
                        
                        if subdivisions_tile is not None:
                            tile_data_vars['subdivisions'] = xr.DataArray(
                                subdivisions_tile,
                                dims=['latitude', 'longitude'],
                                coords={
                                    'latitude': tile_geobox.coords['latitude'].values.round(5),
                                    'longitude': tile_geobox.coords['longitude'].values.round(5)
                                }
                            )

                        tile_ds = xr.Dataset(tile_data_vars)

                        # Write tile to zarr region
                        tile_ds.to_zarr(
                            output_path,
                            region='auto',
                            mode='r+',
                            consolidated=False
                        )

                        processed_tiles += 1
                        
                        if processed_tiles % 100 == 0:
                            logger.info(f"Processed {processed_tiles}/{total_tiles} tiles")
                            
                    except Exception as e:
                        logger.warning(f"Error processing tile [{ix}, {iy}]: {e}")
                        processed_tiles += 1
                        continue

            logger.info(f"Completed processing all {total_tiles} tiles")
            return True
            
        except Exception as e:
            logger.exception(f"Error processing GADM tiles: {e}")
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

    def _get_or_create_geobox(self):
        """Get or create geobox using the common utility."""
        misc_level1_dir = os.path.join(self.get_hpc_output_path('vector'), "misc")
        return get_or_create_geobox(self.hpc_root, misc_level1_dir)

    def finalize(self) -> None:
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {e}")
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MiscPreprocessor':
        """Create an instance from configuration dictionary."""
        return cls(**config)

    def _rasterize_country_classifications_target(self, target: Dict[str, Any]) -> bool:
        """Rasterize country classifications by mapping to GADM country grid."""
        try:
            import xarray as xr
            import numpy as np
            from zarr.codecs import BloscCodec

            # Get input paths
            classifications_parquet = self._strip_remote_prefix(target['source_files'][0])
            gadm_zarr = self._strip_remote_prefix(target['source_files'][1])
            output_path = self._strip_remote_prefix(target['output_path'])

            # Check for existence unless overwrite is True
            if not self.overwrite and os.path.exists(output_path):
                logger.info(f"Skipping country classifications rasterization, output already exists: {output_path}")
                return True

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Load the country classifications table
            logger.info(f"Loading country classifications from {classifications_parquet}")
            classifications_df = pd.read_parquet(classifications_parquet)
            
            # Load the GADM countries grid
            logger.info(f"Loading GADM countries grid from {gadm_zarr}")
            countries_grid = xr.open_zarr(gadm_zarr, chunks="auto", consolidated=False)
            countries_grid = countries_grid.countries.astype("int16").compute()
            
            # Load the country code mapping
            gadm_dir = os.path.dirname(gadm_zarr)
            country_mapping_file = os.path.join(gadm_dir, "country_code_mapping.json")
            
            if not os.path.exists(country_mapping_file):
                logger.error(f"Country mapping file not found: {country_mapping_file}")
                return False
            
            with open(country_mapping_file, 'r') as f:
                country_code_to_id = json.load(f)
            
            # Merge classifications with country codes
            logger.info("Merging classifications with country IDs")
            classifications_df['country_id'] = classifications_df['iso3'].map(
                lambda x: country_code_to_id.get(x, 0)
            )
            
            # Get classification columns (all except iso3 and country_id)
            classification_cols = [col for col in classifications_df.columns 
                                  if col not in ['iso3', 'country_id']]
            
            logger.info(f"Creating grids for {len(classification_cols)} classification variables")
            
            # Create grids for each classification variable using efficient isin() method
            classification_arrays = {}
            for col in classification_cols:
                logger.info(f"Processing classification variable: {col}")
                
                # Get country IDs where this classification is True
                country_ids_with_classification = classifications_df.loc[
                    ((classifications_df[col] == True) & classifications_df["country_id"] != 0), 
                    "country_id"
                ].unique()
                
                # Use isin to create boolean mask efficiently
                classification_array = countries_grid.isin(country_ids_with_classification)
                
                # Add attributes
                classification_array.attrs = {
                    'description': f'{col} classification grid (True/False)',
                    '_FillValue': -1
                }
                
                classification_arrays[col] = classification_array
            
            # Create dataset
            logger.info("Creating output dataset")
            ds = xr.Dataset(
                classification_arrays,
                attrs={
                    'description': 'Country classifications grid (HDI and World Bank income groups)',
                    'source': 'UNDP HDI and World Bank income classifications',
                    'date_created': datetime.now().isoformat(),
                    'crs': str(countries_grid.attrs.get('crs', 'EPSG:4326')),
                    'note': 'Boolean values: True where classification applies, False otherwise'
                }
            )
            
            # Add CRS
            if 'crs' in countries_grid.attrs:
                ds = ds.rio.write_crs(countries_grid.attrs['crs'])
            
            # Setup encoding - use bool dtype for minimal storage
            compressor = BloscCodec(cname="lz4", clevel=5, shuffle='bitshuffle', blocksize=0)
            encoding = {}
            for var_name in classification_arrays.keys():
                encoding[var_name] = {
                    "chunks": (512, 512),
                    "compressors": compressor,
                    "dtype": "bool"
                }
            
            # Write to zarr
            logger.info(f"Writing country classifications grid to {output_path}")
            ds.to_zarr(output_path, zarr_format=3, consolidated=False, encoding=encoding, mode="w")
            
            logger.info("Country classifications rasterization complete")
            
            # Close datasets
            countries_grid.close()
            
            return True
            
        except Exception as e:
            logger.exception(f"Error in country classifications rasterization: {e}")
            return False
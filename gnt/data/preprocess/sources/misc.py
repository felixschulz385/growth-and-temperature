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

from gnt.data.preprocess.sources.base import AbstractPreprocessor
from gnt.data.download.sources.misc import MiscDataSource

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
                files (List[Dict]): Required - List of file configurations
                output_path (str): Required - Path for processed outputs
                local_index_dir (str): Required - Directory for parquet index
                hpc_output_base (str): Base path for HPC outputs
                simplify_tolerance (float): Tolerance for polygon simplification
                rasterize (bool): Whether to create raster versions
                temp_dir (str): Directory for temporary files
                dask_threads (int): Number of Dask threads
                dask_memory_limit (int): Dask memory limit in GB
                grid_resolution (float): Grid resolution for rasterization
                grid_bounds (List[float]): Grid bounds [west, south, east, north]
        """
        super().__init__(**kwargs)
        
        # Required parameters
        files_config = kwargs.get('files')
        # Use output_path if provided, else default to "misc"
        self.data_path = kwargs.get('output_path') or "misc"
        hpc_target = kwargs.get('hpc_target')
        # Strip remote prefix from hpc_target if present
        self.hpc_root = self._strip_remote_prefix(hpc_target)
        if not self.hpc_root:
            raise ValueError("hpc.target is required for HPC mode")
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

        logger.info(f"Initialized MiscPreprocessor in HPC mode")
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
                files=files_config,
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
            stage: Processing stage ('simplify' or 'rasterize')  
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
            
            if stage == 'simplify':
                return self._generate_processing_targets(file_paths)
            elif stage == 'rasterize':
                return self._generate_rasterization_targets()
            else:
                raise ValueError(f"Unknown stage: {stage}")
                
        except Exception as e:
            logger.error(f"Error reading parquet index: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return []

    def _generate_processing_targets(self, files: List[str]) -> List[Dict]:
        """Generate initial processing targets for misc data."""
        targets = []
        
        # Group files by data type
        osm_files = []
        gadm_files = []
        
        for file_path in files:
            # Strip remote prefix from file paths
            clean_file_path = self._strip_remote_prefix(file_path)
            filename = os.path.basename(clean_file_path)
            
            if any(pattern in filename.lower() for pattern in ['osm', 'land-polygons']):
                osm_files.append(clean_file_path)
            elif 'gadm' in filename.lower():
                gadm_files.append(clean_file_path)
        
        logger.info(f"Categorized files: {len(osm_files)} OSM, {len(gadm_files)} GADM")
        
        # OSM land polygons target
        if osm_files:
            source_file = osm_files[0]
            target = {
                'data_type': 'osm',
                'stage': 'simplify',
                'source_files': [source_file],
                'output_path': f"{self.get_hpc_output_path('simplify')}/osm/land_polygons_simplified.gpkg",
                'dependencies': [],
                'metadata': {
                    'data_type': 'osm',
                    'processing_type': 'simplify_polygons',
                    'source_count': len(osm_files),
                    'source_file': source_file
                }
            }
            targets.append(target)
        
        # GADM boundaries target  
        if gadm_files:
            source_file = gadm_files[0]
            target = {
                'data_type': 'gadm',
                'stage': 'simplify',
                'source_files': [source_file],
                'output_path': f"{self.get_hpc_output_path('simplify')}/gadm/gadm_levels_simplified.gpkg",
                'dependencies': [],
                'metadata': {
                    'data_type': 'gadm',
                    'processing_type': 'simplify_boundaries',
                    'source_count': len(gadm_files),
                    'source_file': source_file
                }
            }
            targets.append(target)
        
        return targets
    
    def _generate_rasterization_targets(self) -> List[Dict]:
        """Generate rasterization targets for processed vector data."""
        targets = []
        
        if not self.rasterize:
            return targets
        
        # OSM land mask rasterization
        osm_vector_path = f"{self.get_hpc_output_path('simplify')}/osm/land_polygons_simplified.gpkg"
        if os.path.exists(osm_vector_path):
            target = {
                'data_type': 'osm_raster',
                'stage': 'rasterize',
                'source_files': [osm_vector_path],
                'output_path': f"{self.get_hpc_output_path('rasterize')}/osm/land_mask.zarr",
                'dependencies': [osm_vector_path],
                'metadata': {
                    'data_type': 'osm_raster',
                    'processing_type': 'rasterize_land_mask'
                }
            }
            targets.append(target)
        
        # GADM countries grid rasterization  
        gadm_vector_path = f"{self.get_hpc_output_path('simplify')}/gadm/gadm_level0_simplified.gpkg"
        if os.path.exists(gadm_vector_path):
            target = {
                'data_type': 'gadm_raster',
                'stage': 'rasterize',
                'source_files': [gadm_vector_path],
                'output_path': f"{self.get_hpc_output_path('rasterize')}/gadm/countries_grid.zarr",
                'dependencies': [gadm_vector_path],
                'metadata': {
                    'data_type': 'gadm_raster',
                    'processing_type': 'rasterize_countries'
                }
            }
            targets.append(target)
        
        return targets
    
    def get_hpc_output_path(self, stage: str) -> str:
        """Get output path for a given stage, relative to the HPC root."""
        if stage == "simplify":
            base_path = os.path.join(self.hpc_root, self.data_path, "processed", "stage_0")
        elif stage == "rasterize":
            base_path = os.path.join(self.hpc_root, self.data_path, "processed", "stage_2")
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        # Ensure the output path is also stripped of any remote prefixes
        return self._strip_remote_prefix(base_path)

    def process_target(self, target: Dict[str, Any]) -> bool:
        """Process a single preprocessing target."""
        data_type = target['metadata']['data_type']
        processing_type = target['metadata']['processing_type']
        
        logger.info(f"Processing target: {data_type} - {processing_type}")
        
        try:
            if data_type == 'osm' and processing_type == 'simplify_polygons':
                return self._process_osm_target(target)
            elif data_type == 'gadm' and processing_type == 'simplify_boundaries':
                return self._process_gadm_target(target)
            elif data_type == 'osm_raster' and processing_type == 'rasterize_land_mask':
                return self._rasterize_osm_target(target)
            elif data_type == 'gadm_raster' and processing_type == 'rasterize_countries':
                return self._rasterize_gadm_target(target)
            else:
                logger.error(f"Unknown target type: {data_type} - {processing_type}")
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
        if output_base and os.path.exists(output_base):
            expected_outputs = [Path(output_base) / f"gadm_levelADM_{i}_simplified.gpkg" for i in range(1, 6)]
        if not self.overwrite and expected_outputs and all(os.path.exists(str(p)) for p in expected_outputs):
            logger.info(f"Skipping GADM processing, all outputs already exist in: {output_base}")
            return True

        # Ensure output directory exists
        os.makedirs(output_base, exist_ok=True)
        
        # Extract and process GADM data
        extract_dir = os.path.join(self.temp_dir, "gadm_extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extract zip file
        with zipfile.ZipFile(source_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        # Find the geopackage
        geopackages = list(Path(extract_dir).glob("*.gpkg"))
        if not geopackages:
            raise Exception("No geopackage found in GADM extract")
        
        geopackage_path = str(geopackages[0])
        logger.info(f"Found GADM geopackage: {geopackage_path}")
        
        # Get layers
        layers = gpd.list_layers(geopackage_path)
        
        # Process each level
        for level in layers.name.tolist():
            logger.info(f"Processing GADM level {level}: {geopackage_path}")
            
            # Read the shapefile
            gdf = gpd.read_file(geopackage_path, engine="pyogrio", layer = level)

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
                            
            ds = xr.Dataset(
                    data_vars={'land_mask': land_mask},
                    attrs={
                        'description': 'Land/water mask (1=land, 0=water)',
                        'source': 'OpenStreetMap land polygons',
                        'date_created': datetime.now().isoformat(),
                        'crs': str(geobox.crs)
                    }
                )

            compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
            encoding = {'land_mask': {'compressor': compressor}}

            logger.info(f"Writing land mask to zarr file at {output_path}")
            ds.to_zarr(output_path, encoding=encoding, consolidated=True, mode="w")

            logger.info("OSM land mask rasterization complete")
            return True

        except Exception as e:
            logger.exception(f"Error in OSM rasterization: {e}")
            return False

    def _rasterize_gadm_target(self, target: Dict[str, Any]) -> bool:
        """Rasterize GADM countries target."""
        try:
            from odc.geo.xr import rasterize
            from odc.geo.geom import Geometry
            import xarray as xr
            import zarr
            import numpy as np

            source_file = self._strip_remote_prefix(target['source_files'][1])
            output_path = self._strip_remote_prefix(target['output_path'])
            output_dir = os.path.dirname(output_path)

            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Load the source vector file (should be gadm_level0_simplified.gpkg)
            gdf = gpd.read_file(source_file, engine="pyogrio")
            logger.info(f"Loaded {len(gdf)} country polygons from {source_file}")

            if 'GID_0' in gdf.columns:
                country_codes = sorted(gdf['GID_0'].unique())
            else:
                country_codes = [str(i) for i in range(len(gdf))]

            code_to_id = {code: i+1 for i, code in enumerate(country_codes)}

            # Use DaskClientContextManager as a context manager
            with self._initialize_dask_client() as client:
                try:
                    dashboard_link = getattr(client, "dashboard_link", None)
                    if dashboard_link:
                        logger.info(f"Created Dask client for GADM rasterization: {dashboard_link}")

                    geobox = self._get_or_create_geobox()

                    logger.info(f"Creating empty countries grid of shape {geobox.shape}")
                    countries_grid = xr.DataArray(
                        data=np.zeros(geobox.shape, dtype=np.uint16),
                        dims=['y', 'x'],
                        coords={
                            'y': geobox.coords['y'],
                            'x': geobox.coords['x']
                        },
                        attrs={'crs': str(geobox.crs)}
                    ).chunk(2000)

                    logger.info(f"Rasterizing {len(gdf)} country boundaries")
                    for idx, row in gdf.iterrows():
                        if idx % 10 == 0:
                            logger.info(f"Rasterizing country {idx+1}/{len(gdf)}")

                        if 'GID_0' in gdf.columns:
                            code = row['GID_0']
                        else:
                            code = str(idx)

                        value = code_to_id[code]

                        geom = Geometry(row.geometry, crs=str(gdf.crs))
                        country_mask = rasterize(geom, geobox, dtype='uint16', fill=0, value=value, chunks=2000)
                        countries_grid = xr.where(country_mask > 0, country_mask, countries_grid)

                    ds = xr.Dataset(
                        data_vars={'countries': countries_grid},
                        attrs={
                            'description': 'Country ID grid (0=no country)',
                            'source': 'GADM administrative boundaries',
                            'date_created': datetime.now().isoformat(),
                            'crs': str(geobox.crs)
                        }
                    )

                    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
                    encoding = {'countries': {'compressor': compressor}}

                    logger.info(f"Writing countries grid to zarr file at {output_path}")
                    ds.to_zarr(output_path, encoding=encoding, consolidated=True)

                except Exception as e:
                    logger.exception(f"Error in GADM countries rasterization (Dask context): {e}")
                    return False

            # Save mapping file in output directory
            mapping_file = os.path.join(output_dir, "country_code_mapping.json")
            with open(mapping_file, 'w') as f:
                json.dump(code_to_id, f, indent=2)

            logger.info("GADM countries rasterization complete")
            return True

        except Exception as e:
            logger.exception(f"Error in GADM countries rasterization: {e}")
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
        """
        Extract the geobox from a successful EOG VIIRS download and save it to a file
        in the misc level 0 directory. If the file is gzipped, unpack it first.
        Returns:
            geobox: The extracted geobox object.
        Raises:
            RuntimeError: If no successful VIIRS download is found.
        """
        import gzip
        import shutil

        # Store the pickle to the misc level 0 directory
        misc_level0_dir = os.path.join(self.get_hpc_output_path('simplify'), "misc")
        os.makedirs(misc_level0_dir, exist_ok=True)
        geobox_local = os.path.join(misc_level0_dir, "viirs_geobox.pkl")

        # If geobox already exists, load and return it
        if os.path.exists(geobox_local):
            logger.info(f"Geobox pickle already exists, loading from {geobox_local}")
            with open(geobox_local, 'rb') as f:
                geobox = pickle.load(f)
            return geobox

        parquet_path = os.path.join(self.hpc_root, "hpc_data_index/parquet_eog_viirs.parquet")
        if not os.path.exists(parquet_path):
            raise RuntimeError(f"Parquet index not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        # Try to find a successful download
        if 'status_category' in df.columns:
            ok = df[df['status_category'] == 'completed']
        elif 'download_status' in df.columns:
            ok = df[df['download_status'] == 'completed']
        else:
            raise RuntimeError("No status column found in parquet index")

        if ok.empty:
            raise RuntimeError("No successful VIIRS download found in index")

        # Use the first successful file
        viirs_local = os.path.join(self.hpc_root, "eog/viirs/raw", ok.iloc[0]['relative_path'])
        if not os.path.exists(viirs_local):
            raise RuntimeError(f"VIIRS file does not exist: {viirs_local}")

        # If the file is gzipped, unpack it to a temp location
        if viirs_local.endswith(".gz"):
            unpacked_path = viirs_local[:-3]
            if not os.path.exists(unpacked_path):
                logger.info(f"Unpacking gzipped VIIRS file: {viirs_local} -> {unpacked_path}")
                with gzip.open(viirs_local, 'rb') as f_in, open(unpacked_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            viirs_to_open = unpacked_path
        else:
            viirs_to_open = viirs_local

        # Open the file and extract the geobox
        viirs_data = rxr.open_rasterio(viirs_to_open, chunks="auto")
        geobox = viirs_data.odc.geobox

        with open(geobox_local, 'wb') as f:
            pickle.dump(geobox, f)
        logger.info(f"Saved geobox to {geobox_local}")

        return geobox

    def summarize_annual_means(self) -> None:
        """Implementation of abstract method - not applicable for misc data."""
        logger.warning("summarize_annual_means not applicable for misc data preprocessor")
    
    def project_to_unified_grid(self) -> None:
        """Implementation of abstract method - handled in single-stage workflow."""
        logger.info("Grid projection included in single-stage workflow")
    
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
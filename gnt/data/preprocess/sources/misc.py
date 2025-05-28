import os
import tempfile
import logging
import zipfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import geopandas as gpd
from datetime import datetime
import json
import re

from gnt.data.preprocess.sources.base import AbstractPreprocessor
from gnt.data.common.gcs.client import GCSClient
from gnt.data.common.index.preprocessing_index import PreprocessingIndex

logger = logging.getLogger(__name__)

class MiscPreprocessor(AbstractPreprocessor):
    """
    Simplified preprocessor for auxiliary boundary data.
    
    This class handles preprocessing of auxiliary data sources like:
    - OpenStreetMap land polygons
    - GADM administrative boundaries
    
    It uses a single-stage workflow to:
    1. Extract data from source files
    2. Simplify geometries and standardize formats
    3. Create derived products (like rasterized versions)
    """
    
    # Default grid parameters
    DEFAULT_GRID_RESOLUTION = 0.008333  # ~1km at equator
    DEFAULT_CRS = "EPSG:4326"           # WGS84
    
    def __init__(self, **kwargs):
        """
        Initialize the misc data preprocessor.
        
        Args:
            **kwargs: Configuration parameters including:
                bucket_name (str): GCS bucket name
                output_path (str): Path for processed outputs
                simplify_tolerance (float): Tolerance for polygon simplification
                rasterize (bool): Whether to create raster versions
                version (str): Processing version
                temp_dir (str): Directory for temporary files
        """
        super().__init__(**kwargs)
        
        # Settings - match EOG style
        self.bucket_name = kwargs.get('bucket_name', 'growthandheat')
        self.output_path = kwargs.get('output_path', 'auxiliary/boundaries').rstrip('/')
        
        # Processing parameters
        self.simplify_tolerance = kwargs.get('simplify_tolerance', 0.001)  # ~100m at equator
        self.rasterize = kwargs.get('rasterize', True)
        self.grid_resolution = kwargs.get('grid_resolution', self.DEFAULT_GRID_RESOLUTION)
        self.grid_bounds = kwargs.get('grid_bounds', [-180, -90, 180, 90])  # Global by default
        
        # Setup
        self.version = kwargs.get('version', 'v1')
        self.temp_dir = self._setup_temp_dir(kwargs.get('temp_dir'))
        
        # Initialize GCS client
        self.gcs_client = GCSClient(self.bucket_name)
        
        # Initialize preprocessing index
        self.preprocessing_index = PreprocessingIndex(
            bucket_name=self.bucket_name,
            data_path=self.output_path,  # Use output_path for consistency
            version=self.version,
            temp_dir=self.temp_dir
        )
        
        # Fixed data source configurations - using consistent path structure
        self.data_sources = {
            'osm': {
                'source': f"{self.output_path}/osm/land-polygons-complete-4326.zip",
                'extract_dir': os.path.join(self.temp_dir, "osm_extracted"),
                'output_dir': f"{self.output_path}/osm"  # Use annual_path from index
            },
            'gadm': {
                'source': f"{self.output_path}/gadm/gadm_410-levels.zip",
                'extract_dir': os.path.join(self.temp_dir, "gadm_extracted"),
                'output_dir': f"{self.output_path}/gadm"  # Use annual_path from index
            }
        }
        
        # Dask configuration
        self.dask_threads = kwargs.get('dask_threads', None)  # None means use all available cores
        self.dask_memory_limit = kwargs.get('dask_memory_limit', None)  # None means 75% of system memory
        
        logger.info(f"Initialized simplified MiscPreprocessor for boundary data")
        
    def _setup_temp_dir(self, temp_dir):
        """Set up temporary directory."""
        if not temp_dir:
            temp_dir = tempfile.mkdtemp(prefix="misc_processor_")
        else:
            os.makedirs(temp_dir, exist_ok=True)
        return temp_dir
    
    def preprocess(self) -> bool:
        """
        Main entry point for preprocessing. Processes all data sources sequentially.
        
        Returns:
            True if successful, False otherwise
        """
        success_results = []
        
        # Process OSM data
        logger.info("Processing OSM land polygons")
        osm_success = self._process_osm()
        success_results.append(osm_success)
        
        # Process GADM data
        logger.info("Processing GADM administrative boundaries")
        gadm_success = self._process_gadm()
        success_results.append(gadm_success)
        
        # Save index
        self.preprocessing_index.save()
        
        # Return overall success
        return all(success_results)
    
    def _process_osm(self) -> bool:
        """
        Process OSM land polygons in a single workflow.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get source details
            source_path = self.data_sources['osm']['source']
            extract_dir = self.data_sources['osm']['extract_dir']
            output_dir = self.data_sources['osm']['output_dir']
            
            # Check if vector data already processed (simplified polygons)
            vector_processed = False
            vector_file_hash = None
            vector_output_path = f"{output_dir}/land_polygons_simplified.gpkg"
            # Check if zarr data already processed
            zarr_processed = False
            zarr_output_path = f"{output_dir}/land_mask.zarr"
            
            # Find existing vector file in index
            all_files = self.preprocessing_index.get_files(
                stage=PreprocessingIndex.STAGE_SPATIAL,
                status=PreprocessingIndex.STATUS_COMPLETED
            )
            
            for file in all_files:
                # Check if this is the simplified OSM vector data
                metadata = file.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                
                if (metadata.get('data_type') == 'osm' and 
                    file.get('blob_path') == vector_output_path and
                    metadata.get('processing_stage') == 'simplify'):
                    logger.info(f"Found existing simplified OSM land polygons in index")
                    vector_processed = True
                    vector_file_hash = file.get('file_hash')
                    break

                if (metadata.get('data_type') == 'osm_raster' and 
                    file.get('blob_path') == zarr_output_path and
                    metadata.get('format') == 'zarr'):
                    logger.info(f"Found existing OSM land mask zarr in index")
                    zarr_processed = True
                    break

            # If both vector and zarr are processed, we can skip
            if vector_processed and (zarr_processed or not self.rasterize):
                logger.info("OSM data already fully processed, skipping")
                return True
            
            # If vector not processed, process it
            if not vector_processed:
                # Download and process OSM data to create simplified polygons
                local_zip = "/Users/felixschulz/Downloads/land-polygons-complete-4326.zip"  # Using local file
                
                # Register processing in index
                file_hash = self.preprocessing_index.add_file(
                    stage=PreprocessingIndex.STAGE_RAW,
                    year=datetime.now().year,
                    status=PreprocessingIndex.STATUS_PROCESSING,
                    blob_path=source_path,
                    metadata={
                        "data_type": "osm",
                        "processing_start": datetime.now().isoformat()
                    }
                )
                
                # Extract zip file
                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(local_zip, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Find the shapefile
                shapefiles = list(Path(extract_dir).glob("**/*.shp"))
                if not shapefiles:
                    raise Exception("No shapefiles found in OSM extract")
                
                shapefile_path = str(shapefiles[-1])
                logger.info(f"Found OSM shapefile: {shapefile_path}")
                
                # Read the shapefile
                gdf = gpd.read_file(shapefile_path, engine="pyogrio")
                
                # Update index with raw file info
                self.preprocessing_index.update_file_status(
                    file_hash=file_hash,
                    status=PreprocessingIndex.STATUS_COMPLETED,
                    metadata={
                        "geometry_type": gdf.geometry.type.unique().tolist(),
                        "crs": str(gdf.crs),
                        "feature_count": len(gdf),
                        "extract_dir": extract_dir,
                        "shapefile_path": shapefile_path
                    }
                )
                
                # Process the shapefile - simplify geometries
                logger.info(f"Simplifying OSM polygons with tolerance {self.simplify_tolerance}")
                gdf_simplified = gdf.copy()
                gdf_simplified['geometry'] = gdf_simplified.geometry.simplify(
                    tolerance=self.simplify_tolerance,
                    preserve_topology=True
                )
                
                # Save processed version
                os.makedirs(os.path.dirname(f"{self.temp_dir}/osm"), exist_ok=True)
                output_filename = "land_polygons_simplified.gpkg"
                output_path = os.path.join(self.temp_dir, "osm", output_filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                gdf_simplified.to_file(output_path, driver="GPKG")
                
                # Register processed file
                vector_file_hash = self.preprocessing_index.add_file(
                    stage=PreprocessingIndex.STAGE_SPATIAL,
                    year=datetime.now().year,
                    status=PreprocessingIndex.STATUS_PROCESSING,
                    blob_path=vector_output_path,
                    parent_hash=file_hash,
                    metadata={
                        "data_type": "osm",
                        "original_feature_count": len(gdf),
                        "simplified_feature_count": len(gdf_simplified),
                        "processing_stage": "simplify"
                    }
                )
                
                # Upload to GCS
                os.makedirs(os.path.dirname(f"{output_dir}/{output_filename}"), exist_ok=True)
                self.gcs_client.upload_file(output_path, vector_output_path)
                
                # Update index
                self.preprocessing_index.update_file_status(
                    file_hash=vector_file_hash,
                    status=PreprocessingIndex.STATUS_COMPLETED,
                    metadata={
                        "original_size_mb": os.path.getsize(shapefile_path) / (1024 * 1024),
                        "simplified_size_mb": os.path.getsize(output_path) / (1024 * 1024),
                        "processing_end": datetime.now().isoformat()
                    }
                )
            else:
                # Load existing simplified data for rasterization if needed
                if self.rasterize and not zarr_processed:
                    local_gpkg = os.path.join(self.temp_dir, "osm", "land_polygons_simplified.gpkg")
                    os.makedirs(os.path.dirname(local_gpkg), exist_ok=True)
                    
                    # # Download existing simplified polygons
                    # if not self.gcs_client.download_file(vector_output_path, local_gpkg):
                    #     raise Exception(f"Failed to download existing simplified polygons from {vector_output_path}")
                    local_gpkg = "/Users/felixschulz/Desktop/land_polygons_simplified.gpkg"
                    
                    # Load the data
                    gdf_simplified = gpd.read_file(local_gpkg, engine = "pyogrio")
                    logger.info(f"Loaded existing simplified polygons for rasterization: {len(gdf_simplified)} features")
            
            # Rasterize if requested and not already done
            if self.rasterize and not zarr_processed:
                logger.info("Proceeding with rasterization of land polygons")
                self._rasterize_osm_with_odc(gdf_simplified, vector_file_hash)
            
            logger.info("OSM processing complete")
            return True
            
        except Exception as e:
            logger.exception(f"Error processing OSM data: {e}")
            
            # Update index with error if file_hash exists
            if 'file_hash' in locals():
                self.preprocessing_index.update_file_status(
                    file_hash=file_hash,
                    status=PreprocessingIndex.STATUS_FAILED,
                    metadata={
                        "error": str(e),
                        "processing_end": datetime.now().isoformat()
                    }
                )
            
            return False
    
    def _process_gadm(self) -> bool:
        """
        Process GADM administrative boundaries in a single workflow.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get source details
            source_path = self.data_sources['gadm']['source']
            extract_dir = self.data_sources['gadm']['extract_dir']
            output_dir = self.data_sources['gadm']['output_dir']
            
            # Check if already processed - using compatible method
            all_files = self.preprocessing_index.get_files(
                stage=PreprocessingIndex.STAGE_SPATIAL,
                status=PreprocessingIndex.STATUS_COMPLETED
            )
            
            # Filter the files manually to check for GADM data with level 0
            existing_files = []
            for file in all_files:
                # Check if metadata is a string and parse it
                metadata = file.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        import json
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                        
                # Check if this is GADM data with level 0
                if metadata.get('data_type') == 'gadm' and metadata.get('level') == 0:
                    existing_files.append(file)
            
            if existing_files:
                logger.info("GADM data already processed, skipping")
                return True
            
            # Register processing in index
            file_hash = self.preprocessing_index.add_file(
                stage=PreprocessingIndex.STAGE_RAW,
                year=datetime.now().year,
                status=PreprocessingIndex.STATUS_PROCESSING,
                blob_path=source_path,
                metadata={
                    "data_type": "gadm",
                    "processing_start": datetime.now().isoformat()
                }
            )
            
            # Download zip file
            local_zip = os.path.join(self.temp_dir, os.path.basename(source_path))
            if not self.gcs_client.download_file(source_path, local_zip):
                raise Exception(f"Failed to download {source_path}")
            
            # Extract zip file
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(local_zip, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find shapefiles for different levels
            levels = {}
            for level in range(5):  # GADM has levels 0-4
                pattern = f"*level{level}*.shp"
                shapefiles = list(Path(extract_dir).glob(pattern))
                if shapefiles:
                    levels[level] = str(shapefiles[0])
            
            if not levels:
                raise Exception("No GADM level shapefiles found in extract")
            
            # Update index with raw file info
            self.preprocessing_index.update_file_status(
                file_hash=file_hash,
                status=PreprocessingIndex.STATUS_COMPLETED,
                metadata={
                    "levels_found": list(levels.keys()),
                    "extract_dir": extract_dir
                }
            )
            
            # Process each level
            success = True
            for level, shapefile_path in levels.items():
                level_success = self._process_gadm_level(level, shapefile_path, file_hash, output_dir)
                if not level_success:
                    logger.error(f"Failed to process GADM level {level}")
                    success = False
            
            logger.info("GADM processing complete")
            return success
            
        except Exception as e:
            logger.exception(f"Error processing GADM data: {e}")
            
            # Update index with error
            if 'file_hash' in locals():
                self.preprocessing_index.update_file_status(
                    file_hash=file_hash,
                    status=PreprocessingIndex.STATUS_FAILED,
                    metadata={
                        "error": str(e),
                        "processing_end": datetime.now().isoformat()
                    }
                )
            
            return False
    
    def _process_gadm_level(self, level, shapefile_path, parent_hash, output_dir):
        """Process a specific GADM level."""
        try:
            logger.info(f"Processing GADM level {level}: {shapefile_path}")
            
            # Read the shapefile
            gdf = gpd.read_file(shapefile_path)
            
            # Register in index
            level_file_hash = self.preprocessing_index.add_file(
                stage=PreprocessingIndex.STAGE_SPATIAL,
                year=datetime.now().year,
                status=PreprocessingIndex.STATUS_PROCESSING,
                blob_path=f"{output_dir}/gadm_level{level}_simplified.gpkg",
                parent_hash=parent_hash,
                metadata={
                    "data_type": "gadm",
                    "level": level,
                    "geometry_type": gdf.geometry.type.unique().tolist(),
                    "crs": str(gdf.crs),
                    "feature_count": len(gdf),
                    "processing_stage": "extract"
                }
            )
            
            # Simplify geometries
            logger.info(f"Simplifying GADM level {level} with tolerance {self.simplify_tolerance}")
            gdf_simplified = gdf.copy()
            gdf_simplified['geometry'] = gdf_simplified.geometry.simplify(
                tolerance=self.simplify_tolerance,
                preserve_topology=True
            )
            
            # For country level (0), create a simpler version
            if level == 0:
                # Select key columns for country level
                columns_to_keep = ['GID_0', 'NAME_0', 'geometry']
                columns_to_keep = [col for col in columns_to_keep if col in gdf_simplified.columns]
                gdf_simplified = gdf_simplified[columns_to_keep]
            
            # Save processed version
            output_filename = f"gadm_level{level}_simplified.gpkg"
            os.makedirs(os.path.join(self.temp_dir, "gadm"), exist_ok=True)
            output_path = os.path.join(self.temp_dir, "gadm", output_filename)
            
            gdf_simplified.to_file(output_path, driver="GPKG")
            
            # Upload to GCS
            output_blob_path = f"{output_dir}/{output_filename}"
            self.gcs_client.upload_file(output_path, output_blob_path)
            
            # If this is country level and rasterization is requested
            if level == 0 and self.rasterize:
                self._rasterize_gadm_countries_with_odc(gdf_simplified, level_file_hash, output_dir)
            
            # Update index
            self.preprocessing_index.update_file_status(
                file_hash=level_file_hash,
                status=PreprocessingIndex.STATUS_COMPLETED,
                metadata={
                    "simplified_feature_count": len(gdf_simplified),
                    "columns": list(gdf_simplified.columns),
                    "original_size_mb": os.path.getsize(shapefile_path) / (1024 * 1024),
                    "simplified_size_mb": os.path.getsize(output_path) / (1024 * 1024),
                    "processing_end": datetime.now().isoformat()
                }
            )
            
            logger.info(f"GADM level {level} processing complete")
            return True
            
        except Exception as e:
            logger.exception(f"Error processing GADM level {level}: {e}")
            
            # Update index with error
            if 'level_file_hash' in locals():
                self.preprocessing_index.update_file_status(
                    file_hash=level_file_hash,
                    status=PreprocessingIndex.STATUS_FAILED,
                    metadata={
                        "error": str(e),
                        "processing_end": datetime.now().isoformat()
                    }
                )
            
            return False
    
    def _rasterize_gadm_countries(self, gdf, parent_file_hash, output_dir):
        """Create a rasterized version of GADM countries."""
        try:
            import rasterio
            from rasterio import features
            import numpy as np
            
            # Define output grid
            xmin, ymin, xmax, ymax = self.grid_bounds
            width = int((xmax - xmin) / self.grid_resolution)
            height = int((ymax - ymin) / self.grid_resolution)
            
            # Output file details
            output_filename = "countries_grid.tif"
            output_blob_path = f"{output_dir}/{output_filename}"
            
            # Register the file
            file_hash = self.preprocessing_index.add_file(
                stage=PreprocessingIndex.STAGE_SPATIAL,
                year=datetime.now().year,
                status=PreprocessingIndex.STATUS_PROCESSING,
                blob_path=output_blob_path,
                parent_hash=parent_file_hash,
                metadata={
                    "data_type": "gadm_raster",
                    "width": width,
                    "height": height,
                    "resolution": self.grid_resolution,
                    "processing_stage": "rasterize"
                }
            )
            
            # Create a mapping of country codes to numeric IDs
            if 'GID_0' in gdf.columns:
                country_codes = sorted(gdf['GID_0'].unique())
            else:
                country_codes = [str(i) for i in range(len(gdf))]
                
            code_to_id = {code: i+1 for i, code in enumerate(country_codes)}
            
            # Create raster transform
            transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)
            
            # Create empty raster (0 = no data)
            raster = np.zeros((height, width), dtype=np.uint16)
            
            # Rasterize each country with its numeric ID
            for idx, row in gdf.iterrows():
                if 'GID_0' in gdf.columns:
                    code = row['GID_0']
                else:
                    code = str(idx)
                    
                value = code_to_id[code]
                
                # Rasterize this country's geometry
                country_mask = features.rasterize(
                    [(row.geometry, value)],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint16
                )
                
                # Add to main raster
                raster = np.maximum(raster, country_mask)
            
            # Save to GeoTIFF
            os.makedirs(os.path.join(self.temp_dir, "gadm"), exist_ok=True)
            output_path = os.path.join(self.temp_dir, "gadm", output_filename)
            
            with rasterio.open(
                output_path, 'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=np.uint16,
                crs=self.DEFAULT_CRS,
                transform=transform,
                compress='lzw'
            ) as dst:
                dst.write(raster, 1)
                
                # Add metadata about country codes
                dst.update_tags(countries=json.dumps(code_to_id))
            
            # Also save the country code mapping
            mapping_file = os.path.join(self.temp_dir, "gadm", "country_code_mapping.json")
            with open(mapping_file, 'w') as f:
                json.dump(code_to_id, f, indent=2)
                
            mapping_blob_path = f"{output_dir}/country_code_mapping.json"
            self.gcs_client.upload_file(mapping_file, mapping_blob_path)
            
            # Upload raster to GCS
            self.gcs_client.upload_file(output_path, output_blob_path)
            
            # Update index
            self.preprocessing_index.update_file_status(
                file_hash=file_hash,
                status=PreprocessingIndex.STATUS_COMPLETED,
                metadata={
                    "country_count": len(country_codes),
                    "size_mb": os.path.getsize(output_path) / (1024 * 1024),
                    "processing_end": datetime.now().isoformat(),
                    "country_mapping_path": mapping_blob_path
                }
            )
            
            logger.info(f"GADM countries rasterization complete")
            return True
            
        except Exception as e:
            logger.exception(f"Error rasterizing GADM countries: {e}")
            
            # Update index with error
            if 'file_hash' in locals():
                self.preprocessing_index.update_file_status(
                    file_hash=file_hash,
                    status=PreprocessingIndex.STATUS_FAILED,
                    metadata={
                        "error": str(e),
                        "processing_end": datetime.now().isoformat()
                    }
                )
            
            return False
    
    # Additions to MiscPreprocessor for enhanced rasterization

    def _initialize_dask_client(self):
        """Initialize Dask client for parallel processing."""
        from gnt.data.common.dask.client import DaskClientContextManager
        
        # Use similar parameters to GlassPreprocessor
        dask_params = {
            'threads': self.dask_threads,
            'memory_limit': self.dask_memory_limit,
            'dashboard_port': 8787,
            'temp_dir': os.path.join(self.temp_dir, "dask_workspace")
        }
        return DaskClientContextManager(**dask_params)

    def _get_or_create_geobox(self):
        """
        Get the geobox from GCS or create it from the first VIIRS source file.
        
        Returns:
            ODCGeoBox object for consistent rasterization
        """
        import pickle
        from odc.geo.geobox import GeoBox
        import rioxarray as rxr
        
        geobox_path = "auxiliary/geobox.pkl"
        geobox_local = os.path.join(self.temp_dir, "geobox.pkl")
        
        # Try to download the geobox from GCS
        if self.gcs_client.download_file(geobox_path, geobox_local):
            logger.info(f"Downloaded existing geobox from {geobox_path}")
            try:
                with open(geobox_local, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load geobox: {e}, will create new one")
        
        logger.info("Creating new geobox from VIIRS source file")
        
        # Try to find a VIIRS file using the download index
        viirs_file = None
        try:
            from gnt.data.common.index.download_index import DataDownloadIndex
            from gnt.data.download.sources.factory import create_data_source
            
            # Create a data source instance for VIIRS
            data_source = create_data_source(
                dataset_name="eog",
                config={
                    "base_url": "https://eogdata.mines.edu/nighttime_light/annual/v21/",
                    "file_extensions": ["median_masked.dat.tif.gz"],
                    "output_path": "eog/viirs"
                }
            )
            
            # Initialize download index
            index = DataDownloadIndex(
                bucket_name=self.bucket_name,
                data_source=data_source
            )
            
            # Find the first available VIIRS file
            viirs_files = index.list_successful_files()
            if viirs_files:
                viirs_file = viirs_files[0]
                logger.info(f"Found VIIRS file for geobox creation: {viirs_file}")
        except Exception as e:
            logger.warning(f"Error finding VIIRS file via index: {e}")
        
        if not viirs_file:
            raise Exception(f"Failed to find a VIIRS file {viirs_file}")
        
        # Download the VIIRS file
        viirs_local = os.path.join(self.temp_dir, os.path.basename(viirs_file))
        viirs_local = "/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Job/UNI/Basel/Research/growth-and-temperature/data/VNL_v21_npp_2013_global_vcmcfg_c202205302300.median_masked.dat.tif" # TODO: remove
        # if not self.gcs_client.download_file(viirs_file, viirs_local):
        #     raise Exception(f"Failed to download VIIRS file {viirs_file}")
        
        # Open the file and extract the geobox
        viirs_data = rxr.open_rasterio(viirs_local, chunks="auto")
        geobox = viirs_data.odc.geobox
        
        # Save the geobox to GCS
        with open(geobox_local, 'wb') as f:
            pickle.dump(geobox, f)
        
        self.gcs_client.upload_file(geobox_local, geobox_path)
        logger.info(f"Created and saved new geobox to {geobox_path}")
        
        return geobox

    def _rasterize_osm_with_odc(self, gdf, parent_file_hash):
        """
        Rasterize OSM land polygons using ODC geo tools and Dask.
        
        Args:
            gdf: GeoDataFrame containing land polygons
            parent_file_hash: Hash of the parent file in the preprocessing index
        """
        try:
            from odc.geo.xr import rasterize
            from odc.geo.geom import Geometry
            import shapely
            import xarray as xr
            import zarr
            
            # Output file details
            output_filename = "land_mask.zarr"
            output_blob_path = f"{self.data_sources['osm']['output_dir']}/{output_filename}"
            
            # Register the file in preprocessing index
            file_hash = self.preprocessing_index.add_file(
                stage=PreprocessingIndex.STAGE_SPATIAL,
                year=datetime.now().year,
                status=PreprocessingIndex.STATUS_PROCESSING,
                blob_path=output_blob_path,
                parent_hash=parent_file_hash,
                metadata={
                    "data_type": "osm_raster",
                    "processing_stage": "rasterize",
                    "format": "zarr"
                }
            )
            
            # Create a Dask client for parallel processing
            with self._initialize_dask_client() as client:
                logger.info(f"Created Dask client for OSM rasterization: {client.dashboard_link}")
                
                # Get or create the standard geobox
                geobox = self._get_or_create_geobox()
                
                # Create a single MultiPolygon from all land polygons for efficiency
                logger.info("Creating MultiPolygon from all land geometries")
                land_polygons = shapely.MultiPolygon(gdf.geometry.tolist())
                
                # Create Geometry object with CRS
                geom = Geometry(land_polygons, crs=str(gdf.crs))
                
                # Rasterize using ODC
                logger.info(f"Rasterizing land polygons to grid of shape {geobox.shape}")
                land_mask = rasterize(geom, geobox)
                
                # Convert to Dataset and add metadata
                ds = xr.Dataset(
                    data_vars={'land_mask': land_mask},
                    attrs={
                        'description': 'Land/water mask (1=land, 0=water)',
                        'source': 'OpenStreetMap land polygons',
                        'date_created': datetime.now().isoformat(),
                        'crs': str(geobox.crs)
                    }
                )
                
                # Set up compression for zarr output
                compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
                encoding = {'land_mask': {'compressor': compressor}}
                
                # Save to temporary zarr file
                temp_zarr = os.path.join(self.temp_dir, output_filename)
                logger.info(f"Writing land mask to zarr file at {temp_zarr}")
                ds.to_zarr(temp_zarr, encoding=encoding, consolidated=True)
                
                # Upload to GCS recursively
                logger.info(f"Uploading land mask zarr to {output_blob_path}")
                for root, dirs, files in os.walk(temp_zarr):
                    for file in files:
                        local_path = os.path.join(root, file)
                        rel_path = os.path.relpath(local_path, temp_zarr)
                        cloud_path = f"{output_blob_path}/{rel_path}"
                        self.gcs_client.upload_file(local_path, cloud_path)
            
            # Update index
            self.preprocessing_index.update_file_status(
                file_hash=file_hash,
                status=PreprocessingIndex.STATUS_COMPLETED,
                metadata={
                    "zarr_path": output_blob_path,
                    "shape": list(geobox.shape),
                    "resolution": list(geobox.resolution),
                    "crs": str(geobox.crs),
                    "processing_end": datetime.now().isoformat()
                }
            )
            
            logger.info("OSM land mask rasterization complete")
            return True
            
        except Exception as e:
            logger.exception(f"Error in OSM rasterization: {e}")
            
            # Update index with error
            if 'file_hash' in locals():
                self.preprocessing_index.update_file_status(
                    file_hash=file_hash,
                    status=PreprocessingIndex.STATUS_FAILED,
                    metadata={
                        "error": str(e),
                        "processing_end": datetime.now().isoformat()
                    }
                )
            
            return False

    def _rasterize_gadm_countries_with_odc(self, gdf, parent_file_hash, output_dir):
        """
        Rasterize GADM country boundaries using ODC geo tools and Dask.
        
        Args:
            gdf: GeoDataFrame containing country polygons
            parent_file_hash: Hash of the parent file in the preprocessing index
            output_dir: Directory for output files
        """
        try:
            from odc.geo.xr import rasterize
            from odc.geo.geom import Geometry
            import shapely
            import xarray as xr
            import zarr
            import numpy as np
            
            # Output file details
            output_filename = "countries_grid.zarr"
            output_blob_path = f"{output_dir}/{output_filename}"
            
            # Check if countries grid already exists
            all_files = self.preprocessing_index.get_files(
                stage=PreprocessingIndex.STAGE_SPATIAL,
                status=PreprocessingIndex.STATUS_COMPLETED
            )
            
            for file in all_files:
                # Check if this is the GADM zarr data
                metadata = file.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
            
                if (metadata.get('data_type') == 'gadm_raster' and 
                    file.get('blob_path') == output_blob_path and
                    metadata.get('format') == 'zarr'):
                    logger.info(f"Found existing GADM countries grid zarr in index, skipping rasterization")
                    return True
            
            # Register the file
            file_hash = self.preprocessing_index.add_file(
                stage=PreprocessingIndex.STAGE_SPATIAL,
                year=datetime.now().year,
                status=PreprocessingIndex.STATUS_PROCESSING,
                blob_path=output_blob_path,
                parent_hash=parent_file_hash,
                metadata={
                    "data_type": "gadm_raster",
                    "processing_stage": "rasterize",
                    "format": "zarr"
                }
            )
            
            # Create country code to ID mapping
            if 'GID_0' in gdf.columns:
                country_codes = sorted(gdf['GID_0'].unique())
            else:
                country_codes = [str(i) for i in range(len(gdf))]
        
            code_to_id = {code: i+1 for i, code in enumerate(country_codes)}
            
            # Create a Dask client for parallel processing
            with self._initialize_dask_client() as client:
                logger.info(f"Created Dask client for GADM rasterization: {client.dashboard_link}")
                
                # Get or create the standard geobox
                geobox = self._get_or_create_geobox()
                
                # Create empty array filled with zeros (no data)
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
                
                # Rasterize each country with its ID
                logger.info(f"Rasterizing {len(gdf)} country boundaries")
                for idx, row in gdf.iterrows():
                    if idx % 10 == 0:
                        logger.info(f"Rasterizing country {idx+1}/{len(gdf)}")
                    
                    if 'GID_0' in gdf.columns:
                        code = row['GID_0']
                        name = row.get('NAME_0', code)
                    else:
                        code = str(idx)
                        name = f"Country {idx}"
                    
                    value = code_to_id[code]
                    
                    # Create geometry object with CRS
                    geom = Geometry(row.geometry, crs=str(gdf.crs))
                    
                    # Rasterize this country
                    country_mask = rasterize(geom, geobox, dtype='uint16', 
                                            fill=0, value=value, chunks=2000)
                    
                    # Use maximum to combine with existing data 
                    # (higher values overwrite lower ones if they overlap)
                    countries_grid = xr.where(country_mask > 0, country_mask, countries_grid)
                
                # Convert to Dataset and add metadata
                ds = xr.Dataset(
                    data_vars={'countries': countries_grid},
                    attrs={
                        'description': 'Country ID grid (0=no country)',
                        'source': 'GADM administrative boundaries',
                        'date_created': datetime.now().isoformat(),
                        'crs': str(geobox.crs)
                    }
                )
                
                # Set up compression for zarr output
                compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
                encoding = {'countries': {'compressor': compressor}}
                
                # Save to temporary zarr file
                temp_zarr = os.path.join(self.temp_dir, output_filename)
                logger.info(f"Writing countries grid to zarr file at {temp_zarr}")
                ds.to_zarr(temp_zarr, encoding=encoding, consolidated=True)
                
                # Upload to GCS recursively
                logger.info(f"Uploading countries grid zarr to {output_blob_path}")
                for root, dirs, files in os.walk(temp_zarr):
                    for file in files:
                        local_path = os.path.join(root, file)
                        rel_path = os.path.relpath(local_path, temp_zarr)
                        cloud_path = f"{output_blob_path}/{rel_path}"
                        self.gcs_client.upload_file(local_path, cloud_path)
            
            # Save the country code mapping
            mapping_file = os.path.join(self.temp_dir, "country_code_mapping.json")
            with open(mapping_file, 'w') as f:
                json.dump(code_to_id, f, indent=2)
                
            mapping_blob_path = f"{output_dir}/country_code_mapping.json"
            self.gcs_client.upload_file(mapping_file, mapping_blob_path)
            
            # Update index
            self.preprocessing_index.update_file_status(
                file_hash=file_hash,
                status=PreprocessingIndex.STATUS_COMPLETED,
                metadata={
                    "zarr_path": output_blob_path,
                    "country_count": len(country_codes),
                    "mapping_path": mapping_blob_path,
                    "shape": list(geobox.shape),
                    "resolution": list(geobox.resolution),
                    "crs": str(geobox.crs),
                    "processing_end": datetime.now().isoformat()
                }
            )
            
            logger.info("GADM countries rasterization complete")
            return True
            
        except Exception as e:
            logger.exception(f"Error in GADM countries rasterization: {e}")
            
            # Update index with error
            if 'file_hash' in locals():
                self.preprocessing_index.update_file_status(
                    file_hash=file_hash,
                    status=PreprocessingIndex.STATUS_FAILED,
                    metadata={
                        "error": str(e),
                        "processing_end": datetime.now().isoformat()
                    }
                )
            
            return False
    
    def summarize_annual_means(self) -> None:
        """
        Implementation of abstract method from AbstractPreprocessor.
        For MiscPreprocessor, we just run the single-stage workflow.
        """
        logger.info("Running single-stage workflow for auxiliary data")
        self.preprocess()
    
    def project_to_unified_grid(self) -> None:
        """
        Implementation of abstract method from AbstractPreprocessor.
        Not needed for MiscPreprocessor as we use a single-stage workflow.
        """
        logger.info("Unified grid processing is included in the single-stage workflow")
        # Nothing to do here - processing happens in preprocess()
    
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
        """
        Create an instance of the MiscPreprocessor from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Initialized MiscPreprocessor instance
        """
        return cls(**config)
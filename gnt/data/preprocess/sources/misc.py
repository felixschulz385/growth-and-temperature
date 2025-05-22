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
                data_path (str): Path to the data in GCS
                output_path (str): Path for processed outputs
                simplify_tolerance (float): Tolerance for polygon simplification
                rasterize (bool): Whether to create raster versions
                version (str): Processing version
                temp_dir (str): Directory for temporary files
        """
        super().__init__(**kwargs)
        
        # Settings
        self.bucket_name = kwargs.get('bucket_name', 'growthandheat')
        self.data_path = kwargs.get('data_path', 'auxiliary/boundaries').rstrip('/')
        self.output_path = kwargs.get('output_path', 'auxiliary/boundaries/processed').rstrip('/')
        
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
            data_path=self.data_path,
            version=self.version,
            temp_dir=self.temp_dir
        )
        
        # Fixed data source configurations
        self.data_sources = {
            'osm': {
                'source': f"{self.data_path}/osm/land-polygons-complete-4326.zip",
                'extract_dir': os.path.join(self.temp_dir, "osm_extracted"),
                'output_dir': f"{self.output_path}/osm"
            },
            'gadm': {
                'source': f"{self.data_path}/gadm/gadm_410-levels.zip",
                'extract_dir': os.path.join(self.temp_dir, "gadm_extracted"),
                'output_dir': f"{self.output_path}/gadm"
            }
        }
        
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
            
            # Check if already processed
            existing_files = self.preprocessing_index.get_files(
                stage=PreprocessingIndex.STAGE_SPATIAL,  # Use spatial as the final stage
                status=PreprocessingIndex.STATUS_COMPLETED,
                metadata_filter={"data_type": "osm"}
            )
            
            if existing_files:
                logger.info("OSM data already processed, skipping")
                return True
            
            # Register processing in index
            file_hash = self.preprocessing_index.add_file(
                stage=PreprocessingIndex.STAGE_RAW,
                year=datetime.now().year,  # Current year as reference
                status=PreprocessingIndex.STATUS_PROCESSING,
                blob_path=source_path,
                metadata={
                    "data_type": "osm",
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
            
            # Find the shapefile
            shapefiles = list(Path(extract_dir).glob("**/*.shp"))
            if not shapefiles:
                raise Exception("No shapefiles found in OSM extract")
            
            shapefile_path = str(shapefiles[0])
            logger.info(f"Found OSM shapefile: {shapefile_path}")
            
            # Read the shapefile
            gdf = gpd.read_file(shapefile_path)
            
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
            processed_file_hash = self.preprocessing_index.add_file(
                stage=PreprocessingIndex.STAGE_SPATIAL,  # Final stage
                year=datetime.now().year,
                status=PreprocessingIndex.STATUS_PROCESSING,
                blob_path=f"{output_dir}/{output_filename}",
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
            self.gcs_client.upload_file(output_path, f"{output_dir}/{output_filename}")
            
            # Rasterize if requested
            if self.rasterize:
                self._rasterize_osm(gdf_simplified, processed_file_hash)
            
            # Update index
            self.preprocessing_index.update_file_status(
                file_hash=processed_file_hash,
                status=PreprocessingIndex.STATUS_COMPLETED,
                metadata={
                    "original_size_mb": os.path.getsize(shapefile_path) / (1024 * 1024),
                    "simplified_size_mb": os.path.getsize(output_path) / (1024 * 1024),
                    "processing_end": datetime.now().isoformat()
                }
            )
            
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
    
    def _rasterize_osm(self, gdf, parent_file_hash):
        """Create a rasterized version of OSM land polygons."""
        try:
            import rasterio
            from rasterio import features
            import numpy as np
            
            # Define output grid
            xmin, ymin, xmax, ymax = self.grid_bounds
            width = int((xmax - xmin) / self.grid_resolution)
            height = int((ymax - ymin) / self.grid_resolution)
            
            # Output file details
            output_filename = "land_mask.tif"
            output_blob_path = f"{self.data_sources['osm']['output_dir']}/{output_filename}"
            
            # Register the file
            file_hash = self.preprocessing_index.add_file(
                stage=PreprocessingIndex.STAGE_SPATIAL,
                year=datetime.now().year,
                status=PreprocessingIndex.STATUS_PROCESSING,
                blob_path=output_blob_path,
                parent_hash=parent_file_hash,
                metadata={
                    "data_type": "osm_raster",
                    "width": width,
                    "height": height,
                    "resolution": self.grid_resolution,
                    "processing_stage": "rasterize"
                }
            )
            
            # Create raster transform
            transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)
            
            # Create raster of land=1, ocean=0
            logger.info(f"Rasterizing OSM land polygons to {width}x{height} grid")
            raster = features.rasterize(
                [(geom, 1) for geom in gdf.geometry],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=rasterio.uint8
            )
            
            # Save to GeoTIFF
            output_path = os.path.join(self.temp_dir, "osm", output_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with rasterio.open(
                output_path, 'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=rasterio.uint8,
                crs=self.DEFAULT_CRS,
                transform=transform,
                compress='lzw'
            ) as dst:
                dst.write(raster, 1)
            
            # Upload to GCS
            self.gcs_client.upload_file(output_path, output_blob_path)
            
            # Update index
            self.preprocessing_index.update_file_status(
                file_hash=file_hash,
                status=PreprocessingIndex.STATUS_COMPLETED,
                metadata={
                    "size_mb": os.path.getsize(output_path) / (1024 * 1024),
                    "processing_end": datetime.now().isoformat()
                }
            )
            
            logger.info(f"OSM rasterization complete: {output_blob_path}")
            return True
            
        except Exception as e:
            logger.exception(f"Error rasterizing OSM data: {e}")
            
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
            
            # Check if already processed
            existing_files = self.preprocessing_index.get_files(
                stage=PreprocessingIndex.STAGE_SPATIAL,
                status=PreprocessingIndex.STATUS_COMPLETED,
                metadata_filter={"data_type": "gadm", "level": 0}  # Check for country level
            )
            
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
                self._rasterize_gadm_countries(gdf_simplified, level_file_hash, output_dir)
            
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
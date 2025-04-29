import logging
import json
import os
import tempfile
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
import time
import pandas as pd
from pathlib import Path
import threading

from google.cloud import storage

logger = logging.getLogger(__name__)

class BaseIndex:
    """
    Base class for all data pipeline indices.
    
    Provides common methods for:
    - Index persistence (save/load)
    - GCS interactions
    - Metadata management
    - Thread safety
    """
    
    def __init__(self, bucket_name: str, index_name: str, 
                 prefix: str = "", 
                 index_folder: str = "_index",
                 client=None):
        """
        Initialize the base index.
        
        Args:
            bucket_name: Name of the GCS bucket
            index_name: Name of this specific index
            prefix: Prefix to restrict indexing to a subset of the bucket
            index_folder: Folder within the bucket to store index files
            client: Optional pre-configured storage client
        """
        # Initialize storage client
        self.client = client or storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.index_folder = index_folder
        self.index_name = index_name
        
        # Construct full paths for index files
        self.index_base_path = f"{index_folder}/{index_name}"
        self.metadata_path = f"{self.index_base_path}_metadata.json"
        
        # Initialize metadata
        self.metadata = {
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "stats": {}
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize index state
        self.last_updated = None
        self.last_checked = datetime.now()
        
        # Blob existence cache
        self._blob_existence_cache = {}
        self._blob_cache_max_size = 10000
    
    def _load_parquet_data(self, blob_path, df_name, schema=None):
        """
        Helper function to load a DataFrame from a parquet file in GCS.
        
        Args:
            blob_path: Path to the blob in GCS
            df_name: Name of the dataframe attribute to set on this object
            schema: Optional schema definition for new dataframes
        """
        try:
            blob = self.bucket.blob(blob_path)
            if blob.exists():
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    blob.download_to_filename(temp_file.name)
                    # Use memory-mapped access for large files
                    setattr(self, df_name, pd.read_parquet(temp_file.name))
                    os.unlink(temp_file.name)
                return True
            else:
                logger.debug(f"No data file found at {blob_path}")
                # Initialize empty dataframe if schema provided
                if schema is not None:
                    setattr(self, df_name, pd.DataFrame(columns=schema))
                return False
                
        except Exception as e:
            logger.error(f"Error loading parquet data from {blob_path}: {e}")
            # Initialize empty dataframe if schema provided
            if schema is not None:
                setattr(self, df_name, pd.DataFrame(columns=schema))
            return False
    
    def _save_parquet_data(self, df_name, blob_path):
        """
        Helper function to save a DataFrame as a parquet file in GCS.
        
        Args:
            df_name: Name of the dataframe attribute on this object
            blob_path: Path to the blob in GCS
        """
        df = getattr(self, df_name, None)
        if df is None or df.empty:
            logger.debug(f"No data to save to {blob_path}")
            return
            
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                df.to_parquet(temp_file.name, index=False)
                self.bucket.blob(blob_path).upload_from_filename(temp_file.name)
                os.unlink(temp_file.name)
                
        except Exception as e:
            logger.error(f"Error saving parquet data to {blob_path}: {e}")
    
    def _load_metadata(self):
        """Load metadata JSON from GCS."""
        try:
            blob = self.bucket.blob(self.metadata_path)
            if blob.exists():
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    blob.download_to_filename(temp_file.name)
                    
                    with open(temp_file.name, 'r') as f:
                        self.metadata = json.load(f)
                        
                    if 'last_updated' in self.metadata:
                        self.last_updated = datetime.fromisoformat(self.metadata['last_updated'])
                        
                    os.unlink(temp_file.name)
                
                return True
            else:
                logger.debug(f"No metadata found at {self.metadata_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading metadata from {self.metadata_path}: {e}")
            return False
    
    def _save_metadata(self):
        """Save metadata JSON to GCS."""
        try:
            # Update timestamp
            self.metadata["last_updated"] = datetime.now().isoformat()
            
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                with open(temp_file.name, 'w') as f:
                    json.dump(self.metadata, f)
                
                self.bucket.blob(self.metadata_path).upload_from_filename(temp_file.name)
                os.unlink(temp_file.name)
                
        except Exception as e:
            logger.error(f"Error saving metadata to {self.metadata_path}: {e}")
    
    def is_blob_exists(self, blob_path):
        """Check if a blob exists in the bucket."""
        # Check cache first
        if blob_path in self._blob_existence_cache:
            return self._blob_existence_cache[blob_path]
            
        # Query GCS
        exists = self.bucket.blob(blob_path).exists()
        
        # Update cache
        if len(self._blob_existence_cache) < self._blob_cache_max_size:
            self._blob_existence_cache[blob_path] = exists
            
        return exists
    
    def save(self):
        """Abstract method to save the index. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement save()")
    
    def get_stats(self) -> Dict[str, Any]:
        """Abstract method to get index statistics. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_stats()")
    
    def clear_cache(self):
        """Clear the blob existence cache to ensure fresh reads."""
        with self._lock:
            self._blob_existence_cache.clear()
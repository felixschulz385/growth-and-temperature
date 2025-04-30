import logging
import json
import os
import tempfile
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
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
        
        # Blob existence cache
        self._blob_existence_cache = {}
        self._blob_cache_max_size = 10000
    
    def _load_metadata(self):
        """Load metadata JSON from GCS."""
        try:
            blob = self.bucket.blob(self.metadata_path)
            if blob.exists():
                metadata_json = blob.download_as_text()
                self.metadata = json.loads(metadata_json)
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
            
            # Convert to JSON and upload directly
            metadata_json = json.dumps(self.metadata)
            self.bucket.blob(self.metadata_path).upload_from_string(metadata_json)
            
            return True
        except Exception as e:
            logger.error(f"Error saving metadata to {self.metadata_path}: {e}")
            return False
    
    def is_blob_exists(self, blob_path):
        """Check if a blob exists in the bucket with caching for performance."""
        # Check cache first
        if blob_path in self._blob_existence_cache:
            return self._blob_existence_cache[blob_path]
            
        # Query GCS
        exists = self.bucket.blob(blob_path).exists()
        
        # Update cache
        if len(self._blob_existence_cache) < self._blob_cache_max_size:
            self._blob_existence_cache[blob_path] = exists
            
        return exists
    
    def clear_cache(self):
        """Clear the blob existence cache to ensure fresh reads."""
        with self._lock:
            self._blob_existence_cache.clear()
    
    def save(self):
        """Abstract method to save the index. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement save()")
    
    def get_stats(self) -> Dict[str, Any]:
        """Abstract method to get index statistics. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_stats()")
import logging
import pandas as pd
import json
import os
import tempfile
from typing import Dict, List, Set, Optional
from pathlib import Path
from datetime import datetime
import time

from google.cloud import storage

logger = logging.getLogger(__name__)

class BlobIndex:
    """
    Maintains an efficient index of blobs in a GCS bucket.
    
    This class creates and manages a parquet index file in the bucket that contains
    metadata about blobs, providing fast filtering and searching without
    having to list all blobs every time.
    """
    
    # Constants for index management
    INDEX_FILENAME = "blob_index.parquet"
    INDEX_METADATA_FILENAME = "blob_index_metadata.json"
    DEFAULT_UPDATE_INTERVAL = 3600  # seconds (1 hour)
    
    def __init__(self, bucket_name: str, prefix: str = "", 
                 index_folder: str = "_index",
                 update_interval: int = DEFAULT_UPDATE_INTERVAL,
                 client=None):
        """
        Initialize the blob index.
        
        Args:
            bucket_name: Name of the GCS bucket
            prefix: Prefix to restrict indexing to a subset of the bucket
            index_folder: Folder within the bucket to store index files
            update_interval: How often to check for index updates (seconds)
            client: Optional pre-configured storage client
        """
        # Initialize GCS client directly rather than using GCSClient
        # to avoid circular dependency
        self.client = client or storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.index_folder = index_folder
        self.update_interval = update_interval
        
        # Construct full paths for index files
        prefix_key = self.prefix.replace('/', '_') if self.prefix else "all"
        self.index_path = f"{index_folder}/{prefix_key}_{self.INDEX_FILENAME}"
        self.metadata_path = f"{index_folder}/{prefix_key}_{self.INDEX_METADATA_FILENAME}"
        
        # Index data structure (initialized as empty)
        self.index_df = None
        self.last_updated = None
        self.last_checked = None
        
        # Load or create the index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize the index by either loading or creating it."""
        try:
            # Check if the index metadata exists
            metadata_blob = self.bucket.blob(self.metadata_path)
            if metadata_blob.exists():
                # Download and read the metadata
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    metadata_blob.download_to_filename(temp_file.name)
                    with open(temp_file.name, 'r') as f:
                        metadata = json.load(f)
                        self.last_updated = datetime.fromisoformat(metadata.get('last_updated'))
                    os.unlink(temp_file.name)
            
                # Download and read the index itself if it exists
                index_blob = self.bucket.blob(self.index_path)
                if index_blob.exists():
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        index_blob.download_to_filename(temp_file.name)
                        self.index_df = pd.read_parquet(temp_file.name)
                        os.unlink(temp_file.name)
            
            # Determine if we need to update the index
            current_time = datetime.now()
            self.last_checked = current_time
            
            # Update the index if it's missing, too old, or explicitly requested
            if (self.index_df is None or 
                self.last_updated is None or 
                (current_time - self.last_updated).total_seconds() > self.update_interval):
                logger.info(f"Index is missing or too old. Rebuilding index for prefix '{self.prefix}'")
                self.rebuild_index()
            else:
                logger.info(f"Using existing index for prefix '{self.prefix}', last updated {self.last_updated}")
        
        except Exception as e:
            logger.warning(f"Error initializing index: {e}. Will create a new one.")
            self.rebuild_index()
    
    def rebuild_index(self):
        """Rebuild the complete index from scratch using direct GCS API calls."""
        try:
            logger.info(f"Building blob index for prefix '{self.prefix}'")
            start_time = time.time()
            
            # Use direct blob listing instead of recursively calling list_existing_files
            all_blobs = [blob.name for blob in self.bucket.list_blobs(prefix=self.prefix)]
            
            # Create DataFrame with blob metadata
            records = []
            for blob_name in all_blobs:
                # Skip the index files themselves
                if self.index_folder in blob_name:
                    continue
                    
                # Parse filename parts and extract metadata
                parts = Path(blob_name).parts
                filename = Path(blob_name).name
                extension = Path(blob_name).suffix
                
                # Basic metadata all files will have
                record = {
                    'name': blob_name,
                    'filename': filename, 
                    'extension': extension,
                    'depth': len(parts)
                }
                
                # Add folder paths at different levels (up to 5 levels)
                parts_list = list(parts)
                for i in range(min(5, len(parts_list) - 1)):
                    record[f'folder_{i}'] = parts_list[i]
                
                records.append(record)
            
            # Create the dataframe
            self.index_df = pd.DataFrame(records)
            
            # Save the index
            self._save_index()
            
            elapsed = time.time() - start_time
            logger.info(f"Built index with {len(self.index_df)} entries in {elapsed:.2f} seconds")
        
        except Exception as e:
            logger.error(f"Failed to rebuild index: {str(e)}")
            # Don't re-raise to avoid breaking the calling code
            self.index_df = pd.DataFrame(columns=['name', 'filename', 'extension', 'depth'])
    
    def _save_index(self):
        """Save the index to GCS using direct blob operations."""
        try:
            # Ensure the index dataframe exists
            if self.index_df is None or len(self.index_df) == 0:
                logger.warning("No index data to save")
                return
                
            # Create the index folder if it doesn't exist
            # (Not strictly necessary, but good practice)
            folder_blob = self.bucket.blob(f"{self.index_folder}/")
            if not folder_blob.exists():
                folder_blob.upload_from_string('')
                
            # Save the index file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                self.index_df.to_parquet(temp_file.name, index=False)
                
                index_blob = self.bucket.blob(self.index_path)
                index_blob.upload_from_filename(temp_file.name)
                os.unlink(temp_file.name)
            
            # Update and save metadata
            self.last_updated = datetime.now()
            metadata = {
                'last_updated': self.last_updated.isoformat(),
                'entry_count': len(self.index_df),
                'prefix': self.prefix
            }
            
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                with open(temp_file.name, 'w') as f:
                    json.dump(metadata, f)
                    
                metadata_blob = self.bucket.blob(self.metadata_path)
                metadata_blob.upload_from_filename(temp_file.name)
                os.unlink(temp_file.name)
                
            logger.info(f"Saved index to {self.index_path}")
        
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
    
    def get_matching_files(self, filters: Dict = None, regex: str = None, 
                           max_results: int = None) -> List[str]:
        """
        Get files matching the given filters.
        
        Args:
            filters: Dict of column -> value pairs to filter on
            regex: Regular expression to match against the full path
            max_results: Maximum number of results to return
            
        Returns:
            List of matching file paths
        """
        if self.index_df is None or len(self.index_df) == 0:
            logger.warning("Index is empty")
            return []
        
        # Start with the full index
        result_df = self.index_df
        
        # Apply filters if provided
        if filters:
            for col, value in filters.items():
                if col in result_df.columns:
                    if isinstance(value, list):
                        result_df = result_df[result_df[col].isin(value)]
                    else:
                        result_df = result_df[result_df[col] == value]
        
        # Apply regex if provided
        if regex:
            result_df = result_df[result_df['name'].str.match(regex)]
        
        # Limit results if needed
        if max_results and max_results < len(result_df):
            result_df = result_df.head(max_results)
        
        # Return the paths
        return result_df['name'].tolist()
    
    def get_files_by_prefix(self, prefix: str, max_results: int = None) -> List[str]:
        """
        Get files with the given prefix.
        
        Args:
            prefix: Prefix to match against the full path
            max_results: Maximum number of results to return
            
        Returns:
            List of matching file paths
        """
        if self.index_df is None or len(self.index_df) == 0:
            logger.warning("Index is empty")
            return []
        
        # Filter by prefix
        result_df = self.index_df[self.index_df['name'].str.startswith(prefix)]
        
        # Limit results if needed
        if max_results and max_results < len(result_df):
            result_df = result_df.head(max_results)
        
        # Return the paths
        return result_df['name'].tolist()
    
    def estimate_file_count(self, prefix: str = None) -> int:
        """
        Estimate the number of files with the given prefix.
        
        Args:
            prefix: Prefix to count (if None, counts all files in the index)
            
        Returns:
            Estimated file count
        """
        if self.index_df is None:
            return 0
            
        if prefix:
            return len(self.index_df[self.index_df['name'].str.startswith(prefix)])
        else:
            return len(self.index_df)
            
    def add_to_index(self, blob_names: List[str]) -> None:
        """
        Add new blobs to the existing index without rebuilding it completely.
        
        Args:
            blob_names: List of blob names (paths) to add to the index
            
        Returns:
            None - updates the index in place and saves it
        """
        if not blob_names:
            logger.info("No blobs to add to the index")
            return
            
        # Filter out blobs that should be ignored
        blob_names = [name for name in blob_names if self.index_folder not in name]
        
        # Filter out blobs that are already in the index
        if self.index_df is not None:
            existing_names = set(self.index_df['name'])
            blob_names = [name for name in blob_names if name not in existing_names]
        
        if not blob_names:
            logger.info("All blobs are already in the index")
            return
            
        # Create records for the new blobs
        records = []
        for blob_name in blob_names:
            # Parse filename parts and extract metadata
            parts = Path(blob_name).parts
            filename = Path(blob_name).name
            extension = Path(blob_name).suffix
            
            # Basic metadata all files will have
            record = {
                'name': blob_name,
                'filename': filename, 
                'extension': extension,
                'depth': len(parts)
            }
            
            # Add folder paths at different levels (up to 5 levels)
            parts_list = list(parts)
            for i in range(min(5, len(parts_list) - 1)):
                record[f'folder_{i}'] = parts_list[i]
            
            records.append(record)
        
        # Create a DataFrame for the new records
        new_df = pd.DataFrame(records)
        
        # Append to the existing index
        if self.index_df is None:
            self.index_df = new_df
        else:
            self.index_df = pd.concat([self.index_df, new_df], ignore_index=True)
        
        # Save the updated index
        self._save_index()
        
        logger.info(f"Added {len(records)} new blobs to the index")
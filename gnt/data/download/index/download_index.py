from gnt.data.common.index.base_index import BaseIndex
from typing import Dict, List, Iterator, Tuple, Optional
import pandas as pd
import os
from datetime import datetime
import logging
import hashlib
import tempfile

logger = logging.getLogger(__name__)

class DataDownloadIndex(BaseIndex):
    """Memory-efficient index for downloading files that prioritizes low RAM usage."""
    
    def __init__(self, bucket_name: str, data_source_name: str, client=None):
        # Initialize base index with appropriate paths
        super().__init__(
            bucket_name=bucket_name,
            index_name=f"download_index_{data_source_name}", 
            client=client
        )
        
        self.data_source_name = data_source_name
        
        # Paths for directory-based shards
        self.directories_path = f"{self.index_base_path}_directories.parquet"
        self.completed_dirs_path = f"{self.index_base_path}_completed_directories.parquet"
        
        # Schema definitions
        self.files_schema = [
            "file_hash", "relative_path", "source_url", "destination_blob", 
            "status", "timestamp", "error", "file_size", "metadata"
        ]
        
        self.directories_schema = ["directory_path", "completed_timestamp"]
        
        # Initialize directories DataFrame
        self._completed_directories = pd.DataFrame(columns=self.directories_schema)
        
        # Load only essential data initially
        self._load_parquet_data(self.completed_dirs_path, "_completed_directories", self.directories_schema)
        self._load_metadata()
        
    def _get_directory_shard_path(self, directory: str) -> str:
        """Get the path for a directory shard file."""
        # Create a hash of the directory path for the filename
        dir_hash = hashlib.md5(directory.encode()).hexdigest()[:8]
        return f"{self.index_base_path}_dir_{dir_hash}.parquet"
    
    def _load_directory_shard(self, directory: str) -> pd.DataFrame:
        """Load the shard for a specific directory."""
        shard_path = self._get_directory_shard_path(directory)
        
        # Check if shard exists
        if self.is_blob_exists(shard_path):
            # Load the shard
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                self.bucket.blob(shard_path).download_to_filename(temp_file.name)
                df = pd.read_parquet(temp_file.name)
                os.unlink(temp_file.name)
            return df
        else:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=self.files_schema)
    
    def _save_directory_shard(self, directory: str, df: pd.DataFrame):
        """Save a directory shard back to storage."""
        if df.empty:
            return
            
        shard_path = self._get_directory_shard_path(directory)
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            df.to_parquet(temp_file.name, index=False)
            self.bucket.blob(shard_path).upload_from_filename(temp_file.name)
            os.unlink(temp_file.name)
    
    def add_source_file(self, file_info: Dict):
        """Add a single source file to the index."""
        with self._lock:
            # Extract directory from relative_path
            relative_path = file_info["relative_path"]
            directory = os.path.dirname(relative_path)
            
            # Skip if directory is already completed
            if self.is_directory_completed(directory):
                return
            
            # Load existing directory shard
            df_dir = self._load_directory_shard(directory)
            
            # Check if file already exists in the shard
            file_hash = file_info["file_hash"]
            if file_hash in df_dir["file_hash"].values:
                # File already indexed
                return
                
            # Add status and timestamp if not provided
            if "status" not in file_info:
                file_info["status"] = "indexed"
            if "timestamp" not in file_info:
                file_info["timestamp"] = datetime.now().isoformat()
                
            # Add to directory shard
            df_new = pd.DataFrame([file_info])
            df_dir = pd.concat([df_dir, df_new], ignore_index=True)
            
            # Save directory shard
            self._save_directory_shard(directory, df_dir)
            
            # Update metadata stats
            if "stats" not in self.metadata:
                self.metadata["stats"] = {}
            if "total_indexed" not in self.metadata["stats"]:
                self.metadata["stats"]["total_indexed"] = 0
                
            self.metadata["stats"]["total_indexed"] += 1
            
            # Save metadata periodically (every 1000 files)
            if self.metadata["stats"]["total_indexed"] % 1000 == 0:
                self._save_metadata()
    
    def is_directory_completed(self, directory: str) -> bool:
        """Check if a directory has been fully processed."""
        if self._completed_directories.empty:
            return False
        return directory in self._completed_directories["directory_path"].values
    
    def mark_directory_completed(self, directory: str):
        """Mark a directory as fully processed."""
        with self._lock:
            if not self.is_directory_completed(directory):
                new_row = pd.DataFrame({
                    "directory_path": [directory],
                    "completed_timestamp": [datetime.now().isoformat()]
                })
                self._completed_directories = pd.concat(
                    [self._completed_directories, new_row], ignore_index=True)
                
                # Save completed directories
                self._save_parquet_data("_completed_directories", self.completed_dirs_path)
                
                # Update metadata
                if "stats" not in self.metadata:
                    self.metadata["stats"] = {}
                if "completed_directories" not in self.metadata["stats"]:
                    self.metadata["stats"]["completed_directories"] = 0
                    
                self.metadata["stats"]["completed_directories"] += 1
                self._save_metadata()
    
    def get_file_status(self, file_hash: str) -> Tuple[str, Optional[str]]:
        """
        Get status of a specific file by hash.
        Returns tuple of (status, error_message)
        """
        # This requires efficient search across shards
        # For practical implementation, we could maintain a small lookup table
        # mapping file_hash → directory for quick lookups
        # For now we'll do a simple implementation:
        
        # Use cache for common lookups
        cache_key = f"file_status_{file_hash}"
        if hasattr(self, "_status_cache") and cache_key in self._status_cache:
            return self._status_cache[cache_key]
        
        # Create status cache if it doesn't exist
        if not hasattr(self, "_status_cache"):
            self._status_cache = {}
            
        # List all directory shards
        prefix = f"{self.index_base_path}_dir_"
        for blob in self.bucket.list_blobs(prefix=prefix):
            # Load shard
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                blob.download_to_filename(temp_file.name)
                df = pd.read_parquet(temp_file.name)
                os.unlink(temp_file.name)
                
            # Check if file is in this shard
            if file_hash in df["file_hash"].values:
                row = df[df["file_hash"] == file_hash].iloc[0]
                status = row["status"]
                error = row.get("error") if "error" in row else None
                
                # Cache result (limit cache size)
                if len(self._status_cache) < 10000:  # Limit cache size
                    self._status_cache[cache_key] = (status, error)
                    
                return status, error
        
        # File not found
        return "unknown", None

    def record_download_status(self, file_hash: str, source_url: str, 
                              destination_blob: str, status: str, error: str = None):
        """Update the status of a file download."""
        with self._lock:
            # First, find which directory shard contains this file
            relative_path = None
            
            # Check all directory shards (not optimal but works)
            prefix = f"{self.index_base_path}_dir_"
            for blob in self.bucket.list_blobs(prefix=prefix):
                shard_path = blob.name
                
                # Load shard
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    blob.download_to_filename(temp_file.name)
                    df = pd.read_parquet(temp_file.name)
                    os.unlink(temp_file.name)
                    
                # Check if file is in this shard
                if file_hash in df["file_hash"].values:
                    # Get directory from file info
                    row = df[df["file_hash"] == file_hash].iloc[0]
                    relative_path = row["relative_path"]
                    directory = os.path.dirname(relative_path)
                    
                    # Update status
                    mask = df["file_hash"] == file_hash
                    df.loc[mask, "status"] = status
                    df.loc[mask, "timestamp"] = datetime.now().isoformat()
                    if error:
                        df.loc[mask, "error"] = error
                    
                    # Save directory shard
                    self._save_directory_shard(directory, df)
                    
                    # Update stats based on status
                    if "stats" not in self.metadata:
                        self.metadata["stats"] = {}
                        
                    if status == "success":
                        if "successful_downloads" not in self.metadata["stats"]:
                            self.metadata["stats"]["successful_downloads"] = 0
                        self.metadata["stats"]["successful_downloads"] += 1
                    
                    elif status == "failed":
                        if "failed_downloads" not in self.metadata["stats"]:
                            self.metadata["stats"]["failed_downloads"] = 0
                        self.metadata["stats"]["failed_downloads"] += 1
                    
                    # Save metadata periodically
                    if (self.metadata["stats"].get("successful_downloads", 0) +
                        self.metadata["stats"].get("failed_downloads", 0)) % 100 == 0:
                        self._save_metadata()
                    
                    # Update cache if it exists
                    cache_key = f"file_status_{file_hash}"
                    if hasattr(self, "_status_cache"):
                        self._status_cache[cache_key] = (status, error)
                        
                    return
            
            # If we get here, file wasn't found in any shard
            logger.warning(f"Tried to update status for unknown file: {file_hash}")
    
    def get_pending_downloads(self) -> List[Dict]:
        """Get files that need to be downloaded."""
        pending_files = []
        
        # For each directory that isn't completed
        prefix = f"{self.index_base_path}_dir_"
        for blob in self.bucket.list_blobs(prefix=prefix):
            shard_path = blob.name
            
            # Extract directory from shard path if needed for filtering
            
            # Load shard
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                blob.download_to_filename(temp_file.name)
                df = pd.read_parquet(temp_file.name)
                os.unlink(temp_file.name)
                
            # Get pending files (indexed status)
            pending = df[df["status"] == "indexed"]
            if not pending.empty:
                pending_files.extend(pending.to_dict('records'))
        
        return pending_files
    
    def get_stats(self) -> Dict:
        """Get download statistics."""
        if "stats" not in self.metadata:
            return {
                "successful_downloads": 0,
                "failed_downloads": 0,
                "total_indexed": 0,
                "completed_directories": 0
            }
        return self.metadata["stats"]
    
    def save(self):
        """Save all index components."""
        # Save completed directories
        self._save_parquet_data("_completed_directories", self.completed_dirs_path)
        
        # Save metadata
        self._save_metadata()
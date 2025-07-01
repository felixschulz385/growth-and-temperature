"""
Base SQLite-based index for managing file metadata.
"""

import os
import time
import json
import sqlite3
import logging
import threading
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime

from gnt.data.download.sources.base import BaseDataSource

logger = logging.getLogger(__name__)

class DataDownloadIndex:
    """SQLite-based index for managing file metadata."""
    
    def __init__(self, bucket_name: str, data_source: Any, 
                 client=None, temp_dir=None, save_interval_seconds=300):
        """
        Initialize the index.
        
        Args:
            bucket_name: Storage bucket name (legacy parameter)
            data_source: Data source for indexing
            client: Storage client (optional)
            temp_dir: Temporary directory for index files
            save_interval_seconds: How often to save the index
        """
        self.bucket_name = bucket_name
        self.data_source = data_source
        self.client = client
        self.temp_dir = temp_dir or os.path.join(os.getcwd(), "tmp")
        self.save_interval_seconds = save_interval_seconds
        
        # Extract data source properties
        self.data_source_name = getattr(data_source, "DATA_SOURCE_NAME", "unknown")
        self.data_path = getattr(data_source, "data_path", "unknown")
        
        # Metadata
        self.metadata = {"last_modified": datetime.now().isoformat()}
        
        # Connection management
        self._last_save_time = time.time()
        
        # Set up database
        self._setup_database()
    
    def _setup_database(self):
        """Set up the SQLite database file."""
        # Default implementation - override in subclasses
        self.local_db_path = os.path.join(self.temp_dir, f"index_{self.data_path.replace('/', '_')}.sqlite")
        os.makedirs(os.path.dirname(self.local_db_path), exist_ok=True)
        
        # Connect and create tables
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create standard tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            file_hash TEXT PRIMARY KEY,
            relative_path TEXT,
            source_url TEXT,
            destination_blob TEXT,
            timestamp TEXT,
            file_size INTEGER,
            metadata TEXT
        )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON files(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relative_path ON files(relative_path)')
        conn.commit()
    
    def _get_connection(self):
        """
        Get a database connection for the current thread.
        
        Returns:
            sqlite3.Connection: Database connection
        """
        thread_id = threading.get_ident()
        
        # Create thread-local storage if needed
        if not hasattr(self, '_thread_local'):
            self._thread_local = threading.local()
        
        # Check if this thread already has a connection
        if not hasattr(self._thread_local, 'connections'):
            self._thread_local.connections = {}
        
        # Create a new connection for this thread if needed
        if thread_id not in self._thread_local.connections:
            self._thread_local.connections[thread_id] = sqlite3.connect(
                self.local_db_path,
                timeout=60.0,
                isolation_level=None  # Use autocommit mode
            )
            
            # Enable foreign keys
            self._thread_local.connections[thread_id].execute("PRAGMA foreign_keys = ON")
            
            # Enable WAL mode for better concurrent access
            self._thread_local.connections[thread_id].execute("PRAGMA journal_mode = WAL")
            
            # Set busy timeout
            self._thread_local.connections[thread_id].execute("PRAGMA busy_timeout = 30000")
            
            logger.debug(f"Created new database connection for thread {thread_id}")
        
        return self._thread_local.connections[thread_id]

    def _close_all_connections(self):
        """Close all database connections."""
        if hasattr(self, '_thread_local') and hasattr(self._thread_local, 'connections'):
            thread_id = threading.get_ident()
            
            # Only close connections from the current thread
            if thread_id in self._thread_local.connections:
                conn = self._thread_local.connections[thread_id]
                if conn:
                    try:
                        # Ensure WAL is synced before closing
                        conn.execute("PRAGMA wal_checkpoint(FULL)")
                        conn.close()
                    except sqlite3.Error as e:
                        logger.warning(f"Error closing connection in thread {thread_id}: {e}")
                    
                    # Remove from the connections dict
                    del self._thread_local.connections[thread_id]
                    logger.debug(f"Closed database connection for thread {thread_id}")
        else:
            logger.debug("No connections to close in this thread")
    
    def save(self):
        """Save the index."""
        # Default implementation - override in subclasses
        logger.info("Saving index")
        self._last_save_time = time.time()
    
    def get_stats(self):
        """
        Get statistics about the current index.
        
        Returns:
            dict: Statistics about indexed files
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get total file count
        cursor.execute("SELECT COUNT(*) FROM files")
        total_files = cursor.fetchone()[0]
        
        # Get file size statistics
        cursor.execute("SELECT SUM(file_size) FROM files WHERE file_size IS NOT NULL")
        total_size = cursor.fetchone()[0] or 0
        
        # Return statistics
        return {
            'total_files': total_files,
            'total_size': total_size,
        }
    
    def refresh_index(self, data_source):
        """
        Refresh the index from the data source (to be implemented by subclasses).
        
        Args:
            data_source: Data source to refresh from
        """
        raise NotImplementedError("Subclasses must implement refresh_index")
    
    def build_index_from_source(self, data_source, **kwargs):
        """
        Build index from source (to be implemented by subclasses).
        
        Args:
            data_source: Data source to build index from
            **kwargs: Additional arguments for building the index
        """
        raise NotImplementedError("Subclasses must implement build_index_from_source")
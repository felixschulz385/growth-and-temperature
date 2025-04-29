import pytest
import tempfile
import os
import json
from unittest.mock import MagicMock, patch
from index.download_index import DownloadIndex

class MockGCSClient:
    def __init__(self):
        self.uploaded_files = {}
        self.downloaded_files = {}
        self.existing_blobs = set()
        
    def upload_file(self, local_path, destination_blob):
        with open(local_path, 'r') as f:
            self.uploaded_files[destination_blob] = f.read()
        
    def download_blob_to_file(self, blob_name, file_path):
        if blob_name in self.uploaded_files:
            with open(file_path, 'w') as f:
                f.write(self.uploaded_files[blob_name])
            return True
        return False
        
    def blob_exists(self, blob_name):
        return blob_name in self.existing_blobs or blob_name in self.uploaded_files

@pytest.fixture
def mock_gcs():
    return MockGCSClient()

@pytest.fixture
def download_index(mock_gcs):
    return DownloadIndex(mock_gcs, "test-bucket", "test-source")

def test_download_index_initialization(download_index):
    """Test that a new index is initialized with empty collections"""
    assert isinstance(download_index.index_data["completed_directories"], set)
    assert isinstance(download_index.index_data["successful_downloads"], set)
    assert isinstance(download_index.index_data["failed_downloads"], dict)

def test_mark_directory_completed(download_index):
    """Test marking a directory as completed"""
    directory = "2021/001"
    download_index.mark_directory_completed(directory)
    assert directory in download_index.index_data["completed_directories"]

def test_is_directory_completed(download_index):
    """Test checking if a directory is completed"""
    directory = "2021/002"
    assert not download_index.is_directory_completed(directory)
    download_index.mark_directory_completed(directory)
    assert download_index.is_directory_completed(directory)

def test_record_download_success(download_index):
    """Test recording a successful download"""
    file_url = "https://example.com/file.hdf"
    download_index.record_download_success(file_url)
    assert file_url in download_index.index_data["successful_downloads"]
    assert download_index.index_data["download_stats"]["total_downloaded"] == 1

def test_record_download_failure(download_index):
    """Test recording a failed download"""
    file_url = "https://example.com/bad-file.hdf"
    error = "Connection refused"
    download_index.record_download_failure(file_url, error)
    assert file_url in download_index.index_data["failed_downloads"]
    assert download_index.index_data["download_stats"]["total_failed"] == 1

def test_save_and_load_index(download_index, mock_gcs):
    """Test saving and loading the index"""
    # Add some test data
    download_index.mark_directory_completed("2021/003")
    download_index.record_download_success("https://example.com/file1.hdf")
    download_index.record_download_failure("https://example.com/file2.hdf", "Error")
    
    # Save the index
    download_index.save_index()
    
    # Create a new index with the same GCS client
    new_index = DownloadIndex(mock_gcs, "test-bucket", "test-source")
    
    # Verify the data was loaded from the saved index
    assert "2021/003" in new_index.index_data["completed_directories"]
    assert "https://example.com/file1.hdf" in new_index.index_data["successful_downloads"]
    assert "https://example.com/file2.hdf" in new_index.index_data["failed_downloads"]

def test_thread_safety(download_index):
    """Test that operations are thread-safe"""
    import threading
    
    # Define a function to run in multiple threads
    def add_items(thread_id):
        for i in range(100):
            download_index.record_download_success(f"https://example.com/thread{thread_id}/file{i}.hdf")
            download_index.mark_directory_completed(f"thread{thread_id}/dir{i}")
    
    # Create and run multiple threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=add_items, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Check that all items were added
    success_count = download_index.index_data["download_stats"]["total_downloaded"]
    dir_count = len(download_index.index_data["completed_directories"])
    
    assert success_count == 500  # 5 threads × 100 files
    assert dir_count == 500  # 5 threads × 100 directories
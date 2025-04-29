# tests/test_base.py
import pytest
from download.base import BaseDataSource

class DummyDataSource(BaseDataSource):
    def list_remote_files(self):
        return [("a.txt", "http://example.com/a.txt")]
    
    def local_path(self, remote_file):
        return f"/tmp/{remote_file}"  # simulate a resolved local path

    def download(self, file_url, output_path):
        return f"Downloaded {file_url} to {output_path}"

    def gcs_upload_path(self, relative_path):
        return f"dummy/{relative_path}"

def test_dummy_data_source():
    ds = DummyDataSource()
    files = ds.list_remote_files()
    assert files == [("a.txt", "http://example.com/a.txt")]
    assert ds.gcs_upload_path("a.txt") == "dummy/a.txt"

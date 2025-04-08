# tests/test_workflow.py
import pytest
import tempfile
import os
from unittest.mock import MagicMock

from workflow import run
from download.base import BaseDataSource

class MockDataSource(BaseDataSource):
    
    def __init__(self):
        self.base_url = "http://example.com/"
    
    def list_remote_files(self):
        return [("folder/fake.hdf", "http://example.com/fake.hdf")]
    
    def local_path(self, remote_file):
        return f"/tmp/{remote_file}"  # simulate a resolved local path

    def download(self, file_url, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write("data")

    def gcs_upload_path(self, base_url, relative_path):
        return relative_path

class MockGCSClient:
    def __init__(self, *args, **kwargs):
        self.uploaded = []
    
    def list_existing_files(self):
        return []

    def upload_file(self, local_path, blob_path):
        self.uploaded.append((local_path, blob_path))

def test_workflow_upload(monkeypatch):
    gcs_client = MockGCSClient()
    
    # Monkeypatch the GCSClient instantiation in workflow
    monkeypatch.setattr("workflow.GCSClient", lambda bucket: gcs_client)

    run(MockDataSource(), "my-bucket")
    assert len(gcs_client.uploaded) == 1
    assert gcs_client.uploaded[0][1] == "folder/fake.hdf"

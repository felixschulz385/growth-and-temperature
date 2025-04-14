import pytest
from unittest.mock import patch, MagicMock
from download.harvard import HarvardDataSource


@pytest.fixture
def harvard_source():
    return HarvardDataSource(base_url="10.7910/DVN/ABCDE1", file_extensions=[".csv", ".txt"])


@patch("download.harvard.requests.get")
def test_list_remote_files(mock_get, harvard_source):
    # Mock API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": {
            "latestVersion": {
                "files": [
                    {
                        "label": "dataset-description.txt",
                        "dataFile": {
                            "id": 123456,
                            "originalFileName": "dataset-description.txt"
                        }
                    },
                    {
                        "label": "data.csv",
                        "dataFile": {
                            "id": 789012,
                            "originalFileName": "data.csv"
                        }
                    }
                ]
            }
        }
    }
    mock_response.raise_for_status = lambda: None
    mock_get.return_value = mock_response

    files = harvard_source.list_remote_files()
    
    assert isinstance(files, list)
    assert len(files) == 2
    relative_path, url = files[1]
    assert relative_path.endswith("data.csv")
    assert url == "https://dataverse.harvard.edu/api/access/datafile/789012"


def test_local_path(harvard_source):
    path = harvard_source.local_path("10.7910/DVN/ABCDE1/data.csv")
    assert path == "data/10.7910/DVN/ABCDE1/data.csv"


def test_gcs_upload_path(harvard_source):
    path = harvard_source.gcs_upload_path("10.7910/DVN/ABCDE1", "10.7910/DVN/ABCDE1/data.csv")
    assert path == "harvard/DVN/ABCDE1/data.csv"

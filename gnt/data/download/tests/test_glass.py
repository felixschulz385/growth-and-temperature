# tests/test_glass.py
import pytest
import requests_mock
from download.glass import GLASSDataSource

def test_list_remote_files():
    base_url = "https://fake-glass.org/archive/LST/"

    # Simulate year index page
    html_index = """
    <html><body>
    <a href="2021/">2021</a>
    </body></html>
    """

    # Simulate year page with HDF files
    html_2021 = """
    <html><body>
    <a href="file1.hdf">file1.hdf</a>
    <a href="file2.hdf">file2.hdf</a>
    </body></html>
    """

    with requests_mock.Mocker() as m:
        m.get(base_url, text=html_index)
        m.get("https://fake-glass.org/archive/LST/2021/", text=html_2021)

        ds = GLASSDataSource(base_url)
        files = ds.list_remote_files()

        assert next(files) == ("2021/file1.hdf", "https://fake-glass.org/archive/LST/2021/file1.hdf")

def test_gcs_upload_path():
    ds = GLASSDataSource(base_url="https://glass.hku.hk/archive/LST/MODIS/Daily/1KM/")

    # Relative file path from list_remote_files
    relative_path = "2021/file1.hdf"
    base_url = ds.base_url

    gcs_path = ds.gcs_upload_path(base_url=base_url, relative_path=relative_path)

    # Expecting: glass/LST/MODIS/Daily/1KM/file1.hdf
    expected = "glass/LST/MODIS/Daily/1KM/file1.hdf"

    assert gcs_path == expected
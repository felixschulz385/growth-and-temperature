import os
import pytest
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rxr
import zarr
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch, Mock

from preprocess.glass import GlassPreprocessor
from gcs.client import GCSClient

# Sample file lists for testing
MOCK_MODIS_FILES = [
    "glass/LST/MODIS/Daily/1KM/GLASS06A01.V01.A2000001.h08v05.2022021.hdf",
    "glass/LST/MODIS/Daily/1KM/GLASS06A01.V01.A2000002.h08v05.2022021.hdf",
    "glass/LST/MODIS/Daily/1KM/GLASS06A01.V01.A2000003.h08v05.2022021.hdf",
    "glass/LST/MODIS/Daily/1KM/GLASS06A01.V01.A2000001.h09v05.2022021.hdf",
    "glass/LST/MODIS/Daily/1KM/GLASS06A01.V01.A2000002.h09v05.2022021.hdf",
    "glass/LST/MODIS/Daily/1KM/GLASS06A01.V01.A2000003.h09v05.2022021.hdf",
    "glass/LST/MODIS/Daily/1KM/GLASS06A01.V01.A2001001.h08v05.2022021.hdf",
]

MOCK_AVHRR_FILES = [
    "glass/LST/AVHRR/0.05D/GLASS08B31.V40.A1982001.2021259.hdf",
    "glass/LST/AVHRR/0.05D/GLASS08B31.V40.A1982002.2021259.hdf",
    "glass/LST/AVHRR/0.05D/GLASS08B31.V40.A1982003.2021259.hdf",
    "glass/LST/AVHRR/0.05D/GLASS08B31.V40.A1983001.2021259.hdf",
]


class MockGCSClient:
    """Mock GCSClient for testing GlassPreprocessor."""
    
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.uploaded_files = {}
        self.download_file = MagicMock(side_effect=self._mock_download_file)
        self.upload_file = MagicMock()  # Use MagicMock for upload_file
    
    def list_existing_files(self, prefix):
        """Mock method to list files."""
        if "MODIS" in prefix:
            return MOCK_MODIS_FILES
        elif "AVHRR" in prefix:
            return MOCK_AVHRR_FILES
        return []
    
    def list_blobs_with_limit(self, prefix, limit=1):
        """Mock method to list a limited number of blobs."""
        if "MODIS" in prefix:
            for file in MOCK_MODIS_FILES[:limit]:
                yield file
        elif "AVHRR" in prefix:
            for file in MOCK_AVHRR_FILES[:limit]:
                yield file

    def _mock_download_file(self, source_path, destination_path):
        """Mock method to download a file."""
        # Just create an empty file
        with open(destination_path, 'w') as f:
            f.write("Mock file content")
        return destination_path
    
    def check_if_exists(self, path):
        """Mock method to check if a file exists."""
        return False


@pytest.fixture
def mock_gcs_client():
    """Fixture to create a mock GCS client."""
    return MockGCSClient("mock-bucket")


@pytest.fixture
def temp_paths():
    """Fixture to create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "input"
        output_path = Path(temp_dir) / "output"
        intermediate_path = Path(temp_dir) / "intermediate"
        
        input_path.mkdir()
        output_path.mkdir()
        intermediate_path.mkdir()
        
        yield {
            "temp_dir": temp_dir,
            "input_path": input_path,
            "output_path": output_path,
            "intermediate_path": intermediate_path,
        }


@pytest.fixture
def mock_xarray_data():
    """Fixture to create mock xarray data."""
    # Create a simple DataArray with time, x, and y dimensions
    times = pd.date_range('2000-01-01', periods=3, freq='D')
    x = np.arange(5)
    y = np.arange(5)
    
    data = np.random.rand(len(times), len(y), len(x))
    
    da = xr.DataArray(
        data,
        coords=[
            ('time', times),
            ('y', y),
            ('x', x)
        ],
        name="LST"
    )
    
    return da


class TestGlassPreprocessor:
    """Tests for GlassPreprocessor class."""

    def test_init_defaults(self, temp_paths):
        """Test initialization with default parameters."""
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"]
        )
        
        assert preprocessor.data_source == "MODIS"
        assert len(preprocessor.years) == 21  # Default MODIS years (2000-2020)
        assert preprocessor.grid_cells is None
        assert preprocessor.override is False
        assert preprocessor.path_prefix == preprocessor.MODIS_PATH_PREFIX

    def test_init_avhrr(self, temp_paths):
        """Test initialization with AVHRR data source."""
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"],
            data_source="AVHRR"
        )
        
        assert preprocessor.data_source == "AVHRR"
        assert len(preprocessor.years) == 39  # Default AVHRR years (1982-2020)
        assert preprocessor.path_prefix == preprocessor.AVHRR_PATH_PREFIX

    def test_init_with_years(self, temp_paths):
        """Test initialization with specific years."""
        # Test with list of years
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"],
            years=[2000, 2005, 2010]
        )
        assert preprocessor.years == [2000, 2005, 2010]
        
        # Test with year range
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"],
            years=(2000, 2002)
        )
        assert preprocessor.years == [2000, 2001, 2002]

    def test_init_invalid_data_source(self, temp_paths):
        """Test initialization with invalid data source."""
        with pytest.raises(ValueError, match="Unsupported data source"):
            GlassPreprocessor(
                temp_paths["input_path"],
                temp_paths["output_path"],
                temp_paths["intermediate_path"],
                data_source="INVALID"
            )

    def test_init_invalid_years(self, temp_paths):
        """Test initialization with invalid years."""
        with pytest.raises(ValueError, match="Years must be"):
            GlassPreprocessor(
                temp_paths["input_path"],
                temp_paths["output_path"],
                temp_paths["intermediate_path"],
                years="2000"  # Should be list or tuple
            )

    def test_from_config(self, temp_paths):
        """Test initialization from configuration dictionary."""
        config = {
            "input_path": str(temp_paths["input_path"]),
            "output_path": str(temp_paths["output_path"]),
            "intermediate_path": str(temp_paths["intermediate_path"]),
            "data_source": "AVHRR",
            "years": [1990, 1995, 2000],
            "override": True
        }
        
        preprocessor = GlassPreprocessor.from_config(config)
        
        assert preprocessor.data_source == "AVHRR"
        assert preprocessor.years == [1990, 1995, 2000]
        assert preprocessor.override is True

    @patch("preprocess.glass.GCSClient")
    def test_validate_input(self, mock_gcs_client_class, temp_paths, mock_gcs_client):
        """Test validation of input data."""
        mock_gcs_client_class.return_value = mock_gcs_client
        
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"]
        )
        
        # Should return True because our mock returns files
        assert preprocessor.validate_input() is True

    @patch("preprocess.glass.GCSClient")
    def test_validate_input_no_files(self, mock_gcs_client_class, temp_paths):
        """Test validation of input data when no files are found."""
        mock_client = MagicMock(spec=GCSClient)
        mock_client.list_blobs_with_limit.return_value = []
        mock_gcs_client_class.return_value = mock_client
        
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"]
        )
        
        # Should return False because no files were found
        assert preprocessor.validate_input() is False

    @patch("preprocess.glass.GCSClient")
    def test_validate_input_error(self, mock_gcs_client_class, temp_paths):
        """Test validation of input data when an error occurs."""
        mock_client = MagicMock(spec=GCSClient)
        mock_client.list_blobs_with_limit.side_effect = Exception("Connection error")
        mock_gcs_client_class.return_value = mock_client
        
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"]
        )
        
        # Should return False when an error occurs
        assert preprocessor.validate_input() is False

    @patch("preprocess.glass.GCSClient")
    def test_parse_modis_filenames(self, mock_gcs_client_class, temp_paths, mock_gcs_client):
        """Test parsing MODIS filenames."""
        mock_gcs_client_class.return_value = mock_gcs_client
        
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"]
        )
        
        # Parse the mock MODIS files
        df = preprocessor._parse_modis_filenames(MOCK_MODIS_FILES)
        
        # Check if parsing was successful
        assert len(df) == 7  # We have 7 mock MODIS files
        assert set(df.columns) == {'path', 'year', 'day', 'h', 'v'}
        
        # Check specific values
        first_file = df.iloc[0]
        assert first_file['year'] == 2000
        assert first_file['day'] == 1
        assert first_file['h'] == 8
        assert first_file['v'] == 5

    @patch("preprocess.glass.GCSClient")
    def test_parse_avhrr_filenames(self, mock_gcs_client_class, temp_paths, mock_gcs_client):
        """Test parsing AVHRR filenames."""
        mock_gcs_client_class.return_value = mock_gcs_client
        
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"],
            data_source="AVHRR"
        )
        
        # Parse the mock AVHRR files
        df = preprocessor._parse_avhrr_filenames(MOCK_AVHRR_FILES)
        
        # Check if parsing was successful
        assert len(df) == 4  # We have 4 mock AVHRR files
        assert set(df.columns) == {'path', 'year', 'day'}
        
        # Check specific values
        first_file = df.iloc[0]
        assert first_file['year'] == 1982
        assert first_file['day'] == 1

    @patch("preprocess.glass.GCSClient")
    def test_list_files_from_gcs(self, mock_gcs_client_class, temp_paths, mock_gcs_client):
        """Test listing files from GCS."""
        mock_gcs_client_class.return_value = mock_gcs_client
        
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"]
        )
        
        # List files
        files = preprocessor._list_files_from_gcs()
        
        # Check if the correct files were listed
        assert files == MOCK_MODIS_FILES
        
        # Test AVHRR
        preprocessor.data_source = "AVHRR"
        preprocessor.path_prefix = preprocessor.AVHRR_PATH_PREFIX
        
        files = preprocessor._list_files_from_gcs()
        assert files == MOCK_AVHRR_FILES

    @patch("preprocess.glass.GCSClient")
    def test_filter_by_grid_cells(self, mock_gcs_client_class, temp_paths, mock_gcs_client):
        """Test filtering MODIS data by grid cells."""
        mock_gcs_client_class.return_value = mock_gcs_client
        
        # Create a preprocessor with specific grid cells
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"],
            grid_cells=["h08v05"],
            years=[2000]
        )
        
        # Mock _list_files_from_gcs to return our test data
        preprocessor._list_files_from_gcs = MagicMock(return_value=MOCK_MODIS_FILES)
        
        # Mock _process_file_group to avoid actual processing
        preprocessor._process_file_group = MagicMock()
        
        # Run the processing
        preprocessor._process_modis_data()
        
        # Check that only the h08v05 grid cell files were processed
        # The _process_file_group should be called once for the h08v05/2000 group
        assert preprocessor._process_file_group.call_count == 1
        
        # Get the files_df that was passed to _process_file_group
        call_args = preprocessor._process_file_group.call_args[0][0]
        
        # Check that only h08v05 files were included
        for _, row in call_args.iterrows():
            path = row['path']
            assert "h08v05" in path
            assert "h09v05" not in path

    def test_combine_time_series(self, temp_paths):
        """Test combining time series data."""
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"]
        )
        
        # Create test data arrays
        data_arrays = []
        for i in range(3):
            # Create array with day of year coords
            da = xr.DataArray(
                np.random.rand(5, 5),
                dims=["y", "x"],
                coords={
                    "y": range(5),
                    "x": range(5)
                }
            )
            data_arrays.append(da)
        
        # Create test files_df
        files_df = pd.DataFrame({
            'path': ["path/to/file1.hdf", "path/to/file2.hdf", "path/to/file3.hdf"],
            'year': [2000, 2000, 2000],
            'day': [1, 2, 3]
        })
        
        # Combine the time series
        result = preprocessor._combine_time_series(data_arrays, files_df, 2000)
        
        # Check that the result has the expected structure
        assert "time" in result.dims
        assert len(result.time) == 3
        assert result.time.dtype == "datetime64[ns]"
        
        # Check that the dates were parsed correctly
        expected_dates = pd.to_datetime(["2000001", "2000002", "2000003"], format="%Y%j")
        np.testing.assert_array_equal(result.time.values, expected_dates.values)

    def test_calculate_statistics(self, mock_xarray_data, temp_paths):
        """Test calculating statistics from time series data."""
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"]
        )
        
        # Create a dataset from the mock data
        dataset = mock_xarray_data.to_dataset(name="LST")
        
        # Calculate statistics
        annual_stats, monthly_stats = preprocessor._calculate_statistics(dataset)
        
        # Check that annual stats have the expected variables
        assert "mean" in annual_stats.data_vars
        assert "median" in annual_stats.data_vars
        assert "std" in annual_stats.data_vars
        assert "count" in annual_stats.data_vars
        
        # Check that monthly stats have the expected variables
        assert "mean" in monthly_stats.data_vars
        assert "median" in monthly_stats.data_vars
        assert "std" in monthly_stats.data_vars
        assert "count" in monthly_stats.data_vars
        
        # Check that the time dimensions are correct
        assert len(annual_stats.time) == 1  # One year
        assert len(monthly_stats.time) == 1  # One month (all data is within a single month)

    @patch("preprocess.glass.rxr.open_rasterio")
    @patch("preprocess.glass.GCSClient")
    def test_process_file_group(self, mock_gcs_client_class, mock_open_rasterio, 
                             temp_paths, mock_gcs_client, mock_xarray_data):
        """Test processing a file group."""
        mock_gcs_client_class.return_value = mock_gcs_client
        mock_open_rasterio.return_value = mock_xarray_data
        
        # Create a preprocessor
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"],
            years=[2000]
        )
        
        # Create a test files_df
        files_df = pd.DataFrame({
            'path': ["path/to/file1.hdf", "path/to/file2.hdf", "path/to/file3.hdf"],
            'year': [2000, 2000, 2000],
            'day': [1, 2, 3]
        })
        
        # Mock methods to avoid actual processing
        preprocessor._combine_time_series = MagicMock(return_value=mock_xarray_data)
        preprocessor._calculate_statistics = MagicMock(return_value=(
            mock_xarray_data.to_dataset(name="mean"), 
            mock_xarray_data.to_dataset(name="monthly_mean")
        ))
        preprocessor._upload_to_cloud = MagicMock()
        
        # Process the file group
        output_path = temp_paths["intermediate_path"] / "test_output.zarr"
        preprocessor._process_file_group(files_df, 2000, output_path)
        
        # Check that the download method was called at least once
        assert mock_gcs_client.download_file.call_count >= 1
        
        # Check that our mocked methods were called
        preprocessor._combine_time_series.assert_called_once()
        preprocessor._calculate_statistics.assert_called_once()
        preprocessor._upload_to_cloud.assert_called_once()

    @patch("preprocess.glass.GCSClient")
    def test_upload_to_cloud(self, mock_gcs_client_class, temp_paths, mock_gcs_client):
        """Test uploading processed data to cloud storage."""
        mock_gcs_client_class.return_value = mock_gcs_client
        
        # Create a preprocessor
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"],
            data_source="MODIS",
            years=[2000]
        )
        
        # Create test zarr directories
        annual_path = Path(temp_paths["temp_dir"]) / "test_annual.zarr"
        monthly_path = Path(temp_paths["temp_dir"]) / "test_monthly.zarr"
        annual_path.mkdir()
        monthly_path.mkdir()
        
        # Create some test files in the zarr directories
        (annual_path / "file1.txt").write_text("test")
        (annual_path / "file2.txt").write_text("test")
        (monthly_path / "file1.txt").write_text("test")
        
        # Upload to cloud
        preprocessor._upload_to_cloud(annual_path, monthly_path, 2000)
        
        # Check that the upload method was called for each file
        assert mock_gcs_client.upload_file.call_count == 3

    @patch("preprocess.glass.GCSClient")
    def test_finalize_stage1(self, mock_gcs_client_class, temp_paths, mock_gcs_client):
        """Test finalizing stage 1 processing."""
        mock_gcs_client_class.return_value = mock_gcs_client
        
        # Configure mock to return processed files
        mock_gcs_client.list_existing_files = MagicMock(return_value=[
            "glass/LST/MODIS/Daily/1KM/intermediate/modis_2000_h08v05_annual.zarr/.zmetadata",
            "glass/LST/MODIS/Daily/1KM/intermediate/modis_2000_h09v05_annual.zarr/.zmetadata"
        ])
        
        # Create a preprocessor
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"],
            data_source="MODIS",
            years=[2000]
        )
        
        # Mock temporary file creation and upload
        with patch("tempfile.NamedTemporaryFile") as mock_temp_file, \
             patch("os.unlink") as mock_unlink:
            
            mock_file = MagicMock()
            mock_file.name = "temp_manifest.csv"
            mock_temp_file.return_value.__enter__.return_value = mock_file
            
            # Run finalize_stage1
            preprocessor.finalize_stage1()
            
            # Check that upload was called
            mock_gcs_client.upload_file.assert_called_once()
            mock_unlink.assert_called_once_with("temp_manifest.csv")

    def test_project_to_unified_grid(self, temp_paths):
        """Test project_to_unified_grid method."""
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"]
        )
        
        # This should just log a warning since it's not implemented yet
        with pytest.warns(UserWarning):
            preprocessor.project_to_unified_grid()


# Run the tests
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
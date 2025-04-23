import pytest
import pandas as pd
import numpy as np
import xarray as xr
import tempfile
from pathlib import Path
import os
import shutil
from unittest.mock import MagicMock, patch

from preprocess.glass import GlassPreprocessor
from gcs.client import GCSClient

# Sample data for testing
MOCK_MODIS_FILES = [
    "glass/LST/MODIS/Daily/1KM/GLASS06A01.V01.A2000001.h08v05.2022021.hdf",
    "glass/LST/MODIS/Daily/1KM/GLASS06A01.V01.A2000002.h08v05.2022021.hdf",
    "glass/LST/MODIS/Daily/1KM/GLASS06A01.V01.A2000003.h08v05.2022021.hdf",
    "glass/LST/MODIS/Daily/1KM/GLASS06A01.V01.A2000001.h09v05.2022021.hdf",
    "glass/LST/MODIS/Daily/1KM/GLASS06A01.V01.A2000002.h09v05.2022021.hdf",
    "glass/LST/MODIS/Daily/1KM/GLASS06A01.V01.A2001001.h08v05.2022021.hdf",
]

MOCK_AVHRR_FILES = [
    "glass/LST/AVHRR/0.05D/GLASS08B31.V40.A1982001.2021259.hdf",
    "glass/LST/AVHRR/0.05D/GLASS08B31.V40.A1982002.2021259.hdf",
    "glass/LST/AVHRR/0.05D/GLASS08B31.V40.A1982003.2021259.hdf",
    "glass/LST/AVHRR/0.05D/GLASS08B31.V40.A1983001.2021259.hdf",
]


@pytest.fixture
def temp_paths():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "input"
        output_path = Path(temp_dir) / "output"
        intermediate_path = Path(temp_dir) / "intermediate"

        input_path.mkdir(exist_ok=True)
        output_path.mkdir(exist_ok=True)
        intermediate_path.mkdir(exist_ok=True)

        yield {
            "temp_dir": temp_dir,
            "input_path": input_path,
            "output_path": output_path,
            "intermediate_path": intermediate_path
        }


@pytest.fixture
def mock_gcs_client():
    """Create a mock GCS client."""
    mock_client = MagicMock(spec=GCSClient)
    
    # Configure mock to return sample files
    def mock_list_files(prefix=None):
        if prefix == "glass/LST/MODIS/Daily/1KM/":
            return set(MOCK_MODIS_FILES)
        elif prefix == "glass/LST/AVHRR/0.05D/":
            return set(MOCK_AVHRR_FILES)
        else:
            return set()

    mock_client.list_existing_files.side_effect = mock_list_files
    
    # Mock successful download
    mock_client.download_file.return_value = True
    
    return mock_client


@pytest.fixture
def mock_xarray_data():
    """Create mock xarray data for testing."""
    # Create sample data
    times = pd.date_range('2000-01-01', periods=3)
    lats = np.linspace(30, 40, 10)
    lons = np.linspace(-110, -100, 10)
    
    # Create a DataArray with coordinates
    data = np.random.rand(3, 10, 10)
    da = xr.DataArray(
        data=data,
        dims=["time", "y", "x"],
        coords={
            "time": times,
            "y": lats,
            "x": lons,
        },
    )
    da.name = "mock_data"  # Assign a name to the DataArray
    return da


class TestGlassPreprocessor:
    """Test suite for the GlassPreprocessor class."""

    def test_initialization(self, temp_paths):
        """Test that initialization sets attributes correctly."""
        # Default initialization (MODIS)
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"]
        )
        
        assert preprocessor.data_source == "MODIS"
        assert preprocessor.path_prefix == "glass/LST/MODIS/Daily/1KM/"
        assert len(preprocessor.years) > 0
        assert preprocessor.grid_cells is None
        assert preprocessor.override is False
        
        # AVHRR initialization with specific years
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"],
            data_source="AVHRR",
            years=(1982, 1985),
            override=True
        )
        
        assert preprocessor.data_source == "AVHRR"
        assert preprocessor.path_prefix == "glass/LST/AVHRR/0.05D/"
        assert preprocessor.years == [1982, 1983, 1984, 1985]
        assert preprocessor.grid_cells is None
        assert preprocessor.override is True

    def test_initialization_with_invalid_data_source(self, temp_paths):
        """Test that initialization fails with invalid data source."""
        with pytest.raises(ValueError):
            GlassPreprocessor(
                temp_paths["input_path"],
                temp_paths["output_path"],
                temp_paths["intermediate_path"],
                data_source="INVALID"
            )

    def test_initialization_with_invalid_years(self, temp_paths):
        """Test that initialization fails with invalid years format."""
        with pytest.raises(ValueError):
            GlassPreprocessor(
                temp_paths["input_path"],
                temp_paths["output_path"],
                temp_paths["intermediate_path"],
                years="2000"  # Not a list or tuple
            )

    def test_parse_modis_filenames(self, temp_paths):
        """Test parsing of MODIS filenames."""
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"]
        )
        
        result_df = preprocessor._parse_modis_filenames(MOCK_MODIS_FILES)
        
        # Check that the DataFrame has the expected columns
        assert set(result_df.columns) == {"path", "year", "day", "h", "v"}
        
        # Check that the parsing is correct for a sample file
        sample_row = result_df[result_df.path == MOCK_MODIS_FILES[0]].iloc[0]
        assert sample_row.year == 2000
        assert sample_row.day == 1
        assert sample_row.h == 8
        assert sample_row.v == 5

    def test_parse_avhrr_filenames(self, temp_paths):
        """Test parsing of AVHRR filenames."""
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"],
            data_source="AVHRR"
        )
        
        result_df = preprocessor._parse_avhrr_filenames(MOCK_AVHRR_FILES)
        
        # Check that the DataFrame has the expected columns
        assert set(result_df.columns) == {"path", "year", "day"}
        
        # Check that the parsing is correct for a sample file
        sample_row = result_df[result_df.path == MOCK_AVHRR_FILES[0]].iloc[0]
        assert sample_row.year == 1982
        assert sample_row.day == 1

    @patch("preprocess.glass.GCSClient")
    def test_validate_input_success(self, mock_gcs_client_class, temp_paths, mock_gcs_client):
        """Test the validate_input method when files exist."""
        mock_gcs_client_class.return_value = mock_gcs_client
        
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"]
        )
        
        assert preprocessor.validate_input() is True

    @patch("preprocess.glass.GCSClient")
    def test_validate_input_no_files(self, mock_gcs_client_class, temp_paths):
        """Test the validate_input method when no files exist."""
        mock_client = MagicMock(spec=GCSClient)
        mock_client.list_existing_files.return_value = set()
        mock_gcs_client_class.return_value = mock_client
        
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"]
        )
        
        assert preprocessor.validate_input() is False

    @patch("preprocess.glass.GCSClient")
    def test_validate_input_error(self, mock_gcs_client_class, temp_paths):
        """Test the validate_input method when an error occurs."""
        mock_client = MagicMock(spec=GCSClient)
        mock_client.list_existing_files.side_effect = Exception("Connection error")
        mock_gcs_client_class.return_value = mock_client
        
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"]
        )
        
        assert preprocessor.validate_input() is False

    @patch("preprocess.glass.GCSClient")
    def test_list_files_from_gcs(self, mock_gcs_client_class, temp_paths, mock_gcs_client):
        """Test listing files from GCS."""
        mock_gcs_client_class.return_value = mock_gcs_client
        
        # Test MODIS listing
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"]
        )
        
        files = preprocessor._list_files_from_gcs()
        assert set(files) == set(MOCK_MODIS_FILES)
        
        # Test AVHRR listing
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"],
            data_source="AVHRR"
        )
        
        files = preprocessor._list_files_from_gcs()
        assert set(files) == set(MOCK_AVHRR_FILES)

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
        
        # Mock the combine_time_series and calculate_annual_stats methods
        preprocessor._combine_time_series = MagicMock(return_value=mock_xarray_data)
        preprocessor._calculate_annual_stats = MagicMock(return_value=mock_xarray_data.to_dataset())
        
        # Process the file group
        output_path = temp_paths["intermediate_path"] / "test_output.zarr"
        preprocessor._process_file_group(files_df, 2000, output_path)
        
        # Check that the methods were called correctly
        assert mock_gcs_client.download_file.call_count == 3
        preprocessor._combine_time_series.assert_called_once()
        preprocessor._calculate_annual_stats.assert_called_once()

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

    def test_calculate_annual_stats(self, temp_paths, mock_xarray_data):
        """Test calculating annual statistics."""
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"]
        )
        
        # Calculate statistics
        stats = preprocessor._calculate_annual_stats(mock_xarray_data)
        
        # Check that the output contains the expected variables
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "count" in stats
        assert "monthly_mean" in stats
        assert "monthly_median" in stats
        assert "monthly_std" in stats
        assert "monthly_count" in stats

    @patch("preprocess.glass.GCSClient")
    @patch.object(GlassPreprocessor, "_process_modis_data")
    def test_summarize_annual_means_modis(self, mock_process_modis, mock_gcs_client_class, 
                                         temp_paths, mock_gcs_client):
        """Test summarizing annual means for MODIS data."""
        mock_gcs_client_class.return_value = mock_gcs_client
        
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"],
            years=[2000]
        )
        
        # Run the method
        preprocessor.summarize_annual_means()
        
        # Check that the correct processing method was called
        mock_process_modis.assert_called_once()

    @patch("preprocess.glass.GCSClient")
    @patch.object(GlassPreprocessor, "_process_avhrr_data")
    def test_summarize_annual_means_avhrr(self, mock_process_avhrr, mock_gcs_client_class, 
                                         temp_paths, mock_gcs_client):
        """Test summarizing annual means for AVHRR data."""
        mock_gcs_client_class.return_value = mock_gcs_client
        
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"],
            data_source="AVHRR",
            years=[1982]
        )
        
        # Run the method
        preprocessor.summarize_annual_means()
        
        # Check that the correct processing method was called
        mock_process_avhrr.assert_called_once()

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
        # The _process_file_group should be called once for the h08v05 group
        assert preprocessor._process_file_group.call_count == 1

    @patch("preprocess.glass.GCSClient")
    def test_finalize_stage1(self, mock_gcs_client_class, temp_paths, mock_gcs_client):
        """Test finalizing stage 1 processing."""
        mock_gcs_client_class.return_value = mock_gcs_client
        
        # Create a preprocessor
        preprocessor = GlassPreprocessor(
            temp_paths["input_path"],
            temp_paths["output_path"],
            temp_paths["intermediate_path"],
            data_source="MODIS",
            years=[2000]
        )
        
        # Create some mock processed files
        modis_year_dir = temp_paths["intermediate_path"] / "modis" / "2000"
        modis_year_dir.mkdir(parents=True)
        (modis_year_dir / "h08v05.zarr").mkdir()
        (modis_year_dir / "h09v05.zarr").mkdir()
        
        # Run finalize_stage1
        preprocessor.finalize_stage1()
        
        # Check that manifest file was created
        manifest_path = temp_paths["intermediate_path"] / "MODIS_manifest.csv"
        assert manifest_path.exists()
        
        # Read the manifest and check its content
        manifest = pd.read_csv(manifest_path)
        assert len(manifest) == 2  # Should have two entries
        assert set(manifest.grid_cell.unique()) == {"h08v05", "h09v05"}

    def test_from_config(self, temp_paths):
        """Test creating a GlassPreprocessor from config."""
        config = {
            "input_path": str(temp_paths["input_path"]),
            "output_path": str(temp_paths["output_path"]),
            "intermediate_path": str(temp_paths["intermediate_path"]),
            "data_source": "AVHRR",
            "years": [1982, 1983],
            "override": True
        }
        
        # Create from config
        preprocessor = GlassPreprocessor.from_config(config)
        
        # Check that the attributes were set correctly
        assert preprocessor.data_source == "AVHRR"
        assert preprocessor.years == [1982, 1983]
        assert preprocessor.override is True
        assert preprocessor.input_path == Path(temp_paths["input_path"])
        assert preprocessor.output_path == Path(temp_paths["output_path"])
        assert preprocessor.intermediate_path == Path(temp_paths["intermediate_path"])
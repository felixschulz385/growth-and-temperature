import pytest
import tempfile
from pathlib import Path
import shutil
import logging
from unittest.mock import MagicMock, patch

from preprocess.base import AbstractPreprocessor

# Setup logging for tests
logging.basicConfig(level=logging.INFO)

# Define a concrete implementation for testing
class ConcretePreprocessor(AbstractPreprocessor):
    """Concrete implementation for testing."""
    
    def __init__(self, input_path, output_path, intermediate_path=None, stage="all"):
        super().__init__(input_path, output_path, intermediate_path)
        self.stage = stage

    def validate_input(self):
        return True
        
    def summarize_annual_means(self):
        # Create a test file in the intermediate path
        (self.intermediate_path / "test_file.txt").touch()
    
    def project_to_unified_grid(self):
        # Create a test file in the output path
        (self.output_path / "test_file.txt").touch()
        
    def finalize_stage1(self):
        # Create a manifest file
        (self.intermediate_path / "manifest.txt").touch()
        
    @classmethod
    def from_config(cls, config):
        return cls(
            config["input_path"], 
            config["output_path"],
            config.get("intermediate_path")  # Use .get to handle optional key
        )

    def preprocess(self):
        """Run the preprocessing based on the stage."""
        logging.info("Starting preprocessing...")
        if not self.validate_input():
            raise ValueError("Input validation failed.")
        
        if self.stage == "all":
            self.summarize_annual_means()
            self.finalize_stage1()
            self.project_to_unified_grid()
        elif self.stage == "stage1":
            self.summarize_annual_means()
            self.finalize_stage1()
        elif self.stage == "stage2":
            self.project_to_unified_grid()
        else:
            raise ValueError(f"Invalid stage: {self.stage}")
        
        logging.info("Preprocessing completed successfully")


@pytest.fixture
def temp_paths():
    """Create temporary directories for testing."""
    temp_dir = tempfile.mkdtemp()
    input_path = Path(temp_dir) / "input"
    output_path = Path(temp_dir) / "output"
    intermediate_path = Path(temp_dir) / "intermediate"
    
    # Create the directories
    input_path.mkdir(exist_ok=True)
    output_path.mkdir(exist_ok=True)
    intermediate_path.mkdir(exist_ok=True)
    
    # Return as a dict for easier access
    paths = {
        "temp_dir": temp_dir,
        "input_path": input_path,
        "output_path": output_path,
        "intermediate_path": intermediate_path
    }
    
    yield paths
    
    # Cleanup after test
    shutil.rmtree(temp_dir)


def test_initialization(temp_paths):
    """Test that initialization sets paths correctly."""
    preprocessor = ConcretePreprocessor(
        temp_paths["input_path"], 
        temp_paths["output_path"], 
        temp_paths["intermediate_path"]
    )
    
    assert preprocessor.input_path == Path(temp_paths["input_path"])
    assert preprocessor.output_path == Path(temp_paths["output_path"])
    assert preprocessor.intermediate_path == Path(temp_paths["intermediate_path"])
    assert preprocessor.stage == "all"  # Default stage


def test_initialization_with_string_paths(temp_paths):
    """Test that initialization converts string paths to Path objects."""
    preprocessor = ConcretePreprocessor(
        str(temp_paths["input_path"]), 
        str(temp_paths["output_path"]), 
        str(temp_paths["intermediate_path"])
    )
    
    assert isinstance(preprocessor.input_path, Path)
    assert isinstance(preprocessor.output_path, Path)
    assert isinstance(preprocessor.intermediate_path, Path)


def test_initialization_with_stage(temp_paths):
    """Test that initialization accepts stage parameter."""
    preprocessor = ConcretePreprocessor(
        temp_paths["input_path"], 
        temp_paths["output_path"], 
        temp_paths["intermediate_path"], 
        stage="stage1"
    )
    
    assert preprocessor.stage == "stage1"


def test_preprocess_all_stages(temp_paths):
    """Test that preprocess runs all stages when stage is 'all'."""
    preprocessor = ConcretePreprocessor(
        temp_paths["input_path"], 
        temp_paths["output_path"], 
        temp_paths["intermediate_path"],
        stage="all"
    )
    
    # Mock the methods to verify they're called
    preprocessor.validate_input = MagicMock(return_value=True)
    preprocessor.summarize_annual_means = MagicMock()
    preprocessor.finalize_stage1 = MagicMock()
    preprocessor.project_to_unified_grid = MagicMock()
    
    preprocessor.preprocess()
    
    preprocessor.validate_input.assert_called_once()
    preprocessor.summarize_annual_means.assert_called_once()
    preprocessor.finalize_stage1.assert_called_once()
    preprocessor.project_to_unified_grid.assert_called_once()


def test_preprocess_stage1(temp_paths):
    """Test that preprocess runs only stage1 when specified."""
    preprocessor = ConcretePreprocessor(
        temp_paths["input_path"], 
        temp_paths["output_path"], 
        temp_paths["intermediate_path"],
        stage="stage1"
    )
    
    # Mock the methods to verify they're called
    preprocessor.validate_input = MagicMock(return_value=True)
    preprocessor.summarize_annual_means = MagicMock()
    preprocessor.finalize_stage1 = MagicMock()
    preprocessor.project_to_unified_grid = MagicMock()
    
    preprocessor.preprocess()
    
    preprocessor.validate_input.assert_called_once()
    preprocessor.summarize_annual_means.assert_called_once()
    preprocessor.finalize_stage1.assert_called_once()
    preprocessor.project_to_unified_grid.assert_not_called()  # Should not be called


def test_preprocess_stage2(temp_paths):
    """Test that preprocess runs only stage2 when specified."""
    preprocessor = ConcretePreprocessor(
        temp_paths["input_path"], 
        temp_paths["output_path"], 
        temp_paths["intermediate_path"],
        stage="stage2"
    )
    
    # Mock the methods to verify they're called
    preprocessor.validate_input = MagicMock(return_value=True)
    preprocessor.summarize_annual_means = MagicMock()
    preprocessor.finalize_stage1 = MagicMock()
    preprocessor.project_to_unified_grid = MagicMock()
    
    preprocessor.preprocess()
    
    preprocessor.validate_input.assert_called_once()
    preprocessor.summarize_annual_means.assert_not_called()  # Should not be called
    preprocessor.finalize_stage1.assert_not_called()  # Should not be called
    preprocessor.project_to_unified_grid.assert_called_once()


def test_preprocess_invalid_stage(temp_paths):
    """Test that preprocess raises error for invalid stage."""
    preprocessor = ConcretePreprocessor(
        temp_paths["input_path"], 
        temp_paths["output_path"], 
        temp_paths["intermediate_path"],
        stage="invalid_stage"
    )
    
    # Mock methods to ensure they're not called
    preprocessor.validate_input = MagicMock(return_value=True)
    preprocessor.summarize_annual_means = MagicMock()
    preprocessor.project_to_unified_grid = MagicMock()
    
    with pytest.raises(ValueError):
        preprocessor.preprocess()


def test_validate_input_fails(temp_paths):
    """Test that preprocess stops if validate_input returns False."""
    preprocessor = ConcretePreprocessor(
        temp_paths["input_path"], 
        temp_paths["output_path"], 
        temp_paths["intermediate_path"]
    )
    
    # Override validate_input to return False
    preprocessor.validate_input = MagicMock(return_value=False)
    preprocessor.summarize_annual_means = MagicMock()
    preprocessor.project_to_unified_grid = MagicMock()
    
    with pytest.raises(ValueError):
        preprocessor.preprocess()
        
    # Ensure the other methods weren't called
    preprocessor.summarize_annual_means.assert_not_called()
    preprocessor.project_to_unified_grid.assert_not_called()


def test_from_config(temp_paths):
    """Test that from_config class method works correctly."""
    config = {
        "input_path": temp_paths["input_path"],
        "output_path": temp_paths["output_path"],
        "intermediate_path": temp_paths["intermediate_path"],
        "stage": "stage1",
        "extra_param": "value"
    }
    
    preprocessor = ConcretePreprocessor.from_config(config)
    
    assert preprocessor.input_path == Path(temp_paths["input_path"])
    assert preprocessor.output_path == Path(temp_paths["output_path"])
    assert preprocessor.intermediate_path == Path(temp_paths["intermediate_path"])
    assert preprocessor.stage == "all"  # Default should be used since we didn't handle stage in from_config


def test_actual_processing(temp_paths):
    """Test the actual processing functionality."""
    # Create a concrete preprocessor instance
    preprocessor = ConcretePreprocessor(
        temp_paths["input_path"], 
        temp_paths["output_path"], 
        temp_paths["intermediate_path"]
    )
    
    # Run the preprocessing
    preprocessor.preprocess()
    
    # Check if the expected files were created
    assert (temp_paths["intermediate_path"] / "test_file.txt").exists()
    assert (temp_paths["output_path"] / "test_file.txt").exists()
    assert (temp_paths["intermediate_path"] / "manifest.txt").exists()
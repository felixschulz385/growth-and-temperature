import abc
from pathlib import Path
from typing import Dict, Any, Optional, Union, List


class AbstractPreprocessor(abc.ABC):
    """
    Abstract base class defining the template for preprocessing geodata.
    
    This class implements a template method pattern with two main stages:
    1. Summarize geodata into annual means while preserving subfile structure and projection
    2. Project the data onto a unified grid
    
    Designed for large datasets that remain on disk. Stage 1 and Stage 2 can be run on
    different devices, with the intermediate data transferred between them.
    """
    
    def __init__(self, input_path: Union[str, Path], output_path: Union[str, Path], 
                 intermediate_path: Optional[Union[str, Path]] = None, **kwargs):
        """
        Initialize the preprocessor with input and output paths.
        
        Args:
            input_path: Path to input data
            output_path: Path where final processed data will be stored
            intermediate_path: Path where Stage 1 output will be stored for transfer to Stage 2
                              (if None, uses a subdirectory of output_path)
            **kwargs: Additional configuration parameters
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.intermediate_path = Path(intermediate_path) if intermediate_path else self.output_path / "intermediate"
        self.config = kwargs
        self.stage = kwargs.get('stage', 'all')  # Options: 'stage1', 'stage2', 'all'
        
    def preprocess(self) -> None:
        """
        Execute the preprocessing workflow according to the specified stage.
        
        The workflow can be run in three modes:
        - 'all': Run both Stage 1 and Stage 2 in sequence
        - 'stage1': Only run the summarize_annual_means step
        - 'stage2': Only run the project_to_unified_grid step (assuming Stage 1 data is available)
        """
        # Validate input before processing
        if not self.validate_input():
            raise ValueError(f"Invalid input at {self.input_path}")
            
        if self.stage in ('all', 'stage1'):
            # Ensure intermediate directory exists
            self.intermediate_path.mkdir(parents=True, exist_ok=True)
            
            # Stage 1: Summarize geodata into annual means
            self.summarize_annual_means()
            
            # Finalization for Stage 1 (if only running Stage 1)
            if self.stage == 'stage1':
                self.finalize_stage1()
        
        if self.stage in ('all', 'stage2'):
            # Ensure output directory exists
            self.output_path.mkdir(parents=True, exist_ok=True)
            
            # Stage 2: Project data onto unified grid
            self.project_to_unified_grid()
            
            # Finalization for Stage 2
            self.finalize()
    
    @abc.abstractmethod
    def summarize_annual_means(self) -> None:
        """
        Stage 1: Summarize geodata into annual means.
        
        This method should:
        - Read the input geodata from disk
        - Calculate annual means from the data
        - Preserve the original subfile structure
        - Maintain the original projection
        - Save results to intermediate_path for later use in Stage 2
        
        Note: Results are stored on disk rather than returned in memory
        due to the large data size.
        """
        pass
    
    @abc.abstractmethod
    def project_to_unified_grid(self) -> None:
        """
        Stage 2: Project data onto a unified grid.
        
        This method should:
        - Read data from intermediate_path (output of Stage 1)
        - Reproject it to a standardized grid
        - Save the results to the output_path location
        
        Note: Data is read from disk rather than passed as a parameter
        due to the large data size and potential execution on different devices.
        """
        pass
    
    def finalize_stage1(self) -> None:
        """
        Finalization step after Stage 1 is complete.
        
        This can include:
        - Validation of intermediate output
        - Generation of metadata or manifest for data transfer
        - Compression of output for efficient transfer
        
        By default, does nothing but can be overridden by concrete implementations.
        """
        pass
    
    def finalize(self) -> None:
        """
        Finalization step after Stage 2 is complete.
        
        This can include:
        - Cleanup of temporary files
        - Validation of output data
        - Generation of metadata or reports
        
        By default, does nothing but can be overridden by concrete implementations.
        """
        pass
    
    def validate_input(self) -> bool:
        """
        Validate that the input data exists and is in an expected format.
        
        For Stage 1, checks the original input path.
        For Stage 2, checks the intermediate path.
        
        Returns:
            True if validation passes, False otherwise
        """
        if self.stage == 'stage1' or self.stage == 'all':
            return self.input_path.exists()
        elif self.stage == 'stage2':
            return self.intermediate_path.exists()
        return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return metadata about the preprocessing operation.
        
        Returns:
            Dictionary containing metadata about the preprocessing
        """
        return {
            "input_path": str(self.input_path),
            "intermediate_path": str(self.intermediate_path),
            "output_path": str(self.output_path),
            "stage": self.stage,
            "config": self.config
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AbstractPreprocessor":
        """
        Factory method to create a preprocessor from configuration dictionary.
        
        Args:
            config: Dictionary containing configuration parameters
            
        Returns:
            An instance of the preprocessor
        """
        input_path = config.pop("input_path")
        output_path = config.pop("output_path")
        intermediate_path = config.pop("intermediate_path", None)
        return cls(input_path, output_path, intermediate_path, **config)
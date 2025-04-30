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
    
    def __init__(self, **kwargs):
        """
        Initialize the preprocessor with configuration parameters.
        
        Args:
            **kwargs: Configuration parameters
        """
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
            raise ValueError("Invalid input data")
            
        if self.stage in ('all', 'stage1'):            
            # Stage 1: Summarize geodata into annual means
            self.summarize_annual_means()
            
            # Finalization for Stage 1 (if only running Stage 1)
            if self.stage == 'stage1':
                self.finalize_stage1()
        
        if self.stage in ('all', 'stage2'):            
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
        - Save results for later use in Stage 2
        
        Note: Results are stored on disk rather than returned in memory
        due to the large data size.
        """
        pass
    
    @abc.abstractmethod
    def project_to_unified_grid(self) -> None:
        """
        Stage 2: Project data onto a unified grid.
        
        This method should:
        - Read data from output of Stage 1
        - Reproject it to a standardized grid
        - Save the final results
        
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
        
        Returns:
            True if validation passes, False otherwise
        """
        return True
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return metadata about the preprocessing operation.
        
        Returns:
            Dictionary containing metadata about the preprocessing
        """
        return {
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
        return cls(**config)
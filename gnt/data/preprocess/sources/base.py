import abc
from typing import Dict, Any, List, Tuple
import logging
import os
import xarray as xr

logger = logging.getLogger(__name__)

class AbstractPreprocessor(abc.ABC):
    """
    Abstract base class for geodata preprocessors.
    Enforces the interface for common preprocessing methods.
    """

    def __init__(self, **kwargs):
        self.config = kwargs

    @abc.abstractmethod
    def get_preprocessing_targets(self, stage: str, year_range: Tuple[int, int] = None) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def get_hpc_output_path(self, stage: str) -> str:
        pass

    @abc.abstractmethod
    def process_target(self, target: Dict[str, Any]) -> bool:
        pass

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: Dict[str, Any]):
        pass

    def _process_tabular_target(self, target: Dict[str, Any]) -> bool:
        """
        Default implementation for tabular processing using the common tabularization module.
        This creates parquet tiles that are ready for assembly workflow consumption.
        """
        logger.info("Starting tabular stage processing with parquet tile output")
        
        output_path = self._strip_remote_prefix(target['output_path'])
        source_files = target.get('source_files', [])
        
        override = getattr(self, 'override', False)
        if not override and os.path.exists(output_path):
            logger.info(f"Skipping tabular processing, output already exists: {output_path}")
            return True
        
        if not source_files:
            logger.error("No source files provided for tabular processing")
            return False
        
        source_file = source_files[0]  # Should be single zarr file
        
        if not os.path.exists(source_file):
            logger.error(f"Source file does not exist: {source_file}")
            return False
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # Initialize Dask client if available
            dask_context_manager = getattr(self, '_initialize_dask_client', None)
            if dask_context_manager:
                with dask_context_manager() as client:
                    return self._process_tabular(source_file, output_path)
            else:
                return self._process_tabular(source_file, output_path)
                
        except Exception as e:
            logger.exception(f"Error in tabular processing: {e}")
            return False

    def _process_tabular(self, source_file: str, output_path: str) -> bool:
        """Process tabular conversion using the common implementation."""
        from gnt.data.preprocess.common.tabularization import process_zarr_to_parquet
        
        logger.info("Processing zarr to parquet using common vectorized approach")
        
        # Get hpc_root from instance attributes
        hpc_root = getattr(self, 'hpc_root', None)
        
        # Load the zarr dataset
        logger.info("Loading zarr dataset")
        ds = xr.open_zarr(source_file, consolidated=False, mask_and_scale=False)
        
        # Process using common implementation - creates parquet tiles
        success = process_zarr_to_parquet(
            ds=ds,
            output_path=output_path,
            hpc_root=hpc_root
        )
        
        if success:
            logger.info("Tabular stage processing completed successfully")
        
        return success

    def _strip_remote_prefix(self, path):
        """Remove scp/ssh prefix like user@host: from paths. Override in subclasses if needed."""
        if isinstance(path, str):
            import re
            return re.sub(r"^[^@]+@[^:]+:", "", path)
        return path
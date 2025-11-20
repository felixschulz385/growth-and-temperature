"""
Factory module for creating preprocessor instances.

This module handles the dynamic loading and instantiation of preprocessor classes
based on their names, providing a clean separation between class discovery and usage.
"""

import importlib
import logging
from typing import Any, Dict, Type

logger = logging.getLogger(__name__)

def get_preprocessor_class(preprocessor_name: str) -> Type:
    """
    Dynamically import and return the preprocessor class.
    
    Args:
        preprocessor_name: Name of the preprocessor class to import
        
    Returns:
        The preprocessor class
    """
    try:
        preprocessor_name = preprocessor_name.lower()
        
        if preprocessor_name == 'eog':
            from gnt.data.preprocess.sources.eog import EOGPreprocessor
            return EOGPreprocessor
        elif preprocessor_name.startswith('glass'):
            class_name = 'GlassPreprocessor'
            module_path = f"gnt.data.preprocess.sources.glass"
        elif preprocessor_name in ['ntl_harm', 'ntlharm', 'harmonized_ntl']:
            class_name = 'NTLHarmPreprocessor'
            module_path = f"gnt.data.preprocess.sources.ntl_harm"
        elif preprocessor_name == 'misc':
            from gnt.data.preprocess.sources.misc import MiscPreprocessor
            return MiscPreprocessor
        elif preprocessor_name == 'plad':
            from gnt.data.preprocess.sources.plad import PLADPreprocessor
            return PLADPreprocessor
        elif preprocessor_name == 'berman_mining':
            from gnt.data.preprocess.sources.berman_mining import BermanMiningPreprocessor
            return BermanMiningPreprocessor
        else:
            # By convention, the class name is expected to be CamelCase
            class_name = ''.join(word.capitalize() for word in preprocessor_name.split('_')) + 'Preprocessor'
            module_path = f"gnt.data.preprocess.sources.{preprocessor_name.lower()}"

        module = importlib.import_module(module_path)
        
        if not hasattr(module, class_name):
            raise AttributeError(f"Module {module_path} does not have class {class_name}")
        
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import preprocessor {preprocessor_name}: {str(e)}")
        raise

def create_preprocessor(preprocessor_name: str, config: Dict[str, Any]) -> Any:
    """
    Create a preprocessor instance based on name and configuration.
    
    Args:
        preprocessor_name: Name of the preprocessor to create
        config: Configuration dictionary for the preprocessor
        
    Returns:
        An instance of the specified preprocessor
    """
    PreprocessorClass = get_preprocessor_class(preprocessor_name)
    # Remove any tabular config before passing to preprocessor
    config.pop('tabular_batch_size', None)
    return PreprocessorClass.from_config(config)


def create_source(source_name: str, config: Dict[str, Any]):
    """Create a data source instance for the given source name and configuration."""
    source_name_lower = source_name.lower()
    
    if source_name_lower == 'eog':
        from gnt.data.download.sources.eog import EOGDataSource
        return EOGDataSource(**config)
    elif source_name_lower == 'misc':
        from gnt.data.download.sources.misc import MiscDataSource
        return MiscDataSource(**config)
    elif source_name_lower == 'plad':
        # PLAD doesn't need a separate data source as it works with local files
        return None
    elif source_name_lower == 'berman_mining':
        # Berman mining doesn't need a separate data source as it works with local files
        return None
    # GLASSLSTDataSource
    elif source_name_lower == "glasslstdatasource":
        if 'base_url' in config:
            source_config['base_url'] = config['base_url']
        if 'output_path' in config:
            source_config['output_path'] = config['output_path']
        elif 'data_path' in config:
            source_config['output_path'] = config['data_path']
        if 'file_extensions' in config:
            source_config['file_extensions'] = config['file_extensions']
    else:
        raise ValueError(f"Unknown source: {source_name}")
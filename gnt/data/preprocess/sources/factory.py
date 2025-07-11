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
        # Special case for EOG preprocessor (all caps)
        if "eog" in preprocessor_name.lower():
            class_name = 'EOGPreprocessor'
            module_path = f"gnt.data.preprocess.sources.eog"
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
    return PreprocessorClass.from_config(config)

def get_source_class(source_name: str) -> Type:
    """
    Dynamically import and return the data source class.
    
    Args:
        source_name: Name of the data source class to import
        
    Returns:
        The data source class
    """
    try:
        # Map source names to their modules and classes
        source_mapping = {
            'eog_dmsp': ('gnt.data.download.sources.eog', 'EOGDataSource'),
            'eog_viirs': ('gnt.data.download.sources.eog', 'EOGDataSource'),
            'eog_dvnl': ('gnt.data.download.sources.eog', 'EOGDataSource'),
            'glass_modis': ('gnt.data.download.sources.glass_lst', 'GLASSLSTDataSource'),
            'misc': ('gnt.data.download.sources.misc', 'MiscDataSource'),
        }
        
        if source_name.lower() in source_mapping:
            module_path, class_name = source_mapping[source_name.lower()]
        else:
            # Fallback to convention-based mapping
            class_name = ''.join(word.capitalize() for word in source_name.split('_')) + 'DataSource'
            module_path = f"gnt.data.download.sources.{source_name.lower()}"

        module = importlib.import_module(module_path)
        
        if not hasattr(module, class_name):
            raise AttributeError(f"Module {module_path} does not have class {class_name}")
        
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import data source {source_name}: {str(e)}")
        raise

def create_source(source_name: str, config: Dict[str, Any]) -> Any:
    """
    Create a data source instance based on name and configuration.
    
    Args:
        source_name: Name of the data source to create
        config: Configuration dictionary for the data source
        
    Returns:
        An instance of the specified data source
    """
    SourceClass = get_source_class(source_name)
    
    # Extract relevant parameters for data source creation
    source_config = {}

    # Only pass parameters accepted by the data source class
    # EOGDataSource expects: base_url, file_extensions, output_path
    # MiscDataSource expects: files, output_path
    # GLASSLSTDataSource expects: base_url, file_extensions, output_path

    # EOGDataSource
    if SourceClass.__name__ == "EOGDataSource":
        if 'base_url' in config:
            source_config['base_url'] = config['base_url']
        # Use output_path if available, otherwise fall back to data_path
        if 'output_path' in config:
            source_config['output_path'] = config['output_path']
        elif 'data_path' in config:
            source_config['output_path'] = config['data_path']
        if 'file_extensions' in config:
            source_config['file_extensions'] = config['file_extensions']
    # MiscDataSource
    elif SourceClass.__name__ == "MiscDataSource":
        if 'files' in config:
            source_config['files'] = config['files']
        if 'output_path' in config:
            source_config['output_path'] = config['output_path']
        elif 'data_path' in config:
            source_config['output_path'] = config['data_path']
    # GLASSLSTDataSource
    elif SourceClass.__name__ == "GLASSLSTDataSource":
        if 'base_url' in config:
            source_config['base_url'] = config['base_url']
        if 'output_path' in config:
            source_config['output_path'] = config['output_path']
        elif 'data_path' in config:
            source_config['output_path'] = config['data_path']
        if 'file_extensions' in config:
            source_config['file_extensions'] = config['file_extensions']
    else:
        # Fallback: pass only keys that match the class __init__ signature
        import inspect
        sig = inspect.signature(SourceClass.__init__)
        for k in config:
            if k in sig.parameters and k != 'self':
                source_config[k] = config[k]
    
    return SourceClass(**source_config)
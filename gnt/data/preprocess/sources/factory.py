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
        # Assume preprocessors are in the gnt.data.preprocess package
        module_path = f"gnt.data.preprocess.sources.{preprocessor_name.lower()}"
        module = importlib.import_module(module_path)
        
        # Special case for EOG preprocessor (all caps)
        if preprocessor_name.lower() == "eog":
            class_name = 'EOGPreprocessor'
        else:
            # By convention, the class name is expected to be CamelCase
            class_name = ''.join(word.capitalize() for word in preprocessor_name.split('_')) + 'Preprocessor'
        
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
"""
Configuration handling for data assembly.

Provides functions for loading, validating, and deriving configuration values.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from gnt.data.assemble.utils import strip_remote_prefix
from gnt.data.assemble.constants import (
    DEFAULT_TILE_SIZE,
    DEFAULT_COMPRESSION,
    DEFAULT_RESAMPLING_METHOD,
    DEFAULT_DASK_DASHBOARD_PORT,
    DEFAULT_WORKER_THREADS_PER_CPU,
    DEFAULT_WORKER_FRACTION,
)

logger = logging.getLogger(__name__)


@dataclass
class DaskConfig:
    """Configuration for Dask distributed processing."""
    threads: Optional[int] = None
    memory_limit: Optional[str] = None
    dashboard_port: int = DEFAULT_DASK_DASHBOARD_PORT
    temp_dir: Optional[str] = None
    worker_threads_per_cpu: int = DEFAULT_WORKER_THREADS_PER_CPU
    worker_fraction: float = DEFAULT_WORKER_FRACTION
    
    def to_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs dict for DaskClientContextManager, excluding None values."""
        return {k: v for k, v in {
            'threads': self.threads,
            'memory_limit': self.memory_limit,
            'dashboard_port': self.dashboard_port,
            'temp_dir': self.temp_dir,
            'worker_threads_per_cpu': self.worker_threads_per_cpu,
            'worker_fraction': self.worker_fraction,
        }.items() if v is not None}


@dataclass
class ProcessingConfig:
    """Configuration for tile processing parameters."""
    resolution: Optional[float] = None
    tile_size: int = DEFAULT_TILE_SIZE
    compression: str = DEFAULT_COMPRESSION
    year_range: Optional[tuple] = None
    apply_land_mask: bool = False
    land_mask_path: Optional[str] = None
    dask: DaskConfig = field(default_factory=DaskConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProcessingConfig':
        """Create ProcessingConfig from dictionary."""
        dask_dict = config_dict.get('dask', {})
        dask_config = DaskConfig(
            threads=dask_dict.get('threads'),
            memory_limit=dask_dict.get('memory_limit'),
            dashboard_port=dask_dict.get('dashboard_port', DEFAULT_DASK_DASHBOARD_PORT),
            temp_dir=dask_dict.get('temp_dir'),
            worker_threads_per_cpu=dask_dict.get('worker_threads_per_cpu', DEFAULT_WORKER_THREADS_PER_CPU),
            worker_fraction=dask_dict.get('worker_fraction', DEFAULT_WORKER_FRACTION),
        )
        
        year_range = config_dict.get('year_range')
        if year_range and isinstance(year_range, list):
            year_range = tuple(year_range)
        
        return cls(
            resolution=config_dict.get('resolution'),
            tile_size=config_dict.get('tile_size', DEFAULT_TILE_SIZE),
            compression=config_dict.get('compression', DEFAULT_COMPRESSION),
            year_range=year_range,
            apply_land_mask=config_dict.get('apply_land_mask', False),
            land_mask_path=config_dict.get('land_mask_path'),
            dask=dask_config,
        )


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    path: str
    resampling: str = DEFAULT_RESAMPLING_METHOD
    columns: Optional[List[str]] = None
    column_prefix: Optional[str] = None
    winsorize: Optional[float] = None
    index_cols: List[str] = field(default_factory=lambda: ['pixel_id'])
    
    @classmethod
    def from_dict(cls, name: str, config_dict: Dict[str, Any]) -> 'DatasetConfig':
        """Create DatasetConfig from dictionary."""
        return cls(
            name=name,
            path=config_dict['path'],
            resampling=config_dict.get('resampling', DEFAULT_RESAMPLING_METHOD),
            columns=config_dict.get('columns'),
            column_prefix=config_dict.get('column_prefix'),
            winsorize=config_dict.get('winsorize'),
            index_cols=config_dict.get('index_cols', ['pixel_id']),
        )


def derive_hpc_root(assembly_config: Dict[str, Any], full_config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Derive hpc_root from assembly configuration, checking multiple sources.
    
    Args:
        assembly_config: Assembly configuration dictionary
        full_config: Full configuration dictionary containing HPC settings
        
    Returns:
        HPC root path or None if not found
    """
    # Check full config for hpc settings
    if full_config:
        hpc_config = full_config.get('hpc', {})
        hpc_target = hpc_config.get('target')
        
        if hpc_target:
            return strip_remote_prefix(hpc_target)
    
    logger.warning("Could not derive hpc_root from configuration")
    return None


def apply_cli_overrides(assembly_config: Dict[str, Any], cli_overrides: Dict[str, Any]) -> None:
    """
    Apply CLI overrides to assembly configuration in-place.
    
    Args:
        assembly_config: Assembly configuration to modify
        cli_overrides: CLI override values to apply
    """
    if not cli_overrides:
        return
    
    processing_config = assembly_config.setdefault('processing', {})
    dask_config = processing_config.setdefault('dask', {})
    
    # Dask-related CLI overrides
    dask_overrides = {
        'dask_threads': ('threads', 'dask threads'),
        'dask_memory_limit': ('memory_limit', 'dask memory limit'),
        'temp_dir': ('temp_dir', 'temp dir'),
        'dashboard_port': ('dashboard_port', 'dashboard port'),
    }
    
    for cli_key, (config_key, log_name) in dask_overrides.items():
        if cli_key in cli_overrides:
            dask_config[config_key] = cli_overrides[cli_key]
            logger.info(f"Overriding {log_name} from CLI: {cli_overrides[cli_key]}")
    
    # Processing overrides
    processing_overrides = {
        'tile_size': 'tile size',
        'compression': 'compression',
        'assembly_mode': 'assembly mode',
        'datasource': 'datasource',
        'overwrite': 'overwrite',
    }
    
    for key, log_name in processing_overrides.items():
        if key in cli_overrides:
            processing_config[key] = cli_overrides[key]
            logger.info(f"Overriding {log_name} from CLI: {cli_overrides[key]}")


def validate_assembly_config(assembly_config: Dict[str, Any]) -> List[str]:
    """
    Validate assembly configuration and return list of errors.
    
    Args:
        assembly_config: Assembly configuration to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if 'output_path' not in assembly_config:
        errors.append("Missing required 'output_path' in assembly configuration")
    
    if 'datasets' not in assembly_config:
        errors.append("Missing required 'datasets' in assembly configuration")
    elif not assembly_config['datasets']:
        errors.append("'datasets' configuration is empty")
    else:
        for name, config in assembly_config['datasets'].items():
            if 'path' not in config:
                errors.append(f"Dataset '{name}' missing required 'path' field")
            elif not os.path.exists(config['path']):
                logger.warning(f"Dataset path does not exist: {config['path']}")
            
            # Validate index_cols if specified
            index_cols = config.get('index_cols')
            if index_cols is not None:
                if not isinstance(index_cols, list):
                    errors.append(f"Dataset '{name}' index_cols must be a list, got {type(index_cols)}")
                elif not index_cols:
                    errors.append(f"Dataset '{name}' index_cols cannot be empty")
                elif not all(isinstance(col, str) for col in index_cols):
                    errors.append(f"Dataset '{name}' index_cols must contain only strings")
            else:
                logger.debug(f"Dataset '{name}' using default index_cols: ['pixel_id']")
    
    processing = assembly_config.get('processing', {})
    year_range = processing.get('year_range')
    if year_range:
        if not isinstance(year_range, (list, tuple)) or len(year_range) != 2:
            errors.append("'year_range' must be a list/tuple of two years [start, end]")
        elif year_range[0] > year_range[1]:
            errors.append(f"'year_range' start ({year_range[0]}) must be <= end ({year_range[1]})")
    
    return errors

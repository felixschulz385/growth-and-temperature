"""
Configuration handling for data assembly.

Provides functions for loading, validating, and deriving configuration values.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from src.config.runtime import resolve_data_root
from src.data.assemble.constants import (
    DEFAULT_TILE_SIZE,
    DEFAULT_COMPRESSION,
    DEFAULT_RESAMPLING_METHOD,
    DEFAULT_DASK_DASHBOARD_PORT,
    DEFAULT_WORKER_THREADS_PER_CPU,
    DEFAULT_WORKER_FRACTION,
)
from src.data.assemble.grid_shake import normalize_grid_shake_offsets
from src.data.sources import layout
from src.data.sources.verify import verify_grid_output

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


def derive_data_root(assembly_config: Dict[str, Any], full_config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Derive the local project data root from configuration.
    
    Args:
        assembly_config: Assembly configuration dictionary
        full_config: Full configuration dictionary containing runtime settings
        
    Returns:
        Local project data root or None if not found
    """
    if full_config:
        data_root = resolve_data_root(full_config)
        if data_root:
            return data_root
    
    logger.warning("Could not derive data_root from configuration")
    return None


def derive_hpc_root(assembly_config: Dict[str, Any], full_config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Backward-compatible alias for older callers."""
    return derive_data_root(assembly_config, full_config)


def resolve_dataset_paths(
    assembly_config: Dict[str, Any],
    data_root: str,
    grid_id: str = layout.LEGACY_GRID_ID,
) -> None:
    """Resolve each dataset's `path` from `data_path`+`family` (+ optional
    per-dataset `grid_id`) when `path` isn't given explicitly, via
    `src.data.sources.layout.grid_store_path` -- the same helper every
    migrated PREPARE source uses to compute its own GRID-stage output
    directory, so an assembly config can point at a source's output by name
    instead of hand-pasting a path that silently drifts out of sync when that
    source's layout changes. Mutates `assembly_config['datasets']` in place.

    `grid_id` is the run's grid (from `pipeline.grid`); each dataset inherits it
    unless it sets its own `grid_id`. A dataset with an explicit `path` keeps it
    (escape hatch for one-off/manual paths, e.g. join_on sidecars outside the
    standard GRID layout); a *relative* explicit path is resolved against
    `data_root` so the config stays machine-agnostic (no `${DATA_NOBACKUP}`).
    """
    for name, cfg in assembly_config.get('datasets', {}).items():
        explicit = cfg.get('path')
        if explicit:
            if not os.path.isabs(explicit):
                cfg['path'] = os.path.join(data_root, explicit)
                logger.debug(f"Dataset '{name}': resolved relative path -> {cfg['path']}")
            continue
        data_path = cfg.get('data_path')
        family = cfg.get('family')
        if not data_path or not family:
            continue  # validate_assembly_config reports the resulting missing 'path'
        dataset_grid_id = cfg.get('grid_id', grid_id)
        cfg['path'] = layout.grid_store_path(
            data_root, data_path, grid_id=dataset_grid_id, family=family, suffix=""
        )
        logger.debug(f"Dataset '{name}': resolved path from data_path/family -> {cfg['path']}")


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
        'grid_label': 'grid label',
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
                errors.append(
                    f"Dataset '{name}' path does not exist: {config['path']} "
                    f"-- build it with `data run --source {name} --step prepare`"
                )
            else:
                # Use the same verification kwargs the source itself declares
                # (attached as `_verification` by run_workflow_with_config), so
                # the assembly gate agrees with `data summary`'s `verified`
                # column -- e.g. honoring `sparse_vars` for by-design-sparse
                # columns like snl_mining's mine_priceshock_*.
                vm = config.get('_verification') or {}
                result = verify_grid_output(
                    config['path'],
                    expected_vars=vm.get('expected_vars', config.get('columns')),
                    value_range=tuple(vm['value_range']) if vm.get('value_range') is not None else None,
                    range_vars=vm.get('range_vars'),
                    sparse_vars=vm.get('sparse_vars'),
                )
                if not result.ok:
                    errors.append(f"Dataset '{name}' failed output verification: {result.detail}")

            # join_on datasets are small GID-keyed tables merged directly onto
            # assembled rows (not reprojected pixel-grid data) -- see
            # TileProcessor._apply_join_tables.
            join_on = config.get('join_on')
            if join_on is not None:
                if not isinstance(join_on, str) or not join_on.strip():
                    errors.append(f"Dataset '{name}' join_on must be a non-empty string")

            # resampling: a method string, or a {default, <glob>: <method>} map
            # for per-variable control. resolve_resampling validates the method
            # names up front (no variable list needed for that).
            if 'resampling' in config:
                from src.data.assemble.utils import resolve_resampling

                try:
                    resolve_resampling(config['resampling'], [])
                except ValueError as exc:
                    errors.append(f"Dataset '{name}' invalid 'resampling': {exc}")

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
    derived_pixel_ids = processing.get('derived_pixel_ids')

    if derived_pixel_ids is not None:
        if not isinstance(derived_pixel_ids, dict):
            errors.append("'processing.derived_pixel_ids' must be a mapping of column_name -> resolution")
        else:
            for column_name, raw_value in derived_pixel_ids.items():
                if not isinstance(column_name, str) or not column_name.strip():
                    errors.append("Derived pixel ID column names must be non-empty strings")
                if not isinstance(raw_value, (int, float, str)):
                    errors.append(
                        f"Derived pixel ID resolution for {column_name!r} must be numeric or a known grid label"
                    )

    grid_shake = processing.get('grid_shake')
    if grid_shake is not None:
        try:
            normalize_grid_shake_offsets(grid_shake)
        except ValueError as exc:
            errors.append(f"Invalid 'processing.grid_shake': {exc}")

    year_range = processing.get('year_range')
    if year_range:
        if not isinstance(year_range, (list, tuple)) or len(year_range) != 2:
            errors.append("'year_range' must be a list/tuple of two years [start, end]")
        elif year_range[0] > year_range[1]:
            errors.append(f"'year_range' start ({year_range[0]}) must be <= end ({year_range[1]})")
    
    return errors

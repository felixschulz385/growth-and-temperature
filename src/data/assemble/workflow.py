"""
Main workflow orchestration for data assembly.

This module provides the high-level workflow functions that coordinate
dataset loading, tile processing, and output generation. Implementation
details are delegated to specialized modules.

One assembled table = every source in ``assembly.sources`` merged onto the
canonical grid. The CLI picks the output *grid* resolution (``--grid``) and any
grid-shake variants (``--shake``), not which columns are in the panel. Each
grid-shake variant is a full, identical-schema sibling table written under a
``shake=<label>`` partition next to ``shake=base``.
"""

import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

import copy
import logging
from typing import Dict, Any, Optional

from src.data.pipeline.config import build_context
from src.data.sources.layout import EASE_GRID_ID, LEGACY_GRID_ID

# Import assembly submodules
from src.data.assemble.config import (
    derive_data_root,
    apply_cli_overrides,
    resolve_dataset_paths,
    validate_assembly_config,
)
from src.data.assemble.loaders import resolve_land_mask_path
from src.data.assemble.metadata import create_assembly_metadata
from src.data.assemble.grid_shake import resolve_shake_selection
from src.data.assemble.sql_engine import DuckDBConfig, run_sql_assembly
from src.data.assemble.constants import (
    DEFAULT_GRID_LABEL,
    DEFAULT_TILE_SIZE,
    GRID_RESOLUTIONS_M,
)

logger = logging.getLogger(__name__)


def run_assembly(assembly_config: Dict[str, Any], full_config: Optional[Dict[str, Any]] = None):
    """
    Run one assembled-table pass: one grid resolution, one grid-shake variant.

    Steps:
    1. Resolve each source's grid-store path and validate the configuration
       (hard gate -- raises on a missing/broken source, see
       :func:`validate_assembly_config`).
    2. Hand the whole pass to :func:`run_sql_assembly` -- one DuckDB process
       that block-aggregates every ``cell_id``-keyed source onto the requested
       ``--grid`` factor, full-outer-joins them on ``(pixel_id[, year])``,
       merges any ``join_on`` sidecars, and writes tile-partitioned parquet
       under ``output_path``. No Dask, no per-tile Python loop.

    Args:
        assembly_config: One variant's assembly configuration (already carries
            ``output_path``, ``datasets``, and ``processing`` with grid/shake keys)
        full_config: Full pipeline config dict (``paths``/``remote``/``pipeline``)
    """
    logger.info(f"Starting assembly: {assembly_config.get('description', 'Unknown')}")

    data_root = derive_data_root(assembly_config, full_config)
    grid_id = LEGACY_GRID_ID
    ctx = None
    if full_config:
        ctx = build_context(full_config)
        grid_id = ctx.grid_id

    # Resolve each dataset's `path` from `data_path`+`family` before
    # `validate_assembly_config` checks that `path` exists.
    if data_root:
        resolve_dataset_paths(assembly_config, data_root, grid_id)

    errors = validate_assembly_config(assembly_config)
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        raise ValueError(f"Assembly configuration invalid: {len(errors)} error(s) -- see log above.")

    output_path = assembly_config['output_path']
    processing_config = assembly_config.get('processing', {})

    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Output will be written to: {output_path}")

    if not data_root:
        logger.error("data_root must be specified in config or derivable from runtime settings")
        return

    logger.info(f"Using data_root: {data_root}")

    if ctx is None:
        ctx = build_context(full_config or {})

    # The DuckDB block-aggregation engine only makes sense on the metric,
    # exactly-tileable canonical EASE grid.
    if ctx.grid_id != EASE_GRID_ID:
        raise ValueError(
            f"assemble requires pipeline.grid={EASE_GRID_ID!r} (DuckDB block "
            f"aggregation on the canonical grid); configured grid is {ctx.grid_id!r}."
        )

    resolution = processing_config.get('resolution')
    mode = processing_config.get('assembly_mode', 'create')
    datasource = processing_config.get('datasource')
    if mode == 'update' and not datasource:
        logger.error("Update mode requires datasource to be specified")
        return

    # Default False here to match the pre-rewrite `run_assembly` contract for
    # direct callers; `run_workflow_with_config` always sets this explicitly
    # (from `assembly.land_mask`, default True) for the real pipeline path.
    land_mask_path = None
    if processing_config.get('apply_land_mask', False):
        land_mask_path = processing_config.get('land_mask_path') or resolve_land_mask_path(
            data_root, ctx.grid_id
        )
        if not land_mask_path:
            logger.warning("apply_land_mask is set but no land mask was found; proceeding unmasked")

    if not create_assembly_metadata(output_path, assembly_config):
        logger.warning("Failed to create assembly metadata")

    dcfg = processing_config.get('duckdb', {}) or {}
    duckdb_cfg = DuckDBConfig(
        threads=dcfg.get('threads'),
        memory_limit=dcfg.get('memory_limit'),
        temp_dir=dcfg.get('temp_dir'),
    )

    run_sql_assembly(
        datasets=assembly_config['datasets'],
        output_path=output_path,
        resolution_m=resolution,
        shake_offset=tuple(processing_config.get('shake_offset') or (0.0, 0.0)),
        land_mask_path=land_mask_path,
        compression=processing_config.get('compression', 'zstd'),
        tile_size=processing_config.get('tile_size') or DEFAULT_TILE_SIZE,
        year_range=processing_config.get('year_range'),
        derived_pixel_ids=processing_config.get('derived_pixel_ids'),
        mode=mode,
        datasource=datasource,
        duckdb_cfg=duckdb_cfg,
    )
    logger.info("Assembly pass complete: %s", output_path)


def _resolve_grid_resolution(grid_label: str) -> Optional[float]:
    """Grid label -> output resolution in metres, or None for the native grid."""
    if grid_label not in GRID_RESOLUTIONS_M:
        available = ", ".join(GRID_RESOLUTIONS_M)
        raise ValueError(f"Unknown --grid label {grid_label!r}. Available: {available}")
    if grid_label == DEFAULT_GRID_LABEL:
        return None  # native canonical resolution, no downsampling reprojection
    return GRID_RESOLUTIONS_M[grid_label]


def _source_verification_meta(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """The `verify_grid_output` kwargs a source declares for its final-step
    output -- the same ones `data run`/`data summary` use, so the assembly gate
    agrees with the `verified` column (picking up Python-side defaults like
    snl_mining's `sparse_vars` that aren't spelled out in the config's
    `verification:` block). Best-effort: returns ``{}`` when *name* isn't a
    registered source or the source can't be built here.
    """
    if name not in (config.get("sources") or {}):
        return {}
    try:
        from src.data.pipeline.config import build_context, get_source_config
        from src.data.sources import registry
        from src.data.sources.steps import PipelineStep, TargetSelection

        ctx = build_context(config)
        spec = registry.resolve(name)
        source = registry.create(name, ctx, get_source_config(config, name))
        try:
            final_step = (
                PipelineStep.GRID if PipelineStep.GRID in spec.steps else PipelineStep.PREPARE
            )
            targets = source.plan(final_step, TargetSelection())
        finally:
            source.close()
        for target in targets:
            meta = target.meta or {}
            keys = {
                k: meta[k]
                for k in ("expected_vars", "value_range", "range_vars", "sparse_vars")
                if k in meta
            }
            if keys:
                return keys
    except Exception as exc:  # noqa: BLE001 -- verification tuning is best-effort
        logger.debug("Could not read verification meta for source %r: %s", name, exc)
    return {}


def run_workflow_with_config(config: Dict[str, Any]):
    """
    Entry point for the assembly workflow.

    Reads the single ``assembly:`` block (``output_root`` + per-source merge
    settings under ``sources:``), resolves the requested ``--grid``/``--shake``
    from ``cli_overrides``, and runs one :func:`run_assembly` pass per grid-shake
    variant.

    Args:
        config: Full pipeline config dict, plus an optional ``cli_overrides`` key
            carrying ``grid_label``, ``shake``, dask sizing, ``overwrite``,
            ``assembly_mode``, and ``datasource``.

    Raises:
        ValueError: If the ``assembly:`` block is missing or a grid/shake value
            is unknown.
    """
    assembly = config.get('assembly')
    if not assembly or 'sources' not in assembly:
        raise ValueError(
            "Config has no 'assembly:' block with a 'sources:' mapping "
            "(see orchestration/configs/data.yaml)."
        )

    cli_overrides = dict(config.get('cli_overrides', {}))

    grid_label = cli_overrides.get('grid_label') or DEFAULT_GRID_LABEL
    resolution = _resolve_grid_resolution(grid_label)
    shake_selection = resolve_shake_selection(cli_overrides.get('shake'))

    # Output root is data_root-relative by default, so the config carries no
    # machine-specific absolute path or `${DATA_NOBACKUP}` env dependency.
    data_root = derive_data_root(None, config)
    output_root = assembly.get('output_root') or 'assembled'
    if not os.path.isabs(output_root):
        if not data_root:
            raise ValueError(
                "assembly.output_root is relative but data_root is not resolvable "
                "(set paths.data_root in orchestration/configs/data.local.yaml)."
            )
        output_root = os.path.join(data_root, output_root)

    # Pull each source's own verification tuning once, so the per-variant gate
    # agrees with `data summary`'s `verified` column.
    source_verification = {
        name: _source_verification_meta(config, name)
        for name in assembly['sources']
    }

    base_processing = {
        'compression': assembly.get('compression', 'zstd'),
        'apply_land_mask': assembly.get('land_mask', True),
    }
    if assembly.get('year_range') is not None:
        base_processing['year_range'] = assembly['year_range']
    if assembly.get('tile_size') is not None:
        base_processing['tile_size'] = assembly['tile_size']
    if assembly.get('derived_pixel_ids') is not None:
        base_processing['derived_pixel_ids'] = assembly['derived_pixel_ids']
    if assembly.get('land_mask_path') is not None:
        base_processing['land_mask_path'] = assembly['land_mask_path']

    logger.info(
        "Assembly plan: grid=%s (resolution=%s), shake variants=%s",
        grid_label, resolution, [label for label, _, _ in shake_selection],
    )

    for shake_label, dx, dy in shake_selection:
        datasets = copy.deepcopy(assembly['sources'])
        for name, dcfg in datasets.items():
            if source_verification.get(name):
                dcfg['_verification'] = source_verification[name]
        assembly_config = {
            'description': f"assembly grid={grid_label} shake={shake_label}",
            'output_path': os.path.join(
                output_root, f"grid={grid_label}", f"shake={shake_label}"
            ),
            'datasets': datasets,
            'processing': {
                **base_processing,
                'grid_label': grid_label,
                'resolution': resolution,
                'shake_label': shake_label,
                'shake_offset': [dx, dy],
            },
        }
        apply_cli_overrides(assembly_config, cli_overrides)
        run_assembly(assembly_config, config)

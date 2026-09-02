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
from typing import Dict, Any, List, Optional, Tuple

# Import common utilities
from src.data.common.geobox import get_target_geobox
from src.data.common.dask.client import DaskClientContextManager
from src.data.pipeline.config import build_context
from src.data.sources.layout import EASE_GRID_ID, LEGACY_GRID_ID

# Import assembly submodules
from src.data.assemble.config import (
    derive_data_root,
    apply_cli_overrides,
    resolve_dataset_paths,
    validate_assembly_config,
)
from src.data.assemble.loaders import (
    load_land_mask,
    load_all_datasets,
    prepare_land_mask,
)
from src.data.assemble.processors import TileProcessor
from src.data.assemble.metadata import create_assembly_metadata
from src.data.assemble.tiles import (
    get_available_tiles,
    adjust_tile_size_for_reprojection,
    create_tile_geobox,
)
from src.data.assemble.grid_shake import resolve_shake_selection, shift_geobox_origin
from src.data.assemble.constants import (
    DEFAULT_GRID_LABEL,
    DEFAULT_TILE_SIZE,
    GRID_RESOLUTIONS_M,
)

logger = logging.getLogger(__name__)


def _setup_dask_cluster(processing_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create Dask cluster configuration from processing config.

    Args:
        processing_config: Processing configuration with dask settings

    Returns:
        Dictionary of kwargs for DaskClientContextManager
    """
    from src.data.assemble.config import DaskConfig

    dask_dict = processing_config.get('dask', {})
    dask_config = DaskConfig(
        threads=dask_dict.get('threads'),
        memory_limit=dask_dict.get('memory_limit'),
        dashboard_port=dask_dict.get('dashboard_port', 8787),
        temp_dir=dask_dict.get('temp_dir'),
        worker_threads_per_cpu=dask_dict.get('worker_threads_per_cpu', 2),
        worker_fraction=dask_dict.get('worker_fraction', 0.5),
    )
    return dask_config.to_kwargs()


def _process_all_tiles(
    datasets: List[Tuple],
    land_mask_ds,
    all_tiles: List[Tuple[int, int]],
    target_geobox,
    assembly_config: Dict[str, Any],
    output_path: str,
) -> Tuple[int, int]:
    """
    Process all tiles and return counts of processed and skipped tiles.

    Args:
        datasets: Loaded datasets
        land_mask_ds: Optional land mask dataset
        all_tiles: List of (ix, iy) tile indices
        target_geobox: Target geobox (already coarsened + origin-shifted for the variant)
        assembly_config: Assembly configuration
        output_path: Output directory path

    Returns:
        Tuple of (processed_count, skipped_count)
    """
    processing_config = assembly_config.get('processing', {})
    tile_size = processing_config.get('tile_size', DEFAULT_TILE_SIZE)
    assembly_mode = processing_config.get('assembly_mode', 'create')

    processor = TileProcessor(assembly_config, output_path, target_geobox=target_geobox)
    processed_count = 0
    skipped_count = 0
    overwrite = processing_config.get('overwrite', True)  # Default to True for backward compatibility

    for ix, iy in all_tiles:
        tile_output_path = os.path.join(output_path, f"ix={ix}", f"iy={iy}")
        output_file = os.path.join(tile_output_path, "data.parquet")

        if assembly_mode == 'update' and not os.path.exists(output_file):
            logger.warning(f"Tile ix={ix}, iy={iy} does not exist, skipping in update mode")
            skipped_count += 1
            continue

        # In create mode, skip existing tiles if overwrite=False
        if assembly_mode == 'create' and not overwrite and os.path.exists(output_file):
            logger.info(f"Tile ix={ix}, iy={iy} already exists, skipping (overwrite=False)")
            skipped_count += 1
            continue

        tile_geobox = create_tile_geobox(target_geobox, tile_size, ix, iy)

        try:
            success = processor.process_tile(
                datasets, land_mask_ds, ix, iy, tile_geobox
            )
            if success:
                processed_count += 1
        except Exception as e:
            logger.error(f"Failed to process tile ix={ix}, iy={iy}: {e}")
            continue

    return processed_count, skipped_count


def run_assembly(assembly_config: Dict[str, Any], full_config: Optional[Dict[str, Any]] = None):
    """
    Run one assembled-table pass: one grid resolution, one grid-shake variant.

    Main steps:
    1. Resolve each source's grid-store path and validate the configuration
    2. Build the run's target geobox from ``pipeline.grid`` (`get_target_geobox`),
       coarsen it to ``processing.resolution`` and origin-shift it for
       ``processing.shake_offset`` (if any)
    3. Initialise a Dask cluster, load every source, write the provenance metadata
    4. Process all tiles and write tile-partitioned parquet under ``output_path``

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

    # `processing.resolution` (metres) is only set for a coarser-than-native
    # `--grid`. Coarsening is only defined on the metric EASE grid -- check this
    # before building the geobox so a mis-set grid fails fast.
    resolution = processing_config.get('resolution')
    if resolution is not None and ctx.grid_id != EASE_GRID_ID:
        raise ValueError(
            f"assemble --grid coarsening (resolution={resolution} m) requires "
            f"pipeline.grid={EASE_GRID_ID!r}, but the configured grid is {ctx.grid_id!r}."
        )

    # Target geobox: the grid `pipeline.grid` selects (EASE6933 or legacy 4326).
    target_geobox = get_target_geobox(ctx)
    native_res = abs(target_geobox.resolution.x)

    # Grid-shake: shift the whole run's grid origin once, up front. The offset is
    # a fraction of one *output* cell, expressed here in native pixels.
    shake_offset = processing_config.get('shake_offset') or (0.0, 0.0)
    dx, dy = float(shake_offset[0]), float(shake_offset[1])
    if dx or dy:
        factor = (resolution / native_res) if resolution else 1.0
        target_geobox = shift_geobox_origin(target_geobox, dx * factor, dy * factor)
        logger.info(
            "Grid-shake variant %r: origin shifted by (%.3f, %.3f) output-cells",
            processing_config.get('shake_label'), dx, dy,
        )

    processing_config.setdefault('tile_size', DEFAULT_TILE_SIZE)
    processing_config['tile_size'] = adjust_tile_size_for_reprojection(
        native_res, resolution, processing_config['tile_size']
    )

    # Discover tiles
    logger.info("Discovering available tiles...")
    all_tiles = get_available_tiles(assembly_config, target_geobox)
    logger.info(f"Found {len(all_tiles)} tiles to process")

    if not all_tiles:
        logger.warning("No tiles found to process")
        return

    # Set up Dask cluster
    dask_kwargs = _setup_dask_cluster(processing_config)
    logger.info("Creating Dask cluster for data loading and processing...")

    with DaskClientContextManager(**dask_kwargs) as client:
        logger.info(f"Dask client initialized: {client.dashboard_link}")

        # Load land mask if requested
        land_mask_ds = None
        if processing_config.get('apply_land_mask', False):
            land_mask_path = processing_config.get('land_mask_path')
            land_mask_ds = load_land_mask(
                data_root, target_geobox, processing_config['tile_size'], land_mask_path
            )
            if land_mask_ds is not None:
                land_mask_ds = prepare_land_mask(land_mask_ds)

        # Step 1: Load datasets
        logger.info("Step 1: Loading datasets with alignment checks...")
        try:
            assembly_mode = processing_config.get('assembly_mode', 'create')
            target_datasource = processing_config.get('datasource')

            if assembly_mode == 'update':
                if not target_datasource:
                    logger.error("Update mode requires datasource to be specified")
                    return
                logger.info(f"UPDATE mode: Loading only datasource '{target_datasource}'")
                datasets = load_all_datasets(assembly_config, target_geobox, datasource_filter=target_datasource)
            else:
                logger.info("CREATE mode: Loading all datasets")
                datasets = load_all_datasets(assembly_config, target_geobox)
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            return

        # Step 2: Create metadata
        logger.info("Step 2: Creating assembly metadata...")
        if not create_assembly_metadata(output_path, assembly_config):
            logger.warning("Failed to create assembly metadata")

        # Step 3: Process pixel tiles
        logger.info("Step 3: Processing tiles (source-by-source)...")
        processed_count, skipped_count = _process_all_tiles(
            datasets, land_mask_ds, all_tiles, target_geobox,
            assembly_config, output_path
        )
        logger.info(
            f"Dask processing completed. Processed {processed_count}/{len(all_tiles)} tiles, "
            f"skipped {skipped_count} tiles"
        )


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

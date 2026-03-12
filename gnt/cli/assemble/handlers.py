"""
Handler functions for the ``assemble`` domain.
"""

from __future__ import annotations

import argparse
import importlib
import logging

from gnt.cli.config import load_config_with_env_vars
from gnt.cli.common import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _collect_dask_overrides(args: argparse.Namespace) -> dict:
    """Extract Dask-related CLI overrides from *args* into a dict."""
    overrides = {}
    if getattr(args, "dask_threads", None) is not None:
        overrides["dask_threads"] = args.dask_threads
    if getattr(args, "dask_memory_limit", None) is not None:
        overrides["dask_memory_limit"] = args.dask_memory_limit
        logger.info(f"Overriding dask_memory_limit from CLI: {args.dask_memory_limit}")
    if getattr(args, "temp_dir", None) is not None:
        overrides["temp_dir"] = args.temp_dir
    if getattr(args, "dashboard_port", 8787) != 8787:
        overrides["dashboard_port"] = args.dashboard_port
    if getattr(args, "local_directory", None) is not None:
        overrides["local_directory"] = args.local_directory
        logger.info(f"Overriding local_directory from CLI: {args.local_directory}")
    if getattr(args, "tile_size", None) is not None:
        overrides["tile_size"] = args.tile_size
        logger.info(f"Overriding tile_size from CLI: {args.tile_size}")
    if getattr(args, "compression", None) is not None:
        overrides["compression"] = args.compression
        logger.info(f"Overriding compression from CLI: {args.compression}")
    return overrides


def _run_assembly_workflow(config: dict, source: str, cli_overrides: dict) -> None:
    config = config.copy()
    config["assembly_name"] = source
    config["cli_overrides"] = cli_overrides
    mod = importlib.import_module("gnt.data.assemble.workflow")
    logger.info(f"Running assembly workflow: {source}")
    mod.run_workflow_with_config(config)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def handle_create(args: argparse.Namespace) -> None:
    """``assemble create`` — recreate all tiles."""
    setup_logging(args.log_level, debug=args.debug)
    config = load_config_with_env_vars(args.config)

    cli_overrides = _collect_dask_overrides(args)
    cli_overrides["assembly_mode"] = "create"
    logger.info("Assembly mode: CREATE (recreate all tiles)")

    overwrite = getattr(args, "overwrite", None)
    if overwrite is not None:
        cli_overrides["overwrite"] = overwrite
        logger.info(f"Overriding overwrite from CLI: {overwrite}")

    _run_assembly_workflow(config, args.source, cli_overrides)


def handle_update(args: argparse.Namespace) -> None:
    """``assemble update`` — update existing tiles with a new datasource."""
    setup_logging(args.log_level, debug=args.debug)
    config = load_config_with_env_vars(args.config)

    cli_overrides = _collect_dask_overrides(args)
    cli_overrides["assembly_mode"] = "update"
    cli_overrides["datasource"] = args.datasource
    logger.info(f"Assembly mode: UPDATE datasource '{args.datasource}'")

    _run_assembly_workflow(config, args.source, cli_overrides)


def handle_demean(args: argparse.Namespace) -> None:
    """``assemble demean`` — run demeaning on an assembled dataset."""
    setup_logging(args.log_level, debug=args.debug)
    config = load_config_with_env_vars(args.config)

    override_level = getattr(args, "override_level", 0) or 0
    mod = importlib.import_module("gnt.data.assemble.demean")
    logger.info(
        f"Running demeaning workflow for assembly: {args.source} "
        f"(override_level={override_level})"
    )
    mod.run_workflow_with_config(
        config, assembly_name=args.source, override_level=override_level
    )

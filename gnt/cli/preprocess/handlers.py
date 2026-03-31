"""
Handler functions for the ``preprocess`` domain.
"""

from __future__ import annotations

import argparse
import importlib
import logging

from gnt.cli.config import load_config_with_env_vars
from gnt.cli.common import setup_logging

logger = logging.getLogger(__name__)


def handle_run(args: argparse.Namespace) -> None:
    """``preprocess run`` — preprocess raw source files."""
    setup_logging(args.log_level, debug=args.debug)
    config = load_config_with_env_vars(args.config)

    source = args.source
    if "sources" not in config or source not in config["sources"]:
        raise ValueError(f"Source '{source}' not found in configuration")

    # Apply CLI overrides to config sections before building workflow config
    preprocess_config = config.setdefault("preprocess", {})
    source_config = config.setdefault("sources", {}).setdefault(source, {})

    if getattr(args, "subsource", None):
        preprocess_config["subsource"] = args.subsource
        logger.info(f"Setting subsource from CLI: {args.subsource}")
    if getattr(args, "dask_threads", None) is not None:
        preprocess_config["dask_threads"] = args.dask_threads
        logger.info(f"Overriding dask_threads from CLI: {args.dask_threads}")
    if getattr(args, "dask_memory_limit", None) is not None:
        preprocess_config["dask_memory_limit"] = args.dask_memory_limit
        logger.info(f"Overriding dask_memory_limit from CLI: {args.dask_memory_limit}")
    if getattr(args, "temp_dir", None) is not None:
        preprocess_config["temp_dir"] = args.temp_dir
        logger.info(f"Overriding temp_dir from CLI: {args.temp_dir}")
    if getattr(args, "dashboard_port", 8787) != 8787:
        preprocess_config["dashboard_port"] = args.dashboard_port
        logger.info(f"Overriding dashboard_port from CLI: {args.dashboard_port}")
    if getattr(args, "local_directory", None) is not None:
        preprocess_config["local_directory"] = args.local_directory
        logger.info(f"Overriding local_directory from CLI: {args.local_directory}")
    if getattr(args, "year", None) is not None:
        source_config["year"] = args.year
        logger.info(f"Overriding year from CLI: {args.year}")
    if getattr(args, "year_range", None) is not None:
        source_config["year_range"] = args.year_range
        logger.info(f"Overriding year_range from CLI: {args.year_range}")
    if getattr(args, "grid_cells", None) is not None:
        source_config["grid_cells"] = args.grid_cells
        logger.info(f"Overriding grid_cells from CLI: {args.grid_cells}")
    if getattr(args, "override", False):
        source_config["override"] = True
        logger.info("Override mode enabled from CLI")
    if getattr(args, "stage", None):
        preprocess_config["stage"] = args.stage
        logger.info(f"Setting stage from CLI: {args.stage}")
    if getattr(args, "admin_level", None) is not None:
        source_config["admin_level"] = args.admin_level
        logger.info(f"Overriding admin_level from CLI: {args.admin_level}")

    task_config = preprocess_config.copy()
    task_config["mode"] = getattr(args, "mode", None) or "preprocess"
    stage = getattr(args, "stage", None)
    if stage:
        task_config["stage"] = stage

    wf_config = {
        "source": config["sources"][source],
        "preprocess": preprocess_config,
        "workflow": {
            "tasks": [
                {"type": "preprocess", "config": task_config}
            ]
        },
        "hpc": config.get("hpc", {}),
        "gcs": config.get("gcs", {}),
        "sources": config.get("sources", {}),
        "source_name": source,
    }

    mod = importlib.import_module("gnt.data.preprocess.workflow")
    logger.info("Running unified preprocessing workflow")
    mod.run_workflow_with_config(wf_config)

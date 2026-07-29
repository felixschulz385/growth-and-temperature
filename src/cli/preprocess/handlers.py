"""
Handler functions for the ``preprocess`` domain.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os

from src.cli.config import load_config_with_env_vars
from src.cli.common import setup_logging
from src.config.runtime import (
    get_legacy_hpc_compat_config,
    get_paths_config,
    get_remote_config,
)

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
        "paths": get_paths_config(config),
        "remote": get_remote_config(config),
        "hpc": get_legacy_hpc_compat_config(config),
        "gcs": config.get("gcs", {}),
        "sources": config.get("sources", {}),
        "source_name": source,
    }

    mod = importlib.import_module("src.data.preprocess.workflow")
    logger.info("Running unified preprocessing workflow")
    success = mod.run_workflow_with_config(wf_config)
    if not success:
        raise RuntimeError(f"Preprocessing workflow failed for source '{source}'")


def handle_transfer(args: argparse.Namespace) -> None:
    """``preprocess transfer`` — push a stage's local output to the HPC target.

    docs/design/08-hpc-transfer.md — generic across sources via
    ``AbstractPreprocessor.get_transfer_units()``.
    """
    setup_logging(args.log_level, debug=args.debug)
    config = load_config_with_env_vars(args.config)

    source = args.source
    if "sources" not in config or source not in config["sources"]:
        raise ValueError(f"Source '{source}' not found in configuration")

    if args.direction == "pull":
        raise NotImplementedError(
            "--direction pull is not implemented; included in the CLI for interface "
            "symmetry with HPCIndexSynchronizer's push/pull shape, not because any "
            "current source needs it (docs/design/08-hpc-transfer.md §4)."
        )

    paths_config = get_paths_config(config)
    remote_config = get_remote_config(config)

    preprocessor_config = dict(config["sources"][source])
    preprocessor_config["source"] = source
    preprocessor_config["name"] = source
    preprocessor_config["preprocessor"] = source
    preprocessor_config["stage"] = args.stage
    if paths_config.get("data_root"):
        preprocessor_config["hpc_target"] = paths_config["data_root"]
    if paths_config.get("local_index_dir"):
        preprocessor_config.setdefault("local_index_dir", paths_config["local_index_dir"])
    # `get_transfer_units` doesn't need a specific year -- it lists what's
    # already on local disk for `stage` -- but preprocessor __init__ commonly
    # requires one of year/year_range to be set; supply a wide placeholder,
    # matching the existing validate-task convention in workflow.py.
    if "year" not in preprocessor_config and "year_range" not in preprocessor_config:
        preprocessor_config["year_range"] = [1900, 2100]

    from src.data.preprocess.sources.factory import create_preprocessor
    from src.data.common.hpc.transfer import transfer_units

    preprocessor = create_preprocessor(source, preprocessor_config)
    units = preprocessor.get_transfer_units(args.stage)
    if not units:
        logger.warning("No transfer units found for source '%s' stage '%s'", source, args.stage)
        return

    ssh_target = remote_config.get("ssh_target")
    if not ssh_target:
        raise ValueError("remote.ssh_target is not configured")
    key_file = remote_config.get("key_file")

    manifest_path = None
    local_index_dir = paths_config.get("local_index_dir")
    if local_index_dir:
        manifest_path = os.path.join(local_index_dir, f"transfer_{source}_{args.stage}.parquet")

    logger.info("Transferring %d unit(s) for source '%s' stage '%s'", len(units), source, args.stage)
    success = transfer_units(
        ssh_target=ssh_target,
        key_file=key_file,
        units=units,
        manifest_path=manifest_path,
        override=args.override,
    )
    if not success:
        raise RuntimeError(f"Transfer failed for source '{source}' stage '{args.stage}'")

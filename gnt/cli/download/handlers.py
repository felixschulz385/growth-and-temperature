"""
Handler functions for the ``download`` domain.

Each handler corresponds to one sub-command and is wired via
``parser.set_defaults(func=handle_<name>)``.
Handlers are thin: they load config, build the workflow config dict, then
delegate to :mod:`gnt.data.download.workflow_unified`.
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

def _load_and_validate(args: argparse.Namespace):
    """Load config and return ``(config, source_config)`` for download ops."""
    setup_logging(args.log_level, debug=args.debug)
    config = load_config_with_env_vars(args.config)
    source = args.source

    if "sources" not in config or source not in config["sources"]:
        raise ValueError(f"Source '{source}' not found in configuration")

    return config, config["sources"][source].copy()


def _build_hpc_workflow_config(
    config: dict,
    source: str,
    source_config: dict,
    operation: str,
) -> dict:
    """Build the workflow configuration dict consumed by workflow_unified."""
    return {
        "source": source_config,
        "index": {
            "local_dir": config.get("hpc", {}).get("local_index_dir"),
            "rebuild": operation == "index",
            "only_missing_entrypoints": True,
            "sync_direction": "auto",
        },
        "workflow": {"tasks": []},
        "hpc": config.get("hpc", {}),
        "source_name": source,
    }


def _run_workflow(hpc_workflow_config: dict) -> None:
    """Dispatch to workflow_unified.run_workflow_with_config."""
    mod = importlib.import_module("gnt.data.download.workflow_unified")
    logger.info("Running unified download workflow")
    mod.run_workflow_with_config(hpc_workflow_config)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def handle_index(args: argparse.Namespace) -> None:
    """``download index`` — build / sync the remote file index."""
    config, source_config = _load_and_validate(args)
    wf = _build_hpc_workflow_config(config, args.source, source_config, "index")
    wf["workflow"]["tasks"] = [
        {
            "type": "index",
            "config": {
                "rebuild": False,
                "only_missing_entrypoints": True,
                "sync_direction": "auto",
            },
        }
    ]
    _run_workflow(wf)


def handle_run(args: argparse.Namespace) -> None:
    """``download run`` — download pending files."""
    config, source_config = _load_and_validate(args)

    # Optional: filter misc files
    misc_files = getattr(args, "misc_files", None)
    if args.source == "misc" and misc_files:
        logger.info(f"Filtering misc files to: {misc_files}")
        all_sources = source_config.get("sources", {})
        filtered = {k: v for k, v in all_sources.items() if k in misc_files}
        if not filtered:
            logger.warning(
                f"No matching misc files for filter: {misc_files}. "
                f"Available: {list(all_sources.keys())}"
            )
            return
        source_config["sources"] = filtered
        logger.info(
            f"Will download {len(filtered)} misc files: {list(filtered.keys())}"
        )

    wf = _build_hpc_workflow_config(config, args.source, source_config, "download")

    # Try to report pending count before starting
    try:
        from gnt.data.common.index.unified_index import UnifiedDataIndex
        from gnt.data.download.sources.factory import create_data_source

        ds = create_data_source(wf["source"])
        idx = UnifiedDataIndex(
            bucket_name="",
            data_source=ds,
            local_index_dir=wf["index"]["local_dir"],
            key_file=wf["hpc"].get("key_file"),
            hpc_mode=True,
        )
        pending = idx.count_pending_files()
        logger.info(f"Found {pending} files pending download for {args.source}")
    except Exception as exc:
        logger.debug(f"Could not get pending file count: {exc}")

    dl_cfg = source_config.get("download", {})
    wf["workflow"]["tasks"] = [
        {
            "type": "download",
            "config": {
                "batch_size": dl_cfg.get("batch_size", 50),
                "max_concurrent_downloads": dl_cfg.get("max_concurrent_downloads", 5),
                "tar_max_files": dl_cfg.get("tar_max_files", 100),
                "tar_max_size_mb": dl_cfg.get("tar_max_size_mb", 500),
            },
        }
    ]
    _run_workflow(wf)


# ---------------------------------------------------------------------------
# SNF Mining scraper
# ---------------------------------------------------------------------------

def handle_snf_mining(args: argparse.Namespace) -> None:
    """``download snf-mining`` — run the S&P Global SNF Mining Selenium scraper.

    Invokes :func:`~gnt.data.download.sources.snf_mining.run_full_workflow`
    directly because the scraper is Selenium-based, not HTTP-batch.
    """
    from gnt.cli.common import setup_logging
    from gnt.data.download.sources.snf_mining import Stage, run_full_workflow
    from gnt.data.download.sources.snf_mining.config import DEFAULT_DB_PATH

    setup_logging(args.log_level, debug=args.debug)

    stages = {Stage(stage) for stage in args.stages} if args.stages else None
    force_stages = {Stage(stage) for stage in args.force_stages} if args.force_stages else None
    db_path = args.db or DEFAULT_DB_PATH
    mine_ids = [str(mine_id) for mine_id in args.mine_ids] if args.mine_ids else None

    logger.info(
        "Starting SNF Mining scraper | stages=%s | headless=%s | db=%s",
        [stage.value for stage in stages] if stages else "all",
        args.headless,
        db_path,
    )

    results = run_full_workflow(
        credentials_path=args.credentials,
        db_path=db_path,
        stages=stages,
        headless=args.headless,
        wait=args.wait,
        download_wait=args.download_wait,
        mine_ids=mine_ids,
        subsections=args.subsections,
        max_attempts=args.max_attempts,
        sidebar_reload_attempts=args.sidebar_reload_attempts,
        continue_on_error=not args.fail_fast,
        force_stages=force_stages,
        step_sleep_seconds=args.step_sleep_seconds,
    )
    logger.info("SNF Mining scraper finished: %s", results)

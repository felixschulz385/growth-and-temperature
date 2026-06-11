"""
Handler functions for the ``analysis`` domain.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from gnt.cli.common import setup_logging
from gnt.analysis.runtime_settings import resolve_slurm_partition, scale_memory_limit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Return the project root directory (two levels above gnt/cli)."""
    return Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def handle_run(args: argparse.Namespace) -> None:
    """``analysis run`` — execute a single model via DuckReg."""
    setup_logging(args.log_level, debug=args.debug)
    logger.debug("Starting GNT analysis system: DuckReg")

    from gnt.analysis import AnalysisConfig, run_duckreg

    cfg = AnalysisConfig(args.config)
    output_dir = getattr(args, "output", None) or str(cfg.base_path)

    # --list-models
    if getattr(args, "list_models", False):
        model_names = cfg.get_model_names()
        if not model_names:
            logger.info("No models found in configuration")
            return
        print("\nAvailable models:")
        print("=" * 80)
        for name in model_names:
            spec = cfg.get_model_spec(name)
            print(f"\n{name}")
            print(f"  Description: {spec.get('description', 'N/A')}")
            print(f"  Data source: {spec.get('data_source', 'N/A')}")
            print(f"  Formula    : {spec.get('formula', 'N/A')}")
        print("\n" + "=" * 80)
        return

    model = getattr(args, "model", None)
    if not model:
        raise ValueError("--model is required unless --list-models is specified")

    model_names = cfg.get_model_names()
    if model not in model_names:
        logger.error(f"Unknown model: {model}")
        logger.info(f"Available models: {model_names}")
        raise SystemExit(1)

    logger.debug(f"Running model: {model}")
    run_duckreg(
        cfg,
        model,
        output_dir,
        dataset_override=getattr(args, "dataset", None),
        fixed_effects=getattr(args, "fe", None),
        resolution=getattr(args, "resolution", None),
        clustering=getattr(args, "clustering", None),
        temporal_extent=getattr(args, "temporal_extent", None),
        spatial_extent=getattr(args, "spatial_extent", None),
        se_method=getattr(args, "se_method"),
        fitter=getattr(args, "fitter"),
        fe_method=getattr(args, "fe_method"),
        round_strata=getattr(args, "round_strata"),
        seed=getattr(args, "seed"),
        n_bootstraps=getattr(args, "n_bootstraps"),
        threads=getattr(args, "threads"),
        memory_limit=getattr(args, "memory_limit"),
        max_temp_directory_size=getattr(args, "max_temp_directory_size"),
    )


def handle_submit(args: argparse.Namespace) -> None:
    """``analysis submit`` — submit a SLURM batch job."""
    setup_logging(args.log_level, debug=args.debug)

    from gnt.analysis import AnalysisConfig
    from gnt.analysis.config import seconds_to_slurm_time, PROJECT_ROOT
    from gnt.analysis.slurm import (
        filter_unrun_model_pairs,
        ONE_WEEK_SECONDS,
        make_job_label,
        resolve_explicit_pairs,
        submit_job,
        write_job_script,
    )

    try:
        from duckreg._version import __version__ as _duckreg_ver
    except ImportError:
        _duckreg_ver = "unknown"

    tables = list(getattr(args, "tables", None) or [])
    individual_models = list(getattr(args, "models", None) or [])

    if not tables and not individual_models:
        raise ValueError(
            "analysis submit requires at least one of "
            "--tables <table> [...] or --models <model> [...]"
        )

    cfg = AnalysisConfig(getattr(args, "analysis_config", None) or None)
    pairs, total_secs = resolve_explicit_pairs(
        tables,
        individual_models,
        cfg,
        fixed_effects=getattr(args, "fe", None),
        resolution=getattr(args, "resolution", None),
        clustering=getattr(args, "clustering", None),
        temporal_extent=getattr(args, "temporal_extent", None),
        spatial_extent=getattr(args, "spatial_extent", None),
    )
    pairs, skipped_models = filter_unrun_model_pairs(
        pairs,
        cfg.base_path,
        rerun_existing=getattr(args, "rerun_existing", False),
    )
    identifiers = tables + individual_models

    if skipped_models:
        print(
            f"Skipping {len(skipped_models)} model(s) with existing results. "
            "Pass --rerun-existing to include them."
        )

    if not pairs:
        print("No models left to submit.")
        return

    total_secs = sum(
        cfg.get_model_runtime_seconds_for_spec(model_spec)
        for _, models in pairs
        for model_spec in models
    )

    print(f"Total runtime across all identifiers: {seconds_to_slurm_time(total_secs)}")

    if total_secs > ONE_WEEK_SECONDS:
        logger.error(
            f"Combined runtime {seconds_to_slurm_time(total_secs)} exceeds the "
            "1-week QOS limit. Split the tables across multiple jobs."
        )
        raise SystemExit(1)

    slurm_time = getattr(args, "time", None) or seconds_to_slurm_time(total_secs)
    job_label = make_job_label(identifiers)
    duckdb_memory_limit = scale_memory_limit(args.mem, 0.8)
    partition = resolve_slurm_partition(args.mem, getattr(args, "partition", None))
    slurm_kwargs = {
        "mem":           args.mem,
        "time":          slurm_time,
        "qos":           args.qos,
        "partition":     partition,
        "cpus_per_task": args.cpus_per_task,
    }

    print(f"Creating job script... (duckreg {_duckreg_ver})")
    job_path = write_job_script(
        pairs,
        PROJECT_ROOT,
        job_label,
        _duckreg_ver,
        debug=args.debug,
        runtime_settings={
            "se_method": args.se_method,
            "fitter": args.fitter,
            "fe_method": args.fe_method,
            "round_strata": args.round_strata,
            "seed": args.seed,
            "n_bootstraps": args.n_bootstraps,
            "threads": args.threads,
            "memory_limit": duckdb_memory_limit,
            "max_temp_directory_size": args.max_temp_directory_size,
        },
    )

    print("Submitting job to SLURM...")
    job_id = submit_job(job_path, slurm_kwargs)

    total_models = sum(len(m) for _, m in pairs)
    print(f"\nJob submitted successfully!")
    print(f"  Job ID  : {job_id}")
    print(f"  Tables  : {', '.join(tables) if tables else '—'}")
    print(f"  Models  : {', '.join(individual_models) if individual_models else '—'}")
    print(f"  Total   : {total_models} model(s)")
    print(f"  duckreg : {_duckreg_ver}")
    print(f"  Memory  : {args.mem}")
    print(f"  Partition: {partition}")
    print(f"  DuckDB  : {duckdb_memory_limit}")
    print(f"  Time    : {slurm_time}")
    print(f"  QOS     : {args.qos}")

    os.remove(job_path)


def handle_summary(args: argparse.Namespace) -> None:
    """``analysis summary`` — print table status overview."""
    setup_logging(args.log_level, debug=args.debug)

    from gnt.analysis import AnalysisConfig, summarize_tables

    cfg = AnalysisConfig(getattr(args, "analysis_config", None) or None)
    summarize_tables(cfg)


def handle_tables(args: argparse.Namespace) -> None:
    """``analysis tables`` — render HTML / LaTeX table files."""
    setup_logging(args.log_level, debug=args.debug)
    logger.info("Starting GNT table generation system")

    from gnt.analysis import AnalysisConfig, generate_all_tables

    excel_path = getattr(args, "analysis_config", None) or None
    output_dir = getattr(args, "output_dir", None) or None
    output_formats = getattr(args, "formats", None) or None
    source = getattr(args, "source", None)
    table_names = [source] if source else None

    cfg = AnalysisConfig(excel_path)
    logger.info(f"Generating tables: {table_names or 'all'}")
    generate_all_tables(
        cfg,
        table_names=table_names,
        output_dir=output_dir,
        output_formats=output_formats,
    )
    logger.info("Table generation completed successfully")


def handle_cleanup(args: argparse.Namespace) -> None:
    """``analysis cleanup`` — remove stale analysis result files."""
    setup_logging(args.log_level, debug=args.debug)
    logger.info("Starting analysis results cleanup")

    from gnt.analysis import cleanup_analysis_results

    project_root = _project_root()
    output_dir = getattr(args, "output", None) or str(
        project_root / "output" / "analysis"
    )

    if not Path(output_dir).exists():
        logger.error(f"Output directory not found: {output_dir}")
        raise SystemExit(1)

    logger.info(f"Cleaning up results in: {output_dir}")
    dry_run = getattr(args, "dry_run", False)
    if dry_run:
        logger.info("DRY RUN MODE — no files will be deleted")

    cleanup_analysis_results(output_dir, dry_run=dry_run)

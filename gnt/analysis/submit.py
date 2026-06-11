#!/usr/bin/env python3
"""Submit analysis models to SLURM using the shared analysis helpers."""

from __future__ import annotations

import argparse
import os
import sys

from gnt.analysis.runtime_settings import (
    ANALYSIS_RUNTIME_DEFAULTS,
    resolve_slurm_partition,
    scale_memory_limit,
)

try:
    from duckreg._version import __version__ as _DUCKREG_VERSION
except ImportError:
    _DUCKREG_VERSION = "unknown"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the legacy submission entrypoint."""
    parser = argparse.ArgumentParser(
        description="Submit one SLURM job for selected tables and/or models."
    )
    parser.add_argument(
        "--tables",
        nargs="*",
        metavar="TABLE",
        default=[],
        help="Table names to submit",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        metavar="MODEL",
        default=[],
        help="Individual model names to submit",
    )
    parser.add_argument(
        "--analysis-config",
        help="Path to analysis.xlsx (overrides default)",
    )
    parser.add_argument(
        "--fe",
        choices=["NO", "PX", "PX+CY", "PX+YR", "ADM2", "ADM2+CY", "ADM2+YR"],
        help="Override fixed effects for individually submitted models",
    )
    parser.add_argument(
        "--resolution",
        choices=["500m", "1km", "5km", "50km", "ADM2"],
        help="Override resolution for individually submitted models",
    )
    parser.add_argument(
        "--clustering",
        choices=["ADM2", "Country"],
        help="Override clustering for individually submitted models",
    )
    parser.add_argument(
        "--temporal-extent",
        help="Override temporal extent for individually submitted models (YYYY-YYYY)",
    )
    parser.add_argument(
        "--spatial-extent",
        help="Override spatial extent for individually submitted models",
    )
    parser.add_argument(
        "--mem",
        default="128GB",
        help="SLURM memory request (default: 128GB)",
    )
    parser.add_argument("--time", default=None, help="SLURM time limit override")
    parser.add_argument("--qos", default="1week", help="SLURM QOS (default: 1week)")
    parser.add_argument(
        "--partition",
        default=None,
        help="SLURM partition override (default: auto-select scicore, or bigmem for mem >=256GB)",
    )
    parser.add_argument(
        "--cpus-per-task",
        default=8,
        type=int,
        help="SLURM CPUs per task (default: 8)",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Include models that already have results (default: skip them)",
    )
    parser.add_argument(
        "--se-method",
        default=ANALYSIS_RUNTIME_DEFAULTS["se_method"],
        help=f"DuckReg SE method (default: {ANALYSIS_RUNTIME_DEFAULTS['se_method']})",
    )
    parser.add_argument(
        "--fitter",
        default=ANALYSIS_RUNTIME_DEFAULTS["fitter"],
        help=f"DuckReg fitter (default: {ANALYSIS_RUNTIME_DEFAULTS['fitter']})",
    )
    parser.add_argument(
        "--fe-method",
        default=ANALYSIS_RUNTIME_DEFAULTS["fe_method"],
        help=f"Fixed-effects estimation method (default: {ANALYSIS_RUNTIME_DEFAULTS['fe_method']})",
    )
    parser.add_argument(
        "--round-strata",
        type=int,
        default=ANALYSIS_RUNTIME_DEFAULTS["round_strata"],
        help=f"Round strata setting (default: {ANALYSIS_RUNTIME_DEFAULTS['round_strata']})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=ANALYSIS_RUNTIME_DEFAULTS["seed"],
        help=f"Random seed (default: {ANALYSIS_RUNTIME_DEFAULTS['seed']})",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=ANALYSIS_RUNTIME_DEFAULTS["n_bootstraps"],
        help=f"Number of bootstraps (default: {ANALYSIS_RUNTIME_DEFAULTS['n_bootstraps']})",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=ANALYSIS_RUNTIME_DEFAULTS["threads"],
        help=f"DuckDB threads (default: {ANALYSIS_RUNTIME_DEFAULTS['threads']})",
    )
    parser.add_argument(
        "--max-temp-directory-size",
        default=ANALYSIS_RUNTIME_DEFAULTS["max_temp_directory_size"],
        help=(
            "DuckDB max temp directory size "
            f"(default: {ANALYSIS_RUNTIME_DEFAULTS['max_temp_directory_size']})"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Pass debug mode to submitted analysis runs",
    )
    return parser


def main() -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        from gnt.analysis import AnalysisConfig
        from gnt.analysis.config import PROJECT_ROOT, seconds_to_slurm_time
        from gnt.analysis.slurm import (
            ONE_WEEK_SECONDS,
            filter_unrun_model_pairs,
            make_job_label,
            resolve_explicit_pairs,
            submit_job,
            write_job_script,
        )

        tables = list(args.tables or [])
        models = list(args.models or [])
        if not tables and not models:
            raise ValueError("At least one --tables or --models argument is required")

        cfg = AnalysisConfig(args.analysis_config or None)
        pairs, _ = resolve_explicit_pairs(
            tables,
            models,
            cfg,
            fixed_effects=args.fe,
            resolution=args.resolution,
            clustering=args.clustering,
            temporal_extent=args.temporal_extent,
            spatial_extent=args.spatial_extent,
        )
        pairs, skipped_models = filter_unrun_model_pairs(
            pairs,
            cfg.base_path,
            rerun_existing=args.rerun_existing,
        )

        if skipped_models:
            print(
                f"Skipping {len(skipped_models)} model(s) with existing results. "
                "Pass --rerun-existing to include them."
            )

        if not pairs:
            print("No models left to submit.")
            return 0

        total_seconds = sum(
            cfg.get_model_runtime_seconds_for_spec(model_spec)
            for _, model_specs in pairs
            for model_spec in model_specs
        )
        slurm_time = args.time or seconds_to_slurm_time(total_seconds)

        print(f"\nTotal combined runtime: {seconds_to_slurm_time(total_seconds)}")

        if total_seconds > ONE_WEEK_SECONDS:
            raise ValueError(
                f"Total runtime {seconds_to_slurm_time(total_seconds)} exceeds the "
                f"1-week limit ({seconds_to_slurm_time(ONE_WEEK_SECONDS)}). "
                "Split the tables across multiple jobs."
            )

        identifiers = tables + models
        job_label = make_job_label(identifiers)
        duckdb_memory_limit = scale_memory_limit(args.mem, 0.8)
        partition = resolve_slurm_partition(args.mem, args.partition)

        print(f"\nCreating job script... (duckreg {_DUCKREG_VERSION})")
        job_script_path = write_job_script(
            pairs,
            PROJECT_ROOT,
            job_label,
            _DUCKREG_VERSION,
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
            debug=args.debug,
        )

        print("Submitting job to SLURM...")
        slurm_kwargs = {
            "mem": args.mem,
            "time": slurm_time,
            "qos": args.qos,
            "partition": partition,
            "cpus_per_task": args.cpus_per_task,
        }
        job_id = submit_job(job_script_path, slurm_kwargs)

        total_models = sum(len(model_specs) for _, model_specs in pairs)
        print("\nJob submitted successfully!")
        print(f"  Job ID:      {job_id}")
        print(f"  Tables:      {', '.join(tables) if tables else '—'}")
        print(f"  Models:      {', '.join(models) if models else '—'}")
        print(f"  Total mdls:  {total_models}")
        print(f"  duckreg:     {_DUCKREG_VERSION}")
        print(f"  Memory:      {args.mem}")
        print(f"  Partition:   {partition}")
        print(f"  DuckDB:      {duckdb_memory_limit}")
        print(f"  Time:        {slurm_time}")
        print(f"  QOS:         {args.qos}")

        os.remove(job_script_path)
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

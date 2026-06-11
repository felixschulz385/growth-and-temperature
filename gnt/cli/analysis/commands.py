"""
Argument registration for the ``analysis`` domain.

Sub-commands
------------
run      Run a single model via DuckReg (no SLURM).
submit   Submit one or more table/model sets as a SLURM batch job.
summary  Print a status overview of all tables and their last results.
tables   Render HTML / LaTeX table files.
cleanup  Remove stale result files, keeping the latest per version group.
"""

from __future__ import annotations

import argparse

from gnt.cli.common import add_logging_args
from gnt.analysis.core.config import FIXED_EFFECT_TERMS
from gnt.analysis.core.runtime import ANALYSIS_RUNTIME_DEFAULTS


CLI_FIXED_EFFECT_CHOICES = list(FIXED_EFFECT_TERMS.keys())


def _runtime_default_help(key: str) -> str:
    return (
        f"Default: from analysis.xlsx Settings sheet, "
        f"falling back to {ANALYSIS_RUNTIME_DEFAULTS[key]!r}."
    )


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value, got {value!r}."
    )


def _add_runtime_setting_args(
    parser: argparse.ArgumentParser,
    *,
    include_memory_limit: bool = True,
) -> None:
    """Register analysis runtime settings with CLI-owned defaults."""
    parser.add_argument(
        "--se-method",
        default=None,
        help=f"DuckReg SE method. {_runtime_default_help('se_method')}",
    )
    parser.add_argument(
        "--fitter",
        default=None,
        help=f"DuckReg fitter. {_runtime_default_help('fitter')}",
    )
    parser.add_argument(
        "--fe-method",
        default=None,
        help=f"Fixed-effects estimation method. {_runtime_default_help('fe_method')}",
    )
    parser.add_argument(
        "--round-strata",
        type=int,
        default=None,
        help=f"Round strata setting. {_runtime_default_help('round_strata')}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Random seed. {_runtime_default_help('seed')}",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=None,
        help=f"Number of bootstraps. {_runtime_default_help('n_bootstraps')}",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help=f"DuckDB threads. {_runtime_default_help('threads')}",
    )
    if include_memory_limit:
        parser.add_argument(
            "--memory-limit",
            default=None,
            help=f"DuckDB memory limit. {_runtime_default_help('memory_limit')}",
        )
    parser.add_argument(
        "--max-temp-directory-size",
        default=None,
        help=(
            "DuckDB max temp directory size. "
            f"{_runtime_default_help('max_temp_directory_size')}"
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help=f"Maximum MAP iterations. {_runtime_default_help('max_iterations')}",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help=f"MAP convergence tolerance. {_runtime_default_help('tolerance')}",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=None,
        help=f"MAP convergence check interval. {_runtime_default_help('check_interval')}",
    )
    parser.add_argument(
        "--convergence-sample",
        type=float,
        default=None,
        help=f"MAP convergence sampling fraction. {_runtime_default_help('convergence_sample')}",
    )
    parser.add_argument(
        "--min-iterations-before-check",
        type=int,
        default=None,
        help=(
            "Minimum MAP iterations before non-final convergence checks. "
            f"{_runtime_default_help('min_iterations_before_check')}"
        ),
    )
    parser.add_argument(
        "--check-interval-growth",
        type=_parse_bool,
        default=None,
        metavar="BOOL",
        help=(
            "Whether MAP convergence checks become less frequent over time. "
            f"{_runtime_default_help('check_interval_growth')}"
        ),
    )
    parser.add_argument(
        "--max-check-interval",
        type=int,
        default=None,
        help=f"Upper bound for adaptive MAP check intervals. {_runtime_default_help('max_check_interval')}",
    )
    parser.add_argument(
        "--singleton-pruning",
        choices=["iterative", "one_pass"],
        default=None,
        help=f"Singleton pruning strategy. {_runtime_default_help('singleton_pruning')}",
    )
    parser.add_argument(
        "--fe-order",
        choices=["input", "ascending_groups", "descending_groups"],
        default=None,
        help=f"Order of FE sweeps during MAP. {_runtime_default_help('fe_order')}",
    )
    parser.add_argument(
        "--drop-constant-variables",
        type=_parse_bool,
        default=None,
        metavar="BOOL",
        help=(
            "Whether to skip constant residual columns during MAP. "
            f"{_runtime_default_help('drop_constant_variables')}"
        ),
    )
    parser.add_argument(
        "--residual-type",
        choices=["DOUBLE", "FLOAT"],
        default=None,
        help=f"Residual storage type. {_runtime_default_help('residual_type')}",
    )


def register(top_subparsers: argparse._SubParsersAction) -> None:
    """Register ``analysis`` and its sub-commands on *top_subparsers*."""
    from .handlers import (
        handle_cleanup,
        handle_run,
        handle_submit,
        handle_summary,
        handle_tables,
    )

    analysis_parser = top_subparsers.add_parser(
        "analysis",
        help="Run models, submit SLURM jobs, render tables, and manage results",
        description="Analysis sub-system: model execution, SLURM submission, table rendering.",
    )
    add_logging_args(analysis_parser)
    sub = analysis_parser.add_subparsers(
        dest="analysis_cmd",
        metavar="COMMAND",
    )
    sub.required = True

    # ── run ────────────────────────────────────────────────────────────────
    run_p = sub.add_parser(
        "run",
        help="Run a single model (no SLURM)",
        description="Execute a single analysis model via DuckReg on this machine.",
    )
    add_logging_args(run_p)
    run_p.add_argument(
        "--config",
        default="orchestration/configs/analysis.xlsx",
        help="Path to analysis.xlsx (default: orchestration/configs/analysis.xlsx)",
    )
    run_p.add_argument(
        "--model", "-m",
        required=False,
        help="Model name to run",
    )
    run_p.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    run_p.add_argument(
        "--dataset",
        help="Override dataset path (overrides data_source in specification)",
    )
    run_p.add_argument(
        "--fe",
        choices=CLI_FIXED_EFFECT_CHOICES,
        help="Override fixed effects for this run",
    )
    run_p.add_argument(
        "--resolution",
        choices=["500m", "1km", "5km", "50km", "ADM2"],
        help="Override dataset resolution for this run",
    )
    run_p.add_argument(
        "--clustering",
        choices=["ADM2", "Country"],
        help="Override clustering for this run",
    )
    run_p.add_argument(
        "--temporal-extent",
        help="Override temporal extent for this run (YYYY-YYYY)",
    )
    run_p.add_argument(
        "--spatial-extent",
        help="Override spatial extent for this run",
    )
    run_p.add_argument(
        "--output", "-o",
        help="Output directory for analysis results",
    )
    _add_runtime_setting_args(run_p)
    run_p.set_defaults(func=handle_run)

    # ── submit ─────────────────────────────────────────────────────────────
    submit_p = sub.add_parser(
        "submit",
        help="Submit table/model sets as a SLURM batch job",
        description="Write and submit a SLURM batch script that runs multiple models.",
    )
    add_logging_args(submit_p)
    submit_p.add_argument(
        "--analysis-config",
        help="Path to analysis.xlsx (overrides default)",
    )
    submit_p.add_argument(
        "--tables",
        nargs="+",
        metavar="TABLE",
        required=False,
        help="Table names to submit (all models in each table are included)",
    )
    submit_p.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        required=False,
        help="Individual model names to submit",
    )
    submit_p.add_argument(
        "--fe",
        choices=CLI_FIXED_EFFECT_CHOICES,
        help="Override fixed effects for individually submitted models",
    )
    submit_p.add_argument(
        "--resolution",
        choices=["500m", "1km", "5km", "50km", "ADM2"],
        help="Override resolution for individually submitted models",
    )
    submit_p.add_argument(
        "--clustering",
        choices=["ADM2", "Country"],
        help="Override clustering for individually submitted models",
    )
    submit_p.add_argument(
        "--temporal-extent",
        help="Override temporal extent for individually submitted models (YYYY-YYYY)",
    )
    submit_p.add_argument(
        "--spatial-extent",
        help="Override spatial extent for individually submitted models",
    )
    submit_p.add_argument(
        "--mem",
        default="128GB",
        help="SLURM memory request (default: 128GB)",
    )
    submit_p.add_argument("--time",          default=None,      help="SLURM time limit override")
    submit_p.add_argument("--qos",           default="1week",   help="SLURM QOS (default: 1week)")
    submit_p.add_argument(
        "--partition",
        default=None,
        help="SLURM partition override (default: auto-select scicore, or bigmem for mem >=256GB)",
    )
    submit_p.add_argument("--cpus-per-task", type=int, default=8, help="SLURM CPUs per task (default: 8)")
    submit_p.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Include models that already have results (default: skip them)",
    )
    _add_runtime_setting_args(submit_p, include_memory_limit=False)
    submit_p.set_defaults(func=handle_submit)

    # ── summary ────────────────────────────────────────────────────────────
    summary_p = sub.add_parser(
        "summary",
        help="Print status overview of all tables",
        description="Print a status overview of all tables and their latest results.",
    )
    add_logging_args(summary_p)
    summary_p.add_argument(
        "--analysis-config",
        help="Path to analysis.xlsx (overrides default)",
    )
    summary_p.set_defaults(func=handle_summary)

    # ── tables ─────────────────────────────────────────────────────────────
    tables_p = sub.add_parser(
        "tables",
        help="Render HTML / LaTeX table files",
        description="Generate HTML and/or LaTeX table files from analysis results.",
    )
    add_logging_args(tables_p)
    tables_p.add_argument(
        "--analysis-config",
        help="Path to analysis.xlsx (overrides default)",
    )
    tables_p.add_argument(
        "--source",
        help="Render only this table (default: all tables)",
    )
    tables_p.add_argument(
        "--output-dir",
        help="Output directory for generated tables (overrides config default)",
    )
    tables_p.add_argument(
        "--formats",
        nargs="+",
        choices=["html", "latex", "tex"],
        metavar="FMT",
        help="Output formats: html, latex, tex (default: from Excel or html)",
    )
    tables_p.set_defaults(func=handle_tables)

    # ── cleanup ────────────────────────────────────────────────────────────
    cleanup_p = sub.add_parser(
        "cleanup",
        help="Remove stale analysis result files",
        description=(
            "Prune old result files from the output directory, "
            "keeping only the latest per version group."
        ),
    )
    add_logging_args(cleanup_p)
    cleanup_p.add_argument(
        "--output", "-o",
        help="Output directory to clean (default: output/analysis)",
    )
    cleanup_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    cleanup_p.set_defaults(func=handle_cleanup)

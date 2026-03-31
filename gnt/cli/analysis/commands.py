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
        "--output", "-o",
        help="Output directory for analysis results",
    )
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
        "--source",
        help="Single table or model name (auto-detected; legacy alternative to --tables / --models)",
    )
    submit_p.add_argument("--mem",           default="128GB",   help="SLURM memory (default: 128GB)")
    submit_p.add_argument("--time",          default=None,      help="SLURM time limit override")
    submit_p.add_argument("--qos",           default="1week",   help="SLURM QOS (default: 1week)")
    submit_p.add_argument("--partition",     default="scicore", help="SLURM partition (default: scicore)")
    submit_p.add_argument("--cpus-per-task", type=int, default=8, help="SLURM CPUs per task (default: 8)")
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

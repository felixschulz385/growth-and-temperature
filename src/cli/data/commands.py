"""
Argument registration for the ``data`` domain.

Sub-commands
------------
list       List registered sources, their aliases, steps, and requirements.
summary    Print a concise data-availability overview across all sources/steps.
plan       Print the target list for a (source, step) without running anything.
index      Build/refresh a source's completion index (FETCH-capable sources only).
reconcile  Rebuild a source's ledger from real on-disk/HPC filesystem state.
run        Execute a (source, step)'s pending targets.
transfer   Push a step's local output to the HPC target over SSH.

docs/design/09-integrated-pipeline.md §8, docs/design/10-fetch-ledger.md.
"""

from __future__ import annotations

import argparse

from src.cli.common import add_config_arg, add_logging_args, add_source_arg
from src.data.common.dask.client import DEFAULT_DASHBOARD_PORT


def _add_step_arg(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument(
        "--step",
        choices=["fetch", "prepare", "grid"],
        required=required,
        help="Pipeline step (no default -- docs/design/09-integrated-pipeline.md §2)",
    )


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--years",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Restrict to a year range (inclusive)",
    )
    parser.add_argument(
        "--key",
        nargs="+",
        dest="keys",
        help="Restrict to specific target keys (e.g. one year, or year/tile)",
    )


def register(top_subparsers: argparse._SubParsersAction) -> None:
    """Register ``data`` and its sub-commands on *top_subparsers*."""
    from .handlers import (
        handle_list,
        handle_plan,
        handle_reconcile,
        handle_run,
        handle_summary,
        handle_transfer,
    )

    data_parser = top_subparsers.add_parser(
        "data",
        help="Fetch/prepare/grid a source (replaces download + preprocess run)",
        description=(
            "Run one source through its fetch/prepare/grid lifecycle "
            "(docs/design/09-integrated-pipeline.md)."
        ),
    )
    add_logging_args(data_parser)
    sub = data_parser.add_subparsers(dest="data_cmd", metavar="COMMAND")
    sub.required = True

    # ── list ───────────────────────────────────────────────────────────────
    list_p = sub.add_parser("list", help="List registered sources")
    add_logging_args(list_p)
    list_p.set_defaults(func=handle_list)

    # ── summary ────────────────────────────────────────────────────────────
    summary_p = sub.add_parser(
        "summary",
        help="Print a concise data-availability overview across all sources/steps",
        description=(
            "For every configured source, show which of fetch/prepare/grid it "
            "supports and how much of that step's output already exists locally."
        ),
    )
    add_logging_args(summary_p)
    add_config_arg(summary_p)
    summary_p.add_argument(
        "--source",
        help="Restrict the overview to a single source name (default: all configured sources)",
    )
    summary_p.add_argument(
        "--detailed", action="store_true",
        help="For FETCH rows, break the outstanding count down into never-attempted vs. retrying",
    )
    # Per-target INFO/WARNING chatter from individual sources' plan() (e.g.
    # MODIS logging one line per missing stage-1 year) defeats the point of a
    # *concise* overview across every source -- default this subcommand's
    # --log-level to ERROR (an explicit `--log-level`/`--debug` still wins).
    for action in summary_p._actions:
        if action.dest == "log_level":
            action.default = "ERROR"
            action.help = "Set the logging level (default: ERROR, quieter than other subcommands)"
    summary_p.set_defaults(func=handle_summary)

    # ── plan ───────────────────────────────────────────────────────────────
    plan_p = sub.add_parser("plan", help="Print targets for (source, step) without running them")
    add_logging_args(plan_p)
    add_config_arg(plan_p)
    add_source_arg(plan_p)
    _add_step_arg(plan_p, required=True)
    _add_selection_args(plan_p)
    plan_p.set_defaults(func=handle_plan)

    # ── reconcile ──────────────────────────────────────────────────────────
    reconcile_p = sub.add_parser(
        "reconcile",
        help="Rebuild a source's ledger from real on-disk/HPC filesystem state",
        description=(
            "One-time/occasional bootstrap: reconcile a source's DuckDB ledger against "
            "what's actually on disk and (if configured) on the HPC target, rather than "
            "trusting the ledger's own prior state (docs/design/10-fetch-ledger.md §5)."
        ),
    )
    add_logging_args(reconcile_p)
    add_config_arg(reconcile_p)
    add_source_arg(reconcile_p)
    reconcile_p.add_argument(
        "--step",
        choices=["prepare", "grid", "all"],
        default="all",
        help="Restrict reconciliation to one step (default: every PREPARE/GRID step the source "
             "implements -- FETCH has nothing to reconcile: it's ledger-free, always derived live "
             "from a directory listing).",
    )
    reconcile_p.set_defaults(func=handle_reconcile)

    # ── run ────────────────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Execute a (source, step)'s pending targets")
    add_logging_args(run_p)
    add_config_arg(run_p)
    add_source_arg(run_p)
    _add_step_arg(run_p, required=True)
    _add_selection_args(run_p)
    run_p.add_argument("--override", action="store_true", help="Re-run targets even if already complete")
    # Dask sizing -- carried over from the old `preprocess run` flags of the
    # same name (docs/design/09-integrated-pipeline.md §8's SourceConfig/
    # PipelineContext replaces the kwargs-smearing, not these CLI knobs).
    # generate_slurm_scripts.py emits all four for every non-"simple" job, so
    # their absence here was a latent crash-at-argparse bug caught by the
    # step-9 hard-gate dry run before any real data was touched.
    run_p.add_argument("--dask-threads", type=int, help="Number of Dask threads")
    run_p.add_argument("--dask-memory-limit", help='Dask memory limit per worker (e.g. "4GB")')
    run_p.add_argument("--temp-dir", help="Temporary/spill directory for Dask workers")
    run_p.add_argument(
        "--dashboard-port", type=int, default=DEFAULT_DASHBOARD_PORT,
        help=f"Dask dashboard port (default: {DEFAULT_DASHBOARD_PORT})",
    )
    run_p.set_defaults(func=handle_run)

    # ── transfer ───────────────────────────────────────────────────────────
    transfer_p = sub.add_parser("transfer", help="Push a step's local output to the HPC target over SSH")
    add_logging_args(transfer_p)
    add_config_arg(transfer_p)
    add_source_arg(transfer_p)
    _add_step_arg(transfer_p, required=True)
    transfer_p.add_argument("--direction", choices=["push", "pull"], default="push", help="Transfer direction (default: push)")
    transfer_p.add_argument("--override", action="store_true", help="Re-transfer units already marked completed")
    transfer_p.add_argument(
        "--watch", action="store_true",
        help="Stay running, re-scanning for newly-completed local output and pushing it as it "
             "appears (e.g. alongside a concurrently-running `data run --step fetch`). "
             "Runs until interrupted (Ctrl-C).",
    )
    transfer_p.add_argument(
        "--poll-interval", type=float, default=30.0,
        help="Seconds to wait between re-scans in --watch mode (default: 30)",
    )
    transfer_p.set_defaults(func=handle_transfer)

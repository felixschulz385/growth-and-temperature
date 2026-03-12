"""
Argument registration for the ``preprocess`` domain.

Sub-commands
------------
run   Preprocess raw source files into analysis-ready data.
"""

from __future__ import annotations

import argparse

from gnt.cli.common import add_logging_args


def register(top_subparsers: argparse._SubParsersAction) -> None:
    """Register ``preprocess`` and its sub-commands on *top_subparsers*."""
    from .handlers import handle_run

    preprocess_parser = top_subparsers.add_parser(
        "preprocess",
        help="Preprocess raw source files",
        description="Preprocess raw data files into analysis-ready form.",
    )
    add_logging_args(preprocess_parser)
    sub = preprocess_parser.add_subparsers(
        dest="preprocess_cmd",
        metavar="COMMAND",
    )
    sub.required = True

    # ── run ────────────────────────────────────────────────────────────────
    run_p = sub.add_parser(
        "run",
        help="Preprocess raw source files",
        description="Preprocess raw data files for the specified source.",
    )
    add_logging_args(run_p)
    run_p.add_argument(
        "--config",
        required=True,
        help="Path to unified configuration file (YAML or JSON)",
    )
    run_p.add_argument(
        "--source",
        required=True,
        help="Data source name as defined in the configuration file",
    )
    run_p.add_argument(
        "--mode",
        help="Override operation mode (for advanced usage)",
    )
    run_p.add_argument(
        "--stage",
        help="Processing stage (e.g. annual, spatial, vector)",
    )
    run_p.add_argument(
        "--subsource",
        help="Subsource name for misc preprocessor (e.g. osm, gadm)",
    )
    run_p.add_argument("--year", type=int, help="Specific year to process")
    run_p.add_argument(
        "--year-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Year range to process (start end)",
    )
    run_p.add_argument(
        "--grid-cells",
        nargs="+",
        help="Grid cells to process (MODIS only)",
    )
    run_p.add_argument(
        "--override",
        action="store_true",
        help="Override existing outputs",
    )
    run_p.add_argument(
        "--admin-level",
        type=int,
        choices=[1, 2],
        help="Administrative level for PLAD preprocessor (1 or 2)",
    )
    # Dask configuration
    run_p.add_argument("--dask-threads", type=int, help="Number of Dask threads")
    run_p.add_argument(
        "--dask-memory-limit",
        help='Dask memory limit per worker (e.g. "4GB")',
    )
    run_p.add_argument("--temp-dir", help="Temporary directory for Dask")
    run_p.add_argument(
        "--dashboard-port",
        type=int,
        default=8787,
        help="Dask dashboard port (default: 8787)",
    )
    run_p.add_argument(
        "--local-directory",
        help="Directory for Dask worker spilling",
    )
    run_p.set_defaults(func=handle_run)

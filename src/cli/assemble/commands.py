"""
Argument registration for the ``assemble`` domain.

Sub-commands
------------
create   Recreate all tiles in an assembled dataset.
update   Add or refresh one datasource in an existing assembled dataset.
"""

from __future__ import annotations

import argparse

from src.cli.common import add_logging_args
from src.data.common.dask.client import DEFAULT_DASHBOARD_PORT


def register(top_subparsers: argparse._SubParsersAction) -> None:
    """Register ``assemble`` and its sub-commands on *top_subparsers*."""
    from .handlers import handle_create, handle_update

    assemble_parser = top_subparsers.add_parser(
        "assemble",
        help="Assemble or update the panel dataset",
        description="Build or maintain the assembled panel dataset.",
    )
    add_logging_args(assemble_parser)
    sub = assemble_parser.add_subparsers(
        dest="assemble_cmd",
        metavar="COMMAND",
    )
    sub.required = True

    # ── create ─────────────────────────────────────────────────────────────
    create_p = sub.add_parser(
        "create",
        help="Recreate all tiles (default assembly mode)",
        description="Recreate all tiles for the specified assembly.",
    )
    _add_assemble_common(create_p)
    create_p.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Overwrite existing tiles (default: True)",
    )
    create_p.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Skip existing tiles instead of overwriting",
    )
    create_p.set_defaults(func=handle_create)

    # ── update ─────────────────────────────────────────────────────────────
    update_p = sub.add_parser(
        "update",
        help="Update existing tiles with a new datasource",
        description="Add or refresh one datasource in an existing assembly.",
    )
    _add_assemble_common(update_p)
    update_p.add_argument(
        "--datasource",
        required=True,
        help="Datasource name to update",
    )
    update_p.set_defaults(func=handle_update)


# ---------------------------------------------------------------------------
# Shared argument helpers
# ---------------------------------------------------------------------------

def _add_assemble_common(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by create and update sub-commands."""
    add_logging_args(parser)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to unified configuration file (YAML or JSON)",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Assembly name as defined in the configuration file",
    )
    # Dask configuration
    parser.add_argument("--dask-threads", type=int, help="Number of Dask threads")
    parser.add_argument(
        "--dask-memory-limit",
        help='Dask memory limit per worker (e.g. "4GB")',
    )
    parser.add_argument("--temp-dir", help="Temporary directory")
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=DEFAULT_DASHBOARD_PORT,
        help=f"Dask dashboard port (default: {DEFAULT_DASHBOARD_PORT})",
    )
    parser.add_argument(
        "--local-directory",
        help="Directory for Dask worker spilling",
    )
    parser.add_argument("--tile-size", type=int, help="Tile size override")
    parser.add_argument("--compression", help="Parquet compression format override")

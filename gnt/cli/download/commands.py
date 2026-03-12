"""
Argument registration for the ``download`` domain.

Sub-commands
------------
index   Build or sync the remote file index.
run     Download files from a remote source.
"""

from __future__ import annotations

import argparse

from gnt.cli.common import add_logging_args


def register(top_subparsers: argparse._SubParsersAction) -> None:
    """Register ``download`` and its sub-commands on *top_subparsers*."""
    from .handlers import handle_index, handle_run

    download_parser = top_subparsers.add_parser(
        "download",
        help="Index and download remote data files",
        description="Build the file index or download remote data files.",
    )
    add_logging_args(download_parser)
    sub = download_parser.add_subparsers(
        dest="download_cmd",
        metavar="COMMAND",
    )
    sub.required = True

    # ── index ──────────────────────────────────────────────────────────────
    index_p = sub.add_parser(
        "index",
        help="Build or sync the remote file index",
        description="Build or synchronise the local file index for a data source.",
    )
    _add_download_common(index_p)
    index_p.set_defaults(func=handle_index)

    # ── run (download) ─────────────────────────────────────────────────────
    run_p = sub.add_parser(
        "run",
        help="Download files from a remote source",
        description="Download pending files from a remote source to local storage.",
    )
    _add_download_common(run_p)
    run_p.add_argument(
        "--misc-files",
        nargs="+",
        metavar="KEY",
        help="Specific misc file keys to download (e.g. osm gadm hdi)",
    )
    run_p.set_defaults(func=handle_run)


def _add_download_common(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by all download sub-commands."""
    add_logging_args(parser)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to unified configuration file (YAML or JSON)",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Data source name as defined in the configuration file",
    )

"""
Argument registration for the ``download`` domain.

Sub-commands
------------
index       Build or sync the remote file index.
run         Download files from a remote source.
snf-mining  Run the S&P Global SNF Mining Selenium scraper.
"""

from __future__ import annotations

import argparse

from gnt.cli.common import add_logging_args


def register(top_subparsers: argparse._SubParsersAction) -> None:
    """Register ``download`` and its sub-commands on *top_subparsers*."""
    from .handlers import handle_index, handle_run, handle_snf_mining

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

    # ── snf-mining ─────────────────────────────────────────────────────────
    snf_p = sub.add_parser(
        "snf-mining",
        help="Run the S&P Global SNF Mining Selenium scraper",
        description=(
            "Authenticate against Capital IQ, collect mine IDs from the "
            "screener, export detail workbooks, and parse them. Data is stored in a DuckDB "
            "database under data/snf_mining/raw/."
        ),
    )
    add_logging_args(snf_p)
    snf_p.add_argument(
        "--credentials",
        required=True,
        metavar="PATH",
        help="Path to spglobal.credentials.json",
    )
    snf_p.add_argument(
        "--db",
        metavar="PATH",
        default=None,
        help=(
            "Path to the DuckDB file (default: data/snf_mining/raw/snf_mining.duckdb)"
        ),
    )
    snf_p.add_argument(
        "--stages",
        nargs="+",
        choices=["ids", "detail_exports", "detail_parse"],
        default=None,
        metavar="STAGE",
        help=(
            "Stages to run: 'ids' (collect mine IDs), 'detail_exports' "
            "(scrape subsection workbooks and geometry), and/or "
            "'detail_parse' (parse downloaded workbooks). Defaults to all."
        ),
    )
    snf_p.add_argument(
        "--force-stages",
        nargs="+",
        choices=["detail_exports", "detail_parse"],
        default=None,
        metavar="STAGE",
        help=(
            "Re-run the selected stage(s) completely for the targeted mines by "
            "clearing prior outputs and completion markers first."
        ),
    )
    snf_p.add_argument(
        "--headless",
        action="store_true",
        help="Run Chrome in headless mode (no visible window)",
    )
    snf_p.add_argument(
        "--wait",
        type=int,
        default=10,
        help="Selenium wait timeout in seconds (default: 10)",
    )
    snf_p.add_argument(
        "--download-wait",
        type=int,
        default=90,
        help="Workbook download timeout in seconds (default: 90)",
    )
    snf_p.add_argument(
        "--mine-ids",
        nargs="*",
        default=None,
        metavar="MINE_ID",
        help="Restrict detail export/parse stages to the given mine IDs",
    )
    snf_p.add_argument(
        "--subsections",
        nargs="+",
        default=None,
        metavar="SUBSECTION",
        help=(
            "Restrict detail export/parse stages to specific subsection labels "
            "(case-insensitive match, e.g. --subsections \"Property Profile\" \"Ownership\")."
        ),
    )
    snf_p.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Retry attempts for flaky export/geometry operations (default: 3)",
    )
    snf_p.add_argument(
        "--sidebar-reload-attempts",
        type=int,
        default=2,
        help="Sidebar rediscovery page reload attempts per mine (default: 2)",
    )
    snf_p.add_argument(
        "--step-sleep-seconds",
        type=float,
        default=0.35,
        help="Small sleep interval between browser actions (default: 0.35)",
    )
    snf_p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort detail parsing on the first parse failure",
    )
    snf_p.set_defaults(func=handle_snf_mining)


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

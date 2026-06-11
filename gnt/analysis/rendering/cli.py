"""CLI entrypoint for rendering analysis tables."""

from __future__ import annotations

import argparse

from ..core.config import AnalysisConfig
from .tables import generate_all_tables, summarize_tables


def main() -> int:
    """Run the table rendering CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Work with analysis tables defined in orchestration/configs/analysis.xlsx.\n\n"
            "Modes:\n"
            "  summary   Print a status overview of every table and model.\n"
            "  generate  Render and save table files (default)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["summary", "generate"],
        default="generate",
        help="Operation mode (default: generate)",
    )
    parser.add_argument(
        "--excel",
        default=None,
        metavar="PATH",
        help="Path to analysis.xlsx (default: orchestration/configs/analysis.xlsx)",
    )
    parser.add_argument(
        "--tables",
        nargs="*",
        metavar="TABLE",
        default=None,
        help="Specific table names to process (default: all tables)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Output directory for generated table files (default: output/analysis/tables)",
    )
    parser.add_argument(
        "--formats",
        nargs="*",
        choices=["html", "latex", "tex"],
        metavar="FMT",
        default=None,
        help="Output formats: html, latex, tex (default: from Tables sheet or html)",
    )
    args = parser.parse_args()

    try:
        config = AnalysisConfig(args.excel)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    if args.mode == "summary":
        summarize_tables(config)
        return 0

    generate_all_tables(
        config=config,
        table_names=args.tables,
        output_dir=args.output_dir,
        output_formats=args.formats,
    )
    return 0


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())

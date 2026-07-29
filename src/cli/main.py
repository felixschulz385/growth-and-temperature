"""
src.cli.main — root parser and top-level entry point.

Usage (new-style):
    python -m src.cli.main download index  --config cfg.yaml --source glass
    python -m src.cli.main download run    --config cfg.yaml --source glass
    python -m src.cli.main preprocess run  --config cfg.yaml --source glass
    python -m src.cli.main assemble create --config cfg.yaml --source main
    python -m src.cli.main assemble update --config cfg.yaml --source main --datasource ntl
    python -m src.cli.main assemble demean --config cfg.yaml --source main
    python -m src.cli.main analysis run    --model my_model
    python -m src.cli.main analysis submit --tables table_main
    python -m src.cli.main analysis summary
    python -m src.cli.main analysis tables
    python -m src.cli.main analysis cleanup
    python -m src.cli.main analysis subsets generate
    python -m src.cli.main analysis subsets list

The module is also the delegate for the compatibility shim in ``run.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when invoked directly
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]  # src/cli/main.py → project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.cli import analysis, assemble, download, preprocess

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="src",
        description=(
            "GNT Data System — unified entry point for download, "
            "preprocessing, assembly, and analysis."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  src download index  --config cfg.yaml --source glass
  src download run    --config cfg.yaml --source glass
  src preprocess run  --config cfg.yaml --source glass
  src assemble create --config cfg.yaml --source main_panel
  src assemble update --config cfg.yaml --source main_panel --datasource ntl
  src assemble demean --config cfg.yaml --source main_panel
  src analysis run    --model baseline_ols
  src analysis submit --tables table_main table_robustness
  src analysis summary
  src analysis tables --formats html latex
  src analysis cleanup --dry-run
  src analysis subsets generate
  src analysis subsets list
""",
    )

    subparsers = parser.add_subparsers(
        dest="domain",
        metavar="DOMAIN",
    )
    subparsers.required = True

    # Register each domain
    download.register(subparsers)
    preprocess.register(subparsers)
    assemble.register(subparsers)
    analysis.register(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse *argv* (or ``sys.argv[1:]``) and dispatch to the handler.

    Returns the integer exit code (0 = success, 1 = error).
    """
    # Configure a minimal logger before arg parsing so early errors are visible
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        # Sub-command omitted — print domain-level help
        parser.print_help()
        return 1

    try:
        args.func(args)
        logger.info("Operation completed successfully")
        return 0
    except SystemExit as exc:
        # Handlers may raise SystemExit directly for error conditions
        return int(exc.code) if exc.code is not None else 1
    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

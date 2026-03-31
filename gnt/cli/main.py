"""
gnt.cli.main — root parser and top-level entry point.

Usage (new-style):
    python -m gnt.cli.main download index  --config cfg.yaml --source glass
    python -m gnt.cli.main download run    --config cfg.yaml --source glass
    python -m gnt.cli.main preprocess run  --config cfg.yaml --source glass
    python -m gnt.cli.main assemble create --config cfg.yaml --source main
    python -m gnt.cli.main assemble update --config cfg.yaml --source main --datasource ntl
    python -m gnt.cli.main assemble demean --config cfg.yaml --source main
    python -m gnt.cli.main analysis run    --model my_model
    python -m gnt.cli.main analysis submit --tables table_main
    python -m gnt.cli.main analysis summary
    python -m gnt.cli.main analysis tables
    python -m gnt.cli.main analysis cleanup

The module is also the delegate for the compatibility shim in ``run.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when invoked directly
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]  # gnt/cli/main.py → project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gnt.cli import analysis, assemble, download, preprocess

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="gnt",
        description=(
            "GNT Data System — unified entry point for download, "
            "preprocessing, assembly, and analysis."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gnt download index  --config cfg.yaml --source glass
  gnt download run    --config cfg.yaml --source glass
  gnt preprocess run  --config cfg.yaml --source glass
  gnt assemble create --config cfg.yaml --source main_panel
  gnt assemble update --config cfg.yaml --source main_panel --datasource ntl
  gnt assemble demean --config cfg.yaml --source main_panel
  gnt analysis run    --model baseline_ols
  gnt analysis submit --tables table_main table_robustness
  gnt analysis summary
  gnt analysis tables --formats html latex
  gnt analysis cleanup --dry-run
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

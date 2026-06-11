#!/usr/bin/env python3
"""
Project-local wrapper for the modular GNT CLI.

Examples:
  python run.py download run --config config.yaml --source glass
  python run.py download index --config config.yaml --source glass
  python run.py preprocess run --config config.yaml --source glass
  python run.py assemble create --config config.yaml --source main_panel
  python run.py assemble update --config config.yaml --source main_panel --datasource ntl
  python run.py assemble demean --config config.yaml --source main_panel
  python run.py analysis run --model my_model
  python run.py analysis submit --tables table_main table_robustness
  python run.py analysis summary
  python run.py analysis tables --source table_main --formats html latex
  python run.py analysis cleanup --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    """Run the modular GNT CLI."""
    from gnt.cli.main import main as cli_main

    return cli_main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    sys.exit(main())

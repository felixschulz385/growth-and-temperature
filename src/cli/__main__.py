"""``python -m src.cli`` entry point.

Kept deliberately to an import and a ``sys.exit`` call -- all parser
construction and dispatch lives in ``main.py`` and the per-domain modules.
"""

from __future__ import annotations

import sys

from src.cli.main import main

if __name__ == "__main__":
    sys.exit(main())

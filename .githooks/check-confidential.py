#!/usr/bin/env python3
"""Pre-commit check: keep confidential mine data out of git.

The S&P Global / SNL mine database (and anything derived from it at mine level:
ids, names, coordinates, per-mine attributes, maps of mine points or of mine
counts per grid cell) is licensed and must never be committed. See CLAUDE.md
§ "Confidential mine data". This hook rejects staged files that are likely to
carry it:

1. anything under a ``confidential/`` directory (also git-ignored);
2. data files (.csv .parquet .duckdb .db .xls .xlsx .gpkg .geojson .feather
   .pkl .zarr) -- results tables that are pure estimates can be allowed
   through ``.githooks/confidential-allowlist.txt``;
3. figures and documents (.png .jpg .jpeg .gif .svg .pdf .html) whose path
   mentions mine, mining, snl or corridor, unless allow-listed.

It cannot read content, so it is a backstop, not a guarantee: notebooks are
covered by check-notebook-outputs.py (no outputs), and docs and commit
messages by the CLAUDE.md rule. Stdlib only.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
import sys
from pathlib import Path

HOOK_DIR = Path(__file__).resolve().parent
ALLOWLIST = HOOK_DIR / "confidential-allowlist.txt"

DATA_EXT = (".csv", ".parquet", ".duckdb", ".db", ".xls", ".xlsx", ".gpkg", ".geojson", ".feather", ".pkl", ".zarr")
FIGURE_EXT = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".html")
MINE_WORDS = re.compile(r"mine|mining|snl|corridor", re.IGNORECASE)


def staged_paths() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"], capture_output=True, check=True
    ).stdout.decode()
    return [p for p in out.split("\0") if p]


def allowlist() -> list[str]:
    if not ALLOWLIST.exists():
        return []
    return [l.strip() for l in ALLOWLIST.read_text().splitlines() if l.strip() and not l.strip().startswith("#")]


def reason(path: str, allowed: list[str]) -> str | None:
    if "confidential/" in f"/{path}" or path.startswith("confidential/"):
        return "under a confidential/ directory"
    if any(fnmatch.fnmatch(path, pat) for pat in allowed):
        return None
    low = path.lower()
    if low.endswith(DATA_EXT):
        return "data file"
    if low.endswith(FIGURE_EXT) and MINE_WORDS.search(path):
        return "figure/document whose name mentions mines"
    return None


def main() -> int:
    allowed = allowlist()
    bad = [(p, r) for p in staged_paths() if (r := reason(p, allowed))]
    if not bad:
        return 0
    print("pre-commit: blocked -- these files may carry confidential mine data:\n", file=sys.stderr)
    for p, r in bad:
        print(f"  {p}  ({r})", file=sys.stderr)
    print(
        "\nMine-level data (ids, names, locations, per-mine attributes, maps of mines) must not be\n"
        "committed (CLAUDE.md). Unstage with `git restore --staged <path>`. If a file holds only\n"
        "aggregate estimates, add its path to .githooks/confidential-allowlist.txt.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

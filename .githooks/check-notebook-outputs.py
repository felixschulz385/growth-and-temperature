#!/usr/bin/env python3
"""Pre-commit check: reject staged Jupyter notebooks that carry cell outputs
where outputs could leak confidential mine data (CLAUDE.md).

A notebook "carries output" if any code cell has a non-empty ``outputs`` list or
a non-null ``execution_count``. Notebooks whose repo-relative path matches
``.githooks/notebook-output-whitelist.txt`` are exempt; a ``!pattern`` line
re-blocks a path, and the last matching line wins. A notebook whose code reads a
mine-level source (``MINE_SOURCES``) is blocked even when whitelisted.

Only the *staged* content of each notebook is inspected (``git show :<path>``),
so unstaged scratch state in the working tree never blocks a commit, and
notebooks that already have outputs in history are only flagged once you stage a
fresh change to them.

Stdlib only -- runs under any Python 3, no need for the ``gnt`` env to be active.
"""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path

HOOK_DIR = Path(__file__).resolve().parent
REPO_ROOT = HOOK_DIR.parent
WHITELIST = HOOK_DIR / "notebook-output-whitelist.txt"
# code that reads mine-level data: the SNL and Berman sources and per-mine ids
MINE_SOURCES = re.compile(r"snl_mining|berman_mining|BCRT_|property_id", re.IGNORECASE)


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], capture_output=True, check=True
    ).stdout


def staged_notebooks() -> list[str]:
    out = _git(
        "diff", "--cached", "--name-only", "--diff-filter=ACM", "-z"
    ).decode()
    return [p for p in out.split("\0") if p.endswith(".ipynb")]


def load_whitelist() -> list[str]:
    if not WHITELIST.exists():
        return []
    patterns = []
    for raw in WHITELIST.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            patterns.append(line)
    return patterns


def is_whitelisted(path: str, patterns: list[str]) -> bool:
    allowed = False
    for pat in patterns:
        negate = pat.startswith("!")
        if fnmatch.fnmatch(path, pat[1:] if negate else pat):
            allowed = not negate
    return allowed


def reads_mine_data(nb_bytes: bytes) -> bool:
    try:
        nb = json.loads(nb_bytes)
    except json.JSONDecodeError:
        return False
    return any(
        cell.get("cell_type") == "code" and MINE_SOURCES.search("".join(cell.get("source", [])))
        for cell in nb.get("cells", [])
    )


def offending_kinds(nb_bytes: bytes) -> set[str]:
    """Return which kinds of output content the notebook carries, if any."""
    try:
        nb = json.loads(nb_bytes)
    except json.JSONDecodeError:
        # Malformed JSON: not this hook's job to complain.
        return set()
    kinds: set[str] = set()
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            kinds.add("outputs")
        if cell.get("execution_count") is not None:
            kinds.add("execution_count")
    return kinds


def main() -> int:
    patterns = load_whitelist()
    bad: dict[str, set[str]] = {}
    for path in staged_notebooks():
        nb_bytes = _git("show", f":{path}")
        kinds = offending_kinds(nb_bytes)
        if not kinds:
            continue
        if reads_mine_data(nb_bytes):
            bad[path] = kinds | {"reads mine-level data"}
        elif not is_whitelisted(path, patterns):
            bad[path] = kinds

    if not bad:
        return 0

    red = "\033[31m" if sys.stderr.isatty() else ""
    reset = "\033[0m" if sys.stderr.isatty() else ""
    print(
        f"\n{red}Commit blocked:{reset} staged notebook(s) contain cell outputs "
        "or execution counts.\n",
        file=sys.stderr,
    )
    for path, kinds in sorted(bad.items()):
        print(f"  {path}  ({', '.join(sorted(kinds))})", file=sys.stderr)
    rel_whitelist = WHITELIST.relative_to(REPO_ROOT)
    print(
        "\nStrip outputs before committing, e.g.:\n"
        "  jupyter nbconvert --clear-output --inplace <notebook>\n"
        "  # or, if installed:  nbstripout <notebook>\n"
        "then `git add` the notebook again.\n\n"
        f"Notebooks that read mine-level data must stay stripped. Any other notebook\n"
        f"can keep its outputs through a pattern in\n"
        f"  {rel_whitelist}\n"
        "\nTo bypass this check for one commit:  git commit --no-verify\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Pre-commit check: reject staged Jupyter notebooks that carry cell outputs.

A notebook "carries output" if any code cell has a non-empty ``outputs`` list or
a non-null ``execution_count``. Notebooks whose repo-relative path matches a
pattern in ``.githooks/notebook-output-whitelist.txt`` are exempt.

Only the *staged* content of each notebook is inspected (``git show :<path>``),
so unstaged scratch state in the working tree never blocks a commit, and
notebooks that already have outputs in history are only flagged once you stage a
fresh change to them.

Stdlib only -- runs under any Python 3, no need for the ``gnt`` env to be active.
"""

from __future__ import annotations

import fnmatch
import json
import subprocess
import sys
from pathlib import Path

HOOK_DIR = Path(__file__).resolve().parent
REPO_ROOT = HOOK_DIR.parent
WHITELIST = HOOK_DIR / "notebook-output-whitelist.txt"


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
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


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
        if is_whitelisted(path, patterns):
            continue
        kinds = offending_kinds(_git("show", f":{path}"))
        if kinds:
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
        f"If a notebook is meant to keep its outputs, add a matching pattern to\n"
        f"  {rel_whitelist}\n"
        "\nTo bypass this check for one commit:  git commit --no-verify\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

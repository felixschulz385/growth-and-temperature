"""Small per-unit JSON sidecar files for state that plain `ls` genuinely
can't see (a FETCH unit's retry/error history; a PREPARE tile's completion,
once that lands too). Deliberately dumb: read/write/remove
one JSON file at a time, atomically. No index, no cross-file query -- callers
that need "all outstanding units" get that from a directory listing plus a
handful of these reads, not from this module.

Write discipline mirrors `src/data/sources/verify.py::_write_cache`: temp
file + `os.replace` so a killed-mid-write process never leaves a partially
written status file that looks valid to `json.load`.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

STATUS_SUBDIR = "_status"

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]")


def sanitize_unit_id(unit_id: str) -> str:
    """A unit id (e.g. `"2020/h12v09"`, `"commodity_prices"`) as a single
    filesystem-safe filename component (replace, don't hash, so the file
    stays human-inspectable)."""
    return _UNSAFE_CHARS.sub("_", unit_id)


def status_path(base_dir: str, unit_id: str, *, subdir: str = STATUS_SUBDIR) -> str:
    """Where *unit_id*'s status file lives, sibling to (not inside) whatever
    directory *base_dir* is -- e.g. `<raw_root>/_status/<unit>.json` for a
    FETCH unit, `<output_dir>/_status/<tile>_<year>.json` for a PREPARE tile."""
    return os.path.join(base_dir, subdir, f"{sanitize_unit_id(unit_id)}.json")


def read(path: str) -> Optional[dict]:
    """The status dict at *path*, or `None` if it doesn't exist or is
    corrupt (a killed-mid-write file predating the atomic-replace discipline,
    or simply never written) -- treated as "no status recorded", not an
    error, same as every other "not there yet" case in this codebase."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def write(path: str, data: dict[str, Any]) -> None:
    """Atomically write *data* to *path*, stamping `updated_at`. Caller's
    keys win over the stamp only if they explicitly set `updated_at`
    themselves (unusual, but not blocked)."""
    payload = {"updated_at": datetime.now(timezone.utc).isoformat(), **data}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp_path, path)


def remove(path: str) -> None:
    """Best-effort delete -- clearing a status file (e.g. a FETCH unit that
    previously failed now succeeding) is a cleanup step, not something a
    caller should crash over if it's already gone."""
    try:
        os.remove(path)
    except OSError:
        pass

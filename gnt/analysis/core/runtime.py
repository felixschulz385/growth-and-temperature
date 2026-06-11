"""Shared analysis runtime setting defaults and helpers."""

from __future__ import annotations

import re
from typing import Any, Dict

ANALYSIS_RUNTIME_DEFAULTS: Dict[str, Any] = {
    "se_method": "CRV1",
    "fitter": "duckdb",
    "fe_method": "demean",
    "round_strata": 5,
    "seed": 42,
    "n_bootstraps": 0,
    "threads": 4,
    "memory_limit": "112GB",
    "max_temp_directory_size": "768GB",
}

_MEMORY_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+)\s*$")
_MEMORY_UNIT_TO_GB = {
    "kb": 1 / (1024 ** 2),
    "mb": 1 / 1024,
    "gb": 1,
    "tb": 1024,
}


def scale_memory_limit(memory_limit: str, factor: float) -> str:
    """Scale a memory string like ``112GB`` while preserving its unit."""
    match = _MEMORY_RE.fullmatch(str(memory_limit))
    if match is None:
        raise ValueError(f"Unsupported memory limit format: {memory_limit!r}")

    amount = float(match.group(1)) * factor
    unit = match.group(2)
    if amount.is_integer():
        amount_str = str(int(amount))
    else:
        amount_str = f"{amount:.1f}".rstrip("0").rstrip(".")
    return f"{amount_str}{unit}"


def resolve_slurm_partition(memory_limit: str, partition: str | None = None) -> str:
    """Return the explicit partition or derive one from the memory request."""
    if partition:
        return partition

    match = _MEMORY_RE.fullmatch(str(memory_limit))
    if match is None:
        raise ValueError(f"Unsupported memory limit format: {memory_limit!r}")

    amount = float(match.group(1))
    unit = match.group(2).lower()
    try:
        amount_gb = amount * _MEMORY_UNIT_TO_GB[unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported memory unit in {memory_limit!r}") from exc

    return "bigmem" if amount_gb >= 256 else "scicore"

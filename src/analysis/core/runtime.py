"""Shared analysis runtime setting defaults and helpers."""

from __future__ import annotations

import re
from typing import Any, Dict

ANALYSIS_RUNTIME_DEFAULTS: Dict[str, Any] = {
    "se_method": "CRV1",
    "fitter": "duckdb",
    "fe_method": "demean",
    "compression": 5,
    "seed": 42,
    "n_bootstraps": 0,
    "threads": 4,
    "memory_limit": "112GB",
    "max_temp_directory_size": "768GB",
    "max_iterations": 1000,
    "tolerance": 1e-8,
    "check_interval": 10,
    "convergence_sample": 1.0,
    "min_iterations_before_check": 5,
    "check_interval_growth": True,
    "max_check_interval": 25,
    "singleton_pruning": "iterative",
    "fe_order": "input",
    "drop_constant_variables": False,
    "residual_type": "DOUBLE",
}

THIRTY_MINUTES_SECONDS = 30 * 60
SIX_HOURS_SECONDS = 6 * 3600
ONE_DAY_SECONDS = 24 * 3600
ONE_WEEK_SECONDS = 7 * ONE_DAY_SECONDS
TWO_WEEKS_SECONDS = 14 * ONE_DAY_SECONDS

QOS_RUNTIME_LIMITS = (
    ("30min", THIRTY_MINUTES_SECONDS),
    ("6hours", SIX_HOURS_SECONDS),
    ("1day", ONE_DAY_SECONDS),
    ("1week", ONE_WEEK_SECONDS),
    ("2weeks", TWO_WEEKS_SECONDS),
)

RESOLUTION_CORE_DEFAULTS: Dict[str, int] = {
    "500m": 16,
    "1km": 8,
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


def recommend_slurm_qos(total_runtime_seconds: int) -> str:
    """Return the smallest QoS whose runtime ceiling covers *total_runtime_seconds*."""
    if total_runtime_seconds < 0:
        raise ValueError(
            f"Runtime must be non-negative, got {total_runtime_seconds!r} seconds."
        )

    for qos_name, max_seconds in QOS_RUNTIME_LIMITS:
        if total_runtime_seconds <= max_seconds:
            return qos_name

    raise ValueError(
        "Estimated runtime exceeds the maximum supported QoS limit "
        f"({TWO_WEEKS_SECONDS} seconds / 2weeks)."
    )


def recommended_cores_for_resolution(resolution: str) -> int:
    """Return the default core count for a single resolution label."""
    return RESOLUTION_CORE_DEFAULTS.get(str(resolution).strip(), 4)


def recommended_cores_for_model_specs(model_specs: list[dict[str, Any]]) -> int:
    """Return the auto-selected core count for a batch of model specs.

    The highest requirement among the included resolutions wins so that one
    SLURM allocation can cover every model in the sequential batch.
    """
    if not model_specs:
        return 4
    return max(
        recommended_cores_for_resolution(model_spec.get("resolution", ""))
        for model_spec in model_specs
    )

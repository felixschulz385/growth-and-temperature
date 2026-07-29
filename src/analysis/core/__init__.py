"""Shared analysis configuration and runtime helpers."""

from .config import AnalysisConfig, DEFAULT_EXCEL, PROJECT_ROOT, RESULTS_DIR
from .runtime import (
    ANALYSIS_RUNTIME_DEFAULTS,
    ONE_WEEK_SECONDS,
    QOS_RUNTIME_LIMITS,
    RESOLUTION_CORE_DEFAULTS,
    TWO_WEEKS_SECONDS,
    recommended_cores_for_model_specs,
    recommended_cores_for_resolution,
    recommend_slurm_qos,
    resolve_slurm_partition,
    scale_memory_limit,
)

__all__ = [
    "ANALYSIS_RUNTIME_DEFAULTS",
    "AnalysisConfig",
    "DEFAULT_EXCEL",
    "ONE_WEEK_SECONDS",
    "PROJECT_ROOT",
    "QOS_RUNTIME_LIMITS",
    "RESOLUTION_CORE_DEFAULTS",
    "RESULTS_DIR",
    "TWO_WEEKS_SECONDS",
    "recommended_cores_for_model_specs",
    "recommended_cores_for_resolution",
    "recommend_slurm_qos",
    "resolve_slurm_partition",
    "scale_memory_limit",
]

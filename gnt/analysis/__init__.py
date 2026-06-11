"""Analysis package public API."""

from .core.config import AnalysisConfig, DEFAULT_EXCEL, PROJECT_ROOT, RESULTS_DIR
from .execution.runner import cleanup_analysis_results, run_duckreg
from .rendering.tables import generate_all_tables, summarize_tables

__all__ = [
    "AnalysisConfig",
    "DEFAULT_EXCEL",
    "PROJECT_ROOT",
    "RESULTS_DIR",
    "cleanup_analysis_results",
    "generate_all_tables",
    "run_duckreg",
    "summarize_tables",
]

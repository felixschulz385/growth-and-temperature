"""Legacy compatibility facade for :mod:`gnt.analysis`.

New code should import from ``gnt.analysis`` directly.  This module keeps the
older ``workflow.py`` entrypoints available while delegating all behavior to
the canonical configuration and execution modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .config import AnalysisConfig, PROJECT_ROOT
from .runner import (
    build_geographic_query as _build_geographic_query,
    cleanup_analysis_results as _cleanup_analysis_results,
    load_subset,
    run_duckreg as _run_duckreg,
)


class _WorkflowConfigAdapter:
    """Adapter for the legacy dict returned by :func:`load_config`.

    ``runner.run_duckreg`` only needs ``get_model_spec``.  Keeping the adapter
    here avoids a second execution path while preserving callers that still pass
    the old nested dictionary shape.
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        output = config.get("output", {})
        self.base_path = Path(output.get("base_path", PROJECT_ROOT / "output" / "analysis"))

    def get_model_spec(
        self,
        model_name: str,
        *,
        fixed_effects: Optional[str] = None,
        resolution: Optional[str] = None,
        clustering: Optional[str] = None,
        temporal_extent: Optional[str] = None,
        spatial_extent: Optional[str] = None,
    ) -> Dict[str, Any]:
        if any(
            value is not None
            for value in (
                fixed_effects,
                resolution,
                clustering,
                temporal_extent,
                spatial_extent,
            )
        ):
            raise ValueError(
                "Variant overrides require AnalysisConfig; legacy dict configs "
                "contain only pre-resolved model specifications."
            )

        try:
            spec = self._config["analyses"]["duckreg"]["specifications"][model_name]
        except KeyError as exc:
            available = sorted(
                self._config.get("analyses", {})
                .get("duckreg", {})
                .get("specifications", {})
                .keys()
            )
            raise KeyError(
                f"Model '{model_name}' not found in legacy workflow config. "
                f"Available: {available}"
            ) from exc

        return dict(spec)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load ``analysis.xlsx`` using the canonical :class:`AnalysisConfig`.

    Returns the legacy nested dictionary shape expected by older callers:
    ``{"analyses": {"duckreg": {"specifications": ...}}, "output": ...}``.
    """
    return AnalysisConfig(config_path).as_workflow_config()


def build_geographic_query(spec_config: Dict[str, Any]) -> Optional[str]:
    """Build a Pandas-style geographic query from legacy or canonical keys."""
    if "spatial_extent" not in spec_config and "subset" in spec_config:
        spec_config = dict(spec_config)
        spec_config["spatial_extent"] = spec_config["subset"]
    return _build_geographic_query(spec_config)


def run_duckreg(
    config: AnalysisConfig | Dict[str, Any],
    spec_name: str,
    output_dir: Optional[str] = None,
    verbose: bool = True,
    dataset_override: Optional[str] = None,
) -> Any:
    """Run one DuckReg model through the canonical analysis runner.

    ``config`` may be either an :class:`AnalysisConfig` instance or the legacy
    dictionary produced by :func:`load_config`.
    """
    analysis_config: AnalysisConfig | _WorkflowConfigAdapter
    if isinstance(config, AnalysisConfig):
        analysis_config = config
    else:
        analysis_config = _WorkflowConfigAdapter(config)

    return _run_duckreg(
        analysis_config,
        spec_name,
        output_dir=output_dir,
        verbose=verbose,
        dataset_override=dataset_override,
    )


def cleanup_analysis_results(output_dir: str, dry_run: bool = False) -> None:
    """Prune stale DuckReg result files using the canonical cleanup helper."""
    _cleanup_analysis_results(output_dir, dry_run=dry_run, analysis_type="duckreg")


__all__ = [
    "load_config",
    "load_subset",
    "build_geographic_query",
    "run_duckreg",
    "cleanup_analysis_results",
]

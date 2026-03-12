"""
Result-file I/O helpers for the analysis pipeline.

Functions in this module handle:
* finding the latest result JSON for a model
* loading one or many models from disk (with graceful placeholders for missing ones)
* extracting version, date, and coefficient metadata from the stored JSON format
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_latest_model_result(
    model_name: str,
    base_path: Union[str, Path],
    analysis_type: str = 'duckreg',
) -> Path:
    """Return the path of the most recent ``results_*.json`` for *model_name*.

    Parameters
    ----------
    model_name:
        Model specification name (e.g. ``'ntlharm_twfe_replicate'``).
    base_path:
        Base output directory (``output/analysis``).
    analysis_type:
        Sub-directory under *base_path* (default ``'duckreg'``).

    Raises
    ------
    FileNotFoundError
        When the model directory or any result file is absent.
    """
    base_path = Path(base_path)
    model_dir = base_path / analysis_type / model_name

    if not model_dir.exists():
        raise FileNotFoundError(
            f"No results found for model '{model_name}' "
            f"(analysis_type={analysis_type!r}) at {model_dir}"
        )

    result_files = sorted(model_dir.glob("results_*.json"))
    if not result_files:
        raise FileNotFoundError(
            f"No result files found in {model_dir}. "
            "Expected filenames like 'results_YYYYMMDD_HHMMSS.json'."
        )

    return result_files[-1]   # lexicographic = chronological for this naming scheme


def load_model_result(
    model_name: str,
    base_path: Union[str, Path],
    analysis_type: str = 'duckreg',
) -> Dict[str, Any]:
    """Load and return the latest result dict for *model_name*.

    Raises
    ------
    FileNotFoundError
        When no result file exists (caller decides how to handle missing models).
    """
    path = find_latest_model_result(model_name, base_path, analysis_type)
    with open(path) as fh:
        return json.load(fh)


def list_model_result_files(
    model_name: str,
    base_path: Union[str, Path],
    analysis_type: str = 'duckreg',
) -> List[Path]:
    """Return all ``results_*.json`` paths for *model_name*, oldest first."""
    base_path = Path(base_path)
    model_dir = base_path / analysis_type / model_name
    if not model_dir.exists():
        return []
    return sorted(model_dir.glob("results_*.json"))


# ---------------------------------------------------------------------------
# Batch loading (used by table rendering)
# ---------------------------------------------------------------------------

def load_models_by_name(
    model_specs: List[Union[str, Dict[str, str]]],
    base_path: Union[str, Path],
) -> List[Dict[str, Any]]:
    """Load result dicts for *model_specs*, returning a placeholder for missing ones.

    Parameters
    ----------
    model_specs:
        A list of either:
        * ``str`` — model name, analysis_type defaults to ``'duckreg'``
        * ``dict`` with ``'name'`` / ``'model_name'`` key and optional
          ``'analysis_type'``
    base_path:
        Base output directory.

    Returns
    -------
    list
        One entry per spec; missing models appear as ``{'__missing__': True, …}``.
    """
    models: List[Dict[str, Any]] = []
    unavailable: List[str] = []

    for spec in model_specs:
        if isinstance(spec, str):
            model_name = spec
            analysis_type = 'duckreg'
        elif isinstance(spec, dict):
            model_name = spec.get('name') or spec.get('model_name')
            analysis_type = spec.get('analysis_type', 'duckreg')
            if not model_name:
                raise ValueError(f"Model spec dict must have a 'name' key: {spec}")
        else:
            raise TypeError(f"Expected str or dict, got {type(spec)}: {spec!r}")

        try:
            models.append(load_model_result(model_name, base_path, analysis_type))
        except FileNotFoundError:
            unavailable.append(f"{model_name}")
            models.append({
                '__missing__': True,
                'model_spec': model_name,
                'metadata': {'analysis_type': analysis_type},
            })
        except Exception as exc:
            unavailable.append(f"{model_name} (error: {exc})")
            models.append({
                '__missing__': True,
                'model_spec': model_name,
                'metadata': {'analysis_type': analysis_type},
            })

    if unavailable:
        print("Unavailable models:", ", ".join(unavailable))

    return models


# ---------------------------------------------------------------------------
# Metadata extraction helpers
# ---------------------------------------------------------------------------

def get_model_metadata(model: Dict[str, Any]) -> Dict[str, Any]:
    """Return the metadata block, checking both old and new key names."""
    return model.get('analysis_metadata', model.get('metadata', {}))


def get_model_version(model: Dict[str, Any]) -> str:
    """Extract the duckreg version string (``x.y.z``), or ``'0.0.0'`` if absent."""
    version_info = model.get('version_info', {})
    if version_info:
        v = version_info.get('duckreg_version', '')
        if v and re.match(r'^\d+\.\d+\.\d+$', str(v)):
            return str(v)

    coef_data = model.get('coefficients', {})
    if isinstance(coef_data, dict):
        v = coef_data.get('duckreg_version', '')
        if v and re.match(r'^\d+\.\d+\.\d+$', str(v)):
            return str(v)

    model_stats = model.get('model_statistics', {})
    estimator_type = model_stats.get('estimator_type', '')
    if estimator_type and re.match(r'^\d+\.\d+\.\d+$', str(estimator_type)):
        return str(estimator_type)

    return '0.0.0'


def get_model_date(model: Dict[str, Any]) -> str:
    """Extract the computation date as ``'YYYY-MM-DD'`` or ``'N/A'``."""

    def _parse(timestamp: str) -> str:
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d')
        except Exception:
            return timestamp.split('T')[0] if 'T' in timestamp else timestamp

    version_info = model.get('version_info', {})
    if version_info and 'computed_at' in version_info:
        return _parse(version_info['computed_at'])

    coef_data = model.get('coefficients', {})
    if isinstance(coef_data, dict) and 'computed_at' in coef_data:
        return _parse(coef_data['computed_at'])

    metadata = get_model_metadata(model)
    if 'timestamp' in metadata:
        return _parse(metadata['timestamp'])

    return 'N/A'


def is_2sls_model(model: Dict[str, Any]) -> bool:
    """Return *True* when the result contains a non-empty ``first_stage`` block."""
    return bool(model.get('first_stage'))


def get_coefficient_data(model: Dict[str, Any]) -> Dict[str, Any]:
    """Return the main coefficient block, handling OLS and 2SLS layouts."""
    if is_2sls_model(model):
        return model.get('coefficients', {})
    meta = get_model_metadata(model)
    if meta.get('analysis_type') == 'online_2sls':
        return model.get('second_stage', {}).get('coefficients', {})
    return model.get('coefficients', {})


# ---------------------------------------------------------------------------
# Keep old private-name aliases for backward compatibility
# ---------------------------------------------------------------------------
_find_latest_model_result = find_latest_model_result
_load_models_by_name = load_models_by_name
_get_model_metadata = get_model_metadata
_get_model_version = get_model_version
_get_model_date = get_model_date
_is_2sls_model = is_2sls_model
_get_coefficient_data = get_coefficient_data

"""
Configuration loading utilities for the GNT CLI.

Supports YAML, JSON, and Excel (delegated to gnt.analysis.workflow).
Environment variables in the form ``${VAR}`` or ``${VAR:-default}`` are
expanded inside YAML/JSON values.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Union

import yaml

logger = logging.getLogger(__name__)


def _deep_merge_dicts(base: Any, override: Any) -> Any:
    """Recursively merge dictionaries, preferring values from *override*."""
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def load_config_with_env_vars(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML/JSON/Excel config file with environment variable expansion.

    Parameters
    ----------
    config_path:
        Path to the configuration file.  Supported extensions:
        ``.yaml``, ``.yml``, ``.json``, ``.xlsx``, ``.xls``.

    Returns
    -------
    dict
        Parsed and env-expanded configuration.

    Raises
    ------
    FileNotFoundError
        When the file does not exist.
    ValueError
        When the file extension is not supported.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    # Excel (analysis) configs are handled by gnt.analysis.workflow
    if config_path.suffix.lower() in (".xlsx", ".xls"):
        from gnt.analysis.workflow import load_config as _load_analysis_config
        return _load_analysis_config(config_path)

    env_pattern = re.compile(r"\${([^}^{]+)}")

    def replace_env_vars(value: str) -> str:
        def _replace(match: re.Match) -> str:
            env_var = match.group(1)
            if ":-" in env_var:
                var_name, default_value = env_var.split(":-", 1)
                return os.environ.get(var_name, default_value)
            return os.environ.get(env_var, "")
        return env_pattern.sub(_replace, value)

    def process_item(item: Any) -> Any:
        if isinstance(item, dict):
            return {k: process_item(v) for k, v in item.items()}
        if isinstance(item, list):
            return [process_item(i) for i in item]
        if isinstance(item, str):
            expanded = replace_env_vars(item)
            if expanded != item:
                # Attempt numeric coercion after expansion
                try:
                    if expanded.isdigit() or (
                        expanded.startswith("-") and expanded[1:].isdigit()
                    ):
                        return int(expanded)
                    return float(expanded)
                except ValueError:
                    pass
            return expanded
        return item

    with open(config_path) as fh:
        if config_path.suffix.lower() == ".json":
            raw = json.load(fh)
        elif config_path.suffix.lower() in (".yaml", ".yml"):
            raw = yaml.safe_load(fh)
        else:
            raise ValueError(
                f"Unsupported configuration format: {config_path.suffix}"
            )

    if config_path.suffix.lower() in (".yaml", ".yml"):
        local_config_path = config_path.with_name(
            f"{config_path.stem}.local{config_path.suffix}"
        )
        if local_config_path.exists():
            with open(local_config_path) as fh:
                local_raw = yaml.safe_load(fh) or {}
            raw = _deep_merge_dicts(raw, local_raw)

    return process_item(raw)

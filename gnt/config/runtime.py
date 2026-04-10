"""
Runtime configuration helpers.

These helpers separate local project paths from optional remote/SSH
connection details while remaining backward compatible with the legacy
``hpc`` config block.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


def strip_remote_prefix(path: Optional[str]) -> Optional[str]:
    """Remove ``user@host:`` from an scp-style path."""
    if isinstance(path, str):
        return re.sub(r"^[^@]+@[^:]+:", "", path)
    return path


def get_paths_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return normalized local path settings."""
    config = config or {}
    paths = dict(config.get("paths", {}) or {})
    legacy = config.get("hpc", {}) or {}

    if not paths.get("data_root"):
        legacy_target = legacy.get("target")
        if legacy_target:
            paths["data_root"] = strip_remote_prefix(legacy_target)

    if not paths.get("local_index_dir"):
        local_index_dir = legacy.get("local_index_dir")
        if local_index_dir:
            paths["local_index_dir"] = local_index_dir

    return paths


def get_remote_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return normalized remote/SSH connection settings."""
    config = config or {}
    remote = dict(config.get("remote", {}) or {})
    legacy = config.get("hpc", {}) or {}

    if not remote.get("ssh_target"):
        ssh_target = legacy.get("target")
        if ssh_target:
            remote["ssh_target"] = ssh_target

    if not remote.get("key_file"):
        key_file = legacy.get("key_file")
        if key_file:
            remote["key_file"] = key_file

    return remote


def resolve_data_root(config: Optional[Dict[str, Any]]) -> Optional[str]:
    """Resolve the local project data root."""
    return get_paths_config(config).get("data_root")


def resolve_local_index_dir(config: Optional[Dict[str, Any]]) -> Optional[str]:
    """Resolve the local unified-index directory."""
    return get_paths_config(config).get("local_index_dir")


def resolve_ssh_target(config: Optional[Dict[str, Any]]) -> Optional[str]:
    """Resolve the remote SSH target used by download/index workflows."""
    return get_remote_config(config).get("ssh_target")


def resolve_remote_key_file(config: Optional[Dict[str, Any]]) -> Optional[str]:
    """Resolve the SSH key file used by remote transfer workflows."""
    return get_remote_config(config).get("key_file")


def get_legacy_hpc_compat_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Return a legacy-compatible ``hpc`` block for older internals.

    ``target`` is normalized to the local data root, which is what the
    preprocess and assemble codepaths actually need when reading/writing
    project files.
    """
    data_root = resolve_data_root(config)
    remote = get_remote_config(config)
    paths = get_paths_config(config)

    return {
        "target": data_root,
        "local_index_dir": paths.get("local_index_dir"),
        "key_file": remote.get("key_file"),
        "ssh_target": remote.get("ssh_target"),
    }

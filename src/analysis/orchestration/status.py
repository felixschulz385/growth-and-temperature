"""Lightweight SLURM status and manifest helpers for analysis jobs."""

from __future__ import annotations

import getpass
import json
import re
import subprocess
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..core.config import seconds_to_slurm_time

JOBS_DIR_NAME = "_jobs"
_SQUEUE_FORMAT = "%i|%T|%M|%l|%L|%S|%r|%q|%j|%P"
_SQUEUE_FIELDS = (
    "job_id",
    "state",
    "elapsed",
    "time_limit",
    "time_left",
    "start_time",
    "reason",
    "qos",
    "name",
    "partition",
)
_JOB_ID_RE = re.compile(r"slurm-(\d+)\.(?:log|err)$")
_OOM_PATTERNS = (
    re.compile(r"oom[_ -]?kill", re.IGNORECASE),
    re.compile(r"out of memory", re.IGNORECASE),
    re.compile(r"\bOOM\b", re.IGNORECASE),
)
_FAILURE_PATTERNS = (
    (re.compile(r"\bkilled\b", re.IGNORECASE), "killed"),
    (re.compile(r"\btraceback\b", re.IGNORECASE), "traceback"),
    (re.compile(r"\berror:\b", re.IGNORECASE), "error"),
    (re.compile(r"\bexception\b", re.IGNORECASE), "exception"),
    (re.compile(r"non-zero exit", re.IGNORECASE), "non-zero exit"),
)


def jobs_registry_dir(project_root: Path) -> Path:
    """Return the directory containing analysis job manifests."""
    return project_root / "log" / "analysis" / JOBS_DIR_NAME


def job_manifest_path(project_root: Path, job_id: str) -> Path:
    """Return the manifest path for *job_id*."""
    return jobs_registry_dir(project_root) / f"{job_id}.json"


def _serialise_model_spec(model_spec: Dict[str, Any], runtime_seconds: int) -> Dict[str, Any]:
    return {
        "model_name": model_spec["model_name"],
        "fixed_effects_label": model_spec["fixed_effects_label"],
        "resolution": model_spec["resolution"],
        "temporal_extent": model_spec["temporal_extent"],
        "spatial_extent": model_spec["spatial_extent"],
        "clustering": model_spec["clustering"],
        "variant_path": list(model_spec["variant_path"]),
        "runtime_seconds": runtime_seconds,
        "runtime_slurm": seconds_to_slurm_time(runtime_seconds),
    }


def build_job_manifest(
    *,
    job_id: str,
    job_label: str,
    project_root: Path,
    duckreg_version: str,
    table_model_pairs: List[Tuple[str, List[Dict[str, Any]]]],
    slurm_kwargs: Dict[str, Any],
    runtime_lookup: Dict[Tuple[str, ...], int],
) -> Dict[str, Any]:
    """Build the persisted manifest for a submitted analysis job."""
    batch_log_dir = project_root / "log" / "analysis" / f"table-{job_label}" / duckreg_version
    tables: List[Dict[str, Any]] = []
    global_index = 0

    for table_index, (label, model_specs) in enumerate(table_model_pairs, 1):
        table_models: List[Dict[str, Any]] = []
        for model_index, model_spec in enumerate(model_specs, 1):
            global_index += 1
            key = tuple(model_spec["variant_path"])
            runtime_seconds = runtime_lookup[key]
            table_models.append({
                **_serialise_model_spec(model_spec, runtime_seconds),
                "table_label": label,
                "table_index": table_index,
                "table_model_index": model_index,
                "global_model_index": global_index,
            })
        tables.append({
            "label": label,
            "table_index": table_index,
            "models": table_models,
        })

    return {
        "job_id": str(job_id),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "job_label": job_label,
        "duckreg_version": duckreg_version,
        "slurm": {
            "mem": slurm_kwargs.get("mem"),
            "time": slurm_kwargs.get("time"),
            "qos": slurm_kwargs.get("qos"),
            "partition": slurm_kwargs.get("partition"),
            "cpus_per_task": slurm_kwargs.get("cpus_per_task"),
        },
        "batch_log_path": str(batch_log_dir / f"slurm-{job_id}.log"),
        "batch_err_path": str(batch_log_dir / f"slurm-{job_id}.err"),
        "tables": tables,
    }


def write_job_manifest(
    *,
    project_root: Path,
    job_id: str,
    job_label: str,
    duckreg_version: str,
    table_model_pairs: List[Tuple[str, List[Dict[str, Any]]]],
    slurm_kwargs: Dict[str, Any],
    runtime_lookup: Dict[Tuple[str, ...], int],
) -> Path:
    """Persist one JSON manifest for a submitted analysis job."""
    manifest_dir = jobs_registry_dir(project_root)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_job_manifest(
        job_id=str(job_id),
        job_label=job_label,
        project_root=project_root,
        duckreg_version=duckreg_version,
        table_model_pairs=table_model_pairs,
        slurm_kwargs=slurm_kwargs,
        runtime_lookup=runtime_lookup,
    )
    path = job_manifest_path(project_root, str(job_id))
    path.write_text(json.dumps(manifest, indent=2))
    return path


def load_job_manifest(project_root: Path, job_id: str) -> Optional[Dict[str, Any]]:
    """Load the manifest for *job_id* if it exists."""
    path = job_manifest_path(project_root, str(job_id))
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def read_active_slurm_jobs(
    user: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, str]], Optional[str]]:
    """Return active ``squeue`` jobs keyed by job ID.

    When ``squeue`` is unavailable or returns an error, the function degrades
    gracefully and returns an empty mapping with a warning message.
    """
    if user is None:
        user = getpass.getuser()

    cmd = ["squeue", "--noheader", "--format", _SQUEUE_FORMAT, "--user", user]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return {}, "squeue not available"

    if result.returncode != 0:
        stderr = result.stderr.strip() or "squeue failed"
        return {}, stderr

    jobs: Dict[str, Dict[str, str]] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) != len(_SQUEUE_FIELDS):
            continue
        job = dict(zip(_SQUEUE_FIELDS, parts))
        jobs[job["job_id"]] = job
    return jobs, None


def flatten_manifest_models(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return manifest models in execution order."""
    flat: List[Dict[str, Any]] = []
    for table in manifest.get("tables", []):
        for model in table.get("models", []):
            flat.append(model)
    return sorted(flat, key=lambda row: int(row["global_model_index"]))


def tail_lines(path: Path, max_lines: int = 80) -> List[str]:
    """Return the last *max_lines* from *path* without reading more than needed."""
    if not path.exists():
        return []

    buffer: deque[str] = deque(maxlen=max_lines)
    with open(path, errors="replace") as fh:
        for line in fh:
            buffer.append(line.rstrip("\n"))
    return list(buffer)


def parse_batch_progress(job_log_path: Path, total_models: int) -> Dict[str, Any]:
    """Parse the batch log and return coarse sequential progress metadata."""
    lines = tail_lines(job_log_path, max_lines=200)
    completed_models = 0
    current_table: Optional[str] = None
    for line in lines:
        if "Starting table " in line:
            current_table = line.split("Starting table ", 1)[1].split(" (", 1)[0].strip()
        if "Completed:" in line:
            completed_models += 1

    all_done = any("All models completed successfully!" in line for line in lines)
    current_index: Optional[int] = None
    if not all_done and completed_models < total_models:
        current_index = completed_models + 1

    return {
        "completed_models": min(completed_models, total_models),
        "current_model_index": current_index,
        "current_table": current_table,
        "all_done": all_done,
        "lines": lines,
    }


def latest_model_log_paths(
    project_root: Path,
    model_spec: Dict[str, Any],
) -> Tuple[Optional[Path], Optional[Path]]:
    """Return the latest ``.log`` and ``.err`` files for *model_spec*."""
    model_log_root = project_root / "log" / "analysis" / Path(*model_spec["variant_path"])
    if not model_log_root.exists():
        return None, None

    log_candidates = list(model_log_root.glob("*/slurm-*.log"))
    err_candidates = list(model_log_root.glob("*/slurm-*.err"))
    return _latest_job_file(log_candidates), _latest_job_file(err_candidates)


def _latest_job_file(paths: Iterable[Path]) -> Optional[Path]:
    candidates = list(paths)
    if not candidates:
        return None
    return max(candidates, key=lambda path: (_job_id_for_path(path), str(path)))


def _job_id_for_path(path: Path) -> int:
    match = _JOB_ID_RE.search(path.name)
    return int(match.group(1)) if match else -1


def detect_failure_hint(paths: Iterable[Optional[Path]]) -> Optional[str]:
    """Return a short failure label inferred from recent log/err tails."""
    for path in paths:
        if path is None or not path.exists():
            continue
        joined = "\n".join(tail_lines(path, max_lines=80))
        if not joined.strip():
            continue
        for pattern in _OOM_PATTERNS:
            if pattern.search(joined):
                return "oom"
        for pattern, label in _FAILURE_PATTERNS:
            if pattern.search(joined):
                return label
        if path.suffix == ".err":
            return "stderr output"
    return None

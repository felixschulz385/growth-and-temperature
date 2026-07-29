"""Terminal summary for analysis models and SLURM jobs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.config import AnalysisConfig, PROJECT_ROOT
from ..io.results import get_model_result_status
from ..orchestration.status import (
    detect_failure_hint,
    flatten_manifest_models,
    latest_model_log_paths,
    load_job_manifest,
    parse_batch_progress,
    read_active_slurm_jobs,
)

_STATUS_DISPLAY = {
    "completed": "done",
    "completed_in_job": "done*",
    "queued": "queued",
    "running": "running",
    "failed": "failed",
    "missing": "missing",
}
_COUNTABLE_STATUS = {
    "completed_in_job": "completed",
}


@dataclass
class ModelStatusRecord:
    """Normalized summary row for one variant-aware model spec."""

    table_name: str
    model_name: str
    fixed_effects: str
    resolution: str
    temporal_extent: str
    spatial_extent: str
    clustering: str
    status: str
    count_status: str
    last_run: str
    version: str
    job_id: Optional[str]
    failure_hint: Optional[str]


def summarize_tables(config: AnalysisConfig) -> None:
    """Print a status summary of every configured analysis table."""
    active_jobs, slurm_warning = _collect_active_jobs()
    counts = Counter()
    table_rows: List[Tuple[str, Counter, List[ModelStatusRecord]]] = []

    print(f"\n{'=' * 100}")
    print("  Analysis Summary")
    print(f"  Excel   : {config.excel_path}")
    print(f"  Results : {config.base_path}")
    if slurm_warning:
        print(f"  SLURM   : {slurm_warning}")
    else:
        print(f"  SLURM   : {len(active_jobs)} active analysis job(s)")
    print(f"{'=' * 100}")

    for table_name in _table_names_in_sheet_order(config):
        records = _collect_table_records(config, table_name, active_jobs)
        table_counts = Counter(record.count_status for record in records)
        counts.update(table_counts)
        table_rows.append((table_name, table_counts, records))

    print(
        "  Counts  : "
        f"completed={counts['completed']}  "
        f"queued={counts['queued']}  "
        f"running={counts['running']}  "
        f"failed={counts['failed']}  "
        f"missing={counts['missing']}"
    )

    print("\n  Active SLURM jobs")
    print("  " + "-" * 96)
    if active_jobs:
        for job in active_jobs.values():
            progress = job.get("progress", {})
            completed = progress.get("completed_models", 0)
            total = job.get("total_models", 0)
            current = job.get("current_model_label") or "—"
            detail = job.get("start_time") if job["state"] == "RUNNING" else job.get("reason")
            print(
                f"  {job['job_id']:>9}  {job['state']:<10}  "
                f"time={job.get('elapsed', '—')}/{job.get('time_limit', '—')}  "
                f"models={completed}/{total}  current={current}"
            )
            if detail and detail not in {"N/A", "None", "(null)"}:
                print(f"             detail={detail}")
    else:
        print("  none")

    model_col = 24
    fe_col = 10
    res_col = 7
    temporal_col = 11
    spatial_col = 14
    cluster_col = 9
    for table_name, table_counts, records in table_rows:
        print(
            f"\n  Table: {table_name}  "
            f"(completed={table_counts['completed']} queued={table_counts['queued']} "
            f"running={table_counts['running']} failed={table_counts['failed']} "
            f"missing={table_counts['missing']})"
        )
        print(
            f"  {'Status':<9}  {'Model':<{model_col}}  {'FE':<{fe_col}}  "
            f"{'Res':<{res_col}}  {'Temporal':<{temporal_col}}  {'Spatial':<{spatial_col}}  "
            f"{'Cluster':<{cluster_col}}  {'Last Run':<10}  {'Version':<7}  Job"
        )
        print(
            f"  {'-' * 9}  {'-' * model_col}  {'-' * fe_col}  {'-' * res_col}  "
            f"{'-' * temporal_col}  {'-' * spatial_col}  {'-' * cluster_col}  "
            f"{'-' * 10}  {'-' * 7}  {'-' * 10}"
        )
        for record in records:
            job_id = record.job_id or "—"
            suffix = f" ({record.failure_hint})" if record.failure_hint else ""
            print(
                f"  {_STATUS_DISPLAY[record.status]:<9}  "
                f"{record.model_name:<{model_col}}  "
                f"{record.fixed_effects:<{fe_col}}  "
                f"{record.resolution:<{res_col}}  "
                f"{record.temporal_extent:<{temporal_col}}  "
                f"{record.spatial_extent:<{spatial_col}}  "
                f"{record.clustering:<{cluster_col}}  "
                f"{record.last_run:<10}  "
                f"{record.version:<7}  "
                f"{job_id}{suffix}"
            )

    print()


def _collect_active_jobs() -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    jobs, warning = read_active_slurm_jobs()
    if not jobs:
        return {}, warning

    active_jobs: Dict[str, Dict[str, Any]] = {}
    for job_id, job in jobs.items():
        manifest = load_job_manifest(PROJECT_ROOT, job_id)
        if manifest is None:
            continue

        manifest_models = flatten_manifest_models(manifest)
        progress = parse_batch_progress(Path(manifest["batch_log_path"]), len(manifest_models))
        current_idx = progress.get("current_model_index")
        model_states: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        for model in manifest_models:
            variant_key = tuple(model["variant_path"])
            status = _status_for_manifest_model(job["state"], model, current_idx)
            model_states[variant_key] = {
                "status": status,
                "job_id": job_id,
                "global_model_index": int(model["global_model_index"]),
            }

        current_label = None
        if current_idx is not None:
            for model in manifest_models:
                if int(model["global_model_index"]) == current_idx:
                    current_label = (
                        f"{model['model_name']} "
                        f"[{model['fixed_effects_label']}/{model['resolution']}/"
                        f"{model['temporal_extent']}/{model['spatial_extent']}/"
                        f"{model['clustering']}]"
                    )
                    break

        active_jobs[job_id] = {
            **job,
            "manifest": manifest,
            "progress": progress,
            "total_models": len(manifest_models),
            "current_model_label": current_label,
            "model_states": model_states,
        }

    return active_jobs, warning


def _status_for_manifest_model(
    job_state: str,
    model: Dict[str, Any],
    current_idx: Optional[int],
) -> str:
    if job_state == "RUNNING":
        model_idx = int(model["global_model_index"])
        if current_idx is None:
            return "queued"
        if model_idx < current_idx:
            return "completed_in_job"
        if model_idx == current_idx:
            return "running"
        return "queued"
    return "queued"


def _collect_table_records(
    config: AnalysisConfig,
    table_name: str,
    active_jobs: Dict[str, Dict[str, Any]],
) -> List[ModelStatusRecord]:
    active_index: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for job in active_jobs.values():
        active_index.update(job["model_states"])

    records: List[ModelStatusRecord] = []
    for model_spec in _table_model_specs_in_sheet_order(config, table_name):
        variant_key = tuple(model_spec["variant_path"])
        result_status = get_model_result_status(model_spec, config.base_path)
        active_state = active_index.get(variant_key)

        failure_hint: Optional[str] = None
        status = "missing"
        job_id: Optional[str] = None
        if active_state is not None:
            status = active_state["status"]
            job_id = active_state["job_id"]
        elif result_status["exists"]:
            status = "completed"
        else:
            log_path, err_path = latest_model_log_paths(PROJECT_ROOT, model_spec)
            failure_hint = detect_failure_hint((err_path, log_path))
            if failure_hint:
                status = "failed"

        count_status = _COUNTABLE_STATUS.get(status, status)
        records.append(
            ModelStatusRecord(
                table_name=table_name,
                model_name=model_spec["model_name"],
                fixed_effects=model_spec["fixed_effects_label"],
                resolution=model_spec["resolution"],
                temporal_extent=model_spec["temporal_extent"],
                spatial_extent=model_spec["spatial_extent"],
                clustering=model_spec["clustering"],
                status=status,
                count_status=count_status,
                last_run=result_status["date"],
                version=result_status["version"],
                job_id=job_id,
                failure_hint=failure_hint,
            )
        )
    return records


def _table_names_in_sheet_order(config: AnalysisConfig) -> List[str]:
    """Return table names in the literal workbook row order."""
    rows = config.df_models_in_tables
    seen: set[str] = set()
    names: List[str] = []
    for value in rows["table_name"].tolist():
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan" or text in seen:
            continue
        seen.add(text)
        names.append(text)
    return names


def _table_model_specs_in_sheet_order(
    config: AnalysisConfig,
    table_name: str,
) -> List[Dict[str, Any]]:
    """Return variant-aware model specs in the literal workbook row order."""
    rows = config.df_models_in_tables
    selected = rows[rows["table_name"].astype(str).str.strip() == table_name]

    specs: List[Dict[str, Any]] = []
    for _, row in selected.iterrows():
        model_name = str(row["model_name"]).strip()
        if not model_name or model_name.lower() == "nan":
            continue
        specs.append(
            config.get_model_spec(
                model_name,
                fixed_effects=row.get("Fixed Effects"),
                resolution=row.get("Resolution"),
                clustering=row.get("Clustering"),
                temporal_extent=row.get("Temporal Extent"),
                spatial_extent=row.get("Spatial Extent"),
            )
        )
    return specs

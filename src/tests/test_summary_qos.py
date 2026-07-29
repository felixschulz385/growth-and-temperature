from __future__ import annotations

import json
from pathlib import Path

from src.analysis.core.runtime import (
    ONE_DAY_SECONDS,
    ONE_WEEK_SECONDS,
    recommended_cores_for_model_specs,
    recommended_cores_for_resolution,
    SIX_HOURS_SECONDS,
    THIRTY_MINUTES_SECONDS,
    TWO_WEEKS_SECONDS,
    recommend_slurm_qos,
)
from src.analysis.orchestration.status import write_job_manifest
from src.analysis.rendering import summary


class FakeConfig:
    def __init__(self, base_path: Path, specs: list[dict[str, str]], runtimes: dict[tuple[str, ...], int]):
        self.base_path = base_path
        self.excel_path = Path("analysis.xlsx")
        self._specs = specs
        self._runtimes = runtimes

    def get_all_table_names(self) -> list[str]:
        return ["table1"]

    def get_table_model_specs(self, table_name: str) -> list[dict[str, str]]:
        assert table_name == "table1"
        return self._specs

    def get_model_runtime_seconds_for_spec(self, spec: dict[str, str]) -> int:
        return self._runtimes[tuple(spec["variant_path"])]


def _spec(
    model_name: str,
    fe: str,
    resolution: str = "1km",
    temporal: str = "2000-2020",
    spatial: str = "full_sample",
    clustering: str = "ADM2",
) -> dict[str, str]:
    return {
        "model_name": model_name,
        "fixed_effects_label": fe,
        "resolution": resolution,
        "temporal_extent": temporal,
        "spatial_extent": spatial,
        "clustering": clustering,
        "variant_path": [model_name, fe, resolution, temporal, spatial, clustering],
    }


def _write_result(base_path: Path, spec: dict[str, str], *, version: str = "0.4.3") -> None:
    result_dir = base_path / "duckreg" / Path(*spec["variant_path"])
    result_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "analysis_metadata": {"timestamp": "2026-06-20T10:00:00+00:00"},
        "version_info": {
            "duckreg_version": version,
            "computed_at": "2026-06-20T10:00:00+00:00",
        },
    }
    (result_dir / "results_20260620_100000.json").write_text(json.dumps(result))


def _write_failure_log(project_root: Path, spec: dict[str, str]) -> None:
    log_dir = project_root / "log" / "analysis" / Path(*spec["variant_path"]) / "0.4.3"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "slurm-999.err").write_text("Detected 1 oom_kill event in StepId=999.batch.\n")


def test_recommend_slurm_qos_boundaries() -> None:
    assert recommend_slurm_qos(THIRTY_MINUTES_SECONDS) == "30min"
    assert recommend_slurm_qos(THIRTY_MINUTES_SECONDS + 1) == "6hours"
    assert recommend_slurm_qos(SIX_HOURS_SECONDS) == "6hours"
    assert recommend_slurm_qos(SIX_HOURS_SECONDS + 1) == "1day"
    assert recommend_slurm_qos(ONE_DAY_SECONDS) == "1day"
    assert recommend_slurm_qos(ONE_DAY_SECONDS + 1) == "1week"
    assert recommend_slurm_qos(ONE_WEEK_SECONDS) == "1week"
    assert recommend_slurm_qos(ONE_WEEK_SECONDS + 1) == "2weeks"
    assert recommend_slurm_qos(TWO_WEEKS_SECONDS) == "2weeks"


def test_recommend_slurm_qos_rejects_over_two_weeks() -> None:
    try:
        recommend_slurm_qos(TWO_WEEKS_SECONDS + 1)
    except ValueError as exc:
        assert "2weeks" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_recommended_cores_for_resolution() -> None:
    assert recommended_cores_for_resolution("500m") == 16
    assert recommended_cores_for_resolution("1km") == 8
    assert recommended_cores_for_resolution("5km") == 4
    assert recommended_cores_for_resolution("ADM2") == 4


def test_recommended_cores_for_model_specs_uses_highest_requirement() -> None:
    specs = [
        _spec("coarse_model", "NO", resolution="5km"),
        _spec("medium_model", "PX", resolution="1km"),
        _spec("fine_model", "ADM2", resolution="500m"),
    ]
    assert recommended_cores_for_model_specs(specs) == 16


def test_write_job_manifest_persists_required_fields(tmp_path: Path) -> None:
    project_root = tmp_path
    spec = _spec("queued_model", "PX")
    runtime_lookup = {tuple(spec["variant_path"]): 3600}

    manifest_path = write_job_manifest(
        project_root=project_root,
        job_id="12345",
        job_label="example_job",
        duckreg_version="0.4.3",
        table_model_pairs=[("table1", [spec])],
        slurm_kwargs={
            "mem": "128GB",
            "time": "01:00:00",
            "qos": "6hours",
            "partition": "scicore",
            "cpus_per_task": 8,
        },
        runtime_lookup=runtime_lookup,
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["job_id"] == "12345"
    assert manifest["job_label"] == "example_job"
    assert manifest["slurm"]["qos"] == "6hours"
    assert manifest["slurm"]["time"] == "01:00:00"
    assert manifest["tables"][0]["models"][0]["variant_path"] == spec["variant_path"]


def test_collect_table_records_merges_results_queue_and_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    completed = _spec("completed_model", "NO")
    queued = _spec("queued_model", "PX")
    failed = _spec("failed_model", "ADM2")
    missing = _spec("missing_model", "PX+CY")
    specs = [completed, queued, failed, missing]
    runtimes = {
        tuple(completed["variant_path"]): 1200,
        tuple(queued["variant_path"]): 2400,
        tuple(failed["variant_path"]): 3600,
        tuple(missing["variant_path"]): 4800,
    }
    config = FakeConfig(tmp_path / "output" / "analysis", specs, runtimes)

    _write_result(config.base_path, completed)
    _write_failure_log(tmp_path, failed)
    write_job_manifest(
        project_root=tmp_path,
        job_id="12345",
        job_label="queued_job",
        duckreg_version="0.4.3",
        table_model_pairs=[("table1", [queued])],
        slurm_kwargs={
            "mem": "128GB",
            "time": "00:40:00",
            "qos": "6hours",
            "partition": "scicore",
            "cpus_per_task": 8,
        },
        runtime_lookup={tuple(queued["variant_path"]): runtimes[tuple(queued["variant_path"])]},
    )

    monkeypatch.setattr(summary, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        summary,
        "read_active_slurm_jobs",
        lambda: (
            {
                "12345": {
                    "job_id": "12345",
                    "state": "PENDING",
                    "elapsed": "0:00",
                    "time_limit": "0:40:00",
                    "time_left": "0:40:00",
                    "start_time": "N/A",
                    "reason": "Priority",
                    "qos": "6hours",
                    "name": "queued_job",
                    "partition": "scicore",
                },
                "54321": {
                    "job_id": "54321",
                    "state": "RUNNING",
                    "elapsed": "0:10",
                    "time_limit": "1:00:00",
                    "time_left": "0:50:00",
                    "start_time": "2026-06-20T10:00:00",
                    "reason": "None",
                    "qos": "1day",
                    "name": "other_project",
                    "partition": "scicore",
                }
            },
            None,
        ),
    )

    active_jobs, warning = summary._collect_active_jobs()
    assert warning is None
    assert list(active_jobs) == ["12345"]

    records = summary._collect_table_records(config, "table1", active_jobs)
    statuses = {record.model_name: record.status for record in records}

    assert statuses["completed_model"] == "completed"
    assert statuses["queued_model"] == "queued"
    assert statuses["failed_model"] == "failed"
    assert statuses["missing_model"] == "missing"

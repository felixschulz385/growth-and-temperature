"""Every orchestration/configs/slurm_jobs.yaml entry must resolve against
the source registry and orchestration/configs/data.yaml.

docs/design/09-integrated-pipeline.md §9/§11: the permanent regression test
for the exact bug class that motivated writing generate_slurm_scripts.py in
the first place (broken `--source` names that only failed at SLURM-job
runtime). `data run --slurm` now reads this file at submission time instead
of a generator reading it ahead of time, so this test is the only guard left
against a stale/broken entry.
"""

from pathlib import Path

import pytest
import yaml

from src.data.sources import registry
from src.data.sources.steps import PipelineStep

REPO_ROOT = Path(__file__).resolve().parents[2]
SLURM_JOBS_FILE = REPO_ROOT / "orchestration" / "configs" / "slurm_jobs.yaml"
DATA_YAML_FILE = REPO_ROOT / "orchestration" / "configs" / "data.yaml"


def _jobs():
    with open(SLURM_JOBS_FILE) as f:
        return yaml.safe_load(f)["jobs"]


def _data_yaml_source_keys():
    with open(DATA_YAML_FILE) as f:
        data = yaml.safe_load(f) or {}
    return set((data.get("sources") or {}).keys())


def test_jobs_have_no_duplicate_names():
    names = [job["name"] for job in _jobs()]
    assert len(names) == len(set(names))


def test_cluster_block_has_required_paths():
    with open(SLURM_JOBS_FILE) as f:
        cluster = yaml.safe_load(f)["cluster"]
    for key in ("conda_hook", "python_bin", "project_root", "scratch_prefix"):
        assert cluster.get(key), f"cluster.{key} missing/empty"


@pytest.mark.parametrize("job", _jobs(), ids=lambda j: j["name"])
def test_job_source_resolves_in_registry(job):
    registry.resolve(job["source"])


@pytest.mark.parametrize("job", _jobs(), ids=lambda j: j["name"])
def test_job_step_is_declared_by_its_source(job):
    spec = registry.resolve(job["source"])
    step = PipelineStep(job["step"])
    assert step in spec.steps, f"{job['name']}: source '{job['source']}' does not declare step '{job['step']}'"


@pytest.mark.parametrize("job", _jobs(), ids=lambda j: j["name"])
def test_job_source_is_a_data_yaml_key(job):
    assert job["source"] in _data_yaml_source_keys(), f"{job['name']}: source '{job['source']}' missing from data.yaml"


@pytest.mark.parametrize("job", _jobs(), ids=lambda j: j["name"])
def test_depends_on_references_known_jobs(job):
    names = {j["name"] for j in _jobs()}
    for dep_name in job.get("depends_on", []):
        assert dep_name in names, f"{job['name']}: depends_on references unknown job '{dep_name}'"

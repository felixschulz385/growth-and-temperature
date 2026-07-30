"""Every orchestration/slurm/jobs.yaml entry must resolve against the source
registry and orchestration/configs/data.yaml.

docs/design/09-integrated-pipeline.md §9/§11: the permanent regression test
for the exact bug class that motivated writing generate_slurm_scripts.py in
the first place -- broken `--source` names that only failed at SLURM-job
runtime (e.g. jobs.yaml's old `esacci-preprocess-spatial` job actually ran
`--source acag`; `viirs_annual` and `plad`(old id) didn't resolve to any
`data.yaml` key at all). generate_slurm_scripts.py's own `--check` exercises
the same validation at generation time; this test exercises it in the normal
test suite so CI catches drift without anyone remembering to run the
generator's `--check` mode.
"""

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
JOBS_FILE = REPO_ROOT / "orchestration" / "slurm" / "jobs.yaml"
DATA_YAML_FILE = REPO_ROOT / "orchestration" / "configs" / "data.yaml"

sys.path.insert(0, str(REPO_ROOT / "orchestration" / "slurm"))
import generate_slurm_scripts as gen  # noqa: E402


def _jobs():
    with open(JOBS_FILE) as f:
        return yaml.safe_load(f)["jobs"]


def test_jobs_yaml_has_no_duplicate_names():
    names = [job["name"] for job in _jobs()]
    assert len(names) == len(set(names))


def test_every_job_is_valid():
    errors = gen.validate_jobs(_jobs())
    assert errors == []


def test_every_job_source_is_a_data_yaml_key():
    data_yaml_sources = gen._load_data_yaml_source_keys()
    for job in _jobs():
        assert job["source"] in data_yaml_sources, f"{job['name']}: source '{job['source']}' missing from data.yaml"


@pytest.mark.parametrize("job", _jobs(), ids=lambda j: j["name"])
def test_job_host_is_slurm_or_egress(job):
    assert job.get("host", "slurm") in ("slurm", "egress")


def test_egress_jobs_land_in_scripts_dir_not_slurm_dir():
    for job in _jobs():
        if job.get("host") == "egress":
            _, target = gen.render_job(job)
            assert target.parent == gen.SCRIPTS_DIR


def test_generator_check_reports_no_drift():
    """The on-disk *.sh files must match what the generator would produce --
    i.e. nobody hand-edited a generated script since the last regeneration."""
    for job in _jobs():
        rendered, target = gen.render_job(job)
        assert target.exists(), f"{job['name']}: generated script missing -- run generate_slurm_scripts.py"
        assert target.read_text() == rendered, f"{job['name']}: on-disk script has drifted from jobs.yaml -- regenerate"

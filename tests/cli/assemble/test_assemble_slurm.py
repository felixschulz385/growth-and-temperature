"""`assemble create/update --slurm` rendering/submission logic
(src/cli/assemble/slurm.py) against the real
orchestration/configs/slurm_jobs.yaml -- no real `sbatch` needed (only
exercises render_wrap_command / sbatch_argv / submit --dry-run).
"""

import argparse

import pytest

from src.cli.assemble.slurm import (
    find_assembly_job,
    load_slurm_config,
    render_wrap_command,
    sbatch_argv,
    submit,
)


def _cluster():
    return load_slurm_config()["cluster"]


def _assembly_jobs():
    return load_slurm_config().get("assembly_jobs", [])


def _args(**overrides):
    defaults = dict(
        config="orchestration/configs/data.yaml",
        grid="10km",
        shake="quad",
        datasource=None,
        overwrite=None,
        dashboard_port=8787,
        debug=False,
        dry_run=True,
        slurm_time=None,
        slurm_mem=None,
        slurm_cpus=None,
        slurm_qos=None,
        slurm_partition=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_assembly_jobs_has_a_10km_entry():
    job = find_assembly_job(_assembly_jobs(), "10km")
    assert job is not None and job["grid"] == "10km"


def test_render_wrap_command_create_uses_gnt_and_assemble_create():
    job = find_assembly_job(_assembly_jobs(), "10km")
    wrap = render_wrap_command(job, _cluster(), _args())
    assert "conda activate gnt" in wrap
    assert "run.py assemble create" in wrap
    assert "--grid 10km" in wrap
    assert "--shake quad" in wrap
    assert "--threads $SLURM_CPUS_PER_TASK" in wrap
    assert '--memory-limit "${MEMORY_LIMIT_GB}GB"' in wrap
    assert "assemble_${SLURM_JOB_ID}" in wrap


def test_render_wrap_command_update_passes_datasource():
    job = find_assembly_job(_assembly_jobs(), "1km")
    wrap = render_wrap_command(job, _cluster(), _args(grid="1km", shake="none", datasource="eog_viirs"))
    assert "run.py assemble update" in wrap
    assert "--datasource eog_viirs" in wrap


def test_render_wrap_command_create_no_overwrite_flag():
    job = find_assembly_job(_assembly_jobs(), "10km")
    wrap = render_wrap_command(job, _cluster(), _args(overwrite=False))
    assert "--no-overwrite" in wrap


def test_sbatch_argv_log_dir_is_per_grid():
    job = find_assembly_job(_assembly_jobs(), "10km")
    argv, log_dir = sbatch_argv(job, _cluster(), "WRAP", job_name="assemble-create-10km")
    assert log_dir.endswith("/log/assemble/10km")
    assert "--job-name=assemble-create-10km" in argv
    assert f"--time={job['time']}" in argv
    assert argv[-1] == "--wrap=WRAP"


def test_submit_dry_run_prints_one_sbatch_and_notes_variants(capsys):
    submit(_args(grid="10km", shake="quad", dry_run=True))
    out = capsys.readouterr().out
    assert out.count("sbatch --parsable") == 1
    assert "conda activate gnt" in out
    assert "shake variants ['base', 's0', 's1', 's2']" in out


def test_submit_grid_without_entry_and_without_resources_errors(capsys):
    # "2km" is a valid --grid label but has no assembly_jobs entry.
    assert find_assembly_job(_assembly_jobs(), "2km") is None
    with pytest.raises(SystemExit):
        submit(_args(grid="2km", shake="none"))
    assert "No assembly_jobs entry for grid='2km'" in capsys.readouterr().err


def test_submit_grid_without_entry_but_with_resources_submits(capsys):
    submit(_args(
        grid="2km", shake="none",
        slurm_time="06:00:00", slurm_mem="64G", slurm_cpus=4, slurm_qos="6hours",
        slurm_partition="scicore",
    ))
    out = capsys.readouterr().out
    assert "sbatch --parsable" in out
    assert "--time=06:00:00" in out

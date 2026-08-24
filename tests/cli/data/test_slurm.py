"""`data run --slurm`'s DAG-building/dependency/rendering logic
(src/cli/data/slurm.py) against the real orchestration/configs/slurm_jobs.yaml
(docs/design/10-fetch-ledger.md §6) -- no real `sbatch` needed (only exercises
`build_chain`/`job_dependencies`/`render_wrap_command`/`sbatch_argv`/
`--dry-run`, which never shells out).
"""

import argparse

from src.cli.data.slurm import (
    build_chain,
    job_dependencies,
    load_slurm_config,
    render_wrap_command,
    sbatch_argv,
)
from src.cli.main import main as cli_main


def _jobs():
    return load_slurm_config()["jobs"]


def _cluster():
    return load_slurm_config()["cluster"]


def test_acag_chain_is_a_single_prepare_job():
    # acag's PREPARE does the tiled reprojection work directly -- there is
    # no separate GRID job to chain after it.
    chain = build_chain("acag", _jobs())
    assert [j["name"] for j in chain] == ["acag-prepare"]


def test_plad_chain_includes_gadm_prepare_prerequisite_first():
    chain = build_chain("plad", _jobs())
    assert [j["name"] for j in chain] == ["gadm-prepare", "plad-prepare"]


def test_snl_mining_chain_includes_gadm_and_commodity_prices_prerequisites():
    chain = build_chain("snl_mining", _jobs())
    assert [j["name"] for j in chain] == [
        "gadm-prepare", "commodity_prices-prepare", "snl_mining-prepare",
    ]


def test_country_classifications_chain_includes_gadm_prepare_prerequisite():
    chain = build_chain("country_classifications", _jobs())
    assert [j["name"] for j in chain] == ["gadm-prepare", "country_classifications-prepare"]


def test_modis_from_step_prepare_yields_just_prepare():
    # modis's FETCH step has no slurm_jobs.yaml entry at all (it's never a
    # SLURM job -- see orchestration/scripts/modis-fetch.sh).
    chain = build_chain("modis", _jobs(), from_step="prepare")
    assert [j["name"] for j in chain] == ["modis-prepare"]


def test_modis_default_from_step_also_only_yields_prepare():
    chain = build_chain("modis", _jobs())
    assert [j["name"] for j in chain] == ["modis-prepare"]


def test_gadm_chain_has_no_self_requires_duplication():
    chain = build_chain("gadm", _jobs())
    assert [j["name"] for j in chain] == ["gadm-prepare"]


def test_job_dependencies_scoped_to_each_job_own_step():
    jobs = _jobs()
    chain = build_chain("snl_mining", jobs)
    job_ids = {}
    for job in chain:
        job_dependencies(job, chain, job_ids)
        job_ids[job["name"]] = f"id-{job['name']}"

    known_ids = {
        "gadm-prepare": "id-gadm-prepare",
        "commodity_prices-prepare": "id-commodity_prices-prepare",
    }
    prepare_deps = job_dependencies(chain[2], chain, known_ids)
    assert prepare_deps == ["id-gadm-prepare", "id-commodity_prices-prepare"]


def test_depends_on_escape_hatch_is_additive():
    job = {"name": "x-grid", "source": "acag", "step": "grid", "depends_on": ["some-other-job"]}
    job_ids = {"some-other-job": "id-other"}
    deps = job_dependencies(job, [job], job_ids)
    assert deps == ["id-other"]


def _args(**overrides):
    defaults = dict(
        config="orchestration/configs/data.yaml", override=False, debug=False,
        slurm_time=None, slurm_mem=None, slurm_cpus=None, slurm_qos=None, slurm_partition=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_render_wrap_command_includes_source_and_step():
    job = next(j for j in _jobs() if j["name"] == "acag-prepare")
    wrap_cmd = render_wrap_command(job, _cluster(), _args())
    assert "--source acag" in wrap_cmd
    assert "--step prepare" in wrap_cmd
    assert "conda activate src" in wrap_cmd


def test_render_wrap_command_skips_dask_flags_for_simple_jobs():
    job = next(j for j in _jobs() if j["name"] == "plad-prepare")
    assert job.get("simple") is True
    wrap_cmd = render_wrap_command(job, _cluster(), _args())
    assert "--dask-threads" not in wrap_cmd
    assert "MEMORY_LIMIT_GB" not in wrap_cmd


def test_sbatch_argv_is_well_formed():
    job = next(j for j in _jobs() if j["name"] == "acag-prepare")
    wrap_cmd = render_wrap_command(job, _cluster(), _args())
    argv, log_dir = sbatch_argv(job, _cluster(), wrap_cmd, deps=["123"])
    assert argv[0] == "sbatch"
    assert "--job-name=acag-prepare" in argv
    assert "--dependency=afterok:123" in argv
    assert argv[-1] == f"--wrap={wrap_cmd}"
    assert log_dir.endswith("/log/preprocess/acag")


def test_unknown_source_returns_nonzero(capsys):
    exit_code = cli_main(["data", "run", "--source", "not_a_real_source", "--step", "prepare", "--slurm", "--chain", "--dry-run"])
    assert exit_code == 1


def test_slurm_only_flags_rejected_without_slurm(capsys):
    exit_code = cli_main(["data", "run", "--source", "acag", "--step", "prepare", "--dry-run"])
    assert exit_code == 1

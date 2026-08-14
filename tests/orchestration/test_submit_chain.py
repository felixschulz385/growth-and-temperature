"""submit_chain.py's DAG-building/dependency logic against the real
jobs.yaml (docs/design/10-fetch-ledger.md §6) -- no real `sbatch` needed
(only exercises `build_chain`/`_job_dependencies`/`--dry-run`, which never
shells out).
"""

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
JOBS_FILE = REPO_ROOT / "orchestration" / "slurm" / "jobs.yaml"

sys.path.insert(0, str(REPO_ROOT / "orchestration" / "slurm"))
import submit_chain as sc  # noqa: E402


def _jobs():
    with open(JOBS_FILE) as f:
        return yaml.safe_load(f)["jobs"]


def test_acag_chain_is_a_single_prepare_job():
    # acag's PREPARE now does the tiled reprojection work directly (Plan 2's
    # PREPARE+GRID merge) -- there is no separate GRID job to chain after it.
    chain = sc.build_chain("acag", _jobs())
    assert [j["name"] for j in chain] == ["acag-prepare"]


def test_plad_chain_includes_gadm_prepare_prerequisite_first():
    # plad's former GRID step is renamed PREPARE too (STEPS = FETCH, PREPARE
    # -- Plan 2's PREPARE+GRID merge; plad never had a separate PREPARE step
    # of its own, so this is a pure rename, no merge) -- REQUIRES is on
    # gadm:prepare (scoped to plad's own PREPARE step), so only
    # plad-prepare is plad's own job. gadm's PREPARE now does what used to
    # be its separate GRID step directly -- one job, still pulled in first
    # since that's gadm's own earliest-SLURM-step chain.
    chain = sc.build_chain("plad", _jobs())
    assert [j["name"] for j in chain] == ["gadm-prepare", "plad-prepare"]


def test_snl_mining_chain_includes_gadm_prepare_prerequisite():
    # snl_mining's DuckDB feature-build and tiled rasterization are now one
    # merged PREPARE job (Plan 2's PREPARE+GRID merge) -- there is no
    # separate GRID job to chain after it. REQUIRES also includes
    # commodity_prices:prepare (mine_priceshock_*,
    # src/data/sources/snl_mining/source.py) -- its own prerequisite chain
    # has no jobs of its own (commodity_prices has no REQUIRES), so only
    # commodity_prices-prepare itself is inserted.
    chain = sc.build_chain("snl_mining", _jobs())
    assert [j["name"] for j in chain] == [
        "gadm-prepare", "commodity_prices-prepare", "snl_mining-prepare",
    ]


def test_country_classifications_chain_includes_gadm_prepare_prerequisite():
    # country_classifications now has a single PREPARE job (Plan 2's
    # PREPARE+GRID merge) and REQUIRES is scoped to that same step, so
    # gadm's chain is pulled in before country_classifications-prepare runs.
    chain = sc.build_chain("country_classifications", _jobs())
    assert [j["name"] for j in chain] == ["gadm-prepare", "country_classifications-prepare"]


def test_modis_from_step_grid_skips_egress_fetch():
    # modis-fetch is host: egress, never submitted via this script.
    chain = sc.build_chain("modis", _jobs(), from_step="grid")
    assert [j["name"] for j in chain] == ["modis-grid"]


def test_modis_default_from_step_also_only_yields_grid():
    # No SLURM job exists for modis's fetch step at all (host: egress) --
    # the default "earliest SLURM step" naturally lands on grid.
    chain = sc.build_chain("modis", _jobs())
    assert [j["name"] for j in chain] == ["modis-grid"]


def test_gadm_chain_has_no_self_requires_duplication():
    chain = sc.build_chain("gadm", _jobs())
    assert [j["name"] for j in chain] == ["gadm-prepare"]


def test_job_dependencies_scoped_to_each_job_own_step():
    jobs = _jobs()
    chain = sc.build_chain("snl_mining", jobs)
    job_ids = {}
    for job in chain:
        deps = sc._job_dependencies(job, chain, job_ids)
        job_ids[job["name"]] = f"id-{job['name']}"

    # snl_mining-prepare (index 2: gadm-prepare, commodity_prices-prepare,
    # snl_mining-prepare) carries the REQUIRES entries scoped to its own
    # (merged) PREPARE step: gadm-prepare (polygon geometries for the
    # admin-count spatial join AND GID_N_code_mapping.json for
    # _export_admin_count_tables -- both now produced by gadm's PREPARE
    # directly, Plan 2's PREPARE+GRID merge) and commodity_prices-prepare
    # (mine_priceshock_* price table).
    known_ids = {
        "gadm-prepare": "id-gadm-prepare",
        "commodity_prices-prepare": "id-commodity_prices-prepare",
    }
    prepare_deps = sc._job_dependencies(chain[2], chain, known_ids)
    assert prepare_deps == ["id-gadm-prepare", "id-commodity_prices-prepare"]


def test_depends_on_escape_hatch_is_additive():
    job = {"name": "x-grid", "source": "acag", "step": "grid", "depends_on": ["some-other-job"]}
    job_ids = {"some-other-job": "id-other"}
    deps = sc._job_dependencies(job, [job], job_ids)
    assert deps == ["id-other"]


def test_unknown_source_returns_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["submit_chain.py", "--source", "not_a_real_source", "--dry-run"])
    assert sc.main() == 1

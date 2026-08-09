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


def test_acag_chain_is_prepare_then_grid():
    chain = sc.build_chain("acag", _jobs())
    assert [j["name"] for j in chain] == ["acag-prepare", "acag-grid"]


def test_plad_chain_includes_gadm_prepare_prerequisite_first():
    # plad has no PREPARE step of its own (STEPS = FETCH, GRID) -- REQUIRES
    # is on gadm:prepare, so only plad-grid is plad's own job.
    chain = sc.build_chain("plad", _jobs())
    assert [j["name"] for j in chain] == ["gadm-prepare", "gadm-grid", "plad-grid"]


def test_snl_mining_chain_includes_gadm_prepare_prerequisite():
    # REQUIRES also includes commodity_prices:prepare (mine_priceshock_*,
    # src/data/sources/snl_mining/source.py) -- its own prerequisite chain
    # has no jobs of its own (commodity_prices has no REQUIRES), so only
    # commodity_prices-prepare itself is inserted.
    chain = sc.build_chain("snl_mining", _jobs())
    assert [j["name"] for j in chain] == [
        "gadm-prepare", "gadm-grid", "commodity_prices-prepare", "snl_mining-prepare", "snl_mining-grid",
    ]


def test_country_classifications_chain_includes_gadm_grid_prerequisite():
    chain = sc.build_chain("country_classifications", _jobs())
    assert [j["name"] for j in chain] == [
        "gadm-prepare", "gadm-grid", "country_classifications-prepare", "country_classifications-grid",
    ]


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
    assert [j["name"] for j in chain] == ["gadm-prepare", "gadm-grid"]


def test_job_dependencies_only_on_source_own_first_job():
    jobs = _jobs()
    chain = sc.build_chain("snl_mining", jobs)
    job_ids = {}
    for job in chain:
        deps = sc._job_dependencies(job, chain, job_ids)
        job_ids[job["name"]] = f"id-{job['name']}"

    # snl_mining-prepare (its first own job in the chain, index 3: gadm-prepare,
    # gadm-grid, commodity_prices-prepare, snl_mining-prepare, snl_mining-grid)
    # carries the cross-source REQUIRES dependency on gadm-prepare (polygon
    # geometries for the admin-count spatial join), gadm-grid
    # (GID_N_code_mapping.json, for src/data/sources/snl_mining/source.py's
    # _export_admin_count_tables), and commodity_prices-prepare
    # (mine_priceshock_* price table) -- REQUIRES is source-level, not
    # per-step, so all three apply even though only snl_mining's own GRID
    # step actually reads gadm-grid's output.
    known_ids = {
        "gadm-prepare": "id-gadm-prepare", "gadm-grid": "id-gadm-grid",
        "commodity_prices-prepare": "id-commodity_prices-prepare",
    }
    prepare_deps = sc._job_dependencies(chain[3], chain, known_ids)
    assert prepare_deps == ["id-gadm-prepare", "id-gadm-grid", "id-commodity_prices-prepare"]

    # ...but snl_mining-grid (a later step of the same source) depends only
    # on snl_mining-prepare, not redundantly on its prerequisites again.
    grid_deps = sc._job_dependencies(
        chain[4], chain, {**known_ids, "snl_mining-prepare": "id-snl_mining-prepare"},
    )
    assert grid_deps == ["id-snl_mining-prepare"]


def test_depends_on_escape_hatch_is_additive():
    job = {"name": "x-grid", "source": "acag", "depends_on": ["some-other-job"]}
    job_ids = {"some-other-job": "id-other"}
    deps = sc._job_dependencies(job, [job], job_ids)
    assert deps == ["id-other"]


def test_unknown_source_returns_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["submit_chain.py", "--source", "not_a_real_source", "--dry-run"])
    assert sc.main() == 1

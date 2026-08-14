#!/usr/bin/env python3
"""
Submit a source's SLURM job chain with automatic ``--dependency=afterok``
chaining, derived from ``REQUIRES`` (src/data/sources/registry.py) and each
source's own FETCH->PREPARE->GRID step order.

docs/design/10-fetch-ledger.md §6. Dependency chaining is a submission-time
concern (job IDs don't exist until ``sbatch`` runs), so
``generate_slurm_scripts.py``'s rendered ``.sh`` files stay static -- this
script is the piece that actually calls ``sbatch --dependency=afterok:<ids>``
in the right order.

**Explicit human boundary**: FETCH never runs as a SLURM job (Selenium/
browser/internet-egress constraint -- see jobs.yaml's header comment), so
there is no job id to chain a source's first SLURM job from. Operator flow:
run FETCH manually/on an egress-capable host, confirm via ``data
summary``, then run this script to start PREPARE->GRID. A ``host: egress``
job with ``submit_dependents: true`` (only MODIS's PREPARE today) can
auto-invoke this script at its end, best-effort, if ``sbatch`` happens to be
on that host's PATH (see ``generate_slurm_scripts.py::render_egress_job``).

Usage:
  python orchestration/slurm/submit_chain.py --source acag
  python orchestration/slurm/submit_chain.py --source plad --dry-run
  python orchestration/slurm/submit_chain.py --source modis --from-step prepare
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
JOBS_FILE = SCRIPT_DIR / "jobs.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.sources import registry  # noqa: E402
from src.data.sources.steps import STEP_ORDER, PipelineStep  # noqa: E402


def load_jobs() -> list[dict]:
    with open(JOBS_FILE) as f:
        spec = yaml.safe_load(f)
    return spec["jobs"]


def _slurm_jobs_by_source_step(jobs: list[dict]) -> dict[tuple[str, str], dict]:
    """Only `host: slurm` (default) jobs get an sbatch id to chain from --
    `host: egress` jobs (MODIS's PREPARE) are never submitted here."""
    return {(job["source"], job["step"]): job for job in jobs if job.get("host", "slurm") == "slurm"}


def build_chain(source_id: str, jobs: list[dict], *, from_step: str | None = None) -> list[dict]:
    """The ordered list of `host: slurm` jobs to submit for *source_id*,
    starting from *from_step* (default: its earliest SLURM step). Any
    `REQUIRES` prerequisite source's own chain is resolved first
    (recursively, deduplicated) so its jobs are already in the returned
    list -- and therefore already submitted with an id available -- before
    the dependent source's own job for the step that needs it.
    """
    by_source_step = _slurm_jobs_by_source_step(jobs)
    chain: list[dict] = []
    seen: set[str] = set()

    def add_source_chain(sid: str, start_step: str | None) -> None:
        spec = registry.resolve(sid)
        started = start_step is None
        for step in STEP_ORDER:
            if step is PipelineStep.FETCH:
                continue  # never a SLURM job -- see module docstring
            if not started:
                if step.value == start_step:
                    started = True
                else:
                    continue
            job = by_source_step.get((sid, step.value))
            if job is None:
                continue
            if job["name"] in seen:
                continue
            for requires_id, _requires_step in spec.requires_for(step):
                add_source_chain(requires_id, None)
            chain.append(job)
            seen.add(job["name"])

    add_source_chain(source_id, from_step)
    return chain


def _job_dependencies(job: dict, chain: list[dict], job_ids: dict[str, str]) -> list[str]:
    """SLURM job ids *job* must wait on: the immediately preceding SLURM job
    for the same source in this chain (PREPARE -> GRID) plus every
    `REQUIRES` prerequisite job scoped to *job*'s own step specifically
    (`requires_for()`) -- REQUIRES is per-step, not source-level, so a later
    step's own prerequisite (e.g. snl_mining's GRID needing gadm's GRID) is
    no longer assumed to be already covered by an earlier step's edge.
    Also applies any explicit `depends_on:` escape-hatch names from
    jobs.yaml, unconditionally.
    """
    deps: list[str] = []

    prior_same_source = [j for j in chain[: chain.index(job)] if j["source"] == job["source"]]
    if prior_same_source and prior_same_source[-1]["name"] in job_ids:
        deps.append(job_ids[prior_same_source[-1]["name"]])

    job_step = PipelineStep(job["step"])
    for requires_id, requires_step in registry.resolve(job["source"]).requires_for(job_step):
        requires_job_name = f"{requires_id}-{requires_step.value}"
        if requires_job_name in job_ids and job_ids[requires_job_name] not in deps:
            deps.append(job_ids[requires_job_name])

    for dep_name in job.get("depends_on", []):
        if dep_name in job_ids and job_ids[dep_name] not in deps:
            deps.append(job_ids[dep_name])

    return deps


def submit(chain: list[dict], *, dry_run: bool) -> dict[str, str]:
    job_ids: dict[str, str] = {}
    for job in chain:
        script = SCRIPT_DIR / f"{job['name']}.sh"
        deps = _job_dependencies(job, chain, job_ids)

        cmd = ["sbatch", "--parsable"]
        if deps:
            cmd.append(f"--dependency=afterok:{':'.join(deps)}")
        cmd.append(str(script))

        if dry_run:
            print(" ".join(cmd))
            job_ids[job["name"]] = f"<{job['name']}-id>"
            continue

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, check=True)
        job_id = result.stdout.strip()
        job_ids[job["name"]] = job_id
        suffix = f" (deps: {deps})" if deps else ""
        print(f"Submitted {job['name']} -> job {job_id}{suffix}")

    return job_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Source to submit the SLURM chain for")
    parser.add_argument(
        "--from-step", choices=["prepare", "grid"], default=None,
        help="Start the chain from this step (default: the source's earliest SLURM step)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the sbatch commands without submitting")
    args = parser.parse_args()

    jobs = load_jobs()
    try:
        registry.resolve(args.source)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    chain = build_chain(args.source, jobs, from_step=args.from_step)
    if not chain:
        print(f"No SLURM (host: slurm) jobs found for source '{args.source}'.", file=sys.stderr)
        return 1

    submit(chain, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())

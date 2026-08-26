"""``data run --slurm`` -- submit a (source, step) as a SLURM job via
``sbatch --wrap="..."`` instead of running it in-process, using per-source
resource defaults from ``orchestration/configs/slurm_jobs.yaml`` (overridable
per invocation with ``--slurm-time``/``--slurm-mem``/``--slurm-cpus``/
``--slurm-qos``/``--slurm-partition``).

``--chain`` additionally walks this source's own fetch->prepare->grid order
plus any ``REQUIRES`` prerequisites (src/data/sources/registry.py) and
submits the whole dependency chain with ``sbatch --dependency=afterok:...``,
the same DAG logic the former ``orchestration/slurm/submit_chain.py``
implemented against the former ``orchestration/slurm/jobs.yaml``.

No script file is ever written to disk -- everything ``sbatch`` needs is
built into the ``--wrap`` string and the rest of the ``sbatch`` argv at
submission time.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from src.data.common.dask.client import DEFAULT_DASHBOARD_PORT
from src.data.sources import registry
from src.data.sources.steps import STEP_ORDER, PipelineStep

REPO_ROOT = Path(__file__).resolve().parents[3]
SLURM_CONFIG_FILE = REPO_ROOT / "orchestration" / "configs" / "slurm_jobs.yaml"


def load_slurm_config() -> dict:
    with open(SLURM_CONFIG_FILE) as f:
        return yaml.safe_load(f)


def find_job(jobs: list[dict], source: str, step: str) -> dict | None:
    for job in jobs:
        if job["source"] == source and job["step"] == step:
            return job
    return None


def _apply_resource_overrides(job: dict, args: argparse.Namespace) -> dict:
    job = dict(job)
    if args.slurm_time:
        job["time"] = args.slurm_time
    if args.slurm_mem:
        job["mem"] = args.slurm_mem
    if args.slurm_cpus:
        job["cpus"] = args.slurm_cpus
    if args.slurm_qos:
        job["qos"] = args.slurm_qos
    if args.slurm_partition:
        job["partition"] = args.slurm_partition
    return job


def _adhoc_job(args: argparse.Namespace) -> dict:
    """Build a job dict straight from CLI flags for a (source, step) that has
    no entry in slurm_jobs.yaml -- requires enough of --slurm-time/-mem/-cpus/
    -qos to fully specify an sbatch submission, since there's no per-source
    default to fall back on."""
    required = [
        ("--slurm-time", args.slurm_time),
        ("--slurm-mem", args.slurm_mem),
        ("--slurm-cpus", args.slurm_cpus),
        ("--slurm-qos", args.slurm_qos),
    ]
    missing = [name for name, val in required if not val]
    if missing:
        print(
            f"ERROR: No SLURM defaults for source='{args.source}' step='{args.step}' in "
            f"{SLURM_CONFIG_FILE.relative_to(REPO_ROOT)}, and {', '.join(missing)} "
            "not given to submit ad hoc.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return {
        "name": f"{args.source}-{args.step}",
        "log_key": args.source,
        "source": args.source,
        "step": args.step,
        "time": args.slurm_time,
        "qos": args.slurm_qos,
        "partition": args.slurm_partition,
        "cpus": args.slurm_cpus,
        "mem": args.slurm_mem,
        "mem_fraction": 0.9,
        "temp_dir_prefix": args.source,
    }


def render_wrap_command(job: dict, cluster: dict, args: argparse.Namespace) -> str:
    """The bash command string passed to ``sbatch --wrap``: cd, activate the
    conda env, then the ``data run`` invocation itself."""
    simple = job.get("simple", False)

    run_parts = [
        f'{cluster["python_bin"]} run.py data run',
        f'--config "{args.config}"',
        f"--source {job['source']}",
        f"--step {job['step']}",
    ]
    if args.override:
        run_parts.append("--override")
    if getattr(args, "years", None):
        run_parts.append(f"--years {args.years[0]} {args.years[1]}")
    if getattr(args, "keys", None):
        run_parts.append("--key " + " ".join(shlex.quote(k) for k in args.keys))
    run_parts.extend(job.get("extra_args", []))
    if not simple:
        run_parts += [
            "--dask-threads $SLURM_CPUS_PER_TASK",
            '--dask-memory-limit "${MEMORY_LIMIT_GB}GiB"',
            f'--temp-dir "{cluster["scratch_prefix"]}/{job["temp_dir_prefix"]}_${{SLURM_JOB_ID}}"',
            f"--dashboard-port {job.get('dashboard_port', DEFAULT_DASHBOARD_PORT)}",
        ]
    if getattr(args, "debug", False):
        run_parts.append("--debug")

    lines = [
        f'cd {cluster["project_root"]}',
        f'eval "$({cluster["conda_hook"]} shell.bash hook)"',
        "conda activate src",
    ]
    if not simple:
        mem_fraction = job["mem_fraction"]
        lines.append(f'MEMORY_LIMIT_GB=$(echo "scale=0; $SLURM_MEM_PER_NODE * {mem_fraction} / 1024" | bc)')
    lines.append(" ".join(run_parts))
    return " && ".join(lines)


def sbatch_argv(job: dict, cluster: dict, wrap_cmd: str, *, deps: list[str]) -> tuple[list[str], str]:
    """Return (sbatch argv, log dir) for *job*."""
    log_dir = f'{cluster["project_root"]}/log/preprocess/{job["log_key"]}'
    argv = [
        "sbatch",
        "--parsable",
        f"--job-name={job['name']}",
        f"--output={log_dir}/%x-%j.out",
        f"--error={log_dir}/%x-%j.err",
    ]
    if job.get("partition"):
        argv.append(f"--partition={job['partition']}")
    argv += [
        f"--time={job['time']}",
        f"--qos={job['qos']}",
        f"--cpus-per-task={job.get('cpus', 8)}",
        f"--mem={job['mem']}",
    ]
    if deps:
        argv.append(f"--dependency=afterok:{':'.join(deps)}")
    argv.append(f"--wrap={wrap_cmd}")
    return argv, log_dir


def _submit_one(job: dict, cluster: dict, args: argparse.Namespace, *, deps: list[str], dry_run: bool) -> str:
    wrap_cmd = render_wrap_command(job, cluster, args)
    argv, log_dir = sbatch_argv(job, cluster, wrap_cmd, deps=deps)

    if dry_run:
        print(" ".join(shlex.quote(a) for a in argv))
        return f"<{job['name']}-id>"

    os.makedirs(log_dir, exist_ok=True)
    try:
        result = subprocess.run(argv, capture_output=True, text=True, cwd=REPO_ROOT, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: sbatch failed for {job['name']}: {e.stderr.strip()}", file=sys.stderr)
        raise SystemExit(1) from e
    job_id = result.stdout.strip()
    suffix = f" (deps: {deps})" if deps else ""
    print(f"Submitted {job['name']} -> job {job_id}{suffix}")
    return job_id


def _jobs_by_source_step(jobs: list[dict]) -> dict[tuple[str, str], dict]:
    return {(job["source"], job["step"]): job for job in jobs}


def build_chain(source_id: str, jobs: list[dict], *, from_step: str | None = None) -> list[dict]:
    """The ordered list of jobs to submit for *source_id*, starting from
    *from_step* (default: its earliest step with a jobs.yaml entry). Any
    ``REQUIRES`` prerequisite source's own chain is resolved first
    (recursively, deduplicated) so its jobs are already in the returned
    list -- and therefore already submitted with an id available -- before
    the dependent source's own job for the step that needs it."""
    by_source_step = _jobs_by_source_step(jobs)
    chain: list[dict] = []
    seen: set[str] = set()

    def add_source_chain(sid: str, start_step: str | None) -> None:
        spec = registry.resolve(sid)
        started = start_step is None
        for step in STEP_ORDER:
            if step is PipelineStep.FETCH:
                continue  # never a SLURM job
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


def job_dependencies(job: dict, chain: list[dict], job_ids: dict[str, str]) -> list[str]:
    """SLURM job ids *job* must wait on: the immediately preceding SLURM job
    for the same source in this chain (PREPARE -> GRID) plus every
    ``REQUIRES`` prerequisite job scoped to *job*'s own step specifically,
    plus any explicit ``depends_on:`` escape-hatch names from
    slurm_jobs.yaml."""
    deps: list[str] = []

    prior_same_source = [j for j in chain[: chain.index(job)] if j["source"] == job["source"]]
    if prior_same_source and prior_same_source[-1]["name"] in job_ids:
        deps.append(job_ids[prior_same_source[-1]["name"]])

    by_source_step = _jobs_by_source_step(chain)
    job_step = PipelineStep(job["step"])
    for requires_id, requires_step in registry.resolve(job["source"]).requires_for(job_step):
        requires_job = by_source_step.get((requires_id, requires_step.value))
        if requires_job is None:
            continue
        requires_job_name = requires_job["name"]
        if requires_job_name in job_ids and job_ids[requires_job_name] not in deps:
            deps.append(job_ids[requires_job_name])

    for dep_name in job.get("depends_on", []):
        if dep_name in job_ids and job_ids[dep_name] not in deps:
            deps.append(job_ids[dep_name])

    return deps


def submit(args: argparse.Namespace) -> None:
    """``data run --slurm``'s entry point, called from
    ``src.cli.data.handlers.handle_run``."""
    cfg = load_slurm_config()
    jobs = cfg["jobs"]
    cluster = cfg["cluster"]

    if args.chain:
        chain = build_chain(args.source, jobs, from_step=args.step)
        if not chain:
            print(f"ERROR: No SLURM job(s) found for source '{args.source}' starting at step '{args.step}'.", file=sys.stderr)
            raise SystemExit(1)
        job_ids: dict[str, str] = {}
        for job in chain:
            deps = job_dependencies(job, chain, job_ids)
            resolved = _apply_resource_overrides(job, args) if (job["source"], job["step"]) == (args.source, args.step) else job
            try:
                job_ids[job["name"]] = _submit_one(resolved, cluster, args, deps=deps, dry_run=args.dry_run)
            except SystemExit:
                if job_ids:
                    print(
                        f"NOTE: {len(job_ids)} earlier job(s) in this chain were already submitted "
                        f"and are unaffected: {', '.join(job_ids.values())}",
                        file=sys.stderr,
                    )
                raise
        return

    job = find_job(jobs, args.source, args.step)
    job = _apply_resource_overrides(job, args) if job is not None else _adhoc_job(args)
    _submit_one(job, cluster, args, deps=[], dry_run=args.dry_run)

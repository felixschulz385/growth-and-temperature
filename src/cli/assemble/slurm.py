"""``assemble create/update --slurm`` -- submit an assembly run as a SLURM job
via ``sbatch --wrap="..."`` instead of running it in-process, using per-grid
resource defaults from ``orchestration/configs/slurm_jobs.yaml``'s
``assembly_jobs:`` block (overridable per invocation with ``--slurm-time``/
``--slurm-mem``/``--slurm-cpus``/``--slurm-qos``/``--slurm-partition``).

No script file is written to disk -- everything ``sbatch`` needs is built into
the ``--wrap`` string at submission time, the same pattern as
``src/cli/data/slurm.py``. A ``--shake`` preset runs all its variants
sequentially inside the one job.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from src.data.assemble.grid_shake import resolve_shake_selection

REPO_ROOT = Path(__file__).resolve().parents[3]
SLURM_CONFIG_FILE = REPO_ROOT / "orchestration" / "configs" / "slurm_jobs.yaml"


def load_slurm_config() -> dict:
    with open(SLURM_CONFIG_FILE) as f:
        return yaml.safe_load(f)


def find_assembly_job(assembly_jobs: list | None, grid_label: str) -> dict | None:
    for job in assembly_jobs or []:
        if job.get("grid") == grid_label:
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
    """Build a job dict straight from CLI flags for a ``--grid`` with no
    ``assembly_jobs`` entry -- requires --slurm-time/-mem/-cpus/-qos to fully
    specify an sbatch submission."""
    required = [
        ("--slurm-time", args.slurm_time),
        ("--slurm-mem", args.slurm_mem),
        ("--slurm-cpus", args.slurm_cpus),
        ("--slurm-qos", args.slurm_qos),
    ]
    missing = [name for name, val in required if not val]
    if missing:
        print(
            f"ERROR: No assembly_jobs entry for grid={args.grid!r} in "
            f"{SLURM_CONFIG_FILE.relative_to(REPO_ROOT)}, and {', '.join(missing)} "
            "not given to submit ad hoc.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return {
        "grid": args.grid,
        "time": args.slurm_time,
        "qos": args.slurm_qos,
        "partition": args.slurm_partition,
        "cpus": args.slurm_cpus,
        "mem": args.slurm_mem,
        "mem_fraction": 0.6,
    }


def _subcommand(args: argparse.Namespace) -> str:
    return "update" if getattr(args, "datasource", None) else "create"


def render_wrap_command(job: dict, cluster: dict, args: argparse.Namespace) -> str:
    """The bash command string passed to ``sbatch --wrap``: cd, activate the
    conda env, then the ``assemble`` invocation itself."""
    sub = _subcommand(args)
    run_parts = [
        f'{cluster["python_bin"]} run.py assemble {sub}',
        f'--config "{args.config}"',
        f"--grid {args.grid}",
        f"--shake {args.shake}",
    ]
    if sub == "update":
        run_parts.append(f"--datasource {args.datasource}")
    elif getattr(args, "overwrite", None) is False:
        run_parts.append("--no-overwrite")
    run_parts += [
        "--threads $SLURM_CPUS_PER_TASK",
        '--memory-limit "${MEMORY_LIMIT_GB}GB"',
        f'--temp-dir "{cluster["scratch_prefix"]}/assemble_${{SLURM_JOB_ID}}"',
    ]
    if getattr(args, "debug", False):
        run_parts.append("--debug")

    # DuckDB gets the bulk of the node's RAM; it spills the rest to --temp-dir.
    mem_fraction = job.get("mem_fraction", 0.85)
    lines = [
        f'cd {cluster["project_root"]}',
        f'eval "$({cluster["conda_hook"]} shell.bash hook)"',
        "conda activate gnt",
        f'MEMORY_LIMIT_GB=$(echo "scale=0; $SLURM_MEM_PER_NODE * {mem_fraction} / 1024" | bc)',
        " ".join(run_parts),
    ]
    return " && ".join(lines)


def sbatch_argv(job: dict, cluster: dict, wrap_cmd: str, *, job_name: str) -> tuple[list[str], str]:
    """Return (sbatch argv, log dir) for *job*."""
    log_dir = f'{cluster["project_root"]}/log/assemble/{job["grid"]}'
    argv = [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
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
    argv.append(f"--wrap={wrap_cmd}")
    return argv, log_dir


def submit(args: argparse.Namespace) -> None:
    """``assemble ... --slurm``'s entry point, called from the assemble handlers."""
    cfg = load_slurm_config()
    cluster = cfg["cluster"]

    job = find_assembly_job(cfg.get("assembly_jobs"), args.grid)
    job = _apply_resource_overrides(job, args) if job is not None else _adhoc_job(args)

    sub = _subcommand(args)
    job_name = f"assemble-{sub}-{args.grid}"
    wrap_cmd = render_wrap_command(job, cluster, args)
    argv, log_dir = sbatch_argv(job, cluster, wrap_cmd, job_name=job_name)

    variants = [label for label, _, _ in resolve_shake_selection(getattr(args, "shake", "none"))]

    if args.dry_run:
        print(" ".join(shlex.quote(a) for a in argv))
        if len(variants) > 1:
            print(f"# one job; shake variants {variants} run sequentially inside it")
        return

    os.makedirs(log_dir, exist_ok=True)
    try:
        result = subprocess.run(argv, capture_output=True, text=True, cwd=REPO_ROOT, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: sbatch failed for {job_name}: {e.stderr.strip()}", file=sys.stderr)
        raise SystemExit(1) from e
    print(f"Submitted {job_name} -> job {result.stdout.strip()} (shake variants: {variants})")

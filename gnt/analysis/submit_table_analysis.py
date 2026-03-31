#!/usr/bin/env python3
"""
Submit a single SLURM job to run all models for one or more tables sequentially.

Usage:
    python submit_table_analysis.py <table_id> [<table_id> ...] [--mem=<memory>] [--time=<time>] [--partition=<partition>]

Example:
    python submit_table_analysis.py table1
    python submit_table_analysis.py table1 table2 --mem=64GB
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import pandas as pd
import tempfile

try:
    from duckreg._version import __version__ as _DUCKREG_VERSION
except ImportError:
    _DUCKREG_VERSION = "unknown"

ONE_WEEK_SECONDS = 7 * 24 * 3600


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def parse_runtime_to_seconds(runtime_str: str) -> int:
    """
    Parse a runtime string in HH:MM:SS or D-HH:MM:SS format to seconds.
    """
    runtime_str = str(runtime_str).strip()
    days = 0

    if '-' in runtime_str:
        day_part, time_part = runtime_str.split('-', 1)
        days = int(day_part)
    else:
        time_part = runtime_str

    parts = time_part.split(':')
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        h, m, s = 0, int(parts[0]), int(parts[1])
    else:
        raise ValueError(f"Cannot parse runtime: {runtime_str!r}")

    return days * 86400 + h * 3600 + m * 60 + s


def seconds_to_slurm_time(seconds: int) -> str:
    """Convert seconds to D-HH:MM:SS format for SLURM."""
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{days}-{hours:02d}:{minutes:02d}:{secs:02d}"


def load_excel_sheets(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both relevant sheets from the Excel config."""
    excel_path = project_root / 'orchestration' / 'configs' / 'analysis.xlsx'
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df_models_in_tables = pd.read_excel(excel_path, sheet_name='Models in Tables')
    df_models = pd.read_excel(excel_path, sheet_name='Models')
    return df_models_in_tables, df_models


def get_models_for_table(
    table_id: str,
    df_models_in_tables: pd.DataFrame,
    df_models: pd.DataFrame,
) -> tuple[list, int]:
    """
    Extract model names for a specific table and compute total runtime in seconds.

    Args:
        table_id: Table ID to filter by
        df_models_in_tables: DataFrame from 'Models in Tables' sheet
        df_models: DataFrame from 'Models' sheet

    Returns:
        Tuple of (list of model names sorted by order, total runtime in seconds)
    """
    table_models = df_models_in_tables[
        df_models_in_tables['table_name'] == table_id
    ].sort_values('order')

    if len(table_models) == 0:
        raise ValueError(f"No models found for table: {table_id!r}")

    models = table_models['model_name'].tolist()

    # Look up max_runtime for each model
    model_runtime_map = df_models.set_index('model_name')['max_runtime']
    total_seconds = 0
    for model in models:
        if model not in model_runtime_map.index:
            raise ValueError(
                f"Model {model!r} (from table {table_id!r}) not found in 'Models' sheet"
            )
        runtime_str = model_runtime_map[model]
        total_seconds += parse_runtime_to_seconds(runtime_str)

    return models, total_seconds


def create_job_script(
    table_model_pairs: list[tuple[str, list]],
    project_root: Path,
    job_label: str,
    duckreg_version: str,
) -> str:
    """
    Create a SLURM job script that runs all models for all tables sequentially.

    Args:
        table_model_pairs: List of (table_id, [model_names]) tuples in order
        project_root: Project root path
        job_label: Label used in job name and top-level log dir
        duckreg_version: duckreg version string
    """
    job_log_dir = project_root / 'log' / 'analysis' / f'table-{job_label}' / duckreg_version

    script_content = f"""#!/bin/bash
#SBATCH --job-name=table-{job_label}
#SBATCH --output={job_log_dir}/slurm-%j.log
#SBATCH --error={job_log_dir}/slurm-%j.err

set -eo pipefail

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
set +u
conda activate gnt
set -u

export WD="{project_root}"
export SCRATCH_DIR="${{WD}}/scratch_nobackup/${{SLURM_JOB_ID}}"

mkdir -p "{job_log_dir}"
mkdir -p "${{SCRATCH_DIR}}"

cleanup() {{
    rm -rf "${{SCRATCH_DIR}}"
}}
trap cleanup EXIT

cd "${{WD}}"

echo "[$(date -Is)] Job started for tables: {job_label}"
echo "[$(date -Is)] WD=${{WD}}"
echo "[$(date -Is)] SCRATCH_DIR=${{SCRATCH_DIR}}"

"""

    total_model_count = sum(len(models) for _, models in table_model_pairs)
    global_index = 0

    for table_id, models in table_model_pairs:
        script_content += f"""
# ══════════════════════════════════════════════════════════════════════════════
# Table: {table_id}
# ══════════════════════════════════════════════════════════════════════════════
echo "[$(date -Is)] Starting table {table_id} ({len(models)} models)"

"""
        for i, model in enumerate(models, 1):
            global_index += 1
            model_log_dir = project_root / 'log' / 'analysis' / model / duckreg_version
            script_content += f"""# ── Model {i}/{len(models)} in {table_id}  [{global_index}/{total_model_count} overall]: {model}
mkdir -p "{model_log_dir}"
echo "[$(date -Is)] Running model {i}/{len(models)}: {model}"
python run.py analysis \\
    --config orchestration/configs/analysis.xlsx \\
    --model "{model}" \\
    --output output/analysis \\
    >> "{model_log_dir}/slurm-$SLURM_JOB_ID.log" \\
    2>> "{model_log_dir}/slurm-$SLURM_JOB_ID.err"
echo "[$(date -Is)] Completed: {model}"
rm -rf "${{SCRATCH_DIR:?}}"/* 2>/dev/null || true
echo ""

"""

        script_content += f'echo "[$(date -Is)] Finished table {table_id}"\necho ""\n\n'

    script_content += 'echo "[$(date -Is)] All models completed successfully!"\n'

    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(script_content)
        return f.name


def submit_job(job_script: str, slurm_kwargs: dict) -> str:
    """Submit the job script to SLURM and return the job ID."""
    cmd = ['sbatch']

    if slurm_kwargs.get('mem'):
        cmd.extend(['--mem', slurm_kwargs['mem']])
    if slurm_kwargs.get('time'):
        cmd.extend(['--time', slurm_kwargs['time']])
    if slurm_kwargs.get('qos'):
        cmd.extend(['--qos', slurm_kwargs['qos']])
    if slurm_kwargs.get('partition'):
        cmd.extend(['--partition', slurm_kwargs['partition']])
    if slurm_kwargs.get('cpus_per_task'):
        cmd.extend(['--cpus-per-task', str(slurm_kwargs['cpus_per_task'])])

    cmd.append(job_script)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to submit job: {result.stderr}")

    job_id = result.stdout.strip().split()[-1]
    return job_id


def main():
    """CLI entry-point — delegates to :mod:`gnt.analysis.slurm` and :class:`~gnt.analysis.config.AnalysisConfig`."""
    parser = argparse.ArgumentParser(
        description="Submit a single SLURM job to run all models for one or more tables sequentially"
    )
    parser.add_argument(
        '--tables', nargs='*', metavar='TABLE', default=[],
        help='Table names to submit (all models in each table are included)'
    )
    parser.add_argument(
        '--models', nargs='*', metavar='MODEL', default=[],
        help='Individual model names to submit'
    )
    parser.add_argument('--mem', default='128GB', help='Memory allocation (default: 128GB)')
    parser.add_argument('--time', default=None, help='Time limit override (default: auto from max_runtime sum)')
    parser.add_argument('--qos', default='1week', help='QOS (default: 1week)')
    parser.add_argument('--partition', default='scicore', help='SLURM partition (default: scicore)')
    parser.add_argument('--cpus-per-task', default=8, type=int, help='CPUs per task (default: 8)')

    args = parser.parse_args()

    try:
        from gnt.analysis.config import AnalysisConfig, seconds_to_slurm_time, PROJECT_ROOT
        from gnt.analysis.slurm import (
            write_job_script, submit_job,
            resolve_explicit_pairs, make_job_label, ONE_WEEK_SECONDS,
        )
        import os

        tables = list(args.tables or [])
        models_list = list(args.models or [])
        if not tables and not models_list:
            raise ValueError("At least one --tables or --models argument is required")

        config = AnalysisConfig()
        table_model_pairs, grand_total_seconds = resolve_explicit_pairs(
            tables, models_list, config
        )

        print(f"\nTotal combined runtime: {seconds_to_slurm_time(grand_total_seconds)}")

        if grand_total_seconds > ONE_WEEK_SECONDS:
            raise ValueError(
                f"Total runtime {seconds_to_slurm_time(grand_total_seconds)} exceeds the "
                f"1-week limit ({seconds_to_slurm_time(ONE_WEEK_SECONDS)}). "
                "Split the tables across multiple jobs."
            )

        all_identifiers = tables + models_list
        slurm_time = args.time if args.time else seconds_to_slurm_time(grand_total_seconds)
        job_label = make_job_label(all_identifiers)

        print(f"\nCreating job script... (duckreg {_DUCKREG_VERSION})")
        job_script_path = write_job_script(
            table_model_pairs, PROJECT_ROOT, job_label, _DUCKREG_VERSION
        )

        print("Submitting job to SLURM...")
        slurm_kwargs = {
            'mem': args.mem,
            'time': slurm_time,
            'qos': args.qos,
            'partition': args.partition,
            'cpus_per_task': args.cpus_per_task,
        }

        job_id = submit_job(job_script_path, slurm_kwargs)

        total_models = sum(len(m) for _, m in table_model_pairs)
        print(f"\nJob submitted successfully!")
        print(f"  Job ID:      {job_id}")
        print(f"  Tables:      {', '.join(tables) if tables else '—'}")
        print(f"  Models:      {', '.join(models_list) if models_list else '—'}")
        print(f"  Total mdls:  {total_models}")
        print(f"  duckreg:     {_DUCKREG_VERSION}")
        print(f"  Memory:      {args.mem}")
        print(f"  Time:        {slurm_time}")
        print(f"  QOS:         {args.qos}")

        os.remove(job_script_path)
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

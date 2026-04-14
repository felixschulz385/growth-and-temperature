"""
SLURM job creation and submission utilities.

This module is purely concerned with SLURM: building batch scripts, submitting
them via ``sbatch``, and checking constraints.  All business-logic about which
models belong to which table lives in :mod:`~gnt.analysis.config`.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import (
    AnalysisConfig,
    PROJECT_ROOT,
    seconds_to_slurm_time,
)

ONE_WEEK_SECONDS = 7 * 24 * 3600
DEFAULT_CONDA_HOOK = (
    "/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook"
)
DEFAULT_CONDA_ENV = "gnt"


# ---------------------------------------------------------------------------
# Script building
# ---------------------------------------------------------------------------

def _model_block(
    model: str,
    model_idx: int,
    table_idx: int,
    n_models_in_table: int,
    n_models_total: int,
    table_id: str,
    model_log_dir: Path,
    duckreg_version: str,
) -> str:
    """Return the bash fragment that runs a single model."""
    return (
        f"# ── Model {table_idx}/{n_models_in_table} in {table_id}"
        f"  [{model_idx}/{n_models_total} overall]: {model}\n"
        f'mkdir -p "{model_log_dir}"\n'
        f'echo "[$(date -Is)] Running model {table_idx}/{n_models_in_table}: {model}"\n'
        f"python run.py analysis \\\n"
        f"    --config orchestration/configs/analysis.xlsx \\\n"
        f'    --model "{model}" \\\n'
        f"    --output output/analysis \\\n"
        f'    >> "{model_log_dir}/slurm-$SLURM_JOB_ID.log" \\\n'
        f'    2>> "{model_log_dir}/slurm-$SLURM_JOB_ID.err"\n'
        f'echo "[$(date -Is)] Completed: {model}"\n'
        f'rm -rf "${{SCRATCH_DIR:?}}"/* 2>/dev/null || true\n'
        f'echo ""\n'
        "\n"
    )


def build_job_script(
    table_model_pairs: List[Tuple[str, List[str]]],
    project_root: Path,
    job_label: str,
    duckreg_version: str,
    conda_hook: str = DEFAULT_CONDA_HOOK,
    conda_env: str = DEFAULT_CONDA_ENV,
) -> str:
    """Return a SLURM batch script as a string.

    Parameters
    ----------
    table_model_pairs:
        List of ``(table_id, [model_name, …])`` tuples.
    project_root:
        Absolute path to the project root.
    job_label:
        Human-readable label used in the job name and log directory.
    duckreg_version:
        duckreg version string embedded in the log path.
    conda_hook / conda_env:
        Conda initialisation command and environment name.
    """
    job_log_dir = (
        project_root / 'log' / 'analysis' / f'table-{job_label}' / duckreg_version
    )

    lines: List[str] = [
        "#!/bin/bash",
        f"#SBATCH --job-name=table-{job_label}",
        f"#SBATCH --output={job_log_dir}/slurm-%j.log",
        f"#SBATCH --error={job_log_dir}/slurm-%j.err",
        "",
        "set -eo pipefail",
        "",
        f'eval "$({conda_hook})"',
        "set +u",
        f"conda activate {conda_env}",
        "set -u",
        "",
        f'export WD="{project_root}"',
        'export SCRATCH_DIR="${WD}/scratch_nobackup/${SLURM_JOB_ID}"',
        "",
        f'mkdir -p "{job_log_dir}"',
        'mkdir -p "${SCRATCH_DIR}"',
        "",
        "cleanup() {",
        '    rm -rf "${SCRATCH_DIR}"',
        "}",
        "trap cleanup EXIT",
        "",
        'cd "${WD}"',
        "",
        f'echo "[$(date -Is)] Job started for tables: {job_label}"',
        'echo "[$(date -Is)] WD=${WD}"',
        'echo "[$(date -Is)] SCRATCH_DIR=${SCRATCH_DIR}"',
        "",
    ]

    total_model_count = sum(len(models) for _, models in table_model_pairs)
    global_index = 0

    for table_id, models in table_model_pairs:
        lines += [
            "",
            "# " + "═" * 78,
            f"# Table: {table_id}",
            "# " + "═" * 78,
            f'echo "[$(date -Is)] Starting table {table_id} ({len(models)} models)"',
            "",
        ]
        for i, model in enumerate(models, 1):
            global_index += 1
            model_log_dir = (
                project_root / 'log' / 'analysis' / model / duckreg_version
            )
            lines.append(
                _model_block(
                    model=model,
                    model_idx=global_index,
                    table_idx=i,
                    n_models_in_table=len(models),
                    n_models_total=total_model_count,
                    table_id=table_id,
                    model_log_dir=model_log_dir,
                    duckreg_version=duckreg_version,
                )
            )
        lines += [
            f'echo "[$(date -Is)] Finished table {table_id}"',
            'echo ""',
            "",
        ]

    lines.append('echo "[$(date -Is)] All models completed successfully!"')

    return "\n".join(lines) + "\n"


def write_job_script(
    table_model_pairs: List[Tuple[str, List[str]]],
    project_root: Path,
    job_label: str,
    duckreg_version: str,
    **kwargs,
) -> str:
    """Write a SLURM script to a temp file and return its path."""
    content = build_job_script(
        table_model_pairs, project_root, job_label, duckreg_version, **kwargs
    )
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as fh:
        fh.write(content)
        return fh.name


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def submit_job(job_script: str, slurm_kwargs: Dict[str, str]) -> str:
    """Submit *job_script* via ``sbatch`` and return the numeric job ID.

    Parameters
    ----------
    job_script:
        Path to the batch script file.
    slurm_kwargs:
        Mapping of SLURM option names to values.  Recognised keys:
        ``mem``, ``time``, ``qos``, ``partition``, ``cpus_per_task``.

    Raises
    ------
    RuntimeError
        When ``sbatch`` exits with a non-zero status.
    """
    cmd = ['sbatch']
    if slurm_kwargs.get('mem'):
        cmd += ['--mem', slurm_kwargs['mem']]
    if slurm_kwargs.get('time'):
        cmd += ['--time', slurm_kwargs['time']]
    if slurm_kwargs.get('qos'):
        cmd += ['--qos', slurm_kwargs['qos']]
    if slurm_kwargs.get('partition'):
        cmd += ['--partition', slurm_kwargs['partition']]
    if slurm_kwargs.get('cpus_per_task'):
        cmd += ['--cpus-per-task', str(slurm_kwargs['cpus_per_task'])]
    cmd.append(job_script)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed:\n{result.stderr.strip()}")

    return result.stdout.strip().split()[-1]


# ---------------------------------------------------------------------------
# High-level: resolve table/model args and submit
# ---------------------------------------------------------------------------

def resolve_table_model_pairs(
    identifiers: List[str],
    config: AnalysisConfig,
) -> Tuple[List[Tuple[str, List[str]]], int]:
    """Resolve each identifier to ``(label, [model_names])`` and sum runtimes.

    Each identifier is treated as a table name first; if not found it is tried
    as a model name.  A :class:`ValueError` is raised for unknown identifiers.

    Returns
    -------
    table_model_pairs:
        List of ``(label, [model_names])`` in input order.
    grand_total_seconds:
        Sum of derived runtime budgets for every resolved model.
    """
    table_model_pairs: List[Tuple[str, List[str]]] = []
    grand_total_seconds = 0

    known_tables = set(config.get_all_table_names())
    known_models = set(config.get_model_names())

    for ident in identifiers:
        if ident in known_tables:
            models = config.get_models_for_table(ident)
            secs = config.get_table_runtime_seconds(ident)
        elif ident in known_models:
            models = [ident]
            secs = config.get_model_runtime_seconds(ident)
        else:
            raise ValueError(
                f"'{ident}' is neither a table in 'Models in Tables' "
                f"nor a model in 'Models' sheet."
            )
        table_model_pairs.append((ident, models))
        grand_total_seconds += secs

    return table_model_pairs, grand_total_seconds


def resolve_explicit_pairs(
    tables: List[str],
    models: List[str],
    config: AnalysisConfig,
) -> Tuple[List[Tuple[str, List[str]]], int]:
    """Resolve separately specified tables and individual models into pairs.

    Unlike :func:`resolve_table_model_pairs`, this function does not
    auto-detect identifier types: items in *tables* are always looked up in
    the ``Models in Tables`` sheet and items in *models* are always looked up
    in the ``Models`` sheet.

    Returns
    -------
    table_model_pairs:
        Tables first (in the order given), then individual models.  Each
        entry is a ``(label, [model_names])`` tuple.
    grand_total_seconds:
        Sum of derived runtime budgets for every resolved model.
    """
    pairs: List[Tuple[str, List[str]]] = []
    total_secs = 0

    for table_id in (tables or []):
        models_in_table = config.get_models_for_table(table_id)
        pairs.append((table_id, models_in_table))
        total_secs += config.get_table_runtime_seconds(table_id)

    for model in (models or []):
        pairs.append((model, [model]))
        total_secs += config.get_model_runtime_seconds(model)

    return pairs, total_secs


def make_job_label(identifiers: List[str], max_inline: int = 3) -> str:
    """Return a short human-readable job label from a list of identifiers."""
    if len(identifiers) <= max_inline:
        return "_".join(identifiers)
    return f"{identifiers[0]}_and_{len(identifiers) - 1}_more"

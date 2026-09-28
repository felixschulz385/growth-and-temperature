# SLURM logging

Where SLURM's stdout/stderr ends up for each kind of job. All paths are relative to the project
root on the HPC (`cluster.project_root` in `orchestration/configs/slurm_jobs.yaml`).

| Job | Submitted by | Log directory | Files |
|---|---|---|---|
| Data pipeline (`data run --slurm`) | `src/cli/data/slurm.py` | `log/preprocess/<source>/` | `%x-%j.out`, `%x-%j.err` |
| Assembly (`assemble create/update --slurm`) | `src/cli/assemble/slurm.py` | `log/assemble/<grid>/` | `%x-%j.out`, `%x-%j.err` |
| Maintenance/validation scripts in this directory | `sbatch orchestration/scripts/<name>.sh` | `log/maintenance/<name-or-group>/` | `%x-%j.out`, `%x-%j.err` |
| Analysis (`analysis submit`, `analysis.sh`) | `src/analysis/orchestration/slurm.py` | `log/analysis/<model-or-table>/<duckreg_version>/` | `slurm-<jobid>.log`, `.err` |

`%x` is the job name and `%j` the job id. The two CLI submitters create their log directory
before calling `sbatch`, and submit with `sbatch --wrap`. No generated `.sh` files are involved;
job resources come from `slurm_jobs.yaml` and can be overridden with `--slurm-time`/`-mem`/`-cpus`/
`-qos`/`-partition`.

## Conventions for the scripts in this directory

- `--job-name` equals the script's filename stem, so `%x` resolves to it.
- `--output`/`--error` are relative (`./log/maintenance/...`), so submit from the project root.
- SLURM opens the log files before the script body runs, so the directory must already exist.
  `log/` is git-ignored; after a fresh clone, create it first
  (`mkdir -p log/maintenance/<name>`).

## Exception: the analysis family

`analysis.sh` and `src/analysis/orchestration/slurm.py` use their own layout:
`log/analysis/<model-or-table>/<duckreg_version>/`, `.log`/`.err` extensions, and
`echo "[$(date -Is)] ..."` markers instead of Python `logging`. `scripts/screen_analysis_logs.py`
parses exactly that format and directory depth, so keep the two in sync if either changes.
`analysis.sh` also sets a fallback `--output=./log/_bootstrap/%x-%j.out` to catch anything printed
before its own redirection takes over, e.g. a failed `conda activate`.

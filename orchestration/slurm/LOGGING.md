# SLURM script logging convention

All scripts in this directory except `analysis.sh` (documented exception below) follow one convention for SLURM's own stdout/stderr capture:

- **Directory**: `log/<category>/<key>/`, where:
  - `<category>` is one of `preprocess`, `assemble`, `maintenance` — mirrors `src/cli/<domain>`.
  - `<key>` is the script's `--source` value (e.g. `acag`, `esacci`, `snl_mining`) for source-keyed preprocess/assemble scripts, or the script's own operation name for scripts without a fixed source (`create`, `update`, `demean` for assemble; `compress`, `rechunk` for maintenance).
- **Filename**: always `%x-%j.out` / `%x-%j.err` — SLURM's own job-name/job-id substitutions, never a hand-typed literal.
- **Invariant**: `--job-name` must equal the script's own filename stem (e.g. `esacci-preprocess-spatial.sh` → `--job-name=esacci-preprocess-spatial`). This is what `%x` resolves to, and enforcing it prevents copy-paste bugs where a script's log ends up under another script's directory or job name (this happened twice before this convention: `esacci-preprocess-spatial.sh` was writing into ACAG's log dir, and `plad-preprocess.sh` was using `eog-preprocess`'s job name). Check with `lint_slurm_scripts.sh`.
- **Path form**: pragma paths stay relative (`./log/...`). All scripts here are submitted from the project root by convention — `sbatch` must be invoked from there.

## Directory pre-creation

SLURM opens the `--output`/`--error` files at job start, before the script body runs, and the `#SBATCH` pragma cannot execute a command — so a `mkdir -p` inside the script body cannot help the very first submission after `log/` doesn't exist (it's gitignored, so this happens on every fresh clone and after any `git clean`).

Run once after cloning, and again any time `log/` is wiped:

```bash
bash orchestration/slurm/bootstrap_log_dirs.sh
```

This parses every script's `--output=`/`--error=` line and `mkdir -p`s each directory referenced. It's self-maintaining — adding a new script needs no separate list update.

Each script also keeps a defensive `mkdir -p "$(dirname ...)"` as its first body line, in case a later cleanup job deletes the directory between a bootstrap run and a subsequent submission. This is defense-in-depth, not the primary mechanism — the bootstrap script is what actually runs early enough to matter.

## Linting

```bash
bash orchestration/slurm/lint_slurm_scripts.sh
```

Checks every script (except `analysis.sh`) for the job-name/filename-stem invariant and the `%x-%j.out`/`%x-%j.err` filename pattern. Run this before committing a new or edited `.sh` script.

## Exception: the analysis family

`analysis.sh` and `src/analysis/orchestration/slurm.py` (which generates SLURM scripts for batched model runs) use a different, documented scheme: `log/analysis/<model-or-table>/<duckreg_version>/`, `.log`/`.err` extensions, and manual `echo "[$(date -Is)] ..."` markers instead of Python `logging`. This is deliberate — `scripts/screen_analysis_logs.py` parses that exact format and directory depth. Do not fold the analysis family into the general convention above; if you touch its log paths, keep `screen_analysis_logs.py` in sync.

`analysis.sh` additionally sets a fallback `#SBATCH --output=./log/_bootstrap/%x-%j.out`/`--error=...` to catch anything printed before its manual `exec > ... 2> ...` redirection takes over (e.g. a failed `conda activate`).

#!/usr/bin/env bash
# Verify every SLURM script (except analysis.sh, a documented exception —
# see LOGGING.md) follows the standard job-name/output/error convention.
# Run before committing a new or edited orchestration/slurm/*.sh script.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fail=0

for f in "${SCRIPT_DIR}"/*.sh; do
    base="$(basename "$f" .sh)"
    [ "$base" = "bootstrap_log_dirs" ] && continue
    [ "$base" = "bootstrap_duckdb_extensions" ] && continue
    [ "$base" = "lint_slurm_scripts" ] && continue
    [ "$base" = "analysis" ] && continue

    job_name="$(sed -nE 's/^#SBATCH[[:space:]]+--job-name=([^[:space:]]+).*/\1/p' "$f" | head -1)"
    output="$(sed -nE 's/^#SBATCH[[:space:]]+--output=([^[:space:]]+).*/\1/p' "$f" | head -1)"
    error="$(sed -nE 's/^#SBATCH[[:space:]]+--error=([^[:space:]]+).*/\1/p' "$f" | head -1)"

    if [ "$job_name" != "$base" ]; then
        echo "FAIL: $base.sh — --job-name=$job_name does not match filename stem"
        fail=1
    fi
    if [[ "$output" != *"/%x-%j.out" ]]; then
        echo "FAIL: $base.sh — --output=$output does not end in /%x-%j.out"
        fail=1
    fi
    if [[ "$error" != *"/%x-%j.err" ]]; then
        echo "FAIL: $base.sh — --error=$error does not end in /%x-%j.err"
        fail=1
    fi
done

if [ "$fail" -eq 0 ]; then
    echo "All SLURM scripts follow the logging convention."
fi
exit $fail

#!/usr/bin/env bash
# Create every log directory referenced by #SBATCH --output/--error lines in
# this directory's scripts. Run once after cloning, and again any time log/
# is wiped (it's gitignored) — see LOGGING.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

grep -hoE '#SBATCH[[:space:]]+--(output|error)=\S+' "${SCRIPT_DIR}"/*.sh \
    | sed -E 's/^#SBATCH[[:space:]]+--(output|error)=//' \
    | sed -E 's/%[xjN]//g' \
    | xargs -n1 dirname \
    | sort -u \
    | while read -r dir; do
        mkdir -p "$dir"
        echo "ensured: $dir"
    done

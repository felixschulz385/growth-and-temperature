#!/usr/bin/env bash
# Point this clone at the repo-managed hooks in .githooks/.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath .githooks
echo "core.hooksPath -> .githooks (pre-commit notebook-output check active)"

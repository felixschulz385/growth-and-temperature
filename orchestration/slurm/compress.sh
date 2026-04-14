#!/bin/bash
#SBATCH --job-name=compress-coldstore
#SBATCH --output=./log/compress/%x-%j.log
#SBATCH --error=./log/compress/%x-%j.err
#SBATCH --partition=scicore
#SBATCH --time=3-00:00:00
#SBATCH --qos=1week
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch $0 /path/to/dataset_root"
    exit 1
fi

DATASET_ROOT="${1%/}"
RAW_ROOT="${DATASET_ROOT}/raw"
COLDSTORE_ROOT="${DATASET_ROOT}/coldstore"
DATASET_NAME="$(basename "$DATASET_ROOT")"

if [ ! -d "$RAW_ROOT" ]; then
    echo "Error: input root not found: $RAW_ROOT"
    exit 1
fi

PROJECT_ROOT="/scicore/home/meiera/schulz0022/projects/growth-and-temperature"
PYTHON_BIN="/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/gnt/bin/python"
SCRIPT_PATH="${PROJECT_ROOT}/scripts/compress_agent.py"

cd "$PROJECT_ROOT"
mkdir -p "./log/compress/${DATASET_NAME}"

TMPDIR="/scratch/schulz0022/compress_${DATASET_NAME}_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export TMPDIR

echo "Dataset root: $DATASET_ROOT"
echo "Input root:   $RAW_ROOT"
echo "Output root:  $COLDSTORE_ROOT"
echo "Python:       $PYTHON_BIN"

"$PYTHON_BIN" "$SCRIPT_PATH" \
    --input-root "$RAW_ROOT" \
    --output-root "$COLDSTORE_ROOT" \
    --jobs 1 \
    --level 19 \
    --checksum \
    --delete-source \
    --delete-source-existing \
    --log-file "./log/compress/${DATASET_NAME}/compress-${SLURM_JOB_ID}.log"

rm -rf "$TMPDIR"
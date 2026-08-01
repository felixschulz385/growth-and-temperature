#!/bin/bash
#SBATCH --job-name=validate-hard-gate-gadm
#SBATCH --output=./log/maintenance/hard_gate/%x-%j.out
#SBATCH --error=./log/maintenance/hard_gate/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Hard-gate pilot run: docs/design/09-integrated-pipeline.md step 9's
# validation gate, "vector" archetype, source=gadm (chosen per §5's migration
# table: gadm "seeds common/raster/rasterize.py" and every other vector/
# tabular-join/point source (osm siblings aside) declares REQUIRES on it, so
# it needs to be equivalence-checked before any of them can be).
#
# What this proves: that OLD code (src/data/preprocess/sources/misc.py's
# `_process_gadm_target`/`_rasterize_gadm_target`, subsource=gadm,
# stage=vector/spatial) and NEW code (src/data/sources/misc/gadm.py, step=
# prepare/grid) produce equivalent GeoPackage (PREPARE) and Zarr (GRID)
# output for the same input, via scripts/compare_step_output.py. Nothing
# else in this migration's hard gate is exercised here -- MODIS streaming,
# tabular-join (country_classifications), and point (plad/berman_mining/
# snl_mining) still need their own pilot before step 10 (cutover) can start;
# osm is the same vector archetype as this script but not yet piloted either.
#
# Safety: everything below runs against an ISOLATED scratch data_root, never
# against $DATA_NOBACKUP -- same pattern as validate-hard-gate-acag.sh, same
# reason (old code's stage-based target generation has no per-run scoping to
# rely on instead).
#
# Data required:
#   1. The one already-fetched raw GADM zip under
#      $DATA_NOBACKUP/misc/gadm/gadm_410-levels.zip -- copied into the
#      scratch root, never read from twice, never written to. Falls back to
#      fetching it directly from GADM's public URL (no auth) if missing,
#      same egress caveat as validate-hard-gate-acag.sh.
#   2. GRID's target grid -- see validate-hard-gate-acag.sh's header comment
#      item 2 for the full explanation of what this pickle is and why it's
#      required. This script crops it via scripts/crop_geobox_pickle.py too,
#      but -- unlike ACAG's globally-continuous PM2.5 field -- GADM's content
#      (country/subdivision polygons) is spatially sparse, so the crop is
#      centered on a real place (--lon/--lat, default central Europe) rather
#      than the grid's own pixel midpoint (which for a VIIRS-derived
#      equirectangular grid is lon=0/lat=0, the Gulf of Guinea -- ocean, no
#      GADM content to rasterize, would make this pilot pass vacuously).
#
# Usage:
#   sbatch orchestration/slurm/validate-hard-gate-gadm.sh [WINDOW_PX] [LON] [LAT]
#   (WINDOW_PX defaults to 300, LON/LAT default to 10.0/50.0 -- central
#   Europe, chosen for dense country + subdivision boundaries in a small
#   window so both ADM_0 and ADM_1 rasterization actually get exercised.)

set -euo pipefail

PROJECT_ROOT="/scicore/home/meiera/schulz0022/projects/growth-and-temperature"
PYTHON_BIN="/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/src/bin/python"
WINDOW_PX="${1:-300}"
LON="${2:-10.0}"
LAT="${3:-50.0}"

cd "$PROJECT_ROOT"
mkdir -p "./log/maintenance/hard_gate"

LOG_FILE="./log/maintenance/hard_gate/validate-hard-gate-gadm-${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate src

DATA_NOBACKUP="${DATA_NOBACKUP:-$PROJECT_ROOT/data_nobackup}"
TEST_ROOT="/scratch/schulz0022/hard_gate_gadm_${SLURM_JOB_ID}"
TEST_CONFIG="${TEST_ROOT}/config.yaml"
GADM_URL="https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-levels.zip"
# Both OLD (misc.py::_resolve_source_file_path) and NEW (layout.raw_root())
# unconditionally insert "raw" between data_path and subfolder -- confirmed
# by reading both, not assumed.
RAW_REL="gadm/gadm_410-levels.zip"
PROD_RAW="${DATA_NOBACKUP}/misc/raw/${RAW_REL}"
TEST_RAW="${TEST_ROOT}/misc/raw/${RAW_REL}"

echo "$(date -Is): Hard-gate pilot -- source=gadm window_px=$WINDOW_PX lon=$LON lat=$LAT"
echo "Job ID:     $SLURM_JOB_ID"
echo "Test root:  $TEST_ROOT  (isolated -- production data_nobackup is never written to)"
echo "Log file:   $LOG_FILE"

mkdir -p "$(dirname "$TEST_RAW")" "${TEST_ROOT}/hpc_data_index"

# --- REQUIRED: stage the one raw GADM zip this run needs -------------------
if [ -f "$PROD_RAW" ]; then
    echo "$(date -Is): copying already-fetched raw file from production"
    cp "$PROD_RAW" "$TEST_RAW"
else
    echo "$(date -Is): WARNING -- raw file not found at $PROD_RAW"
    echo "  Falling back to a direct fetch from GADM's public URL."
    echo "  This only works if this node has outbound internet egress."
    curl -L -o "$TEST_RAW" "$GADM_URL"
fi

# --- REQUIRED: a parquet index marking that one file "completed" -----------
# Only OLD code needs this (src/data/preprocess/sources/misc.py::
# get_preprocessing_targets reads it unconditionally); NEW code's PREPARE
# plan (src/data/sources/misc/gadm.py::_plan_prepare) only falls back to the
# index if the raw file itself isn't already on disk, which it is here --
# built anyway so both codepaths have exactly what they each expect, rather
# than relying on that asymmetry.
"$PYTHON_BIN" - "$RAW_REL" "${TEST_ROOT}/hpc_data_index/parquet_misc.parquet" <<'PYEOF'
import sys
import pandas as pd

rel_path, out_path = sys.argv[1], sys.argv[2]
pd.DataFrame([{"relative_path": rel_path, "status_category": "completed"}]).to_parquet(out_path)
print(f"wrote index -> {out_path}")
PYEOF

# --- REQUIRED: seed a CROPPED copy of the shared VIIRS-derived target
# geobox cache, centered on real GADM content -- see the header comment.
PROD_GEOBOX="${DATA_NOBACKUP}/misc/processed/stage_1/misc/viirs_geobox.pkl"
TEST_GEOBOX_DIR="${TEST_ROOT}/misc/processed/stage_1/misc"
mkdir -p "$TEST_GEOBOX_DIR"
if [ -f "$PROD_GEOBOX" ]; then
    "$PYTHON_BIN" scripts/crop_geobox_pickle.py \
        "$PROD_GEOBOX" "${TEST_GEOBOX_DIR}/viirs_geobox.pkl" \
        --window-px "$WINDOW_PX" --lon "$LON" --lat "$LAT"
    echo "$(date -Is): seeded cropped target geobox cache from production"
else
    echo "$(date -Is): ERROR -- no cached target geobox at $PROD_GEOBOX"
    echo "  Run 'pipeline run --source eog_viirs --step grid' once for real first,"
    echo "  or copy a viirs_geobox.pkl from wherever GRID has run before."
    exit 2
fi

# --- self-contained test config: isolated data_root, no HPC remote ---------
# Two source blocks for the SAME on-disk data: `misc` (OLD's nested-subsource
# shape) and `gadm` (NEW's standalone-package shape, defaults for url/name
# already match GADM_URL/RAW_REL so they're omitted here) -- see
# docs/design/09-integrated-pipeline.md §7 for why the shapes differ.
cat > "$TEST_CONFIG" <<EOF
paths:
  data_root: "${TEST_ROOT}"
  local_index_dir: "${TEST_ROOT}/hpc_data_index"
remote:
  ssh_target: ""
  key_file: ""
sources:
  misc:
    type: "misc"
    data_path: "misc"
    sources:
      gadm:
        url: "${GADM_URL}"
        name: "gadm_410-levels.zip"
        subfolder: "gadm"
  gadm:
    type: "gadm"
EOF

run_old() {
    local stage="$1"
    echo "$(date -Is): OLD  preprocess run --source misc --subsource gadm --stage $stage"
    "$PYTHON_BIN" run.py preprocess run --config "$TEST_CONFIG" --source misc --subsource gadm \
        --stage "$stage" --override \
        --dask-threads "$SLURM_CPUS_PER_TASK" --dask-memory-limit 4GiB \
        --temp-dir "${TEST_ROOT}/dask_tmp"
}

run_new() {
    local step="$1"
    echo "$(date -Is): NEW  pipeline run --source gadm --step $step"
    "$PYTHON_BIN" run.py pipeline run --config "$TEST_CONFIG" --source gadm \
        --step "$step" --override \
        --dask-threads "$SLURM_CPUS_PER_TASK" --dask-memory-limit 4GiB \
        --temp-dir "${TEST_ROOT}/dask_tmp"
}

# --- 1. OLD code produces the reference artefacts ---------------------------
run_old vector
run_old spatial

PREPARE_DIR="${TEST_ROOT}/misc/processed/stage_1/gadm"
GRID_DIR="${TEST_ROOT}/misc/processed/stage_2/gadm"
PREPARE_OLD_REF="${PREPARE_DIR}.old_reference"
GRID_OLD_REF="${GRID_DIR}.old_reference"

echo "$(date -Is): snapshotting OLD output before NEW code overwrites the same paths"
cp -a "$PREPARE_DIR" "$PREPARE_OLD_REF"
cp -a "$GRID_DIR" "$GRID_OLD_REF"

# --- 2. NEW code overwrites the same target paths ---------------------------
run_new prepare
run_new grid

# --- 3. Compare ---------------------------------------------------------------
# PREPARE writes one gadm_level<N>_simplified.gpkg per ADM level found in the
# source zip (however many that is -- not hardcoded, both old and new derive
# it from gpd.list_layers() at runtime) -- compare every one found in the OLD
# reference dir, plus flag any whose NEW counterpart is missing entirely.
echo "$(date -Is): comparing PREPARE output"
PREPARE_STATUS=0
shopt -s nullglob
old_level_files=("${PREPARE_OLD_REF}"/gadm_level*_simplified.gpkg)
if [ "${#old_level_files[@]}" -eq 0 ]; then
    echo "  NOT_EQUIVALENT: OLD produced no gadm_level*_simplified.gpkg files at all"
    PREPARE_STATUS=1
fi
for old_file in "${old_level_files[@]}"; do
    fname="$(basename "$old_file")"
    new_file="${PREPARE_DIR}/${fname}"
    "$PYTHON_BIN" scripts/compare_step_output.py "$old_file" "$new_file" || PREPARE_STATUS=$?
done

echo "$(date -Is): comparing GRID output"
GRID_STATUS=0
"$PYTHON_BIN" scripts/compare_step_output.py \
    "${GRID_OLD_REF}/countries_grid.zarr" "${GRID_DIR}/countries_grid.zarr" || GRID_STATUS=$?

echo "=============================================================="
echo "HARD-GATE PILOT RESULT -- source=gadm window_px=$WINDOW_PX lon=$LON lat=$LAT"
echo "  PREPARE (vector  vs prepare): $([ $PREPARE_STATUS -eq 0 ] && echo EQUIVALENT || echo NOT_EQUIVALENT)"
echo "  GRID    (spatial vs grid):    $([ $GRID_STATUS -eq 0 ] && echo EQUIVALENT || echo NOT_EQUIVALENT)"
echo "  Test root (not auto-deleted, inspect or clean up manually): $TEST_ROOT"
echo "=============================================================="

if [ $PREPARE_STATUS -ne 0 ] || [ $GRID_STATUS -ne 0 ]; then
    exit 1
fi
exit 0

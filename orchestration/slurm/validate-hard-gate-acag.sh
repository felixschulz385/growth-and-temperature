#!/bin/bash
#SBATCH --job-name=validate-hard-gate-acag
#SBATCH --output=./log/maintenance/hard_gate/%x-%j.out
#SBATCH --error=./log/maintenance/hard_gate/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --qos=6hours
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Hard-gate pilot run: docs/design/09-integrated-pipeline.md step 9's
# validation gate, "bulk-raster-with-composite" archetype, source=acag
# (chosen because it's the reference migration -- see §5's migration table).
#
# What this proves: that OLD code (src/data/preprocess/sources/acag.py,
# stage=annual/spatial) and NEW code (src/data/sources/acag.py, step=
# prepare/grid) produce byte-equivalent Zarr output for the same input,
# via scripts/compare_step_output.py. Nothing else in this migration's hard
# gate is exercised here -- MODIS streaming, vector (osm/gadm), tabular-join
# (country_classifications), and point (plad/berman_mining/snl_mining) still
# need their own pilot before step 10 (cutover) can start.
#
# Safety: everything below runs against an ISOLATED scratch data_root, never
# against $DATA_NOBACKUP. Two reasons this matters, both found while writing
# this script (not hypothetical):
#   1. Old code's GRID/"spatial" stage ignores --year entirely (see
#      ACAGPreprocessor._gen_spatial_targets) and always reprocesses every
#      annual zarr it finds in stage_1 into ONE shared multi-year Zarr store.
#      Point this at production and a "--year 2020" run would silently
#      overwrite all 26 years of already-computed reprojected PM2.5 with a
#      1-year subset.
#   2. New code's GRID *does* filter by --years (AcagSource._plan_grid) --
#      a real, deliberate behavioural difference from old code, flagged here
#      rather than silently relied upon. It's harmless in this script only
#      because the scratch stage_1 never contains more than one year to begin
#      with, so the filter is a no-op either way.
#
# Data required (verified by actually running this end-to-end locally against
# real fetched data before handing this script off -- see the two REQUIRED
# blocks below plus the geobox note):
#   1. The one already-fetched raw ACAG file for $YEAR under
#      $DATA_NOBACKUP/acag/pm25/raw/GL/Annual/ -- copied into the scratch
#      root, never read from twice, never written to. If it isn't there yet,
#      this script falls back to fetching it directly from ACAG's public Box
#      link (no auth needed), which only works if this node has outbound
#      internet egress. Production FETCH for every source in this repo has
#      always run manually from a machine with internet (see jobs.yaml's
#      header comment: "no download-family SLURM script has ever existed"),
#      so if this node is a compute node without egress, pre-stage the file
#      yourself, e.g. from a login node:
#        mkdir -p $DATA_NOBACKUP/acag/pm25/raw/GL/Annual
#        curl -L -A "Mozilla/5.0" -o $DATA_NOBACKUP/acag/pm25/raw/GL/Annual/V6GL02.04.CNNPM25.GL.${YEAR}01-${YEAR}12.nc \
#          "https://wustl.app.box.com/index.php?rm=box_download_shared_file&shared_name=y143mciw7jz7ft2qe3hccjw65m3xe8f2&file_id=f_<ID>"
#      (file IDs are in src/data/sources/acag.py::AcagSource.KNOWN_FILES)
#   2. GRID's target grid, for BOTH old and new code, is bootstrapped once
#      from an actual local EOG VIIRS raster via
#      src/data/common/geobox/geobox.py::get_or_create_geobox() -- untouched
#      by this migration, shared by old and new SpatialProcessor alike -- and
#      cached as a pickle at <data_root>/misc/processed/stage_1/misc/
#      viirs_geobox.pkl. This is a real, previously-undocumented dependency
#      found while dry-running this exact script: an isolated scratch
#      data_root with no VIIRS history at all makes GRID fail with
#      "Parquet index not found: .../parquet_eog_viirs.parquet" the first
#      time, on ANY raster source, not just acag. On scicore this pickle
#      almost certainly already exists under $DATA_NOBACKUP (every raster
#      GRID job ever run there created or reused it); if it's missing there
#      too, generate it once for real via `data run --source eog_viirs
#      --step grid` before this script.
#
#      This script does NOT copy that pickle in verbatim -- it crops it to a
#      small WINDOW_PX x WINDOW_PX window (centered) via
#      scripts/crop_geobox_pickle.py before pickling the result into the
#      scratch root instead (a small shared tool, since
#      validate-hard-gate-gadm.sh needs the same crop with a real-content
#      lon/lat center, not this script's centered default -- see that
#      script). `GeoBox` supports
#      plain numpy-style slicing (`gbox[y0:y1, x0:x1]`) and the result keeps
#      the same CRS/resolution/affine alignment, just fewer pixels -- so
#      `get_or_create_geobox()` (which only needs `.shape`/`.affine`/`.crs`
#      off whatever it unpickles) reprojects onto a tiny grid instead of the
#      full ~86,401x33,601px global one, with no changes to SpatialProcessor
#      or either acag.py needed. This matters because, post-migration,
#      SpatialProcessor (src/data/common/raster/spatial.py) is now the same
#      function under both OLD and NEW code paths -- this step was never
#      testing whether reprojection is *correct* at global scale (unchanged
#      either way), only whether OLD's and NEW's surrounding plumbing calls
#      it equivalently, which a small window proves just as well. Confirmed
#      by running the full-grid version first: it took 50+ minutes per code
#      path and repeatedly hit dask "Unmanaged memory use is high"/worker-
#      pausing warnings under this job's 2-workers-of-2-threads/12GiB sizing
#      -- pure overhead the window eliminates, not a data-correctness signal.
#
# Also found while dry-running this locally (noted here so it isn't
# mysterious if hit again): a from-scratch dask LocalCluster with the default
# (CPU-count-derived) worker count can exhaust local socket buffers under a
# constrained network namespace (`OSError: [Errno 55] No buffer space
# available`, workers dying repeatedly). Unrelated to pipeline correctness --
# scicore compute nodes shouldn't hit this -- but if this job's dask step
# fails with ENOBUFS/worker-churn, rerun with a smaller --dask-threads.
#
# Usage:
#   sbatch orchestration/slurm/validate-hard-gate-acag.sh [YEAR] [WINDOW_PX]
#   (YEAR defaults to 2020 -- picked because it matches ACAG's "GL"-prefixed
#   naming convention; 2000's entry is anomalously "EU"-tagged, avoid it.
#   WINDOW_PX defaults to 300 -- centered, so ~150px in from the grid's
#   midpoint in every direction; large enough that ACAG's global PM2.5
#   coverage gives compare_step_output.py real numeric variation to check,
#   small enough that the whole run finishes in well under a minute.)

set -euo pipefail

PROJECT_ROOT="/scicore/home/meiera/schulz0022/projects/growth-and-temperature"
PYTHON_BIN="/scicore/home/meiera/schulz0022/miniforge-pypy3/envs/src/bin/python"
YEAR="${1:-2020}"
WINDOW_PX="${2:-300}"
SOURCE="acag"

cd "$PROJECT_ROOT"
mkdir -p "./log/maintenance/hard_gate"

LOG_FILE="./log/maintenance/hard_gate/validate-hard-gate-acag-${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

eval "$(/scicore/home/meiera/schulz0022/miniforge-pypy3/bin/conda shell.bash hook)"
conda activate src

DATA_NOBACKUP="${DATA_NOBACKUP:-$PROJECT_ROOT/data_nobackup}"
TEST_ROOT="/scratch/schulz0022/hard_gate_acag_${SLURM_JOB_ID}"
TEST_CONFIG="${TEST_ROOT}/config.yaml"
RAW_REL="GL/Annual/V6GL02.04.CNNPM25.GL.${YEAR}01-${YEAR}12.nc"
PROD_RAW="${DATA_NOBACKUP}/acag/pm25/raw/${RAW_REL}"
TEST_RAW="${TEST_ROOT}/acag/pm25/raw/${RAW_REL}"

echo "$(date -Is): Hard-gate pilot -- source=$SOURCE year=$YEAR window_px=$WINDOW_PX"
echo "Job ID:     $SLURM_JOB_ID"
echo "Test root:  $TEST_ROOT  (isolated -- production data_nobackup is never written to)"
echo "Log file:   $LOG_FILE"

mkdir -p "$(dirname "$TEST_RAW")" "${TEST_ROOT}/hpc_data_index"

# --- REQUIRED: stage the one raw file this run needs -----------------------
if [ -f "$PROD_RAW" ]; then
    echo "$(date -Is): copying already-fetched raw file from production"
    cp "$PROD_RAW" "$TEST_RAW"
else
    echo "$(date -Is): WARNING -- raw file not found at $PROD_RAW"
    echo "  Falling back to a direct fetch from ACAG's public Box link."
    echo "  This only works if this node has outbound internet egress."
    "$PYTHON_BIN" - "$YEAR" "$TEST_RAW" <<'PYEOF'
import sys
from src.data.sources.acag import AcagSource

year, out_path = sys.argv[1], sys.argv[2]
rel_path = f"GL/Annual/V6GL02.04.CNNPM25.GL.{year}01-{year}12.nc"
file_id = dict(AcagSource.KNOWN_FILES)[rel_path]
url = (
    "https://wustl.app.box.com/index.php"
    f"?rm=box_download_shared_file&shared_name=y143mciw7jz7ft2qe3hccjw65m3xe8f2&file_id=f_{file_id}"
)
import requests
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
resp = requests.get(url, headers=headers, timeout=120)
resp.raise_for_status()
with open(out_path, "wb") as fh:
    fh.write(resp.content)
print(f"fetched {len(resp.content)} bytes -> {out_path}")
PYEOF
fi

# --- REQUIRED: a parquet index marking that one file "completed" -----------
# Both old and new PREPARE code only read relative_path + status_category
# from this file (src/data/preprocess/sources/acag.py::get_preprocessing_targets,
# src/data/sources/acag.py::_plan_prepare) -- built directly here rather than
# through UnifiedDataIndex (1,294 untested lines this migration deliberately
# left untouched, docs/design/09-integrated-pipeline.md §4) to keep this
# pilot's moving parts to exactly what's being validated.
"$PYTHON_BIN" - "$RAW_REL" "${TEST_ROOT}/hpc_data_index/parquet_acag_pm25.parquet" <<'PYEOF'
import sys
import pandas as pd

rel_path, out_path = sys.argv[1], sys.argv[2]
pd.DataFrame([{"relative_path": rel_path, "status_category": "completed"}]).to_parquet(out_path)
print(f"wrote index -> {out_path}")
PYEOF

# --- REQUIRED: seed a CROPPED copy of the shared VIIRS-derived target
# geobox cache -- see the header comment's item 2 for why cropping is safe
# and why full-grid execution here was pure overhead, not a stronger check.
PROD_GEOBOX="${DATA_NOBACKUP}/misc/processed/stage_1/misc/viirs_geobox.pkl"
TEST_GEOBOX_DIR="${TEST_ROOT}/misc/processed/stage_1/misc"
mkdir -p "$TEST_GEOBOX_DIR"
if [ -f "$PROD_GEOBOX" ]; then
    "$PYTHON_BIN" scripts/crop_geobox_pickle.py \
        "$PROD_GEOBOX" "${TEST_GEOBOX_DIR}/viirs_geobox.pkl" --window-px "$WINDOW_PX"
    echo "$(date -Is): seeded cropped target geobox cache from production"
else
    echo "$(date -Is): ERROR -- no cached target geobox at $PROD_GEOBOX"
    echo "  and no EOG VIIRS raw file + index in this scratch root to build one from."
    echo "  Run 'data run --source eog_viirs --step grid' once for real first,"
    echo "  or copy a viirs_geobox.pkl from wherever GRID has run before."
    exit 2
fi

# --- self-contained test config: isolated data_root, no HPC remote ---------
cat > "$TEST_CONFIG" <<EOF
paths:
  data_root: "${TEST_ROOT}"
  local_index_dir: "${TEST_ROOT}/hpc_data_index"
remote:
  ssh_target: ""
  key_file: ""
sources:
  acag:
    type: "acag"
    data_path: "acag/pm25"
    year_range: [${YEAR}, ${YEAR}]
EOF

run_old() {
    local stage="$1"
    echo "$(date -Is): OLD  preprocess run --source $SOURCE --stage $stage --year $YEAR"
    # Without an explicit --dask-threads, dask's LocalCluster auto-detects
    # os.cpu_count() for the WHOLE NODE, not this job's --cpus-per-task
    # allocation -- found by actually running this on scicore, where it
    # spawned 28+ worker processes under a 4-cpu/32G job. Constrain it to
    # match run_new()'s sizing so both codepaths run under the same
    # resources and neither blows past what SBATCH actually reserved.
    "$PYTHON_BIN" run.py preprocess run --config "$TEST_CONFIG" --source "$SOURCE" \
        --stage "$stage" --year "$YEAR" --override \
        --dask-threads "$SLURM_CPUS_PER_TASK" --dask-memory-limit 4GiB \
        --temp-dir "${TEST_ROOT}/dask_tmp"
}

run_new() {
    local step="$1"
    echo "$(date -Is): NEW  data run --source $SOURCE --step $step --years $YEAR $YEAR"
    "$PYTHON_BIN" run.py data run --config "$TEST_CONFIG" --source "$SOURCE" \
        --step "$step" --years "$YEAR" "$YEAR" --override \
        --dask-threads "$SLURM_CPUS_PER_TASK" --dask-memory-limit 4GiB \
        --temp-dir "${TEST_ROOT}/dask_tmp"
}

# --- 1. OLD code produces the reference artefacts ---------------------------
run_old annual
run_old spatial

PREPARE_DIR="${TEST_ROOT}/acag/pm25/processed/stage_1"
GRID_DIR="${TEST_ROOT}/acag/pm25/processed/stage_2"
PREPARE_OLD_REF="${PREPARE_DIR}.old_reference"
GRID_OLD_REF="${GRID_DIR}.old_reference"

echo "$(date -Is): snapshotting OLD output before NEW code overwrites the same paths"
cp -a "$PREPARE_DIR" "$PREPARE_OLD_REF"
cp -a "$GRID_DIR" "$GRID_OLD_REF"

# --- 2. NEW code overwrites the same target paths ---------------------------
run_new prepare
run_new grid

# --- 3. Compare ---------------------------------------------------------------
echo "$(date -Is): comparing PREPARE output"
PREPARE_STATUS=0
"$PYTHON_BIN" scripts/compare_step_output.py \
    "${PREPARE_OLD_REF}/${YEAR}.zarr" "${PREPARE_DIR}/${YEAR}.zarr" || PREPARE_STATUS=$?

echo "$(date -Is): comparing GRID output"
GRID_STATUS=0
"$PYTHON_BIN" scripts/compare_step_output.py \
    "${GRID_OLD_REF}/acag_pm25_timeseries_reprojected.zarr" \
    "${GRID_DIR}/acag_pm25_timeseries_reprojected.zarr" || GRID_STATUS=$?

echo "=============================================================="
echo "HARD-GATE PILOT RESULT -- source=$SOURCE year=$YEAR window_px=$WINDOW_PX"
echo "  PREPARE (annual  vs prepare): $([ $PREPARE_STATUS -eq 0 ] && echo EQUIVALENT || echo NOT_EQUIVALENT)"
echo "  GRID    (spatial vs grid):    $([ $GRID_STATUS -eq 0 ] && echo EQUIVALENT || echo NOT_EQUIVALENT)"
echo "  Test root (not auto-deleted, inspect or clean up manually): $TEST_ROOT"
echo "=============================================================="

if [ $PREPARE_STATUS -ne 0 ] || [ $GRID_STATUS -ne 0 ]; then
    exit 1
fi
exit 0

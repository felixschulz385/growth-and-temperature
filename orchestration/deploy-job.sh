#!/bin/bash

# Usage info
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -m, --mode          Processing mode: 'download' or 'preprocess'"
  echo "  -d, --dataset       Dataset name (e.g., 'glass-avhrr', 'glass-lst-modis')"
  echo "  -c, --config        Path to config JSON file with parameters"
  echo "  -n, --name          Job name prefix (optional, defaults to dataset-mode)"
  echo "  --cpu-request       CPU request value (default: depends on mode)"
  echo "  --cpu-limit         CPU limit value (default: depends on mode)"
  echo "  --memory-request    Memory request value (default: depends on mode)"
  echo "  --memory-limit      Memory limit value (default: depends on mode)"
  echo "  --concurrent        Max concurrent downloads (download mode only)"
  echo "  --queue-size        Max queue size (download mode only)"
  echo "  -h, --help          Show this help message"
  exit 1
}

# Default values
MODE=""
DATASET=""
CONFIG_FILE=""
JOB_NAME_PREFIX=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -m|--mode)
      MODE="$2"
      shift 2
      ;;
    -d|--dataset)
      DATASET="$2"
      shift 2
      ;;
    -c|--config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    -n|--name)
      JOB_NAME_PREFIX="$2"
      shift 2
      ;;
    --cpu-request)
      CPU_REQUEST="$2"
      shift 2
      ;;
    --cpu-limit)
      CPU_LIMIT="$2"
      shift 2
      ;;
    --memory-request)
      MEMORY_REQUEST="$2"
      shift 2
      ;;
    --memory-limit)
      MEMORY_LIMIT="$2"
      shift 2
      ;;
    --concurrent)
      MAX_CONCURRENT_DOWNLOADS="$2"
      shift 2
      ;;
    --queue-size)
      MAX_QUEUE_SIZE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Validate required arguments
if [[ -z "$MODE" || -z "$DATASET" || -z "$CONFIG_FILE" ]]; then
  echo "Error: Missing required arguments"
  usage
fi

# Set job name prefix if not provided
if [[ -z "$JOB_NAME_PREFIX" ]]; then
  JOB_NAME_PREFIX="${DATASET}-${MODE}"
fi

# Set defaults based on mode
if [[ "$MODE" == "download" ]]; then
  CPU_REQUEST=${CPU_REQUEST:-"50m"}
  CPU_LIMIT=${CPU_LIMIT:-"500m"}
  MEMORY_REQUEST=${MEMORY_REQUEST:-"512Mi"}
  MEMORY_LIMIT=${MEMORY_LIMIT:-"1024Mi"}
  MAX_CONCURRENT_DOWNLOADS=${MAX_CONCURRENT_DOWNLOADS:-"2"}
  MAX_QUEUE_SIZE=${MAX_QUEUE_SIZE:-"8"}
elif [[ "$MODE" == "preprocess" ]]; then
  CPU_REQUEST=${CPU_REQUEST:-"8"}
  CPU_LIMIT=${CPU_LIMIT:-"8"}
  MEMORY_REQUEST=${MEMORY_REQUEST:-"16Gi"}
  MEMORY_LIMIT=${MEMORY_LIMIT:-"16Gi"}
else
  echo "Error: Mode must be either 'download' or 'preprocess'"
  usage
fi

# Create unique job and config map names using timestamp
TIMESTAMP=$(date +%s)
JOB_NAME="${JOB_NAME_PREFIX}-${TIMESTAMP}"
CONFIG_MAP_NAME="${JOB_NAME_PREFIX}-config-${TIMESTAMP}"

# Extract the JSON content from the config file
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Error: Config file not found: $CONFIG_FILE"
  exit 1
fi

# Extract parameters from the config file, skipping the first and last line which are braces
CONFIG_PARAMETERS=$(sed '1d;$d' "$CONFIG_FILE")

# Create a temporary configmap file
TEMP_CONFIG_MAP="/tmp/${CONFIG_MAP_NAME}.yaml"
cp "$(dirname "$0")/config-template.yaml" "$TEMP_CONFIG_MAP"

# Replace variables in the configmap template
sed -i '' "s/\${CONFIG_MAP_NAME}/$CONFIG_MAP_NAME/g" "$TEMP_CONFIG_MAP"
sed -i '' "s/\${PROCESSING_MODE}/$MODE/g" "$TEMP_CONFIG_MAP"
sed -i '' "s/\${DATASET_NAME}/$DATASET/g" "$TEMP_CONFIG_MAP"
# Replace the parameters section in the template with the actual parameters from the config file
sed -i '' "s|\${CONFIG_PARAMETERS}|$CONFIG_PARAMETERS|g" "$TEMP_CONFIG_MAP"

# Create a temporary job file
TEMP_JOB_FILE="/tmp/${JOB_NAME}.yaml"
cp "$(dirname "$0")/k8s-job-unified.yaml" "$TEMP_JOB_FILE"

# Replace variables in the job template
sed -i '' "s/\${DATASET_NAME}/$DATASET/g" "$TEMP_JOB_FILE"
sed -i '' "s/name: data-processing-job/name: $JOB_NAME/g" "$TEMP_JOB_FILE"
sed -i '' "s/\${CPU_REQUEST}/$CPU_REQUEST/g" "$TEMP_JOB_FILE"
sed -i '' "s/\${CPU_LIMIT}/$CPU_LIMIT/g" "$TEMP_JOB_FILE"
sed -i '' "s/\${MEMORY_REQUEST}/$MEMORY_REQUEST/g" "$TEMP_JOB_FILE"
sed -i '' "s/\${MEMORY_LIMIT}/$MEMORY_LIMIT/g" "$TEMP_JOB_FILE"
sed -i '' "s/\${PROCESSING_MODE}/$MODE/g" "$TEMP_JOB_FILE"
sed -i '' "s/\${CONFIG_MAP_NAME}/$CONFIG_MAP_NAME/g" "$TEMP_JOB_FILE"
sed -i '' "s/\${MAX_CONCURRENT_DOWNLOADS}/$MAX_CONCURRENT_DOWNLOADS/g" "$TEMP_JOB_FILE"
sed -i '' "s/\${MAX_QUEUE_SIZE}/$MAX_QUEUE_SIZE/g" "$TEMP_JOB_FILE"
# Set config file name to the fixed path in the ConfigMap
sed -i '' "s|\${CONFIG_FILE}|workflow-config.json|g" "$TEMP_JOB_FILE"

# Apply the configmap
echo "Creating ConfigMap $CONFIG_MAP_NAME..."
kubectl apply -f "$TEMP_CONFIG_MAP"

# Apply the job
echo "Deploying job $JOB_NAME..."
kubectl apply -f "$TEMP_JOB_FILE"

echo "Job deployed successfully!"
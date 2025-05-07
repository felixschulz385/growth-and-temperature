#!/bin/bash
set -e

CONFIG_FILE="$1"
echo "Reading config from: $CONFIG_FILE"

# Parse processing mode from config file
if [ -f "$CONFIG_FILE" ]; then
    PROCESSING_MODE=$(grep -o '"processing_mode": *"[^"]*"' "$CONFIG_FILE" | cut -d'"' -f4)
    echo "Detected processing mode from config: $PROCESSING_MODE"
fi

# If processing mode wasn't found in config, use environment variable
if [ -z "$PROCESSING_MODE" ]; then
    echo "No processing mode detected in config, using environment variable: $PROCESSING_MODE"
    if [ -z "$PROCESSING_MODE" ]; then
        echo "Error: PROCESSING_MODE not set in config file or environment"
        exit 1
    fi
fi

echo "Starting job in $PROCESSING_MODE mode..."

if [ "$PROCESSING_MODE" = "download" ]; then
    exec python /app/gnt/data/download/main.py "$CONFIG_FILE"
elif [ "$PROCESSING_MODE" = "preprocess" ]; then
    exec python /app/gnt/data/preprocess/main.py "$CONFIG_FILE" --debug
else
    echo "Error: PROCESSING_MODE must be 'download' or 'preprocess'"
    exit 1
fi
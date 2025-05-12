#!/bin/bash
set -e

CONFIG_FILE="$1"
echo "Reading config from: $CONFIG_FILE"

# Detect file extension to determine format
FILE_EXT="${CONFIG_FILE##*.}"
FILE_EXT_LOWER=$(echo "$FILE_EXT" | tr '[:upper:]' '[:lower:]')

# Parse processing mode from config file based on format (YAML or JSON)
if [ -f "$CONFIG_FILE" ]; then
    if [ "$FILE_EXT_LOWER" = "yaml" ] || [ "$FILE_EXT_LOWER" = "yml" ]; then
        # Parse YAML format
        if command -v yq >/dev/null 2>&1; then
            # If yq is available, use it
            PROCESSING_MODE=$(yq e '.processing_mode' "$CONFIG_FILE" 2>/dev/null)
        else
            # Simple grep-based YAML parsing
            PROCESSING_MODE=$(grep -E '^processing_mode:' "$CONFIG_FILE" | sed 's/^processing_mode:\s*//' | sed 's/^"\(.*\)"$/\1/' | sed "s/^'\(.*\)'$/\1/")
        fi
    else
        # Parse JSON format
        PROCESSING_MODE=$(grep -o '"processing_mode": *"[^"]*"' "$CONFIG_FILE" | cut -d'"' -f4)
    fi
    
    # Handle null, undefined, or empty value
    if [ "$PROCESSING_MODE" = "null" ] || [ "$PROCESSING_MODE" = "~" ] || [ -z "$PROCESSING_MODE" ]; then
        PROCESSING_MODE=""
    fi
    
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
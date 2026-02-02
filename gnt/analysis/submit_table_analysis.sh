#!/bin/bash

# Extract all model names for a specific table from the "Models in Tables" sheet in analysis.xlsx
# Usage: ./run_workflow.sh <table_id>

if [ -z "$1" ]; then
    echo "Error: Table ID required"
    echo "Usage: $0 <table_id>"
    exit 1
fi

TABLE_ID="$1"

models=($(python3 -c "
import pandas as pd
import os
import sys

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath('$0'))
project_root = os.path.join(script_dir, '..', '..')

# Read the Excel file
excel_path = os.path.join(project_root, 'orchestration', 'configs', 'analysis.xlsx')
df = pd.read_excel(excel_path, sheet_name='Models in Tables')

# Filter by table ID and sort by order
table_models = df[df['table'] == '$TABLE_ID'].sort_values('order')

if len(table_models) == 0:
    print(f'Error: No models found for table {sys.argv[1]}', file=sys.stderr)
    sys.exit(1)

# Extract model names
for model in table_models['model_name']:
    print(model)
" "$TABLE_ID"))

# Check if extraction was successful
if [ $? -ne 0 ]; then
    exit 1
fi

if [ ${#models[@]} -eq 0 ]; then
    echo "Error: No models found for table: $TABLE_ID"
    exit 1
fi

echo "Found ${#models[@]} models for table: $TABLE_ID"
echo "Submitting analysis jobs..."

# Run each model in order
for model in "${models[@]}"; do
    echo "  - Submitting: $model"
    sbatch /scicore/home/meiera/schulz0022/projects/growth-and-temperature/orchestration/slurm/analysis.sh "$model"
done

echo "All models submitted for table: $TABLE_ID"
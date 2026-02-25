#!/usr/bin/env python3
"""
Submit a single SLURM job to run all models for a specific table sequentially.

Usage:
    python submit_table_analysis.py <table_id> [--mem=<memory>] [--time=<time>] [--partition=<partition>]

Example:
    python submit_table_analysis.py table1
    python submit_table_analysis.py table2 --mem=64GB --time=04:00:00
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import pandas as pd
import tempfile


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_models_for_table(table_id: str) -> list:
    """
    Extract model names for a specific table from Excel.
    
    Args:
        table_id: Table ID to filter by
        
    Returns:
        List of model names sorted by order
    """
    project_root = get_project_root()
    excel_path = project_root / 'orchestration' / 'configs' / 'analysis.xlsx'
    
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    # Read the "Models in Tables" sheet
    df = pd.read_excel(excel_path, sheet_name='Models in Tables')
    
    # Filter by table ID and sort by order
    table_models = df[df['table'] == table_id].sort_values('order')
    
    if len(table_models) == 0:
        raise ValueError(f"No models found for table: {table_id}")
    
    # Extract model names
    models = table_models['model_name'].tolist()
    return models


def create_job_script(models: list, project_root: Path) -> str:
    """
    Create a SLURM job script that runs all models sequentially.
    
    Args:
        models: List of model names to run
        project_root: Path to project root
        
    Returns:
        Path to the temporary job script
    """
    script_content = f"""#!/bin/bash
#SBATCH --job-name=table-analysis
#SBATCH --output={project_root / 'log' / 'table-analysis-%j.log'}
#SBATCH --error={project_root / 'log' / 'table-analysis-%j.err'}

# Run all models sequentially
set -e  # Exit on first error

cd {project_root}

"""
    
    for i, model in enumerate(models, 1):
        script_content += f"""echo "Running model {i}/{len(models)}: {model}"
python run.py analysis --config orchestration/configs/analysis.xlsx --model {model} --output output/analysis
echo "Model {model} completed successfully"
echo ""

"""
    
    script_content += """echo "All models completed successfully!"
"""
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(script_content)
        return f.name


def submit_job(job_script: str, slurm_kwargs: dict) -> str:
    """
    Submit the job script to SLURM.
    
    Args:
        job_script: Path to job script
        slurm_kwargs: Dictionary of SLURM parameters (mem, time, partition, etc.)
        
    Returns:
        SLURM job ID
    """
    # Build sbatch command
    cmd = ['sbatch']
    
    # Add SLURM parameters
    if slurm_kwargs.get('mem'):
        cmd.extend(['--mem', slurm_kwargs['mem']])
    if slurm_kwargs.get('time'):
        cmd.extend(['--time', slurm_kwargs['time']])
    if slurm_kwargs.get('partition'):
        cmd.extend(['--partition', slurm_kwargs['partition']])
    if slurm_kwargs.get('cpus_per_task'):
        cmd.extend(['--cpus-per-task', str(slurm_kwargs['cpus_per_task'])])
    
    # Add job script
    cmd.append(job_script)
    
    # Submit job
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to submit job: {result.stderr}")
    
    # Extract job ID from output (e.g., "Submitted batch job 12345")
    job_id = result.stdout.strip().split()[-1]
    return job_id


def main():
    parser = argparse.ArgumentParser(
        description="Submit a single SLURM job to run all models for a table sequentially"
    )
    parser.add_argument('table_id', help='Table ID to run models for')
    parser.add_argument('--mem', default='64GB', help='Memory allocation (default: 64GB)')
    parser.add_argument('--time', default='04:00:00', help='Time limit (default: 04:00:00)')
    parser.add_argument('--partition', help='SLURM partition (optional)')
    parser.add_argument('--cpus-per-task', type=int, help='CPUs per task (optional)')
    
    args = parser.parse_args()
    
    try:
        # Get models for table
        print(f"Fetching models for table: {args.table_id}")
        models = get_models_for_table(args.table_id)
        print(f"Found {len(models)} models:")
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
        
        # Create job script
        project_root = get_project_root()
        print(f"\nCreating job script...")
        job_script = create_job_script(models, project_root)
        
        # Submit job
        print(f"Submitting job to SLURM...")
        slurm_kwargs = {
            'mem': args.mem,
            'time': args.time,
            'partition': args.partition,
            'cpus_per_task': args.cpus_per_task,
        }
        # Remove None values
        slurm_kwargs = {k: v for k, v in slurm_kwargs.items() if v is not None}
        
        job_id = submit_job(job_script, slurm_kwargs)
        print(f"\n✓ Job submitted successfully!")
        print(f"  Job ID: {job_id}")
        print(f"  Table: {args.table_id}")
        print(f"  Models: {len(models)}")
        print(f"  Memory: {args.mem}")
        print(f"  Time: {args.time}")
        
        # Clean up temporary script
        os.remove(job_script)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

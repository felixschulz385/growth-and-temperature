"""
Utility script to read analysis results JSON and print human-readable summary to stdout.

Usage:
    python print_results.py <path_to_results.json>
    python print_results.py <path_to_results_dir>  # finds latest results_*.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd


def format_results_summary(results: Dict[str, Any]) -> str:
    """Format results dictionary as human-readable text summary."""
    summary_lines = []
    
    summary_lines.append("=" * 80)
    summary_lines.append("ANALYSIS RESULTS SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    # Metadata section
    if 'metadata' in results:
        meta = results['metadata']
        summary_lines.append("METADATA")
        summary_lines.append("-" * 80)
        summary_lines.append(f"Analysis Type:  {meta.get('analysis_type', 'N/A')}")
        summary_lines.append(f"Specification:  {meta.get('spec_name', 'N/A')}")
        summary_lines.append(f"Description:    {meta.get('description', 'N/A')}")
        summary_lines.append(f"Timestamp:      {meta.get('timestamp', 'N/A')}")
        summary_lines.append(f"Formula:        {meta.get('formula', 'N/A')}")
        summary_lines.append(f"Data Source:    {meta.get('data_source', 'N/A')}")
        if meta.get('query'):
            summary_lines.append(f"Query/Filter:   {meta.get('query', 'N/A')}")
        summary_lines.append("")
    
    # Specification section
    if 'specification' in results:
        spec = results['specification']
        summary_lines.append("SPECIFICATION")
        summary_lines.append("-" * 80)
        
        if 'fe_method' in spec:
            summary_lines.append(f"Fixed Effects Method: {spec.get('fe_method', 'N/A')}")
        if 'n_bootstraps' in spec:
            summary_lines.append(f"Bootstrap Iterations: {spec.get('n_bootstraps', 0)}")
        if 'seed' in spec:
            summary_lines.append(f"Random Seed:          {spec.get('seed', 'N/A')}")
        if 'round_strata' in spec:
            summary_lines.append(f"Round Strata Decimals: {spec.get('round_strata', 'None')}")
        if 'se_type' in spec:
            summary_lines.append(f"Standard Errors:      {spec.get('se_type', 'N/A')}")
        if 'cluster_type' in spec:
            summary_lines.append(f"Cluster Type:         {spec.get('cluster_type', 'N/A')}")
        if 'cluster_vars' in spec and spec.get('cluster_vars'):
            summary_lines.append(f"Cluster Variables:    {', '.join(spec.get('cluster_vars', []))}")
        summary_lines.append("")
    
    # Model statistics section
    if 'model_statistics' in results:
        stats = results['model_statistics']
        summary_lines.append("MODEL STATISTICS")
        summary_lines.append("-" * 80)
        
        if 'n_observations' in stats:
            summary_lines.append(f"Total Observations:    {stats['n_observations']:,}")
        if 'n_obs' in stats:
            summary_lines.append(f"Total Observations:    {stats['n_obs']:,}")
        if 'n_obs_compressed' in stats:
            summary_lines.append(f"Compressed Rows:       {stats['n_obs_compressed']:,}")
        if 'n_compressed_rows' in stats:
            summary_lines.append(f"Compressed Rows:       {stats['n_compressed_rows']:,}")
        if 'compression_ratio' in stats and stats['compression_ratio']:
            summary_lines.append(f"Compression Ratio:     {stats['compression_ratio']:.4f} ({stats['compression_ratio']*100:.2f}%)")
        if 'n_coefficients' in stats:
            summary_lines.append(f"Number of Coefficients: {stats['n_coefficients']}")
        if 'n_features' in stats:
            summary_lines.append(f"Number of Features:     {stats['n_features']}")
        
        if 'r_squared' in stats:
            summary_lines.append(f"R-squared:             {stats['r_squared']:.6f}")
        if 'adj_r_squared' in stats and stats['adj_r_squared'] is not None:
            summary_lines.append(f"Adj. R-squared:        {stats['adj_r_squared']:.6f}")
        
        if 'has_standard_errors' in stats:
            summary_lines.append(f"Standard Errors:       {'Available' if stats['has_standard_errors'] else 'Not Available'}")
        
        summary_lines.append("")
    
    # Regression results section
    if 'coefficients' in results:
        coef = results['coefficients']
        summary_lines.append("REGRESSION RESULTS")
        summary_lines.append("-" * 80)
        
        names = coef.get('names', [])
        estimates = coef.get('estimates', [])
        std_errors = coef.get('std_errors', [])
        t_stats = coef.get('t_statistics', [])
        p_values = coef.get('p_values', [])
        ci_lower = coef.get('conf_int_lower', [])
        ci_upper = coef.get('conf_int_upper', [])
        
        # Build results dataframe
        results_data = {
            'Coefficient': estimates,
        }
        if std_errors:
            results_data['Std. Error'] = std_errors
        if t_stats:
            results_data['t-stat'] = t_stats
        if p_values:
            results_data['P>|t|'] = p_values
        if ci_lower:
            results_data['[0.025'] = ci_lower
        if ci_upper:
            results_data['0.975]'] = ci_upper
        
        results_df = pd.DataFrame(results_data, index=names)
        summary_lines.append(results_df.to_string())
        summary_lines.append("")
    
    # Summary footer
    summary_lines.append("=" * 80)
    summary_lines.append("END OF SUMMARY")
    summary_lines.append("=" * 80)
    
    return "\n".join(summary_lines)


def find_latest_results(directory: Path) -> Path:
    """Find the latest results_*.json file in directory."""
    results_files = sorted(directory.glob("results_*.json"), reverse=True)
    if not results_files:
        raise FileNotFoundError(f"No results_*.json files found in {directory}")
    return results_files[0]


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python print_results.py <path_to_results.json or directory>")
        print("")
        print("Examples:")
        print("  python print_results.py output/analysis/duckreg/my_spec/results_20240115_143022.json")
        print("  python print_results.py output/analysis/duckreg/my_spec/  # uses latest results file")
        sys.exit(1)
    
    path = Path(sys.argv[1]).resolve()
    
    # Determine if path is file or directory
    if path.is_file():
        if not path.name.endswith('.json'):
            print(f"Error: File must be .json format: {path}")
            sys.exit(1)
        results_file = path
    elif path.is_dir():
        results_file = find_latest_results(path)
        print(f"Found latest results file: {results_file.name}\n", file=sys.stderr)
    else:
        print(f"Error: Path not found: {path}")
        sys.exit(1)
    
    # Load and display results
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        summary = format_results_summary(results)
        print(summary)
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

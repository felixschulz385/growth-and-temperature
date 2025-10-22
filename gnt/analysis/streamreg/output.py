"""
Output formatters for regression results.

Each formatter is responsible for converting RegressionResults to a specific format
and saving it to disk. Formatters are stateless and used by RegressionResults.save().
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BaseFormatter:
    """Base class for output formatters."""
    
    @staticmethod
    def format_and_save(results, run_dir: Path, *args, **kwargs) -> None:
        """Format and save results. Must be implemented by subclasses."""
        raise NotImplementedError


class JSONFormatter(BaseFormatter):
    """Format and save results as JSON."""
    
    @staticmethod
    def format_and_save(results, run_dir: Path) -> None:
        """Save results as JSON."""
        results_file = run_dir / "results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved JSON results: {results_file.name}")


class SummaryFormatter(BaseFormatter):
    """Format and save summary report."""
    
    @staticmethod
    def format_and_save(results, run_dir: Path, spec_config: Dict[str, Any]) -> None:
        """Save formatted summary report."""
        summary_file = run_dir / "summary.txt"
        
        with open(summary_file, 'w') as f:
            SummaryFormatter._write_header(f, results, spec_config)
            SummaryFormatter._write_model_stats(f, results)
            SummaryFormatter._write_coefficients(f, results)
            SummaryFormatter._write_cluster_diagnostics(f, results)
            SummaryFormatter._write_first_stage(f, results)
            SummaryFormatter._write_footer(f)
        
        logger.info(f"✓ Saved summary report: summary.txt")
    
    @staticmethod
    def _write_header(f, results, spec_config: Dict) -> None:
        """Write report header."""
        f.write("=" * 80 + "\n")
        f.write("REGRESSION ANALYSIS RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Analysis: {spec_config.get('description', 'N/A')}\n")
        f.write(f"Model Type: {results.model_type.upper()}\n")
        f.write(f"Data Source: {spec_config.get('data_source', 'N/A')}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n" + "-" * 80 + "\n\n")
    
    @staticmethod
    def _write_model_stats(f, results) -> None:
        """Write model statistics."""
        f.write("MODEL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Observations:           {results.n_obs:>15,}\n")
        f.write(f"Features:               {results.n_features:>15}\n")
        f.write(f"R-squared:              {results.r_squared:>15.6f}\n")
        f.write(f"Adjusted R-squared:     {results.adj_r_squared:>15.6f}\n")
        f.write(f"Residual Sum of Squares:{results.rss:>15.2f}\n")
        f.write(f"Root MSE:               {np.sqrt(results.rss/results.n_obs):>15.6f}\n")
        
        if results.f_statistic is not None:
            f.write(f"F-statistic:            {results.f_statistic:>15.4f}\n")
            if results.f_pvalue is not None:
                f.write(f"F-test p-value:         {results.f_pvalue:>15.6f}\n")
            if results.df_model is not None and results.df_resid is not None:
                f.write(f"Degrees of freedom:     {results.df_model:>7,} (model), {results.df_resid:>7,} (resid)\n")
        
        f.write(f"Standard Error Type:    {results.cluster_type:>15}\n")
        f.write("\n" + "-" * 80 + "\n\n")
    
    @staticmethod
    def _write_coefficients(f, results) -> None:
        """Write coefficient table."""
        f.write("COEFFICIENT ESTIMATES\n")
        f.write("-" * 80 + "\n")
        summary_df = results.summary()
        f.write(summary_df.to_string())
        f.write("\n\n")
        f.write("Significance levels: *** p<0.01, ** p<0.05, * p<0.10\n")
        f.write("\n" + "-" * 80 + "\n\n")
    
    @staticmethod
    def _write_cluster_diagnostics(f, results) -> None:
        """Write cluster diagnostics."""
        if not results.cluster_diagnostics:
            return
        
        f.write("CLUSTER DIAGNOSTICS\n")
        f.write("-" * 80 + "\n")
        for dim_name, diag in results.cluster_diagnostics.items():
            f.write(f"\n{dim_name.upper()}:\n")
            f.write(f"  Number of clusters:  {diag['n_clusters']:>10,}\n")
            f.write(f"  Min cluster size:    {diag['min_size']:>10,}\n")
            f.write(f"  Max cluster size:    {diag['max_size']:>10,}\n")
            f.write(f"  Mean cluster size:   {diag['mean_size']:>10.1f}\n")
            f.write(f"  Median cluster size: {diag['median_size']:>10.1f}\n")
            
            if diag.get('warnings'):
                f.write("\n  Warnings:\n")
                for warning in diag['warnings']:
                    f.write(f"    ⚠  {warning}\n")
        f.write("\n" + "-" * 80 + "\n\n")
    
    @staticmethod
    def _write_first_stage(f, results) -> None:
        """Write first stage results for 2SLS."""
        if not results.first_stage_results:
            return
        
        f.write("FIRST STAGE RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for i, fs in enumerate(results.first_stage_results):
            endog_var = results.metadata.get('endogenous_variables', [f"endog_{i}"])[i]
            f.write(f"First Stage {i+1}: {endog_var}\n")
            f.write("-" * 80 + "\n")
            f.write(fs.summary().to_string())
            f.write(f"\n\nR-squared: {fs.r_squared:.6f}\n")
            f.write(f"Adjusted R-squared: {fs.adj_r_squared:.6f}\n")
            
            if fs.f_statistic is not None:
                f.write(f"F-statistic (overall model): {fs.f_statistic:.4f}")
                if fs.f_pvalue is not None:
                    f.write(f" (p-value: {fs.f_pvalue:.6f})")
                if fs.df_model is not None and fs.df_resid is not None:
                    f.write(f" [df_model={fs.df_model}, df_resid={fs.df_resid}]")
                f.write("\n")
            
            if fs.metadata.get('iv_f_statistic') is not None:
                iv_f_stat = fs.metadata['iv_f_statistic']
                iv_f_df = fs.metadata['iv_f_df']
                f.write(f"F-statistic (instruments only): {iv_f_stat:.4f}")
                if iv_f_df is not None:
                    f.write(f" [df_instr={iv_f_df[0]}, df_resid={iv_f_df[1]}]")
                f.write("\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    @staticmethod
    def _write_footer(f) -> None:
        """Write report footer."""
        f.write("=" * 80 + "\n")
        f.write("End of Report\n")
        f.write("=" * 80 + "\n")


class CSVFormatter(BaseFormatter):
    """Format and save coefficient table as CSV."""
    
    @staticmethod
    def format_and_save(results, run_dir: Path) -> None:
        """Save coefficient table as CSV."""
        csv_file = run_dir / "coefficients.csv"
        summary_df = results.summary()
        
        # Add additional columns
        summary_df['feature'] = summary_df.index
        summary_df['n_obs'] = results.n_obs
        summary_df['r_squared'] = results.r_squared
        
        # Reorder columns
        cols = ['feature', 'coefficient', 'std_error', 't_statistic', 'p_value', 'sig', 
                'n_obs', 'r_squared']
        summary_df[cols].to_csv(csv_file, index=False)
        
        logger.info(f"✓ Saved coefficient table: coefficients.csv")


class LaTeXFormatter(BaseFormatter):
    """Format and save LaTeX table."""
    
    @staticmethod
    def format_and_save(results, run_dir: Path) -> None:
        """Save LaTeX-formatted table."""
        latex_file = run_dir / "table.tex"
        summary_df = results.summary()
        
        with open(latex_file, 'w') as f:
            f.write("% Regression Results Table\n")
            f.write("% Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Regression Results}\n")
            f.write("\\begin{tabular}{lcccc}\n")
            f.write("\\hline\\hline\n")
            f.write("Variable & Coefficient & Std. Error & t-stat & p-value \\\\\n")
            f.write("\\hline\n")
            
            for idx, row in summary_df.iterrows():
                sig_marker = row.get('sig', '')
                f.write(f"{idx} & {row['coefficient']:.4f}{sig_marker} & "
                       f"({row['std_error']:.4f}) & {row['t_statistic']:.2f} & "
                       f"{row['p_value']:.3f} \\\\\n")
            
            f.write("\\hline\n")
            f.write(f"Observations & \\multicolumn{{4}}{{c}}{{{results.n_obs:,}}} \\\\\n")
            f.write(f"R-squared & \\multicolumn{{4}}{{c}}{{{results.r_squared:.4f}}} \\\\\n")
            f.write("\\hline\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\begin{minipage}{\\textwidth}\n")
            f.write("\\small\n")
            f.write("\\textit{Notes:} Standard errors in parentheses. ")
            f.write(f"Standard errors are {results.cluster_type}. ")
            f.write("$^{***}$p$<$0.01, $^{**}$p$<$0.05, $^{*}$p$<$0.10.\n")
            f.write("\\end{minipage}\n")
            f.write("\\end{table}\n")
        
        logger.info(f"✓ Saved LaTeX table: table.tex")


class DiagnosticsFormatter(BaseFormatter):
    """Format and save detailed diagnostics."""
    
    @staticmethod
    def format_and_save(results, run_dir: Path, spec_config: Dict[str, Any]) -> None:
        """Save detailed diagnostics."""
        diag_file = run_dir / "diagnostics.txt"
        
        with open(diag_file, 'w') as f:
            f.write("DIAGNOSTIC INFORMATION\n")
            f.write("=" * 80 + "\n\n")
            
            DiagnosticsFormatter._write_specification(f, results, spec_config)
            DiagnosticsFormatter._write_clustering(f, spec_config)
            DiagnosticsFormatter._write_feature_engineering(f, spec_config)
            DiagnosticsFormatter._write_settings(f, spec_config)
            DiagnosticsFormatter._write_coefficient_details(f, results)
            
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f"✓ Saved diagnostics: diagnostics.txt")
    
    @staticmethod
    def _write_specification(f, results, spec_config: Dict) -> None:
        """Write specification details."""
        f.write("SPECIFICATION\n")
        f.write("-" * 80 + "\n")
        if 'formula' in spec_config:
            f.write(f"Formula: {spec_config['formula']}\n")
        f.write(f"Features: {', '.join(results.feature_names)}\n")
        f.write(f"Target: {spec_config.get('target_col', 'N/A')}\n")
        f.write("\n")
    
    @staticmethod
    def _write_clustering(f, spec_config: Dict) -> None:
        """Write clustering information."""
        if spec_config.get('cluster1_col') or spec_config.get('cluster2_col'):
            f.write("CLUSTERING\n")
            f.write("-" * 80 + "\n")
            if spec_config.get('cluster1_col'):
                f.write(f"Cluster Dim 1: {spec_config['cluster1_col']}\n")
            if spec_config.get('cluster2_col'):
                f.write(f"Cluster Dim 2: {spec_config['cluster2_col']}\n")
            f.write("\n")
    
    @staticmethod
    def _write_feature_engineering(f, spec_config: Dict) -> None:
        """Write feature engineering configuration."""
        if 'feature_engineering' in spec_config:
            f.write("FEATURE ENGINEERING\n")
            f.write("-" * 80 + "\n")
            f.write(json.dumps(spec_config['feature_engineering'], indent=2))
            f.write("\n\n")
    
    @staticmethod
    def _write_settings(f, spec_config: Dict) -> None:
        """Write estimation settings."""
        f.write("ESTIMATION SETTINGS\n")
        f.write("-" * 80 + "\n")
        settings = spec_config.get('settings', {})
        for key, value in sorted(settings.items()):
            f.write(f"{key:.<30} {value}\n")
        f.write("\n")
    
    @staticmethod
    def _write_coefficient_details(f, results) -> None:
        """Write detailed coefficient information."""
        f.write("COEFFICIENT DETAILS\n")
        f.write("-" * 80 + "\n")
        for feat_name in results.feature_names:
            coef_info = results.get_coefficient(feat_name)
            ci_lower, ci_upper = results.get_confidence_interval(feat_name, alpha=0.05)
            
            f.write(f"\n{feat_name}:\n")
            f.write(f"  Estimate:     {coef_info['coefficient']:>12.6f}\n")
            f.write(f"  Std Error:    {coef_info['std_error']:>12.6f}\n")
            f.write(f"  t-statistic:  {coef_info['t_statistic']:>12.4f}\n")
            f.write(f"  p-value:      {coef_info['p_value']:>12.6f}\n")
            f.write(f"  95% CI:       [{ci_lower:>11.6f}, {ci_upper:>11.6f}]\n")


class READMEFormatter(BaseFormatter):
    """Format and save README documentation."""
    
    @staticmethod
    def format_and_save(results, run_dir: Path, spec_config: Dict[str, Any], 
                       timestamp: str) -> None:
        """Create README file with analysis overview."""
        readme_file = run_dir / "README.md"
        
        with open(readme_file, 'w') as f:
            READMEFormatter._write_header(f, spec_config)
            READMEFormatter._write_overview(f, results)
            READMEFormatter._write_file_list(f)
            READMEFormatter._write_specification(f, spec_config)
            READMEFormatter._write_top_results(f, results)
            READMEFormatter._write_cluster_info(f, results)
            f.write("---\n")
            f.write(f"*Analysis ID: {timestamp}*\n")
        
        logger.info(f"✓ Created README: README.md")
    
    @staticmethod
    def _write_header(f, spec_config: Dict) -> None:
        """Write README header."""
        f.write(f"# Analysis Results: {spec_config.get('description', 'N/A')}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    @staticmethod
    def _write_overview(f, results) -> None:
        """Write overview section."""
        f.write("## Overview\n\n")
        f.write(f"- **Model Type:** {results.model_type.upper()}\n")
        f.write(f"- **Observations:** {results.n_obs:,}\n")
        f.write(f"- **Features:** {results.n_features}\n")
        f.write(f"- **R²:** {results.r_squared:.4f}\n")
        f.write(f"- **Standard Errors:** {results.cluster_type}\n\n")
    
    @staticmethod
    def _write_file_list(f) -> None:
        """Write file list."""
        f.write("## Files\n\n")
        f.write("| File | Description |\n")
        f.write("|------|-------------|\n")
        f.write("| `summary.txt` | Full formatted analysis report |\n")
        f.write("| `results.json` | Complete results in JSON format |\n")
        f.write("| `coefficients.csv` | Coefficient table (CSV) |\n")
        f.write("| `diagnostics.txt` | Detailed diagnostics and confidence intervals |\n")
        f.write("| `table.tex` | LaTeX-formatted results table |\n")
        f.write("| `config_snapshot.json` | Configuration used for this run |\n\n")
    
    @staticmethod
    def _write_specification(f, spec_config: Dict) -> None:
        """Write specification section."""
        if spec_config.get('formula'):
            f.write("## Specification\n\n")
            f.write(f"```\n{spec_config['formula']}\n```\n\n")
    
    @staticmethod
    def _write_top_results(f, results) -> None:
        """Write top results."""
        f.write("## Quick Results\n\n")
        f.write("### Top 5 Coefficients by Magnitude\n\n")
        summary_df = results.summary()
        top_5 = summary_df.nlargest(5, 'coefficient', keep='all')
        f.write("| Variable | Coefficient | p-value |\n")
        f.write("|----------|-------------|----------|\n")
        for idx, row in top_5.iterrows():
            sig = row.get('sig', '')
            f.write(f"| {idx} | {row['coefficient']:.4f}{sig} | {row['p_value']:.4f} |\n")
        f.write("\n")
    
    @staticmethod
    def _write_cluster_info(f, results) -> None:
        """Write cluster information."""
        if results.cluster_diagnostics:
            f.write("## Cluster Information\n\n")
            for dim_name, diag in results.cluster_diagnostics.items():
                f.write(f"### {dim_name}\n\n")
                f.write(f"- Clusters: {diag['n_clusters']:,}\n")
                f.write(f"- Cluster size: {diag['min_size']:,} - {diag['max_size']:,} "
                       f"(mean: {diag['mean_size']:.1f})\n\n")


class ConfigFormatter:
    """Format and save configuration snapshot."""
    
    @staticmethod
    def save(spec_config: Dict, full_config: Dict, run_dir: Path, timestamp: str) -> None:
        """Save configuration snapshot."""
        config_file = run_dir / "config_snapshot.json"
        snapshot = {
            'specification': spec_config,
            'timestamp': datetime.now().isoformat(),
            'full_config': full_config
        }
        
        with open(config_file, 'w') as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Saved config snapshot: config_snapshot.json")

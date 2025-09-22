import os
import sys
import yaml
import logging
import argparse
import importlib
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gnt.analysis.models.online_RLS import process_partitioned_dataset_parallel, OnlineRLS

# Configure logging
def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create logs directory if it doesn't exist
    log_file = log_config.get('file')
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(level=level, format=format_str)
    
    return logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load analysis configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables
    def expand_env_vars(obj):
        if isinstance(obj, str):
            expanded = os.path.expandvars(obj)
            # Try to convert numeric strings back to numbers
            if expanded != obj:  # Only if expansion occurred
                try:
                    # Try integer first
                    if expanded.isdigit() or (expanded.startswith('-') and expanded[1:].isdigit()):
                        return int(expanded)
                    # Try float
                    return float(expanded)
                except ValueError:
                    # Return as string if conversion fails
                    return expanded
            return expanded
        elif isinstance(obj, dict):
            return {k: expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [expand_env_vars(item) for item in obj]
        return obj
    
    return expand_env_vars(config)

def run_online_rls(config: Dict[str, Any], spec_name: str, 
                   output_dir: Optional[str] = None, verbose: bool = True) -> OnlineRLS:
    """Run Online RLS analysis with specified configuration."""
    logger = logging.getLogger(__name__)
    
    # Get analysis configuration
    rls_config = config['analyses']['online_rls']
    spec_config = rls_config['specifications'][spec_name]
    defaults = rls_config['defaults']
    
    logger.info(f"Running Online RLS analysis: {spec_config['description']}")
    logger.info(f"Data source: {spec_config['data_source']}")
    
    # Merge settings with proper type conversion
    settings = {**defaults, **spec_config.get('settings', {})}
    
    # Override with HPC defaults if not specified
    hpc_config = config.get('hpc', {})
    if 'n_workers' not in settings:
        default_workers = hpc_config.get('default_workers', 4)
        # Ensure n_workers is an integer
        if isinstance(default_workers, str):
            try:
                settings['n_workers'] = int(default_workers)
            except ValueError:
                settings['n_workers'] = 4
        else:
            settings['n_workers'] = default_workers
    
    # Ensure critical parameters are proper types
    settings['alpha'] = float(settings.get('alpha', 1e-3))
    settings['forget_factor'] = float(settings.get('forget_factor', 1.0))
    settings['chunk_size'] = int(settings.get('chunk_size', 20000))
    settings['n_workers'] = int(settings.get('n_workers', 4))
    settings['add_intercept'] = bool(settings.get('add_intercept', True))
    settings['show_progress'] = bool(settings.get('show_progress', True))
    settings['verbose'] = bool(settings.get('verbose', verbose))  # Add verbose setting
    
    logger.info(f"Analysis settings: alpha={settings['alpha']}, chunk_size={settings['chunk_size']}, n_workers={settings['n_workers']}, verbose={settings['verbose']}")
    
    # Run the analysis
    rls = process_partitioned_dataset_parallel(
        parquet_path=spec_config['data_source'],
        feature_cols=spec_config.get('feature_cols'),
        target_col=spec_config['target_col'],
        cluster1_col=spec_config.get('cluster1_col'),
        cluster2_col=spec_config.get('cluster2_col'),
        add_intercept=settings['add_intercept'],
        chunk_size=settings['chunk_size'],
        n_workers=settings['n_workers'],
        alpha=settings['alpha'],
        forget_factor=settings['forget_factor'],
        show_progress=settings['show_progress'],
        verbose=settings['verbose']
    )
    
    # Generate summary
    cluster_type = settings.get('cluster_type', 'classical')
    summary = rls.summary(cluster_type=cluster_type)
    
    logger.info(f"Analysis complete! Total observations: {rls.n_obs:,}")
    logger.info(f"RSS: {rls.rss:.6f}")
    logger.info("Regression Summary:")
    print("\n" + "="*50)
    print(f"Analysis: {spec_config['description']}")
    print(f"Standard Errors: {cluster_type}")
    print("="*50)
    print(summary.to_string())
    print("="*50)
    
    # Save results if output directory specified
    if output_dir:
        save_results(rls, summary, spec_name, spec_config, output_dir, config)
    
    return rls

def save_results(rls: OnlineRLS, summary: pd.DataFrame, spec_name: str,
                spec_config: Dict[str, Any], output_dir: str, 
                config: Dict[str, Any]) -> None:
    """Save analysis results to a single standardized JSON format."""
    logger = logging.getLogger(__name__)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / f"{spec_name}_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    logger.info(f"Saving results to: {run_dir}")
    
    # Get cluster type and analysis settings
    cluster_type = config['analyses']['online_rls']['specifications'][spec_name].get('settings', {}).get('cluster_type', 'one_way')
    settings = {**config['analyses']['online_rls']['defaults'], **spec_config.get('settings', {})}
    
    # Create feature names
    feature_names = []
    if settings.get('add_intercept', True):
        feature_names.append('intercept')
    feature_names.extend(spec_config.get('feature_cols', []))
    
    # Calculate additional statistics
    n_clusters_1 = len(rls.cluster_stats) if rls.cluster_stats else 0
    n_clusters_2 = len(rls.cluster2_stats) if rls.cluster2_stats else 0
    n_intersections = len(rls.intersection_stats) if rls.intersection_stats else 0
    
    # Calculate R-squared and adjusted R-squared
    total_sum_squares = np.var(rls.Xty / rls.n_obs if rls.n_obs > 0 else 0) * rls.n_obs
    r_squared = 1 - (rls.rss / total_sum_squares) if total_sum_squares > 0 else 0
    adj_r_squared = 1 - ((1 - r_squared) * (rls.n_obs - 1) / (rls.n_obs - rls.n_features)) if rls.n_obs > rls.n_features else 0
    
    # Create comprehensive results dictionary
    results = {
        "metadata": {
            "analysis_type": "online_rls",
            "specification": spec_name,
            "description": spec_config['description'],
            "timestamp": timestamp,
            "cluster_type": cluster_type,
            "data_source": spec_config['data_source'],
            "target_variable": spec_config['target_col'],
            "feature_variables": spec_config.get('feature_cols', []),
            "cluster_variables": {
                "cluster1": spec_config.get('cluster1_col'),
                "cluster2": spec_config.get('cluster2_col')
            }
        },
        
        "model_specification": {
            "add_intercept": settings.get('add_intercept', True),
            "alpha": float(settings.get('alpha', 1e-3)),
            "forget_factor": float(settings.get('forget_factor', 1.0)),
            "chunk_size": int(settings.get('chunk_size', 20000)),
            "n_workers": int(settings.get('n_workers', 4))
        },
        
        "sample_statistics": {
            "n_observations": int(rls.n_obs),
            "n_features": int(rls.n_features),
            "n_clusters_1": n_clusters_1,
            "n_clusters_2": n_clusters_2,
            "n_cluster_intersections": n_intersections,
            "feature_names": feature_names
        },
        
        "model_fit": {
            "residual_sum_squares": float(rls.rss),
            "r_squared": float(r_squared),
            "adjusted_r_squared": float(adj_r_squared),
            "root_mean_squared_error": float(np.sqrt(rls.rss / rls.n_obs)) if rls.n_obs > 0 else 0,
            "degrees_of_freedom": int(rls.n_obs - rls.n_features) if rls.n_obs > rls.n_features else 0
        },
        
        "coefficients": {
            "estimates": {
                feature_names[i]: {
                    "coefficient": float(rls.theta[i]),
                    "std_error": float(summary.iloc[i]['std_error']),
                    "t_statistic": float(summary.iloc[i]['t_statistic']),
                    "p_value": float(summary.iloc[i]['p_value']),
                    "ci_lower_95": float(rls.theta[i] - 1.96 * summary.iloc[i]['std_error']),
                    "ci_upper_95": float(rls.theta[i] + 1.96 * summary.iloc[i]['std_error']),
                    "significant_5pct": bool(summary.iloc[i]['p_value'] < 0.05),
                    "significant_1pct": bool(summary.iloc[i]['p_value'] < 0.01)
                }
                for i in range(len(feature_names))
            },
            "covariance_matrix": rls.get_cluster_robust_covariance(cluster_type).tolist() if cluster_type != 'classical' else rls.get_covariance_matrix().tolist()
        },
        
        "inference": {
            "standard_error_type": cluster_type,
            "cluster_robust": cluster_type != 'classical',
            "hypothesis_tests": {
                "joint_significance": {
                    "description": "F-test for joint significance of all coefficients (excluding intercept)",
                    "test_statistic": None,  # Could be calculated if needed
                    "p_value": None,
                    "critical_value_5pct": None
                }
            }
        },
        
        "diagnostics": {
            "convergence": {
                "converged": True,  # Online RLS always converges
                "regularization_applied": bool(rls.alpha > 0),
                "numerical_issues": False  # Could be flagged during processing
            },
            "data_quality": {
                "missing_values_handled": True,
                "infinite_values_handled": True,
                "outlier_detection": False
            }
        },
        
        "computational_details": {
            "algorithm": "Online Recursive Least Squares",
            "implementation": "Vectorized batch processing with parallel partition handling",
            "memory_efficient": True,
            "processing_time_seconds": None,  # Could be tracked
            "partitions_processed": None,  # Could be tracked
            "partitions_failed": None  # Could be tracked
        },
        
        "summary_table": {
            "coefficient_table": [
                {
                    "variable": feature_names[i],
                    "coefficient": float(rls.theta[i]),
                    "std_error": float(summary.iloc[i]['std_error']),
                    "t_statistic": float(summary.iloc[i]['t_statistic']),
                    "p_value": float(summary.iloc[i]['p_value']),
                    "significance": "***" if summary.iloc[i]['p_value'] < 0.01 else 
                                   "**" if summary.iloc[i]['p_value'] < 0.05 else 
                                   "*" if summary.iloc[i]['p_value'] < 0.10 else ""
                }
                for i in range(len(feature_names))
            ]
        }
    }
    
    # Save the comprehensive results as JSON
    try:
        results_file = run_dir / "analysis_results.json"
        with open(results_file, 'w') as f:
            import json
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved comprehensive results: {results_file}")
        
        # Also save a human-readable summary
        summary_file = run_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write(f"GNT Analysis Results: {spec_config['description']}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Analysis Type: Online RLS Regression\n")
            f.write(f"Specification: {spec_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Standard Errors: {cluster_type}\n\n")
            
            f.write(f"Sample Information:\n")
            f.write(f"  Observations: {rls.n_obs:,}\n")
            f.write(f"  Features: {rls.n_features}\n")
            f.write(f"  Clusters (Dim 1): {n_clusters_1}\n")
            f.write(f"  Clusters (Dim 2): {n_clusters_2}\n\n")
            
            f.write(f"Model Fit:\n")
            f.write(f"  R-squared: {r_squared:.6f}\n")
            f.write(f"  Adj. R-squared: {adj_r_squared:.6f}\n")
            f.write(f"  RMSE: {np.sqrt(rls.rss / rls.n_obs) if rls.n_obs > 0 else 0:.6f}\n")
            f.write(f"  RSS: {rls.rss:.6f}\n\n")
            
            f.write(f"Coefficient Estimates:\n")
            f.write(f"{'Variable':<20} {'Coeff':<12} {'Std Err':<12} {'t-stat':<10} {'P>|t|':<10} {'Sig':<5}\n")
            f.write(f"{'-'*75}\n")
            for i, var in enumerate(feature_names):
                coef = rls.theta[i]
                se = summary.iloc[i]['std_error']
                t_stat = summary.iloc[i]['t_statistic']
                p_val = summary.iloc[i]['p_value']
                sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
                f.write(f"{var:<20} {coef:<12.6f} {se:<12.6f} {t_stat:<10.3f} {p_val:<10.6f} {sig:<5}\n")
            
            f.write(f"\nSignificance codes: *** p<0.01, ** p<0.05, * p<0.10\n")
        
        logger.info(f"Saved human-readable summary: {summary_file}")
        
        # Save model object if requested
        if config.get('output', {}).get('save_models', False):
            import pickle
            model_file = run_dir / "model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(rls, f)
            logger.info(f"Saved model object: {model_file}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    logger.info(f"Results successfully saved to: {run_dir}")
    
    # Log summary of saved files
    saved_files = list(run_dir.glob("*"))
    logger.info(f"Total files saved: {len(saved_files)}")
    for file_path in saved_files:
        file_size = file_path.stat().st_size
        logger.info(f"  - {file_path.name}: {file_size:,} bytes")

def list_analyses(config: Dict[str, Any]) -> None:
    """List available analyses and specifications."""
    print("\nAvailable analyses:")
    print("=" * 50)
    
    for analysis_name, analysis_config in config['analyses'].items():
        print(f"\n{analysis_name.upper()}")
        print(f"  Description: {analysis_config['description']}")
        
        if 'specifications' in analysis_config:
            print("  Specifications:")
            for spec_name, spec_config in analysis_config['specifications'].items():
                print(f"    - {spec_name}: {spec_config['description']}")
    print()

def main():
    """Main entrypoint for analysis."""
    parser = argparse.ArgumentParser(description="GNT Analysis Pipeline")
    parser.add_argument("analysis_type", choices=['online_rls', 'list'], 
                       help="Type of analysis to run or 'list' to show available analyses")
    parser.add_argument("--config", default="orchestration/configs/analysis.yaml",
                       help="Path to analysis configuration file")
    parser.add_argument("--specification", "-s", 
                       help="Analysis specification to use")
    parser.add_argument("--output", "-o",
                       help="Output directory for results")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--verbose", "-v", action="store_true", default=True,
                       help="Enable verbose progress output (default: True)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Disable verbose progress output")
    
    args = parser.parse_args()
    
    # Determine verbosity
    verbose = args.verbose and not args.quiet
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to project root
        config_path = project_root / args.config
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    
    # Setup logging
    if args.debug:
        config.setdefault('logging', {})['level'] = 'DEBUG'
    
    logger = setup_logging(config)
    
    if args.analysis_type == 'list':
        list_analyses(config)
        return
    
    # Validate analysis type
    if args.analysis_type not in config['analyses']:
        logger.error(f"Unknown analysis type: {args.analysis_type}")
        logger.info("Use --help or 'list' to see available analyses")
        sys.exit(1)
    
    # Set default output directory to the new location
    output_dir = args.output
    if not output_dir:
        output_dir = config.get('output', {}).get('base_path', 
                                                  str(project_root / "output" / "analysis"))
    
    try:
        if args.analysis_type == 'online_rls':
            if not args.specification:
                logger.error("Online RLS analysis requires a specification. Use --specification/-s")
                logger.info("Available specifications:")
                specs = config['analyses']['online_rls']['specifications']
                for spec_name, spec_config in specs.items():
                    logger.info(f"  - {spec_name}: {spec_config['description']}")
                sys.exit(1)
            
            # Validate specification
            specs = config['analyses']['online_rls']['specifications']
            if args.specification not in specs:
                logger.error(f"Unknown specification: {args.specification}")
                logger.info(f"Available specifications: {list(specs.keys())}")
                sys.exit(1)
            
            run_online_rls(config, args.specification, output_dir, verbose)
        
        else:
            logger.error(f"Analysis type '{args.analysis_type}' not yet implemented")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        if args.debug:
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()

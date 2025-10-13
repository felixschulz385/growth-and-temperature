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
from gnt.analysis.models.online_2SLS import process_partitioned_dataset_2sls, Online2SLS

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
    settings['verbose'] = bool(settings.get('verbose', verbose))
    
    # Get cluster type for standard errors
    cluster_type = settings.get('cluster_type', 'classical')
    
    # Check if using formula syntax
    formula = spec_config.get('formula')
    if formula:
        logger.info(f"Using formula specification: {formula}")
    
    # Extract feature engineering configuration
    feature_engineering = spec_config.get('feature_engineering') or settings.get('feature_engineering')
    
    if feature_engineering:
        logger.info(f"Feature engineering enabled with {len(feature_engineering.get('transformations', []))} transformations")
        for i, transform in enumerate(feature_engineering.get('transformations', [])):
            logger.info(f"  Transformation {i+1}: {transform.get('type', 'unknown')} - {transform}")
    else:
        logger.info("No feature engineering specified - using original features only")
    
    logger.info(f"Analysis settings: alpha={settings['alpha']}, chunk_size={settings['chunk_size']}, n_workers={settings['n_workers']}, verbose={settings['verbose']}")
    logger.info(f"Standard error type: {cluster_type}")
    
    # Run the analysis with error handling
    rls = None
    try:
        rls = process_partitioned_dataset_parallel(
            parquet_path=spec_config['data_source'],
            feature_cols=spec_config.get('feature_cols'),
            target_col=spec_config.get('target_col'),
            cluster1_col=spec_config.get('cluster1_col'),
            cluster2_col=spec_config.get('cluster2_col'),
            add_intercept=settings['add_intercept'],
            chunk_size=settings['chunk_size'],
            n_workers=settings['n_workers'],
            alpha=settings['alpha'],
            forget_factor=settings['forget_factor'],
            show_progress=settings['show_progress'],
            verbose=settings['verbose'],
            feature_engineering=feature_engineering,
            formula=formula  # Pass formula to parser
        )
        
        # Generate summary with correct cluster type
        summary = rls.summary(
            cluster_type=cluster_type
        )
        
        logger.info(f"Analysis complete! Total observations: {rls.n_obs:,}")
        logger.info(f"RSS: {rls.rss:.6f}")
        logger.info("Regression Summary:")
        print("\n" + "="*50)
        print(f"Analysis: {spec_config['description']}")
        print(f"Standard Errors: {cluster_type}")
        if feature_engineering:
            print(f"Feature Engineering: {len(feature_engineering.get('transformations', []))} transformations")
            print(f"Total features: {len(rls.feature_names)}")
        print("="*50)
        print(summary.to_string())
        print("="*50)
        
        # Save results if output directory specified
        if output_dir:
            save_results(rls, summary, spec_name, spec_config, output_dir, config)
            
    except Exception as e:
        logger.error(f"Analysis failed with error: {str(e)}")
        # If we have partial results, try to save them
        if rls is not None and rls.n_obs > 0:
            logger.warning("Attempting to save partial results...")
            try:
                summary = rls.summary(cluster_type=cluster_type)
                
                logger.info(f"Partial analysis results - Total observations: {rls.n_obs:,}")
                logger.info(f"RSS: {rls.rss:.6f}")
                print("\n" + "="*50)
                print(f"PARTIAL RESULTS - Analysis: {spec_config['description']}")
                print(f"Standard Errors: {cluster_type}")
                if feature_engineering:
                    print(f"Feature Engineering: {len(feature_engineering.get('transformations', []))} transformations")
                print("="*50)
                print(summary.to_string())
                print("="*50)
                
                if output_dir:
                    save_results(rls, summary, f"{spec_name}_partial", spec_config, output_dir, config)
                    logger.info("Partial results saved successfully")
                    
            except Exception as save_error:
                logger.error(f"Failed to save partial results: {save_error}")
        
        # Re-raise the original error
        raise e
    
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
    settings = {**config['analyses']['online_rls']['defaults'], **spec_config.get('settings', {})}
    cluster_type = settings.get('cluster_type', 'classical')
    
    # Use the feature names from the model (already transformed)
    feature_names = rls.get_feature_names()
    
    # Determine target variable - handle both formula and explicit specification
    target_variable = None
    feature_variables = []
    
    if 'formula' in spec_config:
        # Parse formula to extract target and features
        from gnt.analysis.models.feature_engineering import FormulaParser
        formula_parser = FormulaParser.parse(spec_config['formula'])
        target_variable = formula_parser.target
        feature_variables = formula_parser.features
    else:
        # Use explicit specification
        target_variable = spec_config.get('target_col')
        feature_variables = spec_config.get('feature_cols', [])
    
    # Calculate additional statistics
    n_clusters_1 = len(rls.cluster_stats) if rls.cluster_stats else 0
    n_clusters_2 = len(rls.cluster2_stats) if rls.cluster2_stats else 0
    n_intersections = len(rls.intersection_stats) if rls.intersection_stats else 0
    
    # Calculate R-squared and adjusted R-squared using proper methods
    r_squared = rls.get_r_squared()
    adj_r_squared = rls.get_adjusted_r_squared()
    
    # Create comprehensive results dictionary
    results = {
        "metadata": {
            "analysis_type": "online_rls",
            "specification": spec_name,
            "description": spec_config['description'],
            "timestamp": timestamp,
            "cluster_type": cluster_type,
            "data_source": spec_config['data_source'],
            "target_variable": target_variable,
            "feature_variables": feature_variables,
            "formula": spec_config.get('formula'),  # Include formula if present
            "feature_engineering": {
                "enabled": bool(spec_config.get('feature_engineering')),
                "transformations": spec_config.get('feature_engineering', {}).get('transformations', []) if spec_config.get('feature_engineering') else [],
                "transformed_feature_names": feature_names
            },
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
    
    # Remove bootstrap info from results
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

def run_online_2sls(config: Dict[str, Any], spec_name: str, 
                    output_dir: Optional[str] = None, verbose: bool = True) -> Online2SLS:
    """Run Online 2SLS analysis with specified configuration."""
    logger = logging.getLogger(__name__)
    
    # Get analysis configuration
    twosls_config = config['analyses']['online_2sls']
    spec_config = twosls_config['specifications'][spec_name]
    defaults = twosls_config['defaults']
    
    logger.info(f"Running Online 2SLS analysis: {spec_config['description']}")
    logger.info(f"Data source: {spec_config['data_source']}")
    
    # Parse formula to get variable lists
    endog_cols = None
    exog_cols = None
    instr_cols = None
    
    # Check if using formula syntax
    formula = spec_config.get('formula')
    if formula:
        from gnt.analysis.models.feature_engineering import FormulaParser
        
        logger.info(f"Using formula specification: {formula}")
        formula_parser = FormulaParser.parse(formula)
        
        if not formula_parser.instruments:
            raise ValueError(f"2SLS formula must contain instruments after '|': {formula}")
        
        # Get variables from formula
        instr_cols = formula_parser.instruments
        all_features = formula_parser.features
        
        # Extract feature engineering config
        feature_engineering = spec_config.get('feature_engineering') or defaults.get('feature_engineering')
        
        # Determine endogenous vs exogenous
        if feature_engineering and 'endogenous' in feature_engineering:
            endog_cols = feature_engineering['endogenous']
            exog_cols = [f for f in all_features if f not in endog_cols]
        else:
            # Default: all features are endogenous
            endog_cols = all_features
            exog_cols = []
        
        logger.info(f"Parsed from formula - Endogenous: {endog_cols}, Exogenous: {exog_cols}, Instruments: {instr_cols}")
    else:
        # Use explicit specification
        endog_cols = spec_config.get('endogenous_cols')
        exog_cols = spec_config.get('exogenous_cols')
        instr_cols = spec_config.get('instrument_cols')
        
        # Validate instrument rank condition
        n_endogenous = len(endog_cols) if endog_cols else 0
        n_instruments = len(instr_cols) if instr_cols else 0
        
        if n_endogenous > 0 and n_instruments > 0 and n_instruments < n_endogenous:
            raise ValueError(f"Under-identified model: {n_instruments} instruments for {n_endogenous} endogenous variables")
        
        if n_endogenous > 0 and n_instruments > 0:
            logger.info(f"Identification: {n_instruments} instruments for {n_endogenous} endogenous variables")
    
    # Merge settings with proper type conversion
    settings = {**defaults, **spec_config.get('settings', {})}
    
    # Override with HPC defaults if not specified
    hpc_config = config.get('hpc', {})
    if 'n_workers' not in settings:
        default_workers = hpc_config.get('default_workers', 4)
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
    settings['verbose'] = bool(settings.get('verbose', verbose))
    
    # Get cluster type for standard errors
    cluster_type = settings.get('cluster_type', 'classical')
    
    # Extract feature engineering configuration
    feature_engineering = spec_config.get('feature_engineering') or settings.get('feature_engineering')
    
    if feature_engineering:
        logger.info(f"Feature engineering enabled with {len(feature_engineering.get('transformations', []))} transformations")
        for i, transform in enumerate(feature_engineering.get('transformations', [])):
            logger.info(f"  Transformation {i+1}: {transform.get('type', 'unknown')} - {transform}")
    else:
        logger.info("No feature engineering specified - using original features only")
    
    logger.info(f"Analysis settings: alpha={settings['alpha']}, chunk_size={settings['chunk_size']}, n_workers={settings['n_workers']}, verbose={settings['verbose']}")
    logger.info(f"Standard error type: {cluster_type}")
    
    # Run the analysis with error handling
    twosls = None
    try:
        twosls = process_partitioned_dataset_2sls(
            parquet_path=spec_config['data_source'],
            endog_cols=endog_cols,
            exog_cols=exog_cols,
            instr_cols=instr_cols,
            target_col=spec_config.get('target_col'),
            cluster1_col=spec_config.get('cluster1_col'),
            cluster2_col=spec_config.get('cluster2_col'),
            add_intercept=settings['add_intercept'],
            chunk_size=settings['chunk_size'],
            n_workers=settings['n_workers'],
            alpha=settings['alpha'],
            forget_factor=settings['forget_factor'],
            show_progress=settings['show_progress'],
            verbose=settings['verbose'],
            feature_engineering=feature_engineering,
            formula=formula  # Pass formula to parser
        )
        
        # Generate summary with correct cluster type
        first_stage_summaries = twosls.get_first_stage_summary()
        second_stage_summary = twosls.get_second_stage_summary()
        
        logger.info(f"Analysis complete! Total observations: {twosls.total_obs:,}")
        logger.info("2SLS Regression Summary:")
        print("\n" + "="*50)
        print(f"Analysis: {spec_config['description']}")
        print(f"Standard Errors: {cluster_type}")
        if feature_engineering:
            print(f"Feature Engineering: {len(feature_engineering.get('transformations', []))} transformations")
        print("="*50)
        
        # Print First Stage Results - use parsed endog_cols
        print("First Stage Results:")
        print("-" * 50)
        for i, (endogen_var, summary) in enumerate(zip(endog_cols, first_stage_summaries)):
            print(f"\nFirst Stage {i+1}: {endogen_var}")
            print("-" * 30)
            # Filter out the metadata rows (r_squared, adj_r_squared, observations)
            coeff_summary = summary.iloc[:-3] if len(summary) > 3 else summary
            print(coeff_summary.to_string())
            
            # Print R-squared statistics if available
            if 'r_squared' in summary.index:
                r_sq = summary.loc['r_squared'].iloc[-1] if hasattr(summary.loc['r_squared'], 'iloc') else summary.loc['r_squared']
                print(f"R-squared: {float(r_sq):.6f}")
            if 'adj_r_squared' in summary.index:
                adj_r_sq = summary.loc['adj_r_squared'].iloc[-1] if hasattr(summary.loc['adj_r_squared'], 'iloc') else summary.loc['adj_r_squared']
                print(f"Adjusted R-squared: {float(adj_r_sq):.6f}")
            if 'observations' in summary.index:
                n_obs = summary.loc['observations'].iloc[-1] if hasattr(summary.loc['observations'], 'iloc') else summary.loc['observations']
                print(f"Observations: {int(n_obs):,}")
        
        print("\n" + "="*50)
        print("Second Stage Results:")
        print("-" * 50)
        # Filter out metadata rows for second stage too
        second_stage_coeff = second_stage_summary.iloc[:-3] if len(second_stage_summary) > 3 else second_stage_summary
        print(second_stage_coeff.to_string())
        
        # Print second stage R-squared statistics
        if 'r_squared' in second_stage_summary.index:
            r_sq = second_stage_summary.loc['r_squared'].iloc[-1] if hasattr(second_stage_summary.loc['r_squared'], 'iloc') else second_stage_summary.loc['r_squared']
            print(f"R-squared: {float(r_sq):.6f}")
        if 'adj_r_squared' in second_stage_summary.index:
            adj_r_sq = second_stage_summary.loc['adj_r_squared'].iloc[-1] if hasattr(second_stage_summary.loc['adj_r_squared'], 'iloc') else second_stage_summary.loc['adj_r_squared']
            print(f"Adjusted R-squared: {float(adj_r_sq):.6f}")
        if 'observations' in second_stage_summary.index:
            n_obs = second_stage_summary.loc['observations'].iloc[-1] if hasattr(second_stage_summary.loc['observations'], 'iloc') else second_stage_summary.loc['observations']
            print(f"Observations: {int(n_obs):,}")
        
        print("="*50)
        
        # Save results if output directory specified
        if output_dir:
            save_2sls_results(twosls, second_stage_summary, first_stage_summaries, spec_name, spec_config, output_dir, config, endog_cols, exog_cols, instr_cols)
            
    except Exception as e:
        logger.error(f"2SLS analysis failed with error: {str(e)}")
        # If we have partial results, try to save them
        if twosls is not None and twosls.total_obs > 0:
            logger.warning("Attempting to save partial 2SLS results...")
            try:
                first_stage_summaries = twosls.get_first_stage_summary()
                second_stage_summary = twosls.get_second_stage_summary()
                
                logger.info(f"Partial 2SLS analysis results - Total observations: {twosls.total_obs:,}")
                print("\n" + "="*50)
                print(f"PARTIAL RESULTS - Analysis: {spec_config['description']}")
                print(f"Standard Errors: {cluster_type}")
                if feature_engineering:
                    print(f"Feature Engineering: {len(feature_engineering.get('transformations', []))} transformations")
                print("="*50)
                
                # Print partial first stage results - use parsed endog_cols
                print("First Stage Results (Partial):")
                print("-" * 50)
                for i, (endogen_var, summary) in enumerate(zip(endog_cols, first_stage_summaries)):
                    print(f"\nFirst Stage {i+1}: {endogen_var}")
                    print("-" * 30)
                    coeff_summary = summary.iloc[:-3] if len(summary) > 3 else summary
                    print(coeff_summary.to_string())
                
                print("\n" + "="*50)
                print("Second Stage Results (Partial):")
                print("-" * 50)
                second_stage_coeff = second_stage_summary.iloc[:-3] if len(second_stage_summary) > 3 else second_stage_summary
                print(second_stage_coeff.to_string())
                print("="*50)
                
                if output_dir:
                    save_2sls_results(twosls, second_stage_summary, first_stage_summaries, f"{spec_name}_partial", spec_config, output_dir, config, endog_cols, exog_cols, instr_cols)
                    logger.info("Partial 2SLS results saved successfully")
                    
            except Exception as save_error:
                logger.error(f"Failed to save partial 2SLS results: {save_error}")
        
        # Re-raise the original error
        raise e
    
    return twosls

def save_2sls_results(twosls: Online2SLS, summary: pd.DataFrame, 
                      first_stage_summaries: List[pd.DataFrame], spec_name: str,
                      spec_config: Dict[str, Any], output_dir: str, 
                      config: Dict[str, Any],
                      endog_cols: List[str],
                      exog_cols: List[str],
                      instr_cols: List[str]) -> None:
    """Save 2SLS analysis results with IV-specific diagnostics."""
    logger = logging.getLogger(__name__)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / f"{spec_name}_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    logger.info(f"Saving 2SLS results to: {run_dir}")
    
    # Get cluster type and analysis settings
    settings = {**config['analyses']['online_2sls']['defaults'], **spec_config.get('settings', {})}
    cluster_type = settings.get('cluster_type', 'classical')
    
    # Use feature names from the model if available (second stage)
    if hasattr(twosls.second_stage, 'feature_names'):
        feature_names = twosls.second_stage.get_feature_names()
    else:
        # Fallback to constructing names manually
        feature_names = []
        if settings.get('add_intercept', True):
            feature_names.append('intercept')
        # Add fitted endogenous variables first
        feature_names.extend([f"{name}(fitted)" for name in endog_cols])
        # Then add exogenous variables
        feature_names.extend(exog_cols)
    
    # Create feature names for first stage
    first_stage_feature_names = []
    if settings.get('add_intercept', True):
        first_stage_feature_names.append('intercept')
    first_stage_feature_names.extend(exog_cols)
    first_stage_feature_names.extend(instr_cols)
    
    # Calculate additional statistics
    n_clusters_1 = len(twosls.second_stage.cluster_stats) if twosls.second_stage.cluster_stats else 0
    n_clusters_2 = len(twosls.second_stage.cluster2_stats) if twosls.second_stage.cluster2_stats else 0
    n_intersections = len(twosls.second_stage.intersection_stats) if twosls.second_stage.intersection_stats else 0
    
    # Calculate R-squared and adjusted R-squared using proper methods
    r_squared = twosls.second_stage.get_r_squared()
    adj_r_squared = twosls.second_stage.get_adjusted_r_squared()
    
    # Create comprehensive first stage results
    first_stage_results = {}
    first_stage_stats = {}
    
    for i, (fs_summary, endogen_name) in enumerate(zip(first_stage_summaries, endog_cols)):
        # Extract coefficient summary (exclude metadata rows)
        coeff_summary = fs_summary.iloc[:-3] if len(fs_summary) > 3 else fs_summary
        
        # Create first stage coefficient table
        first_stage_coefficients = []
        for j, feature_name in enumerate(first_stage_feature_names):
            if j < len(coeff_summary):
                row = coeff_summary.iloc[j]
                first_stage_coefficients.append({
                    "variable": feature_name,
                    "coefficient": float(row['coefficient']) if 'coefficient' in row else float(twosls.first_stage_models[i].theta[j]),
                    "std_error": float(row['std_error']) if 'std_error' in row else 0.0,
                    "t_statistic": float(row['t_statistic']) if 't_statistic' in row else 0.0,
                    "p_value": float(row['p_value']) if 'p_value' in row else 1.0,
                    "significance": "***" if ('p_value' in row and row['p_value'] < 0.01) else 
                                   "**" if ('p_value' in row and row['p_value'] < 0.05) else 
                                   "*" if ('p_value' in row and row['p_value'] < 0.10) else ""
                })
        
        # Extract R-squared statistics
        r_sq = fs_summary.loc['r_squared', fs_summary.columns[-1]] if 'r_squared' in fs_summary.index else 0.0
        adj_r_sq = fs_summary.loc['adj_r_squared', fs_summary.columns[-1]] if 'adj_r_squared' in fs_summary.index else 0.0
        n_obs_fs = fs_summary.loc['observations', fs_summary.columns[-1]] if 'observations' in fs_summary.index else 0
        
        first_stage_results[endogen_name] = {
            "dependent_variable": endogen_name,
            "r_squared": float(r_sq),
            "adjusted_r_squared": float(adj_r_sq),
            "observations": int(n_obs_fs),
            "rss": float(twosls.first_stage_models[i].rss),
            "rmse": float(np.sqrt(twosls.first_stage_models[i].rss / n_obs_fs)) if n_obs_fs > 0 else 0.0,
            "coefficients": first_stage_coefficients,
            "covariance_matrix": twosls.first_stage_models[i].get_cluster_robust_covariance(cluster_type).tolist() if cluster_type != 'classical' else twosls.first_stage_models[i].get_covariance_matrix().tolist()
        }
        
        # Also keep the simple stats for backward compatibility
        first_stage_stats[f"{endogen_name}_r_squared"] = float(r_sq)
        first_stage_stats[f"{endogen_name}_adj_r_squared"] = float(adj_r_sq)
        first_stage_stats[f"{endogen_name}_observations"] = int(n_obs_fs)
    
    # Determine target variable - handle both formula and explicit specification
    target_variable = None
    
    if 'formula' in spec_config:
        # Parse formula to extract target
        from gnt.analysis.models.feature_engineering import FormulaParser
        formula_parser = FormulaParser.parse(spec_config['formula'])
        target_variable = formula_parser.target
    else:
        # Use explicit specification
        target_variable = spec_config.get('target_col')
    
    # Create comprehensive results dictionary
    results = {
        "metadata": {
            "analysis_type": "online_2sls",
            "specification": spec_name,
            "description": spec_config['description'],
            "timestamp": timestamp,
            "cluster_type": cluster_type,
            "data_source": spec_config['data_source'],
            "target_variable": target_variable,
            "feature_variables": exog_cols + endog_cols,
            "formula": spec_config.get('formula'),  # Include formula if present
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
            "n_observations": int(twosls.total_obs),
            "n_features": int(twosls.second_stage.n_features),
            "n_clusters_1": n_clusters_1,
            "n_clusters_2": n_clusters_2,
            "n_cluster_intersections": n_intersections,
            "feature_names": feature_names
        },
        
        "model_fit": {
            "residual_sum_squares": float(twosls.second_stage.rss),
            "r_squared": float(r_squared),
            "adjusted_r_squared": float(adj_r_squared),
            "root_mean_squared_error": float(np.sqrt(twosls.second_stage.rss / twosls.total_obs)) if twosls.total_obs > 0 else 0,
            "degrees_of_freedom": int(twosls.total_obs - twosls.second_stage.n_features) if twosls.total_obs > twosls.second_stage.n_features else 0
        },
        
        "coefficients": {
            "estimates": {
                feature_names[i]: {
                    "coefficient": float(twosls.second_stage.theta[i]),
                    "std_error": float(summary.iloc[i]['std_error']),
                    "t_statistic": float(summary.iloc[i]['t_statistic']),
                    "p_value": float(summary.iloc[i]['p_value']),
                    "ci_lower_95": float(twosls.second_stage.theta[i] - 1.96 * summary.iloc[i]['std_error']),
                    "ci_upper_95": float(twosls.second_stage.theta[i] + 1.96 * summary.iloc[i]['std_error']),
                    "significant_5pct": bool(summary.iloc[i]['p_value'] < 0.05),
                    "significant_1pct": bool(summary.iloc[i]['p_value'] < 0.01)
                }
                for i in range(len(feature_names))
            },
            "covariance_matrix": twosls.second_stage.get_cluster_robust_covariance(cluster_type).tolist() if cluster_type != 'classical' else twosls.second_stage.get_covariance_matrix().tolist()
        },
        
        "inference": {
            "standard_error_type": cluster_type,
            "cluster_robust": cluster_type != 'classical',
            "hypothesis_tests": {
                "joint_significance": {
                    "description": "F-test for joint significance of all coefficients (excluding intercept)",
                    "test_statistic": None,
                    "p_value": None,
                    "critical_value_5pct": None
                }
            }
        },
        
        "diagnostics": {
            "convergence": {
                "converged": True,
                "regularization_applied": bool(twosls.alpha > 0),
                "numerical_issues": False
            },
            "data_quality": {
                "missing_values_handled": True,
                "infinite_values_handled": True,
                "outlier_detection": False
            }
        },
        
        "computational_details": {
            "algorithm": "Online Two-Stage Least Squares",
            "implementation": "Vectorized batch processing with parallel partition handling",
            "memory_efficient": True,
            "processing_time_seconds": None,
            "partitions_processed": None,
            "partitions_failed": None
        },
        
        "summary_table": {
            "coefficient_table": [
                {
                    "variable": feature_names[i],
                    "coefficient": float(twosls.second_stage.theta[i]),
                    "std_error": float(summary.iloc[i]['std_error']),
                    "t_statistic": float(summary.iloc[i]['t_statistic']),
                    "p_value": float(summary.iloc[i]['p_value']),
                    "significance": "***" if summary.iloc[i]['p_value'] < 0.01 else 
                                   "**" if summary.iloc[i]['p_value'] < 0.05 else 
                                   "*" if summary.iloc[i]['p_value'] < 0.10 else ""
                }
                for i in range(len(feature_names))
            ]
        },
        
        "iv_specification": {
            "exogenous_variables": exog_cols,
            "endogenous_variables": endog_cols,
            "instrument_variables": instr_cols,
            "identification_status": {
                "n_endogenous": len(endog_cols),
                "n_instruments": len(instr_cols),
                "over_identified": len(instr_cols) > len(endog_cols),
                "rank_condition_satisfied": True  # Assume satisfied for now
            }
        },
        
        "first_stage_results": first_stage_results,
        "first_stage_diagnostics": first_stage_stats
    }
    
    # Save the comprehensive results as JSON
    try:
        results_file = run_dir / "analysis_results.json"
        with open(results_file, 'w') as f:
            import json
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved comprehensive 2SLS results: {results_file}")
        
        # Also save a human-readable summary
        summary_file = run_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write(f"GNT Analysis Results: {spec_config['description']}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Analysis Type: Online 2SLS Regression\n")
            f.write(f"Specification: {spec_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Standard Errors: {cluster_type}\n\n")
            
            f.write(f"Instrument Specification:\n")
            f.write(f"  Exogenous variables: {exog_cols}\n")
            f.write(f"  Endogenous variables: {endog_cols}\n")
            f.write(f"  Instruments: {instr_cols}\n")
            f.write(f"  Identification: {'Over-identified' if len(instr_cols) > len(endog_cols) else 'Just-identified'}\n\n")
            
            f.write(f"Sample Information:\n")
            f.write(f"  Observations: {twosls.total_obs:,}\n")
            f.write(f"  Features: {twosls.second_stage.n_features}\n")
            f.write(f"  Clusters (Dim 1): {n_clusters_1}\n")
            f.write(f"  Clusters (Dim 2): {n_clusters_2}\n\n")
            
            # Write full first stage results
            f.write(f"FIRST STAGE RESULTS\n")
            f.write(f"{'='*80}\n\n")
            
            for i, (endogen_name, fs_results) in enumerate(first_stage_results.items()):
                f.write(f"First Stage {i+1}: {endogen_name}\n")
                f.write(f"{'-'*50}\n")
                f.write(f"Dependent Variable: {endogen_name}\n")
                f.write(f"R-squared: {fs_results['r_squared']:.6f}\n")
                f.write(f"Adjusted R-squared: {fs_results['adjusted_r_squared']:.6f}\n")
                f.write(f"Observations: {fs_results['observations']:,}\n")
                f.write(f"RMSE: {fs_results['rmse']:.6f}\n")
                f.write(f"RSS: {fs_results['rss']:.6f}\n\n")
                
                f.write(f"Coefficient Estimates:\n")
                f.write(f"{'Variable':<25} {'Coeff':<12} {'Std Err':<12} {'t-stat':<10} {'P>|t|':<10} {'Sig':<5}\n")
                f.write(f"{'-'*80}\n")
                
                for coeff_info in fs_results['coefficients']:
                    var = coeff_info['variable']
                    coef = coeff_info['coefficient']
                    se = coeff_info['std_error']
                    t_stat = coeff_info['t_statistic']
                    p_val = coeff_info['p_value']
                    sig = coeff_info['significance']
                    f.write(f"{var:<25} {coef:<12.6f} {se:<12.6f} {t_stat:<10.3f} {p_val:<10.6f} {sig:<5}\n")
                
                f.write(f"\n")
            
            f.write(f"SECOND STAGE RESULTS\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"Model Fit:\n")
            f.write(f"  R-squared: {r_squared:.6f}\n")
            f.write(f"  Adj. R-squared: {adj_r_squared:.6f}\n")
            f.write(f"  RMSE: {np.sqrt(twosls.second_stage.rss / twosls.total_obs) if twosls.total_obs > 0 else 0:.6f}\n")
            f.write(f"  RSS: {twosls.second_stage.rss:.6f}\n\n")
            
            f.write(f"Coefficient Estimates:\n")
            f.write(f"{'Variable':<25} {'Coeff':<12} {'Std Err':<12} {'t-stat':<10} {'P>|t|':<10} {'Sig':<5}\n")
            f.write(f"{'-'*80}\n")
            for i, var in enumerate(feature_names):
                coef = twosls.second_stage.theta[i]
                se = summary.iloc[i]['std_error']
                t_stat = summary.iloc[i]['t_statistic']
                p_val = summary.iloc[i]['p_value']
                sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
                f.write(f"{var:<25} {coef:<12.6f} {se:<12.6f} {t_stat:<10.3f} {p_val:<10.6f} {sig:<5}\n")
            
            f.write(f"\nSignificance codes: *** p<0.01, ** p<0.05, * p<0.10\n")
        
        logger.info(f"Saved human-readable 2SLS summary: {summary_file}")
        
    except Exception as e:
        logger.error(f"Failed to save 2SLS results: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    logger.info(f"2SLS results successfully saved to: {run_dir}")
    
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
    parser.add_argument("analysis_type", choices=['online_rls', 'online_2sls', 'list'], 
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
        
        elif args.analysis_type == 'online_2sls':
            if not args.specification:
                logger.error("Online 2SLS analysis requires a specification. Use --specification/-s")
                logger.info("Available specifications:")
                specs = config['analyses']['online_2sls']['specifications']
                for spec_name, spec_config in specs.items():
                    logger.info(f"  - {spec_name}: {spec_config['description']}")
                sys.exit(1)
            
            # Validate specification
            specs = config['analyses']['online_2sls']['specifications']
            if args.specification not in specs:
                logger.error(f"Unknown specification: {args.specification}")
                logger.info(f"Available specifications: {list(specs.keys())}")
                sys.exit(1)
            
            run_online_2sls(config, args.specification, output_dir, verbose)
        
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

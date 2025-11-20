import os
import sys
import yaml
import json  # Add missing import
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

# Add streamreg package to path if it's a separate project
streamreg_path = project_root.parent / "streamreg" / "src"
if streamreg_path.exists():
    sys.path.insert(0, str(streamreg_path))

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

def load_continent_mapping() -> Dict[str, List[int]]:
    """
    Load continent to country ID mapping.
    
    Returns:
    --------
    dict mapping continent codes to lists of country IDs
    """
    logger = logging.getLogger(__name__)
    
    # Paths relative to project root
    continents_path = project_root / "data_nobackup" / "misc" / "raw" / "continents" / "continents.csv"
    ids_path = project_root / "data_nobackup" / "misc" / "processed" / "stage_2" / "gadm" / "country_code_mapping.json"
    
    if not continents_path.exists() or not ids_path.exists():
        logger.warning("Continent mapping files not found. Geographic filtering unavailable.")
        return {}
    
    try:
        # Load continent table
        continents = pd.read_csv(continents_path)
        
        # Load country ID mapping
        import json
        with open(ids_path, "r") as f:
            country_to_id = json.load(f)
        
        # Build continent -> country IDs mapping
        continent_map = {}
        for continent_code in continents['Continent_Code'].unique():
            country_codes = continents.query(
                f"Continent_Code == '{continent_code}'"
            )['Three_Letter_Country_Code'].tolist()
            
            # Map country codes to IDs, filtering out missing ones
            country_ids = [
                country_to_id[code] 
                for code in country_codes 
                if code in country_to_id
            ]
            
            continent_map[continent_code] = country_ids
        
        logger.debug(f"Loaded continent mapping: {len(continent_map)} continents")
        return continent_map
        
    except Exception as e:
        logger.warning(f"Failed to load continent mapping: {e}")
        return {}


def load_subset(subset_name: str) -> List[int]:
    """
    Load country IDs from a subset file.
    
    Parameters:
    -----------
    subset_name : str
        Name of the subset (e.g., 'AF', 'asia', 'custom_developed')
        Can be:
        - Continent code: 'AF', 'EU', 'AS', 'NA', 'SA', 'OC'
        - Full subset filename: 'continent_af.json'
        - Custom subset: 'custom_mysubset'
    
    Returns:
    --------
    list of int
        Country IDs in the subset
    """
    logger = logging.getLogger(__name__)
    
    # Path to subsets directory
    subsets_dir = project_root / "data_nobackup" / "subsets"
    
    if not subsets_dir.exists():
        logger.error(f"Subsets directory not found: {subsets_dir}")
        logger.info("Run 'python -m gnt.analysis.subsets' to generate subset files")
        raise FileNotFoundError(f"Subsets directory not found: {subsets_dir}")
    
    # Determine subset file path
    subset_file = None
    
    # Try as continent code (e.g., 'AF' -> 'continent_af.json')
    if len(subset_name) == 2 and subset_name.isupper():
        subset_file = subsets_dir / f"continent_{subset_name.lower()}.json"
    
    # Try as full filename
    elif subset_name.endswith('.json'):
        subset_file = subsets_dir / subset_name
    
    # Try with .json extension
    else:
        subset_file = subsets_dir / f"{subset_name}.json"
    
    if not subset_file.exists():
        # List available subsets
        available = [f.stem for f in subsets_dir.glob("*.json")]
        logger.error(f"Subset file not found: {subset_file}")
        logger.info(f"Available subsets: {available}")
        raise FileNotFoundError(f"Subset '{subset_name}' not found. Available: {available}")
    
    # Load subset
    try:
        with open(subset_file, 'r') as f:
            data = json.load(f)
        
        country_ids = data['country_ids']
        logger.info(f"Loaded subset '{data.get('name', subset_name)}': {len(country_ids)} countries")
        return country_ids
        
    except Exception as e:
        logger.error(f"Failed to load subset from {subset_file}: {e}")
        raise


def build_geographic_query(spec_config: Dict[str, Any]) -> Optional[str]:
    """
    Build query string for geographic filtering using subset files.
    
    Parameters:
    -----------
    spec_config : dict
        Specification configuration
    
    Returns:
    --------
    query : str or None
        Query string for filtering, or None if no geographic filter
    """
    # Check for subset-based filter
    subset_name = spec_config.get('subset')
    country_filter = spec_config.get('countries')
    country_col = spec_config.get('country_col', 'country') 
    
    queries = []
    
    if subset_name:
        # Load country IDs from subset file
        country_ids = load_subset(subset_name)
        if country_ids:
            queries.append(f"{country_col}.isin({country_ids})")
    
    if country_filter:
        # country_filter should be a list of country IDs or codes
        if isinstance(country_filter, list):
            queries.append(f"{country_col}.isin({country_filter})")
        else:
            queries.append(f"{country_col} == {country_filter}")
    
    if queries:
        return " & ".join(f"({q})" for q in queries)
    
    return None


def run_online_rls(config: Dict[str, Any], spec_name: str, 
                   output_dir: Optional[str] = None, verbose: bool = True) -> Any:
    """Run Online RLS analysis with specified configuration."""
    logger = logging.getLogger(__name__)
    
    # Get analysis configuration
    rls_config = config['analyses']['online_rls']
    spec_config = rls_config['specifications'][spec_name]
    defaults = rls_config['defaults']
    
    logger.info(f"Running Online RLS analysis: {spec_config['description']}")
    logger.info(f"Data source: {spec_config['data_source']}")
    
    # Merge settings
    settings = {**defaults, **spec_config.get('settings', {})}
    
    # Import new API - updated import path
    from streamreg.api import OLS
    
    # Setup cluster variable
    cluster = None
    if settings.get('cluster_type') == 'one_way':
        cluster = spec_config.get('cluster1_col')
    elif settings.get('cluster_type') == 'two_way':
        cluster = [spec_config.get('cluster1_col'), spec_config.get('cluster2_col')]
    
    # Get formula
    formula = spec_config.get('formula')
    if not formula:
        # Build formula from explicit specification
        features = spec_config.get('feature_cols', [])
        target = spec_config.get('target_col')
        formula = f"{target} ~ {' + '.join(features)}"
    
    # Build query string for geographic/other filtering
    geo_query = build_geographic_query(spec_config)
    
    # Combine with any user-specified query
    user_query = spec_config.get('query')
    if geo_query and user_query:
        query = f"({geo_query}) & ({user_query})"
    else:
        query = geo_query or user_query
    
    if query:
        logger.info(f"Applying data filter: {query}")
    
    # Create and fit model
    model = OLS(
        formula=formula,
        alpha=settings.get('alpha', 1e-3),
        forget_factor=settings.get('forget_factor', 1.0),
        chunk_size=settings.get('chunk_size', 10000),
        n_workers=settings.get('n_workers'),
        show_progress=settings.get('show_progress', True),
        se_type=settings.get('se_type', 'stata'),
        local_directory=settings.get('local_directory'),
        memory_limit=settings.get('memory_limit')
    )
    
    model.fit(spec_config['data_source'], cluster=cluster, query=query)
    
    # Print results
    logger.info(f"Analysis complete! Total observations: {model.n_obs_:,}")
    logger.info(f"R-squared: {model.r_squared_:.4f}")
    
    print("\n" + "="*80)
    print(f"Analysis: {spec_config['description']}")
    print(f"Standard Errors: {model._cluster_type}")
    print("="*80)
    print(model.summary().to_string())
    print(f"\nR²: {model.r_squared_:.4f} | Adj. R²: {model.results_.adj_r_squared:.4f} | N: {model.n_obs_:,}")
    print("="*80)
    
    # Save comprehensive results
    if output_dir:
        output_path = Path(output_dir) / 'online_rls' / spec_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save coefficients table
        summary_df = model.summary()
        summary_df.to_csv(output_path / 'coefficients.csv')
        
        # Build comprehensive results container
        results_container = {
            'metadata': {
                'analysis_type': 'online_rls',
                'spec_name': spec_name,
                'description': spec_config['description'],
                'timestamp': datetime.now().isoformat(),
                'formula': formula,
                'data_source': spec_config['data_source'],
                'query': query
            },
            'specification': {
                'settings': settings,
                'cluster_type': model._cluster_type,
                'se_type': settings.get('se_type', 'stata'),
                'cluster_vars': cluster if isinstance(cluster, list) else [cluster] if cluster else None
            },
            'model_statistics': {
                'n_obs': int(model.n_obs_),
                'n_features': len(model.coef_),
                'r_squared': float(model.r_squared_),
                'adj_r_squared': float(model.results_.adj_r_squared),
                'rss': float(model.results_.rss) if hasattr(model.results_, 'rss') else None,
                'tss': float(model.results_.tss) if hasattr(model.results_, 'tss') else None,
                'df_model': int(model.results_.df_model) if hasattr(model.results_, 'df_model') else None,
                'df_resid': int(model.results_.df_resid) if hasattr(model.results_, 'df_resid') else None
            },
            'coefficients': {
                'names': summary_df.index.tolist(),
                'estimates': summary_df['coef'].tolist(),
                'std_errors': summary_df['std err'].tolist(),
                't_statistics': summary_df['t'].tolist(),
                'p_values': summary_df['P>|t|'].tolist(),
                'conf_int_lower': summary_df['[0.025'].tolist(),
                'conf_int_upper': summary_df['0.975]'].tolist()
            }
        }
        
        # Create timestamp for results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save comprehensive JSON with timestamp
        json_filename = f'results_{timestamp}.json'
        with open(output_path / json_filename, 'w') as f:
            json.dump(results_container, f, indent=2)
        
        logger.info(f"Results saved to: {output_path / json_filename}")
    
    return model


def run_online_2sls(config: Dict[str, Any], spec_name: str, 
                    output_dir: Optional[str] = None, verbose: bool = True) -> Any:
    """Run Online 2SLS analysis with specified configuration."""
    logger = logging.getLogger(__name__)
    
    # Get analysis configuration
    twosls_config = config['analyses']['online_2sls']
    spec_config = twosls_config['specifications'][spec_name]
    defaults = twosls_config['defaults']
    
    logger.info(f"Running Online 2SLS analysis: {spec_config['description']}")
    
    # Merge settings
    settings = {**defaults, **spec_config.get('settings', {})}
    
    # Import new API - updated import path
    from streamreg.api import TwoSLS
    
    # Setup cluster variable
    cluster = None
    if settings.get('cluster_type') == 'one_way':
        cluster = spec_config.get('cluster1_col')
    elif settings.get('cluster_type') == 'two_way':
        cluster = [spec_config.get('cluster1_col'), spec_config.get('cluster2_col')]
    
    # Get formula and endogenous
    formula = spec_config.get('formula')
    if not formula:
        raise ValueError("2SLS requires formula specification")
    
    feature_engineering = spec_config.get('feature_engineering') or settings.get('feature_engineering')
    endogenous = None
    if feature_engineering and 'endogenous' in feature_engineering:
        endogenous = feature_engineering['endogenous']
    
    # Build query string for geographic/other filtering
    geo_query = build_geographic_query(spec_config)
    
    # Combine with any user-specified query
    user_query = spec_config.get('query')
    if geo_query and user_query:
        query = f"({geo_query}) & ({user_query})"
    else:
        query = geo_query or user_query
    
    if query:
        logger.info(f"Applying data filter: {query}")
    
    # Create and fit model
    model = TwoSLS(
        formula=formula,
        endogenous=endogenous,
        alpha=settings.get('alpha', 1e-3),
        forget_factor=settings.get('forget_factor', 1.0),
        chunk_size=settings.get('chunk_size', 10000),
        n_workers=settings.get('n_workers'),
        show_progress=settings.get('show_progress', True),
        se_type=settings.get('se_type', 'stata'),
        local_directory=settings.get('local_directory'),
        memory_limit=settings.get('memory_limit')
    )
    
    model.fit(spec_config['data_source'], cluster=cluster, query=query)
    
    # Print results
    logger.info(f"Analysis complete! Total observations: {model.results_.n_obs:,}")
    
    print("\n" + "="*80)
    print(f"Analysis: {spec_config['description']}")
    print("="*80)
    
    # Print first stage
    first_stage_summaries = model.summary(stage='first')
    print("\nFIRST STAGE RESULTS:")
    for stage_name, summary_df in first_stage_summaries.items():
        print(f"\n{stage_name}:")
        print(summary_df.to_string())
    
    # Print second stage
    second_stage_summary = model.summary(stage='second')
    print("\nSECOND STAGE RESULTS:")
    print(second_stage_summary.to_string())
    print(f"\nR²: {model.results_.r_squared:.4f} | N: {model.results_.n_obs:,}")
    print("="*80)
    
    # Save comprehensive results
    if output_dir:
        output_path = Path(output_dir) / 'online_2sls' / spec_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save second stage coefficients
        second_stage_summary.to_csv(output_path / 'second_stage_coefficients.csv')
        
        # Save first stage coefficients
        for stage_name, summary_df in first_stage_summaries.items():
            filename = f'first_stage_{stage_name.replace(" ", "_").lower()}.csv'
            summary_df.to_csv(output_path / filename)
        
        # Build comprehensive results container
        results_container = {
            'metadata': {
                'analysis_type': 'online_2sls',
                'spec_name': spec_name,
                'description': spec_config['description'],
                'timestamp': datetime.now().isoformat(),
                'formula': formula,
                'endogenous': endogenous,
                'data_source': spec_config['data_source'],
                'query': query
            },
            'specification': {
                'settings': settings,
                'cluster_type': model._cluster_type if hasattr(model, '_cluster_type') else None,
                'se_type': settings.get('se_type', 'stata'),
                'cluster_vars': cluster if isinstance(cluster, list) else [cluster] if cluster else None
            },
            'second_stage': {
                'model_statistics': {
                    'n_obs': int(model.results_.n_obs),
                    'n_features': len(model.results_.params),
                    'r_squared': float(model.results_.r_squared),
                    'adj_r_squared': float(model.results_.adj_r_squared) if hasattr(model.results_, 'adj_r_squared') else None,
                    'df_model': int(model.results_.df_model) if hasattr(model.results_, 'df_model') else None,
                    'df_resid': int(model.results_.df_resid) if hasattr(model.results_, 'df_resid') else None
                },
                'coefficients': {
                    'names': second_stage_summary.index.tolist(),
                    'estimates': second_stage_summary['coef'].tolist(),
                    'std_errors': second_stage_summary['std err'].tolist(),
                    't_statistics': second_stage_summary['t'].tolist(),
                    'p_values': second_stage_summary['P>|t|'].tolist(),
                    'conf_int_lower': second_stage_summary['[0.025'].tolist(),
                    'conf_int_upper': second_stage_summary['0.975]'].tolist()
                }
            },
            'first_stage': {}
        }
        
        # Add first stage results
        for stage_name, summary_df in first_stage_summaries.items():
            results_container['first_stage'][stage_name] = {
                'coefficients': {
                    'names': summary_df.index.tolist(),
                    'estimates': summary_df['coef'].tolist(),
                    'std_errors': summary_df['std err'].tolist(),
                    't_statistics': summary_df['t'].tolist(),
                    'p_values': summary_df['P>|t|'].tolist(),
                    'conf_int_lower': summary_df['[0.025'].tolist(),
                    'conf_int_upper': summary_df['0.975]'].tolist()
                }
            }
        
        # Create timestamp for results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save comprehensive JSON with timestamp
        json_filename = f'results_{timestamp}.json'
        with open(output_path / json_filename, 'w') as f:
            json.dump(results_container, f, indent=2)
        
        logger.info(f"Results saved to: {output_path / json_filename}")
    
    return model


def run_duckreg(config: Dict[str, Any], spec_name: str, 
                output_dir: Optional[str] = None, verbose: bool = True) -> Any:
    """Run DuckReg compressed OLS analysis with specified configuration."""
    logger = logging.getLogger(__name__)
    
    # Get analysis configuration
    duckreg_config = config['analyses']['duckreg']
    spec_config = duckreg_config['specifications'][spec_name]
    defaults = duckreg_config['defaults']
    
    logger.info(f"Running DuckReg analysis: {spec_config['description']}")
    logger.info(f"Data source: {spec_config['data_source']}")
    
    # Merge settings
    settings = {**defaults, **spec_config.get('settings', {})}
    
    # Import compressed_ols API
    from duckreg import compressed_ols
    
    # Get formula
    formula = spec_config.get('formula')
    if not formula:
        raise ValueError("DuckReg requires formula specification in format 'y ~ x1 + x2 | fe1 + fe2 | 0 | cluster'")
    
    # Build SQL WHERE clause for filtering
    geo_query = build_geographic_query(spec_config)
    user_query = spec_config.get('query')
    
    # Convert pandas query to SQL WHERE clause if needed
    sql_where = None
    if geo_query or user_query:
        # For DuckReg, we need SQL WHERE clause, not pandas query
        # If user provided pandas-style query, convert to SQL
        if user_query:
            # Simple conversion: replace & with AND, | with OR
            sql_where = user_query.replace(' & ', ' AND ').replace(' | ', ' OR ')
        
        # For geographic filtering via subset, build SQL IN clause
        if geo_query:
            # Extract country column and IDs from the pandas query
            subset_name = spec_config.get('subset')
            if subset_name:
                country_col = spec_config.get('country_col', 'country')
                country_ids = load_subset(subset_name)
                geo_sql = f"{country_col} IN ({','.join(map(str, country_ids))})"
                
                if sql_where:
                    sql_where = f"({geo_sql}) AND ({sql_where})"
                else:
                    sql_where = geo_sql
    
    if sql_where:
        logger.info(f"Applying data filter: {sql_where}")
    
    # Prepare DuckDB configuration
    duckdb_kwargs = settings.get('duckdb_kwargs', {})
    
    # Compute scratch directory for database
    # Use ${WD}/scratch_nobackup/${SLURM_JOB_ID} if available
    wd = os.environ.get('WD', project_root)
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'local')
    scratch_path = Path(wd) / 'scratch_nobackup' / slurm_job_id / 'duckreg' / f'analysis_{spec_name}'
    scratch_path.mkdir(exist_ok=True, parents=True)
    
    # Create unique database name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    db_name = str(scratch_path / f'duckreg_{spec_name}_{timestamp}.db')
    
    logger.info(f"Using scratch database at: {db_name}")
    logger.info(f"SLURM_JOB_ID: {slurm_job_id}")
    logger.info(f"WD: {wd}")
    
    # Create and fit model
    model = compressed_ols(
        formula=formula,
        data=spec_config['data_source'],
        subset=sql_where,
        n_bootstraps=settings.get('n_bootstraps', 100),
        round_strata=settings.get('round_strata'),
        seed=settings.get('seed', 42),
        fe_method=settings.get('fe_method', 'mundlak'),
        cache_dir=settings.get('cache_dir'),
        duckdb_kwargs=duckdb_kwargs,
        db_name=db_name  # Pass explicit db_name to write to scratch
    )
    
    # Print results
    logger.info(f"Analysis complete!")
    
    # Read parameters directly from model object
    point_estimate = model.point_estimate.flatten()
    has_vcov = hasattr(model, 'vcov') and model.vcov is not None
    
    # Get coefficient names
    if hasattr(model, 'coef_names_'):
        coef_names = model.coef_names_
    else:
        coef_names = [f'coef_{i}' for i in range(len(point_estimate))]
    
    # Build results DataFrame
    if has_vcov:
        std_err = np.sqrt(np.diag(model.vcov))
        
        results_df = pd.DataFrame({
            'Coefficient': point_estimate,
            'Std. Error': std_err,
            't-stat': point_estimate / std_err
        }, index=coef_names)
        
        # Add p-values and confidence intervals
        from scipy import stats
        t_stats = point_estimate / std_err
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
        ci_lower = point_estimate - 1.96 * std_err
        ci_upper = point_estimate + 1.96 * std_err
        
        results_df['P>|t|'] = p_values
        results_df['[0.025'] = ci_lower
        results_df['0.975]'] = ci_upper
    else:
        results_df = pd.DataFrame({
            'Coefficient': point_estimate
        }, index=coef_names)
    
    print("\n", results_df.to_string())
    print("="*80)
    
    # Save comprehensive results
    if output_dir:
        output_path = Path(output_dir) / 'duckreg' / spec_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get compression statistics from model
        compression_stats = {}
        if hasattr(model, 'df_compressed'):
            compression_stats['n_compressed_rows'] = len(model.df_compressed)
            compression_stats['n_observations'] = int(model.df_compressed['count'].sum()) if 'count' in model.df_compressed.columns else None
            compression_stats['compression_ratio'] = compression_stats['n_compressed_rows'] / compression_stats['n_observations'] if compression_stats['n_observations'] else None
            compression_stats['has_standard_errors'] = has_vcov
        
        # Generate human-readable summary
        text_summary = format_duckreg_summary(spec_config, settings, results_df, model, compression_stats)
        
        # Print to console
        print("\n" + text_summary)
        
        # Build comprehensive results container
        results_container = {
            'metadata': {
                'analysis_type': 'duckreg',
                'spec_name': spec_name,
                'description': spec_config['description'],
                'timestamp': datetime.now().isoformat(),
                'formula': formula,
                'data_source': spec_config['data_source'],
                'query': sql_where
            },
            'specification': {
                'settings': settings,
                'fe_method': settings.get('fe_method', 'mundlak'),
                'n_bootstraps': settings.get('n_bootstraps', 100),
                'round_strata': settings.get('round_strata'),
                'seed': settings.get('seed', 42),
                'duckdb_kwargs': duckdb_kwargs,
                'outcome_vars': model.outcome_vars if hasattr(model, 'outcome_vars') else None,
                'covariates': model.covariates if hasattr(model, 'covariates') else None,
                'fe_cols': model.fe_cols if hasattr(model, 'fe_cols') else None,
                'cluster_col': model.cluster_col if hasattr(model, 'cluster_col') else None
            },
            'model_statistics': {
                'n_bootstraps': settings.get('n_bootstraps', 100),
                'compression_method': settings.get('fe_method', 'mundlak'),
                'has_standard_errors': has_vcov,
                'vcov_available': has_vcov,
                'n_coefficients': len(coef_names),
                'estimator_type': type(model).__name__,
                **compression_stats
            },
            'coefficients': {
                'names': coef_names,
                'estimates': results_df['Coefficient'].tolist(),
            }
        }
        
        # Add standard errors and inference if available
        if has_vcov:
            results_container['coefficients'].update({
                'std_errors': results_df['Std. Error'].tolist(),
                't_statistics': results_df['t-stat'].tolist(),
                'p_values': results_df['P>|t|'].tolist(),
                'conf_int_lower': results_df['[0.025'].tolist(),
                'conf_int_upper': results_df['0.975]'].tolist()
            })
            
            # Add variance-covariance matrix
            vcov_array = model.vcov if hasattr(model.vcov, 'tolist') else np.asarray(model.vcov)
            results_container['vcov_matrix'] = vcov_array.tolist()
        
        # Add SQL queries if available
        if hasattr(model, 'agg_query'):
            results_container['sql_queries'] = {
                'aggregation_query': model.agg_query
            }
            if hasattr(model, 'design_matrix_query'):
                results_container['sql_queries']['design_matrix_query'] = model.design_matrix_query
            if hasattr(model, 'bootstrap_query'):
                results_container['sql_queries']['bootstrap_query'] = model.bootstrap_query
        
        # Create timestamp for results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save text summary with timestamp
        txt_filename = f'results_{timestamp}.txt'
        with open(output_path / txt_filename, 'w') as f:
            f.write(text_summary)
        
        logger.info(f"Text summary saved to: {output_path / txt_filename}")
        
        # Save comprehensive JSON with timestamp
        json_filename = f'results_{timestamp}.json'
        with open(output_path / json_filename, 'w') as f:
            json.dump(results_container, f, indent=2)
        
        logger.info(f"JSON results saved to: {output_path / json_filename}")
        logger.info(f"Database location: {db_name}")
    else:
        # If no output_dir, still generate and print summary for feedback
        compression_stats = {}
        if hasattr(model, 'df_compressed'):
            compression_stats['n_compressed_rows'] = len(model.df_compressed)
            compression_stats['n_observations'] = int(model.df_compressed['count'].sum()) if 'count' in model.df_compressed.columns else None
            compression_stats['compression_ratio'] = compression_stats['n_compressed_rows'] / compression_stats['n_observations'] if compression_stats['n_observations'] else None
            compression_stats['has_standard_errors'] = has_vcov
        
        text_summary = format_duckreg_summary(spec_config, settings, results_df, model, compression_stats)
        print("\n" + text_summary)
    
    return model


def format_duckreg_summary(spec_config: Dict[str, Any], settings: Dict[str, Any], 
                           results_df: pd.DataFrame, model: Any, 
                           compression_stats: Dict[str, Any]) -> str:
    """Format DuckReg results as human-readable text summary."""
    summary_lines = []
    
    summary_lines.append("=" * 80)
    summary_lines.append("DUCKREG REGRESSION ANALYSIS RESULTS")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    # Specification info
    summary_lines.append("SPECIFICATION")
    summary_lines.append("-" * 80)
    summary_lines.append(f"Name:        {spec_config.get('description', 'N/A')}")
    summary_lines.append(f"Formula:     {spec_config.get('formula', 'N/A')}")
    summary_lines.append(f"Data Source: {spec_config.get('data_source', 'N/A')}")
    summary_lines.append(f"Timestamp:   {datetime.now().isoformat()}")
    summary_lines.append("")
    
    # Analysis settings
    summary_lines.append("ANALYSIS SETTINGS")
    summary_lines.append("-" * 80)
    summary_lines.append(f"Fixed Effects Method: {settings.get('fe_method', 'N/A')}")
    summary_lines.append(f"Bootstrap Iterations: {settings.get('n_bootstraps', 0)}")
    summary_lines.append(f"Random Seed:          {settings.get('seed', 'N/A')}")
    summary_lines.append(f"Round Strata Decimals: {settings.get('round_strata', 'None')}")
    summary_lines.append("")
    
    # Data compression statistics
    summary_lines.append("DATA COMPRESSION STATISTICS")
    summary_lines.append("-" * 80)
    if compression_stats.get('n_observations'):
        n_obs = compression_stats['n_observations']
        n_compressed = compression_stats['n_compressed_rows']
        ratio = compression_stats['compression_ratio']
        summary_lines.append(f"Total Observations:    {n_obs:,}")
        summary_lines.append(f"Compressed Rows:       {n_compressed:,}")
        if ratio:
            summary_lines.append(f"Compression Ratio:     {ratio:.4f} ({ratio*100:.2f}%)")
    summary_lines.append("")
    
    # Model statistics
    summary_lines.append("MODEL STATISTICS")
    summary_lines.append("-" * 80)
    summary_lines.append(f"Number of Coefficients: {len(results_df)}")
    summary_lines.append(f"Standard Errors:        {'Available (Bootstrap)' if compression_stats.get('has_standard_errors') else 'Not Available'}")
    summary_lines.append("")
    
    # Regression results table
    summary_lines.append("REGRESSION RESULTS")
    summary_lines.append("-" * 80)
    summary_lines.append(results_df.to_string())
    summary_lines.append("")
    
    # Summary statistics
    summary_lines.append("=" * 80)
    summary_lines.append("ANALYSIS COMPLETE")
    summary_lines.append("=" * 80)
    
    return "\n".join(summary_lines)


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
    parser.add_argument("analysis_type", choices=['online_rls', 'online_2sls', 'duckreg', 'list'], 
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
    parser.add_argument("--local_directory",
                       help="Override local directory for model output")
    parser.add_argument("--dask_memory_limit",
                       help="Override Dask memory limit per worker (e.g., '4GB', '32GB')")
    
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
    
    # Apply local_directory override if provided
    if args.local_directory:
        # Override in both analysis type defaults
        for analysis_type in config.get('analyses', {}).values():
            if 'defaults' in analysis_type:
                analysis_type['defaults']['local_directory'] = args.local_directory
        logger.info(f"Overriding local_directory from CLI: {args.local_directory}")
    
    # Apply memory_limit override if provided
    if args.dask_memory_limit:
        # Override in both analysis type defaults
        for analysis_type in config.get('analyses', {}).values():
            if 'defaults' in analysis_type:
                analysis_type['defaults']['memory_limit'] = args.dask_memory_limit
        logger.info(f"Overriding memory_limit from CLI: {args.dask_memory_limit}")
    
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
        
        elif args.analysis_type == 'duckreg':
            if not args.specification:
                logger.error("DuckReg analysis requires a specification. Use --specification/-s")
                logger.info("Available specifications:")
                specs = config['analyses']['duckreg']['specifications']
                for spec_name, spec_config in specs.items():
                    logger.info(f"  - {spec_name}: {spec_config['description']}")
                sys.exit(1)
            
            # Validate specification
            specs = config['analyses']['duckreg']['specifications']
            if args.specification not in specs:
                logger.error(f"Unknown specification: {args.specification}")
                logger.info(f"Available specifications: {list(specs.keys())}")
                sys.exit(1)
            
            run_duckreg(config, args.specification, output_dir, verbose)
        
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
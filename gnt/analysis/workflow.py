import os
import sys
import yaml
import json
import logging
import argparse
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


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    log_file = log_config.get('file')
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
        )
    else:
        logging.basicConfig(level=level, format=format_str)
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load analysis configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    def expand_env_vars(obj):
        if isinstance(obj, str):
            expanded = os.path.expandvars(obj)
            if expanded != obj:
                try:
                    if expanded.isdigit() or (expanded.startswith('-') and expanded[1:].isdigit()):
                        return int(expanded)
                    return float(expanded)
                except ValueError:
                    return expanded
            return expanded
        elif isinstance(obj, dict):
            return {k: expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [expand_env_vars(item) for item in obj]
        return obj
    
    return expand_env_vars(config)


def load_subset(subset_name: str) -> List[int]:
    """Load country IDs from a subset file."""
    logger = logging.getLogger(__name__)
    subsets_dir = project_root / "data_nobackup" / "subsets"
    
    if not subsets_dir.exists():
        raise FileNotFoundError(f"Subsets directory not found: {subsets_dir}")
    
    if len(subset_name) == 2 and subset_name.isupper():
        subset_file = subsets_dir / f"continent_{subset_name.lower()}.json"
    elif subset_name.endswith('.json'):
        subset_file = subsets_dir / subset_name
    else:
        subset_file = subsets_dir / f"{subset_name}.json"
    
    if not subset_file.exists():
        available = [f.stem for f in subsets_dir.glob("*.json")]
        raise FileNotFoundError(f"Subset '{subset_name}' not found. Available: {available}")
    
    with open(subset_file, 'r') as f:
        data = json.load(f)
    
    country_ids = data['country_ids']
    logger.info(f"Loaded subset '{data.get('name', subset_name)}': {len(country_ids)} countries")
    return country_ids


def build_geographic_query(spec_config: Dict[str, Any]) -> Optional[str]:
    """Build query string for geographic filtering."""
    subset_name = spec_config.get('subset')
    country_filter = spec_config.get('countries')
    country_col = spec_config.get('country_col', 'country')
    
    queries = []
    if subset_name:
        country_ids = load_subset(subset_name)
        if country_ids:
            queries.append(f"{country_col}.isin({country_ids})")
    
    if country_filter:
        if isinstance(country_filter, list):
            queries.append(f"{country_col}.isin({country_filter})")
        else:
            queries.append(f"{country_col} == {country_filter}")
    
    return " & ".join(f"({q})" for q in queries) if queries else None


def get_compression_stats(model: Any) -> Dict[str, Any]:
    """Extract compression statistics from model."""
    stats = {
        'n_observations': getattr(model, 'n_obs', None),
        'n_compressed_rows': getattr(model, 'n_compressed_rows', None),
        'has_standard_errors': hasattr(model, 'vcov') and model.vcov is not None,
        'se_type': getattr(model, 'se', None),
    }
    
    if stats['n_compressed_rows'] is None and hasattr(model, 'df_compressed') and model.df_compressed is not None:
        stats['n_compressed_rows'] = len(model.df_compressed)
        if 'count' in model.df_compressed.columns:
            stats['n_observations'] = int(model.df_compressed['count'].sum())
    
    if stats['n_compressed_rows'] and stats['n_observations']:
        stats['compression_ratio'] = 1 - stats['n_compressed_rows'] / stats['n_observations']
    else:
        stats['compression_ratio'] = None
    
    return stats


def _extract_first_stage_statistics(model: Any, logger: logging.Logger) -> Dict[str, Any]:
    """Extract first stage statistics using the model's built-in methods."""
    first_stage_results = {}
    
    for endog_var, fs_result in model.first_stage.items():
        # Use the FirstStageResults.to_dict() method
        fs_info = fs_result.to_dict()
        first_stage_results[endog_var] = fs_info
        
        # Log F-statistic
        f_stat = fs_result.f_statistic
        is_weak = fs_result.is_weak_instrument
        if f_stat is not None:
            logger.info(f"First stage {endog_var}: F={f_stat:.2f}, weak={is_weak}")
    
    return first_stage_results


def run_duckreg(config: Dict[str, Any], spec_name: str, 
                output_dir: Optional[str] = None, verbose: bool = True,
                dataset_override: Optional[str] = None) -> Any:
    """Run DuckReg compressed OLS analysis."""
    logger = logging.getLogger(__name__)
    
    duckreg_config = config['analyses']['duckreg']
    spec_config = duckreg_config['specifications'][spec_name]
    defaults = duckreg_config['defaults']
    settings = {**defaults, **spec_config.get('settings', {})}
    
    data_source = dataset_override or spec_config['data_source']
    formula = spec_config.get('formula')
    if not formula:
        raise ValueError("DuckReg requires formula specification")
    
    logger.info(f"Running DuckReg analysis: {spec_config['description']}")
    logger.info(f"Data source: {data_source}")
    
    # Build SQL WHERE clause
    sql_where = None
    subset_name = spec_config.get('subset')
    if subset_name:
        country_col = spec_config.get('country_col', 'country')
        country_ids = load_subset(subset_name)
        sql_where = f"{country_col} IN ({','.join(map(str, country_ids))})"
    
    user_query = spec_config.get('query')
    if user_query:
        user_sql = user_query.replace(' & ', ' AND ').replace(' | ', ' OR ')
        sql_where = f"({sql_where}) AND ({user_sql})" if sql_where else user_sql
    
    if sql_where:
        logger.info(f"Applying filter: {sql_where}")
    
    # Setup database path
    wd = os.environ.get('WD', project_root)
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'local')
    scratch_path = Path(wd) / 'scratch_nobackup' / slurm_job_id / 'duckreg' / f'analysis_{spec_name}'
    scratch_path.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    db_name = str(scratch_path / f'duckreg_{spec_name}_{timestamp}.db')
    
    logger.info(f"Database: {db_name}")
    
    fitter = spec_config.get('fitter', settings.get('fitter', 'numpy'))
    se_method = spec_config.get('se_method', settings.get('se_method', 'analytical'))
    logger.info(f"Fitter: {fitter}, SE method: {se_method}")
    
    # Import and fit model
    from duckreg import compressed_ols
    
    model = compressed_ols(
        formula=formula,
        data=data_source,
        subset=sql_where,
        n_bootstraps=settings.get('n_bootstraps', 100),
        round_strata=settings.get('round_strata'),
        seed=settings.get('seed', 42),
        fe_method=settings.get('fe_method', 'mundlak'),
        duckdb_kwargs=settings.get('duckdb_kwargs', {}),
        db_name=db_name,
        n_jobs=settings.get('n_jobs', 1),
        se_method=se_method,
        fitter=fitter
    )
    
    logger.info("Analysis complete!")
    
    # Use unified summary functions
    compression_stats = get_compression_stats(model)
    compression_stats['se_method'] = se_method
    
    # Print results using model's unified summary methods
    results_df = model.summary_df()
    if verbose and not results_df.empty:
        print("\n", results_df.to_string())
        print("=" * 80)
    
    # Use model's print_summary if 2SLS, otherwise print text version
    if hasattr(model, 'print_summary'):
        print()
        model.print_summary(precision=4, include_diagnostics=True)
    
    # Save results if output_dir provided
    if output_dir:
        output_path = Path(output_dir) / 'duckreg' / spec_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get full model summary dict for comprehensive output
        model_summary = model.summary()
        
        results_container = {
            'metadata': {
                'analysis_type': 'duckreg',
                'spec_name': spec_name,
                'description': spec_config['description'],
                'timestamp': datetime.now().isoformat(),
                'formula': formula,
                'data_source': data_source,
                'query': sql_where
            },
            'specification': {
                'settings': settings,
                'fe_method': settings.get('fe_method', 'mundlak'),
                'se_method': se_method,
                'outcome_vars': model_summary.get('outcome_vars'),
                'covariates': model_summary.get('covariates'),
                'fe_cols': model_summary.get('fe_cols'),
                'cluster_col': model_summary.get('cluster_col'),
            },
            'model_statistics': {
                **compression_stats,
                'n_coefficients': len(results_df),
                'estimator_type': model_summary.get('estimator_type'),
            },
        }
        
        # Add coefficients from model's results
        if model_summary.get('coefficients'):
            results_container['coefficients'] = model_summary['coefficients']
        
        # Add vcov if available
        if compression_stats['has_standard_errors']:
            results_container['vcov_matrix'] = np.asarray(model.vcov).tolist()
        
        # Add first stage results for 2SLS using model's built-in methods
        if hasattr(model, 'first_stage') and model.first_stage:
            results_container['first_stage'] = _extract_first_stage_statistics(model, logger)
            results_container['specification'].update({
                'endogenous_vars': model_summary.get('endogenous_vars'),
                'instrument_vars': model_summary.get('instrument_vars'),
                'exogenous_vars': model_summary.get('exogenous_vars'),
            })
            results_container['model_statistics']['weak_instruments'] = model.has_weak_instruments()
            results_container['model_statistics']['first_stage_f_stats'] = model.get_first_stage_f_stats()
            
            logger.info(f"First stage results stored for {len(model.first_stage)} endogenous variable(s)")
        
        # Save files
        with open(output_path / f'results_{timestamp}.txt', 'w') as f:
            f.write(f"Analysis: {spec_config['description']}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Formula: {formula}\n")
            f.write("=" * 80 + "\n\n")
            f.write(results_df.to_string())
            f.write("\n\n" + "=" * 80 + "\n")
        
        with open(output_path / f'results_{timestamp}.json', 'w') as f:
            json.dump(results_container, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
    
    return model


def run_online_rls(config: Dict[str, Any], spec_name: str, 
                   output_dir: Optional[str] = None, verbose: bool = True,
                   dataset_override: Optional[str] = None) -> Any:
    """Run Online RLS analysis with specified configuration."""
    logger = logging.getLogger(__name__)
    
    rls_config = config['analyses']['online_rls']
    spec_config = rls_config['specifications'][spec_name]
    defaults = rls_config['defaults']
    settings = {**defaults, **spec_config.get('settings', {})}
    
    data_source = dataset_override or spec_config['data_source']
    logger.info(f"Running Online RLS: {spec_config['description']}")
    
    from streamreg.api import OLS
    
    cluster = None
    if settings.get('cluster_type') == 'one_way':
        cluster = spec_config.get('cluster1_col')
    elif settings.get('cluster_type') == 'two_way':
        cluster = [spec_config.get('cluster1_col'), spec_config.get('cluster2_col')]
    
    formula = spec_config.get('formula')
    if not formula:
        features = spec_config.get('feature_cols', [])
        target = spec_config.get('target_col')
        formula = f"{target} ~ {' + '.join(features)}"
    
    geo_query = build_geographic_query(spec_config)
    user_query = spec_config.get('query')
    query = f"({geo_query}) AND ({user_query})" if geo_query and user_query else (geo_query or user_query)
    
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
    
    model.fit(data_source, cluster=cluster, query=query)
    
    logger.info(f"Complete! N={model.n_obs_:,}, R²={model.r_squared_:.4f}")
    print(model.summary().to_string())
    
    if output_dir:
        output_path = Path(output_dir) / 'online_rls' / spec_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary_df = model.summary()
        summary_df.to_csv(output_path / 'coefficients.csv')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results = {
            'metadata': {'spec_name': spec_name, 'timestamp': datetime.now().isoformat()},
            'model_statistics': {'n_obs': int(model.n_obs_), 'r_squared': float(model.r_squared_)},
            'coefficients': {'names': summary_df.index.tolist(), 'estimates': summary_df['coef'].tolist()}
        }
        with open(output_path / f'results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    return model


def run_online_2sls(config: Dict[str, Any], spec_name: str, 
                    output_dir: Optional[str] = None, verbose: bool = True,
                    dataset_override: Optional[str] = None) -> Any:
    """Run Online 2SLS analysis."""
    logger = logging.getLogger(__name__)
    
    twosls_config = config['analyses']['online_2sls']
    spec_config = twosls_config['specifications'][spec_name]
    defaults = twosls_config['defaults']
    settings = {**defaults, **spec_config.get('settings', {})}
    
    data_source = dataset_override or spec_config['data_source']
    formula = spec_config.get('formula')
    if not formula:
        raise ValueError("2SLS requires formula")
    
    logger.info(f"Running Online 2SLS: {spec_config['description']}")
    
    from streamreg.api import TwoSLS
    
    cluster = None
    if settings.get('cluster_type') == 'one_way':
        cluster = spec_config.get('cluster1_col')
    elif settings.get('cluster_type') == 'two_way':
        cluster = [spec_config.get('cluster1_col'), spec_config.get('cluster2_col')]
    
    feature_engineering = spec_config.get('feature_engineering') or settings.get('feature_engineering')
    endogenous = feature_engineering.get('endogenous') if feature_engineering else None
    
    geo_query = build_geographic_query(spec_config)
    user_query = spec_config.get('query')
    query = f"({geo_query}) & ({user_query})" if geo_query and user_query else (geo_query or user_query)
    
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
    
    model.fit(data_source, cluster=cluster, query=query)
    
    logger.info(f"Complete! N={model.results_.n_obs:,}")
    print(model.summary(stage='second').to_string())
    
    return model


def list_analyses(config: Dict[str, Any]) -> None:
    """List available analyses."""
    print("\nAvailable analyses:")
    print("=" * 50)
    for name, cfg in config['analyses'].items():
        print(f"\n{name.upper()}: {cfg['description']}")
        if 'specifications' in cfg:
            for spec_name, spec_cfg in cfg['specifications'].items():
                print(f"  - {spec_name}: {spec_cfg['description']}")


def main():
    """Main entrypoint."""
    parser = argparse.ArgumentParser(description="GNT Analysis Pipeline")
    parser.add_argument("analysis_type", choices=['online_rls', 'online_2sls', 'duckreg', 'list'])
    parser.add_argument("--config", default="orchestration/configs/analysis.yaml")
    parser.add_argument("--specification", "-s")
    parser.add_argument("--output", "-o")
    parser.add_argument("--dataset")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = project_root / args.config
    if not config_path.exists():
        print(f"Config not found: {args.config}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    if args.debug:
        config.setdefault('logging', {})['level'] = 'DEBUG'
    
    logger = setup_logging(config)
    
    if args.analysis_type == 'list':
        list_analyses(config)
        return
    
    output_dir = args.output or config.get('output', {}).get('base_path', str(project_root / "output" / "analysis"))
    
    runners = {
        'online_rls': run_online_rls,
        'online_2sls': run_online_2sls,
        'duckreg': run_duckreg,
    }
    
    if args.analysis_type not in config['analyses']:
        logger.error(f"Unknown analysis: {args.analysis_type}")
        sys.exit(1)
    
    if not args.specification:
        logger.error(f"Specification required. Use -s/--specification")
        sys.exit(1)
    
    specs = config['analyses'][args.analysis_type]['specifications']
    if args.specification not in specs:
        logger.error(f"Unknown specification: {args.specification}. Available: {list(specs.keys())}")
        sys.exit(1)
    
    try:
        runners[args.analysis_type](config, args.specification, output_dir, not args.quiet, args.dataset)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
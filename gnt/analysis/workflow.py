import os
import sys
import yaml
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
import openpyxl
import shutil

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


def get_directory_size(path: Path) -> int:
    """Get total size of a directory in bytes."""
    try:
        if not path.exists():
            return 0
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
    except Exception:
        return 0


def bytes_to_gb(bytes_size: int) -> float:
    """Convert bytes to GB."""
    return bytes_size / (1024 ** 3)


def format_size_string(bytes_size: int, max_size_gb: int = 512) -> str:
    """Format byte size as GB string with a maximum cap."""
    size_gb = bytes_to_gb(bytes_size)
    # Triple the size, but cap at max_size_gb
    recommended_size_gb = min(size_gb * 3, max_size_gb)
    return f"{int(recommended_size_gb)}GB"


def load_config(config_path: str) -> Dict[str, Any]:
    """Load analysis configuration from Excel or YAML file."""
    config_path = Path(config_path)
    
    # Check file extension
    if config_path.suffix.lower() in ['.xlsx', '.xls']:
        # Load defaults from the YAML analysis configuration
        yaml_config_path = config_path.parent / "analysis.yaml"
        yaml_defaults = {
            'se_method': 'analytical',
            'fitter': 'duckdb',
            'n_bootstraps': 100,
            'seed': 42,
            'fe_method': 'mundlak',
            'round_strata': 5,
            'n_jobs': 1,
            'duckdb_kwargs': {}
        }
        
        if yaml_config_path.exists():
            try:
                with open(yaml_config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                if 'analyses' in yaml_config and 'duckreg' in yaml_config['analyses']:
                    yaml_defaults = yaml_config['analyses']['duckreg'].get('defaults', yaml_defaults)
            except Exception:
                pass  # Use hardcoded defaults if YAML loading fails
        
        # Load from Excel - read from "Models" sheet
        df = pd.read_excel(config_path, sheet_name='Models')
        
        # Build config structure from Excel data
        config = {
            'analyses': {
                'duckreg': {
                    'description': 'DuckReg compressed OLS/2SLS analysis',
                    'specifications': {},
                    'defaults': yaml_defaults  # Use defaults from YAML
                }
            },
            'output': {
                'base_path': str(project_root / "output" / "analysis")
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
        
        # Process each row as a specification
        for _, row in df.iterrows():
            model_name = row['model_name']
            
            # Build formula from components
            dependent = row['dependent']
            independent = row['independent']
            
            # Parse fixed effects (0 means none, otherwise column names)
            fixed_effects = []
            if pd.notna(row['fixed_effects']) and str(row['fixed_effects']) != '0':
                fixed_effects = [fe.strip() for fe in str(row['fixed_effects']).split(',')]
            
            # Parse instruments (0 means OLS, otherwise IV formula)
            instruments = []
            if pd.notna(row['instruments']) and str(row['instruments']) != '0':
                instruments = [inst.strip() for inst in str(row['instruments']).split(',')]
            
            # Build formula
            if instruments:
                # IV formula: dependent ~ exog + [endog ~ instruments] | fixed_effects
                formula = f"{dependent} ~ {independent}"
                if fixed_effects:
                    formula += " | " + " + ".join(fixed_effects)
            else:
                # OLS formula: dependent ~ independent | fixed_effects
                formula = f"{dependent} ~ {independent}"
                if fixed_effects:
                    formula += " | " + " + ".join(fixed_effects)
            
            # Build clustering specification
            cluster_col = None
            if pd.notna(row['clustering']) and str(row['clustering']) != '0':
                cluster_col = str(row['clustering'])
            
            # Build specification
            spec = {
                'description': f"{row.get('section', 'Analysis')} - {row.get('subsection', model_name)}",
                'data_source': str(row['data_source']),
                'formula': formula,
                'settings': {}
            }
            
            # Expand data_source path
            data_source_value = str(row['data_source'])
            # Handle both direct paths and simple filenames
            if not data_source_value.startswith('/') and not data_source_value.startswith('${'):
                # If it's a simple filename like "5km", expand it
                data_source_expanded = os.path.expandvars(f"${{WD}}/data_nobackup/assembled/{data_source_value}.parquet")
            else:
                # Otherwise expand as-is
                data_source_expanded = os.path.expandvars(data_source_value)
            
            spec['data_source'] = data_source_expanded
            
            # Calculate directory size and set max_temp_directory_size
            # First expand environment variables for size calculation
            data_source_for_size = os.path.expandvars(data_source_expanded)
            data_dir = Path(data_source_for_size.replace('.parquet', ''))
            dir_size = get_directory_size(data_dir)
            
            # Store the calculated temp dir size in settings for later use
            if dir_size > 0:
                calculated_size = format_size_string(dir_size)
                spec['settings']['_max_temp_directory_size'] = calculated_size
                # Log this for debugging
                logger.info(f"Model {model_name}: Data dir size ~{bytes_to_gb(dir_size):.1f}GB -> max_temp_directory_size: {calculated_size}")
            
            # Add clustering if specified
            if cluster_col:
                spec['cluster1_col'] = cluster_col
                spec['settings']['cluster_type'] = 'one_way'
            
            # Add query if specified
            if pd.notna(row.get('query')) and str(row['query']).strip():
                spec['query'] = str(row['query'])
            
            # Add subset if specified
            if pd.notna(row.get('subset')) and str(row['subset']).strip():
                spec['subset'] = str(row['subset'])
            
            # Override defaults with row-specific settings
            if pd.notna(row.get('se_method')):
                spec['se_method'] = str(row['se_method'])
            if pd.notna(row.get('fitter')):
                spec['fitter'] = str(row['fitter'])
            if pd.notna(row.get('n_bootstraps')):
                spec['settings']['n_bootstraps'] = int(row['n_bootstraps'])
            if pd.notna(row.get('seed')):
                spec['settings']['seed'] = int(row['seed'])
            if pd.notna(row.get('fe_method')):
                spec['settings']['fe_method'] = str(row['fe_method'])
            if pd.notna(row.get('round_strata')):
                spec['settings']['round_strata'] = int(row['round_strata'])
            if pd.notna(row.get('n_jobs')):
                spec['settings']['n_jobs'] = int(row['n_jobs'])
            
            config['analyses']['duckreg']['specifications'][model_name] = spec
        
        return config
    
    else:
        # Load from YAML (legacy support)
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
    
    # filter debug output handled in summary block below
    
    # Setup database path
    wd = os.environ.get('WD', project_root)
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'local')
    scratch_path = Path(wd) / 'scratch_nobackup' / slurm_job_id / 'duckreg' / f'analysis_{spec_name}'
    scratch_path.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    db_name = str(scratch_path / f'duckreg_{spec_name}_{timestamp}.db')
    
    # determine fitter and SE method
    fitter = spec_config.get('fitter', settings.get('fitter', 'duckdb'))
    se_method = spec_config.get('se_method', settings.get('se_method', 'analytical'))
    
    # Prepare duckdb_kwargs with environment variable expansion
    duckdb_kwargs = settings.get('duckdb_kwargs', {}).copy()
    
    # Expand environment variables in duckdb_kwargs
    for key, value in duckdb_kwargs.items():
        if isinstance(value, str):
            duckdb_kwargs[key] = os.path.expandvars(value)
    
    # If calculated max_temp_directory_size exists in settings, use it
    if '_max_temp_directory_size' in spec_config.get('settings', {}):
        duckdb_kwargs['max_temp_directory_size'] = spec_config['settings']['_max_temp_directory_size']
        # summary block will show duckdb_kwargs
    
    # Build a consolidated information block for logging
    info_lines = [
        "=== DuckReg analysis summary ===",
        f"Description: {spec_config['description']}",
        f"Data source: {data_source}",
        f"Formula: {formula}",
    ]
    if sql_where:
        info_lines.append(f"Filter: {sql_where}")
    info_lines.append("Settings:")
    for key, val in settings.items():
        info_lines.append(f"  {key}: {val}")
    info_lines.append(f"Fitter: {fitter}, SE method: {se_method}")
    info_lines.append(f"Database path: {db_name}")
    info_lines.append("=" * 30)
    logger.info("\n" + "\n".join(info_lines))

    # Import and fit model
    from duckreg import duckreg
    from duckreg.utils.summary import format_model_summary
    
    model = duckreg(
        formula=formula,
        data=data_source,
        subset=sql_where,
        n_bootstraps=settings.get('n_bootstraps', 0),
        round_strata=settings.get('round_strata'),
        seed=settings.get('seed', 42),
        fe_method=settings.get('fe_method', 'demean'),
        duckdb_kwargs=duckdb_kwargs,
        db_name=db_name,
        n_jobs=settings.get('n_jobs', 1),
        se_method=se_method,
        fitter=fitter
    )
    
    logger.info("Analysis complete!")
    
    # Get standardized model summary from duckreg
    model_summary = model.summary()
    
    # Format comprehensive output using summary.py formatter
    text_output = format_model_summary(model_summary, spec_config, precision=4)
    
    # Print to console
    print("\n" + text_output)
    
    # Save results if output_dir provided
    if output_dir:
        output_path = Path(output_dir) / 'duckreg' / spec_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Store results with workflow-specific context
        results_container = {
            # Analysis metadata (workflow-specific context)
            'analysis_metadata': {
                'analysis_type': 'duckreg',
                'spec_name': spec_name,
                'description': spec_config['description'],
                'formula': formula,
                'data_source': data_source,
                'query': sql_where,
                'settings': settings,
            },
            # Standardized model results from duckreg
            # Includes: version_info, model_spec, sample_info, coefficients, [first_stage]
            **model_summary,
        }
        
        # Log version info for tracking
        version_info = model_summary.get('version_info', {})
        duckreg_version = version_info.get('duckreg_version', 'unknown')
        logger.info(f"Results computed with duckreg version: {duckreg_version}")
        
        # Log first stage info for 2SLS
        if model_summary.get('first_stage'):
            logger.info(f"First stage results stored for {len(model_summary['first_stage'])} endogenous variable(s)")
        
        # Save comprehensive text output (same as console)
        with open(output_path / f'results_{timestamp}.txt', 'w') as f:
            f.write(text_output)
        
        # Save coefficients as CSV for easy loading
        coefficients_df = model.summary_df()
        if not coefficients_df.empty:
            coefficients_df.to_csv(output_path / 'coefficients.csv')
        
        # Save first stage results as CSV files (if IV)
        first_stage = model_summary.get('first_stage', {})
        for endog, fs_dict in first_stage.items():
            if isinstance(fs_dict, dict) and 'coef_names' in fs_dict:
                fs_df = pd.DataFrame({
                    'variable': fs_dict.get('coef_names', []),
                    'coefficient': fs_dict.get('coefficients', []),
                    'std_error': fs_dict.get('std_errors', []),
                    't_stat': fs_dict.get('t_statistics', []),
                    'p_value': fs_dict.get('p_values', []),
                })
                if 'ci_lower' in fs_dict and 'ci_upper' in fs_dict:
                    fs_df['ci_lower'] = fs_dict['ci_lower']
                    fs_df['ci_upper'] = fs_dict['ci_upper']
                
                safe_name = endog.replace(' ', '_').replace('/', '_')
                fs_df.to_csv(output_path / f'first_stage_{safe_name}.csv', index=False)
        
        # Save complete JSON with all metadata and results
        with open(output_path / f'results_{timestamp}.json', 'w') as f:
            json.dump(results_container, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
    
    return model


def cleanup_analysis_results(output_dir: str, dry_run: bool = False) -> None:
    """
    Cleanup regression results: keep only the latest result per model grouped by minor version (X in y.X.z).
    
    Reads duckreg version from result JSON files and groups results by the second version component.
    For each model and version group, keeps only the most recent result.
    
    Args:
        output_dir: Base output directory containing analysis results
        dry_run: If True, only show what would be deleted without actually deleting
    """
    logger = logging.getLogger(__name__)
    
    output_path = Path(output_dir) / 'duckreg'
    if not output_path.exists():
        logger.warning(f"DuckReg output directory not found: {output_path}")
        return
    
    total_deleted = 0
    total_kept = 0
    
    # Process each specification directory
    for spec_dir in output_path.iterdir():
        if not spec_dir.is_dir():
            continue
        
        # Collect result JSON files (which contain version info)
        result_files = list(spec_dir.glob("results_*.json"))
        if not result_files:
            logger.debug(f"No result files in {spec_dir.name}")
            continue
        
        # Group by minor version (X in y.X.z) and timestamp
        version_groups = {}  # version_minor -> {timestamp: [files]}
        
        for json_file in result_files:
            try:
                # Read version info from JSON
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract duckreg version
                version_info = data.get('version_info', {})
                duckreg_version = version_info.get('duckreg_version', 'unknown')
                
                # Parse version: y.X.z -> extract X (default to 0.0.0 if unparseable)
                if duckreg_version == 'unknown':
                    duckreg_version = '0.0.0'
                
                version_parts = duckreg_version.split('.')
                if len(version_parts) >= 2 and version_parts[1].isdigit():
                    version_minor = int(version_parts[1])
                else:
                    logger.warning(f"Could not parse version from {json_file.name}: {duckreg_version}, assuming 0.0.0")
                    version_minor = 0  # Default to 0.0.0 when unparseable
                
                # Extract timestamp from filename: results_YYYYMMDD_HHMMSS.json
                parts = json_file.stem.split('_', 1)
                if len(parts) == 2 and len(parts[1]) == 15:
                    timestamp = datetime.strptime(parts[1], '%Y%m%d_%H%M%S')
                else:
                    logger.warning(f"Invalid timestamp in filename: {json_file.name}")
                    continue
                
                # Group files
                if version_minor not in version_groups:
                    version_groups[version_minor] = {}
                if timestamp not in version_groups[version_minor]:
                    version_groups[version_minor][timestamp] = []
                
                # Find associated files (txt, json, csv)
                base_name = f"results_{parts[1]}"
                associated_files = [json_file]
                
                # Look for matching txt file
                txt_file = spec_dir / f"{base_name}.txt"
                if txt_file.exists():
                    associated_files.append(txt_file)
                
                version_groups[version_minor][timestamp].extend(associated_files)
                
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
                continue
        
        # For each version group, keep only the latest
        for version_minor, timestamp_dict in version_groups.items():
            if not timestamp_dict:
                continue
            
            sorted_timestamps = sorted(timestamp_dict.keys(), reverse=True)
            latest_timestamp = sorted_timestamps[0]
            
            version_str = f"0.{version_minor}.x" if version_minor >= 0 else "unknown"
            
            for ts, files in timestamp_dict.items():
                if ts == latest_timestamp:
                    logger.info(f"Keeping {spec_dir.name} (v{version_str}, {ts}): {len(files)} files")
                    total_kept += len(files)
                else:
                    for file_path in files:
                        if dry_run:
                            logger.info(f"Would delete: {file_path.relative_to(output_path)}")
                        else:
                            file_path.unlink()
                            logger.debug(f"Deleted: {file_path.relative_to(output_path)}")
                        total_deleted += 1
    
    if dry_run:
        logger.info(f"Dry run complete: would delete {total_deleted} files, keep {total_kept} files")
    else:
        logger.info(f"Cleanup complete: deleted {total_deleted} files, kept {total_kept} files")


def main():
    """Main entrypoint."""
    parser = argparse.ArgumentParser(description="GNT Analysis Pipeline (DuckReg)")
    parser.add_argument("--config", default="orchestration/configs/analysis.xlsx", 
                       help="Path to analysis config file (Excel or YAML)")
    parser.add_argument("--model", "-m", help="Model name (specification) to run")
    parser.add_argument("--list", "-l", action="store_true", help="List available models")
    parser.add_argument("--output", "-o", help="Output directory for results")
    parser.add_argument("--dataset", help="Override dataset path for analysis")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Disable verbose output")
    parser.add_argument("--se-method", choices=['analytical', 'bootstrap', 'none'], 
                       help="Standard error method (overrides config)")
    parser.add_argument("--fitter", choices=['numpy', 'duckdb'], 
                       help="Fitter type (overrides config)")
    
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
    
    # Handle --list option
    if args.list:
        specs = config['analyses']['duckreg']['specifications']
        if not specs:
            print("No models found in configuration")
            sys.exit(0)
        
        print("\nAvailable models:")
        print("=" * 80)
        for model_name, spec in specs.items():
            print(f"\n{model_name}")
            print(f"  Description: {spec.get('description', 'N/A')}")
            print(f"  Data source: {spec.get('data_source', 'N/A')}")
            print(f"  Formula: {spec.get('formula', 'N/A')}")
            print(f"  SE method: {spec.get('se_method', 'default')}")
            print(f"  Fitter: {spec.get('fitter', 'default')}")
        print("\n" + "=" * 80)
        sys.exit(0)
    
    # Require --model for actual execution
    if not args.model:
        logger.error("Model name required. Use --model <model_name> or --list to see available models")
        sys.exit(1)
    
    output_dir = args.output or config.get('output', {}).get('base_path', 
                                                            str(project_root / "output" / "analysis"))
    
    if 'duckreg' not in config.get('analyses', {}):
        logger.error("DuckReg analysis not configured in config file")
        sys.exit(1)
    
    specs = config['analyses']['duckreg']['specifications']
    if args.model not in specs:
        logger.error(f"Unknown model: {args.model}. Available: {list(specs.keys())}")
        sys.exit(1)
    
    # Apply CLI overrides
    if args.se_method:
        config['analyses']['duckreg']['defaults']['se_method'] = args.se_method
        logger.info(f"Overriding se_method from CLI: {args.se_method}")
    
    if args.fitter:
        config['analyses']['duckreg']['defaults']['fitter'] = args.fitter
        logger.info(f"Overriding fitter from CLI: {args.fitter}")
    
    try:
        logger.info(f"Running DuckReg model: {args.model}")
        run_duckreg(config, args.model, output_dir, not args.quiet, args.dataset)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
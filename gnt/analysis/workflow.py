import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add streamreg package to path if it's a separate project
streamreg_path = project_root.parent / "streamreg" / "src"
if streamreg_path.exists():
    sys.path.insert(0, str(streamreg_path))

# ---------------------------------------------------------------------------
# New canonical API — prefer importing from gnt.analysis directly:
#
#   from gnt.analysis import AnalysisConfig, run_duckreg, cleanup_analysis_results
#
# The functions below are kept for backward compatibility with run.py and other
# callers that import from this module directly.  New code should use the
# package-level API instead.
# ---------------------------------------------------------------------------




def load_config(config_path: str) -> Dict[str, Any]:
    """Load analysis configuration from an Excel workbook.

    Reads two sheets:
    - ``Settings``: columns ``key`` / ``value`` — global default settings.
    - ``Models``  : one row per model specification.
    """
    config_path = Path(config_path)
    if config_path.suffix.lower() not in ['.xlsx', '.xls']:
        raise ValueError(f"Configuration must be an Excel .xlsx file; got: {config_path.suffix}")

    # ── Settings sheet → defaults ───────────────────────────────────────────
    settings_df = pd.read_excel(config_path, sheet_name='Settings')
    defaults: Dict[str, Any] = {}
    for _, row in settings_df.iterrows():
        key = str(row['key']).strip()
        val = row['value']
        if pd.isna(val):
            continue
        expanded = os.path.expandvars(str(val).strip())
        try:
            defaults[key] = int(expanded)
        except ValueError:
            try:
                defaults[key] = float(expanded)
            except ValueError:
                defaults[key] = expanded

    # ── Models sheet → specifications ───────────────────────────────────────
    df = pd.read_excel(config_path, sheet_name='Models')
    specifications: Dict[str, Any] = {}
    for _, row in df.iterrows():
        model_name = str(row['model_name']).strip()

        # Formula construction
        dependent  = str(row['dependent']).strip()
        independent = str(row['independent']).strip()

        fixed_effects: List[str] = []
        fe_raw = str(row.get('fixed_effects', '0')).strip()
        if fe_raw not in ('0', 'nan', ''):
            fixed_effects = [fe.strip() for fe in fe_raw.split(',')]

        instruments_raw = str(row.get('instruments', '0')).strip()
        has_iv = instruments_raw not in ('0', 'nan', '')

        formula = f"{dependent} ~ {independent}"
        if fixed_effects:
            formula += " | " + " + ".join(fixed_effects)
        if has_iv:
            formula += f" | {instruments_raw}"

        # Data source: bare name → assembled parquet path
        data_source = str(row['data_source']).strip()
        if not data_source.startswith('/') and not data_source.startswith('$'):
            data_source = os.path.expandvars(
                f"data_nobackup/assembled/{data_source}.parquet"
            )
        else:
            data_source = os.path.expandvars(data_source)

        # Per-model settings overrides (only columns explicitly present)
        _SETTING_KEYS = (
            'se_method', 'fitter', 'fe_method', 'round_strata', 'seed',
            'n_bootstraps', 'threads', 'memory_limit', 'max_temp_directory_size',
        )
        model_settings: Dict[str, Any] = {}
        for key in _SETTING_KEYS:
            val = row.get(key)
            if val is not None and pd.notna(val) and str(val).strip() not in ('', 'nan'):
                if isinstance(val, float) and val == int(val):
                    val = int(val)
                model_settings[key] = val

        spec: Dict[str, Any] = {
            'description': (
                f"{row.get('section', 'Analysis')} - {row.get('subsection', model_name)}"
            ),
            'data_source': data_source,
            'formula': formula,
            'settings': model_settings,
        }

        # Clustering
        cluster = str(row.get('clustering', '0')).strip()
        if cluster not in ('0', 'nan', ''):
            spec['cluster1_col'] = cluster

        # Optional SQL filter
        query = str(row.get('query', '')).strip()
        if query and query != 'nan':
            spec['query'] = query

        # Optional geographic subset (country list)
        subset = str(row.get('subset', '')).strip()
        if subset and subset != 'nan':
            spec['subset'] = subset

        specifications[model_name] = spec

    return {
        'analyses': {
            'duckreg': {
                'specifications': specifications,
                'defaults': defaults,
            }
        },
        'output': {'base_path': str(project_root / 'output' / 'analysis')},
    }


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
    
    # Fitter and SE method (per-model settings override defaults)
    fitter = settings.get('fitter', 'duckdb')
    se_method_raw = settings.get('se_method', 'HC1')

    # Build CRV se_method dict when a cluster column is configured
    cluster_col = spec_config.get('cluster1_col')
    if cluster_col and isinstance(se_method_raw, str) and se_method_raw.startswith('CRV'):
        se_method = {se_method_raw: cluster_col}
    else:
        se_method = se_method_raw

    # DuckDB resource kwargs
    threads = int(settings.get('threads', 1))
    memory_limit = settings.get('memory_limit')
    if isinstance(memory_limit, str):
        memory_limit = os.path.expandvars(memory_limit)
    max_temp_directory_size = settings.get('max_temp_directory_size')
    if isinstance(max_temp_directory_size, str):
        max_temp_directory_size = os.path.expandvars(max_temp_directory_size)
    
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
    
    # Build bootstrap config (only relevant when se_method='BS')
    _n_bootstraps = settings.get('n_bootstraps', 0)
    bootstrap_config = {'n': _n_bootstraps} if _n_bootstraps and _n_bootstraps > 0 else None

    # Collect optional resource kwargs to avoid passing None values
    resource_kwargs = {'threads': threads}
    if memory_limit is not None:
        resource_kwargs['memory_limit'] = memory_limit
    if max_temp_directory_size is not None:
        resource_kwargs['max_temp_directory_size'] = max_temp_directory_size

    model = duckreg(
        formula=formula,
        data=data_source,
        subset=sql_where,
        bootstrap=bootstrap_config,
        round_strata=settings.get('round_strata'),
        seed=settings.get('seed', 42),
        fe_method=settings.get('fe_method', 'demean'),
        db_name=db_name,
        se_method=se_method,
        fitter=fitter,
        **resource_kwargs,
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

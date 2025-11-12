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
    country_col = spec_config.get('country_col', 'countries') 
    
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
    
    # Save results using object-oriented interface
    if output_dir:
        model.results_.save(
            output_dir=output_dir,
            spec_name=spec_name,
            spec_config=spec_config,
            full_config=config
        )
    
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
    print("\nSECOND STAGE RESULTS:")
    print(model.summary(stage='second').to_string())
    print(f"\nR²: {model.results_.r_squared:.4f} | N: {model.results_.n_obs:,}")
    print("="*80)
    
    # Save results using object-oriented interface
    if output_dir:
        model.results_.save(
            output_dir=output_dir,
            spec_name=spec_name,
            spec_config=spec_config,
            full_config=config
        )
    
    return model


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
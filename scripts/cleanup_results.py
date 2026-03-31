import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_logging(verbose: bool) -> logging.Logger:
    """Setup logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def cleanup_results(output_dir: Path, logger: logging.Logger, dry_run: bool = False) -> None:
    """Delete all but the latest regression result for each model/spec."""
    model_types = ['duckreg', 'online_rls', 'online_2sls']
    
    for model_type in model_types:
        model_path = output_dir / model_type
        if not model_path.exists():
            logger.warning(f"Model directory not found: {model_path}")
            continue
        
        for spec_dir in model_path.iterdir():
            if not spec_dir.is_dir():
                continue
            
            # Collect result files (txt and json)
            result_files = list(spec_dir.glob("results_*.txt")) + list(spec_dir.glob("results_*.json"))
            if not result_files:
                logger.info(f"No result files in {spec_dir}")
                continue
            
            # Group by timestamp (extract from filename)
            timestamped_files = {}
            for file_path in result_files:
                # Filename format: results_YYYYMMDD_HHMMSS.txt or .json
                parts = file_path.stem.split('_', 1)
                if len(parts) == 2 and len(parts[1]) == 15:  # Assume 15 chars for timestamp
                    try:
                        timestamp = datetime.strptime(parts[1], '%Y%m%d_%H%M%S')
                        if timestamp not in timestamped_files:
                            timestamped_files[timestamp] = []
                        timestamped_files[timestamp].append(file_path)
                    except ValueError:
                        logger.warning(f"Invalid timestamp in filename: {file_path}")
            
            if not timestamped_files:
                logger.warning(f"No valid timestamped files in {spec_dir}")
                continue
            
            # Sort timestamps descending (latest first)
            sorted_timestamps = sorted(timestamped_files.keys(), reverse=True)
            latest_timestamp = sorted_timestamps[0]
            files_to_keep = timestamped_files[latest_timestamp]
            
            # Delete older files
            for ts, files in timestamped_files.items():
                if ts == latest_timestamp:
                    logger.info(f"Keeping latest files for {spec_dir.name}: {[f.name for f in files]}")
                    continue
                for file_path in files:
                    if dry_run:
                        logger.info(f"Would delete: {file_path}")
                    else:
                        file_path.unlink()
                        logger.info(f"Deleted: {file_path}")

def main():
    """Main entrypoint."""
    parser = argparse.ArgumentParser(description="Cleanup regression results: keep only the latest per model/spec.")
    parser.add_argument("--output_dir", type=str, help="Base output directory (default: project output/analysis)")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be deleted without actually deleting")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "output" / "analysis"
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        sys.exit(1)
    
    logger = setup_logging(args.verbose)
    logger.info(f"Starting cleanup in {output_dir} (dry_run={args.dry_run})")
    
    cleanup_results(output_dir, logger, args.dry_run)
    
    logger.info("Cleanup complete.")

if __name__ == "__main__":
    main()

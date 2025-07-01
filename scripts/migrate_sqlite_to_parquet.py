"""
Script to migrate SQLite index databases to Parquet format for better performance.

This script:
1. Finds all SQLite index files in a directory
2. Converts them to Parquet format with enhanced schema
3. Creates backups of original SQLite files
4. Validates the migration was successful
5. Provides detailed reports on the conversion

Usage:
    python scripts/migrate_sqlite_to_parquet.py [--index-dir PATH] [--dry-run] [--backup-dir PATH]
"""

import os
import sys
import argparse
import logging
import sqlite3
import shutil
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import platform
import re

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd
    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False
    print("PyArrow not available. Install with: pip install pyarrow pandas")
    sys.exit(1)

# Configure logging with UTF-8 encoding for Windows
if platform.system() == 'Windows':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('sqlite_to_parquet_migration.log', encoding='utf-8')
        ]
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('sqlite_to_parquet_migration.log')
        ]
    )

logger = logging.getLogger(__name__)


class SQLiteToParquetMigrator:
    """Migrates SQLite index databases to Parquet format."""
    
    def __init__(self, index_dir: str, backup_dir: str = None, dry_run: bool = False):
        """
        Initialize the migrator.
        
        Args:
            index_dir: Directory containing SQLite index files
            backup_dir: Directory for backups (default: index_dir/backups)
            dry_run: If True, only analyze without making changes
        """
        self.index_dir = Path(index_dir)
        self.backup_dir = Path(backup_dir) if backup_dir else self.index_dir / "backups"
        self.dry_run = dry_run
        
        # Create backup directory
        if not dry_run:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'files_found': 0,
            'files_migrated': 0,
            'files_skipped': 0,
            'files_failed': 0,
            'records_migrated': 0,
            'size_reduction_bytes': 0
        }
    
    def find_sqlite_files(self) -> List[Path]:
        """Find all SQLite index files in the directory."""
        logger.info(f"Scanning for SQLite index files in: {self.index_dir}")
        
        sqlite_files = []
        
        # Look for files matching index patterns
        patterns = [
            "index_*.sqlite",
            "download_*.sqlite",
            "*_index.sqlite"
        ]
        
        for pattern in patterns:
            found_files = list(self.index_dir.glob(pattern))
            sqlite_files.extend(found_files)
            logger.info(f"Found {len(found_files)} files matching pattern '{pattern}'")
        
        # Remove duplicates
        sqlite_files = list(set(sqlite_files))
        
        self.stats['files_found'] = len(sqlite_files)
        logger.info(f"Total SQLite index files found: {len(sqlite_files)}")
        
        return sqlite_files
    
    def analyze_sqlite_file(self, db_path: Path) -> Dict[str, Any]:
        """Analyze a SQLite file to understand its structure."""
        logger.info(f"Analyzing SQLite file: {db_path.name}")
        
        analysis = {
            'file_path': str(db_path),
            'file_size': db_path.stat().st_size,
            'has_files_table': False,
            'total_records': 0,
            'schema_info': {},
            'sample_records': [],
            'can_migrate': False,
            'migration_type': 'none'
        }
        
        try:
            # Connect to database
            conn = sqlite3.connect(str(db_path), timeout=30)
            cursor = conn.cursor()
            
            # Check if files table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='files'")
            if cursor.fetchone():
                analysis['has_files_table'] = True
                
                # Get table schema
                cursor.execute("PRAGMA table_info(files)")
                schema = cursor.fetchall()
                analysis['schema_info'] = {row[1]: row[2] for row in schema}
                
                logger.info(f"Schema columns: {list(analysis['schema_info'].keys())}")
                
                # Check if schema is compatible for migration
                required_columns = ['file_hash', 'relative_path', 'source_url']
                has_required = all(col in analysis['schema_info'] for col in required_columns)
                
                if has_required:
                    analysis['can_migrate'] = True
                    analysis['migration_type'] = 'full_migration'
                    
                    # Get total record count
                    cursor.execute("SELECT COUNT(*) FROM files")
                    analysis['total_records'] = cursor.fetchone()[0]
                    
                    # Get sample records
                    cursor.execute("SELECT * FROM files LIMIT 5")
                    analysis['sample_records'] = cursor.fetchall()
                else:
                    missing_cols = [col for col in required_columns if col not in analysis['schema_info']]
                    logger.warning(f"Database missing required columns: {missing_cols}")
                    analysis['migration_type'] = 'incompatible_schema'
            else:
                logger.warning("No 'files' table found in database")
                analysis['migration_type'] = 'no_files_table'
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error analyzing {db_path.name}: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def migrate_sqlite_to_parquet(self, db_path: Path, analysis: Dict[str, Any]) -> bool:
        """Migrate a SQLite file to Parquet format."""
        logger.info(f"Migrating SQLite file: {db_path.name}")
        
        if not analysis['can_migrate']:
            reason = analysis['migration_type']
            logger.warning(f"Cannot migrate {db_path.name}: {reason}")
            self.stats['files_failed'] += 1
            return False
        
        if self.dry_run:
            logger.info("DRY RUN: Would migrate this file")
            return True
        
        try:
            # Create backup
            backup_path = self.backup_dir / f"{db_path.name}.backup.{int(time.time())}"
            shutil.copy2(db_path, backup_path)
            logger.info(f"Created backup: {backup_path.name}")
            
            # Generate Parquet file path
            parquet_dir = db_path.parent / f"parquet_{db_path.stem}"
            parquet_dir.mkdir(exist_ok=True)
            parquet_file = parquet_dir / "files.parquet"
            
            # Read data from SQLite
            logger.info("Reading data from SQLite...")
            conn = sqlite3.connect(str(db_path), timeout=60)
            
            # Read all data into pandas DataFrame
            df = pd.read_sql_query("SELECT * FROM files", conn)
            conn.close()
            
            if len(df) == 0:
                logger.warning("No data to migrate")
                return True
            
            logger.info(f"Read {len(df)} records from SQLite")
            
            # Enhance the DataFrame with computed columns
            df = self._enhance_dataframe(df)
            
            # Convert to PyArrow Table
            table = pa.Table.from_pandas(df, preserve_index=False)
            
            # Write to Parquet
            logger.info("Writing to Parquet format...")
            pq.write_table(
                table,
                parquet_file,
                compression='snappy',
                row_group_size=100000,
                write_statistics=True,
                use_dictionary=True
            )
            
            # Save metadata
            metadata = {
                'migrated_from': str(db_path),
                'migration_timestamp': datetime.now().isoformat(),
                'original_size': analysis['file_size'],
                'parquet_size': parquet_file.stat().st_size,
                'record_count': len(df),
                'compression_ratio': parquet_file.stat().st_size / analysis['file_size']
            }
            
            metadata_file = parquet_dir / "migration_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update statistics
            self.stats['files_migrated'] += 1
            self.stats['records_migrated'] += len(df)
            self.stats['size_reduction_bytes'] += analysis['file_size'] - parquet_file.stat().st_size
            
            # Verify migration
            if self._verify_migration(parquet_file, analysis['total_records']):
                logger.info(f"Successfully migrated {db_path.name}")
                
                # Create a .migrated marker file next to the original SQLite file
                marker_file = db_path.with_suffix('.migrated')
                with open(marker_file, 'w') as f:
                    f.write(f"Migrated to Parquet on {datetime.now().isoformat()}\n")
                    f.write(f"Parquet location: {parquet_file}\n")
                
                return True
            else:
                logger.error("Migration verification failed")
                return False
                
        except Exception as e:
            logger.error(f"Error migrating {db_path.name}: {e}")
            self.stats['files_failed'] += 1
            return False
    
    def _enhance_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhance DataFrame with computed columns for better Parquet performance."""
        logger.info("Enhancing DataFrame with computed columns...")
        
        # Add computed columns
        df['year'] = None
        df['day_of_year'] = None
        df['status_category'] = 'pending'
        df['date_added'] = pd.Timestamp.now()
        
        # Extract year and day from relative_path or filename
        for idx, row in df.iterrows():
            try:
                # Try to extract year and day from filename
                filename = Path(row['relative_path']).name if pd.notna(row['relative_path']) else ''
                # Pattern for GLASS files: A2000055 format
                match = re.search(r'A(\d{4})(\d{3})', filename)
                if match:
                    df.at[idx, 'year'] = int(match.group(1))
                    df.at[idx, 'day_of_year'] = int(match.group(2))
            except Exception:
                pass
        
        # Compute status category from timestamp
        def get_status_category(timestamp):
            if pd.isna(timestamp) or timestamp == '' or timestamp is None:
                return 'pending'
            elif timestamp == 'DOWNLOADING':
                return 'downloading'
            elif isinstance(timestamp, str) and timestamp.startswith('FAILED:'):
                return 'failed'
            elif isinstance(timestamp, str) and timestamp.startswith('20'):
                return 'completed'
            else:
                return 'other'
        
        df['status_category'] = df['timestamp'].apply(get_status_category)
        
        # Convert data types for better Parquet storage
        df['file_size'] = pd.to_numeric(df['file_size'], errors='coerce').fillna(0).astype('int64')
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int32')
        df['day_of_year'] = pd.to_numeric(df['day_of_year'], errors='coerce').astype('Int32')
        
        return df
    
    def _verify_migration(self, parquet_file: Path, expected_records: int) -> bool:
        """Verify that the Parquet migration was successful."""
        try:
            # Read the Parquet file and check record count
            table = pq.read_table(parquet_file)
            actual_records = len(table)
            
            if actual_records == expected_records:
                logger.info(f"Migration verified: {actual_records} records match expected count")
                return True
            else:
                logger.error(f"Migration verification failed: {actual_records} != {expected_records}")
                return False
                
        except Exception as e:
            logger.error(f"Error verifying migration: {e}")
            return False
    
    def generate_report(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive migration report."""
        logger.info("Generating migration report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.stats.copy(),
            'files': []
        }
        
        for analysis in analyses:
            file_report = {
                'file': analysis['file_path'],
                'size_mb': round(analysis['file_size'] / 1024 / 1024, 2),
                'total_records': analysis['total_records'],
                'can_migrate': analysis['can_migrate'],
                'migration_type': analysis['migration_type'],
                'schema_columns': list(analysis['schema_info'].keys()) if analysis.get('schema_info') else []
            }
            
            if 'error' in analysis:
                file_report['error'] = analysis['error']
            
            report['files'].append(file_report)
        
        # Add summary statistics
        report['summary']['total_size_mb'] = sum(f['size_mb'] for f in report['files'])
        report['summary']['migratable_files'] = sum(1 for f in report['files'] if f['can_migrate'])
        report['summary']['size_reduction_mb'] = round(self.stats['size_reduction_bytes'] / 1024 / 1024, 2)
        
        if self.stats['files_found'] > 0:
            report['summary']['migration_rate'] = (
                self.stats['files_migrated'] / self.stats['files_found'] * 100
            )
        else:
            report['summary']['migration_rate'] = 0
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_file: str = None):
        """Save the migration report to a file."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"sqlite_to_parquet_migration_report_{timestamp}.json"
        
        output_path = Path(output_file)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Migration report saved to: {output_path}")
    
    def run_migration(self) -> Dict[str, Any]:
        """Run the complete migration process."""
        logger.info("Starting SQLite to Parquet migration process")
        logger.info(f"Index directory: {self.index_dir}")
        logger.info(f"Backup directory: {self.backup_dir}")
        logger.info(f"Dry run mode: {self.dry_run}")
        
        # Find all SQLite files
        sqlite_files = self.find_sqlite_files()
        
        if not sqlite_files:
            logger.warning("No SQLite files found to migrate")
            return self.generate_report([])
        
        # Analyze each file
        analyses = []
        for db_path in sqlite_files:
            logger.info(f"\n{'='*60}")
            analysis = self.analyze_sqlite_file(db_path)
            analyses.append(analysis)
            
            # Migrate if possible
            if analysis['can_migrate']:
                success = self.migrate_sqlite_to_parquet(db_path, analysis)
                if success:
                    logger.info(f"✓ Successfully migrated {db_path.name}")
                else:
                    logger.error(f"✗ Failed to migrate {db_path.name}")
            else:
                logger.info(f"→ Skipped {db_path.name} (cannot migrate)")
                self.stats['files_skipped'] += 1
        
        # Generate and save report
        logger.info(f"\n{'='*60}")
        report = self.generate_report(analyses)
        self.save_report(report)
        
        # Print summary
        logger.info("MIGRATION SUMMARY:")
        logger.info(f"Files found: {self.stats['files_found']}")
        logger.info(f"Files migrated: {self.stats['files_migrated']}")
        logger.info(f"Files skipped: {self.stats['files_skipped']}")
        logger.info(f"Files failed: {self.stats['files_failed']}")
        logger.info(f"Records migrated: {self.stats['records_migrated']}")
        logger.info(f"Size reduction: {self.stats['size_reduction_bytes'] / 1024 / 1024:.1f} MB")
        
        if self.stats['size_reduction_bytes'] > 0:
            reduction_percent = (self.stats['size_reduction_bytes'] / 
                               sum(a['file_size'] for a in analyses if a['can_migrate'])) * 100
            logger.info(f"Storage savings: {reduction_percent:.1f}%")
        
        return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate SQLite index databases to Parquet format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--index-dir',
        default='C:\\Users\\schulz0022\\hpc_data_index',
        help='Directory containing SQLite index files'
    )
    
    parser.add_argument(
        '--backup-dir',
        help='Directory for backups (default: index-dir/backups)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Analyze files without making changes'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate index directory
    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        logger.error(f"Index directory does not exist: {index_dir}")
        return 1
    
    # Run migration
    try:
        migrator = SQLiteToParquetMigrator(
            index_dir=str(index_dir),
            backup_dir=args.backup_dir,
            dry_run=args.dry_run
        )
        
        report = migrator.run_migration()
        
        # Exit with error code if any migrations failed
        if migrator.stats['files_failed'] > 0:
            logger.error("Some files failed to migrate")
            return 1
        
        logger.info("SQLite to Parquet migration completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Migration interrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error during migration: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

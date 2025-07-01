"""
Script to convert legacy index databases to the new enhanced timestamp format.

This script:
1. Finds all legacy SQLite index files
2. Analyzes their current structure and data
3. Converts timestamp formats to the new enhanced system
4. Creates backups of original files
5. Validates the conversion was successful

Usage:
    python scripts/convert_legacy_indices.py [--index-dir PATH] [--dry-run] [--backup-dir PATH]
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

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Configure logging with UTF-8 encoding for Windows
if platform.system() == 'Windows':
    # On Windows, ensure UTF-8 encoding for file handler
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('index_conversion.log', encoding='utf-8')
        ]
    )
else:
    # On other systems, use default encoding
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('index_conversion.log')
        ]
    )

logger = logging.getLogger(__name__)


class LegacyIndexConverter:
    """Converts legacy index databases to new enhanced timestamp format."""
    
    def __init__(self, index_dir: str, backup_dir: str = None, dry_run: bool = False):
        """
        Initialize the converter.
        
        Args:
            index_dir: Directory containing index files
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
            'files_converted': 0,
            'files_skipped': 0,
            'files_failed': 0,
            'records_analyzed': 0,
            'records_converted': 0
        }
    
    def find_index_files(self) -> List[Path]:
        """Find all SQLite index files in the directory."""
        logger.info(f"Scanning for index files in: {self.index_dir}")
        
        index_files = []
        
        # Look for files matching index patterns
        patterns = [
            "index_*.sqlite",
            "download_*.sqlite",
            "*_index.sqlite"
        ]
        
        for pattern in patterns:
            found_files = list(self.index_dir.glob(pattern))
            index_files.extend(found_files)
            logger.info(f"Found {len(found_files)} files matching pattern '{pattern}'")
        
        # Remove duplicates
        index_files = list(set(index_files))
        
        self.stats['files_found'] = len(index_files)
        logger.info(f"Total index files found: {len(index_files)}")
        
        return index_files
    
    def analyze_index_file(self, db_path: Path) -> Dict[str, Any]:
        """Analyze an index file to understand its current structure."""
        logger.info(f"Analyzing index file: {db_path.name}")
        
        analysis = {
            'file_path': str(db_path),
            'file_size': db_path.stat().st_size,
            'has_files_table': False,
            'total_records': 0,
            'timestamp_analysis': {
                'null_timestamps': 0,
                'empty_timestamps': 0,
                'iso_timestamps': 0,
                'downloading_status': 0,
                'failed_status': 0,
                'other_formats': 0
            },
            'schema_info': {},
            'sample_records': [],
            'needs_conversion': False,
            'conversion_type': 'none',
            'schema_compatible': False
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
                
                # Check if schema is compatible with new format
                required_columns = ['relative_path', 'source_url', 'timestamp']
                has_required = all(col in analysis['schema_info'] for col in required_columns)
                analysis['schema_compatible'] = has_required
                
                if not has_required:
                    missing_cols = [col for col in required_columns if col not in analysis['schema_info']]
                    logger.warning(f"Database missing required columns: {missing_cols}")
                    analysis['conversion_type'] = 'incompatible_schema'
                    analysis['needs_conversion'] = False  # Cannot convert incompatible schema
                    conn.close()
                    return analysis
                
                # Get total record count
                cursor.execute("SELECT COUNT(*) FROM files")
                analysis['total_records'] = cursor.fetchone()[0]
                self.stats['records_analyzed'] += analysis['total_records']
                
                logger.info(f"Total records: {analysis['total_records']}")
                
                # Analyze timestamps if table has timestamp column
                if 'timestamp' in analysis['schema_info']:
                    timestamp_analysis = self._analyze_timestamps(cursor)
                    analysis['timestamp_analysis'] = timestamp_analysis
                    
                    # Get sample records - use available columns
                    available_columns = ['relative_path', 'timestamp']
                    if 'file_hash' in analysis['schema_info']:
                        available_columns.insert(0, 'file_hash')
                    
                    column_list = ', '.join(available_columns)
                    cursor.execute(f"SELECT {column_list} FROM files LIMIT 10")
                    analysis['sample_records'] = cursor.fetchall()
                    
                    # Determine if conversion is needed
                    analysis['needs_conversion'], analysis['conversion_type'] = self._determine_conversion_need(timestamp_analysis)
                else:
                    logger.warning("No timestamp column found in files table")
                    analysis['conversion_type'] = 'no_timestamp_column'
            else:
                logger.warning("No 'files' table found in database")
                analysis['conversion_type'] = 'no_files_table'
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error analyzing {db_path.name}: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _analyze_timestamps(self, cursor: sqlite3.Cursor) -> Dict[str, int]:
        """Analyze the timestamp patterns in the database."""
        logger.info("Analyzing timestamp patterns...")
        
        analysis = {
            'null_timestamps': 0,
            'empty_timestamps': 0,
            'iso_timestamps': 0,
            'downloading_status': 0,
            'failed_status': 0,
            'other_formats': 0
        }
        
        # Count different timestamp patterns
        cursor.execute("SELECT COUNT(*) FROM files WHERE timestamp IS NULL")
        analysis['null_timestamps'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM files WHERE timestamp = ''")
        analysis['empty_timestamps'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM files WHERE timestamp LIKE '20%' AND LENGTH(timestamp) > 10")
        analysis['iso_timestamps'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM files WHERE timestamp = 'DOWNLOADING'")
        analysis['downloading_status'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM files WHERE timestamp LIKE 'FAILED:%'")
        analysis['failed_status'] = cursor.fetchone()[0]
        
        # Calculate other formats
        total_counted = sum(analysis.values())
        cursor.execute("SELECT COUNT(*) FROM files")
        total_records = cursor.fetchone()[0]
        analysis['other_formats'] = total_records - total_counted
        
        logger.info(f"Timestamp analysis: {analysis}")
        return analysis
    
    def _determine_conversion_need(self, timestamp_analysis: Dict[str, int]) -> Tuple[bool, str]:
        """Determine if conversion is needed and what type."""
        
        # If we already have the new format markers, no conversion needed
        if timestamp_analysis['downloading_status'] > 0 or timestamp_analysis['failed_status'] > 0:
            logger.info("Database already uses enhanced timestamp format")
            return False, 'already_enhanced'
        
        # If we have old-style status timestamps, convert them
        if timestamp_analysis['other_formats'] > 0:
            logger.info("Database has legacy status values that need conversion")
            return True, 'legacy_status'
        
        # If we only have NULL/empty and ISO timestamps, it's the simple format
        if (timestamp_analysis['null_timestamps'] > 0 or 
            timestamp_analysis['empty_timestamps'] > 0 or 
            timestamp_analysis['iso_timestamps'] > 0):
            logger.info("Database uses simple timestamp format - compatible, no conversion needed")
            return False, 'simple_format_compatible'
        
        return False, 'unknown'
    
    def convert_index_file(self, db_path: Path, analysis: Dict[str, Any]) -> bool:
        """Convert an index file to the new format."""
        logger.info(f"Converting index file: {db_path.name}")
        
        if not analysis['needs_conversion']:
            reason = analysis['conversion_type']
            if reason == 'incompatible_schema':
                logger.warning("File has incompatible schema and cannot be converted")
                self.stats['files_failed'] += 1
                return False
            else:
                logger.info("File does not need conversion, skipping")
                self.stats['files_skipped'] += 1
                return True
        
        if self.dry_run:
            logger.info("DRY RUN: Would convert this file")
            return True
        
        try:
            # Create backup
            backup_path = self.backup_dir / f"{db_path.name}.backup.{int(time.time())}"
            shutil.copy2(db_path, backup_path)
            logger.info(f"Created backup: {backup_path.name}")
            
            # Connect to database
            conn = sqlite3.connect(str(db_path), timeout=30)
            cursor = conn.cursor()
            
            # Start transaction
            cursor.execute("BEGIN TRANSACTION")
            
            records_converted = 0
            
            if analysis['conversion_type'] == 'legacy_status':
                # Convert legacy status values
                records_converted = self._convert_legacy_status_values(cursor)
            
            # Commit changes
            cursor.execute("COMMIT")
            conn.close()
            
            self.stats['files_converted'] += 1
            self.stats['records_converted'] += records_converted
            
            logger.info(f"Successfully converted {records_converted} records in {db_path.name}")
            
            # Verify conversion
            if self._verify_conversion(db_path):
                logger.info("Conversion verified successfully")
                return True
            else:
                logger.error("Conversion verification failed")
                return False
                
        except Exception as e:
            logger.error(f"Error converting {db_path.name}: {e}")
            self.stats['files_failed'] += 1
            return False
    
    def _convert_legacy_status_values(self, cursor: sqlite3.Cursor) -> int:
        """Convert legacy status values to new enhanced timestamp format."""
        logger.info("Converting legacy status values...")
        
        records_converted = 0
        
        # Get all records with non-standard timestamp values
        cursor.execute("""
            SELECT file_hash, timestamp 
            FROM files 
            WHERE timestamp IS NOT NULL 
            AND timestamp != '' 
            AND timestamp NOT LIKE '20%'
            AND timestamp != 'DOWNLOADING'
            AND timestamp NOT LIKE 'FAILED:%'
        """)
        
        legacy_records = cursor.fetchall()
        logger.info(f"Found {len(legacy_records)} legacy status records to convert")
        
        for file_hash, old_timestamp in legacy_records:
            new_timestamp = self._convert_timestamp_value(old_timestamp)
            
            if new_timestamp != old_timestamp:
                cursor.execute("""
                    UPDATE files 
                    SET timestamp = ? 
                    WHERE file_hash = ?
                """, (new_timestamp, file_hash))
                records_converted += 1
                
                if records_converted % 1000 == 0:
                    logger.info(f"Converted {records_converted} records so far...")
        
        return records_converted
    
    def _convert_timestamp_value(self, old_value: str) -> str:
        """Convert a single timestamp value to new format."""
        if not old_value or old_value in ['', 'NULL']:
            return None
        
        # Already in new format
        if old_value == 'DOWNLOADING' or old_value.startswith('FAILED:'):
            return old_value
        
        # Already ISO timestamp
        if old_value.startswith('20') and len(old_value) > 10:
            return old_value
        
        # Convert common legacy status values
        status_map = {
            'pending': None,
            'queued': None,
            'downloading': 'DOWNLOADING',
            'downloaded': datetime.now().isoformat(),
            'transferred': datetime.now().isoformat(),
            'completed': datetime.now().isoformat(),
            'success': datetime.now().isoformat(),
            'failed': 'FAILED:Legacy conversion',
            'error': 'FAILED:Legacy conversion',
            'cancelled': 'FAILED:Cancelled',
            'timeout': 'FAILED:Timeout'
        }
        
        old_lower = old_value.lower()
        if old_lower in status_map:
            return status_map[old_lower]
        
        # Check if it looks like an error message
        if any(word in old_lower for word in ['error', 'fail', 'timeout', 'cancel']):
            return f'FAILED:{old_value}'
        
        # Default: treat as completed if it's not obviously an error
        logger.warning(f"Unknown timestamp value '{old_value}', treating as completed")
        return datetime.now().isoformat()
    
    def _verify_conversion(self, db_path: Path) -> bool:
        """Verify that the conversion was successful."""
        logger.info(f"Verifying conversion of {db_path.name}")
        
        try:
            conn = sqlite3.connect(str(db_path), timeout=30)
            cursor = conn.cursor()
            
            # Check that we don't have any unexpected timestamp formats
            cursor.execute("""
                SELECT COUNT(*) FROM files 
                WHERE timestamp IS NOT NULL 
                AND timestamp != '' 
                AND timestamp NOT LIKE '20%'
                AND timestamp != 'DOWNLOADING'
                AND timestamp NOT LIKE 'FAILED:%'
            """)
            
            unexpected_count = cursor.fetchone()[0]
            conn.close()
            
            if unexpected_count > 0:
                logger.warning(f"Found {unexpected_count} records with unexpected timestamp formats")
                return False
            
            logger.info("Conversion verification passed")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying conversion: {e}")
            return False
    
    def generate_report(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive conversion report."""
        logger.info("Generating conversion report...")
        
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
                'needs_conversion': analysis['needs_conversion'],
                'conversion_type': analysis['conversion_type'],
                'schema_compatible': analysis.get('schema_compatible', False),
                'timestamp_breakdown': analysis['timestamp_analysis']
            }
            
            if 'error' in analysis:
                file_report['error'] = analysis['error']
            
            # Add schema information
            if analysis.get('schema_info'):
                file_report['schema_columns'] = list(analysis['schema_info'].keys())
            
            report['files'].append(file_report)
        
        # Add summary statistics
        report['summary']['total_size_mb'] = sum(f['size_mb'] for f in report['files'])
        report['summary']['compatible_files'] = sum(1 for f in report['files'] if f.get('schema_compatible', False))
        report['summary']['incompatible_files'] = sum(1 for f in report['files'] if not f.get('schema_compatible', False))
        
        if self.stats['files_found'] > 0:
            report['summary']['conversion_rate'] = (
                self.stats['files_converted'] / self.stats['files_found'] * 100
            )
        else:
            report['summary']['conversion_rate'] = 0
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_file: str = None):
        """Save the conversion report to a file."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"index_conversion_report_{timestamp}.json"
        
        output_path = Path(output_file)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Conversion report saved to: {output_path}")
    
    def run_conversion(self) -> Dict[str, Any]:
        """Run the complete conversion process."""
        logger.info("Starting legacy index conversion process")
        logger.info(f"Index directory: {self.index_dir}")
        logger.info(f"Backup directory: {self.backup_dir}")
        logger.info(f"Dry run mode: {self.dry_run}")
        
        # Find all index files
        index_files = self.find_index_files()
        
        if not index_files:
            logger.warning("No index files found to convert")
            return self.generate_report([])
        
        # Analyze each file
        analyses = []
        for db_path in index_files:
            logger.info(f"\n{'='*60}")
            analysis = self.analyze_index_file(db_path)
            analyses.append(analysis)
            
            # Convert if needed
            if analysis['needs_conversion']:
                success = self.convert_index_file(db_path, analysis)
                if success:
                    logger.info(f"✓ Successfully converted {db_path.name}")
                else:
                    logger.error(f"✗ Failed to convert {db_path.name}")
            else:
                logger.info(f"-> Skipped {db_path.name} (no conversion needed)")
        
        # Generate and save report
        logger.info(f"\n{'='*60}")
        report = self.generate_report(analyses)
        self.save_report(report)
        
        # Print summary
        logger.info("CONVERSION SUMMARY:")
        logger.info(f"Files found: {self.stats['files_found']}")
        logger.info(f"Files converted: {self.stats['files_converted']}")
        logger.info(f"Files skipped: {self.stats['files_skipped']}")
        logger.info(f"Files failed: {self.stats['files_failed']}")
        logger.info(f"Records analyzed: {self.stats['records_analyzed']}")
        logger.info(f"Records converted: {self.stats['records_converted']}")
        
        # Add schema compatibility summary
        compatible_count = sum(1 for a in analyses if a.get('schema_compatible', False))
        logger.info(f"Schema compatible files: {compatible_count}/{len(analyses)}")
        
        return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert legacy index databases to enhanced timestamp format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--index-dir',
        default='C:\\Users\\schulz0022\\hpc_data_index',
        help='Directory containing index files'
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
    
    # Run conversion
    try:
        converter = LegacyIndexConverter(
            index_dir=str(index_dir),
            backup_dir=args.backup_dir,
            dry_run=args.dry_run
        )
        
        report = converter.run_conversion()
        
        # Exit with error code if any conversions failed
        if converter.stats['files_failed'] > 0:
            logger.error("Some files failed to convert")
            return 1
        
        logger.info("Index conversion completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Conversion interrupted by user")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error during conversion: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
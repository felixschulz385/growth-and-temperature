# GNT Data System Scripts

This directory contains utility scripts for the GNT data system.

## Index Conversion Scripts

### convert_legacy_indices.py

Converts existing legacy index databases to the new enhanced timestamp format.

#### Features:
- **Automatic Discovery**: Finds all SQLite index files in a directory
- **Schema Compatibility Check**: Verifies database schema before conversion
- **Comprehensive Analysis**: Analyzes current timestamp patterns and database structure
- **Safe Conversion**: Creates backups before making any changes
- **Verification**: Validates that conversion was successful
- **Detailed Reporting**: Generates JSON reports with conversion statistics
- **Dry Run Mode**: Test conversions without making changes
- **Unicode Support**: Handles Unicode characters in logs on Windows

#### Usage:

```bash
# Analyze indices without making changes
python scripts/convert_legacy_indices.py --index-dir "C:\Users\username\hpc_data_index" --dry-run

# Convert indices with backup
python scripts/convert_legacy_indices.py --index-dir "C:\Users\username\hpc_data_index"

# Convert with custom backup location
python scripts/convert_legacy_indices.py \
    --index-dir "C:\Users\username\hpc_data_index" \
    --backup-dir "C:\Backups\index_backups"

# Verbose output
python scripts/convert_legacy_indices.py --index-dir "..." --verbose
```

#### Schema Compatibility:

The script checks for required columns before attempting conversion:

**Required columns:**
- `relative_path` - Path to the file relative to source
- `source_url` - URL where file can be downloaded
- `timestamp` - Download/processing status timestamp

**Optional columns:**
- `file_hash` - Unique hash for the file
- `destination_blob` - Target path for the file
- `file_size` - Size of the file in bytes
- `metadata` - Additional file metadata (JSON)

**Incompatible databases:**
- Missing required columns cannot be converted
- Will be reported but skipped during conversion
- May require manual schema migration

#### What it converts:

1. **Legacy Status Values**: Converts old string status values to new format:
   - `pending`, `queued` → `NULL` (pending)
   - `downloading` → `DOWNLOADING`
   - `completed`, `success`, `downloaded` → ISO timestamp
   - `failed`, `error` → `FAILED:Legacy conversion`

2. **Preserves Existing Data**: 
   - NULL/empty timestamps (pending files)
   - ISO timestamps (completed files)
   - Already converted enhanced format

3. **Schema Validation**:
   - Checks for required columns
   - Reports incompatible schemas
   - Provides detailed schema information in reports

#### Output:

- **Backup Files**: `*.backup.{timestamp}` in backup directory
- **Conversion Log**: `index_conversion.log` (UTF-8 encoded)
- **JSON Report**: `index_conversion_report_{timestamp}.json`

#### Safety Features:

- Creates backups before any modifications
- Uses database transactions for atomic updates
- Verifies conversion success
- Detailed logging of all operations
- Dry-run mode for testing
- Schema compatibility validation

#### Enhanced Timestamp Format:

After conversion, timestamps use this format:
- `NULL` or `''` = Pending download
- `'DOWNLOADING'` = Currently downloading
- `'FAILED:{error_message}'` = Download/transfer failed
- ISO timestamp = Successfully completed

This allows the async downloader to:
- Query pending files: `WHERE timestamp IS NULL OR timestamp = ''`
- Retry failed files: `WHERE timestamp LIKE 'FAILED:%'`
- Track download progress with clear status indicators

#### Common Issues:

1. **Schema Incompatibility**: Old databases missing required columns
   - **Solution**: Check report for required columns, may need manual migration

2. **Unicode Logging Errors**: Windows encoding issues
   - **Solution**: Script now uses UTF-8 encoding for log files

3. **Database Corruption**: Corrupted SQLite files
   - **Solution**: Script creates backups and attempts recovery

4. **Permission Issues**: Cannot write to backup directory
   - **Solution**: Ensure write permissions to backup location

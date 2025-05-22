#!/usr/bin/env python3
"""
Helper script for transferring data between Google Cloud and university cluster.

This script provides utilities for:
1. Exporting preprocessing index from GCS 
2. Creating transfer manifests for needed files
3. Downloading needed files to a transfer directory
4. Importing the index on the university cluster
"""
from pathlib import Path
import os
import sys
import logging
import argparse
from datetime import datetime
import json
import tempfile
from typing import List, Dict, Any

from gnt.data.common.index.preprocessing_index import PreprocessingIndex
from gnt.data.common.gcs.client import GCSClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def export_index(args):
    """Export the preprocessing index from GCS."""
    # Create index in cloud mode
    index = PreprocessingIndex(
        bucket_name=args.bucket,
        data_path=args.data_path,
        version=args.version,
        temp_dir=args.temp_dir
    )
    
    # Export the index
    export_path = index.export_index(
        output_path=args.output if args.output else None,
        include_completed=not args.skip_completed
    )
    
    # Also export as CSV for inspection
    csv_files = index.export_to_csv(
        output_dir=os.path.dirname(export_path),
        separate_stages=False
    )
    
    print(f"Index exported to: {export_path}")
    for stage, csv_path in csv_files.items():
        print(f"CSV for {stage}: {csv_path}")


def create_transfer_manifest(args):
    """Create a manifest of files that need to be transferred to the cluster."""
    # Load index from exported file or create new one
    if args.index:
        index = PreprocessingIndex(
            bucket_name=args.bucket,
            data_path=args.data_path,
            version=args.version,
            temp_dir=args.temp_dir
        )
        # Import the provided index
        index.import_index(args.index, merge_mode="update")
    else:
        # Connect directly to cloud index
        index = PreprocessingIndex(
            bucket_name=args.bucket,
            data_path=args.data_path,
            version=args.version,
            temp_dir=args.temp_dir
        )
    
    # Mark annual files for transfer if needed
    if args.mark_annual:
        # Find annual files that are completed but not marked for transfer
        annual_files = index.get_files(
            stage=PreprocessingIndex.STAGE_ANNUAL,
            status=PreprocessingIndex.STATUS_COMPLETED
        )
        
        count = 0
        for file in annual_files:
            # Mark for transfer to cluster
            if index.mark_for_transfer(file['file_hash'], destination="cluster"):
                count += 1
        
        print(f"Marked {count} annual files for transfer")
    
    # Create the transfer manifest
    manifest_path = index.create_transfer_manifest(args.output)
    print(f"Transfer manifest created at: {manifest_path}")


def download_files(args):
    """Download files listed in the transfer manifest."""
    # Load the manifest file
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)
    
    # Create output directory
    output_dir = args.output_dir or os.path.join(os.getcwd(), "downloaded_files")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create GCS client
    gcs_client = GCSClient(args.bucket)
    
    # Create index to update transfer status
    index = PreprocessingIndex(
        bucket_name=args.bucket,
        data_path=args.data_path,
        version=args.version,
        temp_dir=args.temp_dir
    )
    
    # Download files
    successful = 0
    failed = 0
    
    for file_entry in manifest["files"]:
        file_hash = file_entry["file_hash"]
        blob_path = file_entry["blob_path"]
        year = file_entry["year"]
        grid_cell = file_entry.get("grid_cell")
        
        # Create target directory
        target_dir = output_dir
        if year:
            target_dir = os.path.join(target_dir, str(year))
            if grid_cell:
                target_dir = os.path.join(target_dir, f"gridcell{grid_cell}")
                
        os.makedirs(target_dir, exist_ok=True)
        
        # Get filename from blob path
        filename = os.path.basename(blob_path.rstrip('/'))
        local_path = os.path.join(target_dir, filename)
        
        try:
            # Download file
            print(f"Downloading {blob_path} to {local_path}")
            if blob_path.endswith('.zarr/'):
                # Handle zarr directories
                gcs_client.download_directory(blob_path, local_path)
            else:
                # Handle regular files
                gcs_client.download_file(blob_path, local_path)
                
            # Update index
            index.update_transfer_status(file_hash, "downloaded")
            
            # Update local path in index
            file_info = index.get_file_by_hash(file_hash)
            if file_info:
                index.update_file_status(
                    file_hash=file_hash,
                    local_path=local_path
                )
            
            successful += 1
            print(f"✓ Downloaded {blob_path}")
            
        except Exception as e:
            print(f"✗ Failed to download {blob_path}: {str(e)}")
            failed += 1
    
    # Save the updated index
    index.save()
    
    # Export the updated index for transfer to the cluster
    export_path = index.export_index(
        output_path=os.path.join(output_dir, f"preprocess_index_{datetime.now().strftime('%Y%m%d')}.sqlite")
    )
    
    print(f"\nDownload complete: {successful} succeeded, {failed} failed")
    print(f"Updated index exported to: {export_path}")


def import_cluster(args):
    """Import the index on the university cluster and prepare for processing."""
    # Create index in local mode
    index = PreprocessingIndex(
        bucket_name=args.bucket,
        data_path=args.data_path,
        version=args.version,
        temp_dir=args.temp_dir,
        local_mode=True,
        local_data_dir=args.data_dir
    )
    
    # Import the index from the downloaded file
    success = index.import_index(args.index_file, merge_mode="update")
    if not success:
        print(f"Failed to import index from {args.index_file}")
        return 1
    
    # Plan stage 2 processing
    if args.years:
        # Convert years string to list of integers
        years = [int(y.strip()) for y in args.years.split(',')]
        print(f"Planning Stage 2 processing for specific years: {years}")
        plan = index.plan_stage2_processing(years=years, limit=args.limit)
    else:
        # Process any available annual files
        print(f"Planning Stage 2 processing for any available annual files (limit: {args.limit})")
        plan = index.plan_stage2_processing(limit=args.limit)
    
    # Save the processing plan
    plan_path = os.path.join(args.temp_dir or os.getcwd(), f"stage2_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(plan_path, 'w') as f:
        json.dump(
            {
                "version": index.version,
                "created": datetime.now().isoformat(),
                "file_count": len(plan),
                "files": plan
            },
            f,
            indent=2
        )
    
    print(f"Stage 2 processing plan created at {plan_path}")
    print(f"Found {len(plan)} files ready for spatial transformation")


def main():
    """Parse arguments and execute the appropriate command."""
    parser = argparse.ArgumentParser(description="Helper for transferring data between GCS and university cluster")
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument("--data-path", required=True, help="Data path in the bucket")
    parser.add_argument("--version", default="v1", help="Processing version")
    parser.add_argument("--temp-dir", help="Directory for temporary files")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Export index command
    export_parser = subparsers.add_parser("export-index", help="Export the preprocessing index from GCS")
    export_parser.add_argument("--output", help="Output path for the exported index")
    export_parser.add_argument("--skip-completed", action="store_true", help="Skip completed files in export")
    
    # Create transfer manifest command
    manifest_parser = subparsers.add_parser("create-manifest", help="Create a transfer manifest")
    manifest_parser.add_argument("--index", help="Path to previously exported index file")
    manifest_parser.add_argument("--output", help="Output path for the manifest file")
    manifest_parser.add_argument("--mark-annual", action="store_true", help="Mark all completed annual files for transfer")
    
    # Download files command
    download_parser = subparsers.add_parser("download", help="Download files from the transfer manifest")
    download_parser.add_argument("--manifest", required=True, help="Path to the transfer manifest file")
    download_parser.add_argument("--output-dir", help="Directory to download files to")
    
    # Import on cluster command
    cluster_parser = subparsers.add_parser("import-cluster", help="Import the index on the university cluster")
    cluster_parser.add_argument("--index-file", required=True, help="Path to the exported index file")
    cluster_parser.add_argument("--data-dir", required=True, help="Local directory with downloaded data files")
    cluster_parser.add_argument("--years", help="Comma-separated list of years to process (e.g. '2000,2001,2002')")
    cluster_parser.add_argument("--limit", type=int, default=100, help="Maximum number of files to include in plan")
    
    args = parser.parse_args()
    
    if args.command == "export-index":
        export_index(args)
    elif args.command == "create-manifest":
        create_transfer_manifest(args)
    elif args.command == "download":
        download_files(args)
    elif args.command == "import-cluster":
        import_cluster(args)
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
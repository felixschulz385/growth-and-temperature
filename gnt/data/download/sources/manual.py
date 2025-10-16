"""
Manual download data sources for datasets requiring manual authentication.
Prompts users to provide paths to manually downloaded files.
"""

import os
import logging
import hashlib
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

from gnt.data.download.sources.base import BaseDataSource

logger = logging.getLogger(__name__)


class ManualDataSource(BaseDataSource):
    """
    Base class for data sources that require manual download.
    Prompts user for file paths and handles file transfers.
    """
    
    def __init__(self, files: List[Dict[str, str]], output_path: str, 
                 download_instructions: str = None):
        """
        Initialize manual data source.
        
        Args:
            files: List of file configurations with 'name', 'description', 'url' (for reference)
            output_path: Relative path for data storage
            download_instructions: Instructions for manual download
        """
        self.DATA_SOURCE_NAME = "manual"
        self.files = files
        self.data_path = output_path
        self.has_entrypoints = False
        self.download_instructions = download_instructions
        
        # Schema for consistency
        self.schema_dtypes = {
            'file_size': 'int64',
            'download_status': 'string',
            'status_category': 'string'
        }
    
    def get_file_hash(self, file_url: str) -> str:
        """Generate unique hash for file based on expected filename."""
        return hashlib.md5(file_url.encode('utf-8')).hexdigest()
    
    def list_remote_files(self, entrypoint: dict = None) -> List[tuple]:
        """
        List files that need to be manually provided.
        Returns list of (relative_path, reference_url) tuples.
        """
        result = []
        for file_config in self.files:
            name = file_config.get('name')
            url = file_config.get('url', '')
            subfolder = file_config.get('subfolder', '')
            
            if subfolder:
                relative_path = f"{subfolder}/{name}"
            else:
                relative_path = name
            
            result.append((relative_path, url))
        
        return result
    
    def local_path(self, relative_path: str) -> str:
        """
        Generate local path for a file based on its relative path.
        
        Args:
            relative_path: Relative path of the file
            
        Returns:
            Full local path where the file should be stored
        """
        return os.path.join("data", self.data_path, relative_path)
    
    def gcs_upload_path(self, base_url: str, relative_path: str) -> str:
        """
        Generate destination path for the file (legacy GCS method).
        
        Args:
            base_url: Base URL (not used for manual sources)
            relative_path: Relative path of the file
            
        Returns:
            Destination path in the format expected by the system
        """
        return f"{self.data_path}/{relative_path}"
    
    def prompt_for_file_path(self, file_config: Dict[str, str]) -> Optional[str]:
        """
        Prompt user to provide path to manually downloaded file.
        
        Args:
            file_config: File configuration dict
            
        Returns:
            Path to local file or None if skipped
        """
        name = file_config.get('name')
        description = file_config.get('description', '')
        url = file_config.get('url', '')
        
        print("\n" + "="*70)
        print(f"Manual Download Required: {name}")
        print("="*70)
        
        if description:
            print(f"\nDescription: {description}")
        
        if url:
            print(f"\nReference URL: {url}")
        
        if self.download_instructions:
            print(f"\nInstructions:\n{self.download_instructions}")
        
        print("\nPlease download this file manually and provide the path below.")
        print("Or press Enter to skip this file.")
        
        while True:
            file_path = input("\nFile path: ").strip()
            
            if not file_path:
                logger.info(f"Skipping {name}")
                return None
            
            # Expand user directory and resolve path
            file_path = Path(file_path).expanduser().resolve()
            
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                continue
            
            if not file_path.is_file():
                print(f"Error: Path is not a file: {file_path}")
                continue
            
            logger.info(f"Found file: {file_path} ({file_path.stat().st_size / 1024 / 1024:.1f} MB)")
            return str(file_path)
    
    def download(self, file_url: str, output_path: str, session=None) -> None:
        """
        'Download' by prompting user for file path and copying.
        
        Args:
            file_url: Reference URL (not actually used for download)
            output_path: Destination path
            session: Not used for manual downloads
        """
        # Find the file configuration
        file_config = None
        for fc in self.files:
            if fc.get('url') == file_url or fc.get('name') in file_url:
                file_config = fc
                break
        
        if not file_config:
            logger.error(f"Could not find file configuration for {file_url}")
            raise ValueError(f"Unknown file: {file_url}")
        
        # Prompt for file path
        source_path = self.prompt_for_file_path(file_config)
        
        if not source_path:
            raise FileNotFoundError(f"User skipped manual download for {file_config.get('name')}")
        
        # Copy file to output path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        logger.info(f"Copying {source_path} to {output_path}")
        shutil.copy2(source_path, output_path)
        logger.info(f"Successfully copied file ({Path(output_path).stat().st_size / 1024 / 1024:.1f} MB)")
    
    async def download_async(self, file_url: str, output_path: str, session=None) -> None:
        """Async version delegates to synchronous download."""
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.download, file_url, output_path, session)
    
    def get_all_entrypoints(self) -> List[Dict[str, Any]]:
        """Manual sources don't use entrypoints."""
        return []
    
    def filename_to_entrypoint(self, relative_path: str) -> Optional[Dict[str, Any]]:
        """Manual sources don't use entrypoints."""
        return None


class BermanMiningDataSource(ManualDataSource):
    """
    Data source for Berman et al. Mining data from ICPSR.
    Requires manual download due to authentication requirements.
    """
    
    def __init__(self, output_path: str = "berman_mining"):
        """Initialize Berman Mining data source."""
        
        files = [
            {
                'name': 'BCRT_baseline.dta',
                'description': 'Berman et al. Mining Conflict Resource Traps - Baseline Data',
                'url': 'https://www.openicpsr.org/openicpsr/project/113068/version/V1/view',
                'subfolder': 'baseline'
            }
        ]
        
        instructions = """
1. Visit: https://www.openicpsr.org/openicpsr/project/113068/version/V1/view
2. Log in or create an ICPSR account
3. Navigate to Data/BCRT_baseline.dta
4. Download the file to your local machine
5. Provide the path to the downloaded file below
"""
        
        super().__init__(
            files=files,
            output_path=output_path,
            download_instructions=instructions
        )
        
        self.DATA_SOURCE_NAME = "berman_mining"

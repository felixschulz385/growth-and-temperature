"""
Ultra-lightweight async download mechanism for HPC workflows.
Polls files from index, downloads, tars, transfers to HPC, and updates status.
"""

import os
import time
import asyncio
import aiohttp
import aiofiles
import logging
import tempfile
import tarfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class AsyncHPCDownloader:
    """Ultra-lightweight async downloader for HPC workflows."""
    
    def __init__(self, data_source, index, hpc_client, staging_dir: str, 
                 batch_size: int = 5, max_concurrent: int = 5, 
                 tar_max_files: int = 100, tar_max_size_mb: int = 500):
        """
        Initialize the async downloader.
        
        Args:
            data_source: Data source for downloads
            index: Unified data index
            hpc_client: HPC client for transfers
            staging_dir: Local staging directory
            batch_size: Number of files to poll per batch
            max_concurrent: Maximum concurrent downloads
            tar_max_files: Maximum files per tar archive
            tar_max_size_mb: Maximum tar archive size in MB
        """
        self.data_source = data_source
        self.index = index
        self.hpc_client = hpc_client
        self.staging_dir = Path(staging_dir)
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.tar_max_files = tar_max_files
        self.tar_max_size_bytes = tar_max_size_mb * 1024 * 1024
        
        # Create directories
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir = self.staging_dir / "downloads"
        self.tar_dir = self.staging_dir / "tars"
        self.download_dir.mkdir(exist_ok=True)
        self.tar_dir.mkdir(exist_ok=True)
        
        # Async state
        self.session = None
        self.download_semaphore = asyncio.Semaphore(max_concurrent)
        self.running = False
        
        # Stats
        self.stats = {
            'files_downloaded': 0,
            'files_failed': 0,
            'tars_created': 0,
            'tars_transferred': 0,
            'bytes_downloaded': 0,
            'start_time': None
        }
    
    async def start(self):
        """Start the async downloader."""
        logger.info("Starting async HPC downloader")
        self.running = True
        self.stats['start_time'] = time.time()
        
        # Create aiohttp session
        connector = aiohttp.TCPConnector(limit=20, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=300, connect=30)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        
        try:
            await self._run_download_loop()
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the async downloader."""
        logger.info("Stopping async HPC downloader")
        self.running = False
        
        if self.session:
            await self.session.close()
            self.session = None
        
        # Log final stats
        elapsed = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        logger.info(f"Download session complete: "
                   f"{self.stats['files_downloaded']} files downloaded, "
                   f"{self.stats['tars_transferred']} tars transferred in {elapsed:.1f}s")
    
    async def _run_download_loop(self):
        """Main download loop with improved efficiency and index synchronization."""
        logger.info("Starting download loop")
        
        # Pull index from HPC before starting downloads
        await self._sync_index_before_downloads()
        
        # Check initial pending count
        loop = asyncio.get_event_loop()
        pending_count = await loop.run_in_executor(None, self.index.count_pending_files)
        logger.info(f"Starting with {pending_count} files pending download")
        
        consecutive_empty_polls = 0
        max_empty_polls = 3  # Stop after 3 consecutive empty polls
        
        while self.running and consecutive_empty_polls < max_empty_polls:
            try:
                # Poll for files to download
                files_to_download = await self._poll_files()
                
                if not files_to_download:
                    consecutive_empty_polls += 1
                    logger.info(f"No files to download (attempt {consecutive_empty_polls}/{max_empty_polls}), waiting...")
                    await asyncio.sleep(30)  # Wait 30 seconds before polling again
                    continue
                
                # Reset empty poll counter
                consecutive_empty_polls = 0
                
                logger.info(f"Processing batch of {len(files_to_download)} files")
                
                # Download files concurrently
                downloaded_files = await self._download_files_batch(files_to_download)
                
                if downloaded_files:
                    # Create tar archive
                    tar_path = await self._create_tar_archive(downloaded_files)
                    
                    if tar_path:
                        # Transfer to HPC
                        success = await self._transfer_and_extract(tar_path, downloaded_files)
                        
                        if success:
                            # Update status in database
                            await self._update_file_statuses(downloaded_files, 'success')
                            self.stats['tars_transferred'] += 1
                            
                            # Sync index to HPC after successful batch
                            await self._sync_index_after_batch()
                        else:
                            await self._update_file_statuses(downloaded_files, 'failed')
                        
                        # Cleanup
                        await self._cleanup_files(downloaded_files, tar_path)
                
                # Log progress
                self._log_progress()
                
            except Exception as e:
                logger.error(f"Error in download loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying
        
        if consecutive_empty_polls >= max_empty_polls:
            logger.info("Download loop completed - no more files to download")
        else:
            logger.info("Download loop stopped")
    
    async def _poll_files(self) -> List[Dict[str, Any]]:
        """Poll the index for files to download efficiently."""
        loop = asyncio.get_event_loop()
        
        def get_pending_files():
            try:
                # Use optimized Parquet query
                files = self.index.query_pending_files(self.batch_size)
                
                # Log some basic stats
                if files:
                    total_size = sum(f.get('file_size', 0) for f in files if f.get('file_size'))
                    logger.debug(f"Queried {len(files)} files, total size: {total_size / 1024 / 1024:.1f} MB")
                
                return files
                
            except Exception as e:
                logger.error(f"Error polling files: {e}")
                return []
        
        return await loop.run_in_executor(None, get_pending_files)
    
    async def _download_single_file(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Download a single file."""
        async with self.download_semaphore:
            file_hash = file_info['file_hash']
            source_url = file_info['source_url']
            relative_path = file_info['relative_path']
            
            # Create local file path
            filename = Path(relative_path).name
            local_path = self.download_dir / f"{file_hash}_{filename}"
            
            try:
                logger.debug(f"Downloading {relative_path}")
                
                # Download file without updating index status
                if hasattr(self.data_source, 'download_async'):
                    await self.data_source.download_async(source_url, str(local_path), self.session)
                else:
                    # Fallback to generic aiohttp download
                    await self._generic_download(source_url, local_path)
                
                # Get file size
                file_size = local_path.stat().st_size
                self.stats['bytes_downloaded'] += file_size
                self.stats['files_downloaded'] += 1
                
                return {
                    'success': True,
                    'file_info': file_info,
                    'local_path': str(local_path),
                    'file_size': file_size
                }
                
            except Exception as e:
                logger.error(f"Failed to download {relative_path}: {e}")
                # Clean up partial file
                if local_path.exists():
                    local_path.unlink()
                
                self.stats['files_failed'] += 1
                return {'success': False, 'error': str(e)}
    
    async def _generic_download(self, url: str, local_path: Path):
        """Generic aiohttp download."""
        async with self.session.get(url) as response:
            response.raise_for_status()
            
            async with aiofiles.open(local_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    await f.write(chunk)
    
    async def _create_tar_archive(self, downloaded_files: List[Dict[str, Any]]) -> Optional[str]:
        """Create a tar archive from downloaded files preserving directory structure."""
        if not downloaded_files:
            return None
        
        # Generate tar filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tar_filename = f"batch_{timestamp}_{len(downloaded_files)}.tar.gz"
        tar_path = self.tar_dir / tar_filename
        
        logger.info(f"Creating tar archive {tar_filename} with {len(downloaded_files)} files")
        
        loop = asyncio.get_event_loop()
        
        def create_tar():
            try:
                with tarfile.open(tar_path, "w:gz") as tar:
                    for file_data in downloaded_files:
                        local_path = Path(file_data['local_path'])
                        relative_path = file_data['file_info']['relative_path']
                        
                        # Preserve the full directory structure in the tar archive
                        # relative_path is like '2000/055/GLASS06A01.V01.A2000055.h01v10.2022021.hdf'
                        # This will maintain the 2000/055/ directory structure when extracted
                        tar.add(local_path, arcname=relative_path)
                        logger.debug(f"Added to tar: {local_path} -> {relative_path}")
                
                return str(tar_path)
            except Exception as e:
                logger.error(f"Error creating tar archive: {e}")
                return None
        
        result = await loop.run_in_executor(None, create_tar)
        
        if result:
            self.stats['tars_created'] += 1
            logger.info(f"Created tar archive: {tar_path} ({tar_path.stat().st_size / 1024 / 1024:.1f} MB)")
        
        return result
    
    async def _transfer_and_extract(self, tar_path: str, downloaded_files: List[Dict[str, Any]]) -> bool:
        """Transfer tar to HPC and extract with proper path handling."""
        try:
            tar_filename = Path(tar_path).name
            
            # Use relative paths - the HPCClient will combine these with base_path
            # All paths are relative to the HPC target base directory
            remote_tar_path = f"{self.index.data_path}/tar/{tar_filename}"
            remote_extract_dir = f"{self.index.data_path}/raw"
            
            logger.info(f"Transferring {tar_filename} to HPC")
            logger.debug(f"Remote tar path (relative): {remote_tar_path}")
            logger.debug(f"Remote extract dir (relative): {remote_extract_dir}")
            
            # Ensure remote directories exist
            loop = asyncio.get_event_loop()
            tar_dir_path = f"{self.index.data_path}/tar"
            
            await loop.run_in_executor(None, self.hpc_client.ensure_directory, tar_dir_path)
            await loop.run_in_executor(None, self.hpc_client.ensure_directory, remote_extract_dir)
            
            # Transfer tar file using rsync
            rsync_options = {
                "compress": True,
                "archive": True,
                "partial": True,
                "verbose": False
            }
            
            success, summary = await loop.run_in_executor(
                None, 
                self.hpc_client.rsync_transfer,
                tar_path, 
                remote_tar_path, 
                True,  # source_is_local
                rsync_options,
                False  # show_progress
            )
            
            if not success:
                logger.error(f"Failed to transfer {tar_filename}: {summary}")
                return False
            
            logger.info(f"Successfully transferred {tar_filename}, extracting...")
            
            # Extract on HPC - paths will be resolved by HPCClient
            extract_success = await loop.run_in_executor(
                None,
                self.hpc_client.extract_tar,
                remote_tar_path,
                remote_extract_dir
            )
            
            if extract_success:
                logger.info(f"Successfully extracted {tar_filename}")
                
                # Verify extraction
                verification_success = await self._verify_extracted_files(downloaded_files, remote_extract_dir)
                
                if verification_success:
                    logger.info(f"Verified extraction of {len(downloaded_files)} files")
                    
                    # Clean up remote tar file
                    # Build full path for removal command
                    if not remote_tar_path.startswith("/") and self.hpc_client.base_path:
                        full_tar_path = f"{self.hpc_client.base_path}/{remote_tar_path}"
                    else:
                        full_tar_path = remote_tar_path
                    
                    await loop.run_in_executor(
                        None,
                        self.hpc_client.execute_command,
                        f"rm -f '{full_tar_path}'"
                    )
                    
                    return True
                else:
                    logger.error(f"Failed to verify extracted files from {tar_filename}")
                    return False
            else:
                logger.error(f"Failed to extract {tar_filename}")
                return False
                
        except Exception as e:
            logger.error(f"Error in transfer and extract: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return False

    async def _verify_extracted_files(self, downloaded_files: List[Dict[str, Any]], remote_extract_dir: str) -> bool:
        """Verify that files were successfully extracted on HPC."""
        try:
            loop = asyncio.get_event_loop()
            
            # Sample a few files to verify extraction
            files_to_check = downloaded_files[:min(5, len(downloaded_files))]
            
            for file_data in files_to_check:
                relative_path = file_data['file_info']['relative_path']
                # Build the remote file path - HPCClient will resolve with base_path
                remote_file_path = f"{remote_extract_dir}/{relative_path}"
                
                logger.debug(f"Verifying extracted file: {remote_file_path}")
                
                # Check if file exists on HPC
                exists = await loop.run_in_executor(
                    None,
                    self.hpc_client.check_file_exists,
                    remote_file_path
                )
                
                if not exists:
                    logger.warning(f"File not found on HPC after extraction: {remote_file_path}")
                    return False
            
            logger.debug(f"Verified {len(files_to_check)} sample files on HPC")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying extracted files: {e}")
            return False

    async def _update_file_statuses(self, files: List[Dict[str, Any]], status: str):
        """Update file statuses for a batch using vectorized Parquet operations - only called after HPC verification."""
        loop = asyncio.get_event_loop()
        
        def update_statuses():
            try:
                # Prepare batch updates for vectorized operation
                updates = []
                for file_data in files:
                    if isinstance(file_data, dict) and 'file_info' in file_data:
                        file_hash = file_data['file_info']['file_hash']
                        file_size = file_data.get('file_size')
                    else:
                        file_hash = file_data['file_hash']
                        file_size = None
                    
                    update_dict = {'file_hash': file_hash}
                    
                    if status == 'success':
                        # Only update to completed after successful HPC extraction and verification
                        update_dict['status'] = 'completed'
                        logger.debug(f"Marking file {file_hash} as completed after HPC verification")
                    elif status == 'failed':
                        update_dict['status'] = 'FAILED:Transfer or extraction failed'
                        logger.debug(f"Marking file {file_hash} as failed")
                    
                    if file_size is not None:
                        update_dict['file_size'] = file_size
                    
                    updates.append(update_dict)
                
                # Use vectorized batch update for efficiency
                self.index.update_file_statuses_batch(updates)
                
            except Exception as e:
                logger.error(f"Error updating file statuses: {e}")
        
        await loop.run_in_executor(None, update_statuses)
    
    async def _download_files_batch(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Download a batch of files concurrently."""
        tasks = []
        for file_info in files:
            task = asyncio.create_task(self._download_single_file(file_info))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        downloaded_files = []
        failed_files = []
        
        for i, result in enumerate(results):
            if isinstance(result, dict) and result.get('success'):
                downloaded_files.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Download failed for {files[i]['relative_path']}: {result}")
                failed_files.append(files[i])
                self.stats['files_failed'] += 1
            else:
                # Download failed but not an exception
                failed_files.append(files[i])
        
        # Only update failed files immediately, successful files are updated after HPC verification
        if failed_files:
            await self._update_file_statuses(failed_files, 'failed')
        
        return downloaded_files

    async def _cleanup_files(self, downloaded_files: List[Dict[str, Any]], tar_path: str):
        """Clean up local files after successful transfer to HPC."""
        try:
            loop = asyncio.get_event_loop()
            
            def cleanup():
                # Remove downloaded files
                for file_data in downloaded_files:
                    try:
                        local_path = Path(file_data['local_path'])
                        if local_path.exists():
                            local_path.unlink()
                            logger.debug(f"Cleaned up local file: {local_path}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up {file_data['local_path']}: {e}")
                
                # Remove tar file
                try:
                    tar_file = Path(tar_path)
                    if tar_file.exists():
                        tar_file.unlink()
                        logger.debug(f"Cleaned up tar file: {tar_path}")
                except Exception as e:
                    logger.warning(f"Error cleaning up tar file {tar_path}: {e}")
            
            await loop.run_in_executor(None, cleanup)
            
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")

    def _log_progress(self):
        """Log download progress."""
        elapsed = time.time() - self.stats['start_time']
        logger.info(f"Progress: {self.stats['files_downloaded']} downloaded, "
                   f"{self.stats['files_failed']} failed, "
                   f"{self.stats['tars_transferred']} tars transferred, "
                   f"{self.stats['bytes_downloaded'] / 1024 / 1024:.1f} MB "
                   f"in {elapsed:.1f}s")

    async def _sync_index_before_downloads(self):
        """Pull index from HPC before starting downloads."""
        try:
            if not hasattr(self.hpc_client, 'target'):
                logger.debug("No HPC target available for index sync")
                return True
            
            logger.info("Syncing index from HPC before downloads")
            loop = asyncio.get_event_loop()
            
            success = await loop.run_in_executor(
                None,
                self.index.ensure_synced_index,
                self.hpc_client.target,
                'pull',  # Always pull before downloads
                False,   # force
                getattr(self.hpc_client, 'key_file', None)
            )
            
            if success:
                logger.info("Successfully synced index from HPC")
            else:
                logger.warning("Failed to sync index from HPC, continuing with local index")
            
            return success
            
        except Exception as e:
            logger.error(f"Error syncing index from HPC: {e}")
            return False

    async def _sync_index_after_batch(self):
        """Push index to HPC after processing a batch."""
        try:
            if not hasattr(self.hpc_client, 'target'):
                logger.debug("No HPC target available for index sync")
                return True
            
            logger.info("Syncing index to HPC after batch")
            loop = asyncio.get_event_loop()
            
            success = await loop.run_in_executor(
                None,
                self.index.sync_index_with_hpc,
                self.hpc_client.target,
                'push',  # Always push after processing
                True,    # sync_entrypoints
                False,   # force
                getattr(self.hpc_client, 'key_file', None)
            )
            
            if success:
                logger.info("Successfully synced index to HPC")
            else:
                logger.warning("Failed to sync index to HPC")
            
            return success
            
        except Exception as e:
            logger.error(f"Error syncing index to HPC: {e}")
            return False

async def run_async_download_workflow(data_source, index, hpc_client, context, config):
    """
    Run the async download workflow.
    
    Args:
        data_source: Data source for downloads
        index: Unified data index
        hpc_client: HPC client
        context: Workflow context
        config: Download configuration
    """
    logger.info("Starting async download workflow")
    
    # Extract configuration
    batch_size = config.get('batch_size', 50)
    max_concurrent = config.get('max_concurrent_downloads', 5)
    tar_max_files = config.get('tar_max_files', 100)
    tar_max_size_mb = config.get('tar_max_size_mb', 500)
    
    # Create downloader
    downloader = AsyncHPCDownloader(
        data_source=data_source,
        index=index,
        hpc_client=hpc_client,
        staging_dir=context.staging_dir,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        tar_max_files=tar_max_files,
        tar_max_size_mb=tar_max_size_mb
    )
    
    try:
        await downloader.start()
        return True
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Error in async download workflow: {e}")
        return False
    finally:
        await downloader.stop()
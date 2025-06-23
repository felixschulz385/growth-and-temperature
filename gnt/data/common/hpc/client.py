"""
Client for interacting with HPC systems via SSH and rsync.

This module provides functionality for transferring files between local workstations
and HPC clusters, executing commands remotely, and managing file synchronization.
"""
import os
import time
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Optional

logger = logging.getLogger(__name__)

class HPCClient:
    """
    Client for interacting with HPC systems using SSH and rsync.
    
    Provides functionality for:
    - File transfers with rsync
    - Remote command execution with SSH
    - Directory and file management
    - Index synchronization
    """
    
    def __init__(self, host_target: str, key_file: Optional[str] = None):
        """
        Initialize the HPC client.
        
        Args:
            host_target: SSH target in format user@host:/path or just user@host
            key_file: Path to SSH private key file (optional)
        """
        self.host_target = host_target
        self.key_file = key_file
        
        # Parse host and path components
        if ":" in host_target:
            self.host, self.base_path = host_target.split(":", 1)
            # Remove trailing slash for consistency
            self.base_path = self.base_path.rstrip('/')
        else:
            self.host = host_target
            self.base_path = ""  # Empty string if no path provided
    
        logger.debug(f"Initialized HPC client with host: {self.host}, base path: {self.base_path}")
        
    def ensure_directory(self, remote_path: str) -> bool:
        """
        Ensure a directory exists on the HPC system.
        
        Args:
            remote_path: Path on the HPC system
            
        Returns:
            bool: Whether the operation was successful
        """
        logger.debug(f"Ensuring directory exists on HPC: {remote_path}")
        
        # Build full path
        if not remote_path.startswith("/"):
            remote_path = f"{self.base_path}/{remote_path}"
        
        # Create directory via SSH
        try:
            ssh_cmd = self._build_ssh_command(f"mkdir -p {remote_path}")
            subprocess.run(ssh_cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to create directory on HPC: {e}")
            return False

    def ensure_directory_path(self, remote_path: str) -> bool:
        """
        Ensure the full remote path and all parents exist on the HPC system.
        
        Args:
            remote_path: Path on the HPC system (can be a file path; will create parent dirs)
            
        Returns:
            bool: Whether the operation was successful
        """
        # If path is a file path, get the directory portion
        remote_dir = os.path.dirname(remote_path)
        
        return self.ensure_directory(remote_dir)

    def check_file_exists(self, remote_path: str) -> bool:
        """
        Check if a file exists on the remote system.
        
        Args:
            remote_path: Full path to the remote file
            
        Returns:
            bool: True if the file exists, False otherwise
        """
        try:
            # Build the command to check if file exists
            cmd = ["ssh"]
            if self.key_file:
                cmd.extend(["-i", self.key_file])
            cmd.append(self.host)
            cmd.append(f"test -f {remote_path} && echo 'exists' || echo 'not found'")
            
            # Run the command
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True
            )
            
            # Check the output
            return result.stdout.strip() == "exists"
            
        except Exception as e:
            logger.error(f"Error checking if file exists: {e}")
            return False
    
    def get_file_info(self, remote_path: str) -> Dict[str, Any]:
        """
        Get information about a file on the HPC system.
        
        Args:
            remote_path: Path to the file on HPC
            
        Returns:
            Dict with file information:
            - exists: Whether the file exists
            - size: File size in bytes (or None if file doesn't exist)
            - modified: Modification timestamp (or None if file doesn't exist)
        """
        logger.debug(f"Getting file info on HPC: {remote_path}")
        
        # Build full path
        if not remote_path.startswith("/"):
            remote_path = f"{self.base_path}/{remote_path}"
        
        result = {
            'exists': False,
            'size': None,
            'modified': None
        }
        
        try:
            # Check if file exists
            ssh_cmd = self._build_ssh_command(f"test -f {remote_path} && echo exists || echo missing")
            check_result = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
            
            if "exists" in check_result.stdout:
                result['exists'] = True
                
                # Get file size
                ssh_cmd = self._build_ssh_command(f"stat -c %s {remote_path}")
                size_result = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
                result['size'] = int(size_result.stdout.strip())
                
                # Get modification time
                ssh_cmd = self._build_ssh_command(f"stat -c %Y {remote_path}")
                mod_result = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
                result['modified'] = int(mod_result.stdout.strip())
            
            return result
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to get file info on HPC: {e}")
            return result
    
    def execute_command(self, command: str) -> Tuple[bool, str, str]:
        """
        Execute a command on the HPC system.
        
        Args:
            command: Command to execute
            
        Returns:
            Tuple containing (success, stdout, stderr)
        """
        logger.debug(f"Executing command on HPC: {command}")
        
        try:
            ssh_cmd = self._build_ssh_command(command)
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
            return True, result.stdout, result.stderr
        except subprocess.SubprocessError as e:
            logger.error(f"Command execution failed on HPC: {e}")
            if isinstance(e, subprocess.CalledProcessError):
                return False, e.stdout, e.stderr
            return False, "", str(e)
    
    def execute_sqlite_query(self, db_path: str, query: str) -> Tuple[bool, List[Any]]:
        """
        Execute an SQLite query on the remote system.
        
        Args:
            db_path: Path to the SQLite database on HPC
            query: SQL query to execute
            
        Returns:
            Tuple containing (success, results)
        """
        logger.debug(f"Executing SQLite query on HPC: {query}")
        
        # Build full path
        if not db_path.startswith("/"):
            db_path = f"{self.base_path}/{db_path}"
        
        try:
            # Escape single quotes in the query for shell compatibility
            escaped_query = query.replace("'", "'\\''")
            
            ssh_cmd = self._build_ssh_command(f"sqlite3 -csv '{db_path}' '{escaped_query}'")
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
            
            # Parse CSV output
            rows = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    rows.append(line.split(','))
            
            return True, rows
        except subprocess.SubprocessError as e:
            logger.error(f"SQLite query failed on HPC: {e}")
            return False, []
    
    def rsync_transfer(
        self, 
        source_path: str, 
        target_path: str, 
        source_is_local: bool = True,
        options: Dict[str, Any] = None,
        show_progress: bool = True
    ) -> Tuple[bool, str]:
        """
        Transfer files using rsync.
        
        Args:
            source_path: Source path (local or remote)
            target_path: Target path (remote or local)
            source_is_local: Whether the source is local (True) or remote (False)
            options: Dictionary of rsync options
            show_progress: Whether to show progress information
            
        Returns:
            Tuple containing (success, output)
        """
        options = options or {
            "compress": True,
            "archive": True,
            "partial": True,
            "checksum": True,
            "verbose": True,
            "bwlimit": 0  # 0 means no limit
        }
        
        # Build rsync command
        rsync_cmd = ["rsync"]
        
        # Add options
        if options.get("compress"):
            rsync_cmd.append("-z")
        if options.get("archive"):
            rsync_cmd.append("-a")
        if options.get("partial"):
            rsync_cmd.append("--partial")
        if options.get("checksum"):
            rsync_cmd.append("--checksum")
        if options.get("ignore_times", False):
            rsync_cmd.append("--ignore-times")
        if options.get("delete", False):
            rsync_cmd.append("--delete")
        if options.get("verbose"):
            rsync_cmd.append("-v")
        if options.get("bwlimit") and options.get("bwlimit") > 0:
            rsync_cmd.extend(["--bwlimit", str(options["bwlimit"])])
        
        # Add progress flag if requested
        if show_progress:
            rsync_cmd.append("--progress")
        
        # Build SSH command with options INSIDE the ssh command string
        ssh_opts = []
        if self.key_file:
            # Expand user directory if path contains tilde
            expanded_key_file = os.path.expanduser(self.key_file)
            
            # Check if key file exists
            if os.path.isfile(expanded_key_file):
                ssh_opts.append(f"-i {expanded_key_file}")
            else:
                logger.warning(f"SSH key file not found: {self.key_file} (expanded to {expanded_key_file})")

        # Build the ssh command string with all options
        ssh_cmd = f"ssh {' '.join(ssh_opts)}" if ssh_opts else "ssh"
        
        # Add the SSH command to rsync
        rsync_cmd.extend(["-e", ssh_cmd])
    
        # Format paths for rsync
        if source_is_local:
            # Local to remote transfer
            formatted_source = source_path
            formatted_target = f"{self.host}:{target_path}"
            logger.info(f"Transferring from local {source_path} to HPC {target_path}")
        else:
            # Remote to local transfer
            formatted_source = f"{self.host}:{source_path}"
            formatted_target = target_path
            logger.info(f"Transferring from HPC {source_path} to local {target_path}")
    
        # Add source and destination to command
        rsync_cmd.append(formatted_source)
        rsync_cmd.append(formatted_target)
    
        start_time = time.time()
    
        try:
            # Execute the transfer
            logger.debug(f"Executing rsync command: {' '.join(rsync_cmd)}")
            result = subprocess.run(rsync_cmd, check=True, capture_output=True, text=True)
    
            elapsed = time.time() - start_time
            logger.info(f"Transfer completed in {elapsed:.1f} seconds")
    
            # Extract progress information
            output_lines = result.stdout.splitlines()
            summary = ""
            if len(output_lines) > 2:
                summary = output_lines[-2]
    
            return True, summary
    
        except subprocess.SubprocessError as e:
            logger.error(f"Rsync transfer failed: {e}")
            if isinstance(e, subprocess.CalledProcessError):
                return False, f"STDOUT: {e.stdout}\nSTDERR: {e.stderr}"
            return False, str(e)
    
    def extract_tar(self, tar_path: str, extract_dir: str) -> bool:
        """
        Extract a tar file on the HPC system.
        
        Args:
            tar_path: Path to the tar file on the HPC system
            extract_dir: Directory to extract to on the HPC system
            
        Returns:
            bool: Whether the extraction was successful
        """
        logger.debug(f"Extracting tar on HPC: {tar_path} to {extract_dir}")
        
        # Build full paths
        if not tar_path.startswith("/"):
            tar_path = f"{self.base_path}/{tar_path}"
        if not extract_dir.startswith("/"):
            extract_dir = f"{self.base_path}/{extract_dir}"
        
        # Build SSH command
        extract_command = f"mkdir -p {extract_dir} && tar -xzf {tar_path} -C {extract_dir} && echo 'Extraction complete'"
        ssh_cmd = self._build_ssh_command(extract_command)
        
        try:
            # Execute the SSH command
            result = subprocess.run(ssh_cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully extracted tar on HPC")
            return True
        except subprocess.SubprocessError as e:
            logger.error(f"Extraction failed on HPC: {e}")
            return False

    def _build_ssh_command(self, remote_command: str) -> List[str]:
        """
        Build an SSH command with the appropriate options.
        
        Args:
            remote_command: Command to execute on the remote system
            
        Returns:
            List containing the SSH command with arguments
        """
        ssh_cmd = ["ssh"]
        
        # Add key file if specified
        if self.key_file:
            # Expand user directory if path contains tilde
            expanded_key_file = os.path.expanduser(self.key_file)
            
            # Check if key file exists
            if os.path.isfile(expanded_key_file):
                # Add the key file to the command
                ssh_cmd.extend(["-i", expanded_key_file])
                
                # Add options to prevent password prompting
                ssh_cmd.extend(["-o", "PasswordAuthentication=no"])
                ssh_cmd.extend(["-o", "BatchMode=yes"])
            else:
                logger.warning(f"SSH key file not found: {self.key_file} (expanded to {expanded_key_file})")
    
        # Add host and command
        ssh_cmd.append(self.host)
        ssh_cmd.append(remote_command)
    
        return ssh_cmd


class HPCIndexSynchronizer:
    """
    Handles synchronization of index files between local and HPC systems.
    """
    
    def __init__(self, client: HPCClient, local_index_dir: str, remote_index_dir: str = "hpc_data_index"):
        """
        Initialize the index synchronizer.
        
        Args:
            client: HPC client for remote operations
            local_index_dir: Local directory for storing index files
            remote_index_dir: Remote directory on the HPC system for index files
        """
        self.client = client
        self.local_index_dir = local_index_dir
        
        # Check if the client has a base path defined (from hpc_target) and use it
        if hasattr(client, 'base_path') and client.base_path and client.base_path != "/home":
            # Combine the base path with the remote index directory
            self.remote_index_dir = f"{client.base_path}/{remote_index_dir}"
        else:
            # Use just the remote_index_dir if no base path is defined
            self.remote_index_dir = remote_index_dir
            
        # Ensure local directory exists
        os.makedirs(local_index_dir, exist_ok=True)
        
        # Ensure remote directory exists
        self.client.ensure_directory(self.remote_index_dir)
        
        # Log the actual remote path for debugging
        logger.debug(f"Remote index directory set to: {self.remote_index_dir}")
    
    def sync_index(
        self, 
        data_path: str, 
        direction: str = "push", 
        sync_entrypoints: bool = True, 
        force: bool = False
    ) -> bool:
        """
        Synchronize index files between local and HPC.
        
        Args:
            data_path: Path identifier for the data source
            direction: 'push' to send local index to HPC, 'pull' to get index from HPC
            sync_entrypoints: Whether to also sync entrypoints file
            force: Whether to force sync even if timestamps indicate no changes
            
        Returns:
            bool: Whether the sync was successful
        """
        logger.info(f"{direction.capitalize()}ing index for {data_path} to/from HPC")
        
        # Generate filenames
        safe_path = data_path.replace('/', '_')
        db_filename = f"download_{safe_path}.sqlite"
        meta_filename = f"download_{safe_path}_meta.json"
        entrypoints_filename = f"entrypoints_{safe_path}.json"
        
        # Local and remote paths
        local_db_path = os.path.join(self.local_index_dir, db_filename)
        local_meta_path = os.path.join(self.local_index_dir, meta_filename)
        local_entrypoints_path = os.path.join(self.local_index_dir, entrypoints_filename)
        
        remote_db_path = f"{self.remote_index_dir}/{db_filename}"
        remote_meta_path = f"{self.remote_index_dir}/{meta_filename}"
        remote_entrypoints_path = f"{self.remote_index_dir}/{entrypoints_filename}"
        
        # Configure rsync options
        rsync_options = {
            "compress": True,
            "archive": True,
            "partial": True,
            "checksum": True,
            "ignore_times": force,
            "verbose": True
        }
        
        # Perform sync based on direction
        try:
            if direction.lower() == "push":
                # Transfer database
                if os.path.exists(local_db_path):
                    db_success, _ = self.client.rsync_transfer(
                        local_db_path, 
                        remote_db_path, 
                        source_is_local=True,
                        options=rsync_options,
                        show_progress=False
                    )
                    if not db_success:
                        logger.error("Failed to push database to HPC")
                        return False
                
                # Transfer metadata
                if os.path.exists(local_meta_path):
                    meta_success, _ = self.client.rsync_transfer(
                        local_meta_path, 
                        remote_meta_path, 
                        source_is_local=True,
                        options=rsync_options,
                        show_progress=False
                    )
                    if not meta_success:
                        logger.warning("Failed to push metadata to HPC")
                
                # Transfer entrypoints if requested
                if sync_entrypoints and os.path.exists(local_entrypoints_path):
                    entrypoints_success, _ = self.client.rsync_transfer(
                        local_entrypoints_path, 
                        remote_entrypoints_path, 
                        source_is_local=True,
                        options=rsync_options,
                        show_progress=False
                    )
                    if not entrypoints_success:
                        logger.warning("Failed to push entrypoints to HPC")
                
                logger.info(f"Successfully pushed index for {data_path} to HPC")
                return True
                
            elif direction.lower() == "pull":
                # Check if remote database exists
                if not self.client.check_file_exists(remote_db_path):
                    logger.warning(f"Remote index file not found at {remote_db_path}")
                    return False
                
                # Transfer database
                db_success, _ = self.client.rsync_transfer(
                    remote_db_path, 
                    local_db_path, 
                    source_is_local=False,
                    options=rsync_options,
                    show_progress=False
                )
                if not db_success:
                    logger.error("Failed to pull database from HPC")
                    return False
                
                # Transfer metadata
                meta_success, _ = self.client.rsync_transfer(
                    remote_meta_path, 
                    local_meta_path, 
                    source_is_local=False,
                    options=rsync_options,
                    show_progress=False
                )
                if not meta_success:
                    logger.warning("Failed to pull metadata from HPC (may not exist)")
                
                # Transfer entrypoints if requested
                if sync_entrypoints:
                    entrypoints_success, _ = self.client.rsync_transfer(
                        remote_entrypoints_path, 
                        local_entrypoints_path, 
                        source_is_local=False,
                        options=rsync_options,
                        show_progress=False
                    )
                    if not entrypoints_success:
                        logger.warning("Failed to pull entrypoints from HPC (may not exist)")
                
                logger.info(f"Successfully pulled index for {data_path} from HPC")
                return True
            
            else:
                logger.error(f"Invalid sync direction: {direction}")
                return False
            
        except Exception as e:
            logger.error(f"Error during index sync: {e}")
            import traceback
            logger.debug(f"Full error: {traceback.format_exc()}")
            return False
    
    def compare_indices(self, data_path: str) -> Dict[str, Any]:
        """
        Compare local and remote index files to determine which is newer/better.
        
        Args:
            data_path: Path identifier for the data source
            
        Returns:
            Dict with comparison results:
            - local_exists: Whether local index exists
            - remote_exists: Whether remote index exists
            - local_file_count: Number of files in local index
            - remote_file_count: Number of files in remote index
            - local_modified: When local index was last modified
            - remote_modified: When remote index was last modified
            - recommendation: 'push', 'pull', or 'up-to-date'
        """
        result = {
            'local_exists': False,
            'remote_exists': False,
            'local_file_count': 0,
            'remote_file_count': 0,
            'local_modified': None,
            'remote_modified': None,
            'recommendation': 'push'  # Default
        }
        
        try:
            # Generate filename
            safe_path = data_path.replace('/', '_')
            db_filename = f"download_{safe_path}.sqlite"
            
            # Local and remote paths
            local_db_path = os.path.join(self.local_index_dir, db_filename)
            remote_db_path = f"{self.remote_index_dir}/{db_filename}"
            
            # Check local existence and stats
            if os.path.exists(local_db_path):
                result['local_exists'] = True
                result['local_modified'] = int(os.path.getmtime(local_db_path))
                
                # Get local file count 
                try:
                    import sqlite3
                    conn = sqlite3.connect(local_db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM files")
                    result['local_file_count'] = cursor.fetchone()[0]
                    conn.close()
                except Exception as e:
                    logger.warning(f"Error getting local file count: {e}")
            
            # Check remote existence and stats
            remote_info = self.client.get_file_info(remote_db_path)
            
            if remote_info['exists']:
                result['remote_exists'] = True
                result['remote_modified'] = remote_info['modified']
                
                # Get remote file count
                success, rows = self.client.execute_sqlite_query(
                    remote_db_path, 
                    "SELECT COUNT(*) FROM files"
                )
                
                if success and rows and rows[0]:
                    try:
                        result['remote_file_count'] = int(rows[0][0])
                    except (ValueError, IndexError):
                        logger.warning(f"Could not parse remote file count: {rows}")
            
            # Determine recommendation
            if result['local_exists'] and result['remote_exists']:
                # If both exist, compare by file count and timestamp
                if result['local_file_count'] > result['remote_file_count']:
                    result['recommendation'] = 'push'
                elif result['local_file_count'] < result['remote_file_count']:
                    result['recommendation'] = 'pull'
                else:
                    # Same file count, check timestamps
                    if result['local_modified'] and result['remote_modified']:
                        if result['local_modified'] > result['remote_modified']:
                            result['recommendation'] = 'push'
                        elif result['local_modified'] < result['remote_modified']:
                            result['recommendation'] = 'pull'
                        else:
                            result['recommendation'] = 'up-to-date'
            elif result['local_exists']:
                result['recommendation'] = 'push'
            elif result['remote_exists']:
                result['recommendation'] = 'pull'
            
            return result
            
        except Exception as e:
            logger.error(f"Error comparing indices: {e}")
            import traceback
            logger.debug(f"Full error: {traceback.format_exc()}")
            return result
    
    def ensure_synced_index(self, data_path: str, sync_direction: str = 'auto', force: bool = False) -> bool:
        """
        Ensure the index is synchronized between local and the HPC system.
        
        Args:
            data_path: Path identifier for the data source
            sync_direction: 'auto', 'push', 'pull', or 'none'
            force: Whether to force sync even if timestamps indicate no changes
            
        Returns:
            bool: Whether synchronization was successful or wasn't needed
        """
        if sync_direction.lower() == 'none':
            logger.info("Index synchronization disabled")
            return True
        
        if sync_direction.lower() == 'auto':
            # Compare indices to determine best action
            comparison = self.compare_indices(data_path)
            
            logger.info(f"Index comparison: Local {comparison['local_file_count']} files, " +
                        f"Remote {comparison['remote_file_count']} files")
            
            if comparison['recommendation'] == 'up-to-date':
                logger.info("Indices are up-to-date, no synchronization needed")
                return True
                
            sync_direction = comparison['recommendation']
            logger.info(f"Auto-detected best sync direction: {sync_direction}")
        
        # Perform the sync
        return self.sync_index(
            data_path=data_path,
            direction=sync_direction,
            sync_entrypoints=True,
            force=force
        )

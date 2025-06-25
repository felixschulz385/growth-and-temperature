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
        
        # Flag to track SQLite availability on remote system
        self._sqlite_available = None
        self._sqlite_path = None

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
    
    def check_sqlite_availability(self) -> bool:
        """
        Check if SQLite is available on the remote system.
        
        Returns:
            bool: Whether SQLite is available
        """
        if self._sqlite_available is not None:
            return self._sqlite_available
            
        try:
            # Try to run a simple SQLite command
            ssh_cmd = self._build_ssh_command("command -v sqlite3")
            result = subprocess.run(ssh_cmd, capture_output=True, text=True)
            
            # If the command returns a path, SQLite is available
            if result.returncode == 0 and result.stdout.strip() != "":
                self._sqlite_available = True
                self._sqlite_path = result.stdout.strip()
                logger.debug(f"Found SQLite at: {self._sqlite_path}")
                return True
            
            # Try with module load if available (common on HPC systems)
            ssh_cmd = self._build_ssh_command("module avail sqlite 2>&1 | grep -i sqlite")
            result = subprocess.run(ssh_cmd, capture_output=True, text=True)
            
            if "sqlite" in result.stdout.lower():
                # Module exists, try loading it and checking sqlite3
                ssh_cmd = self._build_ssh_command("module load sqlite 2>/dev/null && command -v sqlite3")
                result = subprocess.run(ssh_cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout.strip() != "":
                    self._sqlite_available = True
                    self._sqlite_path = "module load sqlite && " + result.stdout.strip()
                    logger.debug(f"SQLite available via module system")
                    return True
            
            # SQLite not available
            self._sqlite_available = False
            logger.warning("SQLite (sqlite3 command) not available on the remote system. Some functionality will be limited.")
            return False
            
        except Exception as e:
            logger.error(f"Error checking for SQLite availability: {e}")
            self._sqlite_available = False
            return False
    
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
        
        # Check if SQLite is available
        if not self.check_sqlite_availability():
            logger.error("Cannot execute SQLite query: sqlite3 command not found on remote system")
            return False, []
        
        # Build full path
        if not db_path.startswith("/"):
            db_path = f"{self.base_path}/{db_path}"
        
        try:
            # Escape single quotes in the query for shell compatibility
            escaped_query = query.replace("'", "'\\''")
            
            # Build the SQLite command using the detected path/method
            if self._sqlite_path and "module load" in self._sqlite_path:
                # Use module load approach
                sqlite_command = f"{self._sqlite_path} -csv"
            else:
                # Direct command
                sqlite_command = "sqlite3 -csv"
            
            ssh_cmd = self._build_ssh_command(f"{sqlite_command} '{db_path}' '{escaped_query}'")
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
            
            # Parse CSV output
            rows = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    rows.append(line.split(','))
            
            return True, rows
        except subprocess.SubprocessError as e:
            logger.error(f"SQLite query failed on HPC: {e}")
            if isinstance(e, subprocess.CalledProcessError):
                logger.debug(f"SQLite error details - stdout: {e.stdout}, stderr: {e.stderr}")
            return False, []

    def rsync_transfer(
        self, 
        source_path: str, 
        target_path: str, 
        source_is_local: bool = True,
        options: Dict[str, Any] = None,
        show_progress: bool = True,
        progress_callback=None,
        return_process=False
    ) -> Union[Tuple[bool, str], Tuple[bool, str, subprocess.Popen]]:
        """
        Transfer files using rsync.
        
        Args:
            source_path: Source path (local or remote)
            target_path: Target path (remote or local)
            source_is_local: Whether the source is local (True) or remote (False)
            options: Dictionary of rsync options
            show_progress: Whether to show progress information
            progress_callback: Optional callback function for progress updates
            return_process: Whether to return the subprocess.Popen object
            
        Returns:
            Tuple containing (success, output) or (success, output, process)
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
        if options.get("partial-dir"):
            rsync_cmd.extend(["--partial-dir", options.get("partial-dir")])
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
        
        # Build SSH command with key file
        ssh_cmd = ["ssh"]
        
        # Add key file if specified - use expanded path to ensure ~ is resolved
        if self.key_file:
            expanded_key_file = os.path.expanduser(self.key_file)
            if os.path.isfile(expanded_key_file):
                ssh_cmd.extend(["-i", expanded_key_file])
                
                # Add options to prevent password prompting
                ssh_cmd.extend(["-o", "PasswordAuthentication=no"])
                ssh_cmd.extend(["-o", "BatchMode=yes"])
            else:
                logger.warning(f"SSH key file not found: {self.key_file} (expanded to {expanded_key_file})")
        
        # Join the SSH command with quotes to handle spaces in paths
        ssh_cmd_str = " ".join(ssh_cmd)
        rsync_cmd.extend(["-e", ssh_cmd_str])
    
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
        
        logger.debug(f"Executing rsync command: {' '.join(rsync_cmd)}")
        
        if return_process or progress_callback:
            # For real-time progress tracking or returning the process
            try:
                process = subprocess.Popen(
                    rsync_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                output_lines = []
                error_lines = []
                
                # Process output in real-time for progress tracking
                if progress_callback:
                    for line in process.stdout:
                        output_lines.append(line)
                        # Extract transfer progress information
                        progress_info = self._parse_rsync_progress(line)
                        if progress_info:
                            progress_callback(progress_info)
                    
                    # Collect any errors
                    for line in process.stderr:
                        error_lines.append(line)
                else:
                    # Just collect output without callbacks
                    stdout, stderr = process.communicate()
                    output_lines = stdout.splitlines()
                    error_lines = stderr.splitlines()
                
                # Wait for completion and get return code
                return_code = process.wait()
                
                if return_code == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"Transfer completed in {elapsed:.1f} seconds")
                    
                    # Extract summary from output
                    summary = "Transfer complete"
                    for line in reversed(output_lines):
                        if "bytes/sec" in line or "files transferred" in line:
                            summary = line.strip()
                            break
                    
                    if return_process:
                        return True, summary, process
                    return True, summary
                else:
                    logger.error(f"Rsync transfer failed with code {return_code}")
                    error_summary = "\n".join(error_lines) if error_lines else "Unknown error"
                    
                    if return_process:
                        return False, error_summary, process
                    return False, error_summary
                    
            except Exception as e:
                logger.error(f"Error during rsync transfer: {e}")
                if return_process:
                    return False, str(e), None
                return False, str(e)
        else:
            # Simple version without real-time tracking
            try:
                # Execute the transfer
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
    
    def _parse_rsync_progress(self, line):
        """Parse rsync progress output line for status information."""
        line = line.strip()
        progress_info = {"message": line}
        
        # Look for bytes transferred information
        if "%" in line and "to-check" not in line:
            try:
                # Extract bytes information
                if "bytes/sec" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.endswith("/s") or part.endswith("/sec"):
                            # Found speed indicator, the preceding part should be bytes transferred
                            if i > 0 and parts[i-1].isdigit():
                                progress_info["bytes_transferred"] = int(parts[i-1])
                                break
                
                # Extract percentage
                if "%" in line:
                    percent_part = next((p for p in line.split() if "%" in p), None)
                    if percent_part:
                        try:
                            progress_info["percent"] = float(percent_part.replace("%", ""))
                        except ValueError:
                            pass
                            
            except Exception:
                pass  # Ignore parsing errors
                
        return progress_info

    def extract_tar(self, tar_path: str, extraction_dir: str) -> bool:
        """
        Extract a tar file on the HPC system.
        
        Args:
            tar_path: Path to the tar file on HPC
            extraction_dir: Directory to extract to on HPC
            
        Returns:
            bool: Whether the extraction was successful
        """
        logger = logging.getLogger(__name__)
        
        # Make sure the extraction directory exists
        self.ensure_directory(extraction_dir)
        
        try:
            # Build the command to extract tar
            # Using tar -xzf <tar_path> -C <extraction_dir>
            # The -C option changes to the extraction directory first
            # FIX: Use --strip-components=1 to remove the top-level directory
            # This removes the batch_* directory structure and places files directly in the raw dir
            cmd = f"cd {extraction_dir} && tar -xzf {tar_path} --strip-components=1"
            
            # Execute SSH command
            result, stdout, stderr = self.execute_command(cmd)
            
            if result:
                logger.info(f"Successfully extracted {tar_path} to {extraction_dir}")
                return True
            else:
                logger.error(f"Failed to extract {tar_path}: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error extracting tar file {tar_path}: {e}")
            return False

    def _ssh_execute(self, command: str) -> tuple:
        """
        DEPRECATED: Use execute_command instead.
        Execute a command on the remote SSH host.
        
        Args:
            command: Command to execute
        
        Returns:
            Tuple: (success: bool, stdout: str, stderr: str)
        """
        logger.debug(f"DEPRECATED: _ssh_execute called. Use execute_command instead. Command: {command}")
        return self.execute_command(command)

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
                
                # Check if SQLite is available on remote system
                sqlite_available = self.client.check_sqlite_availability()
                
                # Get remote file count
                if sqlite_available:
                    success, rows = self.client.execute_sqlite_query(
                        remote_db_path, 
                        "SELECT COUNT(*) FROM files"
                    )
                    
                    if success and rows and rows[0]:
                        try:
                            result['remote_file_count'] = int(rows[0][0])
                        except (ValueError, IndexError):
                            logger.warning(f"Could not parse remote file count: {rows}")
                else:
                    # SQLite not available - use file timestamps for comparison
                    logger.info("SQLite not available on remote system, using timestamp comparison")
                    
                    # If local index exists, estimate remote count based on file size and timestamp
                    if result['local_exists']:
                        if remote_info['size'] > os.path.getsize(local_db_path) * 0.9:  
                            # If remote file is similar size or larger, assume same or more files
                            result['remote_file_count'] = result['local_file_count']
                        else:
                            # Otherwise, estimate proportionally by file size
                            size_ratio = remote_info['size'] / os.path.getsize(local_db_path)
                            result['remote_file_count'] = int(result['local_file_count'] * size_ratio)
            
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

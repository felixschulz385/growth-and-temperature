"""
Client for interacting with HPC systems via SSH and rsync.

This module provides functionality for transferring files between local workstations
and HPC clusters, executing commands remotely, and managing file synchronization.
"""
import os
import time
import logging
import subprocess
import platform
import shutil
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Optional

logger = logging.getLogger(__name__)

class HPCClient:
    """Client for interacting with HPC systems via SSH and rsync."""
    
    def __init__(self, target: str, key_file: str = None):
        """
        Initialize HPC client.
        
        Args:
            target: SSH target in format user@host:/path or user@host
            key_file: Path to SSH private key file (optional)
        """
        self.target = target
        self.key_file = key_file
        
        # Parse target to extract host and path
        if ":" in target:
            self.ssh_target, self.base_path = target.split(":", 1)
        else:
            self.ssh_target = target
            self.base_path = ""
        
        # Also set host attribute for backward compatibility
        self.host = self.ssh_target
        
        # Initialize cached attributes
        self._sqlite_available = None
        self._sqlite_path = None
        self._rsync_available = shutil.which("rsync") is not None
        
        # Normalize key file path for Windows compatibility
        if self.key_file:
            self.key_file = self._normalize_key_path(self.key_file)
            logger.debug(f"Using SSH key: {self.key_file}")
            
            # Verify key file exists
            if not os.path.exists(self.key_file):
                logger.warning(f"SSH key file not found: {self.key_file}")
            else:
                logger.debug(f"SSH key file verified: {self.key_file}")

    def _normalize_key_path(self, key_file: str) -> str:
        """Normalize SSH key file path for cross-platform compatibility."""
        from pathlib import Path
        
        # Expand user directory and resolve path
        key_path = Path(key_file).expanduser().resolve()
        
        # On Windows, convert to string with forward slashes for SSH
        if platform.system() == 'Windows':
            # SSH on Windows expects forward slashes
            return str(key_path).replace('\\', '/')
        else:
            return str(key_path)

    def _get_ssh_command_base(self) -> List[str]:
        """Get base SSH command with proper key handling."""
        cmd = ["ssh"]
        
        if self.key_file:
            cmd.extend(["-i", self.key_file])
        
        # Add common SSH options for non-interactive, secure connections
        cmd.extend([
            "-o", "BatchMode=yes",  # Don't prompt for passwords
            "-o", "StrictHostKeyChecking=no",  # Don't prompt for host key verification
            "-o", "UserKnownHostsFile=/dev/null",  # Don't save host keys
            "-o", "LogLevel=ERROR",  # Reduce verbose output
            "-o", "ConnectTimeout=30",  # Connection timeout
            "-o", "PasswordAuthentication=no",  # Explicitly disable password auth
            "-o", "PubkeyAuthentication=yes",  # Ensure pubkey auth is enabled
            "-o", "PreferredAuthentications=publickey"  # Only use public key auth
        ])
        
        cmd.append(self.ssh_target)
        return cmd

    def execute_command(self, command: str, timeout: int = 300) -> Tuple[bool, str, str]:
        """
        Execute a command on the HPC system.
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        ssh_cmd = self._get_ssh_command_base()
        ssh_cmd.append(command)
        
        logger.debug(f"Executing SSH command: {' '.join(ssh_cmd[:4])} [command hidden]")
        
        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )
            
            success = result.returncode == 0
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            logger.error(f"SSH command timed out after {timeout} seconds")
            return False, "", "Command timed out"
        except Exception as e:
            logger.error(f"Error executing SSH command: {e}")
            return False, "", str(e)

    def rsync_transfer(self, source_path: str, target_path: str, source_is_local: bool = True,
                      options: Dict[str, Any] = None, show_progress: bool = False,
                      progress_callback=None, return_process: bool = False):
        """
        Transfer files using rsync with proper SSH key handling.
        
        Args:
            source_path: Source file/directory path
            target_path: Target file/directory path  
            source_is_local: Whether source is local (True) or remote (False)
            options: Rsync options dictionary
            show_progress: Whether to show progress
            progress_callback: Callback function for progress updates
            return_process: Whether to return the process object
            
        Returns:
            Tuple of (success, summary) or (success, summary, process) if return_process=True
        """
        # Check if rsync is available
        rsync_available = shutil.which("rsync") is not None
        if not rsync_available:
            logger.warning("rsync command not found. Will use PowerShell fallback for file transfers.")
            # For now, let's implement a simple copy fallback
            return self._fallback_transfer(source_path, target_path, source_is_local)
        
        # Build rsync command with SSH options
        cmd = ["rsync"]
        
        # Add SSH options including key file
        ssh_opts = []
        if self.key_file:
            ssh_opts.extend(["-i", self.key_file])
        
        # Add SSH connection options
        ssh_opts.extend([
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no", 
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=30"
        ])
        
        # Set SSH command for rsync
        cmd.extend(["-e", f"ssh {' '.join(ssh_opts)}"])
        
        # Add rsync options
        if options:
            if options.get("archive", True):
                cmd.append("-a")
            if options.get("verbose", False):
                cmd.append("-v")
            if options.get("compress", False):
                cmd.append("-z")
            if options.get("partial", False):
                cmd.append("--partial")
            if show_progress or options.get("progress", False):
                cmd.append("--progress")
            if options.get("bwlimit"):
                cmd.extend(["--bwlimit", str(options["bwlimit"])])
        
        # Construct source and target paths
        if source_is_local:
            src = source_path
            
            # For remote target, combine base_path with target_path if target_path is relative
            if not target_path.startswith("/") and self.base_path:
                full_target_path = f"{self.base_path}/{target_path}"
            else:
                full_target_path = target_path
            
            dst = f"{self.ssh_target}:{full_target_path}"
        else:
            # For remote source, combine base_path with source_path if source_path is relative
            if not source_path.startswith("/") and self.base_path:
                full_source_path = f"{self.base_path}/{source_path}"
            else:
                full_source_path = source_path
            
            src = f"{self.ssh_target}:{full_source_path}"
            dst = target_path
        
        cmd.extend([src, dst])
        
        logger.debug(f"Executing rsync command: {' '.join(cmd[:6])} [paths hidden]")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            if return_process:
                # Return immediately with process for monitoring
                return True, "Transfer started", process
            
            # Monitor progress if callback provided
            if progress_callback and show_progress:
                self._monitor_rsync_progress(process, progress_callback)
            
            stdout, stderr = process.communicate()
            success = process.returncode == 0
            
            if success:
                return True, f"Transfer completed successfully"
            else:
                logger.error(f"rsync failed: {stderr}")
                return False, f"Transfer failed: {stderr}"
                
        except Exception as e:
            logger.error(f"Error executing rsync: {e}")
            return False, f"Transfer error: {str(e)}"

    def _fallback_transfer(self, source_path: str, target_path: str, source_is_local: bool = True):
        """
        Fallback transfer method using SCP when rsync is not available.
        
        Args:
            source_path: Source file path
            target_path: Target file path
            source_is_local: Whether source is local
            
        Returns:
            Tuple of (success, summary)
        """
        # Use SCP as fallback with proper SSH key handling
        cmd = ["scp"]
        
        if self.key_file:
            cmd.extend(["-i", self.key_file])
        
        # Add SCP options
        cmd.extend([
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=30"
        ])
        
        # Add paths
        if source_is_local:
            cmd.extend([source_path, f"{self.ssh_target}:{target_path}"])
        else:
            cmd.extend([f"{self.ssh_target}:{source_path}", target_path])
        
        logger.debug(f"Executing SCP fallback: {' '.join(cmd[:4])} [paths hidden]")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                check=False
            )
            
            success = result.returncode == 0
            if success:
                return True, "SCP transfer completed successfully"
            else:
                logger.error(f"SCP failed: {result.stderr}")
                return False, f"SCP transfer failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, "SCP transfer timed out"
        except Exception as e:
            logger.error(f"Error executing SCP: {e}")
            return False, f"SCP error: {str(e)}"

    def ensure_directory(self, remote_path: str) -> bool:
        """
        Ensure a directory exists on the HPC system.
        
        Args:
            remote_path: Path on the HPC system (can be relative or absolute)
            
        Returns:
            bool: Whether the operation was successful
        """
        logger.debug(f"Ensuring directory exists on HPC: {remote_path}")
        
        # Build full path by combining base_path with remote_path
        if not remote_path.startswith("/") and self.base_path:
            full_remote_path = f"{self.base_path}/{remote_path}"
        else:
            full_remote_path = remote_path
        
        logger.debug(f"Full remote path: {full_remote_path}")
        
        # Create directory via SSH using consistent options
        try:
            success, stdout, stderr = self.execute_command(f"mkdir -p '{full_remote_path}'")
            return success
        except Exception as e:
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
            remote_path: Path to the remote file (can be relative or absolute)
            
        Returns:
            bool: True if the file exists, False otherwise
        """
        try:
            # Build full path by combining base_path with remote_path
            if not remote_path.startswith("/") and self.base_path:
                full_remote_path = f"{self.base_path}/{remote_path}"
            else:
                full_remote_path = remote_path
            
            logger.debug(f"Checking file existence: {full_remote_path}")
            
            # Use test -e to check for file or directory existence
            success, stdout, stderr = self.execute_command(f"if [ -f '{full_remote_path}' ]; then echo exists; else echo missing; fi")
            
            if success:
                return stdout.strip() == "exists"
            else:
                logger.debug(f"Command failed to check file existence: {stderr}")
                return False
            
        except Exception as e:
            logger.error(f"Error checking if file exists: {e}")
            return False

    def get_file_info(self, remote_path: str) -> Dict[str, Any]:
        """
        Get information about a file on the HPC system.
        
        Args:
            remote_path: Path to the file on HPC (can be relative or absolute)
            
        Returns:
            Dict with file information
        """
        logger.debug(f"Getting file info on HPC: {remote_path}")
        
        # Build full path by combining base_path with remote_path
        if not remote_path.startswith("/") and self.base_path:
            full_remote_path = f"{self.base_path}/{remote_path}"
        else:
            full_remote_path = remote_path
        
        result = {
            'exists': False,
            'size': None,
            'modified': None
        }
        
        try:
            # Check if file exists using consistent SSH options
            success, stdout, stderr = self.execute_command(f"test -f '{full_remote_path}' && echo exists || echo missing")
            
            if success and "exists" in stdout:
                result['exists'] = True
                
                # Get file size
                success, stdout, stderr = self.execute_command(f"stat -c %s '{full_remote_path}'")
                if success and stdout.strip():
                    result['size'] = int(stdout.strip())
                
                # Get modification time
                success, stdout, stderr = self.execute_command(f"stat -c %Y '{full_remote_path}'")
                if success and stdout.strip():
                    result['modified'] = int(stdout.strip())
            
            return result
        except Exception as e:
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
            # Try to run a simple SQLite command using consistent SSH options
            success, stdout, stderr = self.execute_command("command -v sqlite3")
            
            # If the command returns a path, SQLite is available
            if success and stdout.strip():
                self._sqlite_available = True
                self._sqlite_path = stdout.strip()
                logger.debug(f"Found SQLite at: {self._sqlite_path}")
                return True
            
            # Try with module load if available (common on HPC systems)
            success, stdout, stderr = self.execute_command("module avail sqlite 2>&1 | grep -i sqlite")
            
            if success and "sqlite" in stdout.lower():
                # Module exists, try loading it and checking sqlite3
                success, stdout, stderr = self.execute_command("module load sqlite 2>/dev/null && command -v sqlite3")
                
                if success and stdout.strip():
                    self._sqlite_available = True
                    self._sqlite_path = "module load sqlite && " + stdout.strip()
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
            db_path: Path to the SQLite database on HPC (can be relative or absolute)
            query: SQL query to execute
            
        Returns:
            Tuple containing (success, results)
        """
        logger.debug(f"Executing SQLite query on HPC: {query}")
        
        # Check if SQLite is available
        if not self.check_sqlite_availability():
            logger.error("Cannot execute SQLite query: sqlite3 command not found on remote system")
            return False, []
        
        # Build full path by combining base_path with db_path
        if not db_path.startswith("/") and self.base_path:
            full_db_path = f"{self.base_path}/{db_path}"
        else:
            full_db_path = db_path
        
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
            
            # Execute using consistent SSH options
            success, stdout, stderr = self.execute_command(f"{sqlite_command} '{full_db_path}' '{escaped_query}'")
            
            if success:
                # Parse CSV output
                rows = []
                for line in stdout.strip().split('\n'):
                    if line:
                        rows.append(line.split(','))
                return True, rows
            else:
                logger.error(f"SQLite query failed: {stderr}")
                return False, []
                
        except Exception as e:
            logger.error(f"SQLite query failed on HPC: {e}")
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
        Transfer files using rsync with proper path handling.
        
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
        # Use PowerShell fallback if rsync is not available
        if not self._rsync_available:
            logger.info("Using PowerShell fallback for file transfer")
            if source_is_local:
                # Upload using PowerShell
                return self._powershell_upload(source_path, target_path, options)
            else:
                # Download using PowerShell
                return self._powershell_download(source_path, target_path, options)
        
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
    
        # Format paths for rsync with proper base path handling
        if source_is_local:
            # Local to remote transfer
            formatted_source = source_path
            
            # For remote target, combine base_path with target_path if target_path is relative
            if not target_path.startswith("/") and self.base_path:
                full_target_path = f"{self.base_path}/{target_path}"
            else:
                full_target_path = target_path
            
            formatted_target = f"{self.ssh_target}:{full_target_path}"
            logger.info(f"Transferring from local {source_path} to HPC {full_target_path}")
        else:
            # Remote to local transfer
            # For remote source, combine base_path with source_path if source_path is relative
            if not source_path.startswith("/") and self.base_path:
                full_source_path = f"{self.base_path}/{source_path}"
            else:
                full_source_path = source_path
            
            formatted_source = f"{self.ssh_target}:{full_source_path}"
            formatted_target = target_path
            logger.info(f"Transferring from HPC {full_source_path} to local {target_path}")
    
        # Add source and destination to command
        rsync_cmd.append(formatted_source)
        rsync_cmd.append(formatted_target)
    
        start_time = time.time()
        
        logger.debug(f"Executing rsync command: {' '.join(rsync_cmd[:6])} [paths hidden]")
        
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
    
    def _powershell_upload(
        self, 
        local_path: str, 
        remote_path: str, 
        options: Dict[str, Any] = None
    ) -> Tuple[bool, str]:
        """
        Upload a file using PowerShell SCP as a fallback when rsync is not available.
        
        Args:
            local_path: Local source file path
            remote_path: Remote target path (can be relative or absolute)
            options: Transfer options (limited support)
            
        Returns:
            Tuple containing (success, output)
        """
        # Build full remote path by combining base_path with remote_path if remote_path is relative
        if not remote_path.startswith("/") and self.base_path:
            full_remote_path = f"{self.base_path}/{remote_path}"
        else:
            full_remote_path = remote_path
        
        logger.info(f"PowerShell upload: {local_path} -> {full_remote_path}")
        
        try:
            # Ensure remote directory exists using the relative path (ensure_directory handles base_path)
            remote_dir = os.path.dirname(remote_path)
            if remote_dir:
                self.ensure_directory(remote_dir)
            
            # Build PowerShell command using scp
            ps_cmd = ["powershell", "-Command"]
            
            # Construct the SCP command
            scp_cmd = f"scp"
            
            # Add key file if specified
            if self.key_file:
                expanded_key_file = os.path.expanduser(self.key_file)
                if os.path.isfile(expanded_key_file):
                    scp_cmd += f" -i '{expanded_key_file}'"
            
            # Add source and destination using the full remote path
            # Handle paths with spaces by quoting them
            quoted_local_path = f"'{local_path}'" if " " in local_path else local_path
            scp_cmd += f" {quoted_local_path} {self.ssh_target}:{full_remote_path}"
            
            # Complete the PowerShell command
            ps_cmd.append(scp_cmd)
            
            start_time = time.time()
            logger.debug(f"Executing PowerShell command: {ps_cmd}")
            
            # Execute the command
            result = subprocess.run(ps_cmd, check=True, capture_output=True, text=True)
            
            elapsed = time.time() - start_time
            logger.info(f"PowerShell transfer completed in {elapsed:.1f} seconds")
            
            return True, "Transfer completed successfully"
            
        except subprocess.SubprocessError as e:
            logger.error(f"PowerShell transfer failed: {e}")
            if isinstance(e, subprocess.CalledProcessError):
                return False, f"STDOUT: {e.stdout}\nSTDERR: {e.stderr}"
            return False, str(e)
    
    def _powershell_download(
        self, 
        remote_path: str, 
        local_path: str, 
        options: Dict[str, Any] = None
    ) -> Tuple[bool, str]:
        """
        Download a file using PowerShell SCP as a fallback when rsync is not available.
        
        Args:
            remote_path: Remote source path (can be relative or absolute)
            local_path: Local target file path
            options: Transfer options (limited support)
            
        Returns:
            Tuple containing (success, output)
        """
        # Build full remote path by combining base_path with remote_path if remote_path is relative
        if not remote_path.startswith("/") and self.base_path:
            full_remote_path = f"{self.base_path}/{remote_path}"
        else:
            full_remote_path = remote_path
        
        logger.info(f"PowerShell download: {full_remote_path} -> {local_path}")
        
        try:
            # Ensure local directory exists
            local_dir = os.path.dirname(local_path)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)
            
            # Build PowerShell command using scp
            ps_cmd = ["powershell", "-Command"]
            
            # Construct the SCP command
            scp_cmd = f"scp"
            
            # Add key file if specified
            if self.key_file:
                expanded_key_file = os.path.expanduser(self.key_file)
                if os.path.isfile(expanded_key_file):
                    scp_cmd += f" -i '{expanded_key_file}'"
            
            # Add source and destination using the full remote path
            # Handle paths with spaces by quoting them
            quoted_local_path = f"'{local_path}'" if " " in local_path else local_path
            scp_cmd += f" {self.ssh_target}:{full_remote_path} {quoted_local_path}"
            
            # Complete the PowerShell command
            ps_cmd.append(scp_cmd)
            
            start_time = time.time()
            logger.debug(f"Executing PowerShell command: {ps_cmd}")
            
            # Execute the command
            result = subprocess.run(ps_cmd, check=True, capture_output=True, text=True)
            
            elapsed = time.time() - start_time
            logger.info(f"PowerShell transfer completed in {elapsed:.1f} seconds")
            
            return True, "Transfer completed successfully"
            
        except subprocess.SubprocessError as e:
            logger.error(f"PowerShell transfer failed: {e}")
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
            tar_path: Path to the tar file on HPC (can be relative or absolute)
            extraction_dir: Directory to extract to on HPC (can be relative or absolute)
            
        Returns:
            bool: Whether the extraction was successful
        """
        logger.debug(f"Extracting tar file on HPC: {tar_path} to {extraction_dir}")
        
        # Build full paths by combining base_path with relative paths
        if not tar_path.startswith("/") and self.base_path:
            full_tar_path = f"{self.base_path}/{tar_path}"
        else:
            full_tar_path = tar_path
            
        if not extraction_dir.startswith("/") and self.base_path:
            full_extraction_dir = f"{self.base_path}/{extraction_dir}"
        else:
            full_extraction_dir = extraction_dir
        
        # Make sure the extraction directory exists
        self.ensure_directory(extraction_dir)  # Pass original path, ensure_directory will handle base_path
        
        try:
            # Build the command to extract tar with proper quoting
            # Using tar -xzf <tar_path> -C <extraction_dir>
            # The -C option changes to the extraction directory first
            cmd = f"cd '{full_extraction_dir}' && tar -xzf '{full_tar_path}'"
            
            # Execute SSH command with consistent options
            success, stdout, stderr = self.execute_command(cmd)
            
            if success:
                logger.info(f"Successfully extracted {full_tar_path} to {full_extraction_dir}")
                return True
            else:
                logger.error(f"Failed to extract {full_tar_path}: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error extracting tar file {full_tar_path}: {e}")
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
        
        # Add comprehensive SSH options for secure, non-interactive connections
        ssh_cmd.extend([
            "-o", "BatchMode=yes",  # Don't prompt for passwords
            "-o", "StrictHostKeyChecking=no",  # Don't prompt for host key verification  
            "-o", "UserKnownHostsFile=/dev/null",  # Don't save host keys
            "-o", "LogLevel=ERROR",  # Reduce verbose output
            "-o", "ConnectTimeout=30",  # Connection timeout
            "-o", "PasswordAuthentication=no",  # Explicitly disable password auth
            "-o", "PubkeyAuthentication=yes",  # Ensure pubkey auth is enabled
            "-o", "PreferredAuthentications=publickey",  # Only use public key auth
            "-o", "IdentitiesOnly=yes"  # Only use explicitly specified identity files
        ])
        
        if not self.key_file:
            logger.warning(f"No SSH key file specified for connection to {self.ssh_target}")
    
        # Add host and command - use ssh_target instead of self.host
        ssh_cmd.append(self.ssh_target)
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
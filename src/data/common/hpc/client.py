"""
Client for interacting with HPC systems via SSH and rsync.

This module provides functionality for transferring files between local workstations
and HPC clusters, executing commands remotely, and managing file synchronization.
"""
import os
import re
import time
import logging
import subprocess
import platform
import shlex
import shutil
from pathlib import Path
from typing import Callable, List, Dict, Any, Union, Tuple, Optional

logger = logging.getLogger(__name__)

#: Matches a scicore-style `transferNN.<domain>` hostname, capturing the
#: node number and domain suffix so a same-numbered `loginNN` fallback host
#: can be derived (transfer12.scicore.unibas.ch -> login12.scicore.unibas.ch).
_TRANSFER_HOST_RE = re.compile(r"^transfer(\d+)(\..+)$")


class HPCClient:
    """Client for interacting with HPC systems via SSH and rsync.

    `transfer12.scicore.unibas.ch` (this project's configured transfer
    node) has been observed to fail ~40% of SSH connection attempts
    (`kex_exchange_identification: read: Software caused connection abort`,
    exit 255) while the same-numbered login node, `login12`, is reliable
    from the same machine at the same moment -- confirmed by repeated live
    testing, not a one-off. Every SSH/SCP operation below retries against
    the configured host first (a transfer node is the *intended* place for
    bulk data movement, so it stays the default), then falls back to the
    derived login-node host as a last resort if the primary is exhausted --
    see `_run_with_fallback()`. The login-node fallback is deliberately a
    last resort, not a first choice: login nodes are shared interactive
    infrastructure, so routing every push through one by default would
    trade one kind of overload risk for another.
    """

    def __init__(self, target: str, key_file: str = None):
        """
        Initialize HPC client.

        Args:
            target: SSH target in format user@host:/path or user@host
            key_file: Path to SSH private key file (optional)
        """
        self.target = target
        self.key_file = key_file
        # Ceiling for the PowerShell/scp fallback transfer subprocess -- a second
        # line of defense behind BatchMode=yes (see _scp_non_interactive_opts()).
        self.transfer_timeout = 600

        # Parse target to extract host and path
        if ":" in target:
            self.ssh_target, self.base_path = target.split(":", 1)
        else:
            self.ssh_target = target
            self.base_path = ""

        # Also set host attribute for backward compatibility
        self.host = self.ssh_target

        self.fallback_ssh_target = self._derive_login_fallback(self.ssh_target)
        if self.fallback_ssh_target:
            logger.debug(f"Login-node fallback available: {self.fallback_ssh_target}")

        # Initialize cached attributes
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

    @staticmethod
    def _derive_login_fallback(ssh_target: str) -> Optional[str]:
        """`user@transferNN.<domain>` -> `user@loginNN.<domain>`, or `None`
        if `ssh_target`'s host doesn't match that naming pattern (nothing
        safe to derive)."""
        user_part, sep, host_part = ssh_target.rpartition("@")
        m = _TRANSFER_HOST_RE.match(host_part)
        if not m:
            return None
        login_host = f"login{m.group(1)}{m.group(2)}"
        return f"{user_part}{sep}{login_host}"

    def _run_with_fallback(
        self, attempt: Callable[[str], Tuple[bool, Any, Any]], *, op_name: str,
        max_retries: int, retry_backoff: float,
    ) -> Tuple[bool, Any, Any]:
        """Run `attempt(target)` -- returning `(success, ...)` -- against
        `self.ssh_target` up to `max_retries` times (exponential backoff),
        then one final attempt against `self.fallback_ssh_target` if the
        primary is exhausted and a fallback host is available.

        Per-attempt failures log at DEBUG -- a bulk run makes thousands of
        these calls, and most transient bounces resolve within a couple of
        retries, so WARNING-per-attempt was drowning out genuinely
        actionable log lines. Only the *outcome* (falling back, or fully
        exhausted) logs at WARNING/ERROR.
        """
        result: Tuple[bool, Any, Any] = (False, None, None)
        for attempt_n in range(1, max_retries + 1):
            result = attempt(self.ssh_target)
            if result[0]:
                return result
            if attempt_n < max_retries:
                delay = retry_backoff * (2 ** (attempt_n - 1))
                logger.debug(
                    f"{op_name} failed on {self.ssh_target} (attempt {attempt_n}/{max_retries}), retrying in {delay:.0f}s"
                )
                time.sleep(delay)

        if not self.fallback_ssh_target:
            logger.error(f"{op_name} failed on {self.ssh_target} after {max_retries} attempts (no fallback host configured)")
            return result

        logger.warning(f"{op_name} failed on {self.ssh_target} after {max_retries} attempts -- falling back to {self.fallback_ssh_target}")
        fallback_result = attempt(self.fallback_ssh_target)
        if fallback_result[0]:
            logger.info(f"{op_name} succeeded via fallback host {self.fallback_ssh_target}")
            return fallback_result
        logger.error(f"{op_name} failed on both {self.ssh_target} and fallback {self.fallback_ssh_target}")
        return fallback_result

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

    def _check_paths(self, remote_paths: List[str], test_flag: str, *, log_label: str) -> Dict[str, bool]:
        if not remote_paths:
            return {}

        resolved = {}
        for path in remote_paths:
            if not path.startswith("/") and self.base_path:
                resolved[path] = f"{self.base_path}/{path}"
            else:
                resolved[path] = path

        script = " ; ".join(
            f"if [ {test_flag} {shlex.quote(full)} ]; then echo 'EXISTS:{i}'; else echo 'MISSING:{i}'; fi"
            for i, full in enumerate(resolved.values())
        )
        success, stdout, stderr = self.execute_command(script)
        if not success:
            logger.debug(f"Command failed to check {log_label} existence: {stderr}")
            return {path: False for path in remote_paths}

        orig_paths = list(resolved.keys())
        results = {path: False for path in remote_paths}
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("EXISTS:"):
                results[orig_paths[int(line[len("EXISTS:"):])]] = True
            elif line.startswith("MISSING:"):
                results[orig_paths[int(line[len("MISSING:"):])]] = False
        return results

    def check_files_exist(self, remote_paths: List[str]) -> Dict[str, bool]:
        """Check existence of multiple remote files in one round trip.

        Same ``[ -f ... ]`` semantics and base_path resolution as
        `check_file_exists`, but batches the whole list into a single
        `execute_command` call (one SSH subprocess total) instead of one
        subprocess per path -- callers that used to sample several files via
        repeated `check_file_exists` calls (each paying its own SSH
        handshake) should use this instead.
        """
        return self._check_paths(remote_paths, "-f", log_label="file")

    def check_paths_exist(self, remote_paths: List[str]) -> Dict[str, bool]:
        """Existence check for a mix of files and directories, in one round
        trip -- `check_files_exist`'s `[ -f ... ]` reports a directory (e.g.
        an already-pushed Zarr store) as missing even when it's fully
        present, so callers that need to skip already-transferred
        directory-shaped output must use this (`[ -e ... ]`) instead."""
        return self._check_paths(remote_paths, "-e", log_label="path")

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

    def execute_command(self, command: str, *, max_retries: int = 4, retry_backoff: float = 2.0) -> Tuple[bool, str, str]:
        """
        Execute a command on the HPC system.

        Args:
            command: Command to execute

        Returns:
            Tuple containing (success, stdout, stderr)

        Retries against the configured host (exponential backoff, see
        `_run_with_fallback`'s docstring for why the transfer node needs
        this), then falls back to the derived login-node host as a last
        resort -- see the class docstring. A single failed `extract_tar()`
        call (this function's caller) fails its *entire* batch (up to
        `tar_max_files` units) with no retry above this level, so losing
        the whole batch to a connection hiccup is expensive enough to
        justify the fallback rather than just giving up.
        """
        logger.debug(f"Executing command on HPC: {command}")

        def attempt(target: str) -> Tuple[bool, str, str]:
            try:
                ssh_cmd = self._build_ssh_command(command, target=target)
                result = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
                return True, result.stdout, result.stderr
            except subprocess.SubprocessError as e:
                if isinstance(e, subprocess.CalledProcessError):
                    return False, e.stdout, e.stderr
                return False, "", str(e)

        return self._run_with_fallback(attempt, op_name="SSH command", max_retries=max_retries, retry_backoff=retry_backoff)
    
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
            logger.debug("Using PowerShell fallback for file transfer")
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
    
    def _build_scp_argv(self, *, target: str, local_path: str, remote_path: str, upload: bool) -> List[str]:
        """Full `powershell -NoProfile -Command "scp ..."` argv for one
        upload/download attempt against `target`. `-NoProfile`: this is an
        ad hoc scp wrapper, not an interactive shell -- loading the user's
        profile is unnecessary and fragile (e.g. a broken conda
        auto-activate hook there can abort the whole -Command script before
        scp even runs, misreporting a profile error as a transfer
        failure)."""
        scp_cmd = f"scp {self._scp_non_interactive_opts()}"
        if self.key_file:
            expanded_key_file = os.path.expanduser(self.key_file)
            if os.path.isfile(expanded_key_file):
                scp_cmd += f" -i '{expanded_key_file}'"

        quoted_local_path = f"'{local_path}'" if " " in local_path else local_path
        if upload:
            scp_cmd += f" {quoted_local_path} {target}:{remote_path}"
        else:
            scp_cmd += f" {target}:{remote_path} {quoted_local_path}"
        return ["powershell", "-NoProfile", "-Command", scp_cmd]

    def _run_scp(self, *, local_path: str, remote_path: str, upload: bool, op_name: str) -> Tuple[bool, str]:
        """One scp transfer via `_run_with_fallback` -- see its docstring
        and the class docstring for why this retries against the configured
        host before falling back to the derived login-node host."""

        def attempt(target: str) -> Tuple[bool, str]:
            ps_cmd = self._build_scp_argv(target=target, local_path=local_path, remote_path=remote_path, upload=upload)
            logger.debug(f"Executing PowerShell command: {ps_cmd}")
            start_time = time.time()
            try:
                # BatchMode (via _scp_non_interactive_opts) makes a broken
                # connection fail immediately instead of dropping to an
                # interactive password prompt; the subprocess timeout is a
                # second line of defense against the process hanging
                # regardless.
                subprocess.run(ps_cmd, check=True, capture_output=True, text=True, timeout=self.transfer_timeout)
                logger.debug(f"PowerShell transfer to {target} completed in {time.time() - start_time:.1f}s")
                return True, "Transfer completed successfully"
            except subprocess.TimeoutExpired:
                return False, f"transfer timed out after {self.transfer_timeout}s"
            except subprocess.SubprocessError as e:
                if isinstance(e, subprocess.CalledProcessError):
                    return False, f"STDOUT: {e.stdout}\nSTDERR: {e.stderr}"
                return False, str(e)

        return self._run_with_fallback(attempt, op_name=op_name, max_retries=3, retry_backoff=3.0)

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

        logger.debug(f"PowerShell upload: {local_path} -> {full_remote_path}")

        # Ensure remote directory exists using the relative path (ensure_directory handles base_path)
        remote_dir = os.path.dirname(remote_path)
        if remote_dir:
            self.ensure_directory(remote_dir)

        return self._run_scp(
            local_path=local_path, remote_path=full_remote_path, upload=True,
            op_name=f"PowerShell upload ({os.path.basename(local_path)})",
        )

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

        logger.debug(f"PowerShell download: {full_remote_path} -> {local_path}")

        local_dir = os.path.dirname(local_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)

        return self._run_scp(
            local_path=local_path, remote_path=full_remote_path, upload=False,
            op_name=f"PowerShell download ({os.path.basename(local_path)})",
        )

    def _scp_non_interactive_opts(self) -> str:
        """`-o` flags for the ad hoc `scp` fallback command, mirroring
        `_get_ssh_command_base()`'s options -- without `BatchMode=yes` a
        broken key/connection makes `scp` silently drop to an interactive
        password prompt, which hangs the whole pipeline (observed: a stuck
        upload blocked for over a day) instead of failing so retry logic can
        take over."""
        return (
            "-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
            "-o LogLevel=ERROR -o ConnectTimeout=30 -o PasswordAuthentication=no"
        )

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

    def _build_ssh_command(self, remote_command: str, *, target: Optional[str] = None) -> List[str]:
        """
        Build an SSH command with the appropriate options.

        Args:
            remote_command: Command to execute on the remote system
            target: SSH destination to use instead of `self.ssh_target` --
                `_run_with_fallback()` passes the derived login-node host
                here for its fallback attempt.

        Returns:
            List containing the SSH command with arguments
        """
        target = target or self.ssh_target
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
            logger.warning(f"No SSH key file specified for connection to {target}")

        ssh_cmd.append(target)
        ssh_cmd.append(remote_command)

        return ssh_cmd

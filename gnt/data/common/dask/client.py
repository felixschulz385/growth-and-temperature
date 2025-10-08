import os
import re
import logging
import dask
from dask.distributed import Client, LocalCluster
import psutil
import asyncio
import time

logger = logging.getLogger(__name__)

def init_dask_client(threads=None, memory_limit=None, dashboard_port=8787, 
                     temp_dir=None, worker_threads_per_cpu=2, 
                     worker_fraction=0.5, return_cluster=False):
    """
    Initialize a Dask client for distributed processing with optimized memory management.
    
    Args:
        threads (int, optional): Total number of threads to use. If None, uses all available cores.
        memory_limit (str or int, optional): Memory limit for workers (e.g., '4GB'). 
                                           If None, defaults to 75% of system memory.
        dashboard_port (int): Port for the Dask dashboard. Default is 8787.
        temp_dir (str, optional): Directory for temporary files. If None, uses /tmp.
        worker_threads_per_cpu (int): Number of threads per worker. Default is 2.
        worker_fraction (float): Fraction of CPUs to use for workers. Default is 0.5.
        return_cluster (bool): Whether to return the cluster along with client. Default is False.
    
    Returns:
        Client: Dask distributed client object
        or
        tuple: (Client, Cluster) if return_cluster=True
    
    Example:
        >>> client = init_dask_client()
        >>> # Use client for distributed computing
        >>> client.close()  # Close client when done
        
        # Or with custom parameters
        >>> client = init_dask_client(threads=8, memory_limit='4GB')
    """
    try:
        # Set up temporary directory
        if temp_dir is None:
            temp_dir = "/tmp/dask_temp"
        
        os.makedirs(temp_dir, exist_ok=True)
        
        # Configure Dask for better memory management and stability
        dask.config.set({
            "temporary_directory": temp_dir,
            "distributed.worker.memory.target": 0.70,  # Target memory threshold (70%)
            "distributed.worker.memory.spill": 0.80,   # Spill to disk threshold
            "distributed.worker.memory.pause": 0.90,   # Pause worker at this threshold
            "distributed.worker.memory.terminate": 0.95, # Critical threshold
            "distributed.worker.daemon": False,        # Disable daemon mode for better cleanup
            "distributed.comm.timeouts.connect": "60s", # Increase connection timeout
            "distributed.comm.timeouts.tcp": "60s",     # Increase TCP timeout
            "distributed.worker.connections.outgoing": 50, # Limit outgoing connections
            "distributed.worker.connections.incoming": 10, # Limit incoming connections
            "distributed.scheduler.idle-timeout": "1h",    # Keep scheduler alive longer
            "distributed.worker.heartbeat-interval": "5s", # More frequent heartbeats
            "distributed.scheduler.worker-ttl": "300s",    # Worker time-to-live
            # Improve shutdown behavior
            "distributed.worker.use-file-locking": False,  # Reduce file system contention
            "distributed.worker.multiprocessing-method": "spawn",  # Better process cleanup
            "distributed.nanny.pre-spawn-environ": {"OMP_NUM_THREADS": "1"},  # Prevent thread conflicts
        })
        
        # Determine number of workers and threads per worker
        if threads is None:
            available_cpus = psutil.cpu_count(logical=True)
            n_workers = max(1, int(available_cpus * worker_fraction))
        else:
            n_workers = max(1, int(threads / worker_threads_per_cpu))
        
        threads_per_worker = worker_threads_per_cpu
        
        
        # Default memory limit if not specified (70% of system memory for safety)
        if memory_limit is None:
            total_memory = psutil.virtual_memory().total
            memory_limit = int(0.70 * total_memory / n_workers)
        elif isinstance(memory_limit, str):
            memory_per_worker = int(int(re.search(r"\d*", memory_limit).group(0)) / n_workers)
            memory_limit = str(memory_per_worker) + re.search(r"[GB]i*B", memory_limit).group(0)
        else:
            memory_limit = memory_limit / n_workers
                    
        # Create LocalCluster with specific worker configuration
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            processes=True,  # Use processes instead of threads for better isolation
            dashboard_address=f':{dashboard_port}',
            local_directory=temp_dir,
            silence_logs=False,  # Keep logs for debugging
            death_timeout="30s",  # Reduced timeout for worker shutdown
            # Additional parameters for better shutdown behavior
            worker_class="distributed.Nanny",  # Use nannies for better process management
        )
        
        client = Client(cluster)
        logger.info(f"Dask client initialized with {n_workers} workers, "
                   f"{threads_per_worker} threads per worker")
        logger.info(f"Memory limit per worker: {memory_limit}")
        logger.info(f"Dask dashboard available at: {client.dashboard_link}")
        
        if return_cluster:
            return client, cluster
        else:
            return client
            
    except Exception as e:
        logger.error(f"Failed to initialize Dask client: {str(e)}")
        if return_cluster:
            return None, None
        else:
            return None

def close_client(client):
    """
    Safely close a Dask client with improved shutdown handling.
    
    Args:
        client: Dask client to close
    """
    if client is not None:
        try:
            # Cancel any pending futures first
            logger.debug("Cancelling any pending futures...")
            try:
                # Get all futures and cancel them
                futures = client.futures
                if futures:
                    logger.debug(f"Cancelling {len(futures)} pending futures")
                    client.cancel(futures)
                    # Wait briefly for cancellation to complete
                    time.sleep(0.5)
            except Exception as e:
                logger.debug(f"Error cancelling futures: {e}")
            
            # Close the client
            logger.debug("Closing Dask client...")
            client.close(timeout=10)  # 10 second timeout
            logger.debug("Closed Dask client")
        except Exception as e:
            logger.warning(f"Error closing Dask client: {str(e)}")

def close_cluster(cluster):
    """
    Safely close a Dask cluster with improved timeout handling and worker cleanup.
    
    Args:
        cluster: Dask cluster to close
    """
    if cluster is not None:
        try:
            logger.info("Shutting down Dask cluster...")
            
            # First, try to retire workers gracefully
            try:
                logger.debug("Retiring workers gracefully...")
                cluster.retire_workers(n_workers=len(cluster.workers))
                time.sleep(2)  # Give workers time to retire
            except Exception as e:
                logger.debug(f"Error retiring workers: {e}")
            
            # Close the cluster with timeout
            try:
                cluster.close(timeout=15)  # Reduced timeout to 15 seconds
                logger.info("Dask cluster closed successfully")
            except Exception as e:
                logger.warning(f"Graceful cluster shutdown failed: {e}")
                
                # Force shutdown if graceful shutdown fails
                logger.info("Attempting forced cluster shutdown...")
                try:
                    # Kill worker processes directly if they exist
                    if hasattr(cluster, 'workers'):
                        for worker_name, worker_info in cluster.workers.items():
                            try:
                                if hasattr(worker_info, 'process') and worker_info.process:
                                    logger.debug(f"Terminating worker process {worker_name}")
                                    worker_info.process.terminate()
                            except Exception as worker_e:
                                logger.debug(f"Error terminating worker {worker_name}: {worker_e}")
                    
                    # Final forced close with very short timeout
                    cluster.close(timeout=3)
                    logger.info("Forced cluster shutdown completed")
                    
                except Exception as e2:
                    logger.error(f"Forced cluster shutdown also failed: {e2}")
                    
                    # Last resort: try to kill any remaining processes
                    try:
                        _cleanup_zombie_processes()
                    except Exception as e3:
                        logger.debug(f"Error in zombie process cleanup: {e3}")
                        
        except Exception as e:
            logger.error(f"Error during Dask cluster shutdown: {e}")

def _cleanup_zombie_processes():
    """Clean up any zombie Dask worker processes."""
    try:
        import signal
        import subprocess
        
        # Find any remaining dask-worker processes
        try:
            result = subprocess.run(['pgrep', '-f', 'dask-worker'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                logger.debug(f"Found {len(pids)} zombie dask-worker processes")
                
                for pid in pids:
                    try:
                        pid_int = int(pid.strip())
                        logger.debug(f"Killing zombie process {pid_int}")
                        os.kill(pid_int, signal.SIGTERM)
                        time.sleep(0.5)
                        # If still alive, use SIGKILL
                        try:
                            os.kill(pid_int, signal.SIGKILL)
                        except ProcessLookupError:
                            pass  # Process already dead
                    except (ValueError, ProcessLookupError, PermissionError) as e:
                        logger.debug(f"Could not kill process {pid}: {e}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # pgrep not available or timeout
            pass
            
    except Exception as e:
        logger.debug(f"Error in zombie process cleanup: {e}")

class DaskClientContextManager:
    """
    Context manager for Dask client to ensure proper cleanup with improved shutdown handling.
    
    Example:
        >>> with DaskClientContextManager() as client:
        >>>     # Use client for distributed computing
        >>>     result = client.submit(func, *args).result()
    """
    
    def __init__(self, **kwargs):
        """Initialize with the same parameters as init_dask_client."""
        self.client = None
        self.cluster = None
        self.kwargs = kwargs
        self.return_cluster = kwargs.get('return_cluster', False)
        
    def __enter__(self):
        """Set up the Dask client when entering context."""
        if self.return_cluster:
            self.client, self.cluster = init_dask_client(**self.kwargs)
            return self.client
        else:
            # Always get the cluster for proper cleanup, even if not returned
            kwargs_with_cluster = self.kwargs.copy()
            kwargs_with_cluster['return_cluster'] = True
            self.client, self.cluster = init_dask_client(**kwargs_with_cluster)
            return self.client
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up Dask resources on context exit."""
        logger.debug("Exiting Dask client context")
        
        try:
            if self.client and not self.client.closed:
                # Try to retire workers gracefully before closing
                try:
                    logger.debug("Retiring workers gracefully...")
                    # Use the scheduler's retire_workers method through the client
                    if hasattr(self.client, 'retire_workers'):
                        self.client.retire_workers()
                    elif self.cluster and hasattr(self.cluster.scheduler, 'retire_workers'):
                        # Alternative: call through scheduler
                        self.cluster.scheduler.retire_workers()
                    else:
                        logger.debug("Retire workers not available, skipping")
                except Exception as e:
                    logger.debug(f"Error retiring workers: {e}")
                
                # Close client
                logger.debug("Closing Dask client...")
                self.client.close()
                
            if self.cluster and not self.cluster.closed:
                logger.debug("Closing Dask cluster...")
                self.cluster.close()
                
        except Exception as e:
            logger.warning(f"Error during Dask cleanup: {e}")
        finally:
            # Ensure references are cleared
            self.client = None
            self.cluster = None
            
        logger.debug("Dask context cleanup completed")
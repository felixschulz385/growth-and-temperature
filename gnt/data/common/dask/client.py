import os
import re
import logging
import dask
from dask.distributed import Client, LocalCluster
import psutil

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
        
        # Configure Dask for better memory management
        dask.config.set({
            "temporary_directory": temp_dir,
            "distributed.worker.memory.target": 0.75,  # Target memory threshold (75%)
            "distributed.worker.memory.spill": 0.85,   # Spill to disk threshold
            "distributed.worker.memory.pause": 0.95,   # Pause worker at this threshold
            "distributed.worker.memory.terminate": 0.98, # Critical threshold
        })
        
        # Determine number of workers and threads per worker
        if threads is None:
            available_cpus = psutil.cpu_count(logical=True)
            n_workers = max(1, int(available_cpus * worker_fraction))
        else:
            n_workers = max(1, int(threads / worker_threads_per_cpu))
        
        threads_per_worker = worker_threads_per_cpu
        
        
        # Default memory limit if not specified (75% of system memory)
        if memory_limit is None:
            total_memory = psutil.virtual_memory().total
            memory_limit = int(0.75 * total_memory / n_workers)
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
        )
        
        client = Client(cluster)
        logger.info(f"Dask client initialized with {n_workers} workers, "
                   f"{threads_per_worker} threads per worker")
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
    Safely close a Dask client.
    
    Args:
        client: Dask client to close
    """
    if client is not None:
        try:
            client.close()
            logger.debug("Closed Dask client")
        except Exception as e:
            logger.warning(f"Error closing Dask client: {str(e)}")

class DaskClientContextManager:
    """
    Context manager for Dask client to ensure proper cleanup.
    
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
        
    def __enter__(self):
        """Set up the Dask client when entering context."""
        if self.kwargs.get('return_cluster', False):
            self.client, self.cluster = init_dask_client(**self.kwargs)
            return self.client
        else:
            self.client = init_dask_client(**self.kwargs)
            return self.client
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure the client is closed when exiting context."""
        close_client(self.client)
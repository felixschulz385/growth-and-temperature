import os
import logging
import subprocess
from typing import Dict, Any, Optional
from pyspark.sql import SparkSession
import psutil

logger = logging.getLogger(__name__)

# Global flag to track if Java has been checked
_java_checked = False
_java_available = False

def _configure_spark_logging():
    """Configure Spark and py4j logging to reduce verbose output."""
    # Reduce py4j logging noise
    py4j_logger = logging.getLogger("py4j")
    py4j_logger.setLevel(logging.WARNING)
    
    # Reduce Spark logging noise
    spark_loggers = [
        "pyspark",
        "py4j.java_gateway", 
        "py4j.clientserver",
        "org.apache.spark",
        "org.eclipse.jetty",
        "org.sparkproject.jetty"
    ]
    
    for logger_name in spark_loggers:
        spark_logger = logging.getLogger(logger_name)
        spark_logger.setLevel(logging.WARNING)

def load_java_module():
    """Load Java module using module system if available."""
    global _java_checked, _java_available
    
    # Return cached result if already checked
    if _java_checked:
        return _java_available
    
    try:
        # Check if Java is already available first (most common case)
        try:
            result = subprocess.run(['java', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("Java is already available in PATH")
                _java_checked = True
                _java_available = True
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Try environment modules only if Java not found
        if os.path.exists('/usr/share/Modules/init/python.py'):
            try:
                import sys
                sys.path.insert(0, '/usr/share/Modules/init')
                import python as module_python
                module_python.module('load', 'Java/21.0.2')
                logger.info("Loaded Java/21.0.2 module using environment modules")
                _java_checked = True
                _java_available = True
                return True
            except Exception as e:
                logger.debug(f"Failed to load Java module via Python module system: {e}")
        
        # Fallback: try using subprocess to load module
        try:
            result = subprocess.run(['module', 'load', 'Java/21.0.2'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("Loaded Java/21.0.2 module using subprocess")
                _java_checked = True
                _java_available = True
                return True
            else:
                logger.debug(f"Module load failed: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug(f"Module command not available or timed out: {e}")
        
        logger.warning("Could not load Java module or find Java in PATH")
        _java_checked = True
        _java_available = False
        return False
        
    except Exception as e:
        logger.warning(f"Error loading Java module: {e}")
        _java_checked = True
        _java_available = False
        return False

def create_spark_session(config: Optional[Dict[str, Any]] = None, 
                        app_name: str = "GNTDataProcessing",
                        driver_memory: str = None,
                        driver_cores: str = None,
                        temp_dir: str = None,
                        **spark_configs) -> SparkSession:
    """
    Create and configure Spark session based on configuration.
    
    Args:
        config: Configuration dictionary containing spark settings
        app_name: Name for the Spark application
        driver_memory: Memory allocation for driver (e.g., '4g')
        driver_cores: Number of cores for driver
        temp_dir: Temporary directory for Spark
        **spark_configs: Additional Spark configuration options
        
    Returns:
        SparkSession: Configured Spark session
    """
    try:
        # Configure logging first to reduce noise
        _configure_spark_logging()
        
        # Ensure Java is loaded (only check once)
        if not load_java_module():
            logger.warning("Java module loading failed, but continuing - Spark may fail if Java is not available")
        
        # Handle configuration sources
        spark_config = config.get('spark', {}) if config else {}
        
        # Get configuration values with defaults and parameter overrides
        app_name = app_name or spark_config.get('app_name', 'GNTDataProcessing')
        driver_memory = driver_memory or spark_config.get('driver_memory') or _get_default_memory()
        driver_cores = str(driver_cores or spark_config.get('driver_cores') or _get_default_cores())
        temp_dir = temp_dir or spark_config.get('temp_dir') or '/tmp/spark'
        executor_memory = spark_config.get('executor_memory', driver_memory)
        spark_cpus = str(spark_config.get('spark_cpus', driver_cores))
        
        # Create temp directory
        os.makedirs(temp_dir, exist_ok=True)
        
        # Build Spark session with comprehensive configuration
        builder = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED") \
            .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS") \
            .config("spark.driver.memory", driver_memory) \
            .config("spark.executor.memory", executor_memory) \
            .config("spark.driver.cores", driver_cores) \
            .config("spark.executor.cores", spark_cpus) \
            .config("spark.sql.ansi.enabled", "false") \
            .config("spark.local.dir", temp_dir) \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.driver.extraJavaOptions", "--add-modules=jdk.incubator.vector") \
            .config("spark.executor.memoryFraction", "0.8") \
            .config("spark.storage.memoryFraction", "0.5") \
            .config("spark.sql.adaptive.shuffle.targetPostShuffleInputSize", "128MB") \
            .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB") \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
            .config("spark.sql.broadcastTimeout", "36000") \
            .config("spark.sql.execution.arrow.fallback.enabled", "true") \
            .config("spark.sql.files.maxPartitionBytes", "134217728") \
            .config("spark.sql.files.openCostInBytes", "4194304") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .config("spark.sql.adaptive.localShuffleReader.enabled", "true")
        
        # Add any additional spark configurations from config
        additional_configs = spark_config.get('configs', {})
        for key, value in additional_configs.items():
            builder = builder.config(key, value)
        
        # Add any passed spark_configs
        for key, value in spark_configs.items():
            spark_key = key if key.startswith('spark.') else f'spark.{key}'
            builder = builder.config(spark_key, value)
        
        # Create the session
        spark = builder.getOrCreate()
        
        # Set Spark log level to reduce verbosity
        spark.sparkContext.setLogLevel("WARN")
        
        logger.info(f"Created Spark session '{app_name}' with {driver_memory} memory and {driver_cores} cores")
        logger.info(f"Spark UI available at: {spark.sparkContext.uiWebUrl}")
        logger.info(f"Spark temp directory: {temp_dir}")
        logger.info(f"Memory configuration: executor={executor_memory}, memoryFraction=0.8, storageFraction=0.5")
        
        return spark
        
    except Exception as e:
        logger.error(f"Failed to create Spark session: {str(e)}")
        raise

def _get_default_memory() -> str:
    """Get default memory allocation based on system resources."""
    try:
        # Check for SLURM environment variable first
        spark_memory_gb = os.environ.get('SPARK_MEMORY_GB')
        if spark_memory_gb:
            return f"{spark_memory_gb}g"
        
        # Use 50% of available memory, minimum 2GB
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        default_memory_gb = max(2, int(total_memory_gb * 0.5))
        return f"{default_memory_gb}g"
    except Exception:
        return "4g"  # Fallback

def _get_default_cores() -> str:
    """Get default core count based on system resources."""
    try:
        # Check for SLURM environment variable first
        spark_cpus = os.environ.get('SPARK_CPUS')
        if spark_cpus:
            return str(spark_cpus)
        
        # Use 75% of available cores, minimum 2
        total_cores = psutil.cpu_count(logical=True)
        default_cores = max(2, int(total_cores * 0.75))
        return str(default_cores)
    except Exception:
        return "4"  # Fallback

def close_spark_session(spark: SparkSession):
    """
    Safely close a Spark session.
    
    Args:
        spark: Spark session to close
    """
    if spark is not None:
        try:
            logger.info("Closing Spark session...")
            spark.stop()
            logger.info("Spark session closed successfully")
        except Exception as e:
            logger.warning(f"Error closing Spark session: {str(e)}")

class SparkSessionContextManager:
    """
    Context manager for Spark session to ensure proper cleanup.
    
    Example:
        >>> with SparkSessionContextManager(app_name="MyApp") as spark:
        >>>     df = spark.read.parquet("data.parquet")
        >>>     # Use spark session
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize with the same parameters as create_spark_session.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional parameters for create_spark_session
        """
        self.spark = None
        self.config = config
        self.kwargs = kwargs
        
    def __enter__(self) -> SparkSession:
        """Set up the Spark session when entering context."""
        self.spark = create_spark_session(self.config, **self.kwargs)
        return self.spark
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure the Spark session is closed when exiting context."""
        if self.spark is not None:
            close_spark_session(self.spark)
            self.spark = None

def get_spark_session_for_assembly(assembly_config: Dict[str, Any]) -> SparkSession:
    """
    Convenience function to create a Spark session specifically for assembly operations.
    
    Args:
        assembly_config: Assembly configuration containing spark settings
        
    Returns:
        SparkSession: Configured Spark session for assembly
    """
    spark_config = assembly_config.get('spark', {})
    
    return create_spark_session(
        config={'spark': spark_config},
        app_name=spark_config.get('app_name', 'DataAssembly')
    )

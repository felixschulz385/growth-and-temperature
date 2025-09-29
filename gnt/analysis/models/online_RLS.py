import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import logging
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os
from tqdm import tqdm
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OnlineRLS:
    """
    Online Recursive Least Squares with cluster-robust standard errors.
    Handles large datasets that don't fit in memory.
    """
    
    def __init__(self, n_features: int, alpha: float = 1e-3, forget_factor: float = 1.0, 
                 batch_size: int = 1000):
        """
        Initialize Online RLS.
        
        Parameters:
        -----------
        n_features : int
            Number of features (including intercept if desired)
        alpha : float
            Regularization parameter for numerical stability (increased default)
        forget_factor : float
            Forgetting factor (1.0 = no forgetting, <1.0 = exponential forgetting)
        batch_size : int
            Batch size for vectorized processing
        """
        self.n_features = n_features
        self.alpha = alpha
        self.forget_factor = forget_factor
        self.batch_size = batch_size
        
        # Initialize parameter estimates
        self.theta = np.zeros(n_features)
        
        # Initialize precision matrix (inverse of covariance) - better initialization
        self.P = (1.0 / alpha) * np.eye(n_features)
        
        # Track X'X and X'y for proper parameter estimation
        self.XtX = alpha * np.eye(n_features)  # Regularized X'X
        self.Xty = np.zeros(n_features)        # X'y
        
        # Tracking statistics
        self.n_obs = 0
        self.rss = 0.0  # Residual sum of squares
        
        # Add statistics for proper R-squared calculation
        self.sum_y = 0.0       # Sum of y
        self.sum_y_squared = 0.0  # Sum of y²
        
        # For cluster-robust SE computation
        self.cluster_stats = defaultdict(lambda: {
            'X_sum': np.zeros(n_features),
            'residual_sum': 0.0,
            'count': 0,
            'XtX': np.zeros((n_features, n_features)),
            'X_residual_sum': np.zeros(n_features),
            'Xy': np.zeros(n_features)  # NEW: store X_c' y_c
        })
        
        self.cluster2_stats = defaultdict(lambda: {
            'X_sum': np.zeros(n_features),
            'residual_sum': 0.0,
            'count': 0,
            'XtX': np.zeros((n_features, n_features)),
            'X_residual_sum': np.zeros(n_features),
            'Xy': np.zeros(n_features)
        })
        
        self.intersection_stats = defaultdict(lambda: {
            'X_sum': np.zeros(n_features),
            'residual_sum': 0.0,
            'count': 0,
            'XtX': np.zeros((n_features, n_features)),
            'X_residual_sum': np.zeros(n_features),
            'Xy': np.zeros(n_features)
        })

    def _validate_and_clean_data(self, X: np.ndarray, y: np.ndarray, 
                                cluster1: Optional[np.ndarray] = None,
                                cluster2: Optional[np.ndarray] = None) -> Tuple:
        """Validate and clean input data with better logging."""
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        
        # Check for completely empty data first
        if X.shape[0] == 0 or y.shape[0] == 0:
            logger.debug("Empty input data")
            return X, y, cluster1, cluster2
        
        # Remove any NaN/inf values
        finite_X = np.isfinite(X).all(axis=1)
        finite_y = np.isfinite(y)
        valid_mask = finite_X & finite_y
        
        n_invalid = (~valid_mask).sum()
        n_total = len(valid_mask)
        
        if n_invalid > 0:
            # Only log if significant portion is invalid or if all are invalid
            if n_invalid == n_total:
                logger.debug("No valid observations in chunk")
            elif n_invalid > n_total * 0.1:  # More than 10% invalid
                logger.warning(f"Removed {n_invalid}/{n_total} ({n_invalid/n_total*100:.1f}%) invalid observations")
        
            X = X[valid_mask]
            y = y[valid_mask]
            if cluster1 is not None:
                cluster1 = cluster1[valid_mask]
            if cluster2 is not None:
                cluster2 = cluster2[valid_mask]
        
        return X, y, cluster1, cluster2

    def partial_fit(self, X: np.ndarray, y: np.ndarray, 
                   cluster1: Optional[np.ndarray] = None,
                   cluster2: Optional[np.ndarray] = None) -> 'OnlineRLS':
        """Update RLS estimates with new batch of data using vectorized operations."""
        
        X, y, cluster1, cluster2 = self._validate_and_clean_data(X, y, cluster1, cluster2)
        
        if X.shape[0] == 0:
            #logger.warning("No valid observations in batch")
            return self
        
        # Use vectorized batch processing for efficiency
        if X.shape[0] <= self.batch_size:
            self._update_vectorized(X, y, cluster1, cluster2)
        else:
            # Process in smaller chunks to manage memory
            for i in range(0, X.shape[0], self.batch_size):
                end_idx = min(i + self.batch_size, X.shape[0])
                X_chunk = X[i:end_idx]
                y_chunk = y[i:end_idx]
                cluster1_chunk = cluster1[i:end_idx] if cluster1 is not None else None
                cluster2_chunk = cluster2[i:end_idx] if cluster2 is not None else None
                
                self._update_vectorized(X_chunk, y_chunk, cluster1_chunk, cluster2_chunk)
        
        return self
    
    def _update_vectorized(self, X: np.ndarray, y: np.ndarray,
                          cluster1: Optional[np.ndarray] = None,
                          cluster2: Optional[np.ndarray] = None) -> None:
        """Vectorized RLS update for a batch of observations."""
        n_batch = X.shape[0]
        
        # Update sufficient statistics first
        self.XtX += X.T @ X
        self.Xty += X.T @ y
        self.n_obs += n_batch
        
        # Update statistics for R-squared calculation
        self.sum_y += np.sum(y)
        self.sum_y_squared += np.sum(y**2)
        
        # Solve for parameters using accumulated sufficient statistics
        try:
            self.theta = np.linalg.solve(self.XtX, self.Xty)
        except np.linalg.LinAlgError:
            # Add more regularization if singular
            regularized_XtX = self.XtX + self.alpha * 10 * np.eye(self.n_features)
            self.theta = np.linalg.solve(regularized_XtX, self.Xty)
            logger.warning("Added extra regularization due to singular matrix")
        
        # Update precision matrix for covariance estimation
        try:
            self.P = np.linalg.inv(self.XtX)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            self.P = np.linalg.pinv(self.XtX)
            logger.warning("Used pseudo-inverse for precision matrix")
        
        # Compute residuals and RSS
        predictions = X @ self.theta
        errors = y - predictions
        self.rss = np.sum(errors**2)
        # UPDATED calls include y
        self._update_cluster_stats(X, y, errors, cluster1, self.cluster_stats)
        self._update_cluster_stats(X, y, errors, cluster2, self.cluster2_stats)
        if cluster1 is not None and cluster2 is not None:
            intersection_ids = [f"{cluster1[i]}_{cluster2[i]}" for i in range(len(cluster1))]
            self._update_cluster_stats(X, y, errors, intersection_ids, self.intersection_stats)
    
    def _update_cluster_stats(self, X: np.ndarray, y: np.ndarray, errors: np.ndarray,
                              cluster_ids: Optional[np.ndarray],
                              stats_dict: Dict) -> None:
        """Update per-cluster sufficient statistics: XtX_c, X'u_c, X'y_c."""
        if cluster_ids is None:
            return
        if isinstance(cluster_ids, list):
            cluster_ids = np.array(cluster_ids)
        unique_clusters = np.unique(cluster_ids)
        for cluster_id in unique_clusters:
            mask = cluster_ids == cluster_id
            if not np.any(mask):
                continue
            Xc = X[mask]
            ec = errors[mask]
            yc = y[mask]
            stats = stats_dict[cluster_id]
            stats['X_sum'] += np.sum(Xc, axis=0)
            stats['residual_sum'] += np.sum(ec)
            stats['count'] += Xc.shape[0]
            stats['XtX'] += Xc.T @ Xc
            # X'u_c
            stats['X_residual_sum'] += (Xc * ec.reshape(-1, 1)).sum(axis=0)
            # X'y_c
            stats['Xy'] += Xc.T @ yc

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X = np.atleast_2d(X)
        return X @ self.theta
    
    def get_covariance_matrix(self) -> np.ndarray:
        """Get parameter covariance matrix (non-robust)."""
        sigma2 = self.rss / max(1, self.n_obs - self.n_features)
        return sigma2 * self.P
    
    def get_cluster_robust_covariance(self, cluster_type: str = 'one_way') -> np.ndarray:
        """Compute cluster-robust covariance matrix."""
        if cluster_type == 'one_way':
            return self._compute_cluster_covariance(self.cluster_stats)
        elif cluster_type == 'two_way':
            # Cameron-Gelbach-Miller formula: V = V1 + V2 - V_intersection
            V1 = self._compute_cluster_covariance(self.cluster_stats)
            V2 = self._compute_cluster_covariance(self.cluster2_stats)
            V_int = self._compute_cluster_covariance(self.intersection_stats)
            return V1 + V2 - V_int
        else:
            raise ValueError("cluster_type must be 'one_way' or 'two_way'")
    
    def _compute_cluster_covariance(self, stats_dict: Dict) -> np.ndarray:
        """Compute cluster-robust covariance from cluster statistics."""
        # Bread (X'X)^{-1}
        XtX_inv = self.P
        
        # Meat: sum over clusters of (X_c' * residual_c)' * (X_c' * residual_c)
        meat = np.zeros((self.n_features, self.n_features))
        
        for cluster_id, stats in stats_dict.items():
            if stats['count'] > 0:
                # Use the correct sum of score vectors directly
                meat += np.outer(stats['X_residual_sum'], stats['X_residual_sum'])
        
        # Sandwich estimator
        n_clusters = len(stats_dict)
        if n_clusters <= 1:
            warnings.warn("Insufficient clusters for robust standard errors")
            return self.get_covariance_matrix()
        
        # Small sample correction
        correction = n_clusters / (n_clusters - 1) * (self.n_obs - 1) / (self.n_obs - self.n_features)
        
        return correction * XtX_inv @ meat @ XtX_inv
    
    def get_standard_errors(self, cluster_type: str = 'classical') -> np.ndarray:
        """Get standard errors."""
        if cluster_type == 'classical':
            cov_matrix = self.get_covariance_matrix()
        else:
            cov_matrix = self.get_cluster_robust_covariance(cluster_type)
        
        return np.sqrt(np.diag(cov_matrix))
    
    def summary(self, cluster_type: str = 'classical', 
               bootstrap: bool = False, 
               n_bootstrap: int = 999,
               bootstrap_cluster_var: str = None,
               n_workers: int = 1) -> pd.DataFrame:
        """Get regression summary (bootstrapping removed)."""
        # Get standard errors
        se = self.get_standard_errors(cluster_type)
        t_stats = self.theta / se
        
        # Simple two-tailed p-values (assuming normal distribution)
        from scipy import stats
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
        
        # Create summary DataFrame
        summary_df = pd.DataFrame({
            'coefficient': self.theta,
            'std_error': se,
            't_statistic': t_stats,
            'p_value': p_values
        })
        
        # Remove all bootstrap logic
        return summary_df

    def get_total_sum_squares(self) -> float:
        """Compute the total sum of squares (TSS) properly."""
        if self.n_obs <= 1:
            return 0.0
        
        # TSS = sum(y²) - (sum(y)²/n)
        y_mean = self.sum_y / self.n_obs
        return self.sum_y_squared - (self.sum_y**2) / self.n_obs
        
    def get_r_squared(self) -> float:
        """Calculate R-squared properly."""
        tss = self.get_total_sum_squares()
        if tss <= 0:
            return 0.0
        return 1.0 - (self.rss / tss)
        
    def get_adjusted_r_squared(self) -> float:
        """Calculate adjusted R-squared."""
        r_squared = self.get_r_squared()
        if self.n_obs <= self.n_features:
            return 0.0
        return 1.0 - ((1.0 - r_squared) * (self.n_obs - 1) / (self.n_obs - self.n_features))

    def merge_statistics(self, other: 'OnlineRLS') -> None:
        """Merge statistics from another OnlineRLS instance."""
        
        # Merge sufficient statistics properly
        self.XtX += other.XtX - self.alpha * np.eye(self.n_features)  # Remove one regularization
        self.Xty += other.Xty
        self.n_obs += other.n_obs
        self.rss += other.rss
        
        # Merge R-squared statistics
        self.sum_y += other.sum_y
        self.sum_y_squared += other.sum_y_squared
        
        # Recompute parameters from merged sufficient statistics
        try:
            self.theta = np.linalg.solve(self.XtX, self.Xty)
            self.P = np.linalg.inv(self.XtX)
        except np.linalg.LinAlgError:
            # Add regularization if needed
            regularized_XtX = self.XtX + self.alpha * np.eye(self.n_features)
            self.theta = np.linalg.solve(regularized_XtX, self.Xty)
            self.P = np.linalg.inv(regularized_XtX)
            logger.warning("Added regularization during merge")
        
        # Merge cluster statistics
        self._merge_cluster_stats(other.cluster_stats, self.cluster_stats)
        self._merge_cluster_stats(other.cluster2_stats, self.cluster2_stats)
        self._merge_cluster_stats(other.intersection_stats, self.intersection_stats)
            
    def _merge_cluster_stats(self, source_stats: Dict, target_stats: Dict) -> None:
        """Merge cluster statistics."""
        for cluster_id, stats in source_stats.items():
            if cluster_id in target_stats:
                target_stats[cluster_id]['X_sum'] += stats['X_sum']
                target_stats[cluster_id]['residual_sum'] += stats['residual_sum']
                target_stats[cluster_id]['count'] += stats['count']
                target_stats[cluster_id]['XtX'] += stats['XtX']
                target_stats[cluster_id]['X_residual_sum'] += stats['X_residual_sum']
                target_stats[cluster_id]['Xy'] += stats['Xy']  # NEW
            else:
                target_stats[cluster_id] = {
                    'X_sum': stats['X_sum'].copy(),
                    'residual_sum': stats['residual_sum'],
                    'count': stats['count'],
                    'XtX': stats['XtX'].copy(),
                    'X_residual_sum': stats['X_residual_sum'].copy(),
                    'Xy': stats['Xy'].copy()  # NEW
                }

def process_partition_worker(args: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int, Dict, Dict, Dict]:
    """Worker function to process a single partition with better error handling and data validation."""
    (partition_file, feature_cols, target_col, cluster1_col, cluster2_col, 
     add_intercept, n_features, alpha, forget_factor, chunk_size, verbose) = args
    
    worker_logger = logging.getLogger(f"worker_{mp.current_process().pid}")
    worker_logger.info(f"Processing partition: {partition_file}")
    
    try:
        # Initialize local RLS
        local_rls = OnlineRLS(n_features=n_features, alpha=alpha, 
                             forget_factor=forget_factor, batch_size=min(chunk_size, 5000))
        
        # Use PyArrow to read partition
        import pyarrow.parquet as pq
        
        parquet_file = pq.ParquetFile(partition_file)
        total_rows = parquet_file.metadata.num_rows
        worker_logger.info(f"Partition has {total_rows:,} rows")
        
        # Adaptive chunk sizing based on partition size
        if total_rows > 50_000_000:  # Very large partitions
            effective_chunk_size = min(chunk_size, 10000)
        elif total_rows > 10_000_000:  # Large partitions
            effective_chunk_size = min(chunk_size, 20000)
        else:
            effective_chunk_size = min(chunk_size, 50000)
        
        worker_logger.info(f"Using chunk size: {effective_chunk_size:,}")
        
        # Early data validation - check first small batch
        first_batch = next(parquet_file.iter_batches(batch_size=1000))
        first_df = first_batch.to_pandas()
        
        # Validate columns exist
        missing_cols = []
        for col in feature_cols:
            if col not in first_df.columns:
                missing_cols.append(col)
        if target_col not in first_df.columns:
            missing_cols.append(target_col)
        
        if missing_cols:
            worker_logger.error(f"Missing required columns: {missing_cols}")
            return (alpha * np.eye(n_features), np.zeros(n_features), np.zeros(n_features), 0.0, 0, {}, {}, {})
        
        # Check data quality in first batch
        y_sample = first_df[target_col].values
        valid_y_ratio = np.isfinite(y_sample).mean()
        
        if valid_y_ratio < 0.01:  # Less than 1% valid data
            worker_logger.warning(f"Partition has very low data quality ({valid_y_ratio*100:.1f}% valid). Skipping.")
            return (alpha * np.eye(n_features), np.zeros(n_features), np.zeros(n_features), 0.0, 0, {}, {}, {})
        
        worker_logger.info(f"Data quality check: {valid_y_ratio*100:.1f}% valid observations in sample")
        
        # Reset file iterator
        parquet_file = pq.ParquetFile(partition_file)
        
        chunks_processed = 0
        valid_chunks = 0
        
        # Process file in chunks
        for batch in parquet_file.iter_batches(batch_size=effective_chunk_size):
            try:
                chunk_df = batch.to_pandas()
                chunks_processed += 1
                
                if chunk_df.empty:
                    continue
                
                # Extract and validate data
                X = chunk_df[feature_cols].values.astype(np.float32)
                y = chunk_df[target_col].values.astype(np.float32)
                
                # Quick validity check
                valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
                valid_ratio = valid_mask.mean()
                
                if valid_ratio < 0.001:  # Less than 0.1% valid
                    continue  # Skip chunk instead of logging debug message
                
                if not valid_mask.all():
                    X = X[valid_mask]
                    y = y[valid_mask]
                
                if X.shape[0] == 0:
                    continue  # Skip empty chunks silently
                
                # Add intercept if requested
                if add_intercept:
                    intercept = np.ones((X.shape[0], 1), dtype=np.float32)
                    X = np.column_stack([intercept, X])
                
                # Prepare cluster variables
                cluster1 = None
                cluster2 = None
                if cluster1_col and cluster1_col in chunk_df.columns:
                    cluster1 = chunk_df[cluster1_col].values
                    if not valid_mask.all():
                        cluster1 = cluster1[valid_mask]
                if cluster2_col and cluster2_col in chunk_df.columns:
                    cluster2 = chunk_df[cluster2_col].values
                    if not valid_mask.all():
                        cluster2 = cluster2[valid_mask]
                
                # Update local RLS
                local_rls.partial_fit(X, y, cluster1, cluster2)
                valid_chunks += 1
                
            except Exception as e:
                worker_logger.error(f"Error processing chunk {chunks_processed}: {e}")
                continue
            
            finally:
                # Cleanup
                del chunk_df
                if 'X' in locals():
                    del X
                if 'y' in locals():
                    del y
        
        # Log summary for this partition
        if valid_chunks == 0:
            worker_logger.warning(f"No valid chunks processed in partition {partition_file.name}")
        else:
            worker_logger.info(f"Processed {valid_chunks}/{chunks_processed} valid chunks")
        
        worker_logger.info(f"Completed partition: {partition_file.name}")
        
        return (local_rls.XtX, local_rls.Xty, local_rls.theta, local_rls.rss, local_rls.n_obs,
                dict(local_rls.cluster_stats), dict(local_rls.cluster2_stats), 
                dict(local_rls.intersection_stats))
                
    except Exception as e:
        worker_logger.error(f"Fatal error processing partition {partition_file}: {str(e)}")
        # Return empty results
        empty_XtX = alpha * np.eye(n_features)
        empty_Xty = np.zeros(n_features)
        empty_theta = np.zeros(n_features)
        return (empty_XtX, empty_Xty, empty_theta, 0.0, 0, {}, {}, {})

def discover_partitions(parquet_path: Union[str, Path]) -> List[Path]:
    """Discover all partition files in a partitioned parquet dataset."""
    parquet_path = Path(parquet_path)
    logger.info(f"Discovering partitions in {parquet_path}")
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet path {parquet_path} does not exist")
    
    # Find all .parquet files recursively
    partition_files = list(parquet_path.rglob("*.parquet"))
    
    if not partition_files:
        raise ValueError(f"No parquet files found in {parquet_path}")
    
    logger.info(f"Found {len(partition_files)} partition files")
    return sorted(partition_files)

def get_optimal_workers() -> int:
    """Determine optimal number of workers from SLURM environment or system."""
    # Try SLURM environment variables first
    for env_var in ['SLURM_CPUS_PER_TASK', 'SLURM_NTASKS', 'SLURM_JOB_CPUS_PER_NODE']:
        slurm_cpus = os.environ.get(env_var)
        if slurm_cpus:
            n_workers = int(slurm_cpus)
            logger.info(f"Using {env_var}: {n_workers} workers")
            return n_workers
    
    # Fall back to system CPU count
    n_workers = mp.cpu_count()
    logger.info(f"No SLURM environment detected, using system CPU count: {n_workers} workers")
    return n_workers


def process_partitioned_dataset_parallel(
    parquet_path: Union[str, Path],
    feature_cols: List[str] = None,
    target_col: str = None,
    cluster1_col: str = None,
    cluster2_col: str = None,
    add_intercept: bool = True,
    chunk_size: int = 10000,  # Smaller default
    n_workers: int = None,
    alpha: float = 1e-3,
    forget_factor: float = 1.0,
    show_progress: bool = True,
    verbose: bool = True,
    max_workers_override: Optional[int] = None
    # Remove all bootstrap arguments
    ) -> OnlineRLS:
        """Process partitioned parquet dataset in parallel (bootstrapping removed)."""
        start_time = time.time()
        parquet_path = Path(parquet_path)
        
        if n_workers is None:
            n_workers = get_optimal_workers()
        
        # Reduce workers for very large datasets to avoid memory pressure
        if n_workers > 6:
            n_workers = min(n_workers, 4)  # Cap at 4 workers for stability
            logger.info(f"Reduced workers to {n_workers} for large dataset processing")
        
        logger.info(f"Starting parallel processing with {n_workers} workers")
        logger.info(f"Dataset path: {parquet_path}")
        logger.info(f"Verbosity: {'enabled' if verbose else 'disabled'}")
        
        # Log SLURM environment info if available
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        if slurm_job_id:
            logger.info(f"Running in SLURM job {slurm_job_id}")
            slurm_node = os.environ.get('SLURM_NODELIST', 'unknown')
            logger.info(f"SLURM node(s): {slurm_node}")
        
        # Discover all partitions
        partition_files = discover_partitions(parquet_path)
        
        # Add partition pre-filtering based on size/location
        logger.info(f"Pre-filtering {len(partition_files)} partitions...")
        
        # Quick check for obviously problematic partitions
        valid_partitions = []
        skipped_partitions = []
        
        for partition_file in partition_files:
            try:
                # Check file size - skip very small files
                file_size = partition_file.stat().st_size
                if file_size < 1024:  # Less than 1KB
                    skipped_partitions.append(str(partition_file))
                    continue
                
                valid_partitions.append(partition_file)
                
            except Exception as e:
                logger.warning(f"Cannot access partition {partition_file}: {e}")
                skipped_partitions.append(str(partition_file))
        
        if skipped_partitions:
            logger.info(f"Skipped {len(skipped_partitions)} problematic partitions")
        
        partition_files = valid_partitions
        logger.info(f"Processing {len(partition_files)} valid partitions")
        
        # For very large datasets, process a sample first to determine structure
        sample_size = min(5, len(partition_files))  # Use first 5 partitions as sample
        sample_files = partition_files[:sample_size]
        
        logger.info(f"Using {sample_size} partitions to determine data structure")
        
        # Read first partition to determine feature structure
        first_df = None
        for partition_file in sample_files:
            try:
                # Read a small sample from the partition
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(partition_file)
                
                # Read just the first batch to get column structure
                first_batch = next(parquet_file.iter_batches(batch_size=1000))
                first_df = first_batch.to_pandas()
                
                break  # Successfully read structure
                
            except Exception as e:
                logger.warning(f"Failed to read partition {partition_file} for structure detection: {e}")
                continue
        
        if first_df is None:
            raise ValueError("Could not read any partition to determine data structure")
        
        if feature_cols is None:
            exclude_cols = [target_col, cluster1_col, cluster2_col]
            feature_cols = [col for col in first_df.select_dtypes(include=[np.number]).columns 
                        if col not in exclude_cols]
        
        n_features = len(feature_cols)
        if add_intercept:
            n_features += 1
        
        # Create feature names for display
        display_feature_names = []
        if add_intercept:
            display_feature_names.append("intercept")
        display_feature_names.extend(feature_cols)
        
        logger.info(f"Features: {feature_cols}")
        logger.info(f"Target: {target_col}")
        logger.info(f"Total features (with intercept): {n_features}")
        logger.info(f"Chunk size: {chunk_size:,}")
        
        # Check if target and cluster columns exist
        missing_cols = []
        if target_col not in first_df.columns:
            missing_cols.append(target_col)
        if cluster1_col and cluster1_col not in first_df.columns:
            missing_cols.append(cluster1_col)
        if cluster2_col and cluster2_col not in first_df.columns:
            missing_cols.append(cluster2_col)
        
        if missing_cols:
            raise ValueError(f"Missing required columns in data: {missing_cols}")
        
        # Initialize simple progress tracking
        total_partitions = len(partition_files)
        completed_partitions = 0
        
        # Create main progress bar if requested
        main_pbar = None
        if show_progress:
            main_pbar = tqdm(
                total=total_partitions, 
                desc="Processing partitions", 
                unit="partitions",
                disable=not verbose  # Control visibility based on verbosity
            )
        
        # Prepare arguments for workers
        worker_args = [
            (partition_file, feature_cols, target_col, cluster1_col, cluster2_col,
            add_intercept, n_features, alpha, forget_factor, chunk_size, verbose)
            for partition_file in partition_files
        ]
        
        # Initialize main RLS instance
        main_rls = OnlineRLS(n_features=n_features, alpha=alpha, forget_factor=forget_factor)
        
        # Process partitions in parallel
        logger.info(f"Processing {len(partition_files)} partitions in parallel...")
        
        try:
            # Use ProcessPoolExecutor with reduced max_workers and timeout
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Submit all tasks
                future_to_partition = {
                    executor.submit(process_partition_worker, args): args[0] 
                    for args in worker_args
                }
                
                failed_partitions = []
                successful_partitions = 0
                empty_partitions = 0  # Track partitions with no valid data
                
                # Remove timeout to prevent premature termination
                for future in as_completed(future_to_partition):
                    partition_file = future_to_partition[future]
                    
                    try:
                        result = future.result(timeout=1200)  # 20 minute timeout per partition
                        XtX_update, Xty_update, theta_update, rss_update, n_obs, cluster_stats, cluster2_stats, intersection_stats = result
                        
                        if n_obs > 0:
                            # Merge results
                            temp_rls = OnlineRLS(n_features=n_features, alpha=alpha)
                            temp_rls.XtX = XtX_update
                            temp_rls.Xty = Xty_update
                            temp_rls.theta = theta_update
                            temp_rls.rss = rss_update
                            temp_rls.n_obs = n_obs
                            temp_rls.cluster_stats = defaultdict(lambda: {
                                'X_sum': np.zeros(n_features),
                                'residual_sum': 0.0,
                                'count': 0,
                                'XtX': np.zeros((n_features, n_features)),
                                'X_residual_sum': np.zeros(n_features),
                                'Xy': np.zeros(n_features)
                            }, cluster_stats)
                            temp_rls.cluster2_stats = defaultdict(lambda: {
                                'X_sum': np.zeros(n_features),
                                'residual_sum': 0.0,
                                'count': 0,
                                'XtX': np.zeros((n_features, n_features)),
                                'X_residual_sum': np.zeros(n_features),
                                'Xy': np.zeros(n_features)
                            }, cluster2_stats)
                            temp_rls.intersection_stats = defaultdict(lambda: {
                                'X_sum': np.zeros(n_features),
                                'residual_sum': 0.0,
                                'count': 0,
                                'XtX': np.zeros((n_features, n_features)),
                                'X_residual_sum': np.zeros(n_features),
                                'Xy': np.zeros(n_features)
                            }, intersection_stats)
                            main_rls.merge_statistics(temp_rls)
                            successful_partitions += 1
                        else:
                            empty_partitions += 1
                        
                        completed_partitions += 1
                        
                        # Update progress with better statistics
                        if main_pbar:
                            main_pbar.update(1)
                            main_pbar.set_postfix({
                                'completed': completed_partitions,
                                'successful': successful_partitions,
                                'empty': empty_partitions,
                                'failed': len(failed_partitions),
                                'total_obs': f"{main_rls.n_obs:,}",
                                'RSS': f"{main_rls.rss:.2e}" if main_rls.rss > 0 else "0"
                            })
                        
                        # Log current coefficients periodically
                        if verbose and (completed_partitions % 25 == 0 or completed_partitions == total_partitions):
                            logger.info(f"Progress update - Partition {completed_partitions}/{total_partitions}")
                            coeff_str = ", ".join([f"{name}={coeff:.4f}" for name, coeff in zip(display_feature_names, main_rls.theta)])
                            logger.info(f"Current coefficients: {coeff_str}")
                            logger.info(f"RSS: {main_rls.rss:.6f}, Observations: {main_rls.n_obs:,}")
                        
                    except Exception as e:
                        failed_partitions.append(str(partition_file))
                        completed_partitions += 1
                        logger.error(f"Failed to process partition {partition_file}: {str(e)}")
                        
                        # Update progress even for failed partitions
                        if main_pbar:
                            main_pbar.update(1)
                            main_pbar.set_postfix({
                                'completed': completed_partitions,
                                'successful': successful_partitions,
                                'empty': empty_partitions,
                                'failed': len(failed_partitions),
                                'total_obs': f"{main_rls.n_obs:,}",
                                'RSS': f"{main_rls.rss:.2e}" if main_rls.rss > 0 else "0"
                            })
        
        except Exception as e:
            logger.error(f"Critical error in parallel processing: {str(e)}")
            # Continue with whatever results we have
        
        finally:
            if main_pbar:
                main_pbar.close()
        
        # Enhanced reporting
        logger.info(f"Processing completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Successful partitions: {successful_partitions}/{total_partitions}")
        logger.info(f"Empty partitions (no valid data): {empty_partitions}")
        logger.info(f"Failed partitions: {len(failed_partitions)}")
        logger.info(f"Total observations processed: {main_rls.n_obs:,}")
        
        if main_rls.n_obs == 0:
            logger.error("No observations were successfully processed!")
            logger.error("This suggests a fundamental data quality issue.")
            logger.error("Check: 1) Column names are correct, 2) Data types are numeric, 3) Files contain valid data")
            raise ValueError("Failed to process any data. Check partition files and column names.")
        
        # Warn if we have very few successful partitions
        success_rate = successful_partitions / total_partitions
        if success_rate < 0.5:
            logger.warning(f"Low success rate: {success_rate*100:.1f}% of partitions processed successfully")
            logger.warning("Consider checking data quality or reducing parallelism")
        
        return main_rls


def process_large_dataset(file_path: str, chunk_size: int = 10000,
                        feature_cols: List[str] = None,
                        target_col: str = None,
                        cluster1_col: str = None,
                        cluster2_col: str = None,
                        add_intercept: bool = True,
                        alpha: float = 1e-3,
                        verbose: bool = True) -> OnlineRLS:
    """Process large parquet dataset with online RLS."""
    logger.info(f"Processing single parquet file: {file_path}")
    logger.info(f"Verbosity: {'enabled' if verbose else 'disabled'}")
    
    import pyarrow.parquet as pq
    
    # Read parquet file metadata
    parquet_file = pq.ParquetFile(file_path)
    
    # Initialize RLS (we'll update n_features after seeing first batch)
    rls = None
    
    logger.info(f"Processing {parquet_file.metadata.num_rows:,} rows in chunks of {chunk_size:,}")
    
    # Process in chunks
    chunk_pbar = tqdm(
        total=parquet_file.metadata.num_rows,
        desc="Processing chunks",
        unit="rows",
        disable=not verbose
    ) if verbose else None
    
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
        df = batch.to_pandas()
        
        # Prepare features
        if feature_cols is None:
            # Use all numeric columns except target and cluster columns
            exclude_cols = [target_col, cluster1_col, cluster2_col]
            feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                        if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Add intercept if requested
        if add_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        # Initialize RLS on first batch
        if rls is None:
            rls = OnlineRLS(n_features=X.shape[1], alpha=alpha)
            logger.info(f"Initialized RLS with {X.shape[1]} features")
        
        # Prepare cluster variables
        cluster1 = df[cluster1_col].values if cluster1_col else None
        cluster2 = df[cluster2_col].values if cluster2_col else None
        
        # Update RLS
        rls.partial_fit(X, y, cluster1, cluster2)
        
        if chunk_pbar:
            chunk_pbar.update(len(df))
        
        if verbose and (i + 1) % 10 == 0:
            logger.info(f"Processed {(i+1)*chunk_size:,} rows...")
    
    if chunk_pbar:
        chunk_pbar.close()
    
    logger.info("Processing complete!")
    return rls
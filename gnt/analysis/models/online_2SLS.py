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

# Import the base OnlineRLS class
from .online_RLS import OnlineRLS, discover_partitions, get_optimal_workers

# Configure logging
logger = logging.getLogger(__name__)

class Online2SLS(OnlineRLS):
    """
    Online Two-Stage Least Squares with cluster-robust standard errors.
    Extends OnlineRLS to handle instrumental variables estimation.
    """
    
    def __init__(self, n_exogenous: int, n_endogenous: int, n_instruments: int, 
                 alpha: float = 1e-3, forget_factor: float = 1.0, 
                 batch_size: int = 1000):
        """
        Initialize Online 2SLS.
        
        Parameters:
        -----------
        n_exogenous : int
            Number of exogenous variables (including intercept if desired)
        n_endogenous : int  
            Number of endogenous variables
        n_instruments : int
            Number of instruments (should be >= n_endogenous for identification)
        """
        # Total features for second stage = exogenous + endogenous
        super().__init__(n_features=n_exogenous + n_endogenous, alpha=alpha, 
                        forget_factor=forget_factor, batch_size=batch_size)
        
        self.n_exogenous = n_exogenous
        self.n_endogenous = n_endogenous  
        self.n_instruments = n_instruments
        self.n_all_instruments = n_exogenous + n_instruments  # Total instruments
        
        # First stage: regress each endogenous var on all instruments
        self.first_stage_models = []
        for i in range(n_endogenous):
            fs_model = OnlineRLS(n_features=self.n_all_instruments, alpha=alpha, 
                               forget_factor=forget_factor, batch_size=batch_size)
            self.first_stage_models.append(fs_model)
        
        # Store sufficient statistics for 2SLS variance calculation
        # We need Z'Z, Z'X, Z'y where Z=[exogenous, instruments], X=[exogenous, endogenous]
        self.ZtZ = alpha * np.eye(self.n_all_instruments)  # Instruments cross-product
        self.ZtX = np.zeros((self.n_all_instruments, n_exogenous + n_endogenous))  # Z'[exog,endog]
        self.Zty = np.zeros(self.n_all_instruments)  # Z'y
        
        # Track rank condition for identification
        self.rank_deficient = False
        
    def partial_fit(self, X_exog: np.ndarray, X_endog: np.ndarray, 
                   instruments: np.ndarray, y: np.ndarray,
                   cluster1: Optional[np.ndarray] = None,
                   cluster2: Optional[np.ndarray] = None) -> 'Online2SLS':
        """
        Update 2SLS estimates with new batch of data.
        
        Parameters:
        -----------
        X_exog : array-like, shape (n_obs, n_exogenous)
            Exogenous variables (including intercept if desired)
        X_endog : array-like, shape (n_obs, n_endogenous)  
            Endogenous variables
        instruments : array-like, shape (n_obs, n_instruments)
            Instrumental variables (excluding exogenous variables)
        y : array-like, shape (n_obs,)
            Dependent variable
        """
        
        # Validate and clean data
        X_exog, y, cluster1, cluster2 = self._validate_and_clean_data(X_exog, y, cluster1, cluster2)
        X_endog, _, _, _ = self._validate_and_clean_data(X_endog, y, cluster1, cluster2)
        instruments, _, _, _ = self._validate_and_clean_data(instruments, y, cluster1, cluster2)
        
        if X_exog.shape[0] == 0 or X_endog.shape[0] == 0 or instruments.shape[0] == 0:
            #logger.warning("No valid observations in batch")
            return self
        
        # Construct full instrument matrix Z = [X_exog, instruments]
        Z = np.column_stack([X_exog, instruments])
        X_full = np.column_stack([X_exog, X_endog])
        
        # Update sufficient statistics
        self.ZtZ += Z.T @ Z
        self.ZtX += Z.T @ X_full  
        self.Zty += Z.T @ y
        
        # Update first stage models (regress each endogenous var on Z)
        for i, fs_model in enumerate(self.first_stage_models):
            fs_model.partial_fit(Z, X_endog[:, i], cluster1, cluster2)
        
        # Predict endogenous variables from first stage
        X_endog_pred = np.zeros_like(X_endog)
        for i, fs_model in enumerate(self.first_stage_models):
            X_endog_pred[:, i] = fs_model.predict(Z)
        
        # Second stage: regress y on [X_exog, X_endog_pred]
        X_second_stage = np.column_stack([X_exog, X_endog_pred])
        
        # Update second stage using parent class method
        super().partial_fit(X_second_stage, y, cluster1, cluster2)
        
        return self
    
    def get_2sls_covariance_matrix(self) -> np.ndarray:
        """
        Compute proper 2SLS covariance matrix accounting for generated regressors.
        Uses the standard 2SLS formula: (X'P_Z X)^{-1} * sigma^2
        where P_Z = Z(Z'Z)^{-1}Z' is the projection matrix onto instruments.
        """
        try:
            # Compute P_Z = Z(Z'Z)^{-1}Z' implicitly through sufficient statistics
            ZtZ_inv = np.linalg.inv(self.ZtZ)
            
            # X'P_Z X = X'Z(Z'Z)^{-1}Z'X = (Z'X)'(Z'Z)^{-1}(Z'X)
            XtPzX = self.ZtX.T @ ZtZ_inv @ self.ZtX
            
            # 2SLS variance: sigma^2 * (X'P_Z X)^{-1}
            sigma2 = self.rss / max(1, self.n_obs - self.n_features)
            
            return sigma2 * np.linalg.inv(XtPzX)
            
        except np.linalg.LinAlgError:
            self.rank_deficient = True
            warnings.warn("Rank deficient instrument matrix - identification may be weak")
            return super().get_covariance_matrix()  # Fall back to OLS
    
    def get_first_stage_statistics(self) -> Dict[str, Any]:
        """Return first stage diagnostics including F-statistics for weak instruments."""
        first_stage_stats = {}
        
        for i, fs_model in enumerate(self.first_stage_models):
            # Compute F-stat for excluded instruments
            # This requires testing H0: coefficients on excluded instruments = 0
            
            # Get covariance matrix for first stage
            fs_cov = fs_model.get_cluster_robust_covariance() if hasattr(fs_model, 'cluster_stats') else fs_model.get_covariance_matrix()
            
            # Extract coefficients and variance for excluded instruments
            excluded_coefs = fs_model.theta[self.n_exogenous:]  # Exclude exogenous vars
            excluded_cov = fs_cov[self.n_exogenous:, self.n_exogenous:]
            
            # F-statistic = beta'(Var(beta))^{-1}beta / q
            try:
                f_stat = excluded_coefs.T @ np.linalg.inv(excluded_cov) @ excluded_coefs / self.n_instruments
                first_stage_stats[f'endogenous_{i}_f_stat'] = float(f_stat)
                first_stage_stats[f'endogenous_{i}_weak_instruments'] = f_stat < 10  # Stock-Yogo critical value
            except np.linalg.LinAlgError:
                first_stage_stats[f'endogenous_{i}_f_stat'] = np.nan
                first_stage_stats[f'endogenous_{i}_weak_instruments'] = True
        
        return first_stage_stats

# Add worker function for 2SLS
def process_partition_worker_2sls(args: Tuple) -> Tuple:
    """Worker function for 2SLS estimation across partitions."""
    (partition_file, exogenous_cols, endogenous_cols, instrument_cols, target_col, 
     cluster1_col, cluster2_col, add_intercept, n_exogenous, n_endogenous, 
     n_instruments, alpha, forget_factor, chunk_size, verbose) = args
    
    worker_logger = logging.getLogger(f"worker_2sls_{mp.current_process().pid}")
    worker_logger.info(f"Processing 2SLS partition: {partition_file}")
    
    try:
        # Initialize local 2SLS
        local_2sls = Online2SLS(
            n_exogenous=n_exogenous, 
            n_endogenous=n_endogenous,
            n_instruments=n_instruments, 
            alpha=alpha,
            forget_factor=forget_factor, 
            batch_size=min(chunk_size, 5000)
        )
        
        # Use PyArrow to read partition
        import pyarrow.parquet as pq
        
        parquet_file = pq.ParquetFile(partition_file)
        total_rows = parquet_file.metadata.num_rows
        worker_logger.info(f"Partition has {total_rows:,} rows")
        
        # Adaptive chunk sizing based on partition size
        if total_rows > 50_000_000:  # Very large partitions
            effective_chunk_size = min(chunk_size, 8000)
        elif total_rows > 10_000_000:  # Large partitions
            effective_chunk_size = min(chunk_size, 15000)
        else:
            effective_chunk_size = min(chunk_size, 25000)
        
        worker_logger.info(f"Using chunk size: {effective_chunk_size:,}")
        
        # Early data validation - check first small batch
        first_batch = next(parquet_file.iter_batches(batch_size=1000))
        first_df = first_batch.to_pandas()
        
        # Validate columns exist
        missing_cols = []
        all_required_cols = exogenous_cols + endogenous_cols + instrument_cols + [target_col]
        if cluster1_col:
            all_required_cols.append(cluster1_col)
        if cluster2_col:
            all_required_cols.append(cluster2_col)
            
        for col in all_required_cols:
            if col not in first_df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            worker_logger.error(f"Missing required columns: {missing_cols}")
            return (
                alpha * np.eye(local_2sls.n_all_instruments),  # ZtZ
                np.zeros((local_2sls.n_all_instruments, n_exogenous + n_endogenous)),  # ZtX
                np.zeros(local_2sls.n_all_instruments),  # Zty
                0.0, 0, {}, {}, {},  # rss, n_obs, cluster stats
                [alpha * np.eye(local_2sls.n_all_instruments) for _ in range(n_endogenous)],  # first stage XtX
                [np.zeros(local_2sls.n_all_instruments) for _ in range(n_endogenous)]  # first stage Xty
            )
        
        # Check data quality in first batch
        y_sample = first_df[target_col].values
        valid_y_ratio = np.isfinite(y_sample).mean()
        
        if valid_y_ratio < 0.01:  # Less than 1% valid data
            worker_logger.warning(f"Partition has very low data quality ({valid_y_ratio*100:.1f}% valid). Skipping.")
            return (
                alpha * np.eye(local_2sls.n_all_instruments),
                np.zeros((local_2sls.n_all_instruments, n_exogenous + n_endogenous)),
                np.zeros(local_2sls.n_all_instruments),
                0.0, 0, {}, {}, {},
                [alpha * np.eye(local_2sls.n_all_instruments) for _ in range(n_endogenous)],
                [np.zeros(local_2sls.n_all_instruments) for _ in range(n_endogenous)]
            )
        
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
                
                # Extract data
                X_exog = chunk_df[exogenous_cols].values.astype(np.float32) if exogenous_cols else np.empty((len(chunk_df), 0), dtype=np.float32)
                X_endog = chunk_df[endogenous_cols].values.astype(np.float32)
                instruments = chunk_df[instrument_cols].values.astype(np.float32)
                y = chunk_df[target_col].values.astype(np.float32)
                
                # Quick validity check
                valid_mask_exog = np.isfinite(X_exog).all(axis=1) if X_exog.shape[1] > 0 else np.ones(len(chunk_df), dtype=bool)
                valid_mask_endog = np.isfinite(X_endog).all(axis=1)
                valid_mask_instr = np.isfinite(instruments).all(axis=1)
                valid_mask_y = np.isfinite(y)
                valid_mask = valid_mask_exog & valid_mask_endog & valid_mask_instr & valid_mask_y
                
                valid_ratio = valid_mask.mean()
                
                if valid_ratio < 0.001:  # Less than 0.1% valid
                    continue
                
                if not valid_mask.all():
                    X_exog = X_exog[valid_mask]
                    X_endog = X_endog[valid_mask]
                    instruments = instruments[valid_mask]
                    y = y[valid_mask]
                
                if len(y) == 0:
                    continue
                
                # Add intercept if requested
                if add_intercept:
                    intercept = np.ones((len(y), 1), dtype=np.float32)
                    if X_exog.shape[1] > 0:
                        X_exog = np.column_stack([intercept, X_exog])
                    else:
                        X_exog = intercept
                
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
                
                # Update local 2SLS
                local_2sls.partial_fit(X_exog, X_endog, instruments, y, cluster1, cluster2)
                valid_chunks += 1
                
            except Exception as e:
                worker_logger.error(f"Error processing chunk {chunks_processed}: {e}")
                continue
            
            finally:
                # Cleanup
                del chunk_df
                if 'X_exog' in locals():
                    del X_exog, X_endog, instruments, y
        
        # Log summary for this partition
        if valid_chunks == 0:
            worker_logger.warning(f"No valid chunks processed in partition {partition_file.name}")
        else:
            worker_logger.info(f"Processed {valid_chunks}/{chunks_processed} valid chunks")
        
        worker_logger.info(f"Completed 2SLS partition: {partition_file.name}")
        
        # Return 2SLS-specific statistics
        first_stage_XtX = [fs_model.XtX for fs_model in local_2sls.first_stage_models]
        first_stage_Xty = [fs_model.Xty for fs_model in local_2sls.first_stage_models]
        
        return (
            local_2sls.ZtZ, local_2sls.ZtX, local_2sls.Zty,
            local_2sls.rss, local_2sls.n_obs,
            dict(local_2sls.cluster_stats), dict(local_2sls.cluster2_stats), 
            dict(local_2sls.intersection_stats),
            first_stage_XtX, first_stage_Xty
        )
                
    except Exception as e:
        worker_logger.error(f"Fatal error processing 2SLS partition {partition_file}: {str(e)}")
        # Return empty results
        return (
            alpha * np.eye(n_exogenous + n_instruments + (1 if add_intercept else 0)),
            np.zeros((n_exogenous + n_instruments + (1 if add_intercept else 0), n_exogenous + n_endogenous + (1 if add_intercept else 0))),
            np.zeros(n_exogenous + n_instruments + (1 if add_intercept else 0)),
            0.0, 0, {}, {}, {},
            [alpha * np.eye(n_exogenous + n_instruments + (1 if add_intercept else 0)) for _ in range(n_endogenous)],
            [np.zeros(n_exogenous + n_instruments + (1 if add_intercept else 0)) for _ in range(n_endogenous)]
        )


def process_partitioned_dataset_2sls(
    parquet_path: Union[str, Path],
    exogenous_cols: List[str],
    endogenous_cols: List[str], 
    instrument_cols: List[str],
    target_col: str,
    cluster1_col: str = None,
    cluster2_col: str = None,
    add_intercept: bool = True,
    chunk_size: int = 10000,
    n_workers: int = None,
    alpha: float = 1e-3,
    forget_factor: float = 1.0,
    show_progress: bool = True,
    verbose: bool = True
) -> Online2SLS:
    """Process partitioned dataset with 2SLS estimation."""
    start_time = time.time()
    parquet_path = Path(parquet_path)
    
    if n_workers is None:
        n_workers = get_optimal_workers()
    
    # Reduce workers for 2SLS due to increased memory requirements
    if n_workers > 4:
        n_workers = min(n_workers, 3)  # Cap at 3 workers for 2SLS
        logger.info(f"Reduced workers to {n_workers} for 2SLS processing")
    
    logger.info(f"Starting 2SLS parallel processing with {n_workers} workers")
    logger.info(f"Dataset path: {parquet_path}")
    logger.info(f"Verbosity: {'enabled' if verbose else 'disabled'}")
    
    # Validate identification
    n_endogenous = len(endogenous_cols)
    n_instruments = len(instrument_cols)
    n_exogenous = len(exogenous_cols)
    
    if n_instruments < n_endogenous:
        raise ValueError(f"Under-identified: {n_instruments} instruments < {n_endogenous} endogenous variables")
    
    logger.info(f"2SLS specification: {n_endogenous} endogenous, {n_instruments} instruments, {n_exogenous} exogenous")
    
    # Log SLURM environment info if available
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id:
        logger.info(f"Running in SLURM job {slurm_job_id}")
    
    # Discover all partitions
    partition_files = discover_partitions(parquet_path)
    
    # Pre-filter partitions
    logger.info(f"Pre-filtering {len(partition_files)} partitions...")
    valid_partitions = []
    skipped_partitions = []
    
    for partition_file in partition_files:
        try:
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
    
    # Determine data structure from sample partitions
    sample_size = min(3, len(partition_files))
    sample_files = partition_files[:sample_size]
    
    logger.info(f"Using {sample_size} partitions to determine data structure")
    
    # Read first partition to validate column structure
    first_df = None
    for partition_file in sample_files:
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(partition_file)
            first_batch = next(parquet_file.iter_batches(batch_size=1000))
            first_df = first_batch.to_pandas()
            break
        except Exception as e:
            logger.warning(f"Failed to read partition {partition_file} for structure detection: {e}")
            continue
    
    if first_df is None:
        raise ValueError("Could not read any partition to determine data structure")
    
    # Validate all required columns exist
    all_required_cols = exogenous_cols + endogenous_cols + instrument_cols + [target_col]
    if cluster1_col:
        all_required_cols.append(cluster1_col)
    if cluster2_col:
        all_required_cols.append(cluster2_col)
    
    missing_cols = [col for col in all_required_cols if col not in first_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in data: {missing_cols}")
    
    # Calculate dimensions
    n_features_total = n_exogenous + n_endogenous
    n_all_instruments = n_exogenous + n_instruments
    if add_intercept:
        n_features_total += 1
        n_all_instruments += 1
        n_exogenous += 1  # Update for intercept
    
    # Create feature names for display
    display_feature_names = []
    if add_intercept:
        display_feature_names.append("intercept")
    display_feature_names.extend(exogenous_cols)
    display_feature_names.extend(endogenous_cols)
    
    logger.info(f"Exogenous vars: {exogenous_cols}")
    logger.info(f"Endogenous vars: {endogenous_cols}")
    logger.info(f"Instruments: {instrument_cols}")
    logger.info(f"Target: {target_col}")
    logger.info(f"Total second-stage features: {n_features_total}")
    logger.info(f"Total instruments: {n_all_instruments}")
    logger.info(f"Chunk size: {chunk_size:,}")
    
    # Initialize progress tracking
    total_partitions = len(partition_files)
    completed_partitions = 0
    
    # Create main progress bar
    main_pbar = None
    if show_progress:
        main_pbar = tqdm(
            total=total_partitions, 
            desc="Processing 2SLS partitions", 
            unit="partitions",
            disable=not verbose
        )
    
    # Prepare arguments for workers
    worker_args = [
        (partition_file, exogenous_cols, endogenous_cols, instrument_cols, target_col,
         cluster1_col, cluster2_col, add_intercept, n_exogenous, n_endogenous, 
         n_instruments, alpha, forget_factor, chunk_size, verbose)
        for partition_file in partition_files
    ]
    
    # Initialize main 2SLS instance
    main_2sls = Online2SLS(
        n_exogenous=n_exogenous, 
        n_endogenous=n_endogenous,
        n_instruments=n_instruments, 
        alpha=alpha, 
        forget_factor=forget_factor
    )
    
    # Process partitions in parallel
    logger.info(f"Processing {len(partition_files)} partitions with 2SLS in parallel...")
    
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_partition = {
                executor.submit(process_partition_worker_2sls, args): args[0] 
                for args in worker_args
            }
            
            failed_partitions = []
            successful_partitions = 0
            empty_partitions = 0
            
            for future in as_completed(future_to_partition, timeout=7200):  # 2 hour global timeout
                partition_file = future_to_partition[future]
                
                try:
                    result = future.result(timeout=900)  # 15 minute timeout per partition for 2SLS
                    (ZtZ_update, ZtX_update, Zty_update, rss_update, n_obs, 
                     cluster_stats, cluster2_stats, intersection_stats,
                     first_stage_XtX, first_stage_Xty) = result
                    
                    if n_obs > 0:
                        # Merge 2SLS results
                        temp_2sls = Online2SLS(n_exogenous=n_exogenous, n_endogenous=n_endogenous,
                                             n_instruments=n_instruments, alpha=alpha)
                        
                        # Update sufficient statistics
                        temp_2sls.ZtZ = ZtZ_update
                        temp_2sls.ZtX = ZtX_update  
                        temp_2sls.Zty = Zty_update
                        temp_2sls.n_obs = n_obs
                        temp_2sls.rss = rss_update
                        
                        # Update first stage models
                        for i, fs_model in enumerate(temp_2sls.first_stage_models):
                            fs_model.XtX = first_stage_XtX[i]
                            fs_model.Xty = first_stage_Xty[i]
                            fs_model.n_obs = n_obs
                            try:
                                fs_model.theta = np.linalg.solve(fs_model.XtX, fs_model.Xty)
                                fs_model.P = np.linalg.inv(fs_model.XtX)
                            except np.linalg.LinAlgError:
                                regularized_XtX = fs_model.XtX + alpha * np.eye(fs_model.n_features)
                                fs_model.theta = np.linalg.solve(regularized_XtX, fs_model.Xty)
                                fs_model.P = np.linalg.inv(regularized_XtX)
                        
                        # Set cluster statistics
                        temp_2sls.cluster_stats = defaultdict(lambda: {
                            'X_sum': np.zeros(n_features_total),
                            'residual_sum': 0.0,
                            'count': 0,
                            'XtX': np.zeros((n_features_total, n_features_total))
                        }, cluster_stats)
                        temp_2sls.cluster2_stats = defaultdict(lambda: {
                            'X_sum': np.zeros(n_features_total),
                            'residual_sum': 0.0,
                            'count': 0,
                            'XtX': np.zeros((n_features_total, n_features_total))
                        }, cluster2_stats)
                        temp_2sls.intersection_stats = defaultdict(lambda: {
                            'X_sum': np.zeros(n_features_total),
                            'residual_sum': 0.0,
                            'count': 0,
                            'XtX': np.zeros((n_features_total, n_features_total))
                        }, intersection_stats)
                        
                        # Merge with main 2SLS
                        main_2sls.ZtZ += temp_2sls.ZtZ - alpha * np.eye(n_all_instruments)  # Remove duplicate regularization
                        main_2sls.ZtX += temp_2sls.ZtX
                        main_2sls.Zty += temp_2sls.Zty
                        main_2sls.n_obs += temp_2sls.n_obs
                        main_2sls.rss += temp_2sls.rss
                        
                        # Merge first stage models
                        for i, (main_fs, temp_fs) in enumerate(zip(main_2sls.first_stage_models, temp_2sls.first_stage_models)):
                            main_fs.merge_statistics(temp_fs)
                        
                        # Merge cluster statistics
                        main_2sls._merge_cluster_stats(temp_2sls.cluster_stats, main_2sls.cluster_stats)
                        main_2sls._merge_cluster_stats(temp_2sls.cluster2_stats, main_2sls.cluster2_stats)
                        main_2sls._merge_cluster_stats(temp_2sls.intersection_stats, main_2sls.intersection_stats)
                        
                        successful_partitions += 1
                    else:
                        empty_partitions += 1
                    
                    completed_partitions += 1
                    
                    # Update progress
                    if main_pbar:
                        main_pbar.update(1)
                        main_pbar.set_postfix({
                            'completed': completed_partitions,
                            'successful': successful_partitions,
                            'empty': empty_partitions,
                            'failed': len(failed_partitions),
                            'total_obs': f"{main_2sls.n_obs:,}",
                            'RSS': f"{main_2sls.rss:.2e}" if main_2sls.rss > 0 else "0"
                        })
                    
                    # Log progress periodically
                    if verbose and (completed_partitions % 20 == 0 or completed_partitions == total_partitions):
                        logger.info(f"2SLS Progress update - Partition {completed_partitions}/{total_partitions}")
                        logger.info(f"RSS: {main_2sls.rss:.6f}, Observations: {main_2sls.n_obs:,}")
                    
                except Exception as e:
                    failed_partitions.append(str(partition_file))
                    completed_partitions += 1
                    logger.error(f"Failed to process 2SLS partition {partition_file}: {str(e)}")
    
    finally:
        if main_pbar:
            main_pbar.close()
    
    # Finalize 2SLS estimation
    if main_2sls.n_obs > 0:
        try:
            # Compute second stage parameters using 2SLS formula
            ZtZ_inv = np.linalg.inv(main_2sls.ZtZ)
            XtPzX = main_2sls.ZtX.T @ ZtZ_inv @ main_2sls.ZtX
            XtPzy = main_2sls.ZtX.T @ ZtZ_inv @ main_2sls.Zty
            
            main_2sls.theta = np.linalg.solve(XtPzX, XtPzy)
            main_2sls.P = np.linalg.inv(XtPzX)
            
        except np.linalg.LinAlgError:
            logger.warning("Numerical issues in 2SLS estimation, using regularized solution")
            main_2sls.rank_deficient = True
            ZtZ_reg = main_2sls.ZtZ + alpha * 10 * np.eye(n_all_instruments)
            ZtZ_inv = np.linalg.inv(ZtZ_reg)
            XtPzX = main_2sls.ZtX.T @ ZtZ_inv @ main_2sls.ZtX
            XtPzy = main_2sls.ZtX.T @ ZtZ_inv @ main_2sls.Zty
            
            main_2sls.theta = np.linalg.solve(XtPzX + alpha * np.eye(n_features_total), XtPzy)
            main_2sls.P = np.linalg.inv(XtPzX + alpha * np.eye(n_features_total))
    
    # Enhanced reporting
    logger.info(f"2SLS processing completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Successful partitions: {successful_partitions}/{total_partitions}")
    logger.info(f"Empty partitions: {empty_partitions}")
    logger.info(f"Failed partitions: {len(failed_partitions)}")
    logger.info(f"Total observations processed: {main_2sls.n_obs:,}")
    
    if main_2sls.n_obs == 0:
        logger.error("No observations were successfully processed for 2SLS!")
        raise ValueError("Failed to process any data for 2SLS estimation.")
    
    # Log final coefficient estimates
    if verbose:
        coeff_str = ", ".join([f"{name}={coeff:.4f}" for name, coeff in zip(display_feature_names, main_2sls.theta)])
        logger.info(f"Final 2SLS coefficients: {coeff_str}")
    
    return main_2sls
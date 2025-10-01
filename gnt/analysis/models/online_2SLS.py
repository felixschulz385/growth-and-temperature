import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import time
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from collections import defaultdict

# Import functionality from OnlineRLS
from gnt.analysis.models.online_RLS import (
    OnlineRLS, 
    process_partition_worker,
    discover_partitions,
    get_optimal_workers,
    _default_cluster_stats
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Online2SLS:
    """
    Online Two-Stage Least Squares with cluster-robust standard errors.
    Handles large datasets that don't fit in memory.
    
    First stage: Regress endogenous variables on instruments
    Second stage: Regress outcome on predicted endogenous variables and exogenous variables
    """
    
    def __init__(
        self, 
        n_endogenous: int, 
        n_exogenous: int, 
        n_instruments: int,
        add_intercept: bool = True,
        alpha: float = 1e-3, 
        forget_factor: float = 1.0, 
        batch_size: int = 1000
    ):
        """
        Initialize Online 2SLS.
        
        Parameters:
        -----------
        n_endogenous : int
            Number of endogenous variables
        n_exogenous : int
            Number of exogenous variables
        n_instruments : int
            Number of instruments (excluding exogenous variables)
        add_intercept : bool
            Whether to add intercept to both stages
        alpha : float
            Regularization parameter for numerical stability
        forget_factor : float
            Forgetting factor (1.0 = no forgetting, <1.0 = exponential forgetting)
        batch_size : int
            Batch size for vectorized processing
        """
        self.n_endogenous = n_endogenous
        self.n_exogenous = n_exogenous
        self.n_instruments = n_instruments
        self.add_intercept = add_intercept
        self.alpha = alpha
        self.forget_factor = forget_factor
        self.batch_size = batch_size
        
        # Calculate dimensions
        self._calc_dimensions()
        
        # Initialize first stage RLS models (one per endogenous variable)
        self.first_stage_models = [
            OnlineRLS(
                n_features=self.first_stage_dims,
                alpha=alpha,
                forget_factor=forget_factor,
                batch_size=batch_size
            )
            for _ in range(n_endogenous)
        ]
        
        # Initialize second stage model
        self.second_stage = OnlineRLS(
            n_features=self.second_stage_dims,
            alpha=alpha,
            forget_factor=forget_factor,
            batch_size=batch_size
        )
        
        self.first_stage_residuals = []
        self.total_obs = 0
        
    def _calc_dimensions(self):
        """Calculate dimensions for first and second stage models."""
        # First stage: regress endogenous on exogenous + instruments
        self.first_stage_dims = self.n_exogenous + self.n_instruments
        if self.add_intercept:
            self.first_stage_dims += 1
        
        # Second stage: regress outcome on fitted endogenous + exogenous
        self.second_stage_dims = self.n_endogenous + self.n_exogenous
        if self.add_intercept:
            self.second_stage_dims += 1
            
        logger.info(f"First stage dimensions: {self.first_stage_dims}")
        logger.info(f"Second stage dimensions: {self.second_stage_dims}")
    
    def partial_fit(
        self, 
        X_endog: np.ndarray,
        X_exog: np.ndarray,
        Z: np.ndarray,
        y: np.ndarray,
        cluster1: Optional[np.ndarray] = None,
        cluster2: Optional[np.ndarray] = None
    ) -> 'Online2SLS':
        """
        Update 2SLS estimates with new batch of data.
        
        Parameters:
        -----------
        X_endog : array-like
            Endogenous variables, shape (n_samples, n_endogenous)
        X_exog : array-like
            Exogenous variables, shape (n_samples, n_exogenous)
        Z : array-like
            Instruments, shape (n_samples, n_instruments)
        y : array-like
            Outcome variable, shape (n_samples,)
        cluster1, cluster2 : array-like, optional
            Cluster variables for robust standard errors
        """
        # Validate inputs
        X_endog, X_exog, Z, y, cluster1, cluster2 = self._validate_and_clean_data(
            X_endog, X_exog, Z, y, cluster1, cluster2
        )
        
        if len(y) == 0:
            return self
            
        self.total_obs += len(y)
        
        # Create first stage features: [intercept], exogenous, instruments
        first_stage_features = np.column_stack([X_exog, Z])
        if self.add_intercept:
            intercept = np.ones((first_stage_features.shape[0], 1))
            first_stage_features = np.column_stack([intercept, first_stage_features])
        
        # First stage: fit each endogenous variable
        X_endog_hat = np.zeros((X_endog.shape[0], self.n_endogenous))
        
        for i in range(self.n_endogenous):
            self.first_stage_models[i].partial_fit(
                first_stage_features, X_endog[:, i], cluster1, cluster2
            )
            X_endog_hat[:, i] = self.first_stage_models[i].predict(first_stage_features)
        
        # Create second stage features: [intercept], fitted endogenous, exogenous
        second_stage_features = np.column_stack([X_endog_hat, X_exog])
        if self.add_intercept:
            second_stage_features = np.column_stack([
                np.ones((second_stage_features.shape[0], 1)), 
                second_stage_features
            ])
        
        # Second stage: regress y on fitted endogenous + exogenous
        self.second_stage.partial_fit(
            second_stage_features, y, cluster1, cluster2
        )
        
        return self
    
    def _validate_and_clean_data(
        self, 
        X_endog: np.ndarray, 
        X_exog: np.ndarray, 
        Z: np.ndarray,
        y: np.ndarray,
        cluster1: Optional[np.ndarray] = None,
        cluster2: Optional[np.ndarray] = None
    ) -> Tuple:
        """Validate and clean input data."""
        X_endog = np.atleast_2d(X_endog)
        X_exog = np.atleast_2d(X_exog) if X_exog is not None else np.empty((len(y), 0))
        Z = np.atleast_2d(Z)
        y = np.atleast_1d(y)
        
        # Check dimensions
        if X_endog.shape[1] != self.n_endogenous:
            raise ValueError(f"Expected {self.n_endogenous} endogenous variables, got {X_endog.shape[1]}")
        
        if X_exog.shape[1] != self.n_exogenous:
            raise ValueError(f"Expected {self.n_exogenous} exogenous variables, got {X_exog.shape[1]}")
            
        if Z.shape[1] != self.n_instruments:
            raise ValueError(f"Expected {self.n_instruments} instruments, got {Z.shape[1]}")
        
        # Filter invalid observations
        valid_endog = np.isfinite(X_endog).all(axis=1)
        valid_exog = np.isfinite(X_exog).all(axis=1) if X_exog.size > 0 else np.ones_like(y, dtype=bool)
        valid_Z = np.isfinite(Z).all(axis=1)
        valid_y = np.isfinite(y)
        
        valid_mask = valid_endog & valid_exog & valid_Z & valid_y
        
        if not np.all(valid_mask):
            logger.debug(f"Removing {(~valid_mask).sum()}/{len(valid_mask)} invalid observations")
            X_endog = X_endog[valid_mask]
            X_exog = X_exog[valid_mask]
            Z = Z[valid_mask]
            y = y[valid_mask]
            
            if cluster1 is not None:
                cluster1 = cluster1[valid_mask]
            if cluster2 is not None:
                cluster2 = cluster2[valid_mask]
        
        return X_endog, X_exog, Z, y, cluster1, cluster2
    
    def predict(
        self, 
        X_endog: np.ndarray, 
        X_exog: np.ndarray, 
        Z: np.ndarray
    ) -> np.ndarray:
        """
        Make predictions using the 2SLS model.
        
        Parameters:
        -----------
        X_endog : array-like
            Endogenous variables (can be None if Z is provided)
        X_exog : array-like
            Exogenous variables
        Z : array-like
            Instruments (used to predict X_endog if needed)
            
        Returns:
        --------
        y_pred : array-like
            Predicted outcomes
        """
        # Clean inputs
        X_endog = np.atleast_2d(X_endog) if X_endog is not None else None
        X_exog = np.atleast_2d(X_exog) if X_exog is not None else np.empty((Z.shape[0], 0))
        Z = np.atleast_2d(Z)
        
        # Create first stage features
        first_stage_features = np.column_stack([X_exog, Z])
        if self.add_intercept:
            intercept = np.ones((first_stage_features.shape[0], 1))
            first_stage_features = np.column_stack([intercept, first_stage_features])
        
        # Get predicted endogenous variables
        X_endog_hat = np.zeros((Z.shape[0], self.n_endogenous))
        for i in range(self.n_endogenous):
            X_endog_hat[:, i] = self.first_stage_models[i].predict(first_stage_features)
        
        # Create second stage features
        second_stage_features = np.column_stack([X_endog_hat, X_exog])
        if self.add_intercept:
            second_stage_features = np.column_stack([
                np.ones((second_stage_features.shape[0], 1)), 
                second_stage_features
            ])
        
        return self.second_stage.predict(second_stage_features)
    
    def get_first_stage_summary(self, feature_names: List[str] = None) -> List[pd.DataFrame]:
        """
        Get summary of first stage regressions.
        
        Parameters:
        -----------
        feature_names : list of str, optional
            Names of features (exogenous + instruments)
            
        Returns:
        --------
        summaries : list of DataFrames
            One DataFrame per endogenous variable
        """
        summaries = []
        
        for i in range(self.n_endogenous):
            model = self.first_stage_models[i]
            summary = model.summary(cluster_type='two_way')
            
            if feature_names is not None:
                full_names = []
                if self.add_intercept:
                    full_names.append('intercept')
                full_names.extend(feature_names)
                
                if len(full_names) == len(summary):
                    summary['feature'] = full_names
            
            summary.loc['r_squared'] = ['', '', '', model.get_r_squared()]
            summary.loc['adj_r_squared'] = ['', '', '', model.get_adjusted_r_squared()]
            summary.loc['observations'] = ['', '', '', model.n_obs]
            
            summaries.append(summary)
        
        return summaries
    
    def get_second_stage_summary(
        self, 
        endog_names: List[str] = None, 
        exog_names: List[str] = None
    ) -> pd.DataFrame:
        """
        Get summary of second stage regression with corrected standard errors.
        
        Parameters:
        -----------
        endog_names : list of str, optional
            Names of endogenous variables
        exog_names : list of str, optional
            Names of exogenous variables
            
        Returns:
        --------
        summary : DataFrame
            Summary statistics
        """
        # Get basic summary from second stage
        summary = self.second_stage.summary(cluster_type='two_way')
        
        if endog_names is not None or exog_names is not None:
            full_names = []
            if self.add_intercept:
                full_names.append('intercept')
            
            if endog_names is not None:
                full_names.extend([f"{name}(fitted)" for name in endog_names])
            else:
                full_names.extend([f"endog_{i}(fitted)" for i in range(self.n_endogenous)])
                
            if exog_names is not None:
                full_names.extend(exog_names)
            else:
                full_names.extend([f"exog_{i}" for i in range(self.n_exogenous)])
                
            if len(full_names) == len(summary):
                summary['feature'] = full_names
        
        summary.loc['r_squared'] = ['', '', '', self.second_stage.get_r_squared()]
        summary.loc['adj_r_squared'] = ['', '', '', self.second_stage.get_adjusted_r_squared()]
        summary.loc['observations'] = ['', '', '', self.second_stage.n_obs]
        
        return summary
    
    def merge_with(self, other: 'Online2SLS') -> None:
        """
        Merge with another Online2SLS instance.
        
        Parameters:
        -----------
        other : Online2SLS
            Another instance to merge with
        """
        # Check compatibility
        if (self.n_endogenous != other.n_endogenous or
            self.n_exogenous != other.n_exogenous or
            self.n_instruments != other.n_instruments):
            raise ValueError("Cannot merge incompatible Online2SLS instances")
        
        for i in range(self.n_endogenous):
            self.first_stage_models[i].merge_statistics(other.first_stage_models[i])
        
        self.second_stage.merge_statistics(other.second_stage)
        self.total_obs += other.total_obs


def process_partitioned_dataset_2sls(
    parquet_path: Union[str, Path],
    endog_cols: List[str],
    exog_cols: List[str],
    instr_cols: List[str],
    target_col: str,
    cluster1_col: str = None,
    cluster2_col: str = None,
    add_intercept: bool = True,
    chunk_size: int = 10000,
    n_workers: int = None,
    alpha: float = 1e-3,
    forget_factor: float = 1.0,
    show_progress: bool = True,
    verbose: bool = True,
    feature_engineering: Optional[Dict[str, Any]] = None
) -> Online2SLS:
    """Process partitioned parquet dataset in parallel for 2SLS estimation using two passes."""
    start_time = time.time()
    parquet_path = Path(parquet_path)
    
    if n_workers is None:
        n_workers = get_optimal_workers()
    
    if n_workers > 6:
        n_workers = min(n_workers, 4)
        logger.info(f"Reduced workers to {n_workers} for large dataset processing")
    
    logger.info(f"Starting 2SLS two-pass processing with {n_workers} workers")
    logger.info(f"Dataset path: {parquet_path}")
    logger.info(f"Endogenous variables: {endog_cols}")
    logger.info(f"Exogenous variables: {exog_cols}")
    logger.info(f"Instruments: {instr_cols}")
    logger.info(f"Target variable: {target_col}")
    
    # Discover and validate partitions
    partition_files = discover_partitions(parquet_path)
    valid_partitions = []
    for partition_file in partition_files:
        try:
            file_size = partition_file.stat().st_size
            if file_size >= 1024:
                valid_partitions.append(partition_file)
        except Exception as e:
            logger.warning(f"Cannot access partition {partition_file}: {e}")
    
    partition_files = valid_partitions
    logger.info(f"Processing {len(partition_files)} valid partitions")
    
    # Initialize dimensions
    n_endogenous = len(endog_cols)
    n_exogenous = len(exog_cols)
    n_instruments = len(instr_cols)
    
    # First pass: Estimate first stage models
    logger.info("FIRST PASS: Estimating first stage regressions...")
    
    first_stage_base_features = exog_cols + instr_cols
    first_stage_fe_config = None
    
    if feature_engineering:
        first_stage_fe_config = {
            'transformations': [t for t in feature_engineering.get('transformations', []) 
                              if t.get('type') != 'predicted_substitution']
        }
        if not first_stage_fe_config['transformations']:
            first_stage_fe_config = None
    
    # Calculate first stage dimensions
    from gnt.analysis.models.feature_engineering import FeatureTransformer
    
    if first_stage_fe_config or add_intercept:
        fe_config = first_stage_fe_config or {'transformations': []}
        temp_transformer = FeatureTransformer.from_config(fe_config, first_stage_base_features, add_intercept=add_intercept)
        first_stage_dims = temp_transformer.get_n_features()
        first_stage_feature_names = temp_transformer.get_feature_names()
    else:
        first_stage_dims = len(first_stage_base_features)
        first_stage_feature_names = first_stage_base_features.copy()
        if add_intercept:
            first_stage_dims += 1
            first_stage_feature_names = ['intercept'] + first_stage_feature_names
    
    logger.info(f"First stage dimensions: {first_stage_dims}")
    
    first_stage_models = []
    for i in range(n_endogenous):
        logger.info(f"Processing first stage for endogenous variable {i+1}/{n_endogenous}: {endog_cols[i]}")
        
        first_stage_pbar = None
        if show_progress:
            first_stage_pbar = tqdm(
                total=len(partition_files),
                desc=f"First stage {endog_cols[i]}",
                unit="partitions",
                disable=not verbose
            )
        
        worker_args = [
            (partition_file, first_stage_base_features, endog_cols[i], cluster1_col, cluster2_col,
             add_intercept, first_stage_dims, alpha, forget_factor, chunk_size, verbose, 
             first_stage_fe_config)
            for partition_file in partition_files
        ]
        
        first_stage_rls = OnlineRLS(n_features=first_stage_dims, alpha=alpha, forget_factor=forget_factor)
        
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_partition = {
                    executor.submit(process_partition_worker, args): args[0]
                    for args in worker_args
                }
                
                for future in as_completed(future_to_partition):
                    try:
                        result = future.result(timeout=1200)
                        XtX_update, Xty_update, theta_update, rss_update, n_obs, cluster_stats, cluster2_stats, intersection_stats = result
                        
                        if n_obs > 0:
                            temp_rls = OnlineRLS(n_features=first_stage_dims, alpha=alpha)
                            temp_rls.XtX = XtX_update
                            temp_rls.Xty = Xty_update
                            temp_rls.theta = theta_update
                            temp_rls.rss = rss_update
                            temp_rls.n_obs = n_obs
                            temp_rls.cluster_stats = cluster_stats if cluster_stats else {}
                            temp_rls.cluster2_stats = cluster2_stats if cluster2_stats else {}
                            temp_rls.intersection_stats = intersection_stats if intersection_stats else {}
                            first_stage_rls.merge_statistics(temp_rls)
                        
                        if first_stage_pbar:
                            first_stage_pbar.update(1)
                            first_stage_pbar.set_postfix({
                                'obs': f"{first_stage_rls.n_obs:,}",
                                'rss': f"{first_stage_rls.rss:.2e}"
                            })
                        
                    except Exception as e:
                        logger.error(f"Failed to process partition in first stage: {e}")
                        if first_stage_pbar:
                            first_stage_pbar.update(1)
            
        finally:
            if first_stage_pbar:
                first_stage_pbar.close()
        
        first_stage_models.append(first_stage_rls)
        logger.info(f"First stage {i+1} complete: {first_stage_rls.n_obs:,} observations, R²={first_stage_rls.get_r_squared():.4f}")
    #
    logger.info("All first stage regressions complete!")
    
    # Second pass: Estimate second stage with existing worker
    logger.info("SECOND PASS: Estimating second stage regression...")
    
    second_stage_base_features = endog_cols + exog_cols
    second_stage_fe_config = {
        'transformations': []
    }
    
    # Add predicted substitution transformations for each endogenous variable
    for i, endogen_var in enumerate(endog_cols):
        fs_coefficients = first_stage_models[i].theta.tolist()
        
        second_stage_fe_config['transformations'].append({
            'type': 'predicted_substitution',
            'original': endogen_var,
            'predicted': f"{endogen_var}_hat",
            'first_stage_coefficients': fs_coefficients,
            'first_stage_feature_config': first_stage_fe_config,
            'first_stage_feature_names': first_stage_base_features,
            'add_intercept_first_stage': add_intercept
        })
    
    # Add any other feature engineering transformations
    if feature_engineering:
        for transform in feature_engineering.get('transformations', []):
            if transform.get('type') != 'predicted_substitution':
                second_stage_fe_config['transformations'].append(transform)
    
    temp_transformer = FeatureTransformer.from_config(
        second_stage_fe_config, 
        second_stage_base_features,
        add_intercept=add_intercept
    )
    second_stage_dims = temp_transformer.get_n_features()
    
    logger.info(f"Second stage dimensions: {second_stage_dims}")
    
    second_stage_pbar = None
    if show_progress:
        second_stage_pbar = tqdm(
            total=len(partition_files),
            desc="Second stage",
            unit="partitions",
            disable=not verbose
        )
    
    enhanced_fe_config = second_stage_fe_config.copy()
    enhanced_fe_config['_extra_input_columns'] = instr_cols
    enhanced_fe_config['_base_feature_count'] = len(second_stage_base_features)
    
    worker_args = [
        (partition_file, second_stage_base_features, target_col, cluster1_col, cluster2_col,
         add_intercept, second_stage_dims, alpha, forget_factor, chunk_size, verbose, 
         enhanced_fe_config)
        for partition_file in partition_files
    ]
    
    second_stage_rls = OnlineRLS(n_features=second_stage_dims, alpha=alpha, forget_factor=forget_factor)
    
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_partition = {
                executor.submit(process_partition_worker, args): args[0]
                for args in worker_args
            }
            
            total_obs = 0
            for future in as_completed(future_to_partition):
                try:
                    result = future.result(timeout=1200)
                    
                    if result is not None:
                        XtX_update, Xty_update, theta_update, rss_update, n_obs, cluster_stats, cluster2_stats, intersection_stats = result
                        
                        if n_obs > 0:
                            temp_rls = OnlineRLS(n_features=second_stage_dims, alpha=alpha)
                            temp_rls.XtX = XtX_update
                            temp_rls.Xty = Xty_update
                            temp_rls.theta = theta_update
                            temp_rls.rss = rss_update
                            temp_rls.n_obs = n_obs
                            temp_rls.cluster_stats = cluster_stats if cluster_stats else {}
                            temp_rls.cluster2_stats = cluster2_stats if cluster2_stats else {}
                            temp_rls.intersection_stats = intersection_stats if intersection_stats else {}
                            second_stage_rls.merge_statistics(temp_rls)
                            total_obs += n_obs
                    
                    if second_stage_pbar:
                        second_stage_pbar.update(1)
                        second_stage_pbar.set_postfix({
                            'obs': f"{second_stage_rls.n_obs:,}",
                            'rss': f"{second_stage_rls.rss:.2e}"
                        })
                    
                except Exception as e:
                    logger.error(f"Failed to process partition in second stage: {e}")
                    if second_stage_pbar:
                        second_stage_pbar.update(1)
        
    finally:
        if second_stage_pbar:
            second_stage_pbar.close()
    
    # Create final 2SLS model object
    main_model = Online2SLS(
        n_endogenous=n_endogenous,
        n_exogenous=n_exogenous,
        n_instruments=n_instruments,
        add_intercept=add_intercept,
        alpha=alpha,
        forget_factor=forget_factor
    )
    
    main_model.first_stage_models = first_stage_models
    main_model.second_stage = second_stage_rls
    main_model.total_obs = total_obs
    
    # Store feature engineering info
    if feature_engineering:
        main_model._feature_engineering_config = feature_engineering
        main_model._first_stage_fe_config = first_stage_fe_config
        main_model._second_stage_fe_config = second_stage_fe_config
        main_model._first_stage_feature_names = first_stage_feature_names
    
    logger.info(f"2SLS processing completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Total observations processed: {total_obs:,}")
    logger.info(f"Second stage R²: {second_stage_rls.get_r_squared():.4f}")
    
    return main_model
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

# Helper to consistently add intercept
def _add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones. Handles zero-column X."""
    if X is None:
        return None
    if X.shape[1] == 0:
        return np.ones((X.shape[0], 1), dtype=X.dtype)
    return np.column_stack([np.ones((X.shape[0], 1), dtype=X.dtype), X])

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
        # RENAMED: was self.ZtX
        self.ZtX_actual = np.zeros((self.n_all_instruments, n_exogenous + n_endogenous))
        self.Zty = np.zeros(self.n_all_instruments)  # Z'y
        # NEW second-stage (predicted endogenous) accumulators
        # ZtX_hat stores Z'X_hat where X_hat stacks exogenous and first-stage predicted endogenous
        self.ZtX_hat = np.zeros((self.n_all_instruments, self.n_features))
        # Zte accumulates Z'e where e are second-stage residuals (for Sargan under homoskedasticity)
        self.Zte = np.zeros(self.n_all_instruments)
        # Per-cluster instrument-residual moments S_g = Z'_g e_g (for robust Hansen J & cluster-robust covariance)
        self.cluster_instrument_resid = defaultdict(lambda: np.zeros(self.n_all_instruments))
        self.second_stage_initialized = False
        # NEW: first-stage cluster stats (instrument space)
        self.first_stage_cluster_stats = defaultdict(lambda: {
            'Z_sum': np.zeros(self.n_all_instruments),
            'count': 0
        })
        
        # Track rank condition for identification
        self.rank_deficient = False
        
        # Flag to track if first stage is finalized
        self.first_stage_finalized = False
        
        # Store original data for second pass
        self.stored_data = []
    
    def partial_fit(self, X_exog: np.ndarray, X_endog: np.ndarray, 
                   instruments: np.ndarray, y: np.ndarray,
                   cluster1: Optional[np.ndarray] = None,
                   cluster2: Optional[np.ndarray] = None,
                   store_data: bool = True) -> 'Online2SLS':
        """
        Update first stage models with new batch of data.
        
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
        store_data : bool, default=True
            Whether to store data for second pass
        """
        if self.first_stage_finalized:
            warnings.warn("First stage already finalized. Use second_stage_fit instead.")
            return self
        
        # Validate and clean data
        X_exog, y, cluster1, cluster2 = self._validate_and_clean_data(X_exog, y, cluster1, cluster2)
        X_endog, _, _, _ = self._validate_and_clean_data(X_endog, y, cluster1, cluster2)
        instruments, _, _, _ = self._validate_and_clean_data(instruments, y, cluster1, cluster2)
        
        if X_exog.shape[0] == 0 or X_endog.shape[0] == 0 or instruments.shape[0] == 0:
            return self
        
        # Construct full instrument matrix Z = [X_exog, instruments]
        Z = np.column_stack([X_exog, instruments])
        X_full = np.column_stack([X_exog, X_endog])
        # Update first-stage sufficient statistics (actual endogenous X)
        # Apply forgetting factor decay to first-stage IV stats
        if self.forget_factor < 1.0:
            gamma = self.forget_factor
            self.ZtZ *= gamma
            self.ZtX_actual *= gamma
            self.Zty *= gamma
        self.ZtZ += Z.T @ Z
        self.ZtX_actual += Z.T @ X_full
        self.Zty += Z.T @ y
        
        # Update first stage models (regress each endogenous var on Z)
        for i, fs_model in enumerate(self.first_stage_models):
            fs_model.partial_fit(Z, X_endog[:, i], cluster1, cluster2)
        
        # Store data for second pass if requested
        if store_data:
            self.stored_data.append((X_exog, X_endog, instruments, y, 
                                   cluster1.copy() if cluster1 is not None else None,
                                   cluster2.copy() if cluster2 is not None else None))
        
        return self
    
    def finalize_first_stage(self):
        """
        Finalize the first stage estimation after all data has been processed.
        This ensures the first-stage models are fully estimated before the second stage.
        """
        # Check if we already finalized
        if self.first_stage_finalized:
            return
        
        # Nothing to do for the first stage models - they're already fully updated
        # Just mark as finalized
        self.first_stage_finalized = True
        logger.info("First stage estimation finalized.")
        
        # Log first-stage diagnostics
        fs_stats = self.get_first_stage_statistics()
        for i in range(self.n_endogenous):
            logger.info(f"Endogenous variable {i} F-stat: {fs_stats[f'endogenous_{i}_f_stat']:.4f}")
            weak = fs_stats.get(f'endogenous_{i}_weak_instruments', True)
            logger.info(f"Weak instruments detected: {weak}")
        
        return self
    
    def second_stage_fit(self, X_exog: np.ndarray, X_endog: np.ndarray, 
                       instruments: np.ndarray, y: np.ndarray,
                       cluster1: Optional[np.ndarray] = None,
                       cluster2: Optional[np.ndarray] = None) -> 'Online2SLS':
        """
        Update second stage estimates using first stage predictions.
        
        Parameters:
        -----------
        X_exog : array-like, shape (n_obs, n_exogenous)
            Exogenous variables (including intercept if desired)
        X_endog : array-like, shape (n_obs, n_endogenous)  
            Endogenous variables (only needed for dimension check)
        instruments : array-like, shape (n_obs, n_instruments)
            Instrumental variables (excluding exogenous variables)
        y : array-like, shape (n_obs,)
            Dependent variable
        """
        if not self.first_stage_finalized:
            warnings.warn("First stage not finalized. Call finalize_first_stage first.")
            return self
            
        # Validate and clean data
        X_exog, y, cluster1, cluster2 = self._validate_and_clean_data(X_exog, y, cluster1, cluster2)
        X_endog, _, _, _ = self._validate_and_clean_data(X_endog, y, cluster1, cluster2)
        instruments, _, _, _ = self._validate_and_clean_data(instruments, y, cluster1, cluster2)
        
        if X_exog.shape[0] == 0 or X_endog.shape[0] == 0 or instruments.shape[0] == 0:
            return self
        
        # Construct full instrument matrix Z = [X_exog, instruments]
        Z = np.column_stack([X_exog, instruments])
        
        # Predict endogenous variables from finalized first stage models
        X_endog_pred = np.zeros_like(X_endog)
        for i, fs_model in enumerate(self.first_stage_models):
            X_endog_pred[:, i] = fs_model.predict(Z)
        
        # Second stage: regress y on [X_exog, X_endog_pred]
        X_second_stage = np.column_stack([X_exog, X_endog_pred])
        
        # --- NEW: accumulate second-stage moment components BEFORE parameter update ---
        # Residuals computed with current theta (pre-update) -> consistent as n grows
        current_theta = getattr(self, 'theta', np.zeros(self.n_features))
        residuals = y - X_second_stage @ current_theta
        # Z'X_hat
        self.ZtX_hat += Z.T @ X_second_stage
        # Z'e
        self.Zte += Z.T @ residuals
        # Cluster instrument residual accumulation
        if cluster1 is not None and len(cluster1) == len(residuals):
            cluster_ids = cluster1
        elif cluster2 is not None and len(cluster2) == len(residuals):
            cluster_ids = cluster2
        else:
            cluster_ids = np.array(['__all__'] * len(residuals))
        # Aggregate S_g = Z'_g e_g
        for gid in np.unique(cluster_ids):
            mask = (cluster_ids == gid)
            if mask.sum() == 0:
                continue
            Zg = Z[mask]
            eg = residuals[mask]
            self.cluster_instrument_resid[gid] += Zg.T @ eg
        # --- END NEW ---
        
        # Proceed with classic second-stage recursive update
        super().partial_fit(X_second_stage, y, cluster1, cluster2)
        
        return self
    
    def fit_two_stage(self):
        """
        Perform true two-stage least squares by:
        1. Finalizing first stage estimation
        2. Processing all stored data with the finalized first stage models
        """
        if not self.stored_data:
            warnings.warn("No data stored for second stage. Did you run partial_fit with store_data=True?")
            return self
        
        # Finalize first stage if not already done
        if not self.first_stage_finalized:
            self.finalize_first_stage()
        
        # Reset second stage parameters
        self.theta = np.zeros(self.n_features)
        self.XtX = self.alpha * np.eye(self.n_features)
        self.Xty = np.zeros(self.n_features)
        self.n_obs = 0
        self.rss = 0.0
        
        # Reset cluster statistics
        self.cluster_stats = defaultdict(lambda: {
            'X_sum': np.zeros(self.n_features),
            'residual_sum': 0.0,
            'count': 0,
            'XtX': np.zeros((self.n_features, self.n_features)),
            'X_residual_sum': np.zeros(self.n_features),
            'Xy': np.zeros(self.n_features)
        })
        self.cluster2_stats = defaultdict(lambda: {
            'X_sum': np.zeros(self.n_features),
            'residual_sum': 0.0,
            'count': 0,
            'XtX': np.zeros((self.n_features, self.n_features)),
            'X_residual_sum': np.zeros(self.n_features),
            'Xy': np.zeros(self.n_features)
        })
        self.intersection_stats = defaultdict(lambda: {
            'X_sum': np.zeros(self.n_features),
            'residual_sum': 0.0,
            'count': 0,
            'XtX': np.zeros((self.n_features, self.n_features)),
            'X_residual_sum': np.zeros(self.n_features),
            'Xy': np.zeros(self.n_features)
        })
        
        # Process all stored data with second stage fit
        logger.info(f"Starting second stage with {len(self.stored_data)} batches")
        for batch_data in self.stored_data:
            X_exog, X_endog, instruments, y, cluster1, cluster2 = batch_data
            self.second_stage_fit(X_exog, X_endog, instruments, y, cluster1, cluster2)
            
        logger.info(f"Second stage completed with {self.n_obs} observations")
        
        # Free up memory
        self.stored_data = []
        
        return self
        
    def get_2sls_covariance_matrix(self, homoskedastic: bool = True) -> np.ndarray:
        """
        Return 2SLS covariance matrix.
        homoskedastic=True implements classic formula Var(theta) = sigma^2 (X'P_Z X)^-1
        where P_Z = Z(Z'Z)^{-1}Z'. Uses predicted endogenous regressors (X_hat).
        If homoskedastic=False, dispatches to cluster-robust sandwich estimator.
        Assumes second stage has been run (ZtX_hat populated). Falls back gracefully if not.
        """
        if not homoskedastic:
            return self.get_2sls_cluster_robust_covariance()
        try:
            ZtZ_inv = np.linalg.inv(self.ZtZ)
            # If no predicted accumulation yet, fall back to actual (warning)
            if not np.any(self.ZtX_hat):
                warnings.warn("ZtX_hat empty; falling back to ZtX_actual for covariance.")
                ZtX_use = self.ZtX_actual
            else:
                ZtX_use = self.ZtX_hat
            XtPzX = ZtX_use.T @ ZtZ_inv @ ZtX_use
            sigma2 = self.rss / max(1, self.n_obs - self.n_features)
            return sigma2 * np.linalg.inv(XtPzX)
        except Exception as e:
            warnings.warn(f"2SLS homoskedastic covariance failed ({e}); using OLS robust fallback.")
            return super().get_covariance_matrix()

    def get_2sls_cluster_robust_covariance(self) -> np.ndarray:
        """
        Cluster-robust (one-way) 2SLS covariance:
        Var(theta) = A^{-1} * (X'Z W (Σ_g S_g S_g') W Z'X) * A^{-1}
        where:
          S_g = Z'_g e_g
          W = (Z'Z)^{-1} (efficient 2SLS weighting under homoskedasticity)
          A = X'Z W Z'X (all with X = [exog, predicted endog])
        Uses accumulated cluster_instrument_resid (S_g).
        """
        try:
            if not np.any(self.ZtX_hat):
                # Attempt to build ZtX_hat from actual if only first stage done (fallback)
                warnings.warn("ZtX_hat not populated; cannot compute cluster-robust 2SLS covariance. Returning OLS robust.")
                return super().get_cluster_robust_covariance()
            ZtZ_inv = np.linalg.inv(self.ZtZ)
            A = self.ZtX_hat.T @ ZtZ_inv @ self.ZtX_hat
            A_inv = np.linalg.inv(A)
            # Σ_g S_g S_g'
            S_sum = np.zeros((self.n_all_instruments, self.n_all_instruments))
            for vec in self.cluster_instrument_resid.values():
                if np.any(vec):
                    S_sum += np.outer(vec, vec)
            # Meat inner: W S_sum W
            Meat_inner = ZtZ_inv @ S_sum @ ZtZ_inv
            # Full middle: (Z'X)' Meat_inner (Z'X) = ZtX_hat.T @ Meat_inner @ ZtX_hat
            Middle = self.ZtX_hat.T @ Meat_inner @ self.ZtX_hat
            return A_inv @ Middle @ A_inv
        except Exception as e:
            warnings.warn(f"Cluster-robust 2SLS covariance failed ({e}); using OLS cluster-robust fallback.")
            return super().get_cluster_robust_covariance()

    def sargan_test(self) -> Dict[str, float]:
        """
        Classical (homoskedastic) overidentification test (Sargan).
        Uses moment conditions based on predicted endogenous regressors:
          Statistic: J = (Z'e)' (Z'Z)^{-1} (Z'e) / sigma^2
        where Z'e = Z'y - Z'X_hat theta (recomputed if needed),
              sigma^2 = RSS/(n - k).
        Valid only under homoskedasticity; for heteroskedastic/cluster-robust use hansen_j_test().
        """
        if self.n_instruments <= self.n_endogenous:
            return {"test_statistic": np.nan, "p_value": np.nan, "degrees_freedom": 0,
                    "overidentified": False, "note": "Exactly identified or underidentified."}
        try:
            if not np.any(self.ZtX_hat):
                warnings.warn("Second-stage predicted cross-products missing (ZtX_hat). Sargan test unavailable.")
                return {"test_statistic": np.nan, "p_value": np.nan, "degrees_freedom": self.n_instruments - self.n_endogenous}
            ZtZ_inv = np.linalg.inv(self.ZtZ)
            # Recompute Z'e from stored components to reduce sequential bias
            Zte_vec = self.Zty - self.ZtX_hat @ self.theta
            sigma2 = self.rss / max(1, (self.n_obs - self.n_features))
            stat = (Zte_vec.T @ ZtZ_inv @ Zte_vec) / sigma2
            from scipy import stats
            df = self.n_instruments - self.n_endogenous
            p_value = 1 - stats.chi2.cdf(stat, df)
            return {
                "test_statistic": float(stat),
                "p_value": float(p_value),
                "degrees_freedom": df,
                "overidentified": True,
                "robust": False
            }
        except Exception as e:
            warnings.warn(f"Sargan test failed: {e}")
            return {"test_statistic": np.nan, "p_value": np.nan, "error": str(e)}

    def hansen_j_test(self) -> Dict[str, float]:
        """
        Hansen (robust) J overidentification test using cluster-robust covariance of moment conditions.
        Let g_bar = (1/n) Z'e, and S = (1/n) Σ_g (Z'_g e_g)(Z'_g e_g)' (cluster robust).
        J = n * g_bar' S^{-1} g_bar ~ χ²_{L - K}, L = total instruments, K = endogenous vars.
        More reliable under heteroskedasticity / clustering than Sargan.
        """
        if self.n_instruments <= self.n_endogenous:
            return {"test_statistic": np.nan, "p_value": np.nan, "degrees_freedom": 0,
                    "overidentified": False, "note": "Exactly identified or underidentified."}
        try:
            if self.n_obs == 0 or not self.cluster_instrument_resid:
                return {"test_statistic": np.nan, "p_value": np.nan, "degrees_freedom": self.n_instruments - self.n_endogenous,
                        "error": "No second-stage cluster residual moments accumulated."}
            # Aggregate S_g
            S_sum = np.zeros((self.n_all_instruments, self.n_all_instruments))
            Se = np.zeros(self.n_all_instruments)
            for vec in self.cluster_instrument_resid.values():
                if np.any(vec):
                    Se += vec
                    S_sum += np.outer(vec, vec)
            n = self.n_obs
            g_bar = Se / n
            S = S_sum / n  # matches cluster-robust moment covariance scaling
            # Regularize if near singular
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                ridge = 1e-8 * np.eye(S.shape[0])
                S_inv = np.linalg.inv(S + ridge)
            J = n * (g_bar.T @ S_inv @ g_bar)
            from scipy import stats
            df = self.n_instruments - self.n_endogenous
            p_value = 1 - stats.chi2.cdf(J, df)
            return {
                "test_statistic": float(J),
                "p_value": float(p_value),
                "degrees_freedom": df,
                "overidentified": True,
                "robust": True
            }
        except Exception as e:
            warnings.warn(f"Hansen J test failed: {e}")
            return {"test_statistic": np.nan, "p_value": np.nan, "error": str(e)}

    def get_first_stage_statistics(self) -> Dict[str, Any]:
        """
        Return first stage diagnostics:
          - Wald F for excluded instruments
          - Partial R^2 derived from F: R2_partial = (F * q)/(F * q + df2)
          - df1 = number of excluded instruments (q)
          - df2 = n_obs - k (k = total included instruments incl. exogenous)
        Keys preserved for backward compatibility; new keys appended.
        """
        first_stage_stats: Dict[str, Any] = {}
        q = self.n_instruments
        for i, fs_model in enumerate(self.first_stage_models):
            fs_cov = fs_model.get_cluster_robust_covariance() if hasattr(fs_model, 'cluster_stats') else fs_model.get_covariance_matrix()
            excluded_coefs = fs_model.theta[self.n_exogenous:]
            excluded_cov = fs_cov[self.n_exogenous:, self.n_exogenous:]
            df2 = max(1, fs_model.n_obs - fs_model.n_features)
            try:
                inv_excl = np.linalg.inv(excluded_cov)
                wald = excluded_coefs.T @ inv_excl @ excluded_coefs
                f_stat = (wald / q)
                partial_R2 = (f_stat * q) / (f_stat * q + df2) if (f_stat > 0) else 0.0
                # thresholds (approx Stock-Yogo simplified)
                if self.n_instruments <= 5:
                    critical_value = 10.0
                elif self.n_instruments <= 10:
                    critical_value = 11.0
                else:
                    critical_value = 12.0
                prefix = f'endogenous_{i}'
                first_stage_stats[f'{prefix}_f_stat'] = float(f_stat)
                first_stage_stats[f'{prefix}_wald_stat'] = float(wald)
                first_stage_stats[f'{prefix}_partial_R2'] = float(partial_R2)
                first_stage_stats[f'{prefix}_df1'] = q
                first_stage_stats[f'{prefix}_df2'] = df2
                first_stage_stats[f'{prefix}_critical_value'] = critical_value
                first_stage_stats[f'{prefix}_weak_instruments'] = f_stat < critical_value
            except Exception:
                prefix = f'endogenous_{i}'
                first_stage_stats[f'{prefix}_f_stat'] = np.nan
                first_stage_stats[f'{prefix}_wald_stat'] = np.nan
                first_stage_stats[f'{prefix}_partial_R2'] = np.nan
                first_stage_stats[f'{prefix}_df1'] = q
                first_stage_stats[f'{prefix}_df2'] = np.nan
                first_stage_stats[f'{prefix}_critical_value'] = np.nan
                first_stage_stats[f'{prefix}_weak_instruments'] = True
        return first_stage_stats

    def get_comprehensive_first_stage_statistics(self) -> Dict[str, Any]:
        """
        Return comprehensive first stage diagnostics including:
          - Individual regression statistics for each endogenous variable
          - Joint weak instrument tests
          - Instrument relevance measures
          - Model fit statistics
        """
        comprehensive_stats = {}
        
        # Individual first-stage regressions
        for i, fs_model in enumerate(self.first_stage_models):
            var_name = f'endogenous_{i}'
            
            # Basic statistics
            fs_cov = fs_model.get_cluster_robust_covariance() if hasattr(fs_model, 'cluster_stats') else fs_model.get_covariance_matrix()
            excluded_coefs = fs_model.theta[self.n_exogenous:]
            excluded_cov = fs_cov[self.n_exogenous:, self.n_exogenous:]
            df2 = max(1, fs_model.n_obs - fs_model.n_features)
            
            comprehensive_stats[var_name] = {
                'coefficients': {
                    'all': fs_model.theta.tolist(),
                    'exogenous': fs_model.theta[:self.n_exogenous].tolist(),
                    'instruments': excluded_coefs.tolist()
                },
                'standard_errors': {
                    'all': np.sqrt(np.diag(fs_cov)).tolist(),
                    'exogenous': np.sqrt(np.diag(fs_cov[:self.n_exogenous, :self.n_exogenous])).tolist(),
                    'instruments': np.sqrt(np.diag(excluded_cov)).tolist()
                },
                'n_observations': int(fs_model.n_obs),
                'degrees_of_freedom': df2,
                'r_squared': float(fs_model.get_r_squared()),
                'adjusted_r_squared': float(fs_model.get_adjusted_r_squared()),
                'residual_sum_squares': float(fs_model.rss)
            }
            
            # Instrument relevance tests
            try:
                inv_excl = np.linalg.inv(excluded_cov)
                wald = excluded_coefs.T @ inv_excl @ excluded_coefs
                f_stat = (wald / self.n_instruments)
                partial_R2 = (f_stat * self.n_instruments) / (f_stat * self.n_instruments + df2) if (f_stat > 0) else 0.0
                
                # Stock-Yogo critical values (simplified)
                if self.n_instruments <= 5:
                    critical_value = 10.0
                elif self.n_instruments <= 10:
                    critical_value = 11.0
                else:
                    critical_value = 12.0
                
                comprehensive_stats[var_name]['instrument_tests'] = {
                    'f_statistic': float(f_stat),
                    'wald_statistic': float(wald),
                    'partial_r_squared': float(partial_R2),
                    'critical_value_10pct': critical_value,
                    'weak_instruments': bool(f_stat < critical_value),
                    'df_numerator': self.n_instruments,
                    'df_denominator': df2
                }
                
                # Individual instrument t-statistics
                t_stats = excluded_coefs / np.sqrt(np.diag(excluded_cov))
                comprehensive_stats[var_name]['individual_instrument_tests'] = {
                    f'instrument_{j}': {
                        'coefficient': float(excluded_coefs[j]),
                        'std_error': float(np.sqrt(excluded_cov[j, j])),
                        't_statistic': float(t_stats[j]),
                        'p_value': float(2 * (1 - stats.t.cdf(abs(t_stats[j]), df2))),
                        'significant_5pct': bool(abs(t_stats[j]) > stats.t.ppf(0.975, df2))
                    }
                    for j in range(self.n_instruments)
                }
                
            except Exception as e:
                comprehensive_stats[var_name]['instrument_tests'] = {
                    'error': str(e),
                    'weak_instruments': True
                }
                comprehensive_stats[var_name]['individual_instrument_tests'] = {}
        
        # Joint/summary statistics
        all_f_stats = [comprehensive_stats[f'endogenous_{i}']['instrument_tests'].get('f_statistic', 0) 
                      for i in range(self.n_endogenous) 
                      if 'instrument_tests' in comprehensive_stats[f'endogenous_{i}']]
        
        comprehensive_stats['summary'] = {
            'min_f_statistic': float(min(all_f_stats)) if all_f_stats else np.nan,
            'mean_f_statistic': float(np.mean(all_f_stats)) if all_f_stats else np.nan,
            'max_f_statistic': float(max(all_f_stats)) if all_f_stats else np.nan,
            'any_weak_instruments': any(comprehensive_stats[f'endogenous_{i}']['instrument_tests'].get('weak_instruments', True) 
                                       for i in range(self.n_endogenous)),
            'n_endogenous_variables': self.n_endogenous,
            'n_instruments': self.n_instruments,
            'n_exogenous': self.n_exogenous,
            'identification_status': 'over_identified' if self.n_instruments > self.n_endogenous else 'just_identified'
        }
        
        return comprehensive_stats

    def summarize_diagnostics(self) -> Dict[str, Any]:
        """
        Summarize key econometric diagnostics:
          - First-stage F statistics (weak instrument check)
          - Sargan (homoskedastic) over-ID test
          - Hansen J (cluster-robust) over-ID test
          - Condition numbers & ranks (Z'Z and endogenous block of Z'X_hat)
          - Observation count
        """
        diags: Dict[str, Any] = {}
        fs = self.get_first_stage_statistics()
        diags["first_stage_F_stats"] = {k: v for k, v in fs.items() if k.endswith("_f_stat")}
        diags["weak_instruments_flags"] = {k: v for k, v in fs.items() if k.endswith("_weak_instruments")}
        try:
            cond_ZtZ = float(np.linalg.cond(self.ZtZ))
            rank_ZtZ = int(np.linalg.matrix_rank(self.ZtZ))
        except Exception:
            cond_ZtZ, rank_ZtZ = np.nan, np.nan
        if np.any(self.ZtX_hat):
            try:
                endog_block = self.ZtX_hat[:, -self.n_endogenous:]
                cond_ZtX_hat_endog = float(np.linalg.cond(endog_block))
                rank_ZtX_hat_endog = int(np.linalg.matrix_rank(endog_block))
            except Exception:
                cond_ZtX_hat_endog = np.nan
                rank_ZtX_hat_endog = np.nan
        else:
            cond_ZtX_hat_endog = np.nan
            rank_ZtX_hat_endog = np.nan
        diags["cond_ZtZ"] = cond_ZtZ
        diags["rank_ZtZ"] = rank_ZtZ
        diags["cond_ZtX_hat_endog"] = cond_ZtX_hat_endog
        diags["rank_ZtX_hat_endog"] = rank_ZtX_hat_endog
        diags["n_obs"] = int(self.n_obs)
        diags["sargan"] = self.sargan_test()
        diags["hansen_j"] = self.hansen_j_test()
        return diags

    # TEST HOOKS (unit-test friendly)
    def get_internal_state(self) -> Dict[str, Any]:
        """
        Lightweight snapshot for unit tests / smoke tests.
        TODO: add more invariants (e.g., symmetry checks) in future tests.
        """
        return {
            'ZtZ_shape': self.ZtZ.shape,
            'ZtX_actual_shape': self.ZtX_actual.shape,
            'ZtX_hat_populated': bool(np.any(self.ZtX_hat)),
            'Zte_norm': float(np.linalg.norm(self.Zte)),
            'n_obs': self.n_obs,
            'second_stage_initialized': self.second_stage_initialized
        }

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
    """Process partitioned dataset with two-stage least squares estimation."""
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
    
    # Calculate dimensions CORRECTLY
    # Second stage: regress y on [intercept?, exogenous, endogenous]
    n_features_second_stage = n_exogenous + n_endogenous
    if add_intercept:
        n_features_second_stage += 1
    
    # First stage & instruments: [intercept?, exogenous, instruments]
    n_features_instruments = n_exogenous + n_instruments
    if add_intercept:
        n_features_instruments += 1
    
    # For dimension calculations in Online2SLS constructor, we need the counts BEFORE adding intercept
    n_exogenous_for_constructor = n_exogenous
    if add_intercept:
        n_exogenous_for_constructor += 1  # This will be used in constructor
    
    # Create feature names for display (second stage)
    display_feature_names = []
    if add_intercept:
        display_feature_names.append("intercept")
    display_feature_names.extend(exogenous_cols)
    display_feature_names.extend(endogenous_cols)
    
    logger.info(f"Exogenous vars: {exogenous_cols}")
    logger.info(f"Endogenous vars: {endogenous_cols}")
    logger.info(f"Instruments: {instrument_cols}")
    logger.info(f"Target: {target_col}")
    logger.info(f"Add intercept: {add_intercept}")
    logger.info(f"Second-stage features: {n_features_second_stage}")
    logger.info(f"Total instruments: {n_features_instruments}")
    logger.info(f"Chunk size: {chunk_size:,}")
    
    # Initialize progress tracking
    total_partitions = len(partition_files)
    completed_partitions = 0
    
    # Create main progress bar
    main_pbar = None
    if show_progress:
        main_pbar = tqdm(
            total=total_partitions, 
            desc="Processing 2SLS first stage", 
            unit="partitions",
            disable=not verbose
        )
    
    # Initialize main 2SLS instance with constructor counts
    main_2sls = Online2SLS(
        n_exogenous=n_exogenous_for_constructor, 
        n_endogenous=n_endogenous,
        n_instruments=n_instruments, 
        alpha=alpha, 
        forget_factor=forget_factor
    )
    
    # Process first stage in parallel
    logger.info(f"Processing {len(partition_files)} partitions for first stage in parallel...")
    
    # Modified worker args to indicate first stage pass
    worker_args = [
        (partition_file, exogenous_cols, endogenous_cols, instrument_cols, target_col,
         cluster1_col, cluster2_col, add_intercept, n_exogenous, n_endogenous, 
         n_instruments, alpha, forget_factor, chunk_size, verbose, True)  # True for first_stage_only
        for partition_file in partition_files
    ]
    
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
            
            for future in as_completed(future_to_partition, timeout=7200):
                partition_file = future_to_partition[future]
                try:
                    result = future.result(timeout=900)
                    # UPDATED: support 10 or 11 element first-stage tuple
                    if len(result) == 11:
                        (ZtZ_update, ZtX_update, Zty_update, rss_update, n_obs,
                         cluster_stats, cluster2_stats, intersection_stats,
                         first_stage_XtX, first_stage_Xty,
                         first_stage_cluster_stats_part) = result
                    else:
                        (ZtZ_update, ZtX_update, Zty_update, rss_update, n_obs,
                         cluster_stats, cluster2_stats, intersection_stats,
                         first_stage_XtX, first_stage_Xty) = result
                        first_stage_cluster_stats_part = {}
                    if n_obs > 0:
                        # Merge 2SLS first stage results
                        temp_2sls = Online2SLS(n_exogenous=n_exogenous_for_constructor, n_endogenous=n_endogenous,
                                             n_instruments=n_instruments, alpha=alpha)
                        
                        # Update sufficient statistics
                        temp_2sls.ZtZ = ZtZ_update
                        temp_2sls.ZtX_actual = ZtX_update
                        temp_2sls.Zty = Zty_update
                        temp_2sls.n_obs = n_obs
                        temp_2sls.rss = rss_update
                        
                        # Update first stage models
                        for i, fs_model in enumerate(temp_2sls.first_stage_models):
                            fs_model.XtX = first_stage_XtX[i]
                            fs_model.Xty = first_stage_Xty[i]  # Fixed: was first_stage_XtY
                            fs_model.n_obs = n_obs
                            try:
                                fs_model.theta = np.linalg.solve(fs_model.XtX, fs_model.Xty)
                                fs_model.P = np.linalg.inv(fs_model.XtX)
                            except np.linalg.LinAlgError:
                                regularized_XtX = fs_model.XtX + alpha * np.eye(fs_model.n_features)
                                fs_model.theta = np.linalg.solve(regularized_XtX, fs_model.Xty)
                                fs_model.P = np.linalg.inv(regularized_XtX)
                        
                        # Set cluster statistics - use correct feature count
                        temp_2sls.cluster_stats = cluster_stats
                        temp_2sls.cluster2_stats = cluster2_stats
                        temp_2sls.intersection_stats = intersection_stats
                        # Merge
                        main_2sls.ZtZ += temp_2sls.ZtZ - alpha * np.eye(n_features_instruments)
                        main_2sls.ZtX_actual += temp_2sls.ZtX_actual
                        main_2sls.Zty += temp_2sls.Zty
                        main_2sls.n_obs += temp_2sls.n_obs
                        main_2sls.rss += temp_2sls.rss
                        
                        # Merge first stage models
                        for i, (main_fs, temp_fs) in enumerate(zip(main_2sls.first_stage_models, temp_2sls.first_stage_models)):
                            main_fs.merge_statistics(temp_fs)
                        
                        # MERGE first-stage cluster instrument stats (already unpacked)
                        for cid, stats in first_stage_cluster_stats_part.items():
                            fs_target = main_2sls.first_stage_cluster_stats[cid]
                            fs_target['Z_sum'] += stats['Z_sum']
                            fs_target['count'] += stats['count']
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
                            'total_obs': f"{main_2sls.n_obs:,}"
                        })
                    
                except Exception as e:
                    failed_partitions.append(str(partition_file))
                    completed_partitions += 1
                    logger.error(f"Failed to process 2SLS partition {partition_file}: {str(e)}")
        
        # Close first stage progress bar
        if main_pbar:
            main_pbar.close()
            
        # Finalize first stage estimation
        logger.info("Finalizing first stage estimation...")
        main_2sls.finalize_first_stage()

        # Print first-stage regression outputs
        print("\nFirst-stage regression outputs:")
        for i, fs_model in enumerate(main_2sls.first_stage_models):
            coef_str = ", ".join([f"{c:.4f}" for c in fs_model.theta])
            try:
                y_hat = None
                if hasattr(fs_model, "XtX") and hasattr(fs_model, "Xty"):
                    # Estimate R² if possible
                    # Note: This is only approximate for online/partitioned data
                    # If you want exact, you need to accumulate y and y_hat
                    # Here, we just print coefficients
                    pass
                print(f"  Endogenous {i}: coefficients = [{coef_str}]")
            except Exception:
                print(f"  Endogenous {i}: coefficients = [{coef_str}]")
        print("")  # Blank line for separation

        # Now process second stage in parallel
        logger.info(f"Processing second stage using finalized first stage models...")
        
        # New progress bar for second stage
        if show_progress:
            main_pbar = tqdm(
                total=total_partitions, 
                desc="Processing 2SLS second stage", 
                unit="partitions",
                disable=not verbose
            )
        
        # Reset counters for second stage
        completed_partitions = 0
        successful_partitions = 0
        empty_partitions = 0
        failed_partitions = []
        
        # Modified worker args to indicate second stage pass
        worker_args = [
            (partition_file, exogenous_cols, endogenous_cols, instrument_cols, target_col,
             cluster1_col, cluster2_col, add_intercept, n_exogenous, n_endogenous, 
             n_instruments, alpha, forget_factor, chunk_size, verbose, False,  # False for not first_stage_only
             [model.theta for model in main_2sls.first_stage_models])  # Pass first stage coefficients
            for partition_file in partition_files
        ]
        
        # Process second stage
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_partition = {
                executor.submit(process_partition_worker_2sls, args): args[0] 
                for args in worker_args
            }
            
            for future in as_completed(future_to_partition, timeout=7200):  # 2 hour global timeout
                partition_file = future_to_partition[future]
                
                try:
                    result = future.result(timeout=900)  # 15 minute timeout per partition
                    
                    # For second stage, we expect different return values
                    result_len = len(result)
                    # Expect extended tuple with new IV accumulators
                    if result_len == 10:
                        (XtX_update, Xty_update, rss_update, n_obs,
                         cluster_stats, cluster2_stats, intersection_stats,
                         ZtX_hat_update, Zte_update, cluster_z_e_dict) = result
                    else:
                        # Backward compatibility (older worker signature)
                        (XtX_update, Xty_update, rss_update, n_obs,
                         cluster_stats, cluster2_stats, intersection_stats) = result
                        ZtX_hat_update = None
                        Zte_update = None
                        cluster_z_e_dict = {}
                    if n_obs > 0:
                        # Replace hasattr(main_2sls, 'second_stage') with flag already set elsewhere
                        if not main_2sls.second_stage_initialized:
                            # First result, just copy
                            main_2sls.XtX = XtX_update
                            main_2sls.Xty = Xty_update
                            main_2sls.n_obs = n_obs
                            main_2sls.rss = rss_update
                            main_2sls.cluster_stats = cluster_stats
                            main_2sls.cluster2_stats = cluster2_stats
                            main_2sls.intersection_stats = intersection_stats
                            main_2sls.second_stage_initialized = True
                        else:
                            main_2sls.XtX += XtX_update - alpha * np.eye(n_features_second_stage)
                            main_2sls.Xty += Xty_update
                            main_2sls.n_obs += n_obs
                            main_2sls.rss += rss_update
                            
                            # Merge cluster stats
                            main_2sls._merge_cluster_stats(cluster_stats, main_2sls.cluster_stats)
                            main_2sls._merge_cluster_stats(cluster2_stats, main_2sls.cluster2_stats)
                            main_2sls._merge_cluster_stats(intersection_stats, main_2sls.intersection_stats)
                        
                        # NEW: accumulate second-stage IV projections
                        if ZtX_hat_update is not None:
                            main_2sls.ZtX_hat += ZtX_hat_update
                        if Zte_update is not None:
                            main_2sls.Zte += Zte_update
                        for cid, vec in cluster_z_e_dict.items():
                            main_2sls.cluster_instrument_resid[cid] += vec
                        
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
                
                except Exception as e:
                    failed_partitions.append(str(partition_file))
                    completed_partitions += 1
                    logger.error(f"Failed to process second stage for partition {partition_file}: {str(e)}")
        
        # Close second stage progress bar
        if main_pbar:
            main_pbar.close()
    
    finally:
        if main_pbar and main_pbar.n < main_pbar.total:
            main_pbar.close()
    
    # Finalize 2SLS estimation
    if main_2sls.n_obs > 0:
        try:
            # Compute theta using XtX and Xty from second stage
            main_2sls.theta = np.linalg.solve(main_2sls.XtX, main_2sls.Xty)
            main_2sls.P = np.linalg.inv(main_2sls.XtX)
        except np.linalg.LinAlgError:
            logger.warning("Numerical issues in second stage, using regularized solution")
            main_2sls.rank_deficient = True
            XtX_reg = main_2sls.XtX + alpha * 10 * np.eye(n_features_second_stage)
            main_2sls.theta = np.linalg.solve(XtX_reg, main_2sls.Xty)
            main_2sls.P = np.linalg.inv(XtX_reg)
    
    # Enhanced reporting
    logger.info(f"Two-stage least squares completed in {time.time() - start_time:.2f} seconds")
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
    
    diagnostics = main_2sls.summarize_diagnostics()
    logger.info(f"2SLS diagnostics summary: {diagnostics}")
    
    return main_2sls

def process_partition_worker_2sls(args: Tuple):
    """Worker function for 2SLS estimation across partitions."""
    first_stage_only = args[15] if len(args) > 15 else True
    first_stage_theta = args[16] if len(args) > 16 else None
    
    (partition_file, exogenous_cols, endogenous_cols, instrument_cols, target_col, 
     cluster1_col, cluster2_col, add_intercept, n_exogenous, n_endogenous, 
     n_instruments, alpha, forget_factor, chunk_size, verbose) = args[:15]
    
    worker_id = mp.current_process().pid
    stage_name = "first stage" if first_stage_only else "second stage"
    worker_logger = logging.getLogger(f"worker_2sls_{worker_id}")
    worker_logger.info(f"Processing 2SLS {stage_name} for partition: {partition_file}")
    
    try:
        # Calculate correct dimensions for this worker
        n_exogenous_worker = n_exogenous
        if add_intercept:
            n_exogenous_worker += 1
        
        n_features_second_stage = n_exogenous_worker + n_endogenous
        n_all_instruments = n_exogenous_worker + n_instruments
            
        # Initialize models based on stage
        if first_stage_only:
            # For first stage, initialize 2SLS model
            local_model = Online2SLS(
                n_exogenous=n_exogenous_worker, 
                n_endogenous=n_endogenous,
                n_instruments=n_instruments, 
                alpha=alpha,
                forget_factor=forget_factor, 
                batch_size=min(chunk_size, 5000)
            )
            # Initialize first-stage cluster stats dictionary
            first_stage_cluster_stats = defaultdict(lambda: {
                'Z_sum': np.zeros(n_all_instruments),
                'count': 0
            })
        else:
            # For second stage, we'll use an OnlineRLS directly
            local_model = OnlineRLS(
                n_features=n_features_second_stage,
                alpha=alpha,
                forget_factor=forget_factor,
                batch_size=min(chunk_size, 5000)
            )
            
            # Create first stage models with fixed coefficients
            first_stage_models = []
            for i in range(n_endogenous):
                fs_model = OnlineRLS(n_features=n_all_instruments, alpha=alpha)
                fs_model.theta = first_stage_theta[i]
                first_stage_models.append(fs_model)
            
            # Initialize second-stage accumulator variables for IV diagnostics
            ZtX_hat_update = np.zeros((n_all_instruments, n_features_second_stage))
            Zte_update = np.zeros(n_all_instruments)
            cluster_z_e_dict = defaultdict(lambda: np.zeros(n_all_instruments))
        
        # Use PyArrow to read partition
        import pyarrow.parquet as pq
        
        parquet_file = pq.ParquetFile(partition_file)
        total_rows = parquet_file.metadata.num_rows
        worker_logger.info(f"Partition has {total_rows:,} rows for {stage_name}")
        
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
            # Return empty results with correct dimensions
            if first_stage_only:
                n_all_instruments = n_exogenous_worker + n_instruments
                n_second_stage_features = n_exogenous_worker + n_endogenous
                return (
                    alpha * np.eye(n_all_instruments),  # ZtZ
                    np.zeros((n_all_instruments, n_second_stage_features)),  # ZtX
                    np.zeros(n_all_instruments),  # Zty
                    0.0, 0, {}, {}, {},  # rss, n_obs, cluster stats
                    [alpha * np.eye(n_all_instruments) for _ in range(n_endogenous)],  # first stage XtX
                    [np.zeros(n_all_instruments) for _ in range(n_endogenous)]  # first stage Xty
                )
            else:
                # Second stage return values
                return (
                    alpha * np.eye(n_features_second_stage),  # XtX
                    np.zeros(n_features_second_stage),  # Xty
                    0.0, 0, {}, {}, {}  # rss, n_obs, cluster stats
                )
        
        # Check data quality in first batch
        y_sample = first_df[target_col].values
        valid_y_ratio = np.isfinite(y_sample).mean()
        
        if valid_y_ratio < 0.01:  # Less than 1% valid data
            worker_logger.warning(f"Partition has very low data quality ({valid_y_ratio*100:.1f}% valid). Skipping.")
            # Return empty results with correct dimensions
            if first_stage_only:
                n_all_instruments = n_exogenous_worker + n_instruments
                n_second_stage_features = n_exogenous_worker + n_endogenous
                return (
                    alpha * np.eye(n_all_instruments),
                    np.zeros((n_all_instruments, n_second_stage_features)),
                    np.zeros(n_all_instruments),
                    0.0, 0, {}, {}, {},
                    [alpha * np.eye(n_all_instruments) for _ in range(n_endogenous)],
                    [np.zeros(n_all_instruments) for _ in range(n_endogenous)]
                )
            else:
                # Second stage return values
                return (
                    alpha * np.eye(n_features_second_stage),  # XtX
                    np.zeros(n_features_second_stage),  # Xty
                    0.0, 0, {}, {}, {}  # rss, n_obs, cluster stats
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
                
                if first_stage_only:
                    # First stage: just update model
                    local_model.partial_fit(X_exog, X_endog, instruments, y, cluster1, cluster2)
                    
                    # accumulate per-cluster instrument stats
                    if cluster1 is not None and len(cluster1) == len(y):
                        cluster_ids = cluster1
                    elif cluster2 is not None and len(cluster2) == len(y):
                        cluster_ids = cluster2
                    else:
                        cluster_ids = np.array(['__all__'] * len(y))
                    Z = np.column_stack([X_exog, instruments])
                    for cid in np.unique(cluster_ids):
                        m = (cluster_ids == cid)
                        fs_stats = first_stage_cluster_stats[cid]
                        fs_stats['Z_sum'] += Z[m].sum(axis=0)
                        fs_stats['count'] += m.sum()
                else:
                    # Second stage: predict endogenous vars with fixed first stage models
                    Z = np.column_stack([X_exog, instruments])
                    X_endog_pred = np.zeros_like(X_endog)
                    
                    for i, fs_model in enumerate(first_stage_models):
                        X_endog_pred[:, i] = fs_model.predict(Z)
                    
                    # Create second-stage features
                    X_second_stage = np.column_stack([X_exog, X_endog_pred])
                    
                    # Calculate residuals using current theta before update
                    current_theta = getattr(local_model, 'theta', np.zeros(n_features_second_stage))
                    residuals = y - X_second_stage @ current_theta
                    
                    # Accumulate instrument-feature and instrument-residual moments
                    ZtX_hat_update += Z.T @ X_second_stage
                    Zte_update += Z.T @ residuals
                    
                    # Accumulate per-cluster instrument-residual moments
                    if cluster1 is not None and len(cluster1) == len(residuals):
                        cluster_ids = cluster1
                    elif cluster2 is not None and len(cluster2) == len(residuals):
                        cluster_ids = cluster2
                    else:
                        cluster_ids = np.array(['__all__'] * len(residuals))
                    for cid in np.unique(cluster_ids):
                        mask = (cluster_ids == cid)
                        if mask.sum() == 0:
                            continue
                        Zg = Z[mask]
                        eg = residuals[mask]
                        cluster_z_e_dict[cid] += Zg.T @ eg
                    
                    # Update second stage
                    local_model.partial_fit(X_second_stage, y, cluster1, cluster2)
                
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
            worker_logger.info(f"Processed {valid_chunks}/{chunks_processed} valid chunks for {stage_name}")
        
        worker_logger.info(f"Completed 2SLS {stage_name} for partition: {partition_file.name}")
        
        # Return results based on stage
        if first_stage_only:
            # Return first stage statistics
            first_stage_XtX = [fs_model.XtX for fs_model in local_model.first_stage_models]
            first_stage_Xty = [fs_model.Xty for fs_model in local_model.first_stage_models]
            
            return (
                local_model.ZtZ, local_model.ZtX_actual, local_model.Zty,
                local_model.rss, local_model.n_obs,
                dict(local_model.cluster_stats), dict(local_model.cluster2_stats),
                dict(local_model.intersection_stats),
                first_stage_XtX, first_stage_Xty,
                dict(first_stage_cluster_stats)
            )
        else:
            return (
                local_model.XtX, local_model.Xty, local_model.rss, local_model.n_obs,
                dict(local_model.cluster_stats), dict(local_model.cluster2_stats),
                dict(local_model.intersection_stats),
                ZtX_hat_update, Zte_update, dict(cluster_z_e_dict)
            )
    except Exception as e:
        worker_logger.error(f"Fatal error processing 2SLS {stage_name} for partition {partition_file}: {str(e)}")
        # Return empty results with correct dimensions
        n_exogenous_worker = n_exogenous + (1 if add_intercept else 0)
        if first_stage_only:
            n_all_instruments = n_exogenous_worker + n_instruments
            n_second_stage_features = n_exogenous_worker + n_endogenous
            return (
                alpha * np.eye(n_all_instruments),
                np.zeros((n_all_instruments, n_second_stage_features)),
                np.zeros(n_all_instruments),
                0.0, 0, {}, {}, {},
                [alpha * np.eye(n_all_instruments) for _ in range(n_endogenous)],
                [np.zeros(n_all_instruments) for _ in range(n_endogenous)],
                {}
            )
        else:
            n_features_second_stage = n_exogenous_worker + n_endogenous
            n_all_instruments = n_exogenous_worker + n_instruments
            return (
                alpha * np.eye(n_features_second_stage),
                np.zeros(n_features_second_stage),
                0.0, 0, {}, {}, {},
                np.zeros((n_all_instruments, n_features_second_stage)),
                np.zeros(n_all_instruments),
                {}
            )
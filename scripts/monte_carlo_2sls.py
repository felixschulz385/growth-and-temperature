#!/usr/bin/env python3
"""
Monte Carlo Testing Suite for Online2SLS Implementation

This script generates synthetic datasets with known parameters and instruments,
then tests whether the Online2SLS implementation can recover the true coefficients
accurately. Tests include various IV scenarios: weak/strong instruments,
different sample sizes, clustering structures, and endogeneity patterns.
"""

import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import time
import sys
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Add the parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from gnt.analysis.models.online_2SLS import Online2SLS, process_partitioned_dataset_2sls
import pyarrow as pa
import pyarrow.parquet as pq

# Set random seed for reproducibility
np.random.seed(42)

class MonteCarlo2SLSTester:
    """Monte Carlo testing framework for Online2SLS"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = []
        
    def log(self, message: str):
        """Print message if verbose mode is on"""
        if self.verbose:
            print(f"[MC 2SLS Test] {message}")
    
    def generate_synthetic_iv_data(self, 
                                  n_obs: int = 10000,
                                  n_exogenous: int = 2,
                                  n_endogenous: int = 1,
                                  n_instruments: int = 2,
                                  true_beta_exog: np.ndarray = None,
                                  true_beta_endog: np.ndarray = None,
                                  instrument_strength: float = 0.5,
                                  endogeneity_strength: float = 0.3,
                                  noise_std: float = 1.0,
                                  n_clusters1: int = 50,
                                  n_clusters2: int = 20,
                                  missing_rate: float = 0.0) -> pd.DataFrame:
        """
        Generate synthetic IV regression data with known DGP.
        
        Data Generating Process:
        1. Generate instruments Z and exogenous variables X_exog
        2. Generate endogenous variables: X_endog = Z*pi + X_exog*gamma + v (first stage)
        3. Generate outcome: y = X_exog*beta_exog + X_endog*beta_endog + u
        4. Correlation between u and v creates endogeneity
        
        Parameters:
        -----------
        n_obs : int
            Number of observations
        n_exogenous : int
            Number of exogenous variables (excluding intercept)
        n_endogenous : int
            Number of endogenous variables
        n_instruments : int
            Number of excluded instruments
        true_beta_exog : np.ndarray
            True coefficients for exogenous variables [intercept, exog_vars]
        true_beta_endog : np.ndarray
            True coefficients for endogenous variables
        instrument_strength : float
            Strength of instruments (affects first stage F-stat)
        endogeneity_strength : float
            Correlation between structural error and endogenous variable error
        """
        
        if true_beta_exog is None:
            # Default: intercept + exogenous coefficients
            true_beta_exog = np.array([2.0] + [1.0 * ((-1) ** i) for i in range(n_exogenous)])
        
        if true_beta_endog is None:
            # Default endogenous coefficients
            true_beta_endog = np.array([1.5 * ((-1) ** i) for i in range(n_endogenous)])
        
        # Generate cluster IDs
        cluster1_ids = np.random.randint(0, n_clusters1, n_obs)
        cluster2_ids = np.random.randint(0, n_clusters2, n_obs)
        
        # Add cluster-specific effects
        cluster1_effects = np.random.normal(0, 0.3, n_clusters1)
        cluster2_effects = np.random.normal(0, 0.2, n_clusters2)
        cluster_error = (cluster1_effects[cluster1_ids] + 
                        cluster2_effects[cluster2_ids])
        
        # 1. Generate exogenous variables and instruments
        X_exog = np.random.normal(0, 1, (n_obs, n_exogenous))
        Z_instruments = np.random.normal(0, 1, (n_obs, n_instruments))
        
        # Make instruments somewhat correlated for realism
        if n_instruments > 1:
            correlation_matrix = np.full((n_instruments, n_instruments), 0.3)
            np.fill_diagonal(correlation_matrix, 1.0)
            L = np.linalg.cholesky(correlation_matrix)
            Z_instruments = Z_instruments @ L.T
        
        # 2. Generate first stage errors and structural errors (correlated)
        # This correlation creates the endogeneity problem
        error_cov = np.array([[1.0, endogeneity_strength],
                             [endogeneity_strength, 1.0]])
        errors = np.random.multivariate_normal([0, 0], error_cov, n_obs)
        v_first_stage = errors[:, 0]  # First stage error
        u_structural = errors[:, 1]   # Structural error
        
        # Add cluster effects to both error terms
        v_first_stage += cluster_error * 0.5
        u_structural += cluster_error
        
        # 3. Generate endogenous variables (first stage)
        # X_endog = Z*pi + X_exog*gamma + v
        
        # First stage coefficients
        pi_instruments = np.random.normal(instrument_strength, 0.1, 
                                        (n_instruments, n_endogenous))
        gamma_exog = np.random.normal(0.2, 0.05, (n_exogenous, n_endogenous))
        
        X_endog = np.zeros((n_obs, n_endogenous))
        for j in range(n_endogenous):
            X_endog[:, j] = (Z_instruments @ pi_instruments[:, j] + 
                           X_exog @ gamma_exog[:, j] + 
                           v_first_stage * noise_std)
        
        # 4. Generate outcome variable (structural equation)
        # y = intercept + X_exog*beta_exog + X_endog*beta_endog + u
        intercept = true_beta_exog[0]
        y = (intercept + 
             X_exog @ true_beta_exog[1:] + 
             X_endog @ true_beta_endog + 
             u_structural * noise_std)
        
        # Create DataFrame
        exog_names = [f'exog_{i+1}' for i in range(n_exogenous)]
        endog_names = [f'endog_{i+1}' for i in range(n_endogenous)]
        instrument_names = [f'instrument_{i+1}' for i in range(n_instruments)]
        
        df = pd.DataFrame({
            'y': y,
            **{name: X_exog[:, i] for i, name in enumerate(exog_names)},
            **{name: X_endog[:, i] for i, name in enumerate(endog_names)},
            **{name: Z_instruments[:, i] for i, name in enumerate(instrument_names)},
            'cluster1': cluster1_ids,
            'cluster2': cluster2_ids,
            'v_first_stage': v_first_stage,  # For diagnostics
            'u_structural': u_structural     # For diagnostics
        })
        
        # Introduce missing values
        if missing_rate > 0:
            n_missing = int(n_obs * missing_rate)
            missing_indices = np.random.choice(n_obs, n_missing, replace=False)
            
            for idx in missing_indices:
                missing_var = np.random.choice(exog_names + endog_names + instrument_names + ['y'])
                df.loc[idx, missing_var] = np.nan
        
        # Store true parameters and DGP info
        df.attrs['true_beta_exog'] = true_beta_exog
        df.attrs['true_beta_endog'] = true_beta_endog
        df.attrs['true_beta_all'] = np.concatenate([true_beta_exog, true_beta_endog])
        df.attrs['pi_instruments'] = pi_instruments
        df.attrs['gamma_exog'] = gamma_exog
        df.attrs['instrument_strength'] = instrument_strength
        df.attrs['endogeneity_strength'] = endogeneity_strength
        df.attrs['noise_std'] = noise_std
        
        # Calculate theoretical first-stage F-statistics
        # This is approximate - true F-stat would need residual variance
        theoretical_f_stats = []
        for j in range(n_endogenous):
            pi_j = pi_instruments[:, j]
            f_stat_approx = np.sum(pi_j**2) / (noise_std**2 / n_obs)
            theoretical_f_stats.append(f_stat_approx)
        
        df.attrs['theoretical_f_stats'] = theoretical_f_stats
        
        return df
    
    def create_partitioned_dataset(self, df: pd.DataFrame, 
                                 temp_dir: Path,
                                 n_partitions: int = 5) -> Path:
        """Create a partitioned parquet dataset from DataFrame"""
        
        parquet_dir = temp_dir / "test_2sls_data.parquet"
        parquet_dir.mkdir(exist_ok=True)
        
        # Split data into partitions
        partition_size = len(df) // n_partitions
        
        for i in range(n_partitions):
            start_idx = i * partition_size
            if i == n_partitions - 1:
                end_idx = len(df)
            else:
                end_idx = (i + 1) * partition_size
            
            partition_df = df.iloc[start_idx:end_idx].copy()
            
            # Create partition directory structure
            partition_dir = parquet_dir / f"partition_{i}"
            partition_dir.mkdir(exist_ok=True)
            
            # Write partition
            partition_file = partition_dir / "data.parquet"
            table = pa.Table.from_pandas(partition_df)
            pq.write_table(table, partition_file)
        
        return parquet_dir
    
    def run_single_2sls_test(self, 
                           test_name: str,
                           n_obs: int = 10000,
                           n_exogenous: int = 2,
                           n_endogenous: int = 1,
                           n_instruments: int = 2,
                           chunk_size: int = 1000,
                           n_partitions: int = 5,
                           **kwargs) -> Dict:
        """Run a single 2SLS Monte Carlo test"""
        
        self.log(f"Running 2SLS test: {test_name}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Generate synthetic IV data
            start_time = time.time()
            df = self.generate_synthetic_iv_data(
                n_obs=n_obs,
                n_exogenous=n_exogenous,
                n_endogenous=n_endogenous,
                n_instruments=n_instruments,
                **kwargs
            )
            
            true_beta_all = df.attrs['true_beta_all']
            data_gen_time = time.time() - start_time
            
            # Create partitioned dataset
            parquet_path = self.create_partitioned_dataset(df, temp_path, n_partitions)
            
            # Prepare column names
            exog_cols = [f'exog_{i+1}' for i in range(n_exogenous)]
            endog_cols = [f'endog_{i+1}' for i in range(n_endogenous)]
            instrument_cols = [f'instrument_{i+1}' for i in range(n_instruments)]
            
            # Test Online2SLS
            start_time = time.time()
            
            try:
                twosls = process_partitioned_dataset_2sls(
                    parquet_path=parquet_path,
                    exogenous_cols=exog_cols,
                    endogenous_cols=endog_cols,
                    instrument_cols=instrument_cols,
                    target_col='y',
                    cluster1_col='cluster1',
                    cluster2_col='cluster2',
                    add_intercept=True,
                    chunk_size=chunk_size,
                    n_workers=2,
                    verbose=False
                )
                
                processing_time = time.time() - start_time
                success = True
                error_msg = None
                
                # Get estimates
                estimated_beta = twosls.theta
                
                # Calculate standard errors
                se_classical = twosls.get_standard_errors('classical')
                se_cluster1 = twosls.get_standard_errors('one_way')
                
                try:
                    se_cluster2 = twosls.get_standard_errors('two_way')
                except:
                    se_cluster2 = None
                
                # Get first stage diagnostics
                first_stage_stats = twosls.get_first_stage_statistics()
                
                # Check if 2SLS covariance matrix works
                try:
                    cov_2sls = twosls.get_2sls_covariance_matrix()
                    se_2sls = np.sqrt(np.diag(cov_2sls))
                    twosls_cov_success = True
                except:
                    se_2sls = None
                    twosls_cov_success = False
                
            except Exception as e:
                success = False
                error_msg = str(e)
                estimated_beta = np.full(len(true_beta_all), np.nan)
                se_classical = np.full(len(true_beta_all), np.nan)
                se_cluster1 = np.full(len(true_beta_all), np.nan)
                se_cluster2 = None
                se_2sls = None
                first_stage_stats = {}
                twosls_cov_success = False
                processing_time = np.nan
            
            # Benchmark against statsmodels if available and dataset small enough
            benchmark_beta = None
            benchmark_se = None
            benchmark_first_stage_f = None
            
            if n_obs <= 20000 and success:  # Only for smaller datasets
                try:
                    import statsmodels.api as sm
                    from statsmodels.sandbox.regression.gmm import IV2SLS
                    
                    # Prepare data (remove missing values)
                    df_clean = df.dropna()
                    
                    y_bench = df_clean['y'].values
                    X_exog_bench = df_clean[exog_cols].values
                    X_endog_bench = df_clean[endog_cols].values
                    Z_bench = df_clean[instrument_cols].values
                    
                    # Add intercept
                    X_exog_bench = np.column_stack([np.ones(len(X_exog_bench)), X_exog_bench])
                    Z_all_bench = np.column_stack([X_exog_bench, Z_bench])  # All instruments
                    X_all_bench = np.column_stack([X_exog_bench, X_endog_bench])  # All regressors
                    
                    # Fit 2SLS manually using the same approach as our implementation
                    ZtZ_inv = np.linalg.inv(Z_all_bench.T @ Z_all_bench)
                    ZtX = Z_all_bench.T @ X_all_bench
                    Zty = Z_all_bench.T @ y_bench
                    
                    XtPzX = ZtX.T @ ZtZ_inv @ ZtX
                    XtPzy = ZtX.T @ ZtZ_inv @ Zty
                    
                    benchmark_beta = np.linalg.solve(XtPzX, XtPzy)
                    
                    # Calculate standard errors
                    residuals = y_bench - X_all_bench @ benchmark_beta
                    sigma2 = np.sum(residuals**2) / (len(y_bench) - len(benchmark_beta))
                    var_beta = sigma2 * np.linalg.inv(XtPzX)
                    benchmark_se = np.sqrt(np.diag(var_beta))
                    
                    # First stage F-stats (simplified)
                    benchmark_first_stage_f = []
                    for j in range(n_endogenous):
                        y_fs = X_endog_bench[:, j]
                        X_fs = Z_all_bench
                        
                        # Full model
                        beta_fs = np.linalg.solve(X_fs.T @ X_fs, X_fs.T @ y_fs)
                        resid_fs = y_fs - X_fs @ beta_fs
                        rss_full = np.sum(resid_fs**2)
                        
                        # Restricted model (only exogenous)
                        beta_restricted = np.linalg.solve(X_exog_bench.T @ X_exog_bench, X_exog_bench.T @ y_fs)
                        resid_restricted = y_fs - X_exog_bench @ beta_restricted
                        rss_restricted = np.sum(resid_restricted**2)
                        
                        # F-statistic
                        f_stat = ((rss_restricted - rss_full) / n_instruments) / (rss_full / (len(y_fs) - len(beta_fs)))
                        benchmark_first_stage_f.append(f_stat)
                    
                except Exception as e:
                    self.log(f"Benchmark failed: {e}")
            
            # Calculate metrics
            result = {
                'test_name': test_name,
                'n_obs': n_obs,
                'n_exogenous': n_exogenous,
                'n_endogenous': n_endogenous,
                'n_instruments': n_instruments,
                'n_partitions': n_partitions,
                'chunk_size': chunk_size,
                'success': success,
                'error_msg': error_msg,
                'data_gen_time': data_gen_time,
                'processing_time': processing_time,
                'true_beta_all': true_beta_all.copy(),
                'estimated_beta': estimated_beta.copy(),
                'se_classical': se_classical.copy(),
                'se_cluster1': se_cluster1.copy(),
                'se_cluster2': se_cluster2.copy() if se_cluster2 is not None else None,
                'se_2sls': se_2sls.copy() if se_2sls is not None else None,
                'twosls_cov_success': twosls_cov_success,
                'first_stage_stats': first_stage_stats,
                'benchmark_beta': benchmark_beta.copy() if benchmark_beta is not None else None,
                'benchmark_se': benchmark_se.copy() if benchmark_se is not None else None,
                'benchmark_first_stage_f': benchmark_first_stage_f if benchmark_first_stage_f is not None else None,
                'bias': estimated_beta - true_beta_all,
                'abs_bias': np.abs(estimated_beta - true_beta_all),
                'rel_bias': np.abs(estimated_beta - true_beta_all) / np.abs(true_beta_all),
                'n_obs_processed': twosls.n_obs if success else 0,
                'rss': twosls.rss if success else np.nan,
                'rank_deficient': twosls.rank_deficient if success else None,
                'instrument_strength': kwargs.get('instrument_strength', 0.5),
                'endogeneity_strength': kwargs.get('endogeneity_strength', 0.3),
                'theoretical_f_stats': df.attrs.get('theoretical_f_stats', [])
            }
            
            # Additional metrics if benchmark available
            if benchmark_beta is not None:
                result.update({
                    'benchmark_diff': estimated_beta - benchmark_beta,
                    'benchmark_agreement': np.allclose(estimated_beta, benchmark_beta, rtol=1e-2),
                    'se_agreement': np.allclose(se_classical, benchmark_se, rtol=0.2) if benchmark_se is not None else None
                })
            
            # Weak instruments check
            weak_instruments_detected = any(v for k, v in first_stage_stats.items() if 'weak_instruments' in k and v)
            result['weak_instruments_detected'] = weak_instruments_detected
            
            self.log(f"  Success: {success}")
            if success:
                self.log(f"  Max absolute bias: {np.max(result['abs_bias']):.6f}")
                self.log(f"  Max relative bias: {np.max(result['rel_bias']):.6f}")
                self.log(f"  Weak instruments: {weak_instruments_detected}")
                self.log(f"  Processing time: {processing_time:.2f}s")
                self.log(f"  Obs processed: {result['n_obs_processed']:,}/{n_obs}")
            
            return result
    
    def run_2sls_test_suite(self) -> List[Dict]:
        """Run comprehensive 2SLS Monte Carlo test suite"""
        
        self.log("Starting Monte Carlo Test Suite for Online2SLS")
        self.log("=" * 60)
        
        test_scenarios = [
            # Basic functionality tests
            {
                'test_name': '2sls_basic_just_identified',
                'n_obs': 5000,
                'n_exogenous': 1,
                'n_endogenous': 1,
                'n_instruments': 1,  # Just identified
                'chunk_size': 500,
                'instrument_strength': 0.8
            },
            {
                'test_name': '2sls_basic_over_identified',
                'n_obs': 5000,
                'n_exogenous': 2,
                'n_endogenous': 1,
                'n_instruments': 2,  # Over identified
                'chunk_size': 500,
                'instrument_strength': 0.6
            },
            {
                'test_name': '2sls_multiple_endogenous',
                'n_obs': 8000,
                'n_exogenous': 2,
                'n_endogenous': 2,
                'n_instruments': 3,
                'chunk_size': 800,
                'instrument_strength': 0.7
            },
            
            # Different sample sizes
            {
                'test_name': '2sls_small_sample',
                'n_obs': 1000,
                'n_exogenous': 1,
                'n_endogenous': 1,
                'n_instruments': 2,
                'chunk_size': 200,
                'instrument_strength': 0.9
            },
            {
                'test_name': '2sls_large_sample',
                'n_obs': 20000,
                'n_exogenous': 2,
                'n_endogenous': 1,
                'n_instruments': 2,
                'chunk_size': 2000,
                'instrument_strength': 0.5
            },
            
            # Instrument strength variations
            {
                'test_name': '2sls_strong_instruments',
                'n_obs': 5000,
                'n_exogenous': 2,
                'n_endogenous': 1,
                'n_instruments': 2,
                'chunk_size': 500,
                'instrument_strength': 1.2  # Strong
            },
            {
                'test_name': '2sls_weak_instruments',
                'n_obs': 5000,
                'n_exogenous': 2,
                'n_endogenous': 1,
                'n_instruments': 2,
                'chunk_size': 500,
                'instrument_strength': 0.1  # Weak
            },
            
            # Endogeneity strength variations
            {
                'test_name': '2sls_low_endogeneity',
                'n_obs': 5000,
                'n_exogenous': 2,
                'n_endogenous': 1,
                'n_instruments': 2,
                'chunk_size': 500,
                'endogeneity_strength': 0.1
            },
            {
                'test_name': '2sls_high_endogeneity',
                'n_obs': 5000,
                'n_exogenous': 2,
                'n_endogenous': 1,
                'n_instruments': 2,
                'chunk_size': 500,
                'endogeneity_strength': 0.8
            },
            
            # Different noise levels
            {
                'test_name': '2sls_low_noise',
                'n_obs': 5000,
                'n_exogenous': 2,
                'n_endogenous': 1,
                'n_instruments': 2,
                'chunk_size': 500,
                'noise_std': 0.5
            },
            {
                'test_name': '2sls_high_noise',
                'n_obs': 5000,
                'n_exogenous': 2,
                'n_endogenous': 1,
                'n_instruments': 2,
                'chunk_size': 500,
                'noise_std': 2.0
            },
            
            # Different chunk sizes
            {
                'test_name': '2sls_small_chunks',
                'n_obs': 5000,
                'n_exogenous': 2,
                'n_endogenous': 1,
                'n_instruments': 2,
                'chunk_size': 100
            },
            {
                'test_name': '2sls_large_chunks',
                'n_obs': 5000,
                'n_exogenous': 2,
                'n_endogenous': 1,
                'n_instruments': 2,
                'chunk_size': 1500
            },
            
            # Missing data
            {
                'test_name': '2sls_missing_data_5pct',
                'n_obs': 5000,
                'n_exogenous': 2,
                'n_endogenous': 1,
                'n_instruments': 2,
                'chunk_size': 500,
                'missing_rate': 0.05
            },
            
            # Specific coefficient patterns
            {
                'test_name': '2sls_zero_exogenous_effect',
                'n_obs': 5000,
                'n_exogenous': 2,
                'n_endogenous': 1,
                'n_instruments': 2,
                'chunk_size': 500,
                'true_beta_exog': np.array([1.0, 0.0, 0.0]),  # Intercept + two zeros
                'true_beta_endog': np.array([2.0])
            },
            {
                'test_name': '2sls_large_coefficients',
                'n_obs': 5000,
                'n_exogenous': 1,
                'n_endogenous': 1,
                'n_instruments': 2,
                'chunk_size': 500,
                'true_beta_exog': np.array([50.0, -25.0]),
                'true_beta_endog': np.array([100.0])
            }
        ]
        
        # Run all tests
        for scenario in test_scenarios:
            try:
                result = self.run_single_2sls_test(**scenario)
                self.results.append(result)
            except Exception as e:
                self.log(f"Test {scenario['test_name']} failed completely: {e}")
                # Add failed result
                failed_result = {
                    'test_name': scenario['test_name'],
                    'success': False,
                    'error_msg': str(e),
                    **{k: np.nan for k in ['bias', 'abs_bias', 'rel_bias', 'processing_time']}
                }
                self.results.append(failed_result)
        
        return self.results
    
    def analyze_2sls_results(self) -> Dict:
        """Analyze 2SLS test results and generate summary statistics"""
        
        if not self.results:
            return {}
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(self.results)
        
        # Success rate
        success_rate = results_df['success'].mean()
        
        # Filter successful tests
        successful = results_df[results_df['success']].copy()
        
        if len(successful) == 0:
            return {'success_rate': success_rate, 'n_tests': len(self.results)}
        
        # Bias statistics
        bias_stats = {}
        for col in ['abs_bias', 'rel_bias']:
            if col in successful.columns:
                values = []
                for _, row in successful.iterrows():
                    if isinstance(row[col], np.ndarray):
                        values.extend(row[col])
                    else:
                        values.append(row[col])
                
                values = [v for v in values if np.isfinite(v)]
                if values:
                    bias_stats[col] = {
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'max': np.max(values),
                        'q95': np.percentile(values, 95),
                        'q99': np.percentile(values, 99)
                    }
        
        # IV-specific statistics
        iv_stats = {}
        
        # Weak instruments
        if 'weak_instruments_detected' in successful.columns:
            weak_rate = successful['weak_instruments_detected'].mean()
            iv_stats['weak_instruments_rate'] = weak_rate
        
        # 2SLS covariance success
        if 'twosls_cov_success' in successful.columns:
            cov_success_rate = successful['twosls_cov_success'].mean()
            iv_stats['twosls_covariance_success_rate'] = cov_success_rate
        
        # Rank deficiency
        if 'rank_deficient' in successful.columns:
            rank_deficient_rate = successful['rank_deficient'].fillna(False).mean()
            iv_stats['rank_deficient_rate'] = rank_deficient_rate
        
        # Performance statistics
        perf_stats = {}
        if 'processing_time' in successful.columns:
            times = successful['processing_time'].dropna()
            if len(times) > 0:
                perf_stats['processing_time'] = {
                    'mean': times.mean(),
                    'median': times.median(),
                    'min': times.min(),
                    'max': times.max()
                }
        
        # Benchmark agreement
        benchmark_stats = {}
        if 'benchmark_agreement' in successful.columns:
            agreements = successful['benchmark_agreement'].dropna()
            if len(agreements) > 0:
                benchmark_stats['agreement_rate'] = agreements.mean()
        
        summary = {
            'n_tests': len(self.results),
            'success_rate': success_rate,
            'n_successful': len(successful),
            'bias_statistics': bias_stats,
            'iv_statistics': iv_stats,
            'performance_statistics': perf_stats,
            'benchmark_statistics': benchmark_stats
        }
        
        return summary
    
    def create_2sls_diagnostic_plots(self, save_path: Path = None):
        """Create diagnostic plots specific to 2SLS results"""
        
        if not self.results:
            print("No results to plot")
            return
        
        # Filter successful results
        successful_results = [r for r in self.results if r['success']]
        
        if not successful_results:
            print("No successful results to plot")
            return
        
        # Set up the plotting
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Monte Carlo Test Results - Online2SLS Validation', fontsize=16)
        
        # Plot 1: Bias vs True Coefficients
        ax = axes[0, 0]
        for result in successful_results:
            if isinstance(result['true_beta_all'], np.ndarray) and isinstance(result['bias'], np.ndarray):
                ax.scatter(result['true_beta_all'], result['bias'], alpha=0.6, s=20)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('True Coefficient Value')
        ax.set_ylabel('Bias (Estimated - True)')
        ax.set_title('Bias vs True Coefficients')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Instrument Strength vs Bias
        ax = axes[0, 1]
        strengths = []
        max_biases = []
        for result in successful_results:
            if 'instrument_strength' in result and isinstance(result['abs_bias'], np.ndarray):
                strengths.append(result['instrument_strength'])
                max_biases.append(np.max(result['abs_bias']))
        
        if strengths and max_biases:
            ax.scatter(strengths, max_biases, alpha=0.7)
            ax.set_xlabel('Instrument Strength')
            ax.set_ylabel('Maximum Absolute Bias')
            ax.set_title('Instrument Strength vs Bias')
            ax.grid(True, alpha=0.3)
        
        # Plot 3: First Stage F-Statistics
        ax = axes[0, 2]
        f_stats = []
        for result in successful_results:
            if 'first_stage_stats' in result:
                for key, value in result['first_stage_stats'].items():
                    if 'f_stat' in key and np.isfinite(value):
                        f_stats.append(value)
        
        if f_stats:
            ax.hist(f_stats, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(x=10, color='red', linestyle='--', label='Weak IV threshold')
            ax.set_xlabel('First Stage F-Statistic')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of First Stage F-Statistics')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Endogeneity vs Bias
        ax = axes[1, 0]
        endogeneity_strengths = []
        endog_max_biases = []
        for result in successful_results:
            if 'endogeneity_strength' in result and isinstance(result['abs_bias'], np.ndarray):
                endogeneity_strengths.append(result['endogeneity_strength'])
                endog_max_biases.append(np.max(result['abs_bias']))
        
        if endogeneity_strengths and endog_max_biases:
            ax.scatter(endogeneity_strengths, endog_max_biases, alpha=0.7)
            ax.set_xlabel('Endogeneity Strength')
            ax.set_ylabel('Maximum Absolute Bias')
            ax.set_title('Endogeneity Strength vs Bias')
            ax.grid(True, alpha=0.3)
        
        # Plot 5: Processing Time vs Dataset Size
        ax = axes[1, 1]
        n_obs_list = []
        proc_times = []
        for result in successful_results:
            if 'n_obs' in result and 'processing_time' in result:
                if np.isfinite(result['processing_time']):
                    n_obs_list.append(result['n_obs'])
                    proc_times.append(result['processing_time'])
        
        if n_obs_list and proc_times:
            ax.scatter(n_obs_list, proc_times, alpha=0.7)
            ax.set_xlabel('Number of Observations')
            ax.set_ylabel('Processing Time (seconds)')
            ax.set_title('Processing Time vs Dataset Size')
            ax.grid(True, alpha=0.3)
        
        # Plot 6: Success Rate by Test Category
        ax = axes[1, 2]
        test_names = [r['test_name'] for r in self.results]
        success_flags = [r['success'] for r in self.results]
        
        # Group by test category
        categories = {}
        for name, success in zip(test_names, success_flags):
            category = '_'.join(name.split('_')[1:3])  # Extract category
            if category not in categories:
                categories[category] = []
            categories[category].append(success)
        
        cat_names = list(categories.keys())
        success_rates = [np.mean(categories[cat]) for cat in cat_names]
        
        bars = ax.bar(cat_names, success_rates, alpha=0.7)
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate by Test Category')
        ax.set_ylim(0, 1.1)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{rate:.2f}', ha='center', va='bottom')
        
        # Plot 7: 2SLS vs Classical Standard Errors
        ax = axes[2, 0]
        se_classical = []
        se_2sls = []
        
        for result in successful_results:
            if (isinstance(result['se_classical'], np.ndarray) and 
                result['se_2sls'] is not None and isinstance(result['se_2sls'], np.ndarray)):
                se_classical.extend(result['se_classical'])
                se_2sls.extend(result['se_2sls'])
        
        if se_classical and se_2sls:
            ax.scatter(se_classical, se_2sls, alpha=0.6)
            
            # Add diagonal line
            min_se = min(min(se_classical), min(se_2sls))
            max_se = max(max(se_classical), max(se_2sls))
            ax.plot([min_se, max_se], [min_se, max_se], 'r--', alpha=0.7)
            
            ax.set_xlabel('Classical Standard Errors')
            ax.set_ylabel('2SLS Standard Errors')
            ax.set_title('Standard Error Comparison')
            ax.grid(True, alpha=0.3)
        
        # Plot 8: Weak Instruments Detection
        ax = axes[2, 1]
        weak_detected = [r.get('weak_instruments_detected', False) for r in successful_results]
        weak_counts = [sum(weak_detected), len(weak_detected) - sum(weak_detected)]
        labels = ['Weak Instruments', 'Strong Instruments']
        
        if sum(weak_counts) > 0:
            ax.pie(weak_counts, labels=labels, autopct='%1.1f%%')
            ax.set_title('Weak Instruments Detection')
        
        # Plot 9: Benchmark Comparison
        ax = axes[2, 2]
        benchmark_diffs = []
        for result in successful_results:
            if result.get('benchmark_diff') is not None:
                if isinstance(result['benchmark_diff'], np.ndarray):
                    benchmark_diffs.extend(result['benchmark_diff'])
        
        if benchmark_diffs:
            ax.hist(benchmark_diffs, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--')
            ax.set_xlabel('Difference from Benchmark')
            ax.set_ylabel('Frequency')
            ax.set_title('Online2SLS vs Benchmark Differences')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No benchmark\ncomparisons available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Benchmark Comparison')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {save_path}")
        
        plt.show()
    
    def print_2sls_summary_report(self):
        """Print comprehensive 2SLS summary report"""
        
        summary = self.analyze_2sls_results()
        
        print("\n" + "="*80)
        print("MONTE CARLO 2SLS TEST SUMMARY REPORT")
        print("="*80)
        
        print(f"\nTEST OVERVIEW:")
        print(f"  Total tests run: {summary['n_tests']}")
        print(f"  Successful tests: {summary['n_successful']}")
        print(f"  Success rate: {summary['success_rate']:.1%}")
        
        if 'iv_statistics' in summary and summary['iv_statistics']:
            print(f"\nIV-SPECIFIC DIAGNOSTICS:")
            iv_stats = summary['iv_statistics']
            
            if 'weak_instruments_rate' in iv_stats:
                print(f"  Weak instruments detected: {iv_stats['weak_instruments_rate']:.1%}")
            
            if 'twosls_covariance_success_rate' in iv_stats:
                print(f"  2SLS covariance success: {iv_stats['twosls_covariance_success_rate']:.1%}")
            
            if 'rank_deficient_rate' in iv_stats:
                print(f"  Rank deficient cases: {iv_stats['rank_deficient_rate']:.1%}")
        
        if 'bias_statistics' in summary and summary['bias_statistics']:
            print(f"\nBIAS ANALYSIS:")
            if 'abs_bias' in summary['bias_statistics']:
                abs_bias = summary['bias_statistics']['abs_bias']
                print(f"  Absolute Bias:")
                print(f"    Mean: {abs_bias['mean']:.6f}")
                print(f"    Median: {abs_bias['median']:.6f}")
                print(f"    95th percentile: {abs_bias['q95']:.6f}")
                print(f"    Maximum: {abs_bias['max']:.6f}")
            
            if 'rel_bias' in summary['bias_statistics']:
                rel_bias = summary['bias_statistics']['rel_bias']
                print(f"  Relative Bias:")
                print(f"    Mean: {rel_bias['mean']:.4%}")
                print(f"    Median: {rel_bias['median']:.4%}")
                print(f"    95th percentile: {rel_bias['q95']:.4%}")
                print(f"    Maximum: {rel_bias['max']:.4%}")
        
        if 'performance_statistics' in summary and summary['performance_statistics']:
            print(f"\nPERFORMANCE ANALYSIS:")
            if 'processing_time' in summary['performance_statistics']:
                perf = summary['performance_statistics']['processing_time']
                print(f"  Processing Time (seconds):")
                print(f"    Mean: {perf['mean']:.2f}")
                print(f"    Median: {perf['median']:.2f}")
                print(f"    Range: {perf['min']:.2f} - {perf['max']:.2f}")
        
        if 'benchmark_statistics' in summary and summary['benchmark_statistics']:
            print(f"\nBENCHMARK COMPARISON:")
            if 'agreement_rate' in summary['benchmark_statistics']:
                print(f"  Agreement with manual 2SLS: {summary['benchmark_statistics']['agreement_rate']:.1%}")
        
        # Test-specific results
        print(f"\nDETAILED RESULTS BY TEST:")
        for result in self.results:
            status = "PASS" if result['success'] else "FAIL"
            print(f"  {result['test_name']:<35} {status}")
            
            if result['success'] and isinstance(result.get('abs_bias'), np.ndarray):
                max_bias = np.max(result['abs_bias'])
                weak_iv = result.get('weak_instruments_detected', False)
                print(f"    {'':37} Max bias: {max_bias:.6f}, Weak IV: {weak_iv}")
        
        # Overall assessment
        print(f"\nOVERALL ASSESSMENT:")
        if summary['success_rate'] >= 0.95:
            assessment = "EXCELLENT"
        elif summary['success_rate'] >= 0.90:
            assessment = "GOOD"
        elif summary['success_rate'] >= 0.80:
            assessment = "ACCEPTABLE"
        else:
            assessment = "NEEDS IMPROVEMENT"
        
        print(f"  Overall performance: {assessment}")
        
        if 'bias_statistics' in summary and 'rel_bias' in summary['bias_statistics']:
            avg_rel_bias = summary['bias_statistics']['rel_bias']['mean']
            if avg_rel_bias < 0.01:
                bias_assessment = "EXCELLENT (< 1%)"
            elif avg_rel_bias < 0.05:
                bias_assessment = "GOOD (< 5%)"
            elif avg_rel_bias < 0.10:
                bias_assessment = "ACCEPTABLE (< 10%)"
            else:
                bias_assessment = "POOR (≥ 10%)"
            
            print(f"  Bias performance: {bias_assessment}")
        
        print("="*80)


def main():
    """Main function to run 2SLS Monte Carlo tests"""
    
    print("Online2SLS Monte Carlo Validation Test Suite")
    print("=" * 50)
    
    # Initialize tester
    tester = MonteCarlo2SLSTester(verbose=True)
    
    # Run test suite
    print("Running comprehensive 2SLS test suite...")
    results = tester.run_2sls_test_suite()
    
    # Print summary
    tester.print_2sls_summary_report()
    
    # Create diagnostic plots
    print("\nGenerating 2SLS diagnostic plots...")
    try:
        tester.create_2sls_diagnostic_plots()
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    # Save results
    results_path = Path(__file__).parent / "monte_carlo_2sls_results.csv"
    try:
        # Convert results to CSV format
        csv_results = []
        for result in results:
            csv_row = {k: v for k, v in result.items() 
                      if not isinstance(v, (np.ndarray, dict))}
            
            # Add summary statistics for array fields
            for field in ['true_beta_all', 'estimated_beta', 'bias', 'abs_bias', 'rel_bias']:
                if isinstance(result.get(field), np.ndarray):
                    arr = result[field]
                    csv_row[f'{field}_mean'] = np.mean(arr) if len(arr) > 0 else np.nan
                    csv_row[f'{field}_max'] = np.max(arr) if len(arr) > 0 else np.nan
                    csv_row[f'{field}_std'] = np.std(arr) if len(arr) > 0 else np.nan
            
            # Add first stage stats summary
            if isinstance(result.get('first_stage_stats'), dict):
                for key, value in result['first_stage_stats'].items():
                    if isinstance(value, (int, float)):
                        csv_row[f'fs_{key}'] = value
            
            csv_results.append(csv_row)
        
        pd.DataFrame(csv_results).to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")
        
    except Exception as e:
        print(f"Could not save results: {e}")
    
    print("\n2SLS Monte Carlo testing completed!")
    
    return tester


if __name__ == "__main__":
    tester = main()

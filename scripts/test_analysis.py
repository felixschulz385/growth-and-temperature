#!/usr/bin/env python3
"""
Monte Carlo Testing Suite for OnlineRLS Implementation

This script generates synthetic datasets with known parameters and tests whether
the OnlineRLS implementation can recover the true coefficients accurately.
Tests include various scenarios: different sample sizes, clustering structures,
missing data patterns, and numerical stability conditions.
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

from gnt.analysis.models.online_RLS import OnlineRLS, process_partitioned_dataset_parallel
import pyarrow as pa
import pyarrow.parquet as pq

# Set random seed for reproducibility
np.random.seed(42)

class MonteCarloTester:
    """Monte Carlo testing framework for OnlineRLS"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = []
        
    def log(self, message: str):
        """Print message if verbose mode is on"""
        if self.verbose:
            print(f"[MC Test] {message}")
    
    def generate_synthetic_data(self, 
                              n_obs: int = 10000,
                              n_features: int = 3,
                              true_beta: np.ndarray = None,
                              noise_std: float = 1.0,
                              n_clusters1: int = 50,
                              n_clusters2: int = 20,
                              missing_rate: float = 0.0,
                              correlation: float = 0.0) -> pd.DataFrame:
        """
        Generate synthetic regression data with clustering structure.
        
        Parameters:
        -----------
        n_obs : int
            Number of observations
        n_features : int  
            Number of features (excluding intercept)
        true_beta : np.ndarray
            True coefficients [intercept, beta1, beta2, ...]
        noise_std : float
            Standard deviation of error term
        n_clusters1 : int
            Number of first-level clusters
        n_clusters2 : int
            Number of second-level clusters
        missing_rate : float
            Proportion of missing values to introduce
        correlation : float
            Correlation between features
        """
        
        if true_beta is None:
            # Default coefficients: intercept=5, slopes=[2, -1, 0.5, ...]
            true_beta = np.array([5.0] + [2.0 * ((-1) ** i) / (i + 1) for i in range(n_features)])
        
        # Generate correlated features
        if correlation > 0:
            # Create correlation matrix
            corr_matrix = np.full((n_features, n_features), correlation)
            np.fill_diagonal(corr_matrix, 1.0)
            
            # Generate correlated features using Cholesky decomposition
            L = np.linalg.cholesky(corr_matrix)
            uncorr_features = np.random.normal(0, 1, (n_obs, n_features))
            X_raw = uncorr_features @ L.T
        else:
            # Generate independent features
            X_raw = np.random.normal(0, 1, (n_obs, n_features))
        
        # Add some non-linear transformations to make it more realistic
        X_raw[:, 0] = np.random.exponential(1, n_obs)  # Exponential distribution
        if n_features > 1:
            X_raw[:, 1] = np.random.uniform(-2, 2, n_obs)  # Uniform distribution
        if n_features > 2:
            X_raw[:, 2] = np.random.gamma(2, 1, n_obs)  # Gamma distribution
        
        # Create design matrix with intercept
        X = np.column_stack([np.ones(n_obs), X_raw])
        
        # Generate cluster IDs
        cluster1_ids = np.random.randint(0, n_clusters1, n_obs)
        cluster2_ids = np.random.randint(0, n_clusters2, n_obs)
        
        # Add cluster-specific effects for more realistic error structure
        cluster1_effects = np.random.normal(0, 0.5, n_clusters1)
        cluster2_effects = np.random.normal(0, 0.3, n_clusters2)
        
        cluster_error = (cluster1_effects[cluster1_ids] + 
                        cluster2_effects[cluster2_ids])
        
        # Generate outcome with clustered errors
        epsilon = np.random.normal(0, noise_std, n_obs) + cluster_error
        y = X @ true_beta + epsilon
        
        # Create DataFrame
        feature_names = [f'feature_{i+1}' for i in range(n_features)]
        df = pd.DataFrame({
            'y': y,
            **{name: X_raw[:, i] for i, name in enumerate(feature_names)},
            'cluster1': cluster1_ids,
            'cluster2': cluster2_ids,
            'true_y': X @ true_beta  # Store true signal for analysis
        })
        
        # Introduce missing values
        if missing_rate > 0:
            n_missing = int(n_obs * missing_rate)
            missing_indices = np.random.choice(n_obs, n_missing, replace=False)
            
            # Randomly choose which variables to make missing
            for idx in missing_indices:
                missing_var = np.random.choice(feature_names + ['y'])
                df.loc[idx, missing_var] = np.nan
        
        # Store true parameters for validation
        df.attrs['true_beta'] = true_beta
        df.attrs['noise_std'] = noise_std
        df.attrs['n_clusters1'] = n_clusters1
        df.attrs['n_clusters2'] = n_clusters2
        
        return df
    
    def create_partitioned_dataset(self, df: pd.DataFrame, 
                                 temp_dir: Path,
                                 n_partitions: int = 5) -> Path:
        """Create a partitioned parquet dataset from DataFrame"""
        
        parquet_dir = temp_dir / "test_data.parquet"
        parquet_dir.mkdir(exist_ok=True)
        
        # Split data into partitions
        partition_size = len(df) // n_partitions
        
        for i in range(n_partitions):
            start_idx = i * partition_size
            if i == n_partitions - 1:
                # Last partition gets remaining rows
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
    
    def run_single_test(self, 
                       test_name: str,
                       n_obs: int = 10000,
                       n_features: int = 3,
                       true_beta: np.ndarray = None,
                       chunk_size: int = 1000,
                       n_partitions: int = 5,
                       **kwargs) -> Dict:
        """Run a single Monte Carlo test"""
        
        self.log(f"Running test: {test_name}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Generate synthetic data
            start_time = time.time()
            df = self.generate_synthetic_data(
                n_obs=n_obs, 
                n_features=n_features, 
                true_beta=true_beta,
                **kwargs
            )
            
            true_beta = df.attrs['true_beta']
            data_gen_time = time.time() - start_time
            
            # Create partitioned dataset
            parquet_path = self.create_partitioned_dataset(df, temp_path, n_partitions)
            
            # Test OnlineRLS
            start_time = time.time()
            
            try:
                rls = process_partitioned_dataset_parallel(
                    parquet_path=parquet_path,
                    feature_cols=[f'feature_{i+1}' for i in range(n_features)],
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
                estimated_beta = rls.theta
                
                # Calculate standard errors
                se_classical = rls.get_standard_errors('classical')
                se_cluster1 = rls.get_standard_errors('one_way')
                
                try:
                    se_cluster2 = rls.get_standard_errors('two_way')
                except:
                    se_cluster2 = None
                
            except Exception as e:
                success = False
                error_msg = str(e)
                estimated_beta = np.full(len(true_beta), np.nan)
                se_classical = np.full(len(true_beta), np.nan)
                se_cluster1 = np.full(len(true_beta), np.nan)
                se_cluster2 = None
                processing_time = np.nan
            
            # Benchmark against sklearn/statsmodels if dataset is small enough
            benchmark_beta = None
            benchmark_se = None
            
            if n_obs <= 50000:  # Only for smaller datasets
                try:
                    from sklearn.linear_model import LinearRegression
                    
                    # Prepare data (remove missing values)
                    df_clean = df.dropna()
                    X_bench = df_clean[[f'feature_{i+1}' for i in range(n_features)]].values
                    y_bench = df_clean['y'].values
                    
                    # Add intercept
                    X_bench = np.column_stack([np.ones(len(X_bench)), X_bench])
                    
                    # Fit with sklearn
                    lr = LinearRegression(fit_intercept=False)
                    lr.fit(X_bench, y_bench)
                    benchmark_beta = lr.coef_
                    
                    # Calculate standard errors manually
                    residuals = y_bench - X_bench @ benchmark_beta
                    mse = np.sum(residuals**2) / (len(y_bench) - len(benchmark_beta))
                    var_beta = mse * np.linalg.inv(X_bench.T @ X_bench)
                    benchmark_se = np.sqrt(np.diag(var_beta))
                    
                except Exception as e:
                    self.log(f"Benchmark failed: {e}")
            
            # Calculate metrics
            result = {
                'test_name': test_name,
                'n_obs': n_obs,
                'n_features': n_features,
                'n_partitions': n_partitions,
                'chunk_size': chunk_size,
                'success': success,
                'error_msg': error_msg,
                'data_gen_time': data_gen_time,
                'processing_time': processing_time,
                'true_beta': true_beta.copy(),
                'estimated_beta': estimated_beta.copy(),
                'se_classical': se_classical.copy(),
                'se_cluster1': se_cluster1.copy(),
                'se_cluster2': se_cluster2.copy() if se_cluster2 is not None else None,
                'benchmark_beta': benchmark_beta.copy() if benchmark_beta is not None else None,
                'benchmark_se': benchmark_se.copy() if benchmark_se is not None else None,
                'bias': estimated_beta - true_beta,
                'abs_bias': np.abs(estimated_beta - true_beta),
                'rel_bias': np.abs(estimated_beta - true_beta) / np.abs(true_beta),
                'n_obs_processed': rls.n_obs if success else 0,
                'rss': rls.rss if success else np.nan
            }
            
            # Additional metrics if benchmark available
            if benchmark_beta is not None:
                result.update({
                    'benchmark_diff': estimated_beta - benchmark_beta,
                    'benchmark_agreement': np.allclose(estimated_beta, benchmark_beta, rtol=1e-3),
                    'se_agreement': np.allclose(se_classical, benchmark_se, rtol=0.1) if benchmark_se is not None else None
                })
            
            self.log(f"  Success: {success}")
            if success:
                self.log(f"  Max absolute bias: {np.max(result['abs_bias']):.6f}")
                self.log(f"  Max relative bias: {np.max(result['rel_bias']):.6f}")
                self.log(f"  Processing time: {processing_time:.2f}s")
                self.log(f"  Obs processed: {result['n_obs_processed']:,}/{n_obs}")
            
            return result
    
    def run_test_suite(self) -> List[Dict]:
        """Run comprehensive Monte Carlo test suite"""
        
        self.log("Starting Monte Carlo Test Suite for OnlineRLS")
        self.log("=" * 60)
        
        test_scenarios = [
            # Basic functionality tests
            {
                'test_name': 'basic_small',
                'n_obs': 1000,
                'n_features': 2,
                'chunk_size': 100,
                'n_partitions': 3
            },
            {
                'test_name': 'basic_medium',
                'n_obs': 10000,
                'n_features': 3,
                'chunk_size': 500,
                'n_partitions': 5
            },
            {
                'test_name': 'basic_large',
                'n_obs': 50000,
                'n_features': 4,
                'chunk_size': 2000,
                'n_partitions': 10
            },
            
            # Different chunk sizes
            {
                'test_name': 'small_chunks',
                'n_obs': 5000,
                'n_features': 3,
                'chunk_size': 50,
                'n_partitions': 8
            },
            {
                'test_name': 'large_chunks',
                'n_obs': 5000,
                'n_features': 3,
                'chunk_size': 1000,
                'n_partitions': 3
            },
            
            # Different clustering structures
            {
                'test_name': 'many_clusters',
                'n_obs': 10000,
                'n_features': 3,
                'n_clusters1': 100,
                'n_clusters2': 50,
                'chunk_size': 500
            },
            {
                'test_name': 'few_clusters',
                'n_obs': 10000,
                'n_features': 3,
                'n_clusters1': 5,
                'n_clusters2': 3,
                'chunk_size': 500
            },
            
            # Different noise levels
            {
                'test_name': 'low_noise',
                'n_obs': 5000,
                'n_features': 3,
                'noise_std': 0.1,
                'chunk_size': 500
            },
            {
                'test_name': 'high_noise',
                'n_obs': 5000,
                'n_features': 3,
                'noise_std': 5.0,
                'chunk_size': 500
            },
            
            # Correlated features
            {
                'test_name': 'correlated_features',
                'n_obs': 5000,
                'n_features': 4,
                'correlation': 0.7,
                'chunk_size': 500
            },
            
            # Missing data
            {
                'test_name': 'missing_data_5pct',
                'n_obs': 5000,
                'n_features': 3,
                'missing_rate': 0.05,
                'chunk_size': 500
            },
            {
                'test_name': 'missing_data_20pct',
                'n_obs': 5000,
                'n_features': 3,
                'missing_rate': 0.20,
                'chunk_size': 500
            },
            
            # Specific coefficient patterns
            {
                'test_name': 'zero_coefficients',
                'n_obs': 5000,
                'n_features': 5,
                'true_beta': np.array([1.0, 2.0, 0.0, -1.5, 0.0, 0.5]),
                'chunk_size': 500
            },
            {
                'test_name': 'large_coefficients',
                'n_obs': 5000,
                'n_features': 3,
                'true_beta': np.array([100.0, -50.0, 75.0, -25.0]),
                'chunk_size': 500
            },
            {
                'test_name': 'small_coefficients',
                'n_obs': 5000,
                'n_features': 3,
                'true_beta': np.array([0.01, 0.005, -0.003, 0.002]),
                'chunk_size': 500
            }
        ]
        
        # Run all tests
        for scenario in test_scenarios:
            try:
                result = self.run_single_test(**scenario)
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
    
    def analyze_results(self) -> Dict:
        """Analyze test results and generate summary statistics"""
        
        if not self.results:
            return {}
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(self.results)
        
        # Success rate
        success_rate = results_df['success'].mean()
        
        # Filter successful tests for bias analysis
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
        
        # Benchmark agreement (if available)
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
            'performance_statistics': perf_stats,
            'benchmark_statistics': benchmark_stats
        }
        
        return summary
    
    def create_diagnostic_plots(self, save_path: Path = None):
        """Create diagnostic plots for test results"""
        
        if not self.results:
            print("No results to plot")
            return
        
        # Filter successful results
        successful_results = [r for r in self.results if r['success']]
        
        if not successful_results:
            print("No successful results to plot")
            return
        
        # Set up the plotting
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Monte Carlo Test Results - OnlineRLS Validation', fontsize=16)
        
        # Plot 1: Bias vs True Coefficients
        ax = axes[0, 0]
        for result in successful_results:
            if isinstance(result['true_beta'], np.ndarray) and isinstance(result['bias'], np.ndarray):
                ax.scatter(result['true_beta'], result['bias'], alpha=0.6, s=20)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('True Coefficient Value')
        ax.set_ylabel('Bias (Estimated - True)')
        ax.set_title('Bias vs True Coefficients')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Relative Bias Distribution
        ax = axes[0, 1]
        rel_biases = []
        for result in successful_results:
            if isinstance(result['rel_bias'], np.ndarray):
                rel_biases.extend(result['rel_bias'][np.isfinite(result['rel_bias'])])
        
        if rel_biases:
            ax.hist(rel_biases, bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(x=0.01, color='red', linestyle='--', label='1% threshold')
            ax.axvline(x=0.05, color='orange', linestyle='--', label='5% threshold')
            ax.set_xlabel('Relative Bias')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Relative Bias')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Processing Time vs Dataset Size
        ax = axes[0, 2]
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
        
        # Plot 4: Benchmark Comparison (if available)
        ax = axes[1, 0]
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
            ax.set_title('OnlineRLS vs Benchmark Differences')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No benchmark\ncomparisons available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Benchmark Comparison')
        
        # Plot 5: Success Rate by Test Type
        ax = axes[1, 1]
        test_names = [r['test_name'] for r in self.results]
        success_flags = [r['success'] for r in self.results]
        
        # Group by test category
        categories = {}
        for name, success in zip(test_names, success_flags):
            category = name.split('_')[0]  # First part of test name
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
        
        # Plot 6: Standard Error Comparison
        ax = axes[1, 2]
        se_classical = []
        se_cluster = []
        
        for result in successful_results:
            if isinstance(result['se_classical'], np.ndarray) and isinstance(result['se_cluster1'], np.ndarray):
                se_classical.extend(result['se_classical'])
                se_cluster.extend(result['se_cluster1'])
        
        if se_classical and se_cluster:
            ax.scatter(se_classical, se_cluster, alpha=0.6)
            
            # Add diagonal line
            min_se = min(min(se_classical), min(se_cluster))
            max_se = max(max(se_classical), max(se_cluster))
            ax.plot([min_se, max_se], [min_se, max_se], 'r--', alpha=0.7)
            
            ax.set_xlabel('Classical Standard Errors')
            ax.set_ylabel('Cluster-Robust Standard Errors')
            ax.set_title('Standard Error Comparison')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No standard error\ncomparisons available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Standard Error Comparison')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {save_path}")
        
        plt.show()
    
    def print_summary_report(self):
        """Print a comprehensive summary report"""
        
        summary = self.analyze_results()
        
        print("\n" + "="*80)
        print("MONTE CARLO TEST SUMMARY REPORT")
        print("="*80)
        
        print(f"\nTEST OVERVIEW:")
        print(f"  Total tests run: {summary['n_tests']}")
        print(f"  Successful tests: {summary['n_successful']}")
        print(f"  Success rate: {summary['success_rate']:.1%}")
        
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
                print(f"  Agreement with sklearn: {summary['benchmark_statistics']['agreement_rate']:.1%}")
        
        # Test-specific results
        print(f"\nDETAILED RESULTS BY TEST:")
        for result in self.results:
            status = "PASS" if result['success'] else "FAIL"
            print(f"  {result['test_name']:<25} {status}")
            
            if result['success'] and isinstance(result.get('abs_bias'), np.ndarray):
                max_bias = np.max(result['abs_bias'])
                print(f"    {'':27} Max absolute bias: {max_bias:.6f}")
        
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
    """Main function to run Monte Carlo tests"""
    
    print("OnlineRLS Monte Carlo Validation Test Suite")
    print("=" * 50)
    
    # Initialize tester
    tester = MonteCarloTester(verbose=True)
    
    # Run test suite
    print("Running comprehensive test suite...")
    results = tester.run_test_suite()
    
    # Print summary
    tester.print_summary_report()
    
    # Create diagnostic plots
    print("\nGenerating diagnostic plots...")
    try:
        tester.create_diagnostic_plots()
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    # Save results
    results_path = Path(__file__).parent / "monte_carlo_results.csv"
    try:
        # Convert results to a format suitable for CSV
        csv_results = []
        for result in results:
            csv_row = {k: v for k, v in result.items() 
                      if not isinstance(v, np.ndarray)}
            
            # Add summary statistics for array fields
            for field in ['true_beta', 'estimated_beta', 'bias', 'abs_bias', 'rel_bias']:
                if isinstance(result.get(field), np.ndarray):
                    arr = result[field]
                    csv_row[f'{field}_mean'] = np.mean(arr) if len(arr) > 0 else np.nan
                    csv_row[f'{field}_max'] = np.max(arr) if len(arr) > 0 else np.nan
                    csv_row[f'{field}_std'] = np.std(arr) if len(arr) > 0 else np.nan
            
            csv_results.append(csv_row)
        
        pd.DataFrame(csv_results).to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")
        
    except Exception as e:
        print(f"Could not save results: {e}")
    
    print("\nMonte Carlo testing completed!")
    
    return tester


if __name__ == "__main__":
    tester = main()

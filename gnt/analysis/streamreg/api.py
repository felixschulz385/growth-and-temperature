import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, List, Literal
from pathlib import Path
import logging

from gnt.analysis.streamreg.data import StreamData
from gnt.analysis.streamreg.results import RegressionResults
from gnt.analysis.streamreg.formula import FormulaParser
from gnt.analysis.streamreg.transforms import FeatureTransformer
from gnt.analysis.streamreg.estimators.ols import OnlineRLS, ParallelOLSOrchestrator

logger = logging.getLogger(__name__)


class OLS:
    """
    Ordinary Least Squares estimator with streaming/online capabilities.
    
    Usage:
    ------
    >>> model = OLS(formula="y ~ x1 + x2 + I(x1^2)")
    >>> model.fit(data, cluster=['country', 'year'])
    >>> print(model.summary())
    >>> coeffs = model.coef_
    """
    
    def __init__(
        self,
        formula: str,
        alpha: float = 1e-3,
        forget_factor: float = 1.0,
        chunk_size: int = 10000,
        n_workers: Optional[int] = None,
        show_progress: bool = True
    ):
        """
        Initialize OLS estimator.
        
        Parameters:
        -----------
        formula : str
            R-style formula (e.g., "y ~ x1 + x2 + I(x1^2)")
        alpha : float
            Regularization parameter for numerical stability
        forget_factor : float
            Forgetting factor (1.0 = no forgetting)
        chunk_size : int
            Chunk size for processing large datasets
        n_workers : int, optional
            Number of parallel workers (auto-detected if None)
        show_progress : bool
            Show progress bar during fitting
        """
        self.formula = formula
        self.alpha = alpha
        self.forget_factor = forget_factor
        self.chunk_size = chunk_size
        self.n_workers = n_workers
        self.show_progress = show_progress
        
        # Parse formula
        self._parser = FormulaParser.parse(formula)
        
        if self._parser.instruments:
            raise ValueError(
                "Formula contains instruments. Use TwoSLS for IV estimation."
            )
        
        # Results (populated after fit)
        self._results = None
        self._rls_model = None
        self._cluster_type = 'classical'
    
    def fit(
        self,
        data: Union[str, Path, pd.DataFrame, StreamData],
        cluster: Optional[Union[str, List[str]]] = None
    ) -> 'OLS':
        """Fit the OLS model."""
        # Setup data
        if not isinstance(data, StreamData):
            data = StreamData(data, chunk_size=self.chunk_size)
        
        # Validate columns
        required_cols = [self._parser.target] + self._parser.features
        if cluster:
            cluster_cols = [cluster] if isinstance(cluster, str) else cluster
            required_cols.extend(cluster_cols)
        data.validate_columns(required_cols)
        
        # Determine cluster type
        self._cluster_type = 'classical'
        cluster1_col = None
        cluster2_col = None
        
        if cluster:
            if isinstance(cluster, str):
                self._cluster_type = 'one_way'
                cluster1_col = cluster
            elif len(cluster) == 2:
                self._cluster_type = 'two_way'
                cluster1_col, cluster2_col = cluster
            else:
                raise ValueError("cluster must be string or list of 2 strings")
        
        # Create feature engineering config
        feature_engineering = None
        if self._parser.transformations:
            feature_engineering = {'transformations': self._parser.transformations}
        
        # Setup feature transformation
        from gnt.analysis.streamreg.transforms import FeatureTransformer
        
        if feature_engineering or self._parser.has_intercept:
            transformer = FeatureTransformer.from_config(
                feature_engineering or {'transformations': []},
                self._parser.features,
                add_intercept=self._parser.has_intercept
            )
            n_features = transformer.get_n_features()
            feature_names = transformer.get_feature_names()
        else:
            n_features = len(self._parser.features)
            feature_names = self._parser.features.copy()
        
        # Use orchestrator for all data (handles parallelization internally)
        orchestrator = ParallelOLSOrchestrator(
            data=data,
            feature_cols=self._parser.features,
            target_col=self._parser.target,
            cluster1_col=cluster1_col,
            cluster2_col=cluster2_col,
            add_intercept=self._parser.has_intercept,
            n_features=n_features,
            transformed_feature_names=feature_names,
            alpha=self.alpha,
            chunk_size=self.chunk_size,
            n_workers=self.n_workers,
            show_progress=self.show_progress,
            verbose=True,
            feature_engineering=feature_engineering
        )
        
        self._rls_model = orchestrator.fit()
        
        # Store results
        self._results = self._create_results()
        
        return self
    
    def _create_results(self) -> RegressionResults:
        """Create standardized results object."""
        se = self._rls_model.get_standard_errors(self._cluster_type)
        t_stats = self._rls_model.theta / se
        
        from scipy import stats
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
        
        if self._cluster_type == 'classical':
            cov = self._rls_model.get_covariance_matrix()
        else:
            cov = self._rls_model.get_cluster_robust_covariance(self._cluster_type)
        
        # Get F-statistic
        f_stat, df_model, df_resid = self._rls_model.get_f_statistic()
        f_pval = self._rls_model.get_f_pvalue()
        
        cluster_diag = None
        if self._cluster_type != 'classical':
            cluster_diag = {}
            if self._cluster_type == 'one_way':
                cluster_diag['dim1'] = self._rls_model.diagnose_cluster_structure(
                    self._rls_model.cluster_stats, "Cluster1"
                )
            elif self._cluster_type == 'two_way':
                cluster_diag['dim1'] = self._rls_model.diagnose_cluster_structure(
                    self._rls_model.cluster_stats, "Cluster1"
                )
                cluster_diag['dim2'] = self._rls_model.diagnose_cluster_structure(
                    self._rls_model.cluster2_stats, "Cluster2"
                )
        
        return RegressionResults(
            coefficients=self._rls_model.theta,
            std_errors=se,
            feature_names=self._rls_model.get_feature_names(),
            n_obs=self._rls_model.n_obs,
            n_features=self._rls_model.n_features,
            rss=self._rls_model.rss,
            r_squared=self._rls_model.get_r_squared(),
            adj_r_squared=self._rls_model.get_adjusted_r_squared(),
            t_statistics=t_stats,
            p_values=p_values,
            f_statistic=f_stat,
            f_pvalue=f_pval,
            df_model=df_model,
            df_resid=df_resid,
            model_type='ols',
            cluster_type=self._cluster_type,
            covariance_matrix=cov,
            cluster_diagnostics=cluster_diag
        )
    
    def summary(self) -> pd.DataFrame:
        """
        Get regression summary table.
        
        Returns:
        --------
        DataFrame with coefficients, standard errors, t-stats, and p-values
        """
        if self._results is None:
            raise ValueError("Model must be fitted before calling summary()")
        
        return self._results.summary()
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : DataFrame or ndarray
            Features to predict on
        
        Returns:
        --------
        predictions : ndarray
        """
        if self._rls_model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        if isinstance(X, pd.DataFrame):
            X = X[self._parser.features].values
        
        # Apply feature transformation
        transformer = FeatureTransformer.from_config(
            {'transformations': self._parser.transformations or []},
            self._parser.features,
            add_intercept=self._parser.has_intercept
        )
        X_transformed = transformer.transform(X, self._parser.features)
        
        return self._rls_model.predict(X_transformed)
    
    def save_results(
        self,
        output_dir: str,
        spec_name: Optional[str] = None,
        spec_config: Optional[Dict[str, Any]] = None,
        full_config: Optional[Dict[str, Any]] = None,
        formats: Optional[List[str]] = None
    ) -> Path:
        """
        Save regression results to disk.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save results
        spec_name : str, optional
            Name for this specification (used in subdirectory)
        spec_config : dict, optional
            Specification configuration (for documentation)
        full_config : dict, optional
            Full configuration snapshot
        formats : list of str, optional
            Formats to save. Options: 'summary', 'csv', 'json', 'latex', 'readme', 'diagnostics'
            Default: all formats
        
        Returns:
        --------
        run_dir : Path
            Directory where results were saved
        """
        if self._results is None:
            raise ValueError("Model must be fitted before saving results")
        
        return self._results.save(
            output_dir=output_dir,
            spec_name=spec_name,
            spec_config=spec_config,
            full_config=full_config,
            formats=formats
        )
    
    # Scikit-learn style properties
    @property
    def coef_(self) -> np.ndarray:
        """Coefficient estimates."""
        if self._results is None:
            raise ValueError("Model must be fitted first")
        return self._results.coefficients
    
    @property
    def se_(self) -> np.ndarray:
        """Standard errors."""
        if self._results is None:
            raise ValueError("Model must be fitted first")
        return self._results.std_errors
    
    @property
    def n_obs_(self) -> int:
        """Number of observations."""
        if self._results is None:
            raise ValueError("Model must be fitted first")
        return self._results.n_obs
    
    @property
    def r_squared_(self) -> float:
        """R-squared."""
        if self._results is None:
            raise ValueError("Model must be fitted first")
        return self._results.r_squared
    
    @property
    def results_(self) -> RegressionResults:
        """Full results object."""
        if self._results is None:
            raise ValueError("Model must be fitted first")
        return self._results


class TwoSLS:
    """
    Two-Stage Least Squares estimator with streaming/online capabilities.
    
    Usage:
    ------
    >>> model = TwoSLS(formula="y ~ x1 + x2 | z1 + z2", endogenous=['x1'])
    >>> model.fit(data, cluster='country')
    >>> print(model.summary())
    """
    
    def __init__(
        self,
        formula: str,
        endogenous: Optional[List[str]] = None,
        alpha: float = 1e-3,
        forget_factor: float = 1.0,
        chunk_size: int = 10000,
        n_workers: Optional[int] = None,
        show_progress: bool = True
    ):
        """
        Initialize 2SLS estimator.
        
        Parameters:
        -----------
        formula : str
            R-style formula with instruments (e.g., "y ~ x1 + x2 | z1 + z2")
        endogenous : list of str, optional
            List of endogenous variables. If None, all features are endogenous
        alpha : float
            Regularization parameter
        forget_factor : float
            Forgetting factor
        chunk_size : int
            Chunk size for processing
        n_workers : int, optional
            Number of parallel workers
        show_progress : bool
            Show progress bar
        """
        self.formula = formula
        self.endogenous = endogenous
        self.alpha = alpha
        self.forget_factor = forget_factor
        self.chunk_size = chunk_size
        self.n_workers = n_workers
        self.show_progress = show_progress
        
        # Parse formula
        self._parser = FormulaParser.parse(formula)
        
        if not self._parser.instruments:
            raise ValueError(
                "2SLS requires instruments. Use format: y ~ x1 + x2 | z1 + z2"
            )
        
        # Results (populated after fit)
        self._results = None
        self._twosls_model = None
        self._cluster_type = 'classical'
        self._endog_cols = endogenous or self._parser.features
        self._exog_cols = [f for f in self._parser.features if f not in self._endog_cols]
    
    def fit(
        self,
        data: Union[str, Path, pd.DataFrame, StreamData],
        cluster: Optional[Union[str, List[str]]] = None
    ) -> 'TwoSLS':
        """
        Fit the 2SLS model.
        """
        from gnt.analysis.streamreg.estimators.iv import TwoSLSOrchestrator
        
        # Setup data ONCE
        if not isinstance(data, StreamData):
            data = StreamData(data, chunk_size=self.chunk_size)
        
        # Validate columns
        required_cols = [self._parser.target] + self._parser.features + self._parser.instruments
        if cluster:
            cluster_cols = [cluster] if isinstance(cluster, str) else cluster
            required_cols.extend(cluster_cols)
        data.validate_columns(required_cols)
        
        # Determine cluster type
        self._cluster_type = 'classical'
        cluster1_col = None
        cluster2_col = None
        
        if cluster:
            if isinstance(cluster, str):
                self._cluster_type = 'one_way'
                cluster1_col = cluster
            elif len(cluster) == 2:
                self._cluster_type = 'two_way'
                cluster1_col, cluster2_col = cluster
        
        # Feature engineering config
        feature_engineering = {'endogenous': self._endog_cols}
        if self._parser.transformations:
            feature_engineering['transformations'] = self._parser.transformations
        
        # Fit using orchestrator - pass StreamData object directly
        if data.info.source_type != 'partitioned':
            raise NotImplementedError(
                "2SLS currently only supports partitioned parquet datasets"
            )
        
        # Create orchestrator with StreamData object
        orchestrator = TwoSLSOrchestrator(
            data=data,  # Pass StreamData object, not path
            endog_cols=self._endog_cols,
            exog_cols=self._exog_cols,
            instr_cols=self._parser.instruments,
            target_col=self._parser.target,
            cluster1_col=cluster1_col,
            cluster2_col=cluster2_col,
            add_intercept=self._parser.has_intercept,
            alpha=self.alpha,
            chunk_size=self.chunk_size,
            n_workers=self.n_workers,
            show_progress=self.show_progress,
            verbose=True,
            feature_engineering=feature_engineering
        )
        
        self._twosls_model = orchestrator.fit()
        
        # Store results
        self._results = self._create_results()
        
        return self
    
    def _create_results(self) -> RegressionResults:
        """Create standardized results object."""
        # Convert first stage
        first_stage_results = []
        for fs_model in self._twosls_model.first_stage_models:
            se = fs_model.get_standard_errors(self._cluster_type)
            t_stats = fs_model.theta / se
            
            from scipy import stats
            p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
            
            # Get F-statistic for first stage
            f_stat, df_model, df_resid = fs_model.get_f_statistic()
            f_pval = fs_model.get_f_pvalue()
            
            # Add IV-specific F-statistic if available
            iv_f_stat = fs_model.iv_f_statistic
            iv_f_df = fs_model.iv_f_df
            
            fs_result = RegressionResults(
                coefficients=fs_model.theta,
                std_errors=se,
                feature_names=fs_model.get_feature_names(),
                n_obs=fs_model.n_obs,
                n_features=fs_model.n_features,
                rss=fs_model.rss,
                r_squared=fs_model.get_r_squared(),
                adj_r_squared=fs_model.get_adjusted_r_squared(),
                t_statistics=t_stats,
                p_values=p_values,
                f_statistic=f_stat,
                f_pvalue=f_pval,
                df_model=df_model,
                df_resid=df_resid,
                model_type='first_stage',
                cluster_type=self._cluster_type,
                covariance_matrix=fs_model.get_cluster_robust_covariance(self._cluster_type)
                    if self._cluster_type != 'classical' else fs_model.get_covariance_matrix()
            )
            
            # Add IV F-statistic to metadata
            if iv_f_stat is not None:
                fs_result.metadata['iv_f_statistic'] = iv_f_stat
                fs_result.metadata['iv_f_df'] = iv_f_df
            
            first_stage_results.append(fs_result)
        
        # Convert second stage
        ss_model = self._twosls_model.second_stage
        se = ss_model.get_standard_errors(self._cluster_type)
        t_stats = ss_model.theta / se
        
        from scipy import stats
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
        
        # Get F-statistic for second stage
        f_stat, df_model, df_resid = ss_model.get_f_statistic()
        f_pval = ss_model.get_f_pvalue()
        
        result = RegressionResults(
            coefficients=ss_model.theta,
            std_errors=se,
            feature_names=ss_model.get_feature_names(),
            n_obs=ss_model.n_obs,
            n_features=ss_model.n_features,
            rss=ss_model.rss,
            r_squared=ss_model.get_r_squared(),
            adj_r_squared=ss_model.get_adjusted_r_squared(),
            t_statistics=t_stats,
            p_values=p_values,
            f_statistic=f_stat,
            f_pvalue=f_pval,
            df_model=df_model,
            df_resid=df_resid,
            model_type='2sls',
            cluster_type=self._cluster_type,
            covariance_matrix=ss_model.get_cluster_robust_covariance(self._cluster_type)
                if self._cluster_type != 'classical' else ss_model.get_covariance_matrix(),
            first_stage_results=first_stage_results
        )
        result.metadata['endogenous_variables'] = self._endog_cols
        
        return result
    
    def summary(self, stage: str = 'second') -> pd.DataFrame:
        """
        Get regression summary.
        
        Parameters:
        -----------
        stage : str
            'first', 'second', or 'all'
        
        Returns:
        --------
        DataFrame or dict of DataFrames
        """
        if self._results is None:
            raise ValueError("Model must be fitted before calling summary()")
        
        if stage == 'second':
            return self._results.summary()
        elif stage == 'first':
            return {
                f"First Stage {i+1} ({self._endog_cols[i]})": fs.summary()
                for i, fs in enumerate(self._results.first_stage_results)
            }
        elif stage == 'all':
            return {
                **{f"First Stage {i+1} ({self._endog_cols[i]})": fs.summary()
                   for i, fs in enumerate(self._results.first_stage_results)},
                "Second Stage": self._results.summary()
            }
        else:
            raise ValueError("stage must be 'first', 'second', or 'all'")
    
    def save_results(
        self,
        output_dir: str,
        spec_name: Optional[str] = None,
        spec_config: Optional[Dict[str, Any]] = None,
        full_config: Optional[Dict[str, Any]] = None,
        formats: Optional[List[str]] = None
    ) -> Path:
        """
        Save regression results to disk.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save results
        spec_name : str, optional
            Name for this specification (used in subdirectory)
        spec_config : dict, optional
            Specification configuration (for documentation)
        full_config : dict, optional
            Full configuration snapshot
        formats : list of str, optional
            Formats to save. Options: 'summary', 'csv', 'json', 'latex', 'readme', 'diagnostics'
            Default: all formats
        
        Returns:
        --------
        run_dir : Path
            Directory where results were saved
        """
        if self._results is None:
            raise ValueError("Model must be fitted before saving results")
        
        return self._results.save(
            output_dir=output_dir,
            spec_name=spec_name,
            spec_config=spec_config,
            full_config=full_config,
            formats=formats
        )
    
    @property
    def coef_(self) -> np.ndarray:
        """Second stage coefficient estimates."""
        if self._results is None:
            raise ValueError("Model must be fitted first")
        return self._results.coefficients
    
    @property
    def results_(self) -> RegressionResults:
        """Full results object."""
        if self._results is None:
            raise ValueError("Model must be fitted first")
        return self._results


# Convenience functions for backward compatibility
def ols(formula: str, data: Union[str, Path, pd.DataFrame, StreamData],
        cluster: Optional[Union[str, List[str]]] = None, **kwargs) -> RegressionResults:
    """Convenience function for OLS estimation."""
    model = OLS(formula, **kwargs)
    model.fit(data, cluster=cluster)
    return model.results_


def twosls(formula: str, data: Union[str, Path, pd.DataFrame, StreamData],
           endogenous: Optional[List[str]] = None,
           cluster: Optional[Union[str, List[str]]] = None, **kwargs) -> RegressionResults:
    """Convenience function for 2SLS estimation."""
    model = TwoSLS(formula, endogenous=endogenous, **kwargs)
    model.fit(data, cluster=cluster)
    return model.results_

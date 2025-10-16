import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, List, Literal
from pathlib import Path
import logging

from gnt.analysis.models.data import StreamData
from gnt.analysis.models.results import RegressionResults
from gnt.analysis.models.feature_engineering import FormulaParser, FeatureTransformer
from gnt.analysis.models.online_RLS import OnlineRLS, process_partitioned_dataset_parallel

logger = logging.getLogger(__name__)


def ols(
    formula: str,
    data: Union[str, Path, pd.DataFrame, StreamData],
    cluster: Optional[Union[str, List[str]]] = None,
    alpha: float = 1e-3,
    forget_factor: float = 1.0,
    chunk_size: int = 10000,
    n_workers: Optional[int] = None,
    show_progress: bool = True,
    verbose: bool = True,
    **kwargs
) -> RegressionResults:
    """
    Fit OLS regression using streaming/online algorithm.
    
    Parameters:
    -----------
    formula : str
        R-style formula (e.g., "y ~ x1 + x2 + I(x1^2)")
    data : str, Path, DataFrame, or StreamData
        Data source
    cluster : str or list of str, optional
        Column name(s) for cluster-robust standard errors
        - str: one-way clustering
        - list of 2 strings: two-way clustering
    alpha : float
        Regularization parameter
    forget_factor : float
        Forgetting factor (1.0 = no forgetting)
    chunk_size : int
        Chunk size for processing
    n_workers : int, optional
        Number of parallel workers
    show_progress : bool
        Show progress bar
    verbose : bool
        Verbose output
    
    Returns:
    --------
    RegressionResults
        Fitted model results
    """
    # Parse formula
    parser = FormulaParser.parse(formula)
    
    if parser.instruments:
        raise ValueError(
            "Formula contains instruments. Use twosls() for 2SLS estimation."
        )
    
    # Setup data
    if not isinstance(data, StreamData):
        data = StreamData(data, chunk_size=chunk_size)
    
    # Validate columns
    required_cols = [parser.target] + parser.features
    if cluster:
        cluster_cols = [cluster] if isinstance(cluster, str) else cluster
        required_cols.extend(cluster_cols)
    data.validate_columns(required_cols)
    
    # Determine cluster type
    cluster_type = 'classical'
    cluster1_col = None
    cluster2_col = None
    
    if cluster:
        if isinstance(cluster, str):
            cluster_type = 'one_way'
            cluster1_col = cluster
        elif len(cluster) == 2:
            cluster_type = 'two_way'
            cluster1_col, cluster2_col = cluster
        else:
            raise ValueError("cluster must be string or list of 2 strings")
    
    # Create feature engineering config
    feature_engineering = None
    if parser.transformations:
        feature_engineering = {'transformations': parser.transformations}
    
    # Fit model based on data type
    if data.info.source_type == 'partitioned':
        # Use parallel processing for partitioned data
        rls = process_partitioned_dataset_parallel(
            parquet_path=data.info.source_path,
            feature_cols=parser.features,
            target_col=parser.target,
            cluster1_col=cluster1_col,
            cluster2_col=cluster2_col,
            add_intercept=parser.has_intercept,
            chunk_size=chunk_size,
            n_workers=n_workers,
            alpha=alpha,
            forget_factor=forget_factor,
            show_progress=show_progress,
            verbose=verbose,
            feature_engineering=feature_engineering,
            formula=formula
        )
    else:
        # Sequential processing for single file or DataFrame
        rls = _fit_sequential(
            data, parser, cluster1_col, cluster2_col,
            alpha, forget_factor, feature_engineering
        )
    
    # Convert to standardized results
    return _rls_to_results(rls, cluster_type, 'ols')


def twosls(
    formula: str,
    data: Union[str, Path, pd.DataFrame, StreamData],
    endogenous: Optional[List[str]] = None,
    cluster: Optional[Union[str, List[str]]] = None,
    alpha: float = 1e-3,
    forget_factor: float = 1.0,
    chunk_size: int = 10000,
    n_workers: Optional[int] = None,
    show_progress: bool = True,
    verbose: bool = True,
    **kwargs
) -> RegressionResults:
    """
    Fit 2SLS regression using streaming/online algorithm.
    
    Parameters:
    -----------
    formula : str
        R-style formula with instruments (e.g., "y ~ x1 + x2 | z1 + z2")
        Format: target ~ features | instruments
    data : str, Path, DataFrame, or StreamData
        Data source
    endogenous : list of str, optional
        List of endogenous variables from features
        If not specified, all features are treated as endogenous
    cluster : str or list of str, optional
        Column name(s) for cluster-robust standard errors
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
    verbose : bool
        Verbose output
    
    Returns:
    --------
    RegressionResults
        Fitted model results with first stage results included
    """
    from gnt.analysis.models.online_2SLS import process_partitioned_dataset_2sls
    
    # Parse formula
    parser = FormulaParser.parse(formula)
    
    if not parser.instruments:
        raise ValueError(
            "2SLS requires instruments. Use format: y ~ x1 + x2 | z1 + z2"
        )
    
    # Setup data
    if not isinstance(data, StreamData):
        data = StreamData(data, chunk_size=chunk_size)
    
    # Determine endogenous/exogenous split
    if endogenous is None:
        endog_cols = parser.features
        exog_cols = []
    else:
        endog_cols = endogenous
        exog_cols = [f for f in parser.features if f not in endogenous]
    
    # Validate columns
    required_cols = [parser.target] + parser.features + parser.instruments
    if cluster:
        cluster_cols = [cluster] if isinstance(cluster, str) else cluster
        required_cols.extend(cluster_cols)
    data.validate_columns(required_cols)
    
    # Determine cluster type
    cluster_type = 'classical'
    cluster1_col = None
    cluster2_col = None
    
    if cluster:
        if isinstance(cluster, str):
            cluster_type = 'one_way'
            cluster1_col = cluster
        elif len(cluster) == 2:
            cluster_type = 'two_way'
            cluster1_col, cluster2_col = cluster
    
    # Create feature engineering config
    feature_engineering = {'endogenous': endog_cols}
    if parser.transformations:
        feature_engineering['transformations'] = parser.transformations
    
    # Fit model (requires partitioned data for now)
    if data.info.source_type != 'partitioned':
        raise NotImplementedError(
            "2SLS currently only supports partitioned parquet datasets"
        )
    
    twosls_model = process_partitioned_dataset_2sls(
        parquet_path=data.info.source_path,
        endog_cols=endog_cols,
        exog_cols=exog_cols,
        instr_cols=parser.instruments,
        target_col=parser.target,
        cluster1_col=cluster1_col,
        cluster2_col=cluster2_col,
        add_intercept=parser.has_intercept,
        chunk_size=chunk_size,
        n_workers=n_workers,
        alpha=alpha,
        forget_factor=forget_factor,
        show_progress=show_progress,
        verbose=verbose,
        feature_engineering=feature_engineering,
        formula=formula
    )
    
    # Convert to standardized results
    return _twosls_to_results(twosls_model, cluster_type, endog_cols)


def _fit_sequential(
    data: StreamData,
    parser: FormulaParser,
    cluster1_col: Optional[str],
    cluster2_col: Optional[str],
    alpha: float,
    forget_factor: float,
    feature_engineering: Optional[Dict]
) -> OnlineRLS:
    """Fit model sequentially for non-partitioned data."""
    # Create feature transformer
    transformer = FeatureTransformer.from_config(
        feature_engineering or {'transformations': []},
        parser.features,
        add_intercept=parser.has_intercept
    )
    
    n_features = transformer.get_n_features()
    feature_names = transformer.get_feature_names()
    
    # Initialize RLS
    rls = OnlineRLS(
        n_features=n_features,
        alpha=alpha,
        forget_factor=forget_factor,
        feature_names=feature_names
    )
    
    # Required columns for loading
    load_cols = parser.features + [parser.target]
    if cluster1_col:
        load_cols.append(cluster1_col)
    if cluster2_col:
        load_cols.append(cluster2_col)
    
    # Process chunks
    for chunk in data.iter_chunks(columns=load_cols):
        X = chunk[parser.features].values
        y = chunk[parser.target].values
        
        # Apply feature transformation
        X_transformed = transformer.transform(X, parser.features)
        
        # Get cluster variables
        c1 = chunk[cluster1_col].values if cluster1_col else None
        c2 = chunk[cluster2_col].values if cluster2_col else None
        
        # Fit
        rls.partial_fit(X_transformed, y, c1, c2)
    
    return rls


def _rls_to_results(
    rls: OnlineRLS,
    cluster_type: str,
    model_type: str
) -> RegressionResults:
    """Convert OnlineRLS to RegressionResults."""
    # Get standard errors
    se = rls.get_standard_errors(cluster_type)
    
    # Compute statistics
    t_stats = rls.theta / se
    from scipy import stats
    p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
    
    # Get covariance matrix
    if cluster_type == 'classical':
        cov = rls.get_covariance_matrix()
    else:
        cov = rls.get_cluster_robust_covariance(cluster_type)
    
    # Get cluster diagnostics if applicable
    cluster_diag = None
    if cluster_type != 'classical':
        cluster_diag = {}
        if cluster_type == 'one_way':
            cluster_diag['dim1'] = rls.diagnose_cluster_structure(
                rls.cluster_stats, "Cluster1"
            )
        elif cluster_type == 'two_way':
            cluster_diag['dim1'] = rls.diagnose_cluster_structure(
                rls.cluster_stats, "Cluster1"
            )
            cluster_diag['dim2'] = rls.diagnose_cluster_structure(
                rls.cluster2_stats, "Cluster2"
            )
    
    return RegressionResults(
        coefficients=rls.theta,
        std_errors=se,
        feature_names=rls.get_feature_names(),
        n_obs=rls.n_obs,
        n_features=rls.n_features,
        rss=rls.rss,
        r_squared=rls.get_r_squared(),
        adj_r_squared=rls.get_adjusted_r_squared(),
        t_statistics=t_stats,
        p_values=p_values,
        model_type=model_type,
        cluster_type=cluster_type,
        covariance_matrix=cov,
        cluster_diagnostics=cluster_diag
    )


def _twosls_to_results(
    twosls_model,
    cluster_type: str,
    endog_cols: List[str]
) -> RegressionResults:
    """Convert Online2SLS to RegressionResults."""
    # Convert first stage results
    first_stage_results = []
    for fs_model in twosls_model.first_stage_models:
        first_stage_results.append(_rls_to_results(fs_model, cluster_type, 'first_stage'))
    
    # Convert second stage
    second_stage = _rls_to_results(twosls_model.second_stage, cluster_type, '2sls')
    second_stage.first_stage_results = first_stage_results
    second_stage.metadata['endogenous_variables'] = endog_cols
    
    return second_stage

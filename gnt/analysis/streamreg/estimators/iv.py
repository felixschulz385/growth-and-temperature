import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path

from gnt.analysis.streamreg.estimators.ols import (
    OnlineRLS,
    ParallelOLSOrchestrator,
    DEFAULT_ALPHA,
    DEFAULT_CHUNK_SIZE
)

logger = logging.getLogger(__name__)


class Online2SLS:
    """
    Online Two-Stage Least Squares using composition of OnlineRLS instances.
    
    This class orchestrates two-stage estimation:
    - First stage: Regress each endogenous variable on instruments + exogenous
    - Second stage: Regress outcome on fitted endogenous + exogenous
    """
    
    def __init__(
        self, 
        n_endogenous: int, 
        n_exogenous: int, 
        n_instruments: int,
        add_intercept: bool = True,
        alpha: float = DEFAULT_ALPHA,
        forget_factor: float = 1.0,
        batch_size: int = 1000,
        endog_names: Optional[List[str]] = None,
        exog_names: Optional[List[str]] = None,
        instr_names: Optional[List[str]] = None
    ):
        """Initialize Online 2SLS with composition of OLS models."""
        self.n_endogenous = n_endogenous
        self.n_exogenous = n_exogenous
        self.n_instruments = n_instruments
        self.add_intercept = add_intercept
        self.alpha = alpha
        self.forget_factor = forget_factor
        self.batch_size = batch_size
        
        # Store variable names
        self.endog_names = endog_names or [f"endog_{i}" for i in range(n_endogenous)]
        self.exog_names = exog_names or [f"exog_{i}" for i in range(n_exogenous)]
        self.instr_names = instr_names or [f"instr_{i}" for i in range(n_instruments)]
        
        # Calculate dimensions
        self.first_stage_dims = n_exogenous + n_instruments + (1 if add_intercept else 0)
        self.second_stage_dims = n_endogenous + n_exogenous + (1 if add_intercept else 0)
        
        # First stage: one model per endogenous variable
        first_stage_feature_names = self._get_first_stage_feature_names()
        self.first_stage_models = [
            OnlineRLS(
                n_features=self.first_stage_dims,
                alpha=alpha,
                forget_factor=forget_factor,
                batch_size=batch_size,
                feature_names=first_stage_feature_names
            )
            for _ in range(n_endogenous)
        ]
        
        # Second stage: single model
        second_stage_feature_names = self._get_second_stage_feature_names()
        self.second_stage = OnlineRLS(
            n_features=self.second_stage_dims,
            alpha=alpha,
            forget_factor=forget_factor,
            batch_size=batch_size,
            feature_names=second_stage_feature_names
        )
        
        self.total_obs = 0
    
    def _get_first_stage_feature_names(self) -> List[str]:
        """Get feature names for first stage."""
        names = []
        if self.add_intercept:
            names.append("intercept")
        names.extend(self.exog_names)
        names.extend(self.instr_names)
        return names
    
    def _get_second_stage_feature_names(self) -> List[str]:
        """Get feature names for second stage."""
        names = []
        if self.add_intercept:
            names.append("intercept")
        names.extend([f"{name}_hat" for name in self.endog_names])
        names.extend(self.exog_names)
        return names
    
    def partial_fit(
        self, 
        X_endog: np.ndarray,
        X_exog: np.ndarray,
        Z: np.ndarray,
        y: np.ndarray,
        cluster1: Optional[np.ndarray] = None,
        cluster2: Optional[np.ndarray] = None
    ) -> 'Online2SLS':
        """Update 2SLS estimates with new batch of data."""
        # Validate and clean
        X_endog, X_exog, Z, y, cluster1, cluster2 = self._validate_data(
            X_endog, X_exog, Z, y, cluster1, cluster2
        )
        
        if len(y) == 0:
            return self
        
        self.total_obs += len(y)
        
        # Build first stage features: exogenous + instruments
        first_stage_X = self._build_first_stage_features(X_exog, Z)
        
        # First stage: fit each endogenous variable
        X_endog_hat = np.zeros_like(X_endog)
        for i in range(self.n_endogenous):
            self.first_stage_models[i].partial_fit(
                first_stage_X, X_endog[:, i], cluster1, cluster2
            )
            X_endog_hat[:, i] = self.first_stage_models[i].predict(first_stage_X)
        
        # Build second stage features: fitted endogenous + exogenous
        second_stage_X = self._build_second_stage_features(X_endog_hat, X_exog)
        
        # Second stage
        self.second_stage.partial_fit(second_stage_X, y, cluster1, cluster2)
        
        return self
    
    def _validate_data(self, X_endog, X_exog, Z, y, cluster1, cluster2) -> Tuple:
        """Validate and clean input data."""
        X_endog = np.atleast_2d(X_endog)
        X_exog = np.atleast_2d(X_exog) if X_exog.size > 0 else np.empty((len(y), 0))
        Z = np.atleast_2d(Z)
        y = np.atleast_1d(y)
        
        # Validate dimensions
        if X_endog.shape[1] != self.n_endogenous:
            raise ValueError(f"Expected {self.n_endogenous} endogenous, got {X_endog.shape[1]}")
        if X_exog.shape[1] != self.n_exogenous:
            raise ValueError(f"Expected {self.n_exogenous} exogenous, got {X_exog.shape[1]}")
        if Z.shape[1] != self.n_instruments:
            raise ValueError(f"Expected {self.n_instruments} instruments, got {Z.shape[1]}")
        
        # Filter invalid observations
        valid_mask = (
            np.isfinite(X_endog).all(axis=1) & 
            np.isfinite(X_exog).all(axis=1) & 
            np.isfinite(Z).all(axis=1) & 
            np.isfinite(y)
        )
        
        if not valid_mask.all():
            logger.debug(f"Removing {(~valid_mask).sum()}/{len(valid_mask)} invalid observations")
            X_endog = X_endog[valid_mask]
            X_exog = X_exog[valid_mask]
            Z = Z[valid_mask]
            y = y[valid_mask]
            cluster1 = cluster1[valid_mask] if cluster1 is not None else None
            cluster2 = cluster2[valid_mask] if cluster2 is not None else None
        
        return X_endog, X_exog, Z, y, cluster1, cluster2
    
    def _build_first_stage_features(self, X_exog: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Build first stage feature matrix."""
        features = np.column_stack([X_exog, Z]) if X_exog.size > 0 else Z
        if self.add_intercept:
            intercept = np.ones((features.shape[0], 1), dtype=features.dtype)
            features = np.column_stack([intercept, features])
        return features
    
    def _build_second_stage_features(self, X_endog_hat: np.ndarray, X_exog: np.ndarray) -> np.ndarray:
        """Build second stage feature matrix."""
        features = np.column_stack([X_endog_hat, X_exog]) if X_exog.size > 0 else X_endog_hat
        if self.add_intercept:
            intercept = np.ones((features.shape[0], 1), dtype=features.dtype)
            features = np.column_stack([intercept, features])
        return features
    
    def predict(self, X_exog: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Make predictions using 2SLS model."""
        first_stage_X = self._build_first_stage_features(X_exog, Z)
        
        # Predict endogenous variables
        X_endog_hat = np.zeros((Z.shape[0], self.n_endogenous))
        for i in range(self.n_endogenous):
            X_endog_hat[:, i] = self.first_stage_models[i].predict(first_stage_X)
        
        # Predict outcome
        second_stage_X = self._build_second_stage_features(X_endog_hat, X_exog)
        return self.second_stage.predict(second_stage_X)
    
    def get_first_stage_summary(self) -> List[pd.DataFrame]:
        """Get summaries of first stage regressions."""
        return [model.summary(cluster_type='two_way') for model in self.first_stage_models]
    
    def get_second_stage_summary(self) -> pd.DataFrame:
        """Get summary of second stage regression."""
        return self.second_stage.summary(cluster_type='two_way')


class TwoSLSOrchestrator:
    """
    Orchestrates two-pass 2SLS estimation using ParallelOLSOrchestrator.
    
    This separates orchestration logic from estimation logic.
    """
    
    def __init__(
        self,
        data,
        endog_cols: List[str],
        exog_cols: List[str],
        instr_cols: List[str],
        target_col: str,
        cluster1_col: Optional[str] = None,
        cluster2_col: Optional[str] = None,
        add_intercept: bool = True,
        alpha: float = DEFAULT_ALPHA,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        n_workers: Optional[int] = None,
        show_progress: bool = True,
        verbose: bool = True,
        feature_engineering: Optional[Dict[str, Any]] = None
    ):
        """Initialize 2SLS orchestrator."""
        self.data = data
        self.endog_cols = endog_cols
        self.exog_cols = exog_cols
        self.instr_cols = instr_cols
        self.target_col = target_col
        self.cluster1_col = cluster1_col
        self.cluster2_col = cluster2_col
        self.add_intercept = add_intercept
        self.alpha = alpha
        self.chunk_size = chunk_size
        self.n_workers = n_workers
        self.show_progress = show_progress
        self.verbose = verbose
        self.feature_engineering = feature_engineering
    
    def fit(self) -> Online2SLS:
        """Execute two-pass 2SLS estimation."""
        logger.info("Starting 2SLS two-pass estimation")
        
        # PASS 1: Estimate first stage models
        logger.info("PASS 1: Estimating first stage regressions")
        first_stage_models = self._fit_first_stage()
        
        # PASS 2: Estimate second stage with predicted values
        logger.info("PASS 2: Estimating second stage regression")
        second_stage_model = self._fit_second_stage(first_stage_models)
        
        # Construct final 2SLS model
        return self._build_2sls_model(first_stage_models, second_stage_model)
    
    def _fit_first_stage(self) -> List[OnlineRLS]:
        """Fit first stage models (one per endogenous variable)."""
        first_stage_models = []
        first_stage_features = self.exog_cols + self.instr_cols
        
        # Extract first stage feature engineering
        first_stage_fe = self._get_first_stage_fe_config()
        
        # Compute n_features for first stage
        from gnt.analysis.streamreg.transforms import FeatureTransformer
        
        if first_stage_fe or self.add_intercept:
            transformer = FeatureTransformer.from_config(
                first_stage_fe or {'transformations': []},
                first_stage_features,
                add_intercept=self.add_intercept
            )
            first_stage_n_features = transformer.get_n_features()
            first_stage_feature_names = transformer.get_feature_names()
        else:
            first_stage_n_features = len(first_stage_features)
            first_stage_feature_names = first_stage_features.copy()
        
        for i, endog_var in enumerate(self.endog_cols):
            logger.info(f"First stage {i+1}/{len(self.endog_cols)}: {endog_var}")
            
            # Create orchestrator with computed dimensions
            orchestrator = ParallelOLSOrchestrator(
                data=self.data,
                feature_cols=first_stage_features,
                target_col=endog_var,
                cluster1_col=self.cluster1_col,
                cluster2_col=self.cluster2_col,
                add_intercept=self.add_intercept,
                n_features=first_stage_n_features,  # Now properly set
                transformed_feature_names=first_stage_feature_names,  # Now properly set
                alpha=self.alpha,
                chunk_size=self.chunk_size,
                n_workers=self.n_workers,
                show_progress=self.show_progress,
                verbose=self.verbose,
                feature_engineering=first_stage_fe
            )
            
            first_stage_rls = orchestrator.fit()
            first_stage_models.append(first_stage_rls)
            
            logger.info(f"First stage {i+1} complete: R²={first_stage_rls.get_r_squared():.4f}")
        
        return first_stage_models
    
    def _fit_second_stage(self, first_stage_models: List[OnlineRLS]) -> OnlineRLS:
        """Fit second stage model using first stage predictions."""
        second_stage_features = self.endog_cols + self.exog_cols
        second_stage_fe = self._get_second_stage_fe_config(first_stage_models)
        
        # Compute n_features for second stage
        from gnt.analysis.streamreg.transforms import FeatureTransformer
        
        if second_stage_fe or self.add_intercept:
            transformer = FeatureTransformer.from_config(
                second_stage_fe or {'transformations': []},
                second_stage_features,
                add_intercept=self.add_intercept
            )
            second_stage_n_features = transformer.get_n_features()
            second_stage_feature_names = transformer.get_feature_names()
        else:
            second_stage_n_features = len(second_stage_features)
            second_stage_feature_names = second_stage_features.copy()
        
        # Create orchestrator with computed dimensions
        orchestrator = ParallelOLSOrchestrator(
            data=self.data,
            feature_cols=second_stage_features,
            target_col=self.target_col,
            cluster1_col=self.cluster1_col,
            cluster2_col=self.cluster2_col,
            add_intercept=self.add_intercept,
            n_features=second_stage_n_features,  # Now properly set
            transformed_feature_names=second_stage_feature_names,  # Now properly set
            alpha=self.alpha,
            chunk_size=self.chunk_size,
            n_workers=self.n_workers,
            show_progress=self.show_progress,
            verbose=self.verbose,
            feature_engineering=second_stage_fe
        )
        
        return orchestrator.fit()

    def _get_first_stage_fe_config(self) -> Optional[Dict]:
        """Extract feature engineering config for first stage."""
        if not self.feature_engineering:
            return None
        
        # Exclude predicted substitution transformations
        transformations = [
            t for t in self.feature_engineering.get('transformations', [])
            if t.get('type') != 'predicted_substitution'
        ]
        
        return {'transformations': transformations} if transformations else None
    
    def _get_second_stage_fe_config(self, first_stage_models: List[OnlineRLS]) -> Dict:
        """Build feature engineering config for second stage with predicted substitutions."""
        config = {'transformations': []}
        
        # Add predicted substitution for each endogenous variable
        first_stage_features = self.exog_cols + self.instr_cols
        first_stage_fe = self._get_first_stage_fe_config()
        
        for i, endog_var in enumerate(self.endog_cols):
            config['transformations'].append({
                'type': 'predicted_substitution',
                'original': endog_var,
                'predicted': f"{endog_var}_hat",
                'first_stage_coefficients': first_stage_models[i].theta.tolist(),
                'first_stage_feature_config': first_stage_fe,
                'first_stage_feature_names': first_stage_features,
                'add_intercept_first_stage': self.add_intercept
            })
        
        # Add other transformations from original config
        if self.feature_engineering:
            for t in self.feature_engineering.get('transformations', []):
                if t.get('type') != 'predicted_substitution':
                    config['transformations'].append(t)
        
        # Mark that we need to read instruments
        config['_extra_input_columns'] = self.instr_cols
        config['_base_feature_count'] = len(self.endog_cols) + len(self.exog_cols)
        
        return config
    
    def _build_2sls_model(self, first_stage_models: List[OnlineRLS], 
                         second_stage_model: OnlineRLS) -> Online2SLS:
        """Construct final Online2SLS model from components."""
        model = Online2SLS(
            n_endogenous=len(self.endog_cols),
            n_exogenous=len(self.exog_cols),
            n_instruments=len(self.instr_cols),
            add_intercept=self.add_intercept,
            alpha=self.alpha,
            endog_names=self.endog_cols,
            exog_names=self.exog_cols,
            instr_names=self.instr_cols
        )
        
        # Inject fitted models
        model.first_stage_models = first_stage_models
        model.second_stage = second_stage_model
        model.total_obs = second_stage_model.n_obs
        
        # Calculate first-stage instrument F-statistics
        for i, fs_model in enumerate(first_stage_models):
            # Find instrument indices in the first stage feature names
            fs_features = fs_model.get_feature_names()
            instr_indices = []
            instr_features = []
            for j, name in enumerate(fs_features):
                # Check if this feature is an instrument (not intercept or exogenous)
                if name != 'intercept' and name in self.instr_cols:
                    instr_indices.append(j)
                    instr_features.append(name)
            
            if instr_indices:
                # Get instrument coefficients and their covariance submatrix
                instr_coefs = fs_model.theta[instr_indices]
                
                # Get covariance matrix
                if self.cluster1_col or self.cluster2_col:
                    cluster_type = 'two_way' if self.cluster1_col and self.cluster2_col else 'one_way'
                    cov_matrix = fs_model.get_cluster_robust_covariance(cluster_type)
                else:
                    cov_matrix = fs_model.get_covariance_matrix()
                
                instr_cov = cov_matrix[np.ix_(instr_indices, instr_indices)]
                
                # Calculate first-stage F-statistic for instruments only
                try:
                    # F = (β'Σ^(-1)β)/k where k is number of instruments
                    instr_cov_inv = np.linalg.inv(instr_cov)
                    f_stat_iv = float(instr_coefs @ instr_cov_inv @ instr_coefs) / len(instr_indices)
                    
                    # Store the IV F-statistic on the model
                    fs_model.iv_f_statistic = f_stat_iv
                    fs_model.iv_f_df = (len(instr_indices), fs_model.n_obs - fs_model.n_features)
                except:
                    logger.warning(f"Could not calculate IV F-statistic for first stage {i+1}")
                    fs_model.iv_f_statistic = None
                    fs_model.iv_f_df = None
        
        return model


def process_partitioned_dataset_2sls(
    parquet_path: Union[str, Path],
    endog_cols: List[str],
    exog_cols: List[str],
    instr_cols: List[str],
    target_col: str,
    cluster1_col: Optional[str] = None,
    cluster2_col: Optional[str] = None,
    add_intercept: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    n_workers: Optional[int] = None,
    alpha: float = DEFAULT_ALPHA,
    forget_factor: float = 1.0,
    show_progress: bool = True,
    verbose: bool = True,
    feature_engineering: Optional[Dict[str, Any]] = None,
    formula: Optional[str] = None
) -> Online2SLS:
    """
    Process partitioned parquet dataset for 2SLS estimation.
    
    NOTE: This function is kept for backward compatibility.
    New code should use TwoSLSOrchestrator directly with a StreamData object.
    """
    from gnt.analysis.streamreg.data import StreamData
    
    # Parse formula if provided
    if formula is not None:
        from gnt.analysis.streamreg.formula import FormulaParser
        
        logger.debug(f"Parsing 2SLS formula: {formula}")
        parser = FormulaParser.parse(formula)
        
        if not parser.instruments:
            raise ValueError("2SLS formula must contain instruments after '|'")
        
        target_col = parser.target
        instr_cols = parser.instruments
        add_intercept = parser.has_intercept
        
        # Determine endogenous/exogenous split
        if feature_engineering and 'endogenous' in feature_engineering:
            endog_cols = feature_engineering['endogenous']
            exog_cols = [f for f in parser.features if f not in endog_cols]
        else:
            endog_cols = parser.features
            exog_cols = []
        
        if parser.transformations:
            if feature_engineering is None:
                feature_engineering = {}
            feature_engineering['transformations'] = parser.transformations
    
    # Load data
    data = StreamData(parquet_path, chunk_size=chunk_size)
    
    # Validate columns
    required_cols = [target_col] + endog_cols + exog_cols + instr_cols
    if cluster1_col:
        required_cols.append(cluster1_col)
    if cluster2_col:
        required_cols.append(cluster2_col)
    data.validate_columns(required_cols)
    
    # Create and run orchestrator
    orchestrator = TwoSLSOrchestrator(
        data=data,
        endog_cols=endog_cols,
        exog_cols=exog_cols,
        instr_cols=instr_cols,
        target_col=target_col,
        cluster1_col=cluster1_col,
        cluster2_col=cluster2_col,
        add_intercept=add_intercept,
        alpha=alpha,
        chunk_size=chunk_size,
        n_workers=n_workers,
        show_progress=show_progress,
        verbose=verbose,
        feature_engineering=feature_engineering
    )
    
    return orchestrator.fit()

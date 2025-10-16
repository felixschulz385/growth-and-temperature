import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class RegressionResults:
    """
    Standardized regression results container.
    
    Works for OLS, 2SLS, and other regression types.
    """
    # Core results
    coefficients: np.ndarray
    std_errors: np.ndarray
    feature_names: List[str]
    
    # Model fit
    n_obs: int
    n_features: int
    rss: float
    r_squared: float
    adj_r_squared: float
    
    # Inference
    t_statistics: np.ndarray
    p_values: np.ndarray
    
    # Metadata
    model_type: str  # 'ols', '2sls', etc.
    cluster_type: str = 'classical'
    
    # Optional: stage-specific results for 2SLS
    first_stage_results: Optional[List['RegressionResults']] = None
    
    # Optional: covariance matrix
    covariance_matrix: Optional[np.ndarray] = None
    
    # Optional: cluster diagnostics
    cluster_diagnostics: Optional[Dict[str, Any]] = None
    
    # Optional: additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> pd.DataFrame:
        """Get summary table of results."""
        df = pd.DataFrame({
            'coefficient': self.coefficients,
            'std_error': self.std_errors,
            't_statistic': self.t_statistics,
            'p_value': self.p_values
        }, index=self.feature_names)
        
        # Add significance stars
        df['sig'] = df['p_value'].apply(
            lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
        )
        
        return df
    
    def get_coefficient(self, feature_name: str) -> Dict[str, float]:
        """Get coefficient and statistics for a specific feature."""
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found in results")
        
        idx = self.feature_names.index(feature_name)
        return {
            'coefficient': float(self.coefficients[idx]),
            'std_error': float(self.std_errors[idx]),
            't_statistic': float(self.t_statistics[idx]),
            'p_value': float(self.p_values[idx])
        }
    
    def get_confidence_interval(
        self, 
        feature_name: str, 
        alpha: float = 0.05
    ) -> tuple[float, float]:
        """Get confidence interval for a coefficient."""
        from scipy import stats
        
        idx = self.feature_names.index(feature_name)
        coef = self.coefficients[idx]
        se = self.std_errors[idx]
        
        # Critical value from normal distribution
        z_crit = stats.norm.ppf(1 - alpha/2)
        
        return (coef - z_crit * se, coef + z_crit * se)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        result = {
            'model_type': self.model_type,
            'cluster_type': self.cluster_type,
            'n_obs': int(self.n_obs),
            'n_features': int(self.n_features),
            'r_squared': float(self.r_squared),
            'adj_r_squared': float(self.adj_r_squared),
            'rss': float(self.rss),
            'rmse': float(np.sqrt(self.rss / self.n_obs)) if self.n_obs > 0 else 0.0,
            'coefficients': {
                name: {
                    'estimate': float(self.coefficients[i]),
                    'std_error': float(self.std_errors[i]),
                    't_statistic': float(self.t_statistics[i]),
                    'p_value': float(self.p_values[i])
                }
                for i, name in enumerate(self.feature_names)
            }
        }
        
        if self.first_stage_results:
            result['first_stage'] = [
                fs.to_dict() for fs in self.first_stage_results
            ]
        
        if self.cluster_diagnostics:
            result['cluster_diagnostics'] = self.cluster_diagnostics
        
        if self.metadata:
            result['metadata'] = self.metadata
        
        return result
    
    def __repr__(self) -> str:
        return (f"RegressionResults(model={self.model_type}, "
                f"n_obs={self.n_obs:,}, "
                f"n_features={self.n_features}, "
                f"r_squared={self.r_squared:.4f})")

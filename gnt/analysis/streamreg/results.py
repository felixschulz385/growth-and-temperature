import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class RegressionResults:
    """
    Standardized regression results container with built-in formatting and saving.
    
    Works for OLS, 2SLS, and other regression types.
    """
    # Core results - REQUIRED, no defaults
    coefficients: np.ndarray
    std_errors: np.ndarray
    feature_names: List[str]
    
    # Model fit - REQUIRED, no defaults
    n_obs: int
    n_features: int
    rss: float
    r_squared: float
    adj_r_squared: float
    
    # Inference - REQUIRED, no defaults
    t_statistics: np.ndarray
    p_values: np.ndarray
    
    # Metadata - REQUIRED, no defaults
    model_type: str  # 'ols', '2sls', etc.
    
    # Optional fields with defaults come AFTER required fields
    cluster_type: str = 'classical'
    
    # F-test for overall significance - all optional
    f_statistic: Optional[float] = None
    f_pvalue: Optional[float] = None
    df_model: Optional[int] = None
    df_resid: Optional[int] = None
    
    # Optional: stage-specific results for 2SLS
    first_stage_results: Optional[List['RegressionResults']] = None
    
    # Optional: covariance matrix
    covariance_matrix: Optional[np.ndarray] = None
    
    # Optional: cluster diagnostics
    cluster_diagnostics: Optional[Dict[str, Any]] = None
    
    # Optional: additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def save(
        self,
        output_dir: str,
        spec_name: Optional[str] = None,
        spec_config: Optional[Dict[str, Any]] = None,
        full_config: Optional[Dict[str, Any]] = None,
        formats: Optional[List[str]] = None
    ) -> Path:
        """
        Save results to disk in multiple formats.
        
        Parameters:
        -----------
        output_dir : str
            Base directory for outputs
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
        
        Usage:
        ------
        >>> results.save("output/my_analysis", spec_name="baseline")
        """
        from gnt.analysis.streamreg.output import (
            JSONFormatter, SummaryFormatter, CSVFormatter, LaTeXFormatter,
            DiagnosticsFormatter, READMEFormatter, ConfigFormatter
        )
        
        # Create timestamped run directory
        output_path = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if spec_name:
            run_dir = output_path / f"{spec_name}_{timestamp}"
        else:
            run_dir = output_path / timestamp
        
        run_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to: {run_dir}")
        
        # Default formats
        spec_config = spec_config or {}
        formats = formats or ['summary', 'csv', 'json', 'latex', 'readme', 'diagnostics']
        
        # Save each requested format
        if 'json' in formats:
            self.save_json(run_dir)
        
        if 'summary' in formats:
            self.save_summary(run_dir, spec_config)
        
        if 'csv' in formats:
            self.save_csv(run_dir)
        
        if 'latex' in formats:
            self.save_latex(run_dir)
        
        if 'diagnostics' in formats:
            self.save_diagnostics(run_dir, spec_config)
        
        if 'readme' in formats:
            self.save_readme(run_dir, spec_config, timestamp)
        
        # Save config snapshot if provided
        if full_config:
            ConfigFormatter.save(spec_config, full_config, run_dir, timestamp)
        
        logger.info(f"✓ All results saved to: {run_dir}")
        
        return run_dir
    
    def save_json(self, run_dir: Path) -> None:
        """Save results as JSON."""
        from gnt.analysis.streamreg.output import JSONFormatter
        JSONFormatter.format_and_save(self, run_dir)
    
    def save_summary(self, run_dir: Path, spec_config: Optional[Dict[str, Any]] = None) -> None:
        """Save formatted summary report."""
        from gnt.analysis.streamreg.output import SummaryFormatter
        SummaryFormatter.format_and_save(self, run_dir, spec_config or {})
    
    def save_csv(self, run_dir: Path) -> None:
        """Save coefficient table as CSV."""
        from gnt.analysis.streamreg.output import CSVFormatter
        CSVFormatter.format_and_save(self, run_dir)
    
    def save_latex(self, run_dir: Path) -> None:
        """Save LaTeX-formatted table."""
        from gnt.analysis.streamreg.output import LaTeXFormatter
        LaTeXFormatter.format_and_save(self, run_dir)
    
    def save_diagnostics(self, run_dir: Path, spec_config: Optional[Dict[str, Any]] = None) -> None:
        """Save detailed diagnostics."""
        from gnt.analysis.streamreg.output import DiagnosticsFormatter
        DiagnosticsFormatter.format_and_save(self, run_dir, spec_config or {})
    
    def save_readme(self, run_dir: Path, spec_config: Optional[Dict[str, Any]] = None, 
                    timestamp: Optional[str] = None) -> None:
        """Save README documentation."""
        from gnt.analysis.streamreg.output import READMEFormatter
        timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        READMEFormatter.format_and_save(self, run_dir, spec_config or {}, timestamp)
    
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
        }
        
        # Add F-statistic if available
        if self.f_statistic is not None:
            result['f_statistic'] = float(self.f_statistic)
            result['f_pvalue'] = float(self.f_pvalue) if self.f_pvalue is not None else None
            result['df_model'] = int(self.df_model) if self.df_model is not None else None
            result['df_resid'] = int(self.df_resid) if self.df_resid is not None else None
        
        result['coefficients'] = {
            name: {
                'estimate': float(self.coefficients[i]),
                'std_error': float(self.std_errors[i]),
                't_statistic': float(self.t_statistics[i]),
                'p_value': float(self.p_values[i])
            }
            for i, name in enumerate(self.feature_names)
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

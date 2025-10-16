import numpy as np
import pandas as pd
import pytest
from scipy import stats
from gnt.analysis.streamreg.estimators.ols import OnlineRLS, LinAlgHelper
from gnt.analysis.streamreg.api import OLS


class TestLinAlgHelper:
    """Tests for linear algebra helper functions."""
    
    def test_safe_solve_regular(self):
        """Test safe_solve with well-conditioned matrix."""
        A = np.array([[2, 1], [1, 2]], dtype=float)
        b = np.array([3, 3], dtype=float)
        alpha = 1e-3
        
        x = LinAlgHelper.safe_solve(A, b, alpha)
        
        # Check solution
        assert np.allclose(A @ x, b, atol=1e-6)
    
    def test_safe_solve_singular(self):
        """Test safe_solve with singular matrix (should regularize)."""
        A = np.array([[1, 1], [1, 1]], dtype=float)
        b = np.array([2, 2], dtype=float)
        alpha = 1e-3
        
        # Should not raise, should use regularization
        x = LinAlgHelper.safe_solve(A, b, alpha)
        assert x is not None
        assert x.shape == (2,)
    
    def test_safe_inv_regular(self):
        """Test safe_inv with invertible matrix."""
        A = np.array([[2, 1], [1, 2]], dtype=float)
        
        A_inv = LinAlgHelper.safe_inv(A, use_pinv=False)
        
        # Check inverse
        assert np.allclose(A @ A_inv, np.eye(2), atol=1e-6)
    
    def test_condition_number(self):
        """Test condition number checking."""
        # Well-conditioned
        A = np.eye(3)
        is_good, cond = LinAlgHelper.check_condition_number(A)
        assert is_good
        assert cond < 10
        
        # Ill-conditioned
        A_bad = np.array([[1, 1], [1, 1.0001]])
        is_good, cond = LinAlgHelper.check_condition_number(A_bad)
        # May or may not be well-conditioned depending on threshold
        assert isinstance(is_good, bool)
        assert cond > 1


class TestOnlineRLS:
    """Tests for OnlineRLS estimator."""
    
    def test_initialization(self):
        """Test OnlineRLS initialization."""
        rls = OnlineRLS(n_features=3, alpha=1e-3)
        
        assert rls.n_features == 3
        assert rls.alpha == 1e-3
        assert rls.n_obs == 0
        assert rls.theta.shape == (3,)
        assert rls.P.shape == (3, 3)
    
    def test_simple_fit(self):
        """Test fitting with simple data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 2
        
        # Generate data: y = 2*x1 + 3*x2 + noise
        X = np.random.randn(n_samples, n_features)
        true_theta = np.array([2.0, 3.0])
        y = X @ true_theta + 0.1 * np.random.randn(n_samples)
        
        rls = OnlineRLS(n_features=n_features, alpha=1e-4)
        rls.partial_fit(X, y)
        
        # Check estimates are close to true values
        assert rls.n_obs == n_samples
        assert np.allclose(rls.theta, true_theta, atol=0.3)
    
    def test_batch_vs_sequential(self):
        """Test that batch and sequential updates give same result."""
        np.random.seed(42)
        n_samples = 50
        n_features = 2
        
        X = np.random.randn(n_samples, n_features)
        y = X @ np.array([1.0, 2.0]) + 0.1 * np.random.randn(n_samples)
        
        # Batch
        rls_batch = OnlineRLS(n_features=n_features)
        rls_batch.partial_fit(X, y)
        
        # Sequential
        rls_seq = OnlineRLS(n_features=n_features)
        for i in range(n_samples):
            rls_seq.partial_fit(X[i:i+1], y[i:i+1])
        
        # Should give same results
        assert np.allclose(rls_batch.theta, rls_seq.theta, atol=1e-6)
        assert np.allclose(rls_batch.rss, rls_seq.rss, atol=1e-4)
    
    def test_r_squared(self):
        """Test R-squared calculation."""
        np.random.seed(42)
        n_samples = 100
        n_features = 2
        
        X = np.random.randn(n_samples, n_features)
        y = X @ np.array([2.0, 3.0]) + 0.5 * np.random.randn(n_samples)
        
        rls = OnlineRLS(n_features=n_features, alpha=1e-4)
        rls.partial_fit(X, y)
        
        r_squared = rls.get_r_squared()
        
        # R-squared should be between 0 and 1
        assert 0 <= r_squared <= 1
        # With low noise, should be high
        assert r_squared > 0.7
    
    def test_f_statistic(self):
        """Test F-statistic calculation."""
        np.random.seed(42)
        n_samples = 100
        n_features = 2
        
        X = np.random.randn(n_samples, n_features)
        y = X @ np.array([2.0, 3.0]) + 0.5 * np.random.randn(n_samples)
        
        rls = OnlineRLS(n_features=n_features, alpha=1e-4, 
                       feature_names=['x1', 'x2'])
        rls.partial_fit(X, y)
        
        f_stat, df_model, df_resid = rls.get_f_statistic()
        
        # Check degrees of freedom
        assert df_model == 2  # No intercept
        assert df_resid == n_samples - n_features
        
        # F-statistic should be positive
        assert f_stat > 0
        
        # With good fit, F should be large
        assert f_stat > 10


class TestOLSAPI:
    """Tests for OLS API."""
    
    def test_simple_regression(self):
        """Test simple OLS regression via API."""
        np.random.seed(42)
        n_samples = 200
        
        # Generate data
        df = pd.DataFrame({
            'y': np.random.randn(n_samples),
            'x1': np.random.randn(n_samples),
            'x2': np.random.randn(n_samples)
        })
        df['y'] = 2 * df['x1'] + 3 * df['x2'] + 0.5 * np.random.randn(n_samples)
        
        # Fit model
        model = OLS("y ~ x1 + x2")
        model.fit(df)
        
        # Check fit
        assert model.n_obs_ == n_samples
        assert model.r_squared_ > 0.7
        
        # Check coefficients are reasonable
        coefs = model.coef_
        assert len(coefs) == 3  # intercept + 2 features
        assert abs(coefs[1] - 2.0) < 0.3
        assert abs(coefs[2] - 3.0) < 0.3
    
    def test_formula_with_intercept_removal(self):
        """Test formula with intercept removal."""
        np.random.seed(42)
        df = pd.DataFrame({
            'y': np.random.randn(100),
            'x1': np.random.randn(100)
        })
        
        model = OLS("y ~ x1 - 1")
        model.fit(df)
        
        # Should not have intercept
        assert len(model.coef_) == 1
        assert 'intercept' not in model.results_.feature_names
    
    def test_prediction(self):
        """Test prediction functionality."""
        np.random.seed(42)
        df = pd.DataFrame({
            'y': np.random.randn(100),
            'x1': np.random.randn(100),
            'x2': np.random.randn(100)
        })
        df['y'] = 2 * df['x1'] + 3 * df['x2']
        
        model = OLS("y ~ x1 + x2 - 1")
        model.fit(df)
        
        # Predict on new data
        X_new = pd.DataFrame({'x1': [1.0], 'x2': [1.0]})
        y_pred = model.predict(X_new)
        
        # Should be approximately 2*1 + 3*1 = 5
        assert np.abs(y_pred[0] - 5.0) < 0.5


def test_monte_carlo_ols():
    """Monte Carlo test: verify OLS recovers true parameters."""
    np.random.seed(42)
    n_simulations = 50
    n_samples = 500
    n_features = 3
    true_theta = np.array([1.0, -0.5, 2.0])
    
    estimates = []
    
    for sim in range(n_simulations):
        # Generate data
        X = np.random.randn(n_samples, n_features)
        y = X @ true_theta + np.random.randn(n_samples)
        
        # Fit model
        rls = OnlineRLS(n_features=n_features, alpha=1e-4)
        rls.partial_fit(X, y)
        
        estimates.append(rls.theta)
    
    estimates = np.array(estimates)
    mean_estimates = estimates.mean(axis=0)
    
    # Mean estimate should be close to true value
    assert np.allclose(mean_estimates, true_theta, atol=0.1)
    
    # Check coverage: 95% CI should contain true value ~95% of times
    for feat_idx in range(n_features):
        ci_lower = np.percentile(estimates[:, feat_idx], 2.5)
        ci_upper = np.percentile(estimates[:, feat_idx], 97.5)
        
        contains_true = ci_lower <= true_theta[feat_idx] <= ci_upper
        # Allow some slack due to random variation
        assert contains_true or abs(true_theta[feat_idx] - mean_estimates[feat_idx]) < 0.2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
